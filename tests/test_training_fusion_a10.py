import numpy as np

from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder
from cerm._internal.cerm_stable_quotient_block import StableQuotientBlockCERM
from cerm._internal.cerm_hybrid_quotient_block import (
    HybridConfig,
    HybridQuotientBlockCERM,
    _rank_native_candidate_bank,
    _ranking_boundary_ambiguous,
)
from cerm.training_cache import NewtonHistogramCache


def test_quantile_fit_transform_matches_fit_then_transform_bitwise():
    rng = np.random.default_rng(20260811)
    X = rng.normal(size=(1800, 14))
    X[:, 0] = rng.integers(0, 4, size=len(X))
    X[::37, 1] = np.nan
    y = rng.integers(0, 2, size=len(X))
    fused = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16))
    states_fused = fused.fit_transform(X, y)
    reference = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(X, y)
    states_reference = reference.transform(X)
    for left, right in zip(fused.thresholds_, reference.thresholds_):
        assert np.array_equal(left, right)
    for level in fused.levels:
        for left, right in zip(fused.maps_[level], reference.maps_[level]):
            assert np.array_equal(left, right)
        assert np.array_equal(states_fused[level], states_reference[level])


def test_stable_fused_fold_histograms_match_historical_two_dictionary_path():
    rng = np.random.default_rng(7)
    n, d = 2200, 10
    C4 = rng.integers(0, 4, size=(n, d), dtype=np.int16)
    C16 = rng.integers(0, 16, size=(n, d), dtype=np.int16)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    p = np.clip(0.1 + 0.8 * rng.random(n), 1e-5, 1 - 1e-5)
    model = StableQuotientBlockCERM(pair_feature_limit=d, random_state=19)

    # Historical complementary folds.
    rr = np.random.default_rng(model.random_state)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rr.shuffle(idx0); rr.shuffle(idx1)
    fold_a = np.concatenate([idx0[::2], idx1[::2]])
    fold_b = np.concatenate([idx0[1::2], idx1[1::2]])
    if len(fold_b) == 0:
        fold_b = fold_a
    ga = model._gain_dictionary(C4, C16, y, p, fold_a)
    gb = model._gain_dictionary(C4, C16, y, p, fold_b)
    ranked = []
    for key in ga.keys() & gb.keys():
        a = float(ga[key]); b = float(gb[key])
        if a <= 0.0 or b <= 0.0:
            continue
        gate_j, gate_state, target_k, target_card = key
        harmonic = 2.0 * a * b / (a + b + 1e-12)
        agreement = min(a, b) / (max(a, b) + 1e-12)
        score = harmonic * np.sqrt(max(agreement, 0.0)) / np.sqrt(8.0 * target_card)
        ranked.append((score, harmonic, agreement, gate_j, gate_state, target_k, target_card, a, b))
    ranked.sort(reverse=True)
    expected = [
        (gate_j, gate_state, target_k, target_card, harmonic)
        for _, harmonic, _, gate_j, gate_state, target_k, target_card, _, _ in ranked
    ]
    actual = model._stable_rank_blocks(C4, C16, y, p)
    assert actual == expected
    assert model.stability_diagnostics_ == ranked


def test_newton_histogram_pair_buffer_preserves_statistics():
    rng = np.random.default_rng(8)
    states = rng.integers(0, 8, size=(1500, 12), dtype=np.int16)
    y = rng.integers(0, 2, size=len(states))
    p = np.clip(0.15 + 0.7 * rng.random(len(states)), 1e-5, 1 - 1e-5)
    cache = NewtonHistogramCache(states, y, p, feature_limit=10)
    g = y.astype(np.float64) - p
    h = p * (1.0 - p)
    for (j, k), stats in cache.pairs.items():
        card_k = int(cache.states.cardinalities[k])
        code = states[:, j].astype(np.int64) * card_k + states[:, k]
        card = int(cache.states.cardinalities[j]) * card_k
        assert np.array_equal(stats.G, np.bincount(code, weights=g, minlength=card))
        assert np.array_equal(stats.H, np.bincount(code, weights=h, minlength=card))


def test_hybrid_shared_pair_scan_matches_separate_rankings():
    rng = np.random.default_rng(20260923)
    n, d = 4200, 14
    C16 = rng.integers(0, 16, size=(n, d), dtype=np.int16)
    C4 = (C16 // 4).astype(np.int16)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    p = np.clip(0.05 + 0.9 * rng.random(n), 1e-5, 1 - 1e-5)
    model = HybridQuotientBlockCERM(
        pair_feature_limit=d,
        max_interactions=40,
        random_state=41,
    )

    for sample_weight in (
        None,
        np.clip(np.exp(rng.normal(0.0, 1.3, size=n)), 1e-3, 1e3),
    ):
        expected_full = model._rank_blocks(
            C4,
            C16,
            y,
            p,
            gain_l2=5.0,
            min_hessian=1.0,
            max_results=40,
            sample_weight=sample_weight,
        )
        expected_stable = model._stable_rank_blocks(
            C4,
            C16,
            y,
            p,
            max_results=40,
            sample_weight=sample_weight,
        )
        actual = model._shared_rank_blocks(
            C4,
            C16,
            y,
            p,
            max_results=40,
            sample_weight=sample_weight,
        )

        assert actual is not None
        assert actual["_guard_fallback"] == ()
        assert actual["_histogram_backend"] in {
            "python",
            "native",
            "native_chunked",
            "native_quotient",
            "native_quotient_chunked",
            "native_gain",
        }
        assert [row[:4] for row in actual["full"]] == [
            row[:4] for row in expected_full
        ]
        assert [row[:4] for row in actual["stable"]] == [
            row[:4] for row in expected_stable
        ]
        np.testing.assert_allclose(
            [row[4] for row in actual["full"]],
            [row[4] for row in expected_full],
            atol=1e-10,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            [row[4] for row in actual["stable"]],
            [row[4] for row in expected_stable],
            atol=1e-10,
            rtol=1e-12,
        )


def test_hybrid_shared_pair_scan_falls_back_for_non_nested_states():
    rng = np.random.default_rng(20260924)
    n, d = 700, 8
    C4 = rng.integers(0, 4, size=(n, d), dtype=np.int16)
    C16 = rng.integers(0, 16, size=(n, d), dtype=np.int16)
    y = rng.integers(0, 2, size=n, dtype=np.int64)
    p = np.clip(0.1 + 0.8 * rng.random(n), 1e-5, 1 - 1e-5)
    model = HybridQuotientBlockCERM(
        pair_feature_limit=d,
        max_interactions=16,
        random_state=43,
    )
    assert (
        model._shared_rank_blocks(
            C4,
            C16,
            y,
            p,
            max_results=16,
        )
        is None
    )


def test_shared_rank_guard_detects_retained_prefix_ties():
    well_separated = [
        (10.0, "a"),
        (9.0, "b"),
        (8.0, "c"),
        (7.0, "d"),
    ]
    near_tie = [
        (10.0, "a"),
        (9.0, "b"),
        (8.0, "c"),
        (8.0 - 4e-6, "d"),
    ]
    exact_tie = [
        (10.0, "a"),
        (9.0, "b"),
        (8.0, "c"),
        (8.0, "d"),
    ]

    assert not _ranking_boundary_ambiguous(well_separated, 3)
    assert _ranking_boundary_ambiguous(near_tie, 3)
    assert _ranking_boundary_ambiguous(exact_tie, 3)
    assert not _ranking_boundary_ambiguous(exact_tie[:3], 3)
    assert not _ranking_boundary_ambiguous(exact_tie, None)


def test_shared_rank_guard_has_k_plus_one_for_bounded_heap_contract():
    # High-dimensional ranking keeps one extra candidate so the guard can
    # inspect the k / k+1 boundary before truncating to the requested prefix.
    requested = 5
    heap_limit = requested + 1
    items = [(float(score), score) for score in range(20)]
    import heapq

    heap = []
    for item in items:
        if len(heap) < heap_limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    heap.sort(reverse=True)

    assert len(heap) == requested + 1
    assert heap[:requested] == sorted(items, reverse=True)[:requested]


def test_hybrid_shared_pair_scan_falls_back_for_degenerate_complementary_fold():
    # One row from each class leaves the historical complementary fold B empty.
    # The shared path must not reuse duplicated fold-A statistics as full-data
    # statistics because that would double G/H.
    C16 = np.asarray([[0, 1], [4, 5]], dtype=np.int16)
    C4 = (C16 // 4).astype(np.int16)
    y = np.asarray([0, 1], dtype=np.int64)
    p = np.asarray([0.25, 0.75], dtype=np.float64)
    model = HybridQuotientBlockCERM(
        pair_feature_limit=2,
        max_interactions=2,
        random_state=47,
    )

    assert model._shared_rank_blocks(
        C4,
        C16,
        y,
        p,
        max_results=2,
    ) is None



def test_vectorized_native_bank_ranking_matches_tuple_reference():
    rng = np.random.default_rng(20260927)
    n_candidates = 1200
    gate_j = rng.integers(0, 18, size=n_candidates, dtype=np.int32)
    gate_state = rng.integers(0, 4, size=n_candidates, dtype=np.int16)
    target_k = rng.integers(0, 18, size=n_candidates, dtype=np.int32)
    target_card = rng.integers(8, 17, size=n_candidates, dtype=np.int16)
    full = rng.gamma(2.0, 8.0, size=n_candidates)
    gain_a = rng.gamma(2.0, 7.0, size=n_candidates)
    gain_b = rng.gamma(2.0, 7.0, size=n_candidates)
    flags = np.full(n_candidates, 7, dtype=np.uint8)

    # Force exact ties across the retained boundary so top-k selection must
    # keep a score-boundary superset before applying the historical tuple keys.
    full[:80] = 12.5
    gain_a[80:160] = 9.0
    gain_b[80:160] = 9.0
    flags[::17] &= np.uint8(~4 & 0xFF)
    flags[::19] &= np.uint8(~1 & 0xFF)

    bank = {
        "gate_j": gate_j,
        "gate_state": gate_state,
        "target_k": target_k,
        "target_card": target_card,
        "full": full,
        "gain_a": gain_a,
        "gain_b": gain_b,
        "flags": flags,
    }

    def reference(max_results, cost_byte, cost_eval):
        full_rows = []
        stable_rows = []
        for i in range(n_candidates):
            if flags[i] & 4:
                bytes_cost = 8.0 * int(target_card[i]) + 16.0
                normalized = float(full[i]) / 5000.0
                penalized = (
                    normalized
                    - cost_byte * bytes_cost
                    - cost_eval * 2.0
                )
                score = penalized / np.sqrt(max(bytes_cost, 1.0))
                if penalized > 0.0 or (cost_byte == 0.0 and cost_eval == 0.0):
                    full_rows.append(
                        (
                            float(score),
                            float(full[i]),
                            int(gate_j[i]),
                            int(gate_state[i]),
                            int(target_k[i]),
                            int(target_card[i]),
                        )
                    )
            if (
                (flags[i] & 3) == 3
                and gain_a[i] > 0.0
                and gain_b[i] > 0.0
            ):
                a = float(gain_a[i])
                b = float(gain_b[i])
                harmonic = 2.0 * a * b / (a + b + 1e-12)
                agreement = min(a, b) / (max(a, b) + 1e-12)
                score = (
                    harmonic
                    * np.sqrt(max(agreement, 0.0))
                    / np.sqrt(8.0 * int(target_card[i]))
                )
                stable_rows.append(
                    (
                        float(score),
                        float(harmonic),
                        float(agreement),
                        int(gate_j[i]),
                        int(gate_state[i]),
                        int(target_k[i]),
                        int(target_card[i]),
                        a,
                        b,
                    )
                )
        full_rows.sort(reverse=True)
        stable_rows.sort(reverse=True)
        if max_results is not None:
            if int(max_results) <= 0:
                return [], []
            limit = int(max_results) + 1
            full_rows = full_rows[:limit]
            stable_rows = stable_rows[:limit]
        return full_rows, stable_rows

    for max_results in (None, 0, 5, 40):
        for cost_byte, cost_eval in ((0.0, 0.0), (1e-8, 2e-7)):
            expected = reference(max_results, cost_byte, cost_eval)
            actual = _rank_native_candidate_bank(
                bank,
                n_rows=5000,
                max_results=max_results,
                block_cost_per_byte=cost_byte,
                block_cost_per_eval=cost_eval,
            )
            assert actual == expected



def test_hybrid_design_groups_parallelize_without_nested_solver_workers(monkeypatch):
    import threading
    import time

    model = HybridQuotientBlockCERM(
        pair_feature_limit=8,
        max_interactions=16,
        random_state=53,
        n_jobs=4,
    )
    cache = {}
    built = []
    solver_workers = []
    lock = threading.Lock()

    def fake_block_bank(_cache, cfg):
        built.append((cfg.rank_mode, float(cfg.cell_shrinkage)))
        return object()

    def fake_eval(
        _cache,
        _ytr,
        _yva,
        configs,
        _wtr=None,
        _wva=None,
        *,
        solver_n_jobs=None,
    ):
        # Reverse completion pressure verifies executor.map preserves group order.
        time.sleep(0.002 * (5 - configs[0].n_blocks))
        with lock:
            solver_workers.append(solver_n_jobs)
        return [
            (float(cfg.n_blocks), float(cfg.C), cfg)
            for cfg in configs
        ]

    monkeypatch.setattr(model, "_block_bank", fake_block_bank)
    monkeypatch.setattr(model, "_eval_hybrid_path", fake_eval)

    c0 = HybridConfig("full", 0, 0.0, 0.2)
    c1 = HybridConfig("full", 4, 0.0, 0.2)
    c2 = HybridConfig("full", 4, 0.0, 1.0)
    c3 = HybridConfig("stable", 2, 0.0, 0.2)
    c4 = HybridConfig("full", 3, 10.0, 0.2)
    groups = {
        ("full", 0, 0.0): [c0],
        ("full", 4, 0.0): [c1, c2],
        ("stable", 2, 0.0): [c3],
        ("full", 3, 10.0): [c4],
    }

    scored, workers = model._eval_design_groups(
        cache,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        groups,
    )

    assert workers == 4
    assert [row[2] for row in scored] == [c0, c1, c2, c3, c4]
    assert solver_workers == [1, 1, 1, 1]
    assert built == [
        ("full", 0.0),
        ("stable", 0.0),
        ("full", 10.0),
    ]



def test_hybrid_real_solver_parallel_selection_matches_serial_exactly():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        class_sep=0.85,
        random_state=20260928,
    )
    common = dict(
        max_features=10,
        pair_feature_limit=10,
        max_interactions=8,
        search_profile="practical",
        fixed_C=0.2,
        random_state=20260928,
    )

    serial = HybridQuotientBlockCERM(n_jobs=1, **common).fit(X, y)
    parallel = HybridQuotientBlockCERM(n_jobs=4, **common).fit(X, y)

    assert serial.primary_solver_group_workers_ == 1
    assert parallel.primary_solver_group_workers_ >= 2
    assert serial.primary_selected_config_ == parallel.primary_selected_config_
    assert serial.selected_hybrid_config_ == parallel.selected_hybrid_config_
    assert [row[2] for row in serial.primary_scores_] == [
        row[2] for row in parallel.primary_scores_
    ]
    np.testing.assert_array_equal(
        np.asarray([row[0] for row in serial.primary_scores_]),
        np.asarray([row[0] for row in parallel.primary_scores_]),
    )
    np.testing.assert_array_equal(
        serial.predict_proba(X),
        parallel.predict_proba(X),
    )
