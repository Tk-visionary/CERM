from itertools import combinations

import numpy as np
import pytest

from cerm import _experimental_undr_geometry as geom
from cerm.experimental_undr_v21 import (
    UNDRConfig,
    DeficitCandidate,
    ProjectedTrainingGraph,
    Selection,
    _basis,
    _native_resolution_stat_banks,
    resolution_candidates,
    _build_triad_pair_prefix_cache,
    _triad_cube,
    _pair_sweep_bound,
    _pair_sweep_bound_batch,
    _vectorized_native_triad_stage1,
    _conditional_gains_batched,
    _ConditionalPairWorkspace,
    _conditional_gains_from_gram,
    fit_joint,
    _compact_state_level,
    _state_transforms,
    extract_projected_training_graph,
    fit_frozen_undr,
)


class MockEncoder:
    levels=(4,8,16)
    def __init__(self, X):
        self.edges=[]; self.calls=[]
        for j in range(X.shape[1]):
            self.edges.append(np.quantile(X[:,j], np.arange(1,16)/16))
    def _fine(self, X, cols):
        X=np.asarray(X,float); out=np.empty((len(X),len(cols)),np.int16)
        for k,j in enumerate(cols): out[:,k]=np.searchsorted(self.edges[int(j)],X[:,int(j)],side='right')
        return out
    def transform_level_columns(self, X, q, cols):
        self.calls.append((int(q),tuple(int(x) for x in cols)))
        f=self._fine(X,cols)
        return f//(16//int(q))
    def transform(self, X):
        raise AssertionError('full encoder transform should not be used by a10+ integration path')


class FusedMockEncoder(MockEncoder):
    def __init__(self, X):
        super().__init__(X)
        self.fused_calls = []

    def transform_columns(self, X, cols):
        cols = tuple(int(x) for x in cols)
        self.fused_calls.append(cols)
        fine = self._fine(X, cols)
        return {4: fine // 4, 8: fine // 2, 16: fine}

class MockBase:
    def __init__(self,X):
        self.encoder_=MockEncoder(X); self.feature_idx_=np.arange(X.shape[1],dtype=np.int64)


class MockModel:
    def __init__(self,X): self.base_=MockBase(X)
    def decision_function(self,X): return np.zeros(len(X),float)


class MockEstimator:
    def __init__(self,X): self.model_=MockModel(X)


def make(kind,seed=1,n=10000,d=8,strength=1.2):
    r=np.random.default_rng(seed); X=r.uniform(-1,1,(n,d))
    z8=np.minimum(7,((X[:,0]+1)*4).astype(int)); z16=np.minimum(15,((X[:,0]+1)*8).astype(int))
    s8=np.where(z8%2==0,1.,-1.); s16=np.where(z16%2==0,1.,-1.)
    tri=np.sign(X[:,1])*np.sign(X[:,2])*np.sign(X[:,3])
    if kind=='q8': logit=strength*s8
    elif kind=='q16': logit=strength*s16
    elif kind=='triad': logit=strength*tri
    elif kind=='null': logit=np.zeros(n)
    else: raise KeyError(kind)
    y=r.binomial(1,1/(1+np.exp(-logit)))
    return X,y


def split(X,y):
    return X[:8000],y[:8000],X[8000:],y[8000:]


def test_compact_state_level_rejects_wraparound():
    valid = np.asarray([[0, 3], [1, 2]], dtype=np.int16)
    compact = _compact_state_level(valid, 4)
    assert compact.dtype == np.uint8
    np.testing.assert_array_equal(compact, valid)

    with pytest.raises(ValueError, match="outside the requested quotient"):
        _compact_state_level(np.asarray([[0, 4]], dtype=np.int16), 4)
    with pytest.raises(ValueError, match="outside the requested quotient"):
        _compact_state_level(np.asarray([[0, -1]], dtype=np.int16), 4)


def test_projected_traininggraph_path_only():
    X,y=make('q8',2,n=2000); est=MockEstimator(X[:1500])
    g=extract_projected_training_graph(est,X[:1500],y[:1500],maximum_features=6)
    assert set(g.states_by_q)=={4,8,16}
    assert all(values.dtype == np.uint8 for values in g.states_by_q.values())
    assert all(v=='transform_level_columns' for v in g.transform_calls.values())
    assert len(est.model_.base_.encoder_.calls)==3


def test_projected_traininggraph_fuses_available_multilevel_transform():
    X, y = make('q8', 22, n=2200)
    Xt, yt = X[:1700], y[:1700]
    est = MockEstimator(Xt)
    encoder = FusedMockEncoder(Xt)
    est.model_.base_.encoder_ = encoder

    graph = extract_projected_training_graph(
        est, Xt, yt, maximum_features=6
    )
    assert set(graph.states_by_q) == {4, 8, 16}
    assert all(values.dtype == np.uint8 for values in graph.states_by_q.values())
    assert all(
        value == "transform_columns_fused"
        for value in graph.transform_calls.values()
    )
    assert len(encoder.fused_calls) == 1
    assert encoder.calls == []

    fine = encoder._fine(Xt, np.arange(6))
    np.testing.assert_array_equal(graph.states_by_q[4], fine // 4)
    np.testing.assert_array_equal(graph.states_by_q[8], fine // 2)
    np.testing.assert_array_equal(graph.states_by_q[16], fine)


@pytest.mark.parametrize('kind,needle', [('q8','q4->q8:x0'),('q16','q8->q16:x0'),('triad','triad:')])
def test_strong_signal_end_to_end(kind,needle):
    X,y=make(kind,10+len(kind)); Xt,yt,Xh,yh=split(X,y); est=MockEstimator(Xt)
    cfg=UNDRConfig(maximum_features=8,budget=2,maximum_stage2_triads=16)
    prog=fit_frozen_undr(est,Xt,yt,Xh,yh,config=cfg)
    assert prog.audit.accepted
    assert any(needle in name for name in prog.audit.selected_names)
    p=prog.predict_proba(Xh[:50]); assert p.shape==(50,2); assert np.allclose(p.sum(1),1)
    assert prog.model_bytes_>0


def test_state_transforms_keeps_single_level_direct_path():
    X, y = make('q8', 23, n=1800)
    encoder = FusedMockEncoder(X)
    raw = np.arange(6, dtype=np.int64)

    states, paths = _state_transforms(
        encoder, False, X, [16], raw
    )
    assert set(states) == {16}
    assert states[16].dtype == np.uint8
    assert paths == {16: "transform_level_columns"}
    assert len(encoder.calls) == 1
    assert encoder.fused_calls == []


def test_state_transforms_fuses_two_prediction_levels():
    X, y = make('q8', 24, n=1800)
    encoder = FusedMockEncoder(X)
    raw = np.arange(6, dtype=np.int64)

    states, paths = _state_transforms(
        encoder, False, X, [4, 16], raw
    )
    assert set(states) == {4, 16}
    assert states[4].dtype == np.uint8
    assert states[16].dtype == np.uint8
    assert paths == {
        4: "transform_columns_fused",
        16: "transform_columns_fused",
    }
    assert encoder.calls == []
    assert len(encoder.fused_calls) == 1

    fine = encoder._fine(X, raw)
    np.testing.assert_array_equal(states[4], fine // 4)
    np.testing.assert_array_equal(states[16], fine)


def test_rejected_program_is_exact_base_noop():
    X,y=make('null',77); Xt,yt,Xh,yh=split(X,y); est=MockEstimator(Xt)
    prog=fit_frozen_undr(est,Xt,yt,Xh,yh,config=UNDRConfig(maximum_features=8,budget=2))
    base=est.model_.decision_function(Xh)
    if not prog.accepted:
        np.testing.assert_array_equal(prog.decision_function(Xh),base)
        assert prog.model_bytes_==0


def test_budget_zero_is_exact_noop():
    X,y=make('q8',123); Xt,yt,Xh,yh=split(X,y); est=MockEstimator(Xt)
    prog=fit_frozen_undr(est,Xt,yt,Xh,yh,config=UNDRConfig(maximum_features=8,budget=0))
    assert not prog.accepted
    np.testing.assert_array_equal(prog.decision_function(Xh),est.model_.decision_function(Xh))


def test_actual_cerm_shadow_encoder_recovers_q16_cancellation():
    from cerm import CERMClassifier

    rng = np.random.default_rng(211)
    n = 3000
    X = rng.uniform(-1.0, 1.0, size=(n, 6))
    z16 = np.minimum(15, ((X[:, 0] + 1.0) * 8.0).astype(int))
    signal = np.where(z16 % 2 == 0, 1.0, -1.0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-2.0 * signal))).astype(np.int32)
    base = CERMClassifier(
        preset="balanced", max_bins=4, max_features=6,
        max_interaction_features=6, max_interactions=0,
        random_state=211, n_jobs=1,
    ).fit(X[:2100], y[:2100])
    prog = fit_frozen_undr(
        base, X[:2100], y[:2100], X[2100:2550], y[2100:2550],
        config=UNDRConfig(
            maximum_features=6, budget=2,
            maximum_stage2_triads=8, alpha2=0.001,
        ),
    )
    assert prog.audit.accepted
    assert "q8->q16:x0" in prog.audit.selected_names
    assert tuple(prog.audit.search_metadata["state_levels"]) == (4, 8, 16)
    assert all(
        "shadow_selected_features:transform_columns_fused" == value
        for value in prog.audit.search_metadata["transform_calls"].values()
    )



def test_triad_refinement_cache_reproduces_historical_cube_exactly():
    rng = np.random.default_rng(20260929)
    S4 = rng.integers(0, 4, size=(700, 9), dtype=np.int16)
    score = rng.normal(scale=0.8, size=len(S4))
    y = rng.integers(0, 2, size=len(S4), dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    assert np.all(h > 0)

    pair_cache = _build_triad_pair_prefix_cache(
        S4,
        budget_bytes=32 * 1024 * 1024,
    )
    code = np.empty(len(S4), dtype=np.int64)
    for triad in combinations(range(S4.shape[1]), 3):
        a, b, c = triad
        historical_code = (
            (S4[:, a].astype(np.int64) * 4 + S4[:, b]) * 4
            + S4[:, c]
        )
        expected_G = np.bincount(
            historical_code, weights=g, minlength=64
        ).reshape(4, 4, 4)
        expected_H = np.bincount(
            historical_code, weights=h, minlength=64
        ).reshape(4, 4, 4)
        expected_support = np.bincount(
            historical_code, minlength=64
        ).reshape(4, 4, 4) > 0

        actual_G, actual_H, actual_support = _triad_cube(
            S4,
            g,
            h,
            triad,
            code_buffer=code,
            pair_prefix_cache=pair_cache,
        )
        np.testing.assert_array_equal(actual_G, expected_G)
        np.testing.assert_array_equal(actual_H, expected_H)
        np.testing.assert_array_equal(actual_support, expected_support)


def test_triad_refinement_cache_is_bounded_and_reuse_prioritized():
    rng = np.random.default_rng(20260930)
    S4 = rng.integers(0, 4, size=(1000, 12), dtype=np.uint8)
    bank, lookup = _build_triad_pair_prefix_cache(
        S4,
        budget_bytes=2500,
    )

    assert bank.dtype == np.uint8
    assert bank.nbytes <= 2500
    assert bank.shape == (2, len(S4))
    # With equal row cost, b=1 prefixes have the highest downstream reuse.
    assert set(lookup) == {(0, 1), (0, 2)}


def test_undr_triad_audit_reports_refinement_cache_usage(monkeypatch):
    def forced_pair_cache(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        forced_pair_cache,
    )
    X, y = make("triad", 314, n=3000, d=8, strength=1.4)
    Xt, yt, Xh, yh = split(X, y)
    est = MockEstimator(Xt)
    prog = fit_frozen_undr(
        est,
        Xt,
        yt,
        Xh,
        yh,
        config=UNDRConfig(
            maximum_features=8,
            budget=2,
            maximum_stage2_triads=16,
        ),
    )
    meta = prog.audit.search_metadata
    assert meta["triad_histogram_backend"] == "pair_prefix_cache"
    assert meta["triad_support_from_hessian"] is True
    assert meta["triad_pair_cache_bytes"] > 0
    assert meta["triad_pair_cache_pairs"] > 0
    assert meta["triad_pair_cache_hits"] > 0
    assert (
        meta["triad_pair_cache_hits"] + meta["triad_pair_cache_misses"]
        == meta["n_triads"]
    )



def test_vectorized_native_stage1_is_one_sided_against_scalar():
    from scipy.stats import chi2

    rng = np.random.default_rng(20261001)
    critical = float(chi2.ppf(1.0 - 0.001, 27))
    all_deltas = []

    for seed in range(12):
        local = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        n, d = 2200, 10
        S4 = local.integers(0, 4, size=(n, d), dtype=np.uint8)
        score = local.normal(scale=1.5, size=n)
        y = local.integers(0, 2, size=n, dtype=np.int32)
        g, h = geom.logistic_gh(y, score)
        triads = np.asarray(
            list(combinations(range(d), 3)),
            dtype=np.int32,
        )
        bank = np.empty((len(triads), 64, 2), dtype=np.float64)
        scalar_reject = np.zeros(len(triads), dtype=bool)
        scalar_cheap = np.empty(len(triads), dtype=np.float64)

        for index, triad in enumerate(triads):
            a, b, c = (int(value) for value in triad)
            code = (
                (S4[:, a].astype(np.int64) * 4 + S4[:, b]) * 4
                + S4[:, c]
            )
            G = np.bincount(
                code, weights=g, minlength=64
            ).reshape(4, 4, 4)
            H = np.bincount(
                code, weights=h, minlength=64
            ).reshape(4, 4, 4)
            bank[index, :, 0] = G.reshape(-1)
            bank[index, :, 1] = H.reshape(-1)
            valid = H > 0
            full = 0.5 * float(np.sum(G[valid] ** 2 / H[valid]))
            cheap = max(
                0.0,
                full
                - max(
                    # Historical scalar arithmetic.
                    0.5
                    * float(
                        np.sum(
                            np.sum(G, axis=2)[np.sum(H, axis=2) > 0] ** 2
                            / np.sum(H, axis=2)[np.sum(H, axis=2) > 0]
                        )
                    ),
                    0.5
                    * float(
                        np.sum(
                            np.sum(G, axis=1)[np.sum(H, axis=1) > 0] ** 2
                            / np.sum(H, axis=1)[np.sum(H, axis=1) > 0]
                        )
                    ),
                    0.5
                    * float(
                        np.sum(
                            np.sum(G, axis=0)[np.sum(H, axis=0) > 0] ** 2
                            / np.sum(H, axis=0)[np.sum(H, axis=0) > 0]
                        )
                    ),
                ),
            )
            scalar_cheap[index] = cheap
            scalar_reject[index] = bool(
                np.all(valid) and 2.0 * cheap <= critical
            )

        clear_reject, ambiguous, vector_cheap = (
            _vectorized_native_triad_stage1(bank, critical)
        )
        # The vectorized pass is only a prefilter: it may defer extra rows, but
        # it must never remove a row that historical scalar screening retains.
        assert not np.any(clear_reject & ~scalar_reject)
        assert not np.any(clear_reject & ambiguous)
        all_deltas.append(
            float(np.max(np.abs(vector_cheap - scalar_cheap)))
        )

    assert max(all_deltas) <= 1e-12


def test_undr_native_stage1_reports_guarded_vectorized_screen(monkeypatch):
    def forced_native_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        forced_native_bank,
    )
    X, y = make("triad", 315, n=3200, d=8, strength=1.4)
    Xt, yt, Xh, yh = split(X, y)
    prog = fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=UNDRConfig(
            maximum_features=8,
            budget=2,
            maximum_stage2_triads=16,
        ),
    )
    meta = prog.audit.search_metadata
    assert meta["triad_histogram_backend"] == "native"
    assert meta["triad_stage1_backend"] == "vectorized_guarded"
    assert meta["triad_stage1_vectorized_rejects"] > 0
    assert meta["triad_stage1_vectorized_rejects"] <= meta["cheap_safe_reject"]
    assert meta["triad_stage1_ambiguous"] >= 0
    assert any(name.startswith("triad:") for name in prog.audit.selected_names)



@pytest.mark.parametrize("n_rows", [100, 250, 700, 2200])
def test_batched_triad_sweep_matches_scalar_bitwise(n_rows):
    rng = np.random.default_rng(20261020 + int(n_rows))
    n, d = int(n_rows), 10
    S4 = rng.integers(0, 4, size=(n, d), dtype=np.uint8)
    score = rng.normal(scale=1.2, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)

    triads = np.asarray(
        list(combinations(range(d), 3)),
        dtype=np.int32,
    )
    G_bank = np.empty((len(triads), 4, 4, 4), dtype=np.float64)
    H_bank = np.empty_like(G_bank)
    for index, (a, b, c) in enumerate(triads):
        code = (
            (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
            + S4[:, int(c)]
        )
        G_bank[index] = np.bincount(
            code, weights=g, minlength=64
        ).reshape(4, 4, 4)
        H_bank[index] = np.bincount(
            code, weights=h, minlength=64
        ).reshape(4, 4, 4)

    expected = np.asarray(
        [
            _pair_sweep_bound(G, H, H > 0)
            for G, H in zip(G_bank, H_bank)
        ],
        dtype=np.float64,
    )
    actual = _pair_sweep_bound_batch(G_bank, H_bank)
    np.testing.assert_array_equal(actual, expected)


def test_undr_native_path_reports_vectorized_exact_sweep(monkeypatch):
    def forced_native_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        forced_native_bank,
    )
    X, y = make("triad", 316, n=3200, d=8, strength=1.4)
    Xt, yt, Xh, yh = split(X, y)
    prog = fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=UNDRConfig(
            maximum_features=8,
            budget=2,
            maximum_stage2_triads=16,
        ),
    )
    meta = prog.audit.search_metadata
    assert meta["triad_histogram_backend"] == "native"
    assert meta["triad_sweep_backend"] == "vectorized_exact"
    assert meta["triad_sweep_batch_size"] >= 0
    assert any(name.startswith("triad:") for name in prog.audit.selected_names)




def test_triad_score_from_screening_stats_matches_historical_bitwise():
    rng = np.random.default_rng(20261030)
    n, d = 1800, 8
    S4 = rng.integers(0, 4, size=(n, d), dtype=np.uint8)
    score = rng.normal(scale=1.1, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    triad = (1, 4, 6)
    a, b, c = triad
    code = (
        (S4[:, a].astype(np.int64) * 4 + S4[:, b]) * 4
        + S4[:, c]
    )
    G = np.bincount(code, weights=g, minlength=64)
    H = np.bincount(code, weights=h, minlength=64)

    expected = geom.triad_basis(
        S4, y, score, triad, q=4, name="triad:test"
    )
    df, gain, p_ref = geom.triad_score_from_stats(G, H, q=4)

    assert df == expected.df
    assert gain == expected.gain
    assert p_ref == expected.p_ref


def test_native_stage2_reuses_only_solve_cap_stats_and_matches_fallback(monkeypatch):
    def exact_native_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    X, y = make("triad", 317, n=4200, d=8, strength=1.5)
    Xt, yt, Xh, yh = split(X, y)
    cfg = UNDRConfig(
        maximum_features=8,
        budget=2,
        maximum_stage2_triads=7,
    )

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        exact_native_bank,
    )
    native = fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )
    native_meta = native.audit.search_metadata
    assert native_meta["triad_stage2_stats_reuse"] is True
    assert 0 < native_meta["triad_stage2_stats_count"] <= 7
    assert (
        native_meta["triad_stage2_stats_bytes"]
        == native_meta["triad_stage2_stats_count"] * 64 * 2 * 8
    )

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        lambda *args, **kwargs: None,
    )
    fallback = fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    assert native.audit.selected_names == fallback.audit.selected_names
    assert native.audit.selected_families == fallback.audit.selected_families
    assert native.audit.stage2_triad == fallback.audit.stage2_triad
    assert native.audit.triad_exact_solves == fallback.audit.triad_exact_solves
    assert (
        native.audit.search_metadata["triad_exact_diagnostics"]
        == fallback.audit.search_metadata["triad_exact_diagnostics"]
    )



def test_native_stage2_score_path_does_not_materialize_stats_basis(monkeypatch):
    def exact_native_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        exact_native_bank,
    )

    # Search should consume only score-level G/H. Full row-code bases are
    # allowed later when the budgeted candidate is actually materialized.
    score_calls = {"count": 0}
    historical_score = geom.triad_score_from_stats

    def counted_score(*args, **kwargs):
        score_calls["count"] += 1
        return historical_score(*args, **kwargs)

    monkeypatch.setattr(
        "cerm.experimental_undr_v21.geom.triad_score_from_stats",
        counted_score,
    )

    X, y = make("triad", 318, n=3600, d=8, strength=1.45)
    Xt, yt, Xh, yh = split(X, y)
    program = fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=UNDRConfig(
            maximum_features=8,
            budget=2,
            maximum_stage2_triads=9,
        ),
    )
    meta = program.audit.search_metadata
    assert meta["triad_stage2_stats_reuse"] is True
    assert score_calls["count"] == meta["triad_stage2_stats_count"]
    assert score_calls["count"] > 0



def test_resolution_native_state_banks_match_scalar_candidates_exactly(monkeypatch):
    rng = np.random.default_rng(20261101)
    n, d = 2600, 7
    S16 = rng.integers(0, 16, size=(n, d), dtype=np.uint8)
    S8 = S16 // 2
    S4 = S8 // 2
    g = rng.normal(size=n)
    h = np.clip(rng.uniform(0.02, 0.25, size=n), 1e-8, None)
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(d, dtype=np.int64),
        states_by_q={4: S4, 8: S8, 16: S16},
        y=rng.integers(0, 2, size=n, dtype=np.int32),
        score=np.zeros(n, dtype=np.float64),
        g=g,
        h=h,
    )

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_resolution_stat_banks",
        lambda graph: None,
    )
    reference_diagnostics = []
    reference_meta = {}
    reference = resolution_candidates(
        graph,
        0.999,
        diagnostics=reference_diagnostics,
        metadata=reference_meta,
    )

    banks = {}
    total_bytes = 0
    for q, states in graph.states_by_q.items():
        bank = np.empty((d, q, 2), dtype=np.float64)
        for j in range(d):
            bank[j, :, 0] = np.bincount(
                states[:, j], weights=g, minlength=q
            )
            bank[j, :, 1] = np.bincount(
                states[:, j], weights=h, minlength=q
            )
        banks[q] = bank
        total_bytes += bank.nbytes

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_resolution_stat_banks",
        lambda graph: (banks, 4, total_bytes),
    )
    native_diagnostics = []
    native_meta = {}
    actual = resolution_candidates(
        graph,
        0.999,
        diagnostics=native_diagnostics,
        metadata=native_meta,
    )

    assert actual == reference
    assert native_diagnostics == reference_diagnostics
    assert reference_meta["resolution_histogram_backend"] == "scalar"
    assert native_meta["resolution_histogram_backend"] == "native_state_bank"
    assert native_meta["resolution_native_threads"] == 4
    assert native_meta["resolution_native_bank_bytes"] == total_bytes
    assert native_meta["resolution_native_levels"] == [4, 8, 16]
    assert len(graph.basis_stat_cache) == len(actual)
    for candidate in actual:
        assert candidate.name in graph.basis_stat_cache
        G_cached, H_cached = graph.basis_stat_cache[candidate.name]
        np.testing.assert_array_equal(
            G_cached,
            banks[candidate.target_q][candidate.feature, :, 0],
        )
        np.testing.assert_array_equal(
            H_cached,
            banks[candidate.target_q][candidate.feature, :, 1],
        )


def test_resolution_native_bank_rejects_non_nested_state_levels_before_loading_core():
    rng = np.random.default_rng(20261102)
    n, d = 400, 5
    S16 = rng.integers(0, 16, size=(n, d), dtype=np.uint8)
    S8 = S16 // 2
    S4 = (S8 // 2).copy()
    S4[0, 0] = (int(S4[0, 0]) + 1) % 4
    p = np.full(n, 0.5, dtype=np.float64)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(d, dtype=np.int64),
        states_by_q={4: S4, 8: S8, 16: S16},
        y=y,
        score=np.zeros(n, dtype=np.float64),
        g=p - y,
        h=p * (1.0 - p),
    )
    assert _native_resolution_stat_banks(graph) is None



def test_resolution_basis_from_retained_stats_matches_historical_exactly():
    rng = np.random.default_rng(20261120)
    n = 2400
    S16 = rng.integers(0, 16, size=(n, 5), dtype=np.uint8)
    score = rng.normal(scale=1.1, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    feature = 2
    codes = S16[:, feature]
    G = np.bincount(codes, weights=g, minlength=16)
    H = np.bincount(codes, weights=h, minlength=16)

    expected = geom.resolution_basis(
        S16,
        y,
        score,
        feature,
        8,
        16,
        name="q8->q16:test",
    )
    actual = geom.resolution_basis_from_stats(
        S16,
        feature,
        8,
        16,
        G,
        H,
        name="q8->q16:test",
    )

    np.testing.assert_array_equal(actual.codes, expected.codes)
    np.testing.assert_array_equal(actual.A, expected.A)
    np.testing.assert_array_equal(actual.t, expected.t)
    assert actual.df == expected.df
    assert actual.gain == expected.gain
    assert actual.p_ref == expected.p_ref


def test_triad_basis_from_retained_stats_matches_historical_exactly():
    rng = np.random.default_rng(20261121)
    n, d = 2600, 7
    S4 = rng.integers(0, 4, size=(n, d), dtype=np.uint8)
    score = rng.normal(scale=1.0, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    triad = (1, 3, 6)
    a, b, c0 = triad
    codes = (
        (S4[:, a].astype(np.int64) * 4 + S4[:, b]) * 4
        + S4[:, c0]
    )
    G = np.bincount(codes, weights=g, minlength=64)
    H = np.bincount(codes, weights=h, minlength=64)

    expected = geom.triad_basis(
        S4, y, score, triad, q=4, name="triad:test"
    )
    actual = geom.triad_basis_from_stats(
        S4, triad, G, H, q=4, name="triad:test"
    )

    np.testing.assert_array_equal(actual.codes, expected.codes)
    np.testing.assert_array_equal(actual.A, expected.A)
    np.testing.assert_array_equal(actual.t, expected.t)
    assert actual.df == expected.df
    assert actual.gain == expected.gain
    assert actual.p_ref == expected.p_ref


def test_basis_uses_retained_resolution_stats_without_row_rescan(monkeypatch):
    rng = np.random.default_rng(20261122)
    n = 1800
    S16 = rng.integers(0, 16, size=(n, 4), dtype=np.uint8)
    S8 = S16 // 2
    S4 = S8 // 2
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    feature = 1
    G = np.bincount(S16[:, feature], weights=g, minlength=16)
    H = np.bincount(S16[:, feature], weights=h, minlength=16)
    candidate = DeficitCandidate(
        family="resolution",
        name="q8->q16:x1",
        p_ref=1e-6,
        deficit=3.0,
        df=8,
        feature=feature,
        raw_feature=feature,
        source_q=8,
        target_q=16,
    )
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(4, dtype=np.int64),
        states_by_q={4: S4, 8: S8, 16: S16},
        y=y,
        score=score,
        g=g,
        h=h,
        basis_stat_cache={candidate.name: (G.copy(), H.copy())},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("cached basis must not recompute logistic G/H")

    monkeypatch.setattr(geom, "logistic_gh", forbidden)
    basis = _basis(candidate, graph)
    assert basis.df == 8
    assert basis.codes.shape == (n,)



@pytest.mark.parametrize("kind", ["q16", "triad"])
def test_whole_fit_retained_stats_is_exact_noop(monkeypatch, kind):
    import cerm.experimental_undr_v21 as undr

    def exact_resolution_banks(graph):
        banks = {}
        total_bytes = 0
        d = len(graph.selected_raw_features)
        for q, states in graph.states_by_q.items():
            bank = np.empty((d, int(q), 2), dtype=np.float64)
            for j in range(d):
                bank[j, :, 0] = np.bincount(
                    states[:, j],
                    weights=graph.g,
                    minlength=int(q),
                )
                bank[j, :, 1] = np.bincount(
                    states[:, j],
                    weights=graph.h,
                    minlength=int(q),
                )
            banks[int(q)] = bank
            total_bytes += int(bank.nbytes)
        return banks, 4, total_bytes

    def exact_triad_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c0) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c0)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    monkeypatch.setattr(
        undr,
        "_native_resolution_stat_banks",
        exact_resolution_banks,
    )
    monkeypatch.setattr(
        undr,
        "_native_triad_histogram_bank",
        exact_triad_bank,
    )

    X, y = make(kind, 20261130 + len(kind), n=4200, d=8, strength=1.5)
    Xt, yt = X[:3200], y[:3200]
    Xh, yh = X[3200:], y[3200:]
    cfg = UNDRConfig(
        maximum_features=8,
        budget=2,
        maximum_stage2_triads=12,
    )

    cached = undr.fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=cfg,
    )

    historical_build = undr.build_candidate_bank

    def build_then_drop_cache(graph, config):
        bank, family_trials, metadata = historical_build(graph, config)
        graph.basis_stat_cache.clear()
        return bank, family_trials, metadata

    monkeypatch.setattr(
        undr,
        "build_candidate_bank",
        build_then_drop_cache,
    )
    historical = undr.fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=cfg,
    )

    assert cached.audit.selected_names == historical.audit.selected_names
    assert cached.audit.selected_families == historical.audit.selected_families
    assert cached.audit.selection_trace == historical.audit.selection_trace
    assert cached.audit.accepted == historical.audit.accepted
    assert cached.audit.operator_bytes == historical.audit.operator_bytes
    assert cached.audit.holdout_gain == historical.audit.holdout_gain
    assert cached.audit.holdout_radius == historical.audit.holdout_radius
    assert cached.audit.holdout_lcb == historical.audit.holdout_lcb
    np.testing.assert_array_equal(
        cached.decision_function(Xh),
        historical.decision_function(Xh),
    )
    np.testing.assert_array_equal(
        cached.predict_proba(Xh),
        historical.predict_proba(Xh),
    )



def test_span_gain_from_precomputed_gram_matches_historical_exactly():
    rng = np.random.default_rng(20261210)
    n = 2200
    S16 = rng.integers(0, 16, size=(n, 4), dtype=np.uint8)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    bases = [
        geom.resolution_basis(
            S16,
            y,
            score,
            feature,
            8,
            16,
            name=f"b{feature}",
        )
        for feature in (0, 1, 2)
    ]
    _, h = geom.logistic_gh(y, score)
    K, t = geom.gram_and_target(bases, h)
    assert geom.span_gain_from_gram(K, t) == geom.span_gain(bases, h)


def test_batched_conditional_gains_match_historical_exactly(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    rng = np.random.default_rng(20261211)
    n = 2800
    S16 = rng.integers(0, 16, size=(n, 6), dtype=np.uint8)
    score = rng.normal(scale=1.2, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    selected = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"s{feature}"
        )
        for feature in (0, 1)
    ]
    candidates = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"c{feature}"
        )
        for feature in (2, 3, 4, 5)
    ]
    expected = [
        geom.conditional_gain(selected, candidate, h)
        for candidate in candidates
    ]

    def exact_cross(selected_bases, candidate_bases, hessian):
        return [
            np.vstack(
                [
                    geom.cross_gram(left, candidate, hessian)
                    for left in selected_bases
                ]
            )
            for candidate in candidate_bases
        ]

    monkeypatch.setattr(
        undr,
        "_native_conditional_cross_grams",
        exact_cross,
    )
    actual, backend = _conditional_gains_batched(
        selected,
        candidates,
        h,
    )
    assert backend == "native_pair_bank"
    assert actual == expected


def test_batched_conditional_gains_scalar_fallback_is_historical(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    rng = np.random.default_rng(20261212)
    n = 1800
    S16 = rng.integers(0, 16, size=(n, 4), dtype=np.uint8)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    selected = [
        geom.resolution_basis(S16, y, score, 0, 8, 16, name="s0")
    ]
    candidates = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"c{feature}"
        )
        for feature in (1, 2, 3)
    ]
    expected = [
        geom.conditional_gain(selected, candidate, h)
        for candidate in candidates
    ]
    monkeypatch.setattr(
        undr,
        "_native_conditional_cross_grams",
        lambda *args, **kwargs: None,
    )
    actual, backend = _conditional_gains_batched(
        selected,
        candidates,
        h,
    )
    assert backend == "scalar"
    assert actual == expected



def test_whole_fit_batched_conditional_selection_is_exact_noop(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    def exact_resolution_banks(graph):
        banks = {}
        total_bytes = 0
        d = len(graph.selected_raw_features)
        for q, states in graph.states_by_q.items():
            bank = np.empty((d, int(q), 2), dtype=np.float64)
            for j in range(d):
                bank[j, :, 0] = np.bincount(
                    states[:, j], weights=graph.g, minlength=int(q)
                )
                bank[j, :, 1] = np.bincount(
                    states[:, j], weights=graph.h, minlength=int(q)
                )
            banks[int(q)] = bank
            total_bytes += int(bank.nbytes)
        return banks, 4, total_bytes

    def exact_triad_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c0) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c0)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    def exact_cross(selected_bases, candidate_bases, hessian):
        return [
            np.vstack(
                [
                    geom.cross_gram(left, candidate, hessian)
                    for left in selected_bases
                ]
            )
            for candidate in candidate_bases
        ]

    monkeypatch.setattr(
        undr, "_native_resolution_stat_banks", exact_resolution_banks
    )
    monkeypatch.setattr(
        undr, "_native_triad_histogram_bank", exact_triad_bank
    )

    rng = np.random.default_rng(20261220)
    n, d = 5200, 8
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    z16_0 = np.minimum(15, ((X[:, 0] + 1.0) * 8.0).astype(int))
    z16_1 = np.minimum(15, ((X[:, 1] + 1.0) * 8.0).astype(int))
    s0 = np.where(z16_0 % 2 == 0, 1.0, -1.0)
    s1 = np.where(z16_1 % 2 == 0, 1.0, -1.0)
    tri = np.sign(X[:, 2]) * np.sign(X[:, 3]) * np.sign(X[:, 4])
    logit = 1.35 * s0 + 1.15 * s1 + 1.0 * tri
    y = rng.binomial(
        1, 1.0 / (1.0 + np.exp(-logit))
    ).astype(np.int32)
    Xt, yt = X[:4000], y[:4000]
    Xh, yh = X[4000:], y[4000:]
    cfg = UNDRConfig(
        maximum_features=8,
        budget=3,
        maximum_stage2_triads=16,
    )

    monkeypatch.setattr(
        undr, "_native_conditional_cross_grams", exact_cross
    )
    batched = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    monkeypatch.setattr(
        undr,
        "_native_conditional_cross_grams",
        lambda *args, **kwargs: None,
    )
    scalar = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    assert batched.audit.selected_names == scalar.audit.selected_names
    assert batched.audit.selected_families == scalar.audit.selected_families
    assert batched.audit.selection_trace == scalar.audit.selection_trace
    assert batched.audit.accepted == scalar.audit.accepted
    assert batched.audit.operator_bytes == scalar.audit.operator_bytes
    assert batched.audit.holdout_gain == scalar.audit.holdout_gain
    assert batched.audit.holdout_radius == scalar.audit.holdout_radius
    assert batched.audit.holdout_lcb == scalar.audit.holdout_lcb
    np.testing.assert_array_equal(
        batched.decision_function(Xh),
        scalar.decision_function(Xh),
    )
    np.testing.assert_array_equal(
        batched.predict_proba(Xh),
        scalar.predict_proba(Xh),
    )



def test_native_conditional_cross_gram_memory_guard(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    class DummyBasis:
        def __init__(self, rows):
            self.codes = np.zeros(rows, dtype=np.uint8)
            self.nstates = 4
            self.A = np.eye(4)
            self.t = np.zeros(4)
            self.df = 4

    selected = [DummyBasis(16)]
    candidates = [DummyBasis(16), DummyBasis(16)]
    monkeypatch.setattr(
        undr,
        "_CONDITIONAL_STATE_BANK_BUDGET_BYTES",
        1,
    )
    assert undr._native_conditional_cross_grams(
        selected,
        candidates,
        np.ones(16, dtype=np.float64),
    ) is None


def test_native_conditional_cross_gram_single_candidate_supported():
    import cerm.experimental_undr_v21 as undr
    from cerm._internal.cerm_training_core_runtime import native_training_core_supported

    if not native_training_core_supported():
        pytest.skip("native training core unavailable on this platform")

    class DummyBasis:
        def __init__(self):
            self.codes = np.zeros(8, dtype=np.uint8)
            self.nstates = 4
            self.A = np.eye(4)
            self.t = np.zeros(4)
            self.df = 4

    blocks = undr._native_conditional_cross_grams(
        [DummyBasis()],
        [DummyBasis()],
        np.ones(8, dtype=np.float64),
    )
    assert blocks is not None
    assert len(blocks) == 1
    assert blocks[0].shape == (4, 4)



def test_fit_frozen_undr_reports_conditional_workspace_packed_bytes():
    X, y = make("q8", 401, n=3000, d=8, strength=1.5)
    Xt, yt, Xh, yh = split(X, y)
    cfg = UNDRConfig(maximum_features=8, budget=2, maximum_stage2_triads=8)
    prog = fit_frozen_undr(MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg)
    meta = prog.audit.search_metadata
    assert "conditional_workspace_backend" in meta
    assert "conditional_workspace_packed_bytes" in meta
    assert isinstance(meta["conditional_workspace_packed_bytes"], int)
    if prog.audit.selected_names:
        assert meta["conditional_workspace_packed_bytes"] >= 0


def test_incremental_conditional_gram_matches_historical_exactly():
    rng = np.random.default_rng(20261230)
    n = 2600
    S16 = rng.integers(0, 16, size=(n, 6), dtype=np.uint8)
    score = rng.normal(scale=1.0, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    selected = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"s{feature}"
        )
        for feature in (0, 1)
    ]
    candidates = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"c{feature}"
        )
        for feature in (2, 3, 4, 5)
    ]
    K0, t0 = geom.gram_and_target(selected, h)
    cross_blocks = [
        np.vstack(
            [geom.cross_gram(left, candidate, h) for left in selected]
        )
        for candidate in candidates
    ]
    actual = _conditional_gains_from_gram(
        K0,
        t0,
        candidates,
        cross_blocks,
    )
    expected = [
        geom.conditional_gain(selected, candidate, h)
        for candidate in candidates
    ]
    assert [
        (dg, df, p)
        for dg, df, p, _K, _t in actual
    ] == expected


def test_whole_fit_incremental_workspace_is_exact_noop(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    def exact_resolution_banks(graph):
        banks = {}
        total_bytes = 0
        d = len(graph.selected_raw_features)
        for q, states in graph.states_by_q.items():
            bank = np.empty((d, int(q), 2), dtype=np.float64)
            for j in range(d):
                bank[j, :, 0] = np.bincount(
                    states[:, j], weights=graph.g, minlength=int(q)
                )
                bank[j, :, 1] = np.bincount(
                    states[:, j], weights=graph.h, minlength=int(q)
                )
            banks[int(q)] = bank
            total_bytes += int(bank.nbytes)
        return banks, 4, total_bytes

    def exact_triad_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c0) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c0)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    class ExactWorkspace:
        def cross_grams(
            self,
            selected_candidates,
            selected_bases,
            remaining_candidates,
            remaining_bases,
        ):
            return [
                np.vstack(
                    [
                        geom.cross_gram(left, candidate, graph_h)
                        for left in selected_bases
                    ]
                )
                for candidate in remaining_bases
            ]

    monkeypatch.setattr(
        undr, "_native_resolution_stat_banks", exact_resolution_banks
    )
    monkeypatch.setattr(
        undr, "_native_triad_histogram_bank", exact_triad_bank
    )

    rng = np.random.default_rng(20261231)
    n, d = 5200, 8
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    z0 = np.minimum(15, ((X[:, 0] + 1.0) * 8.0).astype(int))
    z1 = np.minimum(15, ((X[:, 1] + 1.0) * 8.0).astype(int))
    z2 = np.minimum(15, ((X[:, 2] + 1.0) * 8.0).astype(int))
    signal = (
        1.35 * np.where(z0 % 2 == 0, 1.0, -1.0)
        + 1.15 * np.where(z1 % 2 == 0, 1.0, -1.0)
        + 1.0 * np.where(z2 % 2 == 0, 1.0, -1.0)
    )
    y = rng.binomial(
        1, 1.0 / (1.0 + np.exp(-signal))
    ).astype(np.int32)
    Xt, yt = X[:4000], y[:4000]
    Xh, yh = X[4000:], y[4000:]
    cfg = UNDRConfig(
        maximum_features=8,
        budget=3,
        maximum_stage2_triads=12,
    )

    original_build = undr._ConditionalPairWorkspace.build
    graph_h = None

    def build_exact(candidates, bases, h):
        nonlocal graph_h
        graph_h = np.asarray(h, dtype=np.float64)
        return ExactWorkspace()

    monkeypatch.setattr(
        undr._ConditionalPairWorkspace,
        "build",
        staticmethod(build_exact),
    )
    incremental = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    monkeypatch.setattr(
        undr._ConditionalPairWorkspace,
        "build",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        undr,
        "_native_conditional_cross_grams",
        lambda *args, **kwargs: None,
    )
    scalar = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    assert incremental.audit.selected_names == scalar.audit.selected_names
    assert incremental.audit.selected_families == scalar.audit.selected_families
    assert incremental.audit.selection_trace == scalar.audit.selection_trace
    assert incremental.audit.accepted == scalar.audit.accepted
    assert incremental.audit.operator_bytes == scalar.audit.operator_bytes
    assert incremental.audit.holdout_gain == scalar.audit.holdout_gain
    assert incremental.audit.holdout_radius == scalar.audit.holdout_radius
    assert incremental.audit.holdout_lcb == scalar.audit.holdout_lcb
    np.testing.assert_array_equal(
        incremental.decision_function(Xh),
        scalar.decision_function(Xh),
    )
    np.testing.assert_array_equal(
        incremental.predict_proba(Xh),
        scalar.predict_proba(Xh),
    )



def test_joint_fit_from_retained_gram_matches_historical_exactly():
    rng = np.random.default_rng(20261310)
    n = 3200
    S16 = rng.integers(0, 16, size=(n, 5), dtype=np.uint8)
    score = rng.normal(scale=1.0, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    bases = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"b{feature}"
        )
        for feature in (0, 1, 2)
    ]
    K, t = geom.gram_and_target(bases, h)
    expected = geom.fit_joint_quotient_lookups(
        bases, h, l2=17.0, lr=0.45
    )
    actual = geom.fit_joint_quotient_lookups_from_gram(
        bases, K, t, l2=17.0, lr=0.45
    )
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_fit_joint_uses_matching_retained_gram_without_row_rescan(monkeypatch):
    rng = np.random.default_rng(20261311)
    n = 2200
    S16 = rng.integers(0, 16, size=(n, 4), dtype=np.uint8)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    bases = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"b{feature}"
        )
        for feature in (0, 1)
    ]
    candidates = [
        DeficitCandidate(
            family="resolution",
            name=f"b{feature}",
            p_ref=1e-6,
            deficit=1.0,
            df=int(bases[feature].df),
            feature=feature,
            raw_feature=feature,
            source_q=8,
            target_q=16,
        )
        for feature in (0, 1)
    ]
    selection = Selection(candidates, bases, [])
    K, t = geom.gram_and_target(bases, h)
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(4, dtype=np.int64),
        states_by_q={16: S16},
        y=y,
        score=score,
        g=g,
        h=h,
        selection_gram_names=tuple(c.name for c in candidates),
        selection_gram_cache=(K.copy(), t.copy()),
    )
    expected = geom.fit_joint_quotient_lookups_from_gram(
        bases, K, t, l2=20.0, lr=0.5
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("matching retained Gram must avoid row rescan")

    monkeypatch.setattr(geom, "fit_joint_quotient_lookups", forbidden)
    actual = fit_joint(selection, graph, UNDRConfig())
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_fit_joint_stale_gram_names_fall_back_historically(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    rng = np.random.default_rng(20261312)
    n = 1800
    S16 = rng.integers(0, 16, size=(n, 3), dtype=np.uint8)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    basis = geom.resolution_basis(
        S16, y, score, 0, 8, 16, name="b0"
    )
    candidate = DeficitCandidate(
        family="resolution",
        name="b0",
        p_ref=1e-6,
        deficit=1.0,
        df=int(basis.df),
        feature=0,
        raw_feature=0,
        source_q=8,
        target_q=16,
    )
    selection = Selection([candidate], [basis], [])
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(3, dtype=np.int64),
        states_by_q={16: S16},
        y=y,
        score=score,
        g=g,
        h=h,
        selection_gram_names=("different",),
        selection_gram_cache=(np.eye(basis.df), basis.t.copy()),
    )
    calls = {"count": 0}
    historical = geom.fit_joint_quotient_lookups

    def counted(*args, **kwargs):
        calls["count"] += 1
        return historical(*args, **kwargs)

    monkeypatch.setattr(geom, "fit_joint_quotient_lookups", counted)
    fit_joint(selection, graph, UNDRConfig())
    assert calls["count"] == 1



def test_whole_fit_retained_final_gram_is_exact_noop(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    def exact_resolution_banks(graph):
        banks = {}
        total_bytes = 0
        d = len(graph.selected_raw_features)
        for q, states in graph.states_by_q.items():
            bank = np.empty((d, int(q), 2), dtype=np.float64)
            for j in range(d):
                bank[j, :, 0] = np.bincount(
                    states[:, j], weights=graph.g, minlength=int(q)
                )
                bank[j, :, 1] = np.bincount(
                    states[:, j], weights=graph.h, minlength=int(q)
                )
            banks[int(q)] = bank
            total_bytes += int(bank.nbytes)
        return banks, 4, total_bytes

    def exact_triad_bank(S4, g, h, triads):
        triads = np.asarray(triads, dtype=np.int32).reshape(-1, 3)
        output = np.empty((len(triads), 64, 2), dtype=np.float64)
        for index, (a, b, c0) in enumerate(triads):
            code = (
                (S4[:, int(a)].astype(np.int64) * 4 + S4[:, int(b)]) * 4
                + S4[:, int(c0)]
            )
            output[index, :, 0] = np.bincount(
                code, weights=g, minlength=64
            )
            output[index, :, 1] = np.bincount(
                code, weights=h, minlength=64
            )
        return output, 4, int(output.nbytes)

    class ExactWorkspace:
        def __init__(self, h):
            self.h = np.asarray(h, dtype=np.float64)

        def cross_grams(
            self,
            selected_candidates,
            selected_bases,
            remaining_candidates,
            remaining_bases,
        ):
            return [
                np.vstack(
                    [
                        geom.cross_gram(left, candidate, self.h)
                        for left in selected_bases
                    ]
                )
                for candidate in remaining_bases
            ]

    monkeypatch.setattr(
        undr, "_native_resolution_stat_banks", exact_resolution_banks
    )
    monkeypatch.setattr(
        undr, "_native_triad_histogram_bank", exact_triad_bank
    )
    monkeypatch.setattr(
        undr._ConditionalPairWorkspace,
        "build",
        staticmethod(
            lambda candidates, bases, h: ExactWorkspace(h)
            if len(candidates) >= 2
            else None
        ),
    )

    rng = np.random.default_rng(20261313)
    n, d = 5200, 8
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    z0 = np.minimum(15, ((X[:, 0] + 1.0) * 8.0).astype(int))
    z1 = np.minimum(15, ((X[:, 1] + 1.0) * 8.0).astype(int))
    z2 = np.minimum(15, ((X[:, 2] + 1.0) * 8.0).astype(int))
    signal = (
        1.4 * np.where(z0 % 2 == 0, 1.0, -1.0)
        + 1.2 * np.where(z1 % 2 == 0, 1.0, -1.0)
        + 1.0 * np.where(z2 % 2 == 0, 1.0, -1.0)
    )
    y = rng.binomial(
        1, 1.0 / (1.0 + np.exp(-signal))
    ).astype(np.int32)
    Xt, yt = X[:4000], y[:4000]
    Xh, yh = X[4000:], y[4000:]
    cfg = UNDRConfig(
        maximum_features=8,
        budget=3,
        maximum_stage2_triads=12,
    )

    retained = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    def force_historical(selection, graph, config):
        return geom.fit_joint_quotient_lookups(
            selection.bases,
            graph.h,
            l2=config.l2,
            lr=config.learning_rate,
        )

    monkeypatch.setattr(undr, "fit_joint", force_historical)
    historical = undr.fit_frozen_undr(
        MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
    )

    assert retained.audit.selected_names == historical.audit.selected_names
    assert retained.audit.selected_families == historical.audit.selected_families
    assert retained.audit.selection_trace == historical.audit.selection_trace
    assert retained.audit.accepted == historical.audit.accepted
    assert retained.audit.operator_bytes == historical.audit.operator_bytes
    assert retained.audit.holdout_gain == historical.audit.holdout_gain
    assert retained.audit.holdout_radius == historical.audit.holdout_radius
    assert retained.audit.holdout_lcb == historical.audit.holdout_lcb
    np.testing.assert_array_equal(
        retained.decision_function(Xh),
        historical.decision_function(Xh),
    )
    np.testing.assert_array_equal(
        retained.predict_proba(Xh),
        historical.predict_proba(Xh),
    )



def test_frozen_program_drops_training_basis_rows_without_mutating_selection():
    import cerm.experimental_undr_v21 as undr

    candidate = DeficitCandidate(
        family="resolution",
        name="q8->q16:x0",
        p_ref=1e-6,
        deficit=2.0,
        df=2,
        feature=0,
        raw_feature=0,
        source_q=8,
        target_q=16,
    )
    basis = geom.StateQuotientBasis(
        name=candidate.name,
        codes=np.arange(128, dtype=np.int64) % 16,
        nstates=16,
        A=np.zeros((16, 2), dtype=np.float64),
        t=np.zeros(2, dtype=np.float64),
        df=2,
        gain=2.0,
        p_ref=1e-6,
        family="resolution",
        meta={},
    )
    selection = Selection(
        candidates=[candidate],
        bases=[basis],
        trace=[{"stage": "first", "candidate": candidate.name}],
    )
    program = undr.FrozenUNDRProgram(
        estimator=object(),
        selected_raw_features=np.asarray([0], dtype=np.int64),
        selection=selection,
        lookups=[np.zeros(16, dtype=np.float64)],
        accepted=True,
        audit=None,
    )

    assert len(selection.bases) == 1
    assert program.selection.candidates == selection.candidates
    assert program.selection.trace == selection.trace
    assert program.selection.bases == []


def test_whole_fit_frozen_program_retains_no_training_bases():
    X, y = make("triad", 20261320, n=3600, d=8, strength=1.5)
    Xt, yt, Xh, yh = split(X, y)
    program = fit_frozen_undr(
        MockEstimator(Xt),
        Xt,
        yt,
        Xh,
        yh,
        config=UNDRConfig(
            maximum_features=8,
            budget=2,
            maximum_stage2_triads=16,
        ),
    )

    assert program.selection.bases == []
    dropped = int(
        program.audit.search_metadata[
            "training_basis_bytes_dropped_at_freeze"
        ]
    )
    if program.selection.candidates:
        assert dropped > 0
    else:
        assert dropped == 0

    decision = program.decision_function(Xh)
    probability = program.predict_proba(Xh)
    assert decision.shape == (len(Xh),)
    assert probability.shape == (len(Xh), 2)
    np.testing.assert_allclose(
        probability[:, 0] + probability[:, 1],
        1.0,
        atol=1e-15,
        rtol=0,
    )



def test_retained_resolution_basis_reuses_state_bank_column_view():
    rng = np.random.default_rng(20261330)
    n = 2400
    S16 = rng.integers(0, 16, size=(n, 5), dtype=np.int16)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    feature = 3
    G = np.bincount(S16[:, feature], weights=g, minlength=16)
    H = np.bincount(S16[:, feature], weights=h, minlength=16)

    basis = geom.resolution_basis_from_stats(
        S16,
        feature,
        8,
        16,
        G,
        H,
        name="view:test",
    )
    assert basis.codes.dtype == S16.dtype
    assert np.shares_memory(basis.codes, S16)
    np.testing.assert_array_equal(basis.codes, S16[:, feature])


def test_retained_triad_basis_compacts_owned_q4_codes_to_uint8():
    rng = np.random.default_rng(20261331)
    n, d = 2600, 7
    S4 = rng.integers(0, 4, size=(n, d), dtype=np.int16)
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    triad = (1, 4, 6)
    historical_codes = (
        (S4[:, triad[0]].astype(np.int64) * 4 + S4[:, triad[1]]) * 4
        + S4[:, triad[2]]
    )
    G = np.bincount(historical_codes, weights=g, minlength=64)
    H = np.bincount(historical_codes, weights=h, minlength=64)

    basis = geom.triad_basis_from_stats(
        S4,
        triad,
        G,
        H,
        q=4,
        name="compact:triad",
    )
    assert basis.codes.dtype == np.uint8
    np.testing.assert_array_equal(basis.codes, historical_codes)


def test_compact_retained_codes_preserve_cross_gram_exactly():
    rng = np.random.default_rng(20261332)
    n = 2800
    S16 = rng.integers(0, 16, size=(n, 4), dtype=np.int16)
    score = rng.normal(scale=1.1, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)

    compact = []
    historical = []
    for feature in (0, 1):
        G = np.bincount(S16[:, feature], weights=g, minlength=16)
        H = np.bincount(S16[:, feature], weights=h, minlength=16)
        compact.append(
            geom.resolution_basis_from_stats(
                S16,
                feature,
                8,
                16,
                G,
                H,
                name=f"compact:{feature}",
            )
        )
        historical.append(
            geom.resolution_basis(
                S16,
                y,
                score,
                feature,
                8,
                16,
                name=f"historical:{feature}",
            )
        )

    actual = geom.cross_gram(compact[0], compact[1], h)
    expected = geom.cross_gram(historical[0], historical[1], h)
    np.testing.assert_array_equal(actual, expected)


def test_scalar_fallback_retains_stat_caches_and_zero_fit_rescans(monkeypatch):
    """Verify scalar fallback retains stat caches and fits from gram without rescans."""
    from cerm._internal.cerm_training_core_runtime import native_training_core_supported

    if not native_training_core_supported():
        pytest.skip("selection Gram reuse requires the native conditional workspace")

    X, y = make("q8", 2026)
    Xt, yt, Xh, yh = split(X, y)
    est = MockEstimator(Xt)

    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_resolution_stat_banks",
        lambda graph: None,
    )
    monkeypatch.setattr(
        "cerm.experimental_undr_v21._native_triad_histogram_bank",
        lambda S4, g, h, triads: None,
    )

    prog = fit_frozen_undr(est, Xt, yt, Xh, yh, config=UNDRConfig(maximum_features=8, budget=2, alpha2=0.5))
    assert prog.audit.search_metadata["resolution_histogram_backend"] == "scalar"
    assert prog.audit.search_metadata["triad_histogram_backend"] == "pair_prefix_cache"
    assert prog.audit.search_metadata["triad_stage2_stats_reuse"] is True
    assert prog.audit.search_metadata["selection_gram_reuse"] is True
    assert prog.audit.search_metadata["selection_stat_cache_count"] >= 1



def test_native_resolution_stat_banks_single_pass_fused_parity():
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    try:
        load_native_training_core()
    except Exception:
        pytest.skip("Native training core unavailable")

    rng = np.random.default_rng(20261103)
    n, d = 3000, 6
    S16 = rng.integers(0, 16, size=(n, d), dtype=np.uint8)
    S8 = S16 // 2
    S4 = S16 // 4
    score = rng.normal(size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    g, h = geom.logistic_gh(y, score)
    graph = ProjectedTrainingGraph(
        selected_raw_features=np.arange(d, dtype=np.int64),
        states_by_q={4: S4, 8: S8, 16: S16},
        y=y,
        score=score,
        g=g,
        h=h,
    )

    result = _native_resolution_stat_banks(graph)
    assert result is not None
    banks, _threads, _bytes = result
    assert set(banks) == {4, 8, 16}

    for j in range(d):
        G16 = np.bincount(S16[:, j], weights=g, minlength=16)
        H16 = np.bincount(S16[:, j], weights=h, minlength=16)
        np.testing.assert_allclose(banks[16][j, :, 0], G16, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(banks[16][j, :, 1], H16, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        banks[8], banks[16].reshape(d, 8, 2, 2).sum(axis=2),
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        banks[4], banks[16].reshape(d, 4, 4, 2).sum(axis=2),
        rtol=1e-12, atol=1e-12,
    )


def test_fused_resolution_banks_preserve_near_tie_candidate_order(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    real_native = undr._native_resolution_stat_banks
    for seed in range(12):
        local = np.random.default_rng(20261104 + seed)
        n, d = 6000, 4
        S16 = local.integers(0, 16, size=(n, d), dtype=np.uint8)
        S16[:, 1] = S16[:, 0]
        row = int(local.integers(0, n))
        S16[row, 1] = np.uint8((int(S16[row, 1]) + 1) % 16)
        S8 = S16 // 2
        S4 = S16 // 4
        score = local.normal(scale=0.7, size=n)
        y = local.integers(0, 2, size=n, dtype=np.int32)
        g, h = geom.logistic_gh(y, score)

        def make_graph():
            return ProjectedTrainingGraph(
                selected_raw_features=np.arange(d, dtype=np.int64),
                states_by_q={4: S4, 8: S8, 16: S16},
                y=y, score=score, g=g, h=h,
            )

        monkeypatch.setattr(undr, "_native_resolution_stat_banks", real_native)
        fused = undr.resolution_candidates(make_graph(), alpha2=1.0)
        monkeypatch.setattr(undr, "_native_resolution_stat_banks", lambda _graph: None)
        scalar = undr.resolution_candidates(make_graph(), alpha2=1.0)

        assert [x.name for x in fused] == [x.name for x in scalar]
        np.testing.assert_allclose(
            [x.p_ref for x in fused], [x.p_ref for x in scalar],
            rtol=1e-11, atol=1e-13,
        )
        np.testing.assert_allclose(
            [x.deficit for x in fused], [x.deficit for x in scalar],
            rtol=1e-11, atol=1e-13,
        )

def test_undr_fused_state_transforms_bitwise_equality():
    from cerm import CERMClassifier

    rng = np.random.default_rng(42)
    n, d = 2000, 10
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    clf = CERMClassifier(max_features=8, random_state=42)
    clf.fit(X, y)
    base = clf.model_.base_
    encoder = base.encoder_
    raw = base.feature_idx_[:8]

    fused_states, fused_paths = _state_transforms(
        encoder, False, X, (4, 8, 16), raw
    )
    assert set(fused_states) == {4, 8, 16}
    assert all(path == "transform_columns_fused" for path in fused_paths.values())
    for q in (4, 8, 16):
        single_state, single_path = _state_transforms(
            encoder, False, X, [q], raw
        )
        assert single_path[q] == "transform_level_columns"
        np.testing.assert_array_equal(fused_states[q], single_state[q])


def test_undr_fused_state_transforms_finite_state_direct_features():
    from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder

    rng = np.random.default_rng(123)
    n = 1500
    X = np.hstack([
        rng.standard_normal((n, 4)),
        rng.integers(0, 4, size=(n, 4)),
    ])
    encoder = NestedQuantileEncoder(
        max_bins=16,
        levels=(4, 8, 16),
        feature_kinds=["numeric"] * 4 + ["categorical_identity"] * 4,
        feature_cardinalities=[None] * 4 + [4] * 4,
    ).fit(X)
    assert not np.any(encoder.direct_state_mask_[:4])
    assert np.all(encoder.direct_state_mask_[4:])
    raw = np.arange(X.shape[1], dtype=np.int64)

    fused_states, fused_paths = _state_transforms(
        encoder, False, X, (4, 8, 16), raw
    )
    assert all(path == "transform_columns_fused" for path in fused_paths.values())
    for q in (4, 8, 16):
        single_state, single_paths = _state_transforms(
            encoder, False, X, [q], raw
        )
        assert single_paths[q] == "transform_level_columns"
        np.testing.assert_array_equal(fused_states[q], single_state[q])


def test_undr_multilevel_projection_benchmark():
    import time
    from cerm import CERMClassifier

    rng = np.random.default_rng(20261021)
    n, d = 50000, 20
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    clf = CERMClassifier(max_features=12, random_state=20261021)
    clf.fit(X, y)
    base = clf.model_.base_
    encoder = base.encoder_
    raw = base.feature_idx_[:12]

    t0 = time.perf_counter()
    fused_states, fused_paths = _state_transforms(
        encoder, False, X, (4, 8, 16), raw
    )
    fused_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fallback_states = {
        q: _compact_state_level(
            encoder.transform_level_columns(X, q, raw),
            q,
        )
        for q in (4, 8, 16)
    }
    fallback_time = time.perf_counter() - t0

    for q in (4, 8, 16):
        np.testing.assert_array_equal(fused_states[q], fallback_states[q])
    speedup = fallback_time / max(fused_time, 1e-9)
    print(
        f"\n[UNDR Multi-level State Extraction Benchmark]\n"
        f"Rows: {n}, Selected Features: {len(raw)}\n"
        f"3-Level Fused Time: {fused_time * 1000:.2f} ms ({fused_paths[4]})\n"
        f"3-Level Fallback Time: {fallback_time * 1000:.2f} ms "
        f"(three transform_level_columns calls)\n"
        f"Speedup: {speedup:.2f}x\n"
        f"State Equality: EXACT MATCH (bitwise identical across all levels)"
    )
    assert fused_time > 0.0
    assert fallback_time > 0.0



def test_fused_native_triad_stage1_parity_across_configs(monkeypatch):
    import cerm.experimental_undr_v21 as undr

    try:
        from cerm._internal.cerm_training_core_runtime import load_native_training_core
        core = load_native_training_core()
    except Exception:
        pytest.skip("native training core unavailable")
    if getattr(core, "_triad_fused_stage1", None) is None:
        pytest.skip("fused triad symbol missing")

    for seed in (320, 321, 322):
        X, y = make("triad", seed, n=3000, d=8, strength=1.4)
        Xt, yt, Xh, yh = split(X, y)
        cfg = UNDRConfig(maximum_features=8, budget=2, maximum_stage2_triads=16)
        fused_prog = fit_frozen_undr(
            MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
        )
        assert fused_prog.audit.search_metadata["triad_stage1_fused"] is True

        with monkeypatch.context() as m:
            m.setattr(
                undr,
                "_native_fused_triad_stage1",
                lambda *args, **kwargs: None,
            )
            unfused_prog = fit_frozen_undr(
                MockEstimator(Xt), Xt, yt, Xh, yh, config=cfg
            )
            assert unfused_prog.audit.search_metadata["triad_stage1_fused"] is False

        assert fused_prog.audit.selected_names == unfused_prog.audit.selected_names
        assert fused_prog.audit.selected_families == unfused_prog.audit.selected_families
        assert fused_prog.audit.stage2_triad == unfused_prog.audit.stage2_triad
        assert fused_prog.audit.triad_exact_solves == unfused_prog.audit.triad_exact_solves
        np.testing.assert_array_equal(
            fused_prog.decision_function(Xh),
            unfused_prog.decision_function(Xh),
        )
