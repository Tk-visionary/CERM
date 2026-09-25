from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


def _semantic_fixture(seed=20260922):
    from cerm.training_graph import BlockColumnBank, SemanticBlockDictionary

    rng = np.random.default_rng(seed)
    n_train, n_valid, d = 320, 120, 7
    C4t = rng.integers(0, 4, size=(n_train, d), dtype=np.uint8)
    C4v = rng.integers(0, 4, size=(n_valid, d), dtype=np.uint8)
    C16t = rng.integers(0, 12, size=(n_train, d), dtype=np.uint8)
    C16v = rng.integers(0, 12, size=(n_valid, d), dtype=np.uint8)
    p_base = np.clip(rng.uniform(0.08, 0.92, size=n_train), 1e-6, 1 - 1e-6)
    terms = [
        (0, 1, 2, 12, 4.0),
        (1, 2, 3, 12, 3.0),
        (2, 0, 4, 12, 2.0),
        (3, 3, 5, 12, 1.0),
    ]
    bank = BlockColumnBank.build(
        C4t,
        C16t,
        C4v,
        C16v,
        p_base,
        terms,
        10.0,
    )
    view = bank.view(len(terms))
    semantic = SemanticBlockDictionary.from_specs(C4t, C4v, view.specs)
    return rng, C4t, C16t, C4v, C16v, view, semantic


def test_semantic_spec_only_builder_matches_block_bank_specs():
    from cerm.training_graph import SemanticBlockDictionary

    rng, C4t, C16t, C4v, _C16v, view, _semantic = _semantic_fixture(20260924)
    p_base = np.clip(rng.uniform(0.08, 0.92, size=len(C4t)), 1e-6, 1 - 1e-6)
    terms = [
        (0, 1, 2, 12, 4.0),
        (1, 2, 3, 12, 3.0),
        (2, 0, 4, 12, 2.0),
        (3, 3, 5, 12, 1.0),
    ]

    from cerm.training_graph import BlockColumnBank

    bank = BlockColumnBank.build(
        C4t, C16t, C4v, _C16v, p_base, terms, 10.0
    )
    expected = bank.view(len(terms))
    actual = SemanticBlockDictionary.build(
        C4t,
        C16t,
        C4v,
        p_base,
        terms,
        10.0,
    )

    assert list(actual.specs) == list(expected.specs)
    assert actual.n_columns == expected.train.shape[1]


def test_semantic_dictionary_reproduces_materialized_block_linear_response():
    rng, _C4t, _C16t, _C4v, C16v, view, semantic = _semantic_fixture()
    coefficient = rng.normal(size=view.valid.shape[1])
    expected = np.asarray(view.valid @ coefficient).ravel()
    empty_base = sparse.csr_matrix((len(C16v), 0), dtype=np.float64)
    actual = semantic.valid_linear_response(
        empty_base,
        C16v,
        coefficient,
        0.0,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)


def test_semantic_tron_matches_materialized_csr_solver_within_guard_tolerance():
    from cerm.training_graph import (
        solve_binary_logistic_path,
        solve_binary_logistic_semantic_blocks,
        semantic_block_solver_available,
    )

    rng, C4t, C16t, C4v, C16v, view, semantic = _semantic_fixture(20260923)
    base_train = sparse.random(
        len(C4t),
        24,
        density=0.16,
        format="csr",
        random_state=20260923,
        dtype=np.float64,
    )
    base_valid = sparse.random(
        len(C4v),
        24,
        density=0.16,
        format="csr",
        random_state=20260924,
        dtype=np.float64,
    )
    base_train.indices = base_train.indices.astype(np.int32, copy=False)
    base_train.indptr = base_train.indptr.astype(np.int32, copy=False)
    base_valid.indices = base_valid.indices.astype(np.int32, copy=False)
    base_valid.indptr = base_valid.indptr.astype(np.int32, copy=False)

    full_train = sparse.hstack([base_train, view.train], format="csr")
    full_valid = sparse.hstack([base_valid, view.valid], format="csr")
    y = rng.integers(0, 2, size=len(C4t), dtype=np.int32)
    if len(np.unique(y)) != 2:
        y[:2] = np.asarray([0, 1], dtype=np.int32)

    reference = solve_binary_logistic_path(
        full_train,
        y,
        full_valid,
        [0.2],
        random_state=17,
        max_iter=1800,
    )[0.2]
    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")
    actual = solve_binary_logistic_semantic_blocks(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        C=0.2,
        random_state=17,
        max_iter=1800,
    )

    assert actual.coefficient.shape == reference.coefficient.shape
    np.testing.assert_allclose(
        actual.coefficient,
        reference.coefficient,
        atol=3e-4,
        rtol=0,
    )
    assert abs(actual.intercept - reference.intercept) <= 3e-4
    np.testing.assert_allclose(
        actual.valid_probability,
        reference.valid_probability,
        atol=1.5e-4,
        rtol=0,
    )



def test_semantic_tron_supports_noncontiguous_state_ids():
    from cerm.training_graph import (
        SemanticBlockDictionary,
        solve_binary_logistic_path,
        solve_binary_logistic_semantic_blocks,
        semantic_block_solver_available,
    )

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    rng = np.random.default_rng(20260925)
    n_train, n_valid = 220, 80
    C4t = np.zeros((n_train, 2), dtype=np.uint8)
    C4v = np.zeros((n_valid, 2), dtype=np.uint8)
    C16t = rng.integers(0, 7, size=(n_train, 2), dtype=np.uint8)
    C16v = rng.integers(0, 7, size=(n_valid, 2), dtype=np.uint8)

    # Deliberately omit states 1, 3, 4, and 6 from the realized effect columns.
    spec = {
        "gate_j": 0,
        "gate_state": 0,
        "target_k": 1,
        "target_card": 7,
        "states": [0, 2, 5],
        "centers": [0.18, 0.21, 0.16],
        "scales": [0.91, 0.87, 0.94],
        "gain": 1.0,
    }
    semantic = SemanticBlockDictionary.from_specs(C4t, C4v, [spec])

    def materialize(states):
        target = states[:, 1]
        cols = []
        for state, center, scale in zip(
            spec["states"], spec["centers"], spec["scales"]
        ):
            cols.append(((target == state).astype(float) - center) * scale)
        return sparse.csr_matrix(np.column_stack(cols))

    block_train = materialize(C16t)
    block_valid = materialize(C16v)
    base_train = sparse.random(
        n_train, 9, density=0.18, format="csr",
        random_state=20260925, dtype=np.float64,
    )
    base_valid = sparse.random(
        n_valid, 9, density=0.18, format="csr",
        random_state=20260926, dtype=np.float64,
    )
    for matrix in (base_train, base_valid):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)

    full_train = sparse.hstack([base_train, block_train], format="csr")
    full_valid = sparse.hstack([base_valid, block_valid], format="csr")
    y = rng.integers(0, 2, size=n_train, dtype=np.int32)
    y[:2] = np.asarray([0, 1], dtype=np.int32)

    reference = solve_binary_logistic_path(
        full_train, y, full_valid, [0.2],
        random_state=23, max_iter=1800,
    )[0.2]
    actual = solve_binary_logistic_semantic_blocks(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        C=0.2,
        random_state=23,
        max_iter=1800,
    )

    np.testing.assert_allclose(
        actual.coefficient,
        reference.coefficient,
        atol=3e-4,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.valid_probability,
        reference.valid_probability,
        atol=1.5e-4,
        rtol=0,
    )



def test_semantic_final_fit_wrapper_matches_materialized_model():
    from cerm.training_graph import (
        fit_binary_logistic_exact,
        fit_binary_logistic_semantic_exact,
        semantic_block_solver_available,
    )

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    rng, C4t, C16t, _C4v, _C16v, view, semantic = _semantic_fixture(20260926)
    base = sparse.random(
        len(C4t),
        21,
        density=0.14,
        format="csr",
        random_state=20260926,
        dtype=np.float64,
    )
    base.indices = base.indices.astype(np.int32, copy=False)
    base.indptr = base.indptr.astype(np.int32, copy=False)
    full = sparse.hstack([base, view.train], format="csr")
    y = rng.integers(0, 2, size=len(C4t), dtype=np.int32)
    y[:2] = np.asarray([0, 1], dtype=np.int32)

    reference = fit_binary_logistic_exact(
        full,
        y,
        C=0.2,
        random_state=29,
        max_iter=1800,
    )
    actual = fit_binary_logistic_semantic_exact(
        base,
        C16t,
        semantic,
        y,
        C=0.2,
        random_state=29,
        max_iter=1800,
    )

    assert actual.classes_.tolist() == [0, 1]
    assert actual.n_features_in_ == full.shape[1]
    np.testing.assert_allclose(
        actual.coef_,
        reference.coef_,
        atol=3e-4,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.intercept_,
        reference.intercept_,
        atol=3e-4,
        rtol=0,
    )
    assert abs(int(actual.n_iter_[0]) - int(reference.n_iter_[0])) <= 1



def test_semantic_prefix_bank_matches_independent_prefix_builds():
    from cerm.training_graph import SemanticBlockBank, SemanticBlockDictionary

    rng = np.random.default_rng(20260927)
    n_train, n_valid, d = 260, 90, 6
    C4t = rng.integers(0, 4, size=(n_train, d), dtype=np.uint8)
    C4v = rng.integers(0, 4, size=(n_valid, d), dtype=np.uint8)
    C16t = rng.integers(0, 10, size=(n_train, d), dtype=np.uint8)
    p_base = np.clip(rng.uniform(0.08, 0.92, size=n_train), 1e-6, 1 - 1e-6)
    weight = np.clip(rng.lognormal(0.0, 0.8, size=n_train), 1e-3, 1e3)
    terms = [
        (0, 1, 2, 10, 5.0),
        (1, 2, 3, 10, 4.0),
        (2, 0, 4, 10, 3.0),
        (3, 3, 5, 10, 2.0),
        (4, 1, 0, 10, 1.0),
    ]
    bank = SemanticBlockBank.build(
        C4t,
        C16t,
        C4v,
        p_base,
        terms,
        10.0,
        sample_weight=weight,
    )
    for k in range(len(terms) + 1):
        expected = SemanticBlockDictionary.build(
            C4t,
            C16t,
            C4v,
            p_base,
            terms[:k],
            10.0,
            sample_weight=weight,
        )
        actual = bank.view(k)
        assert list(actual.specs) == list(expected.specs)
        assert actual.n_columns == expected.n_columns
        np.testing.assert_array_equal(actual.block_target, expected.block_target)
        np.testing.assert_array_equal(actual.coef_offsets, expected.coef_offsets)
        np.testing.assert_array_equal(actual.state_ids, expected.state_ids)
        np.testing.assert_array_equal(actual.centers, expected.centers)
        np.testing.assert_array_equal(actual.scales, expected.scales)
        np.testing.assert_array_equal(
            actual.train_row_offsets, expected.train_row_offsets
        )
        np.testing.assert_array_equal(actual.train_rows, expected.train_rows)


def test_semantic_multi_c_path_matches_separate_semantic_solves():
    from cerm.training_graph import (
        solve_binary_logistic_semantic_blocks,
        solve_binary_logistic_semantic_path,
        semantic_block_solver_available,
    )

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    rng, C4t, C16t, C4v, C16v, _view, semantic = _semantic_fixture(20260928)
    base_train = sparse.random(
        len(C4t), 18, density=0.15, format="csr",
        random_state=20260928, dtype=np.float64,
    )
    base_valid = sparse.random(
        len(C4v), 18, density=0.15, format="csr",
        random_state=20260929, dtype=np.float64,
    )
    for matrix in (base_train, base_valid):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    y = rng.integers(0, 2, size=len(C4t), dtype=np.int32)
    y[:2] = np.asarray([0, 1], dtype=np.int32)

    Cs = [0.2, 1.0]
    path = solve_binary_logistic_semantic_path(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        Cs,
        random_state=31,
        max_iter=1800,
        n_jobs=2,
    )
    for C in Cs:
        expected = solve_binary_logistic_semantic_blocks(
            base_train,
            C16t,
            semantic,
            y,
            base_valid,
            C16v,
            C=C,
            random_state=31,
            max_iter=1800,
        )
        np.testing.assert_array_equal(
            path[C].coefficient, expected.coefficient
        )
        assert path[C].intercept == expected.intercept
        np.testing.assert_array_equal(
            path[C].valid_probability, expected.valid_probability
        )
        assert path[C].n_iter == expected.n_iter



def test_semantic_tron_weighted_zero_rows_match_materialized_solver():
    from cerm.training_graph import (
        solve_binary_logistic_path,
        solve_binary_logistic_semantic_blocks,
        semantic_block_solver_available,
    )

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    rng, C4t, C16t, C4v, C16v, view, semantic = _semantic_fixture(20260930)
    base_train = sparse.random(
        len(C4t), 20, density=0.13, format="csr",
        random_state=20260930, dtype=np.float64,
    )
    base_valid = sparse.random(
        len(C4v), 20, density=0.13, format="csr",
        random_state=20261001, dtype=np.float64,
    )
    for matrix in (base_train, base_valid):
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)

    full_train = sparse.hstack([base_train, view.train], format="csr")
    full_valid = sparse.hstack([base_valid, view.valid], format="csr")
    y = rng.integers(0, 2, size=len(C4t), dtype=np.int32)
    y[:2] = np.asarray([0, 1], dtype=np.int32)
    weight = np.clip(rng.lognormal(0.0, 1.1, size=len(y)), 1e-3, 1e3)
    weight[rng.choice(len(y), size=max(1, len(y) // 12), replace=False)] = 0.0
    # Keep both classes represented among positive-weight rows.
    weight[0] = max(weight[0], 1.0)
    weight[1] = max(weight[1], 1.0)

    reference = solve_binary_logistic_path(
        full_train,
        y,
        full_valid,
        [0.2],
        random_state=37,
        max_iter=1800,
        sample_weight=weight,
    )[0.2]
    actual = solve_binary_logistic_semantic_blocks(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        C=0.2,
        random_state=37,
        max_iter=1800,
        sample_weight=weight,
    )

    np.testing.assert_allclose(
        actual.coefficient,
        reference.coefficient,
        atol=4e-4,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.valid_probability,
        reference.valid_probability,
        atol=2e-4,
        rtol=0,
    )



def test_hybrid_primary_selection_uses_semantic_banks_without_block_materialization():
    from sklearn.datasets import make_classification
    from cerm._internal.cerm_hybrid_quotient_block import HybridQuotientBlockCERM
    from cerm.training_graph import semantic_block_solver_available

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        random_state=20261002,
    )
    model = HybridQuotientBlockCERM(
        max_features=8,
        pair_feature_limit=8,
        max_interactions=8,
        search_profile="practical",
        random_state=20261002,
        n_jobs=2,
    ).fit(X, y)

    assert model.primary_semantic_bank_count_ > 0
    assert model.primary_column_bank_count_ == 0
    assert model.final_solver_backend_ in {
        "semantic_blocks",
        "materialized_csr",
    }
    probability = model.predict_proba(X[:25])
    assert probability.shape == (25, 2)
    assert np.all(np.isfinite(probability))
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)



def test_hybrid_whole_fit_semantic_matches_materialized_fallback(monkeypatch):
    from sklearn.datasets import load_breast_cancer
    import cerm._internal.cerm_hybrid_quotient_block as hybrid_module
    from cerm._internal.cerm_hybrid_quotient_block import HybridQuotientBlockCERM
    from cerm.training_graph import semantic_block_solver_available

    if not semantic_block_solver_available():
        pytest.skip("source-built semantic block solver is unavailable")

    X, y = load_breast_cancer(return_X_y=True)
    common = dict(
        max_features=12,
        pair_feature_limit=10,
        max_interactions=8,
        search_profile="practical",
        fixed_C=0.2,
        random_state=20261003,
        n_jobs=2,
    )

    semantic = HybridQuotientBlockCERM(**common).fit(X, y)
    with monkeypatch.context() as patch:
        patch.setattr(
            hybrid_module,
            "semantic_block_solver_available",
            lambda: False,
        )
        materialized = HybridQuotientBlockCERM(**common).fit(X, y)

    assert semantic.primary_semantic_bank_count_ > 0
    assert materialized.primary_column_bank_count_ > 0
    assert semantic.primary_selected_config_ == materialized.primary_selected_config_
    assert semantic.selected_hybrid_config_ == materialized.selected_hybrid_config_
    np.testing.assert_allclose(
        np.asarray([row[0] for row in semantic.primary_scores_]),
        np.asarray([row[0] for row in materialized.primary_scores_]),
        atol=2e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        semantic.predict_proba(X),
        materialized.predict_proba(X),
        atol=2e-7,
        rtol=0.0,
    )



def test_semantic_native_core_prefers_runtime_loader(monkeypatch):
    import cerm.training_graph as graph
    from cerm._internal import cerm_training_core_runtime as runtime

    class FakeCore:
        _binary_logistic_semantic_blocks = object()

    sentinel = FakeCore()
    graph._semantic_block_native_core.cache_clear()
    monkeypatch.setattr(runtime, "load_native_training_core", lambda: sentinel)
    try:
        assert graph._semantic_block_native_core() is sentinel
        assert graph.semantic_block_solver_available()
    finally:
        graph._semantic_block_native_core.cache_clear()


def test_semantic_native_core_missing_optional_symbol_falls_back(monkeypatch):
    import cerm.training_graph as graph
    from cerm._internal import cerm_training_core_runtime as runtime

    class OldCore:
        _binary_logistic_semantic_blocks = None

    graph._semantic_block_native_core.cache_clear()
    monkeypatch.setattr(runtime, "load_native_training_core", lambda: OldCore())
    try:
        assert not graph.semantic_block_solver_available()
    finally:
        graph._semantic_block_native_core.cache_clear()
