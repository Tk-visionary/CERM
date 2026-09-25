from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score

from cerm._internal.cerm_hierarchical_residual import (
    _binary_mutual_info,
    _pair_parent_map,
    _rank_pairs_diversified,
    _residual_state_map,
)


def _residual_reference(child_to_parent):
    child_to_parent = np.asarray(child_to_parent, dtype=np.int64)
    mapping = np.zeros(len(child_to_parent), dtype=np.int32)
    next_code = 1
    for parent in np.unique(child_to_parent):
        children = np.flatnonzero(child_to_parent == parent)
        for child in children[1:]:
            mapping[int(child)] = next_code
            next_code += 1
    return mapping


def test_binary_mutual_info_matches_sklearn():
    rng = np.random.default_rng(17)
    for card in (2, 4, 8, 31):
        code = rng.integers(0, card, size=500, dtype=np.int64)
        y = rng.integers(0, 2, size=500, dtype=np.int64)
        expected = mutual_info_score(y, code)
        actual = _binary_mutual_info(code, y)
        assert abs(actual - expected) < 1e-12


def test_residual_state_vectorization_preserves_reference_order():
    rng = np.random.default_rng(23)
    for n in (0, 1, 4, 16, 64):
        labels = rng.integers(0, max(1, n // 3 + 1), size=n, dtype=np.int64)
        np.testing.assert_array_equal(
            _residual_state_map(labels),
            _residual_reference(labels),
        )


def test_pair_parent_map_matches_nested_loop():
    map_j = np.asarray([0, 0, 1, 2], dtype=np.int64)
    map_k = np.asarray([0, 2, 1], dtype=np.int64)
    expected = []
    for a in range(len(map_j)):
        for b in range(len(map_k)):
            expected.append(int(map_j[a]) * 3 + int(map_k[b]))
    np.testing.assert_array_equal(
        _pair_parent_map(len(map_j), len(map_k), map_j, map_k, 3),
        np.asarray(expected, dtype=np.int64),
    )


def test_diversified_pair_bank_is_deterministic_bounded_and_row_preserving():
    rng = np.random.default_rng(127)
    states = rng.integers(0, 8, size=(240, 10), dtype=np.int16)
    target = rng.integers(0, 2, size=len(states), dtype=np.int32)
    first = _rank_pairs_diversified(
        states,
        target,
        max_pairs=7,
        feature_limit=8,
        random_state=31,
    )
    second = _rank_pairs_diversified(
        states,
        target,
        max_pairs=7,
        feature_limit=8,
        random_state=31,
    )
    assert first == second
    assert len(first) <= 7
    assert len(first) == len(set(first))
    assert all(0 <= left < right < 8 for left, right in first)


def test_encoded_column_bank_matches_individual_reference_encoders():
    from cerm.training_graph import EncodedColumnBank
    from cerm._internal.cerm_state_design import ReferenceStateEncoder

    rng = np.random.default_rng(101)
    train = np.column_stack([
        rng.integers(0, 4, size=120),
        rng.integers(0, 8, size=120),
        rng.integers(0, 3, size=120),
        rng.integers(0, 6, size=120),
    ])
    valid = np.column_stack([
        rng.integers(0, 4, size=40),
        rng.integers(0, 8, size=40),
        rng.integers(0, 3, size=40),
        rng.integers(0, 6, size=40),
    ])
    bank = EncodedColumnBank.build(train, valid, ReferenceStateEncoder)
    for columns in ([0, 2], [1, 3], [0, 1, 2, 3]):
        expected_encoder = ReferenceStateEncoder()
        expected_train = expected_encoder.fit_transform(train[:, columns])
        expected_valid = expected_encoder.transform(valid[:, columns])
        actual_train, actual_valid = bank.view(columns)
        np.testing.assert_array_equal(actual_train.toarray(), expected_train.toarray())
        np.testing.assert_array_equal(actual_valid.toarray(), expected_valid.toarray())


def test_logistic_path_matches_separate_liblinear_fits():
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from cerm.training_graph import solve_binary_logistic_path

    rng = np.random.default_rng(103)
    X = sparse.csr_matrix(rng.normal(size=(180, 17)))
    y = (rng.normal(size=180) + X[:, 0].toarray().ravel() > 0).astype(int)
    path = solve_binary_logistic_path(
        X[:130], y[:130], X[130:], [0.2, 1.0],
        random_state=11, max_iter=1500,
    )
    for C in (0.2, 1.0):
        model = LogisticRegression(
            C=C, solver="liblinear", max_iter=1500, random_state=11
        ).fit(X[:130], y[:130])
        np.testing.assert_array_equal(path[C].coefficient, model.coef_.ravel())
        assert path[C].intercept == model.intercept_[0]
        np.testing.assert_allclose(
            path[C].valid_probability,
            model.predict_proba(X[130:])[:, 1],
            atol=2e-15,
            rtol=0,
        )


def test_block_column_bank_prefix_matches_individual_builder():
    from cerm.training_graph import BlockColumnBank
    from cerm._internal.cerm_quotient_block import QuotientBlockCERM

    rng = np.random.default_rng(107)
    n_train, n_valid, d = 180, 60, 5
    C4t = rng.integers(0, 4, size=(n_train, d), dtype=np.int16)
    C16t = rng.integers(0, 8, size=(n_train, d), dtype=np.int16)
    C4v = rng.integers(0, 4, size=(n_valid, d), dtype=np.int16)
    C16v = rng.integers(0, 8, size=(n_valid, d), dtype=np.int16)
    p = np.clip(rng.uniform(0.1, 0.9, size=n_train), 1e-6, 1 - 1e-6)
    terms = [
        (0, 1, 2, 8, 3.0),
        (1, 2, 3, 8, 2.0),
        (2, 0, 4, 8, 1.0),
    ]
    bank = BlockColumnBank.build(C4t, C16t, C4v, C16v, p, terms, 10.0)
    model = QuotientBlockCERM(max_features=d, pair_feature_limit=d)
    for k in range(4):
        model.block_specs_ = []
        expected_train = model._block_matrix_fit(C4t, C16t, p, terms[:k], 10.0)
        expected_valid = model._block_matrix_transform(C4v, C16v)
        view = bank.view(k)
        np.testing.assert_allclose(view.train.toarray(), expected_train.toarray(), atol=0, rtol=0)
        np.testing.assert_allclose(view.valid.toarray(), expected_valid.toarray(), atol=0, rtol=0)
        assert list(view.specs) == model.block_specs_


def test_cross_fitted_selection_evaluates_nonzero_block_designs():
    from sklearn.datasets import make_classification
    from cerm import CERMClassifier

    X, y = make_classification(
        n_samples=180,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=109,
    )
    model = CERMClassifier(
        max_features=7,
        pair_feature_limit=7,
        random_state=109,
        selection_strategy="cross_fitted",
        selection_folds=3,
    ).fit(X, y)
    rows = model.model_.cv_selection_results_
    zero_bytes = min(row["mean_bytes"] for row in rows if row["n_blocks"] == 0)
    nonzero = [row for row in rows if row["n_blocks"] > 0]
    assert nonzero
    assert any(row["mean_bytes"] > zero_bytes for row in nonzero)


def test_exact_binary_model_matches_public_liblinear_estimator():
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from cerm.training_graph import fit_binary_logistic_exact

    rng = np.random.default_rng(113)
    X = sparse.random(
        240, 80, density=0.12, format="csr", random_state=113, dtype=np.float64
    )
    y = rng.integers(0, 2, size=240, dtype=np.int32)
    exact = fit_binary_logistic_exact(
        X, y, C=0.2, random_state=19, max_iter=1500
    )
    reference = LogisticRegression(
        C=0.2, solver="liblinear", random_state=19, max_iter=1500
    ).fit(X, y)
    np.testing.assert_array_equal(exact.coef_, reference.coef_)
    np.testing.assert_array_equal(exact.intercept_, reference.intercept_)
    np.testing.assert_array_equal(exact.n_iter_, reference.n_iter_)
    np.testing.assert_array_equal(
        exact.predict_proba(X[:40]), reference.predict_proba(X[:40])
    )


def test_training_graph_artifacts_are_ephemeral():
    from sklearn.datasets import make_classification
    from cerm._internal.cerm_hierarchical_residual import HierarchicalResidualCERM
    from cerm._internal.cerm_hybrid_quotient_block import HybridQuotientBlockCERM

    X, y = make_classification(
        n_samples=140, n_features=6, n_informative=4, random_state=127
    )
    backbone = HierarchicalResidualCERM(
        max_features=6, pair_feature_limit=6, random_state=127
    ).fit(X, y)
    assert not hasattr(backbone, "_fit_training_states_")
    assert not hasattr(backbone, "_fit_training_codes_")
    assert not hasattr(backbone, "_fit_training_design_")

    hybrid = HybridQuotientBlockCERM(
        max_features=6, pair_feature_limit=6, random_state=127
    ).fit(X, y)
    assert not hasattr(hybrid.base_, "_fit_training_states_")
    assert not hasattr(hybrid.base_, "_fit_training_codes_")
    assert not hasattr(hybrid.base_, "_fit_training_design_")
