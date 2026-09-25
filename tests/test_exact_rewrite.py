import numpy as np
from sklearn.datasets import load_digits

from cerm import CERMClassifier
from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder


def test_selected_state_transform_is_exact_projection():
    X, _ = load_digits(return_X_y=True)
    X = X[:200]
    encoder = NestedQuantileEncoder().fit(X)
    columns = np.asarray([7, 1, 31, 5], dtype=np.int64)
    full = encoder.transform(X)
    selected = encoder.transform_columns(X, columns)
    for level in encoder.levels:
        assert np.array_equal(selected[level], full[level][:, columns])


def test_binary_runtime_uses_exact_projected_state_columns():
    rng = np.random.default_rng(20260811)
    X = rng.normal(size=(180, 40))
    y = (X[:, 0] - 0.7 * X[:, 1] + 0.2 * rng.normal(size=len(X)) > 0).astype(int)
    model = CERMClassifier(
        search_effort="balanced",
        state_detail="medium",
        feature_budget=6,
        interaction_search_features=8,
        interaction_budget=4,
        prediction_backend="semantic",
        random_state=20260811,
    ).fit(X, y)

    internal = model.model_
    base = internal.base_
    all_states = base.encoder_.transform(X[:32])
    projected = internal._states(X[:32])
    for level in base.encoder_.levels:
        assert np.array_equal(
            projected[level], all_states[level][:, base.feature_idx_]
        )

    # The exact projected states must feed the same semantic code layout.
    expected_codes = base._build_codes_from_states(projected)
    actual_codes = base._codes(X[:32])
    assert np.array_equal(actual_codes, expected_codes)


def test_shared_direct_lookup_execution_is_bitwise_legacy_equivalent():
    X, y = load_digits(return_X_y=True)
    model = CERMClassifier(
        multiclass_strategy="shared",
        prediction_backend="semantic",
        max_features=20,
        max_interaction_features=12,
        max_interactions=12,
        search_profile="practical",
        reg_lambda=1.0,
        random_state=19,
        n_jobs=1,
    ).fit(X[:1200], y[:1200])
    shared = model.model_
    Xv = X[1200:1400]
    all_states, codes = shared._states_and_codes(Xv)
    legacy = np.tile(shared.intercept_, (len(codes), 1))
    for column, table in enumerate(shared.lookup_):
        state = codes[:, column]
        valid = (state >= 0) & (state < len(table))
        legacy[valid] += table[state[valid]]
    for descriptor, table in zip(shared.extra_descriptors_, shared.extra_lookup_):
        state = (
            all_states[int(descriptor.level)][:, int(descriptor.raw0)]
            == int(descriptor.state)
        ).astype(np.int32)
        legacy += np.asarray(table, dtype=np.float64)[state]
    assert np.array_equal(shared.decision_function(Xv), legacy)


def test_exact_ridge_alpha_path_matches_independent_public_ridge():
    import numpy as np
    from scipy import sparse
    from sklearn.linear_model import Ridge

    from cerm._compat import fit_ridge_lsqr_alpha_path_exact

    rng = np.random.default_rng(20260806)
    design = sparse.random(
        180,
        37,
        density=0.12,
        format="csr",
        random_state=20260806,
        dtype=np.float64,
    )
    target = rng.normal(size=180)
    weights = 1.0 + rng.integers(0, 4, size=180)
    alphas = np.asarray([0.1, 1.0, 10.0], dtype=np.float64)

    for sample_weight in (None, weights):
        coefficients, intercepts, iterations = fit_ridge_lsqr_alpha_path_exact(
            design,
            target,
            alphas,
            sample_weight=sample_weight,
        )
        for index, alpha in enumerate(alphas):
            reference = Ridge(alpha=float(alpha), solver="lsqr").fit(
                design, target, sample_weight=sample_weight
            )
            assert np.array_equal(coefficients[index], reference.coef_)
            # Sparse mean reductions can differ by one final rounding step
            # between BLAS implementations even when coefficients are bitwise
            # identical. Keep the boundary at machine precision.
            np.testing.assert_allclose(
                np.asarray([intercepts[index]]),
                np.asarray([reference.intercept_]),
                atol=4 * np.finfo(np.float64).eps,
                rtol=4 * np.finfo(np.float64).eps,
            )
            assert int(iterations[index]) == int(reference.n_iter_[0])
            np.testing.assert_allclose(
                design @ coefficients[index] + intercepts[index],
                reference.predict(design),
                atol=4 * np.finfo(np.float64).eps,
                rtol=4 * np.finfo(np.float64).eps,
            )
