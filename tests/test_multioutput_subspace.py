import numpy as np
import pandas as pd
import pytest

from cerm import CERMMultiOutputRegressor
from cerm._internal.cerm_multioutput_subspace import output_subspace


def _nonlinear_lowrank(seed=47, n=700, p=12, k=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    latent = np.column_stack(
        [
            (np.abs(x[:, 0]) > 0.8).astype(float),
            ((x[:, 1] * x[:, 2]) > 0.0).astype(float),
            ((np.abs(x[:, 3]) < 0.45) | (np.abs(x[:, 3]) > 1.35)).astype(float),
        ]
    )
    latent -= latent.mean(axis=0, keepdims=True)
    loadings = rng.normal(size=(3, k))
    y = latent @ loadings + 0.2 * rng.normal(size=(n, k))
    return x, y


def test_output_subspace_detects_low_rank_spikes():
    _, y = _nonlinear_lowrank()
    subspace = output_subspace(y, rank_cap=6)
    assert 1 <= subspace.rank <= 3
    assert subspace.eigenvalues[0] > subspace.mp_edge
    assert 0.0 < subspace.explained_energy <= 1.0


def test_shared_subspace_fits_one_vector_model():
    x, y = _nonlinear_lowrank()
    model = CERMMultiOutputRegressor(
        representation_strategy="shared_subspace", n_jobs=1
    ).fit(x, y)
    prediction = model.predict(x[:17])
    assert prediction.shape == (17, y.shape[1])
    assert np.isfinite(prediction).all()
    assert model.estimators_ is None
    assert model.fit_diagnostics_["strategy"] == "shared_subspace_finite_state"
    assert model.fit_diagnostics_["subspace_rank"] >= 1
    assert model.fit_diagnostics_["state_features"] >= 1
    assert model.model_bytes_estimate_ > 0


def test_flat_spectrum_shared_subspace_falls_back_exactly_to_independent():
    rng = np.random.default_rng(20260819)
    n, p, k = 180, 8, 4
    x = rng.normal(size=(n, p))
    q, _ = np.linalg.qr(rng.normal(size=(n, k)))
    y = q * np.sqrt(n)

    independent = CERMMultiOutputRegressor(
        representation_strategy="independent", n_jobs=1
    ).fit(x, y)
    shared = CERMMultiOutputRegressor(
        representation_strategy="shared_subspace", n_jobs=1
    ).fit(x, y)

    np.testing.assert_array_equal(shared.predict(x), independent.predict(x))
    assert shared.fit_diagnostics_["strategy"] == "shared_subspace_fallback_independent"
    assert shared.fit_diagnostics_["subspace_rank"] == 0


def test_shared_subspace_rejects_sample_weight_explicitly():
    x, y = _nonlinear_lowrank(n=200)
    with pytest.raises(ValueError, match="sample_weight is not yet supported"):
        CERMMultiOutputRegressor(
            representation_strategy="shared_subspace", n_jobs=1
        ).fit(x, y, sample_weight=np.ones(len(y)))


def test_unknown_multioutput_strategy_is_rejected():
    x, y = _nonlinear_lowrank(n=200)
    with pytest.raises(ValueError, match="shared_subspace"):
        CERMMultiOutputRegressor(representation_strategy="unknown").fit(x, y)


def test_shared_subspace_respects_public_head_alpha():
    x, y = _nonlinear_lowrank(n=300)
    model = CERMMultiOutputRegressor(
        representation_strategy="shared_subspace", head_alpha=0.25, n_jobs=1
    ).fit(x, y)
    assert model.fit_diagnostics_["strategy"] == "shared_subspace_finite_state"
    assert model.subspace_model_.head_.alpha == pytest.approx(0.25)


def test_multi_value_histogram_cache_matches_python_reference():
    from cerm.training_cache import FiniteStateValueHistogramCache

    rng = np.random.default_rng(20260819)
    states = rng.integers(0, 8, size=(180, 7), dtype=np.uint8)
    values = rng.normal(size=(180, 3))
    pairs = [(0, 1), (0, 4), (2, 6)]
    reference = FiniteStateValueHistogramCache(
        states, values, pairs=pairs, backend="python", n_jobs=1
    )
    candidate = FiniteStateValueHistogramCache(
        states, values, pairs=pairs, backend="auto", n_jobs=1
    )
    for expected, actual in zip(reference.main, candidate.main):
        np.testing.assert_array_equal(actual.mass, expected.mass)
        np.testing.assert_array_equal(actual.sums, expected.sums)
    assert tuple(candidate.pairs) == tuple(reference.pairs)
    for key in pairs:
        np.testing.assert_array_equal(
            candidate.pairs[key].mass, reference.pairs[key].mass
        )
        np.testing.assert_array_equal(
            candidate.pairs[key].sums, reference.pairs[key].sums
        )


def test_histogram_between_group_scores_match_scalar_reference():
    from cerm._internal.cerm_multioutput_subspace import MultiOutputSubspaceFiniteState
    from cerm.regression import _between_group_score
    from cerm.training_cache import FiniteStateValueHistogramCache

    rng = np.random.default_rng(314159)
    states = rng.integers(0, 6, size=(211, 5), dtype=np.uint8)
    values = rng.normal(size=(211, 4))
    cache = FiniteStateValueHistogramCache(
        states, values, pairs=(), backend="python"
    )
    overall = values.mean(axis=0)
    total = np.sum((values - overall) ** 2, axis=0)
    for feature, stats in enumerate(cache.main):
        got = MultiOutputSubspaceFiniteState._between_group_histogram_scores(
            stats, overall, total
        )
        expected = np.asarray(
            [
                _between_group_score(states[:, feature], values[:, r])
                for r in range(values.shape[1])
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-15)


def test_shared_subspace_fast_preparation_matches_legacy_numeric_matrix():
    from cerm.regression import CERMRegressor as HistoricalCERMRegressor

    x, y = _nonlinear_lowrank(n=260)
    target = y[:, 0]
    reference = HistoricalCERMRegressor(n_jobs=1)
    reference._representation_only_fit = True
    reference.fit(x, target)

    fast = HistoricalCERMRegressor(n_jobs=1)
    matrix = CERMMultiOutputRegressor._prepare_shared_subspace_matrix(
        fast, x, target
    )
    np.testing.assert_array_equal(matrix, reference._fit_matrix_cache_)
    np.testing.assert_array_equal(
        fast.adapted_feature_indices_, reference.adapted_feature_indices_
    )
    np.testing.assert_array_equal(
        fast.adapted_feature_names_, reference.adapted_feature_names_
    )
    assert fast.input_columns_ == reference.input_columns_
    assert fast.n_features_in_ == reference.n_features_in_


def test_shared_subspace_fast_preparation_matches_target_aware_typed_matrix():
    from cerm.regression import CERMRegressor as HistoricalCERMRegressor

    rng = np.random.default_rng(271828)
    n = 260
    categories = np.asarray([f"c{i}" for i in range(28)], dtype=object)
    values = rng.choice(categories, size=n)
    effect = {category: rng.normal() for category in categories}
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "category": values,
        }
    )
    target = np.asarray([effect[value] for value in values]) + 0.15 * rng.normal(
        size=n
    )

    reference = HistoricalCERMRegressor(n_jobs=1)
    reference._representation_only_fit = True
    reference.fit(frame, target)

    fast = HistoricalCERMRegressor(n_jobs=1)
    matrix = CERMMultiOutputRegressor._prepare_shared_subspace_matrix(
        fast, frame, target
    )
    np.testing.assert_array_equal(matrix, reference._fit_matrix_cache_)
    np.testing.assert_array_equal(
        fast.adapted_feature_indices_, reference.adapted_feature_indices_
    )
    np.testing.assert_array_equal(
        fast.adapted_feature_names_, reference.adapted_feature_names_
    )
    assert fast.input_columns_ == reference.input_columns_
    assert type(fast.adapter_) is type(reference.adapter_)


def test_shared_subspace_multi_target_lsqr_matches_sklearn_dense_and_sparse():
    from scipy import sparse
    from sklearn.linear_model import Ridge
    from cerm._internal.cerm_multioutput_subspace import _fit_multi_target_lsqr_exact

    rng = np.random.default_rng(161803)
    target = rng.normal(size=(180, 4))
    designs = [
        rng.normal(size=(180, 37)),
        sparse.random(
            180,
            73,
            density=0.12,
            random_state=rng,
            format="csr",
            dtype=np.float64,
        ),
    ]
    for design in designs:
        reference = Ridge(
            alpha=0.25,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-6,
            max_iter=1000,
        ).fit(design, target)
        coefficients, intercepts, iterations = _fit_multi_target_lsqr_exact(
            design,
            target,
            alpha=0.25,
            tol=1e-6,
            max_iter=1000,
        )
        expected_coefficients = np.asarray(reference.coef_, dtype=np.float64)
        if expected_coefficients.ndim == 1:
            expected_coefficients = expected_coefficients[None, :]
        np.testing.assert_array_equal(coefficients, expected_coefficients)
        np.testing.assert_array_equal(
            intercepts,
            np.asarray(reference.intercept_, dtype=np.float64).reshape(-1),
        )
        np.testing.assert_array_equal(
            iterations,
            np.asarray(reference.n_iter_, dtype=np.int32).reshape(-1),
        )



def test_shared_subspace_joblib_roundtrip_preserves_predictions(tmp_path):
    from cerm.multioutput_regression import CERMMultiOutputRegressor

    rng = np.random.default_rng(20260929)
    X = rng.normal(size=(240, 8))
    latent = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.4 * X[:, 1],
            X[:, 2] * X[:, 3] + 0.3 * X[:, 4],
        ]
    )
    mixing = np.asarray(
        [[1.0, 0.2, -0.6], [0.3, 1.1, 0.8]],
        dtype=np.float64,
    )
    y = latent @ mixing + 0.05 * rng.normal(size=(len(X), 3))

    model = CERMMultiOutputRegressor(
        representation_strategy="shared_subspace",
        n_jobs=2,
    ).fit(X, y)
    if model.fit_diagnostics_["strategy"] == "shared_subspace_fallback_independent":
        pytest.skip("constructed target did not retain an output subspace")

    expected = model.predict(X[:40])
    path_out = model.save(tmp_path / "shared_subspace.joblib")
    restored = CERMMultiOutputRegressor.load(path_out)
    np.testing.assert_array_equal(restored.predict(X[:40]), expected)



def test_multioutput_default_strategy_remains_independent():
    from cerm.multioutput_regression import CERMMultiOutputRegressor

    assert CERMMultiOutputRegressor().representation_strategy == "independent"
