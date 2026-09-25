from __future__ import annotations

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from cerm import CERMGeneralizedRegressor, CERMMultiOutputRegressor, CERMRidgeRegressor


def _generalized_params(seed=13):
    return dict(
        preset="balanced",
        max_features=6,
        max_interaction_features=6,
        max_interactions=2,
        max_bins=4,
        random_state=seed,
        max_iter=250,
    )


def _base_regressor(seed=17):
    return CERMRidgeRegressor(
        preset="balanced",
        max_features=6,
        max_interaction_features=6,
        max_interactions=2,
        max_bins=4,
        random_state=seed,
        n_jobs=1,
    )


def test_poisson_exposure_is_exact_rate_weight_transformation():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(260, 5))
    exposure = rng.uniform(0.5, 3.0, size=len(X))
    rate = np.exp(0.2 + 0.45 * X[:, 0] - 0.2 * X[:, 1])
    y = rng.poisson(exposure * rate).astype(float)

    with_exposure = CERMGeneralizedRegressor(
        loss="poisson", **_generalized_params(19)
    ).fit(X, y, exposure=exposure)
    transformed = CERMGeneralizedRegressor(
        loss="poisson", **_generalized_params(19)
    ).fit(X, y / exposure, sample_weight=exposure)

    np.testing.assert_array_equal(
        with_exposure.predict(X, exposure=exposure),
        transformed.predict(X) * exposure,
    )
    assert with_exposure.fit_diagnostics_["exposure_used"] is True


def test_huber_offset_matches_offset_adjusted_target():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(220, 4))
    offset = 0.4 * X[:, 0]
    residual = 1.2 * X[:, 1] + rng.normal(scale=0.3, size=len(X))
    y = offset + residual
    with_offset = CERMGeneralizedRegressor(
        loss="huber", **_generalized_params(23)
    ).fit(X, y, offset=offset)
    adjusted = CERMGeneralizedRegressor(
        loss="huber", **_generalized_params(23)
    ).fit(X, y - offset)
    np.testing.assert_array_equal(
        with_offset.predict(X, offset=offset), adjusted.predict(X) + offset
    )


def test_multi_quantile_predictions_are_ordered_and_share_one_representation():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(360, 6))
    scale = 0.3 + np.abs(X[:, 1])
    y = X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=scale)
    model = CERMGeneralizedRegressor(
        loss="multi_quantile",
        quantiles=(0.1, 0.5, 0.9),
        head_alpha=5e-4,
        **_generalized_params(29),
    ).fit(X, y)
    prediction = model.predict(X[:40])
    assert prediction.shape == (40, 3)
    assert np.all(np.diff(prediction, axis=1) >= 0)
    assert model.fit_diagnostics_["n_heads"] == 3
    assert "shared_multi_quantile_design" in model.fit_diagnostics_[
        "exact_rewrite_passes"
    ]


def test_shared_multioutput_regression_predicts_all_targets_and_beats_mean():
    X, base = make_friedman1(n_samples=420, n_features=6, noise=0.3, random_state=31)
    rng = np.random.default_rng(31)
    y = np.column_stack(
        [
            base,
            0.6 * base + 2.0 * X[:, 0] + rng.normal(scale=0.4, size=len(X)),
            -0.3 * base + X[:, 1] + rng.normal(scale=0.4, size=len(X)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=37
    )
    model = CERMMultiOutputRegressor(
        estimator=_base_regressor(41),
        representation_strategy="shared",
        head_alpha=1.0,
    ).fit(X_train, y_train)
    prediction = model.predict(X_test)
    baseline = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    assert prediction.shape == y_test.shape
    assert mean_squared_error(y_test, prediction) < mean_squared_error(y_test, baseline)
    assert model.fit_diagnostics_["strategy"] == "shared_pc1_finite_state"


def test_independent_multioutput_matches_individual_regressors():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(180, 5))
    y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - 0.5 * X[:, 3]])
    base = _base_regressor(53)
    multi = CERMMultiOutputRegressor(
        estimator=base, representation_strategy="independent"
    ).fit(X, y)
    separate = np.column_stack(
        [_base_regressor(53).fit(X, y[:, index]).predict(X) for index in range(2)]
    )
    np.testing.assert_array_equal(multi.predict(X), separate)
