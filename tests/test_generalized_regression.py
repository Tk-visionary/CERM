from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split

from cerm import CERMGeneralizedRegressor, CERMRidgeRegressor


def _params(seed=17):
    return dict(
        preset="balanced",
        max_features=8,
        max_interaction_features=8,
        max_interactions=4,
        max_bins=8,
        random_state=seed,
        max_iter=300,
    )


def test_huber_is_robust_to_large_training_outliers():
    X, y = make_friedman1(n_samples=520, n_features=8, noise=0.5, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=9
    )
    y_corrupt = y_train.copy()
    y_corrupt[:35] += 80.0
    squared = CERMRidgeRegressor(
        preset="balanced",
        max_features=8,
        max_interaction_features=8,
        max_interactions=4,
        max_bins=8,
        random_state=17,
    ).fit(X_train, y_corrupt)
    robust = CERMGeneralizedRegressor(loss="huber", **_params()).fit(
        X_train, y_corrupt
    )
    assert mean_absolute_error(y_test, robust.predict(X_test)) < mean_absolute_error(
        y_test, squared.predict(X_test)
    )
    assert robust.fit_diagnostics_["loss"] == "huber"


def test_quantile_regression_has_reasonable_empirical_coverage():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(650, 6))
    scale = 0.4 + 1.2 * np.abs(X[:, 1])
    y = 1.5 * X[:, 0] - 0.8 * X[:, 2] + rng.normal(scale=scale)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=5
    )
    model = CERMGeneralizedRegressor(
        loss="quantile", quantile=0.9, head_alpha=5e-4, **_params(23)
    ).fit(X_train, y_train)
    coverage = float(np.mean(y_test <= model.predict(X_test)))
    assert 0.78 <= coverage <= 0.98
    assert model.fit_diagnostics_["quantile"] == 0.9


def test_poisson_regression_outputs_nonnegative_means_and_beats_constant():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(700, 6))
    log_mu = 0.2 + 0.7 * X[:, 0] - 0.45 * X[:, 1] + 0.3 * (X[:, 2] > 0)
    y = rng.poisson(np.exp(log_mu)).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=11
    )
    model = CERMGeneralizedRegressor(
        loss="poisson", head_alpha=1e-4, **_params(29)
    ).fit(X_train, y_train)
    prediction = model.predict(X_test)
    baseline = np.full(len(y_test), max(float(y_train.mean()), 1e-12))
    assert np.all(prediction > 0)
    assert mean_poisson_deviance(y_test, prediction) < mean_poisson_deviance(
        y_test, baseline
    )
    assert model.fit_diagnostics_["representation_target"] == "log1p_y"


def test_sample_weight_reaches_representation_and_generalized_head(tmp_path):
    rng = np.random.default_rng(43)
    n = 180
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "group": np.asarray([f"g{i % 12}" for i in range(n)], dtype=object),
        }
    )
    y = 1.5 * frame["x"].to_numpy() + np.asarray(
        [int(value[1:]) % 4 for value in frame["group"]], dtype=float
    )
    weights = np.ones(n)
    weights[:30] = 5.0
    model = CERMGeneralizedRegressor(
        loss="huber",
        categorical_features=["group"],
        category_policy="ordered",
        **_params(41),
    ).fit(frame, y, sample_weight=weights)
    assert model.fit_diagnostics_["sample_weighted"] is True
    expected = model.predict(frame.iloc[:20])
    restored = CERMGeneralizedRegressor.load(model.save(tmp_path / "generalized.joblib"))
    np.testing.assert_allclose(restored.predict(frame.iloc[:20]), expected)


def test_weighted_ridge_regression_changes_the_fitted_target_emphasis():
    X = np.arange(80.0).reshape(-1, 1)
    y = np.zeros(80)
    y[-10:] = 20.0
    unweighted = CERMRidgeRegressor(
        preset="balanced", max_features=1, max_interactions=0, max_bins=4, random_state=7
    ).fit(X, y)
    weights = np.ones(80)
    weights[-10:] = 20.0
    weighted = CERMRidgeRegressor(
        preset="balanced", max_features=1, max_interactions=0, max_bins=4, random_state=7
    ).fit(X, y, sample_weight=weights)
    assert weighted.predict([[75.0]])[0] > unweighted.predict([[75.0]])[0]
    assert weighted.fit_diagnostics_["sample_weighted"] is True


def test_exact_rewrite_diagnostics_for_numeric_and_typed_paths():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(180, 4))
    y = rng.poisson(np.exp(0.1 + 0.4 * X[:, 0])).astype(float)
    numeric = CERMGeneralizedRegressor(
        loss="poisson",
        preset="balanced",
        max_features=4,
        max_interaction_features=4,
        max_interactions=0,
        max_bins=4,
        random_state=3,
    ).fit(X, y)
    passes = numeric.fit_diagnostics_["exact_rewrite_passes"]
    assert "representation_only_final_fit" in passes
    assert "training_design_reuse" in passes
    assert "small_head_threadpool_limit" in passes
    assert numeric.fit_diagnostics_["head_thread_limit"] == 1

    frame = pd.DataFrame(
        {
            "x": X[:, 0],
            "group": np.asarray([f"g{i % 9}" for i in range(len(X))], dtype=object),
        }
    )
    typed = CERMGeneralizedRegressor(
        loss="huber",
        categorical_features=["group"],
        category_policy="ordered",
        preset="balanced",
        max_features=4,
        max_interaction_features=4,
        max_interactions=0,
        max_bins=4,
        random_state=5,
    ).fit(frame, X[:, 0])
    typed_passes = typed.fit_diagnostics_["exact_rewrite_passes"]
    assert "typed_final_transform_preserved" in typed_passes
    assert "training_design_reuse" not in typed_passes
