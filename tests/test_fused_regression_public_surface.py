from __future__ import annotations

import inspect
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_friedman1, make_regression
from sklearn.model_selection import GridSearchCV

from cerm import (
    CERMFusedRegressor,
    CERMRegressor,
    CERMRidgeRegressor,
    PortableFusedRegressionProgram,
    get_convenience_params,
    get_resource_params,
    get_search_params,
    get_tunable_params,
)
from cerm.regression import CERMRegressor as HistoricalCERMRegressor


@pytest.fixture(scope="module")
def numeric_fit():
    X, y = make_friedman1(
        n_samples=120,
        n_features=6,
        noise=0.8,
        random_state=19,
    )
    model = CERMRegressor(
        n_bins=6,
        max_features=6,
        max_bins=8,
        max_interaction_features=4,
        max_pairs=1,
        random_state=19,
    ).fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def typed_fit():
    rng = np.random.default_rng(29)
    n = 100
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "group": np.asarray(["a", "b", "c", "d"])[rng.integers(0, 4, size=n)],
            "with_missing": rng.normal(size=n),
        }
    )
    frame.loc[::9, "with_missing"] = np.nan
    group_effect = frame["group"].map(
        {"a": -1.0, "b": -0.25, "c": 0.5, "d": 1.25}
    ).to_numpy()
    y = 1.8 * frame["x"].to_numpy() + group_effect + rng.normal(scale=0.2, size=n)
    weights = np.linspace(0.5, 2.5, n)
    model = CERMRegressor(
        n_bins=6,
        max_features=6,
        max_bins=8,
        max_interaction_features=4,
        max_pairs=1,
        max_identity_categories=3,
        random_state=29,
    ).fit(frame, y, sample_weight=weights)
    return model, frame, y, weights


def test_cerm_regressor_is_fused_v2_default():
    signature = inspect.signature(CERMRegressor.__init__)
    assert "regression_strategy" not in signature.parameters
    assert signature.parameters["max_features"].default == 24
    assert signature.parameters["max_interaction_features"].default == 12
    assert signature.parameters["max_pairs"].default == 4
    assert signature.parameters["n_bins"].default == 10
    assert "max_interactions" not in signature.parameters
    assert "reg_lambda" not in signature.parameters

    model = CERMRegressor()
    assert isinstance(model, CERMFusedRegressor)
    assert get_tunable_params(model) == {
        "n_bins": 10,
        "max_bins": 16,
        "max_features": 24,
        "max_interaction_features": 12,
        "max_pairs": 4,
    }
    assert get_search_params(model) == {}
    assert get_resource_params(model) == {}


def test_cerm_fused_regressor_remains_behavior_compatible_name():
    X, y = make_friedman1(
        n_samples=96,
        n_features=6,
        noise=0.7,
        random_state=11,
    )
    params = dict(
        n_bins=6,
        max_features=6,
        max_bins=8,
        max_interaction_features=4,
        max_pairs=1,
        random_state=11,
    )
    default = CERMRegressor(**params).fit(X, y)
    explicit = CERMFusedRegressor(**params).fit(X, y)
    np.testing.assert_array_equal(default.predict(X), explicit.predict(X))
    assert default.selected_kind_ == explicit.selected_kind_


def test_historical_ridge_is_preserved_under_explicit_name():
    signature = inspect.signature(CERMRidgeRegressor.__init__)
    assert signature.parameters["max_features"].default == 64
    assert signature.parameters["max_interaction_features"].default == 24
    assert signature.parameters["max_interactions"].default is None

    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=3.0,
        random_state=7,
    )
    public_legacy = CERMRidgeRegressor(random_state=7).fit(X, y)
    historical = HistoricalCERMRegressor(random_state=7).fit(X, y)
    np.testing.assert_array_equal(public_legacy.predict(X), historical.predict(X))
    assert public_legacy.get_params(deep=True) == historical.get_params(deep=True)


def test_default_fused_is_cloneable():
    model = CERMRegressor()
    params = model.get_params(deep=True)
    cloned = clone(model)
    assert cloned.get_params(deep=True) == params
    cloned.set_params(max_pairs=2, max_features=16)
    assert cloned.max_pairs == 2
    assert cloned.max_features == 16


def test_default_hpo_surface_is_fused_specific():
    model = CERMRegressor()
    assert get_tunable_params(model) == {
        "n_bins": 10,
        "max_bins": 16,
        "max_features": 24,
        "max_interaction_features": 12,
        "max_pairs": 4,
    }
    assert get_search_params(model) == {}
    assert get_resource_params(model) == {}
    convenience = get_convenience_params(model)
    assert convenience["interaction_budget"] == 4
    assert convenience["survival_bins"] == 10
    assert "max_interactions" not in get_tunable_params(model)
    assert "reg_lambda" not in get_tunable_params(model)


def test_grid_search_cv_smoke_uses_default_fused_parameters_only():
    X, y = make_regression(
        n_samples=40,
        n_features=4,
        n_informative=3,
        noise=2.0,
        random_state=13,
    )
    search = GridSearchCV(
        CERMRegressor(
            n_bins=5,
            max_features=4,
            max_bins=4,
            max_interaction_features=3,
            max_pairs=0,
            random_state=13,
        ),
        {"max_features": [3, 4]},
        cv=2,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    ).fit(X, y)
    assert search.best_params_["max_features"] in {3, 4}
    assert isinstance(search.best_estimator_, CERMRegressor)


def test_numeric_prediction_diagnostics_and_feature_names(numeric_fit):
    model, X, _ = numeric_fit
    prediction = model.predict(X[:17])
    survival = model.predict_survival(X[:17])
    assert prediction.shape == (17,)
    assert survival.shape[0] == 17
    assert np.isfinite(prediction).all()
    assert np.isfinite(survival).all()
    assert model.fit_diagnostics_["engine"] == "fused-residual-v2"
    assert model.fit_diagnostics_["public_estimator"] == "CERMRegressor"
    assert model.fit_diagnostics_["public_capacity"]["max_pairs"] == 1
    assert len(model.get_feature_names_out()) == model.n_adapted_features_
    assert model.model_bytes_estimate_ > 0


def test_typed_dataframe_and_sample_weight_contract(typed_fit):
    model, frame, _, weights = typed_fit
    assert model.adapter_ is not None
    assert model.fit_diagnostics_["sample_weighted"] is True
    assert model.fit_diagnostics_["sample_weight_sum"] == pytest.approx(weights.sum())
    expected = model.predict(frame.iloc[:20])
    reordered = model.predict(frame.iloc[:20][list(reversed(frame.columns))])
    np.testing.assert_array_equal(expected, reordered)
    names = model.get_feature_names_out()
    assert any(str(name).startswith("cat") for name in names)
    assert any(str(name).startswith("missing:") for name in names)


def test_semantic_export_roundtrip_and_python_save_load(typed_fit, tmp_path):
    model, frame, _, _ = typed_fit
    expected = model.predict(frame.iloc[:21])

    manifest = model.export(tmp_path / "portable")
    portable = PortableFusedRegressionProgram.load(manifest)
    np.testing.assert_allclose(
        portable.predict(frame.iloc[:21]),
        expected,
        rtol=0.0,
        atol=1e-12,
    )

    restored = CERMRegressor.load(model.save(tmp_path / "default_regressor.joblib"))
    np.testing.assert_array_equal(restored.predict(frame.iloc[:21]), expected)
    assert restored.get_params(deep=True) == model.get_params(deep=True)


def test_native_parity_through_default_estimator(numeric_fit, tmp_path):
    model, X, _ = numeric_fit
    compiled = model.compile_native(tmp_path / "default_regressor")
    probe = X[:31]
    np.testing.assert_allclose(
        compiled.predict(probe),
        model.predict(probe),
        rtol=0.0,
        atol=1e-12,
    )
    assert compiled.artifact_bytes > 0
    assert compiled.compile_seconds >= 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_bins": 2}, "n_bins"),
        ({"max_features": 0}, "max_features"),
        ({"max_bins": 12}, "max_bins"),
        ({"max_interaction_features": 0}, "max_interaction_features"),
        ({"max_pairs": -1}, "max_pairs"),
    ],
)
def test_invalid_default_capacity_parameters(kwargs, match):
    X = np.arange(32.0).reshape(8, 4)
    y = np.linspace(-1.0, 1.0, len(X))
    with pytest.raises((TypeError, ValueError), match=match):
        CERMRegressor(**kwargs).fit(X, y)


def test_ridge_only_parameter_names_require_ridge_estimator():
    with pytest.raises(TypeError):
        CERMRegressor(max_interactions=4)
    with pytest.raises(TypeError):
        CERMRegressor(reg_lambda=1.0)
    ridge = CERMRidgeRegressor(max_interactions=4, reg_lambda=1.0)
    assert ridge.max_interactions == 4
    assert ridge.reg_lambda == 1.0


def test_documented_examples_smoke():
    root = Path(__file__).resolve().parents[1]
    for relative in ("examples/regression.py", "examples/fused_regression.py"):
        source = (root / relative).read_text(encoding="utf-8")
        compile(source, relative, "exec")
        namespace = runpy.run_path(str(root / relative), run_name="__main__")
        assert "model" in namespace
