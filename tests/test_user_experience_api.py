from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from cerm import (
    CERMAliasConflictError,
    CERMClassifier,
    CERMDataSchemaError,
    CERMGeneralizedRegressor,
    CERMParameterError,
    CERMRegressor,
    CERMRidgeRegressor,
    CERMResourceLimitError,
    feature_table,
    inspect_model,
    interaction_table,
    model_summary,
)
from cerm.params import resolve_semantic_alias


def test_schema_error_is_public_and_actionable():
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=11,
    )
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    model = CERMClassifier(
        interaction_order=1,
        resource_policy="ignore",
        random_state=11,
    ).fit(frame, y)

    with pytest.raises(CERMDataSchemaError) as captured:
        model.predict(frame.drop(columns=["c"]).assign(unexpected=0.0))

    message = str(captured.value)
    assert "DataFrame schema mismatch" in message
    assert "missing=['c']" in message
    assert "extra=['unexpected']" in message
    assert "same columns used during fit" in message


def test_parameter_and_alias_errors_have_public_taxonomy_and_guidance():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=3,
    )
    with pytest.raises(CERMParameterError, match="max_bins must be one of"):
        CERMClassifier(max_bins=7).fit(X, y)

    with pytest.raises(CERMAliasConflictError, match="Prefer the user-facing parameter"):
        resolve_semantic_alias(
            semantic_name="state_detail",
            semantic_value="coarse",
            legacy_name="max_bins",
            legacy_value=16,
            legacy_default=8,
            mapping={"coarse": 4, "fine": 16},
        )


def test_resource_error_points_to_prefit_plan():
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    model = CERMClassifier(
        max_features=8,
        pair_feature_limit=8,
        max_pair_evaluations=1,
        resource_policy="raise",
        random_state=13,
    )
    with pytest.raises(CERMResourceLimitError) as captured:
        model.fit(X, y)
    assert "CERM fit resource budget exceeded" in str(captured.value)
    assert "estimate_fit_resources(X)" in str(captured.value)


def test_notebook_inspection_uses_adapted_feature_names():
    X, y = make_classification(
        n_samples=140,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=19,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    model = CERMClassifier(
        interaction_order=1,
        resource_policy="ignore",
        random_state=19,
    ).fit(frame, y)

    features = feature_table(model)
    interactions = interaction_table(model)
    inspection = inspect_model(model)

    assert len(features) == model_summary(model).selected_features
    assert set(features["feature_name"]).issubset(set(frame.columns))
    assert list(features.columns) == [
        "rank",
        "adapted_index",
        "feature_name",
        "used_in_pair",
        "used_in_block",
    ]
    assert interactions.empty
    assert inspection.features.equals(features)
    assert "Selected features" in inspection._repr_html_()


def test_common_inspection_supports_historical_ridge_and_generalized_regression():
    X, y = make_regression(n_samples=90, n_features=5, noise=0.2, random_state=23)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])

    regressor = CERMRidgeRegressor(
        interaction_order=1,
        random_state=23,
    ).fit(frame, y)
    reg_summary = model_summary(regressor)
    assert reg_summary.task == "regression"
    assert len(feature_table(regressor)) == reg_summary.selected_features

    generalized = CERMGeneralizedRegressor(
        loss="huber",
        representation_mode="linear",
        interaction_order=1,
        max_iter=50,
        random_state=23,
    ).fit(frame, y)
    gen_summary = model_summary(generalized)
    assert gen_summary.task == "generalized_regression"
    assert len(feature_table(generalized)) == gen_summary.selected_features
    assert np.isfinite(generalized.predict(frame.iloc[:3])).all()


def test_default_fused_regression_summary_reports_engine():
    X, y = make_regression(n_samples=96, n_features=5, noise=0.3, random_state=29)
    model = CERMRegressor(
        n_bins=6,
        max_features=5,
        max_bins=8,
        max_interaction_features=4,
        max_pairs=1,
        random_state=29,
    ).fit(X, y)
    summary = model_summary(model)
    assert summary.estimator == "CERMRegressor"
    assert summary.task == "regression"
    assert model.fit_diagnostics_["engine"] == "fused-residual-v2"
