from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from cerm import (
    CERMClassifier,
    CERMGeneralizedRegressor,
    CERMMultiLabelClassifier,
    CERMParameterError,
    CERMRegressor,
    CERMRidgeRegressor,
    get_convenience_params,
    get_resource_params,
    get_search_params,
    get_tunable_params,
)


def test_parameter_layers_separate_hpo_search_and_resources():
    model = CERMClassifier(resource_policy="ignore")
    tunable = get_tunable_params(model)
    assert set(tunable) >= {
        "max_bins",
        "max_features",
        "max_interaction_features",
        "max_interactions",
        "interaction_order",
        "reg_lambda",
        "subsample",
        "colsample",
    }
    assert "search_effort" not in tunable
    assert "state_detail" not in tunable
    assert "n_jobs" not in tunable
    assert get_search_params(model)["preset"] == "accurate"
    assert get_resource_params(model)["n_jobs"] == 1
    assert get_convenience_params(model)["state_detail"] == "fine"
    assert get_tunable_params(clone(model)) == tunable


def test_convenience_aliases_resolve_to_same_direct_hpo_values_and_predictions():
    X, y = make_classification(
        n_samples=260, n_features=12, n_informative=7, random_state=71
    )
    semantic = CERMClassifier(
        search_effort="balanced",
        state_detail="medium",
        feature_budget=12,
        interaction_search_features=8,
        interaction_budget=4,
        l2_regularization=2.0,
        random_state=73,
        resource_policy="ignore",
    ).fit(X, y)
    direct = CERMClassifier(
        preset="balanced",
        max_bins=8,
        max_features=12,
        max_interaction_features=8,
        max_interactions=4,
        reg_lambda=2.0,
        random_state=73,
        resource_policy="ignore",
    ).fit(X, y)
    np.testing.assert_array_equal(semantic.predict_proba(X), direct.predict_proba(X))
    assert get_tunable_params(semantic)["max_bins"] == 8
    assert get_tunable_params(semantic)["max_features"] == 12
    assert get_search_params(semantic)["preset"] == "balanced"


def test_layered_view_refreshes_after_set_params_alias():
    model = CERMClassifier(resource_policy="ignore")
    model.set_params(state_detail="medium", feature_budget=20)
    tunable = get_tunable_params(model)
    assert tunable["max_bins"] == 8
    assert tunable["max_features"] == 20


def test_direct_numeric_hpo_surface_works_with_grid_search():
    X, y = make_classification(
        n_samples=220, n_features=10, n_informative=6, random_state=79
    )
    search = GridSearchCV(
        CERMClassifier(
            interaction_order=1,
            max_features=8,
            random_state=83,
            resource_policy="ignore",
        ),
        {"max_bins": [4, 8], "reg_lambda": [0.5, 1.0]},
        cv=2,
        scoring="neg_log_loss",
        n_jobs=1,
    ).fit(X, y)
    assert search.best_params_["max_bins"] in {4, 8}
    assert search.best_params_["reg_lambda"] in {0.5, 1.0}


def test_default_regression_exposes_fused_layered_surface():
    assert get_tunable_params(CERMRegressor()) == {
        "n_bins": 10,
        "max_bins": 16,
        "max_features": 24,
        "max_interaction_features": 12,
        "max_pairs": 4,
    }
    assert get_search_params(CERMRegressor()) == {}
    assert get_resource_params(CERMRegressor()) == {}


def test_historical_ridge_and_generalized_keep_historical_layers():
    ridge = CERMRidgeRegressor()
    assert "max_bins" in get_tunable_params(ridge)
    assert "max_interactions" in get_tunable_params(ridge)
    assert get_resource_params(ridge)["n_jobs"] == 1
    generalized = CERMGeneralizedRegressor(loss="huber")
    assert "max_bins" in get_tunable_params(generalized)
    assert "head_alpha" in get_tunable_params(generalized)


def test_composite_wrapper_has_resource_view_but_no_fake_model_hpo_view():
    wrapper = CERMMultiLabelClassifier(n_jobs=2)
    assert get_resource_params(wrapper)["n_jobs"] == 2
    with pytest.raises(CERMParameterError, match="base estimator"):
        get_tunable_params(wrapper)
