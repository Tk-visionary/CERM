from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from cerm import CERMClassifier, CERMGeneralizedRegressor, CERMRidgeRegressor


def test_classifier_semantic_aliases_are_exact_legacy_equivalents():
    X, y = make_classification(
        n_samples=280,
        n_features=14,
        n_informative=8,
        n_redundant=0,
        random_state=601,
    )
    semantic = CERMClassifier(
        search_effort="balanced",
        state_detail="medium",
        feature_budget=12,
        interaction_search_features=8,
        interaction_budget=4,
        selection_fraction=0.8,
        feature_fraction=0.9,
        l2_regularization=2.0,
        memory_limit_mb=2048.0,
        random_state=607,
        resource_policy="ignore",
    )
    legacy = CERMClassifier(
        preset="balanced",
        max_bins=8,
        max_features=12,
        max_interaction_features=8,
        max_interactions=4,
        subsample=0.8,
        colsample=0.9,
        reg_lambda=2.0,
        max_memory_mb=2048.0,
        random_state=607,
        resource_policy="ignore",
    )
    semantic.fit(X, y)
    legacy.fit(X, y)
    np.testing.assert_array_equal(semantic.predict_proba(X), legacy.predict_proba(X))
    assert semantic.get_user_params()["state_detail"] == "medium"
    assert semantic.get_user_params()["search_effort"] == "balanced"
    assert semantic.explain_params()["interaction_budget"]["value"] == 4
    assert "state_detail='medium'" in semantic.parameter_summary()


def test_ridge_regression_semantic_aliases_are_exact_legacy_equivalents():
    X, y = make_regression(
        n_samples=260,
        n_features=12,
        n_informative=8,
        noise=0.4,
        random_state=613,
    )
    semantic = CERMRidgeRegressor(
        search_effort="balanced",
        state_detail="coarse",
        feature_budget=10,
        interaction_search_features=7,
        interaction_budget=3,
        selection_fraction=0.85,
        feature_fraction=0.9,
        l2_regularization=1.5,
        random_state=617,
    ).fit(X, y)
    legacy = CERMRidgeRegressor(
        preset="balanced",
        max_bins=4,
        max_features=10,
        max_interaction_features=7,
        max_interactions=3,
        subsample=0.85,
        colsample=0.9,
        reg_lambda=1.5,
        random_state=617,
    ).fit(X, y)
    np.testing.assert_array_equal(semantic.predict(X), legacy.predict(X))


def test_generalized_regression_semantic_aliases_are_exact():
    X, y = make_regression(
        n_samples=240,
        n_features=10,
        n_informative=7,
        noise=1.0,
        random_state=619,
    )
    semantic = CERMGeneralizedRegressor(
        loss="huber",
        search_effort="balanced",
        state_detail="medium",
        feature_budget=9,
        interaction_search_features=6,
        interaction_budget=2,
        l2_regularization=2.0,
        random_state=631,
    ).fit(X, y)
    legacy = CERMGeneralizedRegressor(
        loss="huber",
        preset="balanced",
        max_bins=8,
        max_features=9,
        max_interaction_features=6,
        max_interactions=2,
        reg_lambda=2.0,
        random_state=631,
    ).fit(X, y)
    np.testing.assert_array_equal(semantic.predict(X), legacy.predict(X))
    assert semantic.get_user_params()["loss"] == "huber"


def test_semantic_parameters_are_cloneable_and_config_roundtrip():
    estimator = CERMClassifier(
        search_effort="balanced",
        state_detail="medium",
        feature_budget=20,
        interaction_search_features=9,
        interaction_budget=6,
        selection_fraction=0.75,
        feature_fraction=0.8,
        l2_regularization=3.0,
        memory_limit_mb=1024.0,
    )
    cloned = clone(estimator)
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)
    restored = CERMClassifier.from_config(estimator.to_config())
    assert restored.get_params(deep=False) == estimator.get_params(deep=False)


@pytest.mark.parametrize("multiclass_strategy", ["ovr", "shared"])
def test_config_roundtrip_preserves_multiclass_strategy(multiclass_strategy):
    estimator = CERMClassifier(multiclass_strategy=multiclass_strategy)
    config = estimator.to_config()
    assert config.multiclass_strategy == multiclass_strategy
    restored = CERMClassifier.from_config(config)
    assert restored.multiclass_strategy == multiclass_strategy
    assert restored.to_config().multiclass_strategy == multiclass_strategy


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"state_detail": "medium", "max_bins": 4}, "conflicts"),
        ({"feature_budget": 12, "max_features": 10}, "conflicts"),
        ({"search_effort": "balanced", "search_profile": "full_exact"}, "cannot"),
        ({"search_effort": "fast"}, "search_effort"),
    ],
)
def test_conflicting_or_unknown_semantic_parameters_fail_early(kwargs, message):
    estimator = CERMClassifier(**kwargs)
    with pytest.raises(ValueError, match=message):
        estimator.resolve_params()
