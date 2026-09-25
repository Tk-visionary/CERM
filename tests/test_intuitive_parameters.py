from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, make_classification

from cerm import CERMClassifier


def _small_data():
    X, y = make_classification(
        n_samples=260,
        n_features=14,
        n_informative=8,
        n_redundant=0,
        random_state=509,
    )
    return X, y


def test_default_preset_preserves_full_exact_behavior():
    X, y = load_breast_cancer(return_X_y=True)
    kwargs = dict(
        max_features=18,
        pair_feature_limit=14,
        random_state=511,
        resource_policy="ignore",
    )
    default = CERMClassifier(**kwargs).fit(X, y)
    legacy = CERMClassifier(search_profile="full_exact", **kwargs).fit(X, y)
    np.testing.assert_allclose(
        default.predict_proba(X), legacy.predict_proba(X), atol=0.0, rtol=0.0
    )
    assert default.resolved_params_["preset"] == "accurate"
    assert default.resolved_params_["search_profile"] == "full_exact"


def test_balanced_preset_matches_validated_practical_profile():
    X, y = _small_data()
    kwargs = dict(
        max_features=12,
        max_interaction_features=10,
        random_state=521,
        resource_policy="ignore",
    )
    balanced = CERMClassifier(preset="balanced", **kwargs).fit(X, y)
    legacy = CERMClassifier(
        preset="accurate",
        search_profile="practical",
        **kwargs,
    ).fit(X, y)
    np.testing.assert_allclose(
        balanced.predict_proba(X), legacy.predict_proba(X), atol=0.0, rtol=0.0
    )
    assert balanced.fit_diagnostics_.training_graph_candidate_count <= 9


def test_max_interaction_features_alias_controls_pair_budget():
    X, _ = _small_data()
    estimator = CERMClassifier(
        pair_feature_limit=24,
        max_interaction_features=7,
        resource_policy="ignore",
    )
    plan = estimator.estimate_fit_resources(X)
    assert plan.effective_pair_features == 7
    assert plan.pair_candidates_per_ranking == 21
    assert estimator.resolve_params().max_interaction_features == 7


def test_max_interactions_caps_final_block_dictionary():
    X, y = _small_data()
    model = CERMClassifier(
        max_features=12,
        max_interaction_features=10,
        max_interactions=4,
        random_state=523,
        resource_policy="ignore",
    ).fit(X, y)
    assert len(model.model_.block_tables_) <= 4
    assert model.fit_diagnostics_.effective_max_interactions == 4
    assert all(
        row[2].n_blocks <= 4 for row in model.model_.primary_scores_
    )


def test_interaction_order_one_disables_pair_and_block_terms():
    X, y = _small_data()
    model = CERMClassifier(
        max_features=12,
        interaction_order=1,
        random_state=527,
        resource_policy="ignore",
    ).fit(X, y)
    diagnostics = model.fit_diagnostics_
    assert diagnostics.interaction_order == 1
    assert diagnostics.pair_count == 0
    assert diagnostics.fine_pair_count == 0
    assert diagnostics.block_count == 0
    assert diagnostics.training_graph_candidate_count == 1


def test_fixed_reg_lambda_applies_to_backbone_and_hybrid():
    X, y = _small_data()
    model = CERMClassifier(
        max_features=12,
        max_interaction_features=10,
        reg_lambda=2.0,
        random_state=529,
        resource_policy="ignore",
    ).fit(X, y)
    expected_C = 0.5
    assert model.model_.selected_hybrid_config_.C == expected_C
    assert model.model_.base_.best_config_.C == expected_C
    assert {row[2].C for row in model.model_.primary_scores_} == {expected_C}
    assert model.fit_diagnostics_.reg_lambda == 2.0


def test_max_memory_alias_overrides_legacy_memory_limit():
    X, _ = _small_data()
    estimator = CERMClassifier(
        max_memory_mb=1.0,
        max_estimated_peak_memory_mb=4096.0,
        resource_policy="ignore",
    )
    assert estimator.resolve_params().max_memory_mb == 1.0
    assert estimator.estimate_fit_resources(X).estimated_peak_memory_bytes > 0


def test_new_parameters_are_cloneable_and_config_roundtrips():
    estimator = CERMClassifier(
        preset="balanced",
        max_interaction_features=11,
        max_interactions=8,
        interaction_order=2,
        reg_lambda=3.0,
        max_memory_mb=2048.0,
        representation_strategy="adaptive",
        class_specific_budget=8,
    )
    cloned = clone(estimator)
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)
    restored = CERMClassifier.from_config(estimator.to_config())
    assert restored.get_params(deep=False) == estimator.get_params(deep=False)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"preset": "fast"}, "preset"),
        ({"interaction_order": 3}, "interaction_order"),
        ({"max_interactions": -1}, "max_interactions"),
        ({"reg_lambda": 0.0}, "reg_lambda"),
    ],
)
def test_intuitive_parameter_validation(params, message):
    with pytest.raises(ValueError, match=message):
        CERMClassifier(**params).resolve_params()
