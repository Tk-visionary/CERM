from __future__ import annotations

from sklearn.datasets import make_classification

from cerm import CERMClassifier, CERMModelSummary, model_summary


def test_model_summary_is_useful_before_fit():
    summary = model_summary(CERMClassifier())

    assert isinstance(summary, CERMModelSummary)
    assert summary.estimator == "CERMClassifier"
    assert summary.fitted is False
    assert summary.task is None
    assert summary.n_features_in is None
    assert summary.search_effort == "thorough"
    assert summary.state_detail == "fine"
    assert summary.interaction_order == 2
    assert "unfitted" in str(summary)
    assert summary.to_dict()["fitted"] is False


def test_model_summary_reports_compact_fitted_classifier_state():
    X, y = make_classification(
        n_samples=140,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=17,
    )
    model = CERMClassifier(
        max_features=6,
        pair_feature_limit=4,
        interaction_order=1,
        resource_policy="ignore",
        random_state=17,
    ).fit(X, y)

    summary = model_summary(model)

    assert summary.fitted is True
    assert summary.task == "binary"
    assert summary.n_features_in == 6
    assert summary.n_adapted_features == 6
    assert summary.selected_features is not None
    assert summary.selected_features >= 1
    assert summary.pair_count == 0
    assert summary.block_count == 0
    assert summary.fit_seconds is not None
    assert summary.fit_seconds >= 0.0
    assert summary.model_bytes is not None
    assert summary.model_bytes > 0
    assert summary.interaction_order == 1
    assert "main_effects_only" in summary.active_reductions
    rendered = str(summary)
    assert "features:" in rendered
    assert "interactions:" in rendered
    assert "resources:" in rendered
