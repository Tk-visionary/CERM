from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from cerm import CERMClassifier


def test_accepts_multiclass_and_can_disable_it():
    X = np.arange(45, dtype=float).reshape(15, 3)
    y = np.asarray([0, 1, 2] * 5)
    fitted = CERMClassifier(
        preset="balanced",
        max_features=3,
        max_interaction_features=3,
        max_interactions=2,
    ).fit(X, y)
    assert fitted.predict_proba(X).shape == (15, 3)
    with pytest.raises(ValueError, match="multiclass"):
        CERMClassifier(multiclass_strategy="error").fit(X, y)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"representation_strategy": "unknown"}, "representation_strategy"),
        ({"class_specific_budget": -1}, "class_specific_budget"),
        ({"class_specific_budget": 1.5}, "class_specific_budget"),
    ],
)
def test_adaptive_representation_parameter_validation(params, message):
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    with pytest.raises((TypeError, ValueError), match=message):
        CERMClassifier(**params).fit(X, y)


def test_requires_dataframe_after_dataframe_fit():
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "cat": ["a", "b"] * 40})
    y = np.asarray([0, 1] * 40)
    model = CERMClassifier(
        max_features=4,
        pair_feature_limit=4,
        categorical_features="auto",
        random_state=31,
    ).fit(frame, y)
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict_proba(np.zeros((5, 2)))


def test_deterministic_fit():
    X, y = make_classification(
        n_samples=130,
        n_features=5,
        n_informative=3,
        random_state=37,
    )
    kwargs = dict(max_features=5, pair_feature_limit=5, random_state=37)
    a = CERMClassifier(**kwargs).fit(X, y)
    b = CERMClassifier(**kwargs).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=0.0, rtol=0.0)


def test_diversified_ranking_is_explicitly_opt_in():
    X, y = make_classification(
        n_samples=150,
        n_features=7,
        n_informative=4,
        n_redundant=0,
        random_state=47,
    )
    model = CERMClassifier(
        ranking_kind="diversified",
        max_features=7,
        pair_feature_limit=6,
        max_interactions=4,
        prediction_backend="semantic",
        random_state=47,
    ).fit(X, y)
    assert model.ranking_kind == "diversified"
    assert model.model_.base_.ranking_kind == "diversified"
    assert model.predict_proba(X[:12]).shape == (12, 2)
