from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from cerm import CERMClassifier


def test_cross_fitted_selection_is_deterministic_and_compilable(tmp_path):
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=51,
    )
    kwargs = dict(
        max_features=6,
        pair_feature_limit=6,
        random_state=51,
        selection_strategy="cross_fitted",
        selection_folds=3,
    )
    a = CERMClassifier(**kwargs).fit(X, y)
    b = CERMClassifier(**kwargs).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=0, rtol=0)
    assert len(a.model_.cv_selection_results_) > 1
    optimized = a.optimize("balanced")
    np.testing.assert_allclose(
        a.predict_proba(X[:30]), optimized.predict_proba(X[:30]), atol=2e-12, rtol=0
    )
