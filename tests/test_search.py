from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid

from cerm import CERMClassifier, CERMSearchCV


def _data():
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.0,
        random_state=17,
    )
    return X, y


def _estimator():
    return CERMClassifier(
        max_features=6,
        pair_feature_limit=4,
        ranking_kind="newton",
        selection_strategy="two_holdout",
        prediction_backend="optimized",
        random_state=23,
    )


def test_exhaustive_search_evaluates_all_candidates():
    X, y = _data()
    grid = {"ranking_l2": [0.5, 2.0], "pair_feature_limit": [2, 4]}
    search = CERMSearchCV(
        _estimator(),
        grid,
        cv=2,
        cache_prefilter=False,
        refit=True,
    ).fit(X, y)
    assert search.n_candidates_ == len(list(ParameterGrid(grid)))
    assert search.n_evaluated_candidates_ == search.n_candidates_
    assert np.isfinite(search.best_score_)
    assert search.predict_proba(X[:5]).shape == (5, 2)
    assert search.cv_results_["prefilter_selected"].all()


def test_cache_prefilter_reduces_full_cv_candidates():
    X, y = _data()
    grid = {
        "ranking_l2": [0.1, 0.3, 1.0, 3.0],
        "pair_feature_limit": [2, 4],
    }
    search = CERMSearchCV(
        _estimator(),
        grid,
        cv=2,
        cache_prefilter=True,
        prefilter_top_k=2,
        prefilter_min_candidates=4,
        refit=True,
    ).fit(X, y)
    assert search.n_candidates_ == 8
    assert 0 < search.n_evaluated_candidates_ < search.n_candidates_
    skipped = ~search.cv_results_["prefilter_selected"]
    assert skipped.any()
    assert np.isnan(search.cv_results_["mean_test_score"][skipped]).all()
    assert np.isfinite(search.cv_results_["prefilter_score"]).all()
    assert search.search_diagnostics_.proxy_fits == 2
    assert search.search_diagnostics_.full_cv_fits == 2 * search.n_evaluated_candidates_


def test_mi_grid_is_not_prefiltered():
    X, y = _data()
    estimator = _estimator().set_params(ranking_kind="mi")
    grid = {
        "ranking_l2": [0.1, 0.3, 1.0, 3.0],
        "pair_feature_limit": [2, 4],
    }
    search = CERMSearchCV(
        estimator,
        grid,
        cv=2,
        cache_prefilter=True,
        prefilter_top_k=2,
        prefilter_min_candidates=4,
        refit=False,
    ).fit(X, y)
    assert search.n_evaluated_candidates_ == search.n_candidates_
    assert search.search_diagnostics_.prefilter_groups == 0


def test_search_parameter_validation():
    X, y = _data()
    search = CERMSearchCV(_estimator(), {}, prefilter_top_k=0)
    try:
        search.fit(X, y)
    except ValueError as exc:
        assert "prefilter_top_k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_parallel_search_matches_serial_results():
    X, y = _data()
    grid = {"ranking_l2": [0.5, 2.0], "max_interaction_features": [2, 4]}
    common = dict(
        estimator=_estimator(),
        param_grid=grid,
        cv=2,
        cache_prefilter=False,
        refit=True,
        max_full_cv_fits=16,
    )
    serial = CERMSearchCV(n_jobs=1, **common).fit(X, y)
    parallel = CERMSearchCV(n_jobs=2, **common).fit(X, y)
    assert serial.best_params_ == parallel.best_params_
    np.testing.assert_allclose(
        serial.cv_results_["mean_test_score"],
        parallel.cv_results_["mean_test_score"],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        serial.predict_proba(X), parallel.predict_proba(X), atol=0.0, rtol=0.0
    )
