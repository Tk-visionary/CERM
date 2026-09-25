from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from cerm import CERMClassifier, CERMResourceLimitError, CERMSearchCV
from cerm._internal.cerm_hierarchical_residual import (
    _rank_pairs,
    _rank_pairs_newton,
)
from cerm._internal.cerm_quotient_block import QuotientBlockCERM
from cerm._internal.cerm_stable_quotient_block import StableQuotientBlockCERM


def test_dataframe_training_matrix_is_ephemeral():
    n = 180
    frame = pd.DataFrame(
        {
            "x": np.linspace(-2.0, 2.0, n),
            "cat": [f"c{i % 11}" for i in range(n)],
            "missing": np.where(np.arange(n) % 9 == 0, np.nan, np.arange(n)),
        }
    )
    y = (frame["x"].to_numpy() + (np.arange(n) % 11 == 7) > 0).astype(int)
    model = CERMClassifier(
        max_features=8,
        pair_feature_limit=6,
        categorical_features="auto",
        random_state=311,
    ).fit(frame, y)
    assert model.adapter_output_ is None
    assert model.retained_adapter_output_bytes_ == 0
    assert model.transient_adapted_matrix_bytes_ > 0
    assert model.fit_diagnostics_.retained_adapter_output_bytes == 0


def test_resource_plan_and_fail_fast_budget():
    X, y = make_classification(
        n_samples=160,
        n_features=12,
        n_informative=7,
        n_redundant=0,
        random_state=313,
    )
    estimator = CERMClassifier(
        max_features=12,
        pair_feature_limit=10,
        resource_policy="raise",
        max_pair_evaluations=1,
        random_state=313,
    )
    plan = estimator.estimate_fit_resources(X)
    assert plan.pair_candidates_per_ranking == 45
    assert plan.estimated_pair_evaluations > 1
    with pytest.raises(CERMResourceLimitError, match="pair evaluations"):
        estimator.fit(X, y)


def test_search_fit_budget_raises_before_full_cv():
    X, y = make_classification(
        n_samples=140,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=317,
    )
    search = CERMSearchCV(
        CERMClassifier(max_features=6, pair_feature_limit=4, random_state=317),
        {"ranking_l2": [0.5, 1.0], "pair_feature_limit": [2, 4]},
        cv=2,
        cache_prefilter=False,
        refit=False,
        max_full_cv_fits=2,
        budget_policy="raise",
    )
    with pytest.raises(CERMResourceLimitError, match="planned full fits"):
        search.fit(X, y)


def test_streaming_pair_top_k_matches_full_reference():
    rng = np.random.default_rng(331)
    C = rng.integers(0, 16, size=(260, 28), dtype=np.int16)
    y = rng.integers(0, 2, size=260, dtype=np.int32)
    full_mi = _rank_pairs(C, y, 28 * 27 // 2, feature_limit=28)
    top_mi = _rank_pairs(C, y, 17, feature_limit=28)
    assert top_mi == full_mi[:17]
    full_newton = _rank_pairs_newton(C, y, 28 * 27 // 2, feature_limit=28)
    top_newton = _rank_pairs_newton(C, y, 17, feature_limit=28)
    assert top_newton == full_newton[:17]


def test_streaming_block_top_k_matches_full_reference():
    rng = np.random.default_rng(337)
    n, d = 240, 12
    C4 = rng.integers(0, 4, size=(n, d), dtype=np.int16)
    C16 = rng.integers(0, 16, size=(n, d), dtype=np.int16)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    p = np.clip(rng.uniform(0.1, 0.9, size=n), 1e-5, 1 - 1e-5)

    full_model = QuotientBlockCERM(pair_feature_limit=d, random_state=337)
    full = full_model._rank_blocks(C4, C16, y, p, 5.0, 1.0)
    streamed = full_model._rank_blocks(
        C4, C16, y, p, 5.0, 1.0, max_results=9
    )
    assert streamed == full[:9]

    stable_model = StableQuotientBlockCERM(pair_feature_limit=d, random_state=337)
    stable_full = stable_model._stable_rank_blocks(C4, C16, y, p)
    stable_streamed = stable_model._stable_rank_blocks(
        C4, C16, y, p, max_results=9
    )
    assert stable_streamed == stable_full[:9]


def test_practical_profile_reduces_solver_plan_without_pair_truncation():
    X, _ = make_classification(
        n_samples=240,
        n_features=30,
        n_informative=10,
        random_state=347,
    )
    full = CERMClassifier(
        pair_feature_limit=24,
        search_profile="full_exact",
        resource_policy="ignore",
    ).estimate_fit_resources(X)
    practical = CERMClassifier(
        pair_feature_limit=24,
        search_profile="practical",
        resource_policy="ignore",
    ).estimate_fit_resources(X)
    assert practical.effective_pair_features == 24
    assert practical.estimated_pair_evaluations == full.estimated_pair_evaluations
    assert practical.estimated_solver_calls < full.estimated_solver_calls


def test_aggressive_profile_caps_effective_pair_budget():
    X, _ = make_classification(
        n_samples=240,
        n_features=40,
        n_informative=12,
        random_state=349,
    )
    plan = CERMClassifier(
        pair_feature_limit=32,
        search_profile="aggressive",
        resource_policy="ignore",
    ).estimate_fit_resources(X)
    assert plan.effective_pair_features == 20
    assert plan.pair_candidates_per_ranking == 190
