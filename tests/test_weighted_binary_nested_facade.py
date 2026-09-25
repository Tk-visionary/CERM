from __future__ import annotations

import numpy as np

from cerm._internal import cerm_hierarchical_residual_legacy as legacy
from cerm._internal.cerm_hierarchical_residual import (
    HierConfig,
    HierarchicalResidualCERM,
)


def _data(seed=2026081903, n=120, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    score = X[:, 0] + 0.7 * X[:, 1] * X[:, 2] - 0.35 * X[:, 3]
    y = (score + 0.2 * rng.normal(size=n) > 0.0).astype(np.int64)
    return X, y


def _kwargs():
    return dict(
        max_features=6,
        pair_feature_limit=6,
        search_profile="compact",
        max_bins=8,
        selection_subsample=1.0,
        n_jobs=1,
        random_state=20260819,
        selection_rule="best",
        encoder_kind="quantile",
        ranking_kind="mi",
    )


def _assert_encoder_equal(left, right):
    assert left.levels == right.levels
    np.testing.assert_array_equal(left.direct_state_mask_, right.direct_state_mask_)
    np.testing.assert_array_equal(
        left.direct_state_cardinalities_, right.direct_state_cardinalities_
    )
    for first, second in zip(left.thresholds_, right.thresholds_):
        np.testing.assert_array_equal(first, second)
    for level in left.levels:
        np.testing.assert_array_equal(
            left.cardinalities_[level], right.cardinalities_[level]
        )
        for first, second in zip(left.maps_[level], right.maps_[level]):
            np.testing.assert_array_equal(first, second)


def test_binary_none_path_matches_preserved_historical_model_bitwise():
    X, y = _data()
    current = HierarchicalResidualCERM(**_kwargs()).fit(X, y)
    historical = legacy.HierarchicalResidualCERM(**_kwargs()).fit(X, y)
    assert current.best_config_ == historical.best_config_
    np.testing.assert_array_equal(current.feature_idx_, historical.feature_idx_)
    assert current.pairs_ == historical.pairs_
    _assert_encoder_equal(current.encoder_, historical.encoder_)
    np.testing.assert_array_equal(
        current.predict_proba(X[:31]), historical.predict_proba(X[:31])
    )


def test_binary_all_ones_canonicalizes_to_historical_path():
    X, y = _data(seed=2026081904)
    current = HierarchicalResidualCERM(**_kwargs()).fit(
        X, y, sample_weight=np.ones(len(X), dtype=np.float64)
    )
    historical = legacy.HierarchicalResidualCERM(**_kwargs()).fit(X, y)
    assert current.best_config_ == historical.best_config_
    _assert_encoder_equal(current.encoder_, historical.encoder_)
    np.testing.assert_array_equal(
        current.predict_proba(X[:29]), historical.predict_proba(X[:29])
    )


def test_binary_integer_weight_fixed_structure_matches_row_duplication():
    X, y = _data(seed=2026081905, n=96)
    weights = 1 + (np.arange(len(X)) % 3)
    config = HierConfig(n_pairs=4, n_fine_pairs=0, max_main_level=8, C=0.2)

    weighted = HierarchicalResidualCERM(**_kwargs())._fit_structure(
        X, y, config, sample_weight=weights
    )
    repeated_X = np.repeat(X, weights, axis=0)
    repeated_y = np.repeat(y, weights)
    duplicated = legacy.HierarchicalResidualCERM(**_kwargs())._fit_structure(
        repeated_X, repeated_y, config, sample_weight=None
    )

    _assert_encoder_equal(weighted.encoder_, duplicated.encoder_)
    np.testing.assert_array_equal(weighted.feature_idx_, duplicated.feature_idx_)
    assert weighted.pairs_ == duplicated.pairs_
    np.testing.assert_allclose(
        weighted.predict_proba(X[:37]),
        duplicated.predict_proba(X[:37]),
        rtol=0.0,
        atol=2e-12,
    )


def test_binary_zero_weight_extreme_row_does_not_change_fixed_structure():
    X, y = _data(seed=2026081906, n=98)
    weights = 0.5 + (np.arange(len(X)) % 5) / 3.0
    config = HierConfig(n_pairs=4, n_fine_pairs=0, max_main_level=8, C=0.2)

    base = HierarchicalResidualCERM(**_kwargs())._fit_structure(
        X, y, config, sample_weight=weights
    )
    X_extra = np.vstack([X, np.full((1, X.shape[1]), 1e15)])
    y_extra = np.r_[y, 1]
    weight_extra = np.r_[weights, 0.0]
    augmented = HierarchicalResidualCERM(**_kwargs())._fit_structure(
        X_extra, y_extra, config, sample_weight=weight_extra
    )

    _assert_encoder_equal(base.encoder_, augmented.encoder_)
    np.testing.assert_array_equal(base.feature_idx_, augmented.feature_idx_)
    assert base.pairs_ == augmented.pairs_
    np.testing.assert_allclose(
        base.predict_proba(X[:33]),
        augmented.predict_proba(X[:33]),
        rtol=0.0,
        atol=2e-12,
    )
