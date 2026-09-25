from __future__ import annotations

import numpy as np
import pytest

from cerm._internal.cerm_fused_regression import (
    FusedResidualCERMRegressor,
    FusedThresholdHead,
)
from cerm._internal.cerm_hierarchical_residual import (
    NestedQuantileEncoder,
    _mi_columns,
    _rank_pairs as _rank_binary_pairs,
)
from cerm._internal.cerm_weighted_representation import (
    WeightedNestedQuantileEncoder,
    aggregate_feature_scores,
    canonical_sample_weight,
    frequency_weighted_quantile,
    rank_pairs,
)
from cerm.shared_multitask import (
    _aggregate_feature_scores as _historical_feature_scores,
    _rank_pairs as _historical_rank_pairs,
)


def _assert_encoder_equal(left, right):
    assert left.levels == right.levels
    assert left.n_features_in_ == right.n_features_in_
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


def _numeric_data(seed=20260819, n=80, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return X


def test_frequency_quantile_matches_explicit_integer_replication():
    rng = np.random.default_rng(17)
    values = rng.normal(size=41)
    weights = rng.integers(0, 5, size=len(values))
    weights[0] = 3
    quantiles = np.linspace(0.0, 1.0, 23)
    expected = np.quantile(np.repeat(values, weights), quantiles)
    actual = frequency_weighted_quantile(values, quantiles, weights)
    np.testing.assert_array_equal(actual, expected)


def test_weighted_encoder_none_and_ones_use_historical_semantics():
    X = _numeric_data(n=97, d=5)
    historical = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16))
    historical_states = historical.fit_transform(X)

    weighted_none = WeightedNestedQuantileEncoder(max_bins=16, levels=(4, 8, 16))
    none_states = weighted_none.fit_transform(X, sample_weight=None)
    _assert_encoder_equal(weighted_none, historical)
    for level in historical.levels:
        np.testing.assert_array_equal(none_states[level], historical_states[level])

    weighted_ones = WeightedNestedQuantileEncoder(max_bins=16, levels=(4, 8, 16))
    one_states = weighted_ones.fit_transform(
        X, sample_weight=np.ones(len(X), dtype=np.float64)
    )
    _assert_encoder_equal(weighted_ones, historical)
    for level in historical.levels:
        np.testing.assert_array_equal(one_states[level], historical_states[level])


def test_zero_weight_row_does_not_change_weighted_encoder():
    X = _numeric_data(n=73, d=4)
    weights = 0.5 + np.arange(len(X), dtype=np.float64) / len(X)
    base = WeightedNestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
        X, sample_weight=weights
    )

    extra = np.full((1, X.shape[1]), 1e12)
    augmented = WeightedNestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
        np.vstack([X, extra]),
        sample_weight=np.r_[weights, 0.0],
    )
    _assert_encoder_equal(base, augmented)


def test_integer_weighted_encoder_matches_explicit_replication():
    X = _numeric_data(n=61, d=5)
    weights = 1 + (np.arange(len(X)) % 4)
    weighted = WeightedNestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
        X, sample_weight=weights
    )
    duplicated_X = np.repeat(X, weights, axis=0)
    duplicated = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
        duplicated_X
    )
    _assert_encoder_equal(weighted, duplicated)
    for level in weighted.levels:
        np.testing.assert_array_equal(
            weighted.transform(X)[level], duplicated.transform(X)[level]
        )


def test_weighted_multilabel_unary_and_pair_ranking_match_replication():
    rng = np.random.default_rng(211)
    states = rng.integers(0, 8, size=(91, 7), dtype=np.int64)
    labels = np.column_stack(
        [
            (states[:, 0] >= 4).astype(np.int64),
            ((states[:, 1] + states[:, 2]) % 3 == 0).astype(np.int64),
            (states[:, 3] == states[:, 4]).astype(np.int64),
        ]
    )
    weights = 1 + (np.arange(len(states)) % 3)
    repeated_states = np.repeat(states, weights, axis=0)
    repeated_labels = np.repeat(labels, weights, axis=0)

    weighted_scores = aggregate_feature_scores(
        states, labels, "multilabel", sample_weight=weights
    )
    repeated_scores = _historical_feature_scores(
        repeated_states, repeated_labels, "multilabel"
    )
    np.testing.assert_allclose(weighted_scores, repeated_scores, rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(
        np.argsort(-weighted_scores, kind="stable"),
        np.argsort(-repeated_scores, kind="stable"),
    )

    weighted_pairs = rank_pairs(
        states,
        labels,
        "multilabel",
        limit=8,
        feature_limit=7,
        aggregation="mean",
        sample_weight=weights,
    )
    repeated_pairs = _historical_rank_pairs(
        repeated_states,
        repeated_labels,
        "multilabel",
        8,
        7,
        "mean",
    )
    assert weighted_pairs == repeated_pairs


def test_weighted_multiclass_ranking_matches_replication():
    rng = np.random.default_rng(731)
    states = rng.integers(0, 6, size=(83, 6), dtype=np.int64)
    target = ((states[:, 0] + 2 * states[:, 1] + states[:, 2]) % 4).astype(
        np.int64
    )
    weights = 1 + (np.arange(len(states)) % 4)
    repeated_states = np.repeat(states, weights, axis=0)
    repeated_target = np.repeat(target, weights)

    weighted_scores = aggregate_feature_scores(
        states, target, "multiclass", sample_weight=weights
    )
    repeated_scores = _historical_feature_scores(
        repeated_states, repeated_target, "multiclass"
    )
    np.testing.assert_allclose(weighted_scores, repeated_scores, rtol=0.0, atol=2e-15)
    assert rank_pairs(
        states,
        target,
        "multiclass",
        limit=7,
        feature_limit=6,
        aggregation="joint",
        sample_weight=weights,
    ) == _historical_rank_pairs(
        repeated_states,
        repeated_target,
        "multiclass",
        7,
        6,
        "joint",
    )


def test_existing_binary_weighted_mi_matches_replication():
    rng = np.random.default_rng(119)
    states = rng.integers(0, 8, size=(101, 6), dtype=np.int64)
    target = ((states[:, 0] >= 3) ^ (states[:, 2] >= 5)).astype(np.int64)
    weights = 1 + (np.arange(len(states)) % 3)
    repeated_states = np.repeat(states, weights, axis=0)
    repeated_target = np.repeat(target, weights)

    weighted_scores = _mi_columns(states, target, sample_weight=weights)
    repeated_scores = _mi_columns(repeated_states, repeated_target)
    np.testing.assert_allclose(weighted_scores, repeated_scores, rtol=0.0, atol=2e-15)
    assert _rank_binary_pairs(
        states,
        target,
        max_pairs=7,
        feature_limit=6,
        sample_weight=weights,
    ) == _rank_binary_pairs(
        repeated_states,
        repeated_target,
        max_pairs=7,
        feature_limit=6,
    )


def test_fused_shared_representation_and_head_match_replication():
    rng = np.random.default_rng(2026081901)
    X = rng.normal(size=(96, 5))
    residual = (
        0.8 * np.sin(X[:, 0])
        + 0.5 * X[:, 1] * X[:, 2]
        + 0.15 * rng.normal(size=len(X))
    )
    weights = 1 + (np.arange(len(X)) % 3)
    quantiles = np.arange(1, 7, dtype=np.float64) / 7
    thresholds = frequency_weighted_quantile(residual, quantiles, weights)
    thresholds = np.unique(thresholds)
    labels = (residual[:, None] > thresholds[None, :]).astype(np.int32)

    shell = FusedResidualCERMRegressor(
        max_features=5,
        max_interaction_features=5,
        max_pairs=3,
        max_fine_pairs=1,
        max_bins=16,
        n_bins=7,
        max_iter=80,
        tol=1e-6,
    )
    shell._core_feature_kinds_ = None
    shell._core_feature_cardinalities_ = None
    weighted_bank = shell._feature_pair_bank(
        X, labels, X[:11], sample_weight=weights
    )

    repeated_X = np.repeat(X, weights, axis=0)
    repeated_labels = np.repeat(labels, weights, axis=0)
    repeated_bank = shell._feature_pair_bank(
        repeated_X, repeated_labels, X[:11], sample_weight=None
    )
    weighted_encoder, weighted_features, weighted_pairs = weighted_bank[:3]
    repeated_encoder, repeated_features, repeated_pairs = repeated_bank[:3]
    np.testing.assert_array_equal(weighted_features, repeated_features)
    assert weighted_pairs == repeated_pairs
    _assert_encoder_equal(weighted_encoder, repeated_encoder)

    weighted_design = weighted_bank[5].train
    repeated_design = repeated_bank[5].train
    valid_weighted = weighted_bank[5].valid
    valid_repeated = repeated_bank[5].valid
    lower = float(residual.min())
    upper = float(residual.max())
    weighted_head = FusedThresholdHead(C=0.05, max_iter=100, tol=1e-8).fit(
        weighted_design,
        labels,
        thresholds,
        lower,
        upper,
        sample_weight=weights,
    )
    repeated_head = FusedThresholdHead(C=0.05, max_iter=100, tol=1e-8).fit(
        repeated_design,
        repeated_labels,
        thresholds,
        lower,
        upper,
    )
    np.testing.assert_allclose(
        weighted_head.predict_residual(valid_weighted),
        repeated_head.predict_residual(valid_repeated),
        rtol=0.0,
        atol=2e-9,
    )


def test_fused_large_n_arbitrary_weight_smoke():
    rng = np.random.default_rng(2026081902)
    X = rng.normal(size=(150, 5))
    y = 0.9 * X[:, 0] + np.sin(X[:, 1]) + 0.5 * X[:, 2] * X[:, 3]
    weights = 0.2 + rng.lognormal(mean=0.0, sigma=0.8, size=len(X))
    model = FusedResidualCERMRegressor(
        n_bins=6,
        max_features=5,
        max_interaction_features=5,
        max_pairs=2,
        max_fine_pairs=0,
        max_bins=8,
        small_n_threshold=64,
        max_iter=35,
        tol=2e-4,
        random_state=20260819,
    ).fit(X, y, sample_weight=weights)
    prediction = model.predict(X[:17])
    assert prediction.shape == (17,)
    assert np.all(np.isfinite(prediction))
    assert model.fit_diagnostics_["selection_mode"] == "large_n_shared_holdout"


def test_extreme_positive_real_weights_remain_finite():
    values = np.asarray([-3.0, -1.0, 0.5, 2.0, 9.0])
    weights = np.asarray([1e12, 3e11 + 0.5, 2e10 + 0.25, 7e11, 9e9 + 0.75])
    result = frequency_weighted_quantile(
        values, np.asarray([0.1, 0.5, 0.9]), weights
    )
    assert np.all(np.isfinite(result))

    states = np.asarray([[0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.int64)
    labels = np.asarray([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]], dtype=np.int64)
    scores = aggregate_feature_scores(
        states, labels, "multilabel", sample_weight=weights
    )
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize(
    "weights,match",
    [
        ([-1.0, 1.0], "negative"),
        ([np.nan, 1.0], "NaN or infinity"),
        ([0.0, 0.0], "positive"),
        ([1.0], "length mismatch"),
    ],
)
def test_invalid_weight_validation(weights, match):
    with pytest.raises(ValueError, match=match):
        canonical_sample_weight(weights, 2)
