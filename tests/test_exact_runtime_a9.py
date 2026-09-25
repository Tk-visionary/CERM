from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from cerm import CERMClassifier, CERMRidgeRegressor as CERMRegressor
from cerm.regression import FiniteStateRidgeRegressor


def _legacy_regression_lookup(model: FiniteStateRidgeRegressor, X: np.ndarray) -> np.ndarray:
    states = model.encoder_.transform_columns(np.asarray(X, dtype=np.float64), model.feature_idx_)
    active_levels = tuple(level for level in model.levels if level <= model.config_.max_main_level)
    score = np.full(len(X), float(model.intercept_), dtype=np.float64)
    lookup_index = 0
    for feature in range(len(model.feature_idx_)):
        for level in active_levels:
            state = states[level][:, feature]
            table = model.lookup_[lookup_index]
            valid = (state >= 0) & (state < len(table))
            score[valid] += table[state[valid]]
            lookup_index += 1
    if model.pairs_:
        pair_level = max(
            level for level in active_levels
            if level <= min(model.config_.max_main_level, 8)
        )
        pair_states = states[pair_level]
        for pair_index, (left, right) in enumerate(model.pairs_):
            state = (
                pair_states[:, int(left)].astype(np.int64)
                * int(model.pair_cardinalities_[pair_index])
                + pair_states[:, int(right)]
            )
            table = model.lookup_[lookup_index]
            valid = (state >= 0) & (state < len(table))
            score[valid] += table[state[valid]]
            lookup_index += 1
    assert lookup_index == len(model.lookup_)
    return score


def test_direct_parent_thresholds_reproduce_historical_quotient_states():
    rng = np.random.default_rng(90)
    X = rng.normal(size=(1200, 7))
    y = (X[:, 0] > 0).astype(np.float64)
    model = FiniteStateRidgeRegressor(
        max_features=7,
        max_bins=16,
        max_interaction_features=6,
        max_interactions=3,
        include_linear=False,
        random_state=90,
    ).fit(X, y)
    encoder = model.encoder_
    probe = rng.normal(size=(1800, 7))
    historical = encoder.transform(probe)
    for level in encoder.levels:
        direct = encoder.transform_level_columns(probe, level, np.arange(probe.shape[1]))
        np.testing.assert_array_equal(direct, historical[level])


def test_one_level_regression_executor_is_bitwise_equal_to_historical_lookup_stream():
    rng = np.random.RandomState(91)
    X = rng.randn(5000, 18)
    y = 3.0 * (X[:, 0] > 0) + 5.0 * ((X[:, 1] > 0) & (X[:, 2] > 0))
    model = FiniteStateRidgeRegressor(
        max_features=12,
        max_bins=16,
        max_interaction_features=10,
        max_interactions=6,
        preset="accurate",
        include_linear=False,
        random_state=91,
    ).fit(X[:2600], y[:2600])
    expected = _legacy_regression_lookup(model, X)
    actual = model.decision_function(X)
    np.testing.assert_array_equal(actual, expected)


def test_precomposed_pair_execution_maps_reproduce_historical_pair_codes():
    rng = np.random.RandomState(92)
    X = rng.randn(6000, 20)
    y = 2.0 * (X[:, 0] > 0) + 4.0 * ((X[:, 1] > 0) & (X[:, 2] > 0))
    model = FiniteStateRidgeRegressor(
        max_features=14,
        max_bins=16,
        max_interaction_features=12,
        max_interactions=8,
        preset="accurate",
        include_linear=False,
        random_state=92,
    ).fit(X[:3200], y[:3200])
    model._ensure_execution_maps()
    if not model.pairs_:
        pytest.skip("validation selected no pair for this deterministic task")
    states = model._execution_states(X[:1000])
    pair_level_index = int(model.execution_pair_level_index_)
    for pair_index, (left, right, exec_right_card, code_map) in enumerate(model.execution_pair_plans_):
        old_left = model.execution_parent_maps_[left][pair_level_index][states[:, left]]
        old_right = model.execution_parent_maps_[right][pair_level_index][states[:, right]]
        historical = (
            old_left.astype(np.int64) * int(model.pair_cardinalities_[pair_index])
            + old_right
        )
        execution_joint = states[:, left].astype(np.int64) * int(exec_right_card) + states[:, right]
        np.testing.assert_array_equal(code_map[execution_joint], historical)


def test_hybrid_no_block_executor_remains_bitwise_equal_to_base_lookup():
    rng = np.random.default_rng(93)
    X = rng.normal(size=(2600, 12))
    y = (1.2 * X[:, 0] - 0.8 * X[:, 1] > 0).astype(int)
    model = CERMClassifier(
        preset="balanced",
        max_features=10,
        max_interaction_features=8,
        max_interactions=0,
        random_state=93,
    ).fit(X, y)
    hybrid = model.model_
    if getattr(hybrid, "execution_groups_", ()):
        pytest.skip("control task unexpectedly selected conditional blocks")
    probe = rng.normal(size=(1300, 12))
    hybrid.base_._ensure_execution_maps()
    states = hybrid.base_.encoder_.transform_level_columns(
        probe, hybrid.base_.execution_level_, hybrid.base_.feature_idx_
    )
    expected = hybrid.base_._decision_from_execution_states_with_lookup(
        states, hybrid._base_lookup_tables(), hybrid.intercept_, intercept_last=True
    )
    np.testing.assert_array_equal(hybrid.decision_function(probe), expected)


def test_native_regression_compact_state_lowering_is_exact(tmp_path: Path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    rng = np.random.default_rng(94)
    X = rng.normal(size=(2600, 14))
    y = (X[:, 0] > 0).astype(float) + 0.03 * rng.normal(size=len(X))
    model = CERMRegressor(
        preset="accurate",
        max_features=10,
        max_bins=16,
        max_interaction_features=6,
        max_interactions=0,
        include_linear=False,
        random_state=94,
    ).fit(X, y)
    probe = rng.normal(size=(2400, 14))
    compiled = model.compile_native(tmp_path / "reg_a9")
    np.testing.assert_array_equal(compiled.predict(probe), model.predict(probe))


def test_shared_native_required_level_lowering_is_exact(tmp_path: Path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    rng = np.random.default_rng(95)
    X = rng.normal(size=(2200, 12))
    latent = X[:, 0] + 0.25 * rng.normal(size=len(X))
    y = np.digitize(latent, [-0.6, 0.0, 0.6])
    model = CERMClassifier(
        multiclass_strategy="shared",
        preset="balanced",
        max_bins=16,
        max_features=8,
        max_interaction_features=6,
        max_interactions=0,
        random_state=95,
    ).fit(X, y)
    probe = rng.normal(size=(1800, 12))
    compiled = model.compile_native(tmp_path / "shared_a9")
    np.testing.assert_allclose(
        compiled.predict_proba(probe), model.predict_proba(probe), rtol=0.0, atol=3e-16
    )
