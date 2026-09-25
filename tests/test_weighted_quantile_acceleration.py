from __future__ import annotations

import numpy as np
import pytest

from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder
from cerm._internal.cerm_training_core_runtime import (
    TrainingCoreUnavailable,
    load_native_training_core,
)
from cerm._internal.cerm_weighted_representation import frequency_weighted_quantile


def _pure_python_reference_weighted_quantile(values, quantiles, weights):
    active = weights > 0.0
    active_values = values[active]
    active_weights = weights[active]
    order = np.argsort(active_values, kind="stable")
    sorted_values = active_values[order]
    sorted_weights = active_weights[order]

    rounded = np.rint(sorted_weights)
    if np.array_equal(sorted_weights, rounded):
        counts = rounded.astype(np.int64)
        cumulative = np.cumsum(counts, dtype=np.int64)
        total = int(cumulative[-1])
        virtual_index = (float(total) - 1.0) * quantiles
        lower_rank = np.floor(virtual_index).astype(np.int64)
        upper_rank = np.ceil(virtual_index).astype(np.int64)
        fraction = virtual_index - lower_rank
        lower_value = sorted_values[
            np.searchsorted(cumulative, lower_rank, side="right")
        ]
        upper_value = sorted_values[
            np.searchsorted(cumulative, upper_rank, side="right")
        ]
        diff = upper_value - lower_value
        res = lower_value + diff * fraction
        use_upper = fraction >= 0.5
        if np.any(use_upper):
            res[use_upper] = upper_value[use_upper] - diff[use_upper] * (
                1.0 - fraction[use_upper]
            )
        return res

    scale = float(np.max(sorted_weights))
    stable_weights = sorted_weights / scale
    cumulative = np.cumsum(stable_weights, dtype=np.float64)
    positions = (cumulative - 0.5 * stable_weights) / float(cumulative[-1])
    return np.asarray(
        np.interp(
            quantiles,
            positions,
            sorted_values,
            left=float(sorted_values[0]),
            right=float(sorted_values[-1]),
        ),
        dtype=np.float64,
    )


def test_native_weighted_quantile_bitwise_parity():
    """Verify C++ native weighted_quantile matches pure Python reference bit-for-bit."""
    try:
        core = load_native_training_core()
    except TrainingCoreUnavailable:
        pytest.skip("native training core unavailable")

    rng = np.random.default_rng(12345)
    quantiles = np.arange(1, 16, dtype=np.float64) / 16.0

    # Test integer weights against pure Python reference
    for n in (10, 100, 1000, 10000):
        values = rng.standard_normal(n)
        weights_int = rng.integers(1, 10, size=n).astype(np.float64)
        py_ref_int = _pure_python_reference_weighted_quantile(
            values, quantiles, weights_int
        )
        cpp_int = core.weighted_quantile(values, weights_int, quantiles)
        assert np.array_equal(py_ref_int, cpp_int)

    # Test real weights against pure Python reference
    for n in (10, 100, 1000, 10000):
        values = rng.standard_normal(n)
        weights_real = rng.uniform(0.1, 5.0, size=n)
        py_ref_real = _pure_python_reference_weighted_quantile(
            values, quantiles, weights_real
        )
        cpp_real = core.weighted_quantile(values, weights_real, quantiles)
        assert np.allclose(py_ref_real, cpp_real, atol=1e-12, rtol=1e-12)


def test_encoder_direct_nominal_features_untouched():
    """Verify direct and nominal features are untouched by quantile thresholds."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 3))
    X[:, 0] = rng.integers(0, 4, size=n)  # direct feature

    kinds = ["categorical_quotient", "numeric", "numeric"]
    cards = [4, None, None]
    weights = rng.uniform(0.5, 2.0, size=n)

    encoder = NestedQuantileEncoder(
        max_bins=16,
        levels=(4, 8, 16),
        feature_kinds=kinds,
        feature_cardinalities=cards,
    ).fit(X, sample_weight=weights)

    assert encoder.direct_state_mask_[0]
    assert len(encoder.thresholds_[0]) == 0
    assert not encoder.direct_state_mask_[1]
    assert len(encoder.thresholds_[1]) > 0
    assert not encoder.direct_state_mask_[2]
    assert len(encoder.thresholds_[2]) > 0


def test_zero_weighted_rows_filtered_exactly():
    """Verify zero-weighted rows do not affect threshold calculation."""
    rng = np.random.default_rng(999)
    n = 1000
    values = rng.standard_normal(n)
    weights = rng.uniform(0.1, 2.0, size=n)
    # Set half of weights to 0
    weights[:500] = 0.0

    quantiles = np.arange(1, 16) / 16.0
    th_weighted = frequency_weighted_quantile(values, quantiles, weights)
    th_filtered = frequency_weighted_quantile(values[500:], quantiles, weights[500:])

    assert np.array_equal(th_weighted, th_filtered)


def test_contiguous_map_caching_and_multi_level_maps():
    """Verify multi-level mapping arrays are consistent and correctly shared."""
    rng = np.random.default_rng(100)
    n = 500
    p = 5
    X = rng.standard_normal((n, p))
    weights = rng.uniform(0.1, 3.0, size=n)

    encoder = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
        X, sample_weight=weights
    )

    for level in (4, 8, 16):
        assert level in encoder.maps_
        assert len(encoder.maps_[level]) == p
        for j in range(p):
            mapping = encoder.maps_[level][j]
            assert mapping.ndim == 1
            assert int(mapping.max(initial=0)) + 1 == encoder.cardinalities_[level][j]



def test_native_weighted_quantile_near_integer_weights_use_real_semantics():
    try:
        core = load_native_training_core()
    except TrainingCoreUnavailable:
        pytest.skip("native training core unavailable")

    values = np.array([-3.0, -0.5, 1.0, 4.0], dtype=np.float64)
    weights = np.array([1.0, 1.0 + 5e-13, 2.0, 3.0], dtype=np.float64)
    quantiles = np.array([0.125, 0.5, 0.875], dtype=np.float64)
    reference = _pure_python_reference_weighted_quantile(
        values, quantiles, weights
    )
    actual = core.weighted_quantile(values, weights, quantiles)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-12)


def test_public_weighted_quantile_does_not_drop_nonfinite_values():
    values = np.array([0.0, 1.0, np.nan, 3.0], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    quantiles = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    expected = _pure_python_reference_weighted_quantile(
        values, quantiles, weights
    )
    actual = frequency_weighted_quantile(values, quantiles, weights)
    np.testing.assert_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(
        np.nan_to_num(actual, nan=0.0),
        np.nan_to_num(expected, nan=0.0),
        rtol=0.0,
        atol=0.0,
    )


def test_native_weighted_quantile_rejects_invalid_direct_inputs():
    try:
        core = load_native_training_core()
    except TrainingCoreUnavailable:
        pytest.skip("native training core unavailable")

    with pytest.raises(TrainingCoreUnavailable):
        core.weighted_quantile(
            np.array([0.0, np.nan]),
            np.array([1.0, 1.0]),
            np.array([0.5]),
        )
