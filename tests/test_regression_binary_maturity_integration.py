from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cerm import CERMFusedRegressor, PortableFusedRegressionProgram
from cerm._internal.cerm_weighted_representation import frequency_weighted_quantile


def test_weighted_quantile_dual_contract_is_explicit():
    values = np.asarray([0.0, 1.0, 10.0, 100.0])
    quantiles = np.asarray([0.25, 0.5, 0.75])

    integer_weight = np.asarray([1, 1, 1, 3], dtype=np.float64)
    expected_frequency = np.quantile(
        np.repeat(values, integer_weight.astype(np.int64)), quantiles
    )
    actual_frequency = frequency_weighted_quantile(
        values, quantiles, integer_weight
    )
    np.testing.assert_array_equal(actual_frequency, expected_frequency)

    # Frequency semantics intentionally depend on the virtual finite sample
    # size: scaling integer counts need not preserve Type-7 quantiles.
    scaled_frequency = frequency_weighted_quantile(
        values, quantiles, 2.0 * integer_weight
    )
    assert not np.array_equal(actual_frequency, scaled_frequency)

    real_weight = np.asarray([0.3, 0.7, 1.1, 2.3], dtype=np.float64)
    real_base = frequency_weighted_quantile(values, quantiles, real_weight)
    real_scaled = frequency_weighted_quantile(
        values, quantiles, 2.0 * real_weight
    )
    np.testing.assert_allclose(real_base, real_scaled, rtol=0.0, atol=2e-15)


def test_public_fused_portable_roundtrip_preserves_integer_column_labels(tmp_path):
    rng = np.random.default_rng(20260819)
    n = 48
    frame = pd.DataFrame(
        {
            10: rng.normal(size=n),
            20: np.where(np.arange(n) % 3 == 0, "a", "b"),
            30: rng.normal(size=n),
        }
    )
    frame.loc[::7, 30] = np.nan
    y = (
        0.8 * frame[10].to_numpy()
        + 0.5 * (frame[20].to_numpy() == "a").astype(float)
        + np.nan_to_num(frame[30].to_numpy(), nan=0.0)
    )

    model = CERMFusedRegressor(
        n_bins=4,
        max_bins=4,
        max_features=4,
        max_interaction_features=4,
        max_pairs=1,
        random_state=20260819,
    ).fit(frame, y)
    expected = model.predict(frame.iloc[:9])

    manifest = model.export(tmp_path / "portable")
    loaded = PortableFusedRegressionProgram.load(manifest)
    assert loaded.input_columns == (10, 20, 30)
    np.testing.assert_allclose(
        loaded.predict(frame.iloc[:9]), expected, rtol=0.0, atol=1e-12
    )

    with pytest.raises(ValueError):
        loaded.predict(frame.rename(columns={10: "10"}).iloc[:9])


def test_public_fused_embedding_configuration_is_explicitly_unsupported():
    model = CERMFusedRegressor(embedding_features=("embedding",))
    with pytest.raises(ValueError, match="embedding_features are not yet supported"):
        model._make_model()
