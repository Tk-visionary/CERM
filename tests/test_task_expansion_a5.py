import numpy as np
import pytest

from cerm import CERMGeneralizedRegressor
from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder


def _tweedie_like(seed=11, n=500):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    mu = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
    y = rng.poisson(0.8 * mu) * rng.gamma(2.0, 1.0 / 1.7, size=n)
    return X, y


def test_linear_representation_mode_matches_auto_when_auto_selects_linear():
    X, y = _tweedie_like()
    params = dict(
        loss="tweedie",
        tweedie_power=1.5,
        preset="balanced",
        max_features=8,
        max_interaction_features=6,
        max_interactions=4,
        random_state=11,
    )
    auto = CERMGeneralizedRegressor(**params).fit(X, y)
    linear = CERMGeneralizedRegressor(
        **params, representation_mode="linear"
    ).fit(X, y)
    if auto.fit_diagnostics_["selected_config"]["max_main_level"] == 0:
        assert np.array_equal(auto.predict(X), linear.predict(X))
        assert np.array_equal(auto.linear_coef_, linear.linear_coef_)
    assert linear.fit_diagnostics_["selected_config"] == {
        "max_main_level": 0,
        "n_pairs": 0,
        "alpha": 1.0,
    }
    assert "forced_linear_representation" in linear.fit_diagnostics_[
        "exact_rewrite_passes"
    ]


def test_representation_mode_validation():
    X, y = _tweedie_like(n=30)
    with pytest.raises(ValueError, match="representation_mode"):
        CERMGeneralizedRegressor(
            loss="tweedie", representation_mode="unknown"
        ).fit(X, y)


def test_projected_state_transform_is_exact():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 9))
    encoder = NestedQuantileEncoder(max_bins=16).fit(X)
    columns = np.asarray([7, 1, 5, 0], dtype=np.int64)
    full = encoder.transform(X)
    projected = encoder.transform_projected(X[:, columns], columns)
    for level in encoder.levels:
        assert np.array_equal(projected[level], full[level][:, columns])


def test_linear_representation_requires_linear_features():
    X, y = _tweedie_like(n=30)
    with pytest.raises(ValueError, match="requires include_linear"):
        CERMGeneralizedRegressor(
            loss="tweedie", representation_mode="linear", include_linear=False
        ).fit(X, y)
