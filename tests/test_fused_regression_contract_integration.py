from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from cerm._internal.cerm_fused_regression import FusedResidualCERMRegressor
from cerm._internal.cerm_fused_regression_contracts import (
    PortableFusedRegressionProgram,
    compile_typed_fused_regression_native,
    semantic_program_from_typed_fused_regressor,
)


def _typed_problem(seed: int = 20260819):
    rng = np.random.default_rng(seed)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    cat = np.asarray(["a", "b", "c", "d"])[rng.integers(0, 4, size=n)]
    missing = np.arange(n) % 11 == 0
    x_observed = x.copy()
    x_observed[missing] = np.nan
    frame = pd.DataFrame({"x": x_observed, "z": z, "cat": cat})
    cat_effect = np.choose(
        np.searchsorted(np.asarray(["a", "b", "c", "d"]), cat),
        [-0.8, -0.1, 0.4, 1.1],
    )
    y = (
        0.7 * x
        + np.sin(1.8 * z)
        + cat_effect
        + 0.35 * x * z
        + 0.08 * rng.normal(size=n)
    )
    return frame, y


def _fit_typed_raw_model():
    X, y = _typed_problem()
    model = FusedResidualCERMRegressor(
        n_bins=8,
        max_features=8,
        max_interaction_features=8,
        max_pairs=2,
        raw_max_pairs=2,
        max_fine_pairs=0,
        small_n_threshold=1000,
        small_n_folds=3,
        # Force the small-N candidate through the promotion boundary so this
        # focused contract test exercises the residual native/semantic path.
        small_n_min_oof_improvement=-1.0,
        categorical_features=("cat",),
        category_identity="binary",
        missing_policy="always",
        max_iter=60,
        tol=1e-4,
        random_state=20260819,
    ).fit(X, y)
    assert model.selected_kind_ == "raw_fused"
    assert model.adapter_ is not None
    assert model.n_adapted_features_ > model.n_features_in_
    return model, X


def test_typed_semantic_program_preserves_raw_schema_and_adapter(tmp_path):
    model, X = _fit_typed_raw_model()
    program = semantic_program_from_typed_fused_regressor(model)

    assert program.input_n_features == model.n_features_in_
    assert program.core_n_features == model.n_adapted_features_
    assert program.core.metadata["n_features_in"] == model.n_adapted_features_
    np.testing.assert_allclose(
        program.predict(X.iloc[:23]),
        model.predict(X.iloc[:23]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        program.predict_survival(X.iloc[7:19]),
        model.predict_survival(X.iloc[7:19]),
        rtol=0.0,
        atol=1e-12,
    )

    manifest = program.export(tmp_path / "portable")
    loaded = PortableFusedRegressionProgram.load(manifest)
    assert loaded.core_n_features == model.n_adapted_features_
    np.testing.assert_allclose(
        loaded.predict(X.iloc[:31]),
        model.predict(X.iloc[:31]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        loaded.predict_survival(X.iloc[:17]),
        model.predict_survival(X.iloc[:17]),
        rtol=0.0,
        atol=1e-12,
    )

    with pytest.raises((ValueError, KeyError)):
        loaded.predict(X.drop(columns=["cat"]))


def test_typed_semantic_export_is_deterministic(tmp_path):
    model, _ = _fit_typed_raw_model()
    program = semantic_program_from_typed_fused_regressor(model)
    first = tmp_path / "first"
    second = tmp_path / "second"
    program.export(first)
    program.export(second)
    for name in (
        "model.json",
        "model.npz",
        "adapter.json",
        "adapter.npz",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.skipif(shutil.which("c++") is None, reason="C++ compiler unavailable")
def test_typed_native_program_adapts_before_numeric_core(tmp_path):
    model, X = _fit_typed_raw_model()
    compiled = compile_typed_fused_regression_native(model, tmp_path / "typed_fused")

    assert compiled.input_n_features == model.n_features_in_
    assert compiled.core_n_features == model.n_adapted_features_
    np.testing.assert_allclose(
        compiled.predict(X.iloc[:1]),
        model.predict(X.iloc[:1]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compiled.predict(X.iloc[:37]),
        model.predict(X.iloc[:37]),
        rtol=0.0,
        atol=1e-12,
    )
    assert compiled.artifact_bytes > 0
    assert compiled.source_bytes > 0

    with pytest.raises((ValueError, KeyError)):
        compiled.predict(X.drop(columns=["z"]))
