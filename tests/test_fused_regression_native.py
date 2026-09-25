import time

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, make_friedman1, make_regression
from sklearn.model_selection import train_test_split

from cerm._internal.cerm_fused_regression import FusedResidualCERMRegressor
from cerm._internal.cerm_fused_regression_native_codegen import (
    compile_fused_regression_native,
    render_fused_residual_native_source,
)


def _assert_native_contract(model, X, tmp_path, stem):
    compiled = compile_fused_regression_native(model, tmp_path / stem)
    probe = np.ascontiguousarray(X[: min(257, len(X))], dtype=np.float64)

    native = compiled.predict(probe)
    reference = model.predict(probe)
    assert np.max(np.abs(native - reference)) <= 1e-12

    one = probe[:1]
    assert np.max(np.abs(compiled.predict(one) - model.predict(one))) <= 1e-12

    batch = np.tile(probe, (int(np.ceil(50_000 / len(probe))), 1))[:50_000]
    native_batch = compiled.predict(batch)
    reference_batch = model.predict(batch)
    assert np.max(np.abs(native_batch - reference_batch)) <= 1e-12

    warm_predictions = []
    warm_start = time.perf_counter()
    for _ in range(3):
        warm_predictions.append(compiled.predict(probe))
    warm_seconds = time.perf_counter() - warm_start
    for prediction in warm_predictions:
        assert np.max(np.abs(prediction - reference)) <= 1e-12
    assert warm_seconds >= 0.0

    assert compiled.artifact_bytes > 0
    assert compiled.source_bytes > 0
    assert compiled.compile_seconds >= 0.0

    with pytest.raises(ValueError, match="features"):
        compiled.predict(probe[:, :-1])

    if model.selected_kind_ != "current_mean":
        source_a = render_fused_residual_native_source(model)
        source_b = render_fused_residual_native_source(model)
        assert source_a == source_b
        assert compiled.correction_source_path.read_text(encoding="utf-8") == source_a


def test_current_mean_native_contract(tmp_path):
    X, y = make_regression(
        n_samples=2500,
        n_features=12,
        n_informative=8,
        noise=12.0,
        random_state=11,
    )
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=11)
    model = FusedResidualCERMRegressor(random_state=11).fit(Xtr, ytr)
    assert model.selected_kind_ == "current_mean"
    _assert_native_contract(model, Xte, tmp_path, "current_mean")


def test_shared_fused_native_contract(tmp_path):
    X, y = make_friedman1(
        n_samples=2500,
        n_features=10,
        noise=1.0,
        random_state=11,
    )
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=11)
    model = FusedResidualCERMRegressor(random_state=11).fit(Xtr, ytr)
    assert model.selected_kind_ == "fused"
    _assert_native_contract(model, Xte, tmp_path, "shared_fused")


def test_raw_fused_native_contract(tmp_path):
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=47)
    model = FusedResidualCERMRegressor(random_state=47).fit(Xtr, ytr)
    assert model.selected_kind_ == "raw_fused"
    _assert_native_contract(model, Xte, tmp_path, "raw_fused")
