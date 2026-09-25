from __future__ import annotations

import copy

import numpy as np
import pytest

from cerm._internal.cerm_fused_regression import (
    FittedNestedRepresentation,
    FusedResidualCERMRegressor,
    FusedThresholdHead,
    RawMultiResolutionFusedResidual,
)
from cerm._internal.cerm_fused_regression_semantic import (
    FusedRegressionSemanticProgram,
    semantic_program_from_fused_regressor,
)
from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder
from cerm._internal.cerm_nested_representation import (
    build_nested_codes,
    prepare_nested_residual_maps,
)
from cerm._internal.cerm_state_design import ReferenceStateEncoder


def _data():
    rng = np.random.default_rng(20260819)
    X = rng.normal(size=(96, 4))
    y = (
        0.8 * X[:, 0]
        + np.sin(1.7 * X[:, 1])
        + 0.6 * X[:, 0] * X[:, 2]
        + 0.15 * rng.normal(size=len(X))
    )
    return X, y


def _base_shell(X, y):
    model = FusedResidualCERMRegressor(
        max_features=4,
        max_interaction_features=4,
        max_pairs=2,
        max_fine_pairs=1,
        max_bins=16,
        n_bins=8,
        raw_max_pairs=2,
        max_iter=60,
        tol=1e-4,
        random_state=20260819,
    )
    adapted = model._fit_input_adapter(X, y, None)
    model.baseline_ = model._make_baseline(adapted.shape[1]).fit(adapted, y)
    residual = y - model.baseline_.predict(adapted)
    model.lower_, model.upper_, model.thresholds_ = model._thresholds(
        residual, model.n_bins
    )
    model.representation_ = None
    model.head_ = None
    model.raw_correction_ = None
    model.selected_config_ = None
    model.design_dim_ = 0
    return model, residual


def _current_model(X, y):
    model, _ = _base_shell(X, y)
    model.selected_kind_ = "current_mean"
    model.residual_scale_ = 0.0
    return model


def _shared_model(X, y):
    model, residual = _base_shell(X, y)
    encoder = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(X)
    feature_idx = np.asarray([0, 1, 2], dtype=np.int64)
    pairs = ((0, 1), (1, 2))
    fine_pairs = ((0, 1),)
    levels = (4, 8, 16)
    maps = prepare_nested_residual_maps(
        encoder=encoder,
        feature_idx=feature_idx,
        levels=levels,
        pairs=pairs,
        fine_pairs=fine_pairs,
    )
    all_states = encoder.transform(X)
    states = {
        level: values[:, feature_idx]
        for level, values in all_states.items()
    }
    codes = build_nested_codes(
        states,
        encoder=encoder,
        feature_idx=feature_idx,
        levels=levels,
        max_bins=16,
        max_main_level=16,
        pairs=pairs,
        fine_pairs=fine_pairs,
        maps=maps,
        dtype=np.int32,
    )
    state_encoder = ReferenceStateEncoder()
    design = state_encoder.fit_transform(codes)
    labels = model._labels(residual, model.thresholds_)
    head = FusedThresholdHead(
        spline_knots=4, C=0.05, max_iter=60, tol=1e-4
    ).fit(
        design,
        labels,
        model.thresholds_,
        model.lower_,
        model.upper_,
    )
    model.selected_kind_ = "fused"
    model.residual_scale_ = 0.71
    model.head_ = head
    model.representation_ = FittedNestedRepresentation(
        encoder=encoder,
        feature_idx=feature_idx,
        pairs=pairs,
        fine_pairs=fine_pairs,
        levels=levels,
        max_bins=16,
        max_main_level=16,
        maps=maps,
        state_encoder=state_encoder,
    )
    model.design_dim_ = int(design.shape[1])
    return model


def _raw_model(X, y):
    model, residual = _base_shell(X, y)
    raw = RawMultiResolutionFusedResidual(
        n_bins=model.n_bins,
        resolutions=(4, 8, 16),
        pair_resolution=8,
        spline_knots=4,
        max_pairs=2,
        C=0.05,
        max_iter=60,
        tol=1e-4,
    ).fit(X, residual)
    model.selected_kind_ = "raw_fused"
    model.residual_scale_ = 1.0
    model.raw_correction_ = raw
    model.lower_ = float(raw.lower_)
    model.upper_ = float(raw.upper_)
    model.thresholds_ = np.asarray(raw.thresholds_, dtype=np.float64)
    model.design_dim_ = int(raw.state_dim_)
    return model


@pytest.fixture(scope="module")
def fitted_branches():
    X, y = _data()
    return {
        "current_mean": (_current_model(X, y), X),
        "fused": (_shared_model(X, y), X),
        "raw_fused": (_raw_model(X, y), X),
    }


@pytest.mark.parametrize("kind", ["current_mean", "fused", "raw_fused"])
def test_semantic_prediction_parity_for_single_and_batch(fitted_branches, kind):
    model, X = fitted_branches[kind]
    program = semantic_program_from_fused_regressor(model)
    for batch in (X[7:8], X[9:27]):
        np.testing.assert_allclose(
            program.predict(batch),
            model.predict(batch),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            program.predict_survival(batch),
            model.predict_survival(batch),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("kind", ["current_mean", "fused", "raw_fused"])
def test_semantic_export_roundtrip(fitted_branches, tmp_path, kind):
    model, X = fitted_branches[kind]
    program = semantic_program_from_fused_regressor(model)
    manifest = program.export(tmp_path / kind)
    loaded = FusedRegressionSemanticProgram.load(manifest)
    np.testing.assert_allclose(
        loaded.predict(X[:19]),
        model.predict(X[:19]),
        rtol=0.0,
        atol=1e-12,
    )


def test_semantic_export_is_deterministic(fitted_branches, tmp_path):
    model, _ = fitted_branches["fused"]
    program = semantic_program_from_fused_regressor(model)
    first = tmp_path / "first"
    second = tmp_path / "second"
    program.export(first)
    program.export(second)
    for name in ("model.json", "model.npz", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_semantic_ir_rejects_missing_or_corrupt_fields(fitted_branches):
    model, _ = fitted_branches["fused"]
    program = semantic_program_from_fused_regressor(model)

    missing_metadata = copy.deepcopy(program.metadata)
    missing_metadata.pop("selected_kind")
    with pytest.raises(ValueError, match="missing fused regression IR field"):
        FusedRegressionSemanticProgram(missing_metadata, program.arrays)

    missing_arrays = dict(program.arrays)
    missing_arrays.pop("shared__head__threshold_basis")
    with pytest.raises(ValueError, match="missing fused regression IR array"):
        FusedRegressionSemanticProgram(program.metadata, missing_arrays)

    corrupt_arrays = dict(program.arrays)
    corrupt_arrays["shared__lookup_offsets"] = np.asarray(
        [0, 1], dtype=np.int64
    )
    with pytest.raises(ValueError, match="shared lookup table count mismatch"):
        FusedRegressionSemanticProgram(program.metadata, corrupt_arrays)


def test_semantic_load_rejects_corrupt_artifact(fitted_branches, tmp_path):
    model, _ = fitted_branches["current_mean"]
    program = semantic_program_from_fused_regressor(model)
    directory = tmp_path / "corrupt"
    program.export(directory)
    model_npz = directory / "model.npz"
    model_npz.write_bytes(model_npz.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="byte-count mismatch|SHA-256 mismatch"):
        FusedRegressionSemanticProgram.load(directory)
