from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from sklearn.datasets import load_iris

from cerm import CERMClassifier, verify_export
from cerm.shared_multitask import (
    SharedFiniteStateModel,
    SharedFiniteStateProgram,
)


def _shared_model(*, strategy: str, budget: int = 12):
    X, y = load_iris(return_X_y=True)
    return SharedFiniteStateModel(
        task_type="multiclass",
        max_features=4,
        pair_feature_limit=4,
        max_bins=8,
        random_state=20260804,
        max_pairs=2,
        search_profile="practical",
        n_jobs=1,
        multiclass_objective="auto",
        representation_strategy=strategy,
        class_specific_budget=budget,
    ).fit(X, y)


def test_adaptive_shared_model_is_deterministic_and_class_masked():
    X, _ = load_iris(return_X_y=True)
    first = _shared_model(strategy="adaptive")
    second = _shared_model(strategy="adaptive")

    assert first.representation_strategy_ == "adaptive"
    assert first.class_delta_selected_budget_ in {0, 4, 8, 9, 12}
    assert first.class_delta_selected_budget_ <= first.class_specific_budget
    def descriptor_key(item):
        return (
            item.kind,
            item.raw0,
            item.level,
            item.state,
            item.target_class,
            item.candidate_score,
        )
    assert tuple(map(descriptor_key, first.extra_descriptors_)) == tuple(
        map(descriptor_key, second.extra_descriptors_)
    )
    np.testing.assert_allclose(
        first.predict_proba(X), second.predict_proba(X), atol=0.0, rtol=0.0
    )
    for descriptor, table in zip(first.extra_descriptors_, first.extra_lookup_):
        assert descriptor.kind == "class_state_delta"
        other = np.delete(table, descriptor.target_class, axis=1)
        assert np.count_nonzero(other) == 0


def test_adaptive_budget_zero_is_exact_noop_with_versioned_ir(tmp_path: Path):
    X, _ = load_iris(return_X_y=True)
    baseline = _shared_model(strategy="baseline")
    adaptive = _shared_model(strategy="adaptive", budget=0)
    np.testing.assert_allclose(
        adaptive.predict_proba(X), baseline.predict_proba(X), atol=0.0, rtol=0.0
    )
    assert adaptive.representation_strategy_ == "adaptive"
    assert adaptive.class_delta_selected_budget_ == 0

    _, baseline_json = baseline.export_ir(tmp_path / "baseline")
    _, adaptive_json = adaptive.export_ir(tmp_path / "adaptive")
    assert json.loads(baseline_json.read_text())["format"] == "cerm-shared-finite-state-v2"
    assert json.loads(adaptive_json.read_text())["format"] == "cerm-shared-finite-state-v3"


def test_adaptive_public_program_portable_and_native_roundtrip(tmp_path: Path):
    X, y = load_iris(return_X_y=True)
    estimator = CERMClassifier(
        multiclass_strategy="shared",
        shared_multiclass_objective="auto",
        representation_strategy="adaptive",
        class_specific_budget=12,
        preset="balanced",
        max_interactions=2,
        random_state=20260804,
    ).fit(X, y)
    expected = estimator.predict_proba(X)
    assert estimator.fit_diagnostics_["representation_strategy"] == "adaptive"
    assert estimator.fit_diagnostics_["class_delta_selected_budget"] <= 12

    directory = tmp_path / "portable"
    manifest = estimator.program_.export(directory)
    verify_export(manifest)
    assert json.loads(manifest.read_text())["format"] == "cerm-shared-program-v2"
    assert json.loads((directory / "model.json").read_text())["format"] == "cerm-shared-finite-state-v3"
    restored = SharedFiniteStateProgram.load(directory)
    np.testing.assert_allclose(
        restored.predict_proba(X), expected, atol=0.0, rtol=0.0
    )
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)

    if shutil.which("g++") is None:
        pytest.skip("g++ is unavailable")
    compiled = restored.compile_native(tmp_path / "adaptive_native")
    np.testing.assert_allclose(
        compiled.predict_proba(X), expected, atol=1e-14, rtol=0.0
    )


def test_adaptive_ir_rejects_unmasked_class_coefficient(tmp_path: Path):
    model = _shared_model(strategy="adaptive")
    if not model.extra_descriptors_:
        pytest.skip("deterministic fixture selected the no-op prefix")
    npz_path, json_path = model.export_ir(tmp_path / "model")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    target = int(arrays["extra_target_class"][0])
    outputs = int(arrays["extra_lookup_outputs"][0])
    other = (target + 1) % outputs
    arrays["extra_lookup_values"][outputs + other] = 1.0
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(ValueError, match="class mask"):
        SharedFiniteStateModel.load_ir(json_path, npz_path)


def test_adaptive_requires_shared_multiclass_strategy():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="requires multiclass_strategy='shared'"):
        CERMClassifier(representation_strategy="adaptive").fit(X, y)
