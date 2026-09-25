from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_multilabel_classification
from sklearn.model_selection import train_test_split

from cerm import CERMClassifier, CERMMultiLabelClassifier, verify_export
from cerm.shared_multitask import (
    CompiledSharedFiniteStateProgram,
    SharedFiniteStateProgram,
)


def test_shared_multiclass_portable_and_native_roundtrip(tmp_path: Path):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    estimator = CERMClassifier(
        multiclass_strategy="shared",
        preset="balanced",
        random_state=7,
    ).fit(X_train, y_train)
    probability = estimator.predict_proba(X_test)
    assert probability.shape == (len(X_test), 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    assert estimator.fit_diagnostics_["shared_state_encoder"] is True

    export_dir = tmp_path / "portable"
    manifest = estimator.export(export_dir)
    verify_export(manifest)
    restored = SharedFiniteStateProgram.load(export_dir)
    np.testing.assert_allclose(restored.predict_proba(X_test), probability, atol=1e-14)
    with np.load(export_dir / "model.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)

    if shutil.which("g++") is None:
        pytest.skip("g++ is unavailable")
    compiled = estimator.compile_native(tmp_path / "shared_mc")
    np.testing.assert_allclose(compiled.predict_proba(X_test), probability, atol=1e-12)
    saved = tmp_path / "compiled"
    compiled.save(saved, include_source=True)
    verify_export(saved)
    reloaded = CompiledSharedFiniteStateProgram.load(saved)
    np.testing.assert_allclose(reloaded.predict_proba(X_test), probability, atol=1e-12)


def test_shared_multilabel_constant_output_and_native(tmp_path: Path):
    X, y = make_multilabel_classification(
        n_samples=360,
        n_features=14,
        n_classes=3,
        n_labels=2,
        allow_unlabeled=False,
        random_state=3,
    )
    y = np.column_stack([y, np.ones(len(y), dtype=np.int32)])
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    estimator = CERMMultiLabelClassifier(
        estimator=CERMClassifier(preset="balanced", random_state=11),
        representation_strategy="shared",
    ).fit(X_train, y_train)
    probability = estimator.predict_proba(X_test)
    assert probability.shape == (len(X_test), y.shape[1])
    np.testing.assert_array_equal(probability[:, -1], 1.0)
    assert estimator.fit_diagnostics_["shared_state_encoder"] is True

    export_dir = tmp_path / "portable_ml"
    estimator.export(export_dir)
    verify_export(export_dir)
    restored = SharedFiniteStateProgram.load(export_dir)
    np.testing.assert_allclose(restored.predict_proba(X_test), probability, atol=1e-14)

    if shutil.which("g++") is None:
        pytest.skip("g++ is unavailable")
    compiled = estimator.compile_native(tmp_path / "shared_ml")
    np.testing.assert_allclose(compiled.predict_proba(X_test), probability, atol=1e-12)


def test_shared_multiclass_rejects_target_aware_newton_encoder():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="encoder_kind='quantile'"):
        CERMClassifier(
            multiclass_strategy="shared",
            encoder_kind="newton",
        ).fit(X, y)


def test_multilabel_representation_strategy_validation():
    X, y = make_multilabel_classification(
        n_samples=80, n_features=6, n_classes=2, random_state=1
    )
    with pytest.raises(ValueError, match="representation_strategy"):
        CERMMultiLabelClassifier(representation_strategy="invalid").fit(X, y)


def test_shared_head_parallelism_preserves_predictions():
    X, y = load_iris(return_X_y=True)
    serial = CERMClassifier(
        multiclass_strategy="shared",
        preset="balanced",
        random_state=17,
        n_jobs=1,
    ).fit(X, y)
    parallel = CERMClassifier(
        multiclass_strategy="shared",
        preset="balanced",
        random_state=17,
        n_jobs=2,
    ).fit(X, y)
    np.testing.assert_allclose(
        parallel.predict_proba(X), serial.predict_proba(X), atol=0.0, rtol=0.0
    )
    assert parallel.model_.config_ == serial.model_.config_


def test_shared_multilabel_head_parallelism_preserves_predictions():
    X, y = make_multilabel_classification(
        n_samples=180, n_features=10, n_classes=4, random_state=9
    )
    serial = CERMMultiLabelClassifier(
        estimator=CERMClassifier(preset="balanced", random_state=23),
        representation_strategy="shared",
        n_jobs=1,
    ).fit(X, y)
    parallel = CERMMultiLabelClassifier(
        estimator=CERMClassifier(preset="balanced", random_state=23),
        representation_strategy="shared",
        n_jobs=2,
    ).fit(X, y)
    np.testing.assert_allclose(
        parallel.predict_proba(X), serial.predict_proba(X), atol=0.0, rtol=0.0
    )
    assert parallel.program_.model.config_ == serial.program_.model.config_


def test_shared_multinomial_objective_roundtrip(tmp_path):
    from sklearn.datasets import load_iris
    from cerm import CERMClassifier, SharedFiniteStateProgram

    X, y = load_iris(return_X_y=True)
    model = CERMClassifier(
        multiclass_strategy="shared",
        shared_multiclass_objective="multinomial",
        preset="balanced",
        max_interactions=8,
        random_state=13,
    ).fit(X, y)
    probability = model.predict_proba(X[:17])
    assert probability.shape == (17, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert model.fit_diagnostics_["multiclass_objective"] == "multinomial"

    directory = tmp_path / "shared-multinomial"
    model.program_.export(directory)
    restored = SharedFiniteStateProgram.load(directory)
    assert np.allclose(restored.predict_proba(X[:17]), probability, atol=1e-12)


def test_shared_multiclass_objective_validation():
    from cerm import CERMClassifier

    with pytest.raises(ValueError, match="shared_multiclass_objective"):
        CERMClassifier(shared_multiclass_objective="invalid").fit(
            np.arange(24, dtype=float).reshape(12, 2),
            np.asarray([0, 1] * 6),
        )


def test_shared_auto_multiclass_objective_selects_and_roundtrips(tmp_path):
    from sklearn.datasets import load_wine
    from cerm import CERMClassifier, SharedFiniteStateProgram

    X, y = load_wine(return_X_y=True)
    model = CERMClassifier(
        multiclass_strategy="shared",
        shared_multiclass_objective="auto",
        preset="balanced",
        max_interactions=8,
        random_state=17,
    ).fit(X, y)
    selected = model.fit_diagnostics_["multiclass_objective"]
    assert selected in {"ovr", "multinomial", "blend"}
    assert model.fit_diagnostics_["objective_guard_margin"] == pytest.approx(0.005)
    assert model.fit_diagnostics_["pre_guard_multiclass_objective"] in {
        "ovr",
        "multinomial",
        "blend",
    }
    if model.fit_diagnostics_["objective_guard_applied"]:
        assert selected == "ovr"
        assert model.fit_diagnostics_["pre_guard_multiclass_objective"] != "ovr"
        assert model.fit_diagnostics_["non_ovr_validation_advantage"] < 0.005
    assert model.model_.config_.multiclass_objective == selected
    weight = model.model_.config_.multinomial_weight
    assert 0.0 <= weight <= 1.0
    if selected == "blend":
        assert weight in {0.25, 0.5, 0.75}
    directory = tmp_path / "shared-auto"
    model.program_.export(directory)
    restored = SharedFiniteStateProgram.load(directory)
    assert restored.model.config_.multiclass_objective == selected
    assert restored.model.config_.multinomial_weight == weight
    assert np.allclose(restored.predict_proba(X[:11]), model.predict_proba(X[:11]), atol=1e-12)


def test_shared_multiclass_blend_head_is_single_linear_runtime():
    from cerm.shared_multitask import SharedFiniteStateModel

    class Head:
        pass

    ovr = Head()
    ovr.coef_ = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    ovr.intercept_ = np.asarray([0.2, -0.1, 0.0])
    multi = Head()
    multi.coef_ = np.asarray([[0.0, 2.0], [2.0, 0.0], [-2.0, -2.0]])
    multi.intercept_ = np.asarray([-0.2, 0.1, 0.0])

    blended = SharedFiniteStateModel._blend_multiclass_heads(ovr, multi, 0.25)
    np.testing.assert_allclose(blended.coef_, 0.75 * ovr.coef_ + 0.25 * multi.coef_)
    np.testing.assert_allclose(
        blended.intercept_, 0.75 * ovr.intercept_ + 0.25 * multi.intercept_
    )
    assert blended.coef_.shape == ovr.coef_.shape


def test_shared_auto_blend_selection_penalty_is_conservative():
    from cerm.shared_multitask import SharedFiniteStateModel

    assert SharedFiniteStateModel._BLEND_SELECTION_PENALTY == pytest.approx(0.005)
    assert SharedFiniteStateModel._NON_OVR_SELECTION_MARGIN == pytest.approx(0.005)


def test_shared_auto_non_ovr_guard_uses_validation_margin_only():
    from cerm.shared_multitask import SharedFiniteStateModel, SharedStructureConfig

    def row(loss, objective, weight, byte_count=100):
        return (
            loss,
            byte_count,
            SharedStructureConfig(
                n_pairs=2,
                n_fine_pairs=0,
                max_main_level=8,
                C=0.2,
                multiclass_objective=objective,
                multinomial_weight=weight,
            ),
        )

    insufficient = [
        row(0.5000, "ovr", 0.0),
        row(0.4951, "multinomial", 1.0),
    ]
    selected = SharedFiniteStateModel._select_validation_result(insufficient)
    guarded, best_ovr, advantage, applied = (
        SharedFiniteStateModel._guard_auto_validation_result(
            insufficient, selected
        )
    )
    assert applied is True
    assert guarded[2].multiclass_objective == "ovr"
    assert best_ovr == pytest.approx(0.5)
    assert advantage == pytest.approx(0.0049)

    sufficient = [
        row(0.5000, "ovr", 0.0),
        row(0.4950, "multinomial", 1.0),
    ]
    selected = SharedFiniteStateModel._select_validation_result(sufficient)
    guarded, _, advantage, applied = (
        SharedFiniteStateModel._guard_auto_validation_result(sufficient, selected)
    )
    assert applied is False
    assert guarded[2].multiclass_objective == "multinomial"
    assert advantage == pytest.approx(0.005)
