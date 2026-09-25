from __future__ import annotations

"""Typed outer contracts for fused regression semantic and native programs.

The fused regression statistical core is trained on an adapted dense numeric
matrix.  ``FusedResidualCERMRegressor`` itself may now own a
``RegressionTypedAdapter`` and therefore distinguishes the raw input schema
(``n_features_in_``) from the adapted core width (``n_adapted_features_``).

The semantic and native lowerings intentionally remain numeric-core programs.
This module is the single outer boundary that applies the existing regression
typed adapter before dispatching to those core programs.
"""

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cerm_fused_regression_native_codegen import (
    FusedRegressionNativeArtifact,
    compile_fused_regression_native,
)
from .cerm_fused_regression_semantic import (
    FusedRegressionSemanticProgram,
    semantic_program_from_fused_regressor,
)
from ..validation import dense_numeric_matrix, validate_dataframe_schema


def _record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "file": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _core_width(model) -> int:
    value = getattr(model, "n_adapted_features_", None)
    if value is None:
        value = getattr(model, "n_features_in_", None)
    if value is None:
        raise RuntimeError("fused regression model is missing fitted feature width")
    return int(value)


def _numeric_core_view(model):
    """Return a shallow fitted view whose input width is the adapted width.

    Worker lowerings were deliberately written against the numeric fused core.
    After typed-input maturation, the fitted estimator's ``n_features_in_`` is
    the raw schema width.  A shallow view lets the numeric lowerings retain their
    exact semantics without mutating the fitted estimator or duplicating the
    lowering logic.
    """

    view = copy.copy(model)
    view.n_features_in_ = _core_width(model)
    return view


def _prepare_matrix(
    X,
    *,
    adapter,
    input_columns: tuple[str, ...] | None,
    core_n_features: int,
) -> np.ndarray:
    if adapter is None:
        matrix = dense_numeric_matrix(X, input_columns)
    else:
        frame = validate_dataframe_schema(X, input_columns or ())
        transformed = adapter.transform(frame)
        matrix = transformed.matrix if hasattr(transformed, "matrix") else transformed
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != int(core_n_features):
        raise ValueError(
            "fused regression adapted input width mismatch: "
            f"got {matrix.shape[1] if matrix.ndim == 2 else 'non-2D'}, "
            f"expected {int(core_n_features)}"
        )
    return matrix


@dataclass
class PortableFusedRegressionProgram:
    """Portable typed wrapper around the pure numeric fused semantic program."""

    core: FusedRegressionSemanticProgram
    adapter: Any | None
    input_columns: tuple[str, ...] | None
    adapted_feature_names: tuple[str, ...]
    input_n_features: int
    core_n_features: int

    def _matrix(self, X) -> np.ndarray:
        return _prepare_matrix(
            X,
            adapter=self.adapter,
            input_columns=self.input_columns,
            core_n_features=self.core_n_features,
        )

    def predict(self, X) -> np.ndarray:
        return self.core.predict(self._matrix(X))

    decision_function = predict

    def predict_survival(self, X) -> np.ndarray:
        return self.core.predict_survival(self._matrix(X))

    @property
    def selected_kind(self) -> str:
        return self.core.selected_kind

    def export(self, directory: str | Path) -> Path:
        """Export through the existing regression package-v3 envelope."""

        from ..regression import PortableRegressionAdapter, export_regression_adapter_ir

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_npz = directory / "model.npz"
        model_json = directory / "model.json"
        ordered_arrays = {
            key: self.core.arrays[key] for key in sorted(self.core.arrays)
        }
        np.savez_compressed(model_npz, **ordered_arrays)
        model_json.write_text(
            json.dumps(self.core.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        adapter_record = None
        if self.adapter is not None:
            if isinstance(self.adapter, PortableRegressionAdapter):
                adapter_npz, adapter_json = self.adapter.export(directory / "adapter")
                portable = bool(self.adapter.metadata.get("portable", False))
            else:
                adapter_npz, adapter_json = export_regression_adapter_ir(
                    self.adapter, directory / "adapter"
                )
                adapter_metadata = json.loads(
                    adapter_json.read_text(encoding="utf-8")
                )
                portable = bool(adapter_metadata.get("portable", False))
            if not portable:
                raise ValueError(
                    "fused regression semantic export requires a portable regression adapter"
                )
            adapter_record = {
                "format": "cerm-regression-adapter-ir-v1",
                "portable": True,
                "npz": _record(adapter_npz),
                "json": _record(adapter_json),
            }

        manifest = {
            "format": "cerm-python-package-v3",
            "task_type": "regression",
            "n_outputs": 1,
            "input_columns": (
                None if self.input_columns is None else list(self.input_columns)
            ),
            "adapted_feature_names": list(self.adapted_feature_names),
            "feature_indices": None,
            "model": {
                "json": _record(model_json),
                "npz": _record(model_npz),
            },
            "adapter": adapter_record,
            "config": {},
            "metadata": {
                "engine": "fused-residual-v2",
                "selected_kind": self.selected_kind,
                "input_n_features": int(self.input_n_features),
                "core_n_features": int(self.core_n_features),
            },
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest_path

    @classmethod
    def load(cls, directory: str | Path) -> "PortableFusedRegressionProgram":
        from ..io import verify_export
        from ..regression import PortableRegressionAdapter

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = verify_export(manifest_path)
        if (
            manifest.get("format") != "cerm-python-package-v3"
            or manifest.get("task_type") != "regression"
        ):
            raise ValueError(
                "PortableFusedRegressionProgram requires a regression package-v3 export"
            )
        metadata = dict(manifest.get("metadata") or {})
        if metadata.get("engine") != "fused-residual-v2":
            raise ValueError("regression package is not a fused residual program")
        base = manifest_path.parent
        model_record = manifest.get("model")
        if not isinstance(model_record, dict):
            raise ValueError("fused regression export is missing model artifacts")
        model_metadata = json.loads(
            (base / model_record["json"]["file"]).read_text(encoding="utf-8")
        )
        with np.load(base / model_record["npz"]["file"], allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        core = FusedRegressionSemanticProgram(
            metadata=model_metadata,
            arrays=arrays,
        )

        adapter = None
        adapter_record = manifest.get("adapter")
        if adapter_record is not None:
            adapter = PortableRegressionAdapter.load(
                base / adapter_record["json"]["file"],
                base / adapter_record["npz"]["file"],
            )
        input_columns = manifest.get("input_columns")
        adapted_names = manifest.get("adapted_feature_names") or ()
        core_width = int(metadata.get("core_n_features", core.metadata["n_features_in"]))
        input_width = int(metadata.get("input_n_features", core_width))
        return cls(
            core=core,
            adapter=adapter,
            input_columns=(
                None if input_columns is None else tuple(str(x) for x in input_columns)
            ),
            adapted_feature_names=tuple(str(x) for x in adapted_names),
            input_n_features=input_width,
            core_n_features=core_width,
        )


def semantic_program_from_typed_fused_regressor(model) -> PortableFusedRegressionProgram:
    """Lower a fitted typed fused regressor without losing its input contract."""

    core_width = _core_width(model)
    core = semantic_program_from_fused_regressor(_numeric_core_view(model))
    input_width = int(getattr(model, "n_features_in_", core_width))
    names = tuple(str(x) for x in getattr(model, "adapted_feature_names_", ()))
    columns = getattr(model, "input_columns_", None)
    return PortableFusedRegressionProgram(
        core=core,
        adapter=getattr(model, "adapter_", None),
        input_columns=None if columns is None else tuple(columns),
        adapted_feature_names=names,
        input_n_features=input_width,
        core_n_features=core_width,
    )


@dataclass
class CompiledFusedRegressionProgram:
    """Typed wrapper around the numeric fused native artifact."""

    core: FusedRegressionNativeArtifact
    adapter: Any | None
    input_columns: tuple[str, ...] | None
    input_n_features: int
    core_n_features: int

    def _matrix(self, X) -> np.ndarray:
        return _prepare_matrix(
            X,
            adapter=self.adapter,
            input_columns=self.input_columns,
            core_n_features=self.core_n_features,
        )

    def predict(self, X) -> np.ndarray:
        return self.core.predict(self._matrix(X))

    decision_function = predict

    @property
    def selected_kind(self) -> str:
        return str(self.core.selected_kind)

    @property
    def artifact_bytes(self) -> int:
        return int(self.core.artifact_bytes)

    @property
    def source_bytes(self) -> int:
        return int(self.core.source_bytes)

    @property
    def compile_seconds(self) -> float:
        return float(self.core.compile_seconds)


def compile_typed_fused_regression_native(
    model,
    prefix: str | Path,
) -> CompiledFusedRegressionProgram:
    """Compile the adapted numeric core and retain the fitted raw-input adapter."""

    core_width = _core_width(model)
    core = compile_fused_regression_native(_numeric_core_view(model), prefix)
    input_width = int(getattr(model, "n_features_in_", core_width))
    columns = getattr(model, "input_columns_", None)
    return CompiledFusedRegressionProgram(
        core=core,
        adapter=getattr(model, "adapter_", None),
        input_columns=None if columns is None else tuple(columns),
        input_n_features=input_width,
        core_n_features=core_width,
    )
