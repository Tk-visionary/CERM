from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import platform
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import joblib

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from ._internal.cerm_graph_optimizer_v2 import CERMGraphProgramV2

from .validation import dense_numeric_matrix, validate_dataframe_schema
from .portable import PortableTypedAdapter
from ._internal.cerm_typed_quotient_adapters_v4 import (
    TypedQuotientAdapter,
    estimate_typed_adapter_bytes,
    export_typed_adapter_ir,
)


def _numeric_matrix(
    X: pd.DataFrame | np.ndarray,
    input_columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    return dense_numeric_matrix(X, input_columns)


def _select_features(matrix: np.ndarray, feature_indices: tuple[int, ...] | None) -> np.ndarray:
    if feature_indices is None:
        return matrix
    indices = np.asarray(feature_indices, dtype=np.int64)
    if len(indices) == matrix.shape[1] and np.array_equal(indices, np.arange(matrix.shape[1])):
        return matrix
    return np.ascontiguousarray(matrix[:, indices])


def _adapter_nbytes(adapter: Any | None) -> int:
    if adapter is None:
        return 0
    if isinstance(adapter, PortableTypedAdapter):
        return int(adapter.nbytes)
    return int(estimate_typed_adapter_bytes(adapter))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "get_params") and callable(value.get_params):
        return {
            "estimator": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "params": _jsonable(value.get_params(deep=False)),
        }
    return value


@dataclass
class CompiledProgram:
    predict_positive: Any
    adapter: TypedQuotientAdapter | PortableTypedAdapter | None
    input_columns: tuple[str, ...] | None
    source_path: Path
    library_path: Path
    classes: tuple[Any, Any] | None = None
    feature_indices: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _prepare_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is None:
            return _numeric_matrix(X, self.input_columns)
        frame = validate_dataframe_schema(X, self.input_columns or ())
        transformed = self.adapter.transform(frame)
        return transformed.matrix if hasattr(transformed, "matrix") else transformed

    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is not None and self.feature_indices is not None and hasattr(
            self.adapter, "transform_columns"
        ):
            frame = validate_dataframe_schema(X, self.input_columns or ())
            transformed = self.adapter.transform_columns(frame, self.feature_indices)
            return transformed.matrix if hasattr(transformed, "matrix") else transformed
        return _select_features(self._prepare_input(X), self.feature_indices)

    def positive_probability_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        selected = _select_features(matrix, self.feature_indices)
        return np.asarray(self.predict_positive(selected), dtype=np.float64)

    def decision_function_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        p = np.clip(self.positive_probability_from_prepared(matrix), 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(self.predict_positive(self._matrix(X)), dtype=np.float64), 1e-12, 1.0 - 1e-12)
        return np.log(p / (1.0 - p))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p = np.asarray(self.predict_positive(self._matrix(X)), dtype=np.float64)
        return np.column_stack([1.0 - p, p])

    @property
    def model_bytes_estimate(self) -> int:
        adapter_bytes = _adapter_nbytes(self.adapter)
        library_bytes = self.library_path.stat().st_size if self.library_path.is_file() else 0
        return int(adapter_bytes + library_bytes)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        binary = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)
        if self.classes is None:
            return binary
        classes = np.asarray(self.classes)
        return classes[binary]

    @staticmethod
    def _file_record(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _save_legacy_v1(
        self,
        directory: Path,
        *,
        library: Path,
        source_record: dict[str, Any] | None,
    ) -> Path:
        state_path = directory / "runtime_state.joblib"
        joblib.dump(
            {
                "adapter": self.adapter,
                "input_columns": self.input_columns,
                "classes": self.classes,
                "feature_indices": self.feature_indices,
                "metadata": self.metadata,
            },
            state_path,
        )
        manifest = {
            "format": "cerm-compiled-program-v1",
            "library": self._file_record(library),
            "source": source_record,
            "state": self._file_record(state_path),
            "symbol": self.metadata.get("symbol", "cerm_graph_predict"),
            "backend": self.metadata.get("backend"),
            "dtype": self.metadata.get("dtype"),
            "machine": platform.machine(),
            "system": platform.system(),
            "python_implementation": platform.python_implementation(),
            "unsafe_pickle_state": True,
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        return manifest_path

    def save(
        self,
        directory: str | Path,
        *,
        include_source: bool = False,
        include_adapter: bool = True,
    ) -> Path:
        """Persist a reloadable, architecture-specific compiled package.

        Common numeric/categorical adapters use the pickle-free v2 format.
        Unsupported custom category objects or embedding adapters retain the
        experimental v1 joblib fallback when saved as standalone programs.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / self.library_path.name
        shutil.copy2(self.library_path, library)

        source_record = None
        if include_source and self.source_path.is_file():
            source = directory / self.source_path.name
            shutil.copy2(self.source_path, source)
            source_record = self._file_record(source)

        adapter_record = None
        adapter_source = None
        if self.adapter is not None and include_adapter:
            if isinstance(self.adapter, PortableTypedAdapter):
                adapter_npz, adapter_json = self.adapter.export(directory / "adapter")
                portable = bool(self.adapter.metadata.get("portable", False))
            else:
                adapter_npz, adapter_json = export_typed_adapter_ir(
                    self.adapter, directory / "adapter"
                )
                adapter_metadata = json.loads(adapter_json.read_text(encoding="utf-8"))
                portable = bool(adapter_metadata.get("portable", False))
            if not portable:
                adapter_npz.unlink(missing_ok=True)
                adapter_json.unlink(missing_ok=True)
                return self._save_legacy_v1(
                    directory,
                    library=library,
                    source_record=source_record,
                )
            adapter_record = {
                "format": "cerm-typed-adapter-ir-v1",
                "portable": True,
                "npz": self._file_record(adapter_npz),
                "json": self._file_record(adapter_json),
            }
        elif self.adapter is not None:
            adapter_source = "bundle-shared-adapter"

        runtime_path = directory / "runtime.json"
        runtime_payload = {
            "input_columns": _jsonable(self.input_columns),
            "classes": _jsonable(self.classes),
            "feature_indices": _jsonable(self.feature_indices),
            "metadata": _jsonable(self.metadata),
            "adapter_source": adapter_source,
        }
        try:
            runtime_path.write_text(
                json.dumps(runtime_payload, indent=2), encoding="utf-8"
            )
        except TypeError:
            if not include_adapter:
                raise ValueError(
                    "bundle head runtime labels are not JSON-portable"
                )
            return self._save_legacy_v1(
                directory,
                library=library,
                source_record=source_record,
            )
        manifest = {
            "format": "cerm-compiled-program-v2",
            "library": self._file_record(library),
            "source": source_record,
            "runtime": self._file_record(runtime_path),
            "adapter": adapter_record,
            "adapter_source": adapter_source,
            "symbol": self.metadata.get("symbol", "cerm_graph_predict"),
            "backend": self.metadata.get("backend"),
            "dtype": self.metadata.get("dtype"),
            "machine": platform.machine(),
            "system": platform.system(),
            "python_implementation": platform.python_implementation(),
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        adapter_override: PortableTypedAdapter | None = None,
    ) -> "CompiledProgram":
        """Load a package created by :meth:`save` without recompilation."""

        from .native_runtime import load_native_predictor

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        if not manifest_path.is_file():
            raise FileNotFoundError(f"compiled CERM manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_name = manifest.get("format")
        if format_name not in {"cerm-compiled-program-v1", "cerm-compiled-program-v2"}:
            raise ValueError(f"unsupported compiled CERM format: {format_name!r}")
        if manifest.get("machine") != platform.machine():
            raise RuntimeError("compiled CERM artifact targets a different machine architecture")
        if manifest.get("system") != platform.system():
            raise RuntimeError("compiled CERM artifact targets a different operating system")

        def verify(record: dict[str, Any]) -> Path:
            artifact = manifest_path.parent / record["file"]
            payload = artifact.read_bytes()
            if len(payload) != int(record["bytes"]):
                raise ValueError(f"byte-count mismatch for {artifact.name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {artifact.name}")
            return artifact

        library_path = verify(manifest["library"])
        source_record = manifest.get("source")
        source_path = verify(source_record) if source_record is not None else manifest_path.parent / "unavailable.cpp"
        symbol = manifest.get("symbol", "cerm_graph_predict")
        predictor = load_native_predictor(library_path, symbol=symbol)

        if format_name == "cerm-compiled-program-v1":
            state = joblib.load(verify(manifest["state"]))
            adapter = state.get("adapter")
            input_columns = state.get("input_columns")
            classes = state.get("classes")
            feature_indices = state.get("feature_indices")
            metadata = dict(state.get("metadata") or {})
        else:
            runtime = json.loads(verify(manifest["runtime"]).read_text(encoding="utf-8"))
            adapter_record = manifest.get("adapter")
            if adapter_record is not None:
                adapter = PortableTypedAdapter.load(
                    verify(adapter_record["json"]),
                    verify(adapter_record["npz"]),
                )
            elif manifest.get("adapter_source") == "bundle-shared-adapter":
                if adapter_override is None:
                    raise ValueError(
                        "compiled head requires the bundle shared adapter"
                    )
                adapter = adapter_override
            else:
                adapter = None
            input_columns = runtime.get("input_columns")
            classes = runtime.get("classes")
            feature_indices = runtime.get("feature_indices")
            metadata = dict(runtime.get("metadata") or {})

        metadata.update(
            {
                "reloaded": True,
                "symbol": symbol,
                "artifact_manifest": str(manifest_path),
            }
        )
        return cls(
            predict_positive=predictor,
            adapter=adapter,
            input_columns=tuple(input_columns) if input_columns is not None else None,
            source_path=source_path,
            library_path=library_path,
            classes=tuple(classes) if classes is not None else None,
            feature_indices=tuple(feature_indices) if feature_indices is not None else None,
            metadata=metadata,
        )


@dataclass
class OptimizedProgram:
    graph: CERMGraphProgramV2
    adapter: TypedQuotientAdapter | None
    input_columns: tuple[str, ...] | None
    objective: str
    classes: tuple[Any, Any] | None = None
    feature_indices: tuple[int, ...] | None = None

    def _prepare_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is None:
            return _numeric_matrix(X, self.input_columns)
        frame = validate_dataframe_schema(X, self.input_columns or ())
        return self.adapter.transform(frame).matrix

    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is not None and self.feature_indices is not None and hasattr(
            self.adapter, "transform_columns"
        ):
            frame = validate_dataframe_schema(X, self.input_columns or ())
            return self.adapter.transform_columns(frame, self.feature_indices).matrix
        return _select_features(self._prepare_input(X), self.feature_indices)

    def decision_function_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        return self.graph.decision_function(_select_features(matrix, self.feature_indices))

    def positive_probability_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        return self.graph.predict_proba(_select_features(matrix, self.feature_indices))[:, 1]

    def _core_projected_matrix(self, X):
        if (
            self.adapter is not None
            and self.feature_indices is not None
            and hasattr(self.adapter, "transform_columns")
        ):
            outer = np.asarray(self.feature_indices, dtype=np.int64)
            core = np.asarray(self.graph.feature_idx, dtype=np.int64)
            frame = validate_dataframe_schema(X, self.input_columns or ())
            return self.adapter.transform_columns(frame, outer[core]).matrix
        return None

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        projected = self._core_projected_matrix(X)
        if projected is not None and hasattr(self.graph, "decision_function_projected"):
            return self.graph.decision_function_projected(projected)
        return self.graph.decision_function(self._matrix(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        projected = self._core_projected_matrix(X)
        if projected is not None and hasattr(self.graph, "predict_proba_projected"):
            p = self.graph.predict_proba_projected(projected)[:, 1]
        else:
            p = self.graph.predict_proba(self._matrix(X))[:, 1]
        return np.column_stack([1.0 - p, p])

    def compile(
        self,
        prefix: str | Path,
        *,
        dtype: str = "float64",
    ) -> CompiledProgram:
        from ._internal.cerm_graph_int16_codegen import compile_graph_int16_native
        from ._internal.cerm_graph_native_codegen_v2 import compile_graph_native_v2
        from ._internal.cerm_graph_quantization import quantize_graph_int16

        if dtype not in {"float64", "float32", "int16"}:
            raise ValueError("dtype must be 'float64', 'float32', or 'int16'")
        if dtype == "int16":
            quantized = quantize_graph_int16(self.graph)
            predict, source, library = compile_graph_int16_native(quantized, prefix)
            metadata = {
                "backend": "graph-v2-int16",
                "objective": self.objective,
                "dtype": dtype,
                "logit_error_bound": quantized.logit_error_bound,
                "probability_error_bound": quantized.probability_error_bound,
                "compression_ratio": quantized.compression_ratio,
                "symbol": "cerm_graph_predict",
            }
        else:
            graph = self.graph if dtype == "float64" else self.graph.float_copy(dtype)
            project_adapter = self.adapter is not None and self.feature_indices is not None
            predict, source, library = compile_graph_native_v2(
                graph, prefix, projected_input=project_adapter
            )
            metadata = {
                "backend": "graph-v2",
                "objective": self.objective,
                "dtype": dtype,
                "symbol": "cerm_graph_predict",
                "projected_adapter_input": bool(project_adapter),
            }
        compiled_feature_indices = self.feature_indices
        if dtype != "int16" and self.adapter is not None and self.feature_indices is not None:
            outer = np.asarray(self.feature_indices, dtype=np.int64)
            compiled_feature_indices = tuple(
                int(index) for index in outer[np.asarray(graph.feature_idx, dtype=np.int64)]
            )
        return CompiledProgram(
            predict_positive=predict,
            adapter=self.adapter,
            input_columns=self.input_columns,
            source_path=Path(source),
            library_path=Path(library),
            classes=self.classes,
            feature_indices=compiled_feature_indices,
            metadata=metadata,
        )

    def export(self, prefix: str | Path) -> tuple[Path, Path]:
        return self.graph.export(prefix)


@dataclass
class SemanticProgram:
    model: Any
    adapter: TypedQuotientAdapter | None
    input_columns: tuple[str, ...] | None
    adapted_feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...] | None
    classes: tuple[Any, Any]
    library_version: str
    _optimized_cache: dict[str, OptimizedProgram] = field(default_factory=dict, repr=False)

    def _prepare_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is None:
            return _numeric_matrix(X, self.input_columns)
        frame = validate_dataframe_schema(X, self.input_columns or ())
        return self.adapter.transform(frame).matrix

    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.adapter is not None and self.feature_indices is not None and hasattr(
            self.adapter, "transform_columns"
        ):
            frame = validate_dataframe_schema(X, self.input_columns or ())
            return self.adapter.transform_columns(frame, self.feature_indices).matrix
        return _select_features(self._prepare_input(X), self.feature_indices)

    def decision_function_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.decision_function(_select_features(matrix, self.feature_indices))

    def positive_probability_from_prepared(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_select_features(matrix, self.feature_indices))[:, 1]

    def _core_projected_matrix(self, X):
        if (
            self.adapter is not None
            and self.feature_indices is not None
            and hasattr(self.adapter, "transform_columns")
        ):
            core_source = (
                getattr(self.model, "feature_idx_", None)
                if hasattr(self.model, "feature_idx_")
                else getattr(getattr(self.model, "base_", None), "feature_idx_", None)
            )
            if core_source is None:
                return None
            outer = np.asarray(self.feature_indices, dtype=np.int64)
            core = np.asarray(core_source, dtype=np.int64)
            adapted = outer[core]
            frame = validate_dataframe_schema(X, self.input_columns or ())
            return self.adapter.transform_columns(frame, adapted).matrix
        return None

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        projected = self._core_projected_matrix(X)
        if projected is not None and hasattr(self.model, "decision_function_projected"):
            return self.model.decision_function_projected(projected)
        return self.model.decision_function(self._matrix(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        projected = self._core_projected_matrix(X)
        if projected is not None and hasattr(self.model, "predict_proba_projected"):
            p = self.model.predict_proba_projected(projected)[:, 1]
        else:
            p = self.model.predict_proba(self._matrix(X))[:, 1]
        return np.column_stack([1.0 - p, p])

    @property
    def model_bytes_estimate(self) -> int:
        adapter_bytes = 0 if self.adapter is None else estimate_typed_adapter_bytes(self.adapter)
        return int(self.model.model_bytes_estimate_ + adapter_bytes)

    def optimize(self, target: str = "balanced") -> OptimizedProgram:
        if target not in {"memory", "balanced", "latency"}:
            raise ValueError("target must be memory, balanced, or latency")
        from ._internal.cerm_graph_optimizer_v2 import optimize_hybrid_graph_v2

        cached = self._optimized_cache.get(target)
        if cached is not None:
            return cached
        graph = optimize_hybrid_graph_v2(self.model, objective=target)
        optimized = OptimizedProgram(
            graph=graph,
            adapter=self.adapter,
            input_columns=self.input_columns,
            objective=target,
            classes=self.classes,
            feature_indices=self.feature_indices,
        )
        self._optimized_cache[target] = optimized
        return optimized

    def compile_native(self, prefix: str | Path) -> CompiledProgram:
        from ._internal.cerm_hybrid_native_codegen import compile_hybrid_native

        predict, source, library = compile_hybrid_native(self.model, prefix)
        return CompiledProgram(
            predict_positive=predict,
            adapter=self.adapter,
            input_columns=self.input_columns,
            source_path=Path(source),
            library_path=Path(library),
            classes=self.classes,
            feature_indices=self.feature_indices,
            metadata={
                "backend": "semantic-native",
                "symbol": "cerm_predict",
            },
        )

    def autotune(
        self,
        calibration_X: pd.DataFrame | np.ndarray,
        prefix: str | Path,
        *,
        target: str = "latency",
        benchmark_rows: int = 50_000,
    ) -> CompiledProgram:
        from ._internal.cerm_graph_autotune_v2 import autotune_hybrid_graph_v2

        matrix = self._matrix(calibration_X)
        selected = autotune_hybrid_graph_v2(
            self.model,
            matrix,
            prefix,
            target=target,
            benchmark_rows=benchmark_rows,
        )
        return CompiledProgram(
            predict_positive=selected.predict,
            adapter=self.adapter,
            input_columns=self.input_columns,
            source_path=Path(selected.source_path),
            library_path=Path(selected.library_path),
            classes=self.classes,
            feature_indices=self.feature_indices,
            metadata={
                "backend": "graph-autotune-v2",
                "symbol": "cerm_graph_predict",
                "target": target,
                "family": selected.family,
                "objective": selected.objective,
                "dtype": selected.dtype,
                "ns_per_row": selected.ns_per_row,
                "probability_error": selected.probability_error,
                "analytical_probability_bound": selected.analytical_probability_bound,
            },
        )

    def export(
        self,
        directory: str | Path,
        *,
        config: dict[str, Any] | None = None,
        include_adapter: bool = True,
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_npz, model_json = self.model.export_ir(directory / "model")
        adapter_files = None
        if self.adapter is not None and include_adapter:
            adapter_npz, adapter_json = export_typed_adapter_ir(self.adapter, directory / "adapter")
            adapter_files = {"npz": adapter_npz.name, "json": adapter_json.name}
        def file_record(path: Path) -> dict[str, Any]:
            payload = path.read_bytes()
            return {
                "file": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        model_files = {"npz": file_record(model_npz), "json": file_record(model_json)}
        if adapter_files is not None:
            adapter_files = {
                "npz": file_record(directory / adapter_files["npz"]),
                "json": file_record(directory / adapter_files["json"]),
            }
        if self.adapter is not None and not include_adapter:
            adapter_files = {"source": "bundle-shared-adapter"}
        manifest = {
            "format": "cerm-python-package-v2",
            "library_version": self.library_version,
            "config": _jsonable(config or {}),
            "classes": _jsonable(self.classes),
            "positive_class": _jsonable(self.classes[1]),
            "input_columns": _jsonable(self.input_columns),
            "adapted_feature_names": _jsonable(self.adapted_feature_names),
            "feature_indices": _jsonable(self.feature_indices),
            "model": model_files,
            "adapter": adapter_files,
            "model_bytes_estimate": self.model_bytes_estimate,
            "calibration": _jsonable(
                getattr(
                    getattr(self.model, "calibration_result_", None),
                    "to_dict",
                    lambda: None,
                )()
            ),
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def _row_count(X: pd.DataFrame | np.ndarray) -> int:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return int(len(X))


@dataclass
class ConstantBinaryProgram:
    """Portable binary head used for constant multilabel outputs."""

    probability: float
    classes: tuple[Any, Any] = (0, 1)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p = float(np.clip(self.probability, 1e-12, 1.0 - 1e-12))
        return np.full(_row_count(X), np.log(p / (1.0 - p)), dtype=np.float64)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p = np.full(_row_count(X), float(self.probability), dtype=np.float64)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        label = int(self.probability >= 0.5)
        return np.full(_row_count(X), self.classes[label])

    @property
    def model_bytes_estimate(self) -> int:
        return 8

    def optimize(self, target: str = "balanced") -> "ConstantBinaryProgram":
        return self

    def export(self, directory: str | Path, *, config: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": "cerm-constant-binary-v1",
            "task_type": "binary",
            "probability": float(self.probability),
            "classes": _jsonable(self.classes),
            "config": _jsonable(config or {}),
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


@dataclass
class ProgramBundle:
    """ONNX-like task-neutral composition of independent CERM heads.

    The bundle keeps preprocessing and semantic programs attached to each head.
    It provides one output contract for multiclass and multilabel tasks without
    changing the exact binary head implementation.
    """

    programs: tuple[Any, ...]
    task_type: str
    classes: tuple[Any, ...] | None = None
    output_names: tuple[str, ...] | None = None
    threshold: float | tuple[float, ...] = 0.5
    library_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_type not in {"multiclass", "multilabel"}:
            raise ValueError("task_type must be 'multiclass' or 'multilabel'")
        if not self.programs:
            raise ValueError("program bundle requires at least one output head")
        if self.task_type == "multiclass" and self.classes is None:
            raise ValueError("multiclass program bundle requires classes")
        threshold = np.asarray(self.threshold, dtype=np.float64)
        if threshold.ndim > 1 or np.any((threshold <= 0.0) | (threshold >= 1.0)):
            raise ValueError("threshold values must be in (0, 1)")
        if threshold.ndim == 1 and threshold.size not in {1, len(self.programs)}:
            raise ValueError("threshold vector length must match the number of heads")

    @staticmethod
    def _positive_probability(program: Any, X) -> np.ndarray:
        values = np.asarray(program.predict_proba(X), dtype=np.float64)
        if values.ndim == 1:
            return values
        if values.shape[1] == 1:
            return values[:, 0]
        return values[:, 1]

    def _shared_prepared_input(self, X):
        candidates = [
            program for program in self.programs
            if hasattr(program, "_prepare_input")
            and hasattr(program, "decision_function_from_prepared")
        ]
        if not candidates:
            return None
        first = candidates[0]
        adapter = getattr(first, "adapter", object())
        columns = getattr(first, "input_columns", object())
        if any(getattr(program, "adapter", None) is not adapter for program in candidates):
            return None
        if any(getattr(program, "input_columns", None) != columns for program in candidates):
            return None
        # Every non-constant head must support the prepared-input contract.
        for program in self.programs:
            if isinstance(program, ConstantBinaryProgram):
                continue
            if not any(program is candidate for candidate in candidates):
                return None
        return first._prepare_input(X)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        prepared = self._shared_prepared_input(X)
        columns = []
        for program in self.programs:
            if isinstance(program, ConstantBinaryProgram):
                value = np.asarray(program.decision_function(X), dtype=np.float64)
            elif prepared is not None:
                value = np.asarray(program.decision_function_from_prepared(prepared), dtype=np.float64)
            elif hasattr(program, "decision_function"):
                value = np.asarray(program.decision_function(X), dtype=np.float64)
            else:
                p = np.clip(self._positive_probability(program, X), 1e-12, 1 - 1e-12)
                value = np.log(p / (1.0 - p))
            columns.append(value.reshape(-1))
        return np.column_stack(columns)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.task_type == "multilabel":
            prepared = self._shared_prepared_input(X)
            columns = []
            for program in self.programs:
                if isinstance(program, ConstantBinaryProgram):
                    columns.append(self._positive_probability(program, X))
                elif prepared is not None and hasattr(program, "positive_probability_from_prepared"):
                    columns.append(program.positive_probability_from_prepared(prepared))
                else:
                    columns.append(self._positive_probability(program, X))
            positive = np.column_stack(columns)
            return np.clip(positive, 0.0, 1.0)

        # Multiclass coupling operates on one logit per head.  Branch before
        # evaluating positive probabilities so each binary program is invoked
        # only once per request, matching the single-pass execution contract
        # used by compiled bundles.
        logits = self.decision_function(X)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -50.0, 50.0))
        totals = exponential.sum(axis=1, keepdims=True)
        uniform = np.full_like(exponential, 1.0 / exponential.shape[1])
        return np.divide(exponential, totals, out=uniform, where=totals > 1e-15)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probability = self.predict_proba(X)
        if self.task_type == "multilabel":
            threshold = np.asarray(self.threshold, dtype=np.float64)
            return (probability >= threshold).astype(np.int32)
        classes = np.asarray(self.classes)
        return classes[np.argmax(probability, axis=1)]

    @property
    def model_bytes_estimate(self) -> int:
        total = int(sum(getattr(program, "model_bytes_estimate", 0) for program in self.programs))
        seen: set[int] = set()
        for program in self.programs:
            adapter = getattr(program, "adapter", None)
            if adapter is None:
                continue
            identity = id(adapter)
            adapter_bytes = _adapter_nbytes(adapter)
            if identity in seen:
                total -= adapter_bytes
            else:
                seen.add(identity)
        return max(int(total), 0)

    def optimize(self, target: str = "balanced") -> "ProgramBundle":
        optimized = tuple(
            program.optimize(target) if hasattr(program, "optimize") else program
            for program in self.programs
        )
        return ProgramBundle(
            programs=optimized,
            task_type=self.task_type,
            classes=self.classes,
            output_names=self.output_names,
            threshold=self.threshold,
            library_version=self.library_version,
            metadata={**self.metadata, "optimized_for": target},
        )

    def autotune(
        self,
        calibration_X: pd.DataFrame | np.ndarray,
        prefix: str | Path,
        *,
        target: str = "latency",
        benchmark_rows: int = 50_000,
    ) -> "CompiledProgramBundle":
        prefix = Path(prefix)
        compiled: list[Any] = []
        for index, program in enumerate(self.programs):
            if isinstance(program, ConstantBinaryProgram):
                compiled.append(program)
            elif hasattr(program, "autotune"):
                compiled.append(
                    program.autotune(
                        calibration_X,
                        prefix.with_name(f"{prefix.name}_head{index}"),
                        target=target,
                        benchmark_rows=benchmark_rows,
                    )
                )
            elif hasattr(program, "compile_native"):
                compiled.append(program.compile_native(prefix.with_name(f"{prefix.name}_head{index}")))
            else:
                raise NotImplementedError(f"head {index} does not support compilation")
        return CompiledProgramBundle(
            programs=tuple(compiled),
            task_type=self.task_type,
            classes=self.classes,
            output_names=self.output_names,
            threshold=self.threshold,
            metadata={**self.metadata, "backend": "independent-autotuned-heads", "target": target},
        )

    def compile_native(self, prefix: str | Path) -> "CompiledProgramBundle":
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        compiled: list[Any] = []
        for index, program in enumerate(self.programs):
            if isinstance(program, ConstantBinaryProgram):
                compiled.append(program)
            elif hasattr(program, "compile_native"):
                compiled.append(program.compile_native(prefix.with_name(f"{prefix.name}_head{index}")))
            elif hasattr(program, "compile"):
                compiled.append(program.compile(prefix.with_name(f"{prefix.name}_head{index}")))
            else:
                raise NotImplementedError(f"head {index} does not support native compilation")
        return CompiledProgramBundle(
            programs=tuple(compiled),
            task_type=self.task_type,
            classes=self.classes,
            output_names=self.output_names,
            threshold=self.threshold,
            metadata={**self.metadata, "backend": "independent-native-heads"},
        )

    def _shared_adapter(self):
        adapters = [
            getattr(program, "adapter", None)
            for program in self.programs
            if not isinstance(program, ConstantBinaryProgram)
        ]
        if not adapters or adapters[0] is None:
            return None
        first = adapters[0]
        return first if all(adapter is first for adapter in adapters) else None

    def export(self, directory: str | Path, *, config: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shared_adapter = self._shared_adapter()
        shared_adapter_record = None
        if shared_adapter is not None:
            adapter_npz, adapter_json = export_typed_adapter_ir(
                shared_adapter, directory / "shared_adapter"
            )
            def adapter_record(path: Path):
                payload = path.read_bytes()
                return {
                    "file": path.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            shared_adapter_record = {
                "npz": adapter_record(adapter_npz),
                "json": adapter_record(adapter_json),
            }
        head_records = []
        for index, program in enumerate(self.programs):
            head_dir = directory / f"head_{index:03d}"
            if hasattr(program, "export"):
                if shared_adapter is not None and isinstance(program, SemanticProgram):
                    manifest_path = program.export(
                        head_dir, config=config, include_adapter=False
                    )
                else:
                    manifest_path = program.export(head_dir, config=config)
            else:
                manifest_path = None
            if manifest_path is None:
                raise NotImplementedError(f"head {index} does not support portable export")
            payload = manifest_path.read_bytes()
            head_records.append(
                {
                    "index": index,
                    "directory": head_dir.name,
                    "manifest": manifest_path.name,
                    "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                    "uses_shared_adapter": bool(
                        shared_adapter is not None and isinstance(program, SemanticProgram)
                    ),
                }
            )
        manifest = {
            "format": (
                "cerm-program-bundle-v2"
                if shared_adapter_record is not None
                else "cerm-program-bundle-v1"
            ),
            "task_type": self.task_type,
            "n_outputs": len(self.programs),
            "classes": _jsonable(self.classes),
            "output_names": _jsonable(self.output_names),
            "threshold": _jsonable(self.threshold),
            "library_version": self.library_version,
            "metadata": _jsonable(self.metadata),
            "config": _jsonable(config or {}),
            "heads": head_records,
            "shared_adapter": shared_adapter_record,
            "model_bytes_estimate": self.model_bytes_estimate,
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


@dataclass
class CompiledProgramBundle:
    programs: tuple[Any, ...]
    task_type: str
    classes: tuple[Any, ...] | None = None
    output_names: tuple[str, ...] | None = None
    threshold: float | tuple[float, ...] = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def _bundle(self) -> ProgramBundle:
        return ProgramBundle(
            programs=self.programs,
            task_type=self.task_type,
            classes=self.classes,
            output_names=self.output_names,
            threshold=self.threshold,
            metadata=self.metadata,
        )

    def decision_function(self, X):
        return self._bundle().decision_function(X)

    def predict_proba(self, X):
        return self._bundle().predict_proba(X)

    def predict(self, X):
        return self._bundle().predict(X)

    @staticmethod
    def _record(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def save(self, directory: str | Path, *, include_source: bool = False) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        shared_adapter = self._bundle()._shared_adapter()
        shared_adapter_record = None
        if shared_adapter is not None:
            if isinstance(shared_adapter, PortableTypedAdapter):
                adapter_npz, adapter_json = shared_adapter.export(
                    directory / "shared_adapter"
                )
                portable = bool(shared_adapter.metadata.get("portable", False))
            else:
                adapter_npz, adapter_json = export_typed_adapter_ir(
                    shared_adapter, directory / "shared_adapter"
                )
                adapter_metadata = json.loads(
                    adapter_json.read_text(encoding="utf-8")
                )
                portable = bool(adapter_metadata.get("portable", False))
            if portable:
                shared_adapter_record = {
                    "format": "cerm-typed-adapter-ir-v1",
                    "portable": True,
                    "npz": self._record(adapter_npz),
                    "json": self._record(adapter_json),
                }
            else:
                adapter_npz.unlink(missing_ok=True)
                adapter_json.unlink(missing_ok=True)
                shared_adapter = None

        heads = []
        for index, program in enumerate(self.programs):
            head_dir = directory / f"head_{index:03d}"
            if isinstance(program, ConstantBinaryProgram):
                child_manifest = program.export(head_dir)
                kind = "constant"
            else:
                child_manifest = program.save(
                    head_dir,
                    include_source=include_source,
                    include_adapter=shared_adapter_record is None,
                )
                kind = "compiled"
            payload = child_manifest.read_bytes()
            heads.append(
                {
                    "index": index,
                    "directory": head_dir.name,
                    "manifest": child_manifest.name,
                    "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                    "kind": kind,
                    "uses_shared_adapter": bool(
                        shared_adapter_record is not None and kind == "compiled"
                    ),
                }
            )
        manifest = {
            "format": "cerm-compiled-bundle-v2",
            "task_type": self.task_type,
            "classes": _jsonable(self.classes),
            "output_names": _jsonable(self.output_names),
            "threshold": _jsonable(self.threshold),
            "metadata": _jsonable(self.metadata),
            "shared_adapter": shared_adapter_record,
            "heads": heads,
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "CompiledProgramBundle":
        from .io import verify_export

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = verify_export(manifest_path)
        if manifest.get("format") != "cerm-compiled-bundle-v2":
            raise ValueError(
                f"unsupported compiled bundle format: {manifest.get('format')!r}"
            )
        base = manifest_path.parent
        shared_adapter = None
        shared = manifest.get("shared_adapter")
        if shared is not None:
            shared_adapter = PortableTypedAdapter.load(
                base / shared["json"]["file"],
                base / shared["npz"]["file"],
            )
        programs: list[Any] = []
        for head in manifest["heads"]:
            head_dir = base / head["directory"]
            if head["kind"] == "constant":
                child = json.loads((head_dir / head["manifest"]).read_text(encoding="utf-8"))
                programs.append(
                    ConstantBinaryProgram(
                        probability=float(child["probability"]),
                        classes=tuple(child.get("classes", (0, 1))),
                    )
                )
            else:
                programs.append(
                    CompiledProgram.load(
                        head_dir,
                        adapter_override=(
                            shared_adapter if head.get("uses_shared_adapter") else None
                        ),
                    )
                )
        classes = manifest.get("classes")
        output_names = manifest.get("output_names")
        threshold = manifest.get("threshold", 0.5)
        if isinstance(threshold, list):
            threshold = tuple(float(value) for value in threshold)
        return cls(
            programs=tuple(programs),
            task_type=manifest["task_type"],
            classes=tuple(classes) if classes is not None else None,
            output_names=tuple(output_names) if output_names is not None else None,
            threshold=threshold,
            metadata={**dict(manifest.get("metadata") or {}), "reloaded": True},
        )

