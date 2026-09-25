from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .cerm_graph_int16_codegen import compile_graph_int16_native
from .cerm_graph_native_codegen import compile_graph_native
from .cerm_graph_native_codegen_v2 import compile_graph_native_v2
from .cerm_graph_optimizer import optimize_hybrid_graph, validate_program_exactness
from .cerm_graph_optimizer_v2 import optimize_hybrid_graph_v2, validate_v2_exactness
from .cerm_graph_quantization import quantize_graph_int16


@dataclass
class GraphVariantV2:
    family: str
    objective: str
    dtype: str
    ns_per_row: float
    library_bytes: int
    source_bytes: int
    probability_error: float
    analytical_probability_bound: float | None
    predict: Callable[[np.ndarray], np.ndarray]
    program: object
    source_path: Path
    library_path: Path


def _tile(X: np.ndarray, rows: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if len(X) >= rows:
        return np.ascontiguousarray(X[:rows])
    repeats = int(np.ceil(rows / len(X)))
    return np.ascontiguousarray(np.tile(X, (repeats, 1))[:rows])


def _bench(fn, X: np.ndarray, repeats: int = 11) -> float:
    fn(X[: min(256, len(X))])
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn(X)
        values.append((time.perf_counter_ns() - start) / len(X))
    return float(np.median(values))


def autotune_hybrid_graph_v2(
    model,
    calibration_X: np.ndarray,
    prefix: str | Path,
    *,
    target: str = "latency",
    objectives: tuple[str, ...] = ("memory", "balanced", "latency"),
    allow_float32: bool = True,
    allow_int16: bool = True,
    float32_tolerance: float = 2e-7,
    int16_tolerance: float = 5e-5,
    int16_analytical_bound: float = 3e-4,
    benchmark_rows: int = 50_000,
) -> GraphVariantV2:
    """Autotune v1/v2 exact, float32, and per-table int16 backends."""

    if target not in {"latency", "size"}:
        raise ValueError("target must be latency or size")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(calibration_X, dtype=np.float64)
    Xbench = _tile(X, benchmark_rows)
    reference = model.predict_proba(X)[:, 1]
    variants: list[GraphVariantV2] = []

    def add_variant(family, objective, dtype, program, compile_fn, bound=None):
        stem = prefix.parent / f"{prefix.name}_{family}_{objective}_{dtype}"
        predict, cpp, so = compile_fn(program, stem)
        error = float(np.max(np.abs(predict(X) - reference), initial=0.0))
        variants.append(
            GraphVariantV2(
                family=family,
                objective=objective,
                dtype=dtype,
                ns_per_row=_bench(predict, Xbench),
                library_bytes=so.stat().st_size,
                source_bytes=cpp.stat().st_size,
                probability_error=error,
                analytical_probability_bound=bound,
                predict=predict,
                program=program,
                source_path=cpp,
                library_path=so,
            )
        )

    for objective in objectives:
        # V1 remains a candidate because sparse branches and richer scheduling
        # do not dominate every tiny model.
        v1 = optimize_hybrid_graph(model, objective=objective)
        exact = validate_program_exactness(model, v1, X)
        if exact["max_probability_error"] > 1e-12:
            raise RuntimeError("v1 exact graph mismatch")
        add_variant("v1", objective, "float64", v1, compile_graph_native)
        if allow_float32:
            v1f = v1.quantized_copy("float32")
            error = float(np.max(np.abs(v1f.predict_proba(X)[:, 1] - reference)))
            if error <= float32_tolerance:
                add_variant("v1", objective, "float32", v1f, compile_graph_native)

        v2 = optimize_hybrid_graph_v2(model, objective=objective)
        exact2 = validate_v2_exactness(model, v2, X)
        if exact2["max_probability_error"] > 1e-12:
            raise RuntimeError("v2 exact graph mismatch")
        add_variant("v2", objective, "float64", v2, compile_graph_native_v2)
        if allow_float32:
            v2f = v2.float_copy("float32")
            error = float(np.max(np.abs(v2f.predict_proba(X)[:, 1] - reference)))
            if error <= float32_tolerance:
                add_variant("v2", objective, "float32", v2f, compile_graph_native_v2)
        if allow_int16:
            q = quantize_graph_int16(v2)
            error = float(np.max(np.abs(q.predict_proba(X)[:, 1] - reference)))
            if error <= int16_tolerance and q.probability_error_bound <= int16_analytical_bound:
                add_variant(
                    "v2",
                    objective,
                    "int16",
                    q,
                    compile_graph_int16_native,
                    q.probability_error_bound,
                )

    if target == "latency":
        selected = min(
            variants,
            key=lambda v: (v.ns_per_row, v.library_bytes, v.probability_error),
        )
    else:
        selected = min(
            variants,
            key=lambda v: (v.library_bytes, v.ns_per_row, v.probability_error),
        )

    manifest = {
        "format": "cerm-graph-autotune-v2",
        "target": target,
        "selected": {
            "family": selected.family,
            "objective": selected.objective,
            "dtype": selected.dtype,
            "ns_per_row": selected.ns_per_row,
            "library_bytes": selected.library_bytes,
            "probability_error": selected.probability_error,
            "analytical_probability_bound": selected.analytical_probability_bound,
            "source": selected.source_path.name,
            "library": selected.library_path.name,
        },
        "variants": [
            {
                "family": v.family,
                "objective": v.objective,
                "dtype": v.dtype,
                "ns_per_row": v.ns_per_row,
                "library_bytes": v.library_bytes,
                "source_bytes": v.source_bytes,
                "probability_error": v.probability_error,
                "analytical_probability_bound": v.analytical_probability_bound,
            }
            for v in variants
        ],
    }
    prefix.with_suffix(".autotune.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if hasattr(selected.program, "export"):
        selected.program.export(prefix.with_name(prefix.name + "_selected_ir"))
    return selected
