from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cerm_graph_optimizer import PairComponent, _state_index_dtype
from .cerm_graph_optimizer_v2 import CERMGraphProgramV2, PairProgramV2, SparseRowPair


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


@dataclass
class Int16Table:
    values: np.ndarray
    scale: float
    max_abs_error: float

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.int16)
        self.scale = float(self.scale)
        self.max_abs_error = float(self.max_abs_error)

    @property
    def bytes(self) -> int:
        return int(self.values.nbytes + 8)

    def dequantized(self) -> np.ndarray:
        return self.values.astype(np.float64) * self.scale


def quantize_int16(array: np.ndarray) -> Int16Table:
    array = np.asarray(array, dtype=np.float64)
    maximum = float(np.max(np.abs(array), initial=0.0))
    if maximum == 0.0:
        return Int16Table(np.zeros_like(array, dtype=np.int16), 1.0, 0.0)
    scale = maximum / 32767.0
    q = np.clip(np.rint(array / scale), -32767, 32767).astype(np.int16)
    error = float(np.max(np.abs(array - q.astype(np.float64) * scale), initial=0.0))
    return Int16Table(q, scale, error)


@dataclass
class Int16PairComponent:
    left_map: np.ndarray
    right_map: np.ndarray
    table: Int16Table
    sources: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.left_map = np.asarray(self.left_map, dtype=np.int64)
        self.right_map = np.asarray(self.right_map, dtype=np.int64)


@dataclass
class Int16SparseRowPair:
    row_slot: np.ndarray
    row_table: Int16Table
    transposed_from_source: bool = False

    def __post_init__(self):
        self.row_slot = np.asarray(self.row_slot, dtype=np.int16)


@dataclass
class Int16PairProgram:
    left: int
    right: int
    representation: str
    dense_table: Int16Table | None = None
    components: list[Int16PairComponent] = field(default_factory=list)
    sparse: Int16SparseRowPair | None = None
    source_count: int = 0

    @property
    def lookup_count(self) -> int:
        return len(self.components) if self.representation == "components" else 1


@dataclass
class CERMInt16GraphProgram:
    feature_idx: np.ndarray
    thresholds: list[np.ndarray]
    unary_tables: dict[int, Int16Table]
    pair_programs: list[Int16PairProgram]
    intercept: float
    direct_state_mask: np.ndarray | None
    direct_state_cardinalities: np.ndarray | None
    source_table_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.feature_idx = np.asarray(self.feature_idx, dtype=np.int64)
        self.thresholds = [np.asarray(t, dtype=np.float64) for t in self.thresholds]
        n = len(self.feature_idx)
        self.direct_state_mask = np.zeros(n, dtype=bool) if self.direct_state_mask is None else np.asarray(self.direct_state_mask, dtype=bool)
        self.direct_state_cardinalities = np.asarray([len(t)+1 for t in self.thresholds], dtype=np.int64) if self.direct_state_cardinalities is None else np.asarray(self.direct_state_cardinalities, dtype=np.int64)

    @property
    def fine_cardinalities(self) -> np.ndarray:
        threshold_cards = np.asarray([len(t) + 1 for t in self.thresholds], dtype=np.int64)
        return np.where(self.direct_state_mask, self.direct_state_cardinalities, threshold_cards)

    def states(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        state_dtype = _state_index_dtype(
            int(np.max(self.fine_cardinalities, initial=1)) - 1
        )
        out = np.empty((len(X), len(self.feature_idx)), dtype=state_dtype)
        for j, raw_j in enumerate(self.feature_idx):
            values = X[:, int(raw_j)]
            if self.direct_state_mask[j]:
                rounded = np.rint(values).astype(np.int64)
                card = int(self.direct_state_cardinalities[j])
                valid = np.isfinite(values) & (np.abs(values-rounded)<=1e-9) & (rounded>=0) & (rounded<card)
                out[:, j] = np.where(valid, rounded, 0).astype(state_dtype)
            else:
                out[:, j] = np.searchsorted(self.thresholds[j], values, side="right")
        return out

    def decision_function_from_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        score = np.full(len(states), self.intercept, dtype=np.float64)
        for feature, table in self.unary_tables.items():
            score += table.values[states[:, feature]].astype(np.float64) * table.scale
        for pair in self.pair_programs:
            left = states[:, pair.left]
            right = states[:, pair.right]
            if pair.representation == "dense":
                assert pair.dense_table is not None
                score += (
                    pair.dense_table.values[left, right].astype(np.float64)
                    * pair.dense_table.scale
                )
            elif pair.representation == "components":
                for component in pair.components:
                    score += (
                        component.table.values[
                            component.left_map[left], component.right_map[right]
                        ].astype(np.float64)
                        * component.table.scale
                    )
            else:
                assert pair.sparse is not None
                slot = pair.sparse.row_slot[left]
                active = slot >= 0
                if np.any(active):
                    score[active] += (
                        pair.sparse.row_table.values[slot[active], right[active]].astype(np.float64)
                        * pair.sparse.row_table.scale
                    )
        return score

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function_from_states(self.states(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    @property
    def table_bytes(self) -> int:
        total = sum(table.bytes for table in self.unary_tables.values())
        for pair in self.pair_programs:
            if pair.representation == "dense":
                total += pair.dense_table.bytes
            elif pair.representation == "components":
                total += sum(
                    component.table.bytes
                    + component.left_map.astype(
                        _state_index_dtype(int(np.max(component.left_map, initial=0)))
                    ).nbytes
                    + component.right_map.astype(
                        _state_index_dtype(int(np.max(component.right_map, initial=0)))
                    ).nbytes
                    for component in pair.components
                )
            else:
                total += pair.sparse.row_table.bytes + pair.sparse.row_slot.nbytes
        return int(total)

    @property
    def compression_ratio(self) -> float:
        return 0.0 if self.source_table_bytes == 0 else self.table_bytes / self.source_table_bytes

    @property
    def logit_error_bound(self) -> float:
        # Every unary table contributes once. Dense/sparse pair tables contribute
        # at most once; all component tables contribute once each.
        bound = sum(table.max_abs_error for table in self.unary_tables.values())
        for pair in self.pair_programs:
            if pair.representation == "dense":
                bound += pair.dense_table.max_abs_error
            elif pair.representation == "components":
                bound += sum(c.table.max_abs_error for c in pair.components)
            else:
                bound += pair.sparse.row_table.max_abs_error
        return float(bound)

    @property
    def probability_error_bound(self) -> float:
        # Sigmoid is globally 1/4-Lipschitz.
        return 0.25 * self.logit_error_bound

    def export(self, prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        arrays: dict[str, np.ndarray] = {
            "feature_idx": self.feature_idx.astype(np.int32),
            "thresholds": np.asarray(self.thresholds, dtype=object),
            "intercept": np.asarray([self.intercept], dtype=np.float64),
        }
        unary = []
        for i, (feature, table) in enumerate(sorted(self.unary_tables.items())):
            key = f"unary_{i}"
            arrays[key] = table.values
            unary.append(
                {
                    "feature": int(feature),
                    "array": key,
                    "scale": table.scale,
                    "max_abs_error": table.max_abs_error,
                }
            )
        pairs = []
        for i, pair in enumerate(self.pair_programs):
            entry = {
                "left": int(pair.left),
                "right": int(pair.right),
                "representation": pair.representation,
                "source_count": int(pair.source_count),
            }
            if pair.representation == "dense":
                key = f"pair_{i}_dense"
                arrays[key] = pair.dense_table.values
                entry.update(
                    {
                        "array": key,
                        "scale": pair.dense_table.scale,
                        "max_abs_error": pair.dense_table.max_abs_error,
                    }
                )
            elif pair.representation == "components":
                comps = []
                for q, component in enumerate(pair.components):
                    lm = f"pair_{i}_component_{q}_left_map"
                    rm = f"pair_{i}_component_{q}_right_map"
                    tb = f"pair_{i}_component_{q}_table"
                    map_dtype = np.result_type(
                        _state_index_dtype(int(np.max(component.left_map, initial=0))),
                        _state_index_dtype(int(np.max(component.right_map, initial=0))),
                    )
                    arrays[lm] = component.left_map.astype(map_dtype)
                    arrays[rm] = component.right_map.astype(map_dtype)
                    arrays[tb] = component.table.values
                    comps.append(
                        {
                            "left_map": lm,
                            "right_map": rm,
                            "array": tb,
                            "scale": component.table.scale,
                            "max_abs_error": component.table.max_abs_error,
                        }
                    )
                entry["components"] = comps
            else:
                slot = f"pair_{i}_row_slot"
                tb = f"pair_{i}_row_table"
                arrays[slot] = pair.sparse.row_slot
                arrays[tb] = pair.sparse.row_table.values
                entry.update(
                    {
                        "row_slot": slot,
                        "array": tb,
                        "scale": pair.sparse.row_table.scale,
                        "max_abs_error": pair.sparse.row_table.max_abs_error,
                    }
                )
            pairs.append(entry)
        np.savez_compressed(npz_path, **arrays)
        manifest = {
            "format": "cerm-onnx-like-graph-int16-ir-v1",
            "quantization": "symmetric-per-table-int16",
            "source_table_bytes": self.source_table_bytes,
            "quantized_table_bytes": self.table_bytes,
            "compression_ratio": self.compression_ratio,
            "logit_error_bound": self.logit_error_bound,
            "probability_error_bound": self.probability_error_bound,
            "unary": unary,
            "pairs": pairs,
            "metadata": self.metadata,
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path


def quantize_graph_int16(program: CERMGraphProgramV2) -> CERMInt16GraphProgram:
    pairs: list[Int16PairProgram] = []
    for pair in program.pair_programs:
        if pair.representation == "dense":
            pairs.append(
                Int16PairProgram(
                    pair.left,
                    pair.right,
                    "dense",
                    dense_table=quantize_int16(pair.dense_table),
                    source_count=pair.source_count,
                )
            )
        elif pair.representation == "components":
            pairs.append(
                Int16PairProgram(
                    pair.left,
                    pair.right,
                    "components",
                    components=[
                        Int16PairComponent(
                            component.left_map,
                            component.right_map,
                            quantize_int16(component.table),
                            list(component.sources),
                        )
                        for component in pair.components
                    ],
                    source_count=pair.source_count,
                )
            )
        else:
            assert pair.sparse is not None
            pairs.append(
                Int16PairProgram(
                    pair.left,
                    pair.right,
                    "sparse_rows",
                    sparse=Int16SparseRowPair(
                        pair.sparse.row_slot,
                        quantize_int16(pair.sparse.row_table),
                        pair.sparse.transposed_from_source,
                    ),
                    source_count=pair.source_count,
                )
            )
    source_bytes = int(
        sum(np.asarray(t).nbytes for t in program.unary_tables.values())
        + sum(pair.bytes for pair in program.pair_programs)
    )
    return CERMInt16GraphProgram(
        feature_idx=program.feature_idx.copy(),
        thresholds=[t.copy() for t in program.thresholds],
        unary_tables={k: quantize_int16(v) for k, v in program.unary_tables.items()},
        pair_programs=pairs,
        intercept=float(program.intercept),
        direct_state_mask=program.direct_state_mask.copy(),
        direct_state_cardinalities=program.direct_state_cardinalities.copy(),
        source_table_bytes=source_bytes,
        metadata={
            **program.metadata,
            "source_graph_ir": "cerm-onnx-like-graph-ir-v2",
        },
    )
