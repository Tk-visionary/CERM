from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _array_key(array: np.ndarray) -> tuple[str, tuple[int, ...], str]:
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256(a.view(np.uint8)).hexdigest()
    return (a.dtype.str, tuple(a.shape), digest)


def _is_zero(array: np.ndarray, atol: float = 0.0) -> bool:
    if atol <= 0.0:
        return bool(np.all(array == 0.0))
    return bool(np.max(np.abs(array), initial=0.0) <= atol)


def _state_index_dtype(max_state: int) -> np.dtype:
    """Return a lossless unsigned dtype for a finite-state label."""
    max_state = int(max_state)
    if max_state < 0:
        raise ValueError("finite-state labels must be non-negative")
    if max_state <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_state <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_state <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError("finite-state labels above uint32 are not supported by graph backends")


def _state_index_ctype(max_state: int) -> str:
    return {
        np.dtype(np.uint8): "uint8_t",
        np.dtype(np.uint16): "uint16_t",
        np.dtype(np.uint32): "uint32_t",
    }[_state_index_dtype(max_state)]


@dataclass
class PairComponent:
    """One exact pair term evaluated through two state maps and one LUT."""

    left_map: np.ndarray
    right_map: np.ndarray
    table: np.ndarray
    sources: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.left_map = np.asarray(self.left_map, dtype=np.int64)
        self.right_map = np.asarray(self.right_map, dtype=np.int64)
        self.table = np.asarray(self.table, dtype=np.float64)
        if self.table.ndim != 2:
            raise ValueError("pair component table must be two-dimensional")
        if self.left_map.ndim != 1 or self.right_map.ndim != 1:
            raise ValueError("pair component maps must be one-dimensional")
        if len(self.left_map) and int(self.left_map.max()) >= self.table.shape[0]:
            raise ValueError("left map exceeds table cardinality")
        if len(self.right_map) and int(self.right_map.max()) >= self.table.shape[1]:
            raise ValueError("right map exceeds table cardinality")

    @property
    def bytes(self) -> int:
        return int(self.table.nbytes + self.left_map.nbytes + self.right_map.nbytes)


@dataclass
class PairProgram:
    left: int
    right: int
    representation: str  # dense | components
    dense_table: np.ndarray | None = None
    components: list[PairComponent] = field(default_factory=list)
    source_count: int = 0
    factorized: bool = False

    @property
    def lookup_ops(self) -> int:
        return 1 if self.representation == "dense" else len(self.components)


@dataclass
class GraphOptimizationStats:
    objective: str
    raw_nodes: int
    optimized_nodes: int
    raw_lookup_ops: int
    optimized_lookup_ops: int
    raw_map_ops: int
    optimized_map_ops: int
    raw_table_bytes: int
    optimized_table_bytes: int
    optimized_unique_initializer_bytes: int
    unary_fusions: int
    pair_component_fusions: int
    dense_pair_fusions: int
    additive_pair_factorizations: int
    zero_terms_removed: int
    features_pruned: int
    float_dtype: str

    @property
    def node_reduction(self) -> float:
        return 0.0 if self.raw_nodes == 0 else 1.0 - self.optimized_nodes / self.raw_nodes

    @property
    def lookup_reduction(self) -> float:
        return (
            0.0
            if self.raw_lookup_ops == 0
            else 1.0 - self.optimized_lookup_ops / self.raw_lookup_ops
        )


@dataclass
class CERMGraphProgram:
    """ONNX-like normalized finite-state program.

    The program has one quantizer per selected raw feature, one fused unary LUT
    per used feature, and a set of exact pair programs. Pair programs may retain
    multiple small mapped LUTs or become one dense fine-state LUT.
    """

    feature_idx: np.ndarray
    thresholds: list[np.ndarray]
    unary_tables: dict[int, np.ndarray]
    pair_programs: list[PairProgram]
    intercept: float
    direct_state_mask: np.ndarray | None = None
    direct_state_cardinalities: np.ndarray | None = None
    stats: GraphOptimizationStats | None = None
    table_dtype: str = "float64"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.feature_idx = np.asarray(self.feature_idx, dtype=np.int64)
        self.thresholds = [np.asarray(x, dtype=np.float64) for x in self.thresholds]
        if len(self.feature_idx) != len(self.thresholds):
            raise ValueError("feature and threshold counts differ")
        n = len(self.feature_idx)
        if self.direct_state_mask is None:
            self.direct_state_mask = np.zeros(n, dtype=bool)
        else:
            self.direct_state_mask = np.asarray(self.direct_state_mask, dtype=bool)
        if self.direct_state_cardinalities is None:
            self.direct_state_cardinalities = np.asarray([len(t) + 1 for t in self.thresholds], dtype=np.int64)
        else:
            self.direct_state_cardinalities = np.asarray(self.direct_state_cardinalities, dtype=np.int64)
        if len(self.direct_state_mask) != n or len(self.direct_state_cardinalities) != n:
            raise ValueError("direct-state metadata width mismatch")

    @property
    def fine_cardinalities(self) -> np.ndarray:
        threshold_cards = np.asarray([len(t) + 1 for t in self.thresholds], dtype=np.int64)
        return np.where(self.direct_state_mask, self.direct_state_cardinalities, threshold_cards)

    def states(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if len(self.feature_idx) and int(self.feature_idx.max()) >= X.shape[1]:
            raise ValueError("X has fewer columns than the graph expects")
        state_dtype = _state_index_dtype(
            int(np.max(self.fine_cardinalities, initial=1)) - 1
        )
        out = np.empty((len(X), len(self.feature_idx)), dtype=state_dtype)
        for local_j, raw_j in enumerate(self.feature_idx):
            values = X[:, int(raw_j)]
            if self.direct_state_mask[local_j]:
                rounded = np.rint(values).astype(np.int64)
                card = int(self.direct_state_cardinalities[local_j])
                valid = np.isfinite(values) & (np.abs(values - rounded) <= 1e-9) & (rounded >= 0) & (rounded < card)
                out[:, local_j] = np.where(valid, rounded, 0).astype(state_dtype)
            else:
                out[:, local_j] = np.searchsorted(
                    self.thresholds[local_j], values, side="right"
                )
        return out

    def decision_function_from_states(self, fine: np.ndarray) -> np.ndarray:
        fine = np.asarray(fine)
        if fine.ndim != 2 or fine.shape[1] != len(self.feature_idx):
            raise ValueError("fine-state shape mismatch")
        score = np.full(len(fine), self.intercept, dtype=np.float64)
        for feature, table in self.unary_tables.items():
            score += table[fine[:, feature]]
        for pair in self.pair_programs:
            left = fine[:, pair.left]
            right = fine[:, pair.right]
            if pair.representation == "dense":
                assert pair.dense_table is not None
                score += pair.dense_table[left, right]
            else:
                for component in pair.components:
                    score += component.table[
                        component.left_map[left], component.right_map[right]
                    ]
        return score

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function_from_states(self.states(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def quantized_copy(self, dtype: str = "float32") -> "CERMGraphProgram":
        if dtype not in {"float32", "float64"}:
            raise ValueError("only float32 and float64 are supported")
        dt = np.dtype(dtype)
        unary = {k: np.asarray(v, dtype=dt) for k, v in self.unary_tables.items()}
        pairs: list[PairProgram] = []
        for pair in self.pair_programs:
            if pair.representation == "dense":
                pairs.append(
                    PairProgram(
                        pair.left,
                        pair.right,
                        "dense",
                        dense_table=np.asarray(pair.dense_table, dtype=dt),
                        source_count=pair.source_count,
                        factorized=pair.factorized,
                    )
                )
            else:
                pairs.append(
                    PairProgram(
                        pair.left,
                        pair.right,
                        "components",
                        components=[
                            PairComponent(
                                c.left_map,
                                c.right_map,
                                np.asarray(c.table, dtype=dt),
                                list(c.sources),
                            )
                            for c in pair.components
                        ],
                        source_count=pair.source_count,
                        factorized=pair.factorized,
                    )
                )
        return CERMGraphProgram(
            feature_idx=self.feature_idx.copy(),
            thresholds=[x.copy() for x in self.thresholds],
            unary_tables=unary,
            pair_programs=pairs,
            intercept=float(np.asarray(self.intercept, dtype=dt)),
            direct_state_mask=self.direct_state_mask.copy(),
            direct_state_cardinalities=self.direct_state_cardinalities.copy(),
            stats=self.stats,
            table_dtype=dtype,
            metadata=dict(self.metadata),
        )

    def export(self, prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        arrays: dict[str, np.ndarray] = {
            "feature_idx": self.feature_idx.astype(np.int32),
            "thresholds": np.asarray(self.thresholds, dtype=object),
            "direct_state_mask": self.direct_state_mask.astype(np.uint8),
            "direct_state_cardinalities": self.direct_state_cardinalities.astype(np.int32),
            "intercept": np.asarray([self.intercept], dtype=np.float64),
        }
        unary_manifest = []
        for i, (feature, table) in enumerate(sorted(self.unary_tables.items())):
            name = f"unary_{i}"
            arrays[name] = np.asarray(table)
            unary_manifest.append({"feature": int(feature), "array": name})
        pair_manifest = []
        for i, pair in enumerate(self.pair_programs):
            entry: dict[str, Any] = {
                "left": int(pair.left),
                "right": int(pair.right),
                "representation": pair.representation,
                "source_count": int(pair.source_count),
                "factorized": bool(pair.factorized),
            }
            if pair.representation == "dense":
                name = f"pair_{i}_dense"
                arrays[name] = np.asarray(pair.dense_table)
                entry["array"] = name
            else:
                comps = []
                for q, comp in enumerate(pair.components):
                    lm = f"pair_{i}_component_{q}_left_map"
                    rm = f"pair_{i}_component_{q}_right_map"
                    tb = f"pair_{i}_component_{q}_table"
                    map_dtype = np.result_type(
                        _state_index_dtype(int(np.max(comp.left_map, initial=0))),
                        _state_index_dtype(int(np.max(comp.right_map, initial=0))),
                    )
                    arrays[lm] = comp.left_map.astype(map_dtype)
                    arrays[rm] = comp.right_map.astype(map_dtype)
                    arrays[tb] = np.asarray(comp.table)
                    comps.append(
                        {
                            "left_map": lm,
                            "right_map": rm,
                            "table": tb,
                            "sources": list(comp.sources),
                        }
                    )
                entry["components"] = comps
            pair_manifest.append(entry)
        np.savez_compressed(npz_path, **arrays)
        manifest = {
            "format": "cerm-onnx-like-graph-ir-v1",
            "input": "dense_numeric_matrix",
            "state_index_dtype": _state_index_dtype(
                int(np.max(self.fine_cardinalities, initial=1)) - 1
            ).name,
            "table_dtype": self.table_dtype,
            "feature_count": int(len(self.feature_idx)),
            "unary": unary_manifest,
            "pairs": pair_manifest,
            "metadata": self.metadata,
            "stats": None if self.stats is None else self.stats.__dict__,
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path


@dataclass
class _RawGraph:
    feature_idx: np.ndarray
    thresholds: list[np.ndarray]
    direct_state_mask: np.ndarray
    direct_state_cardinalities: np.ndarray
    unary_components: dict[int, list[tuple[np.ndarray, np.ndarray, str]]]
    pair_components: dict[tuple[int, int], list[PairComponent]]
    intercept: float
    raw_nodes: int
    raw_lookup_ops: int
    raw_map_ops: int
    raw_table_bytes: int
    metadata: dict[str, Any]


def _canonical_pair(
    left: int,
    right: int,
    left_map: np.ndarray,
    right_map: np.ndarray,
    table: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]:
    if left <= right:
        return (left, right), left_map, right_map, table
    return (right, left), right_map, left_map, table.T






def _safe_take(table: np.ndarray, indices: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    out = np.zeros(indices.shape, dtype=np.float64)
    valid = (indices >= 0) & (indices < len(table))
    out[valid] = table[indices[valid]]
    return out

def _compile_lookup(encoder, coef: np.ndarray) -> list[np.ndarray]:
    coef = np.asarray(coef, dtype=np.float64)
    out: list[np.ndarray] = []
    position = 0
    drop = encoder.drop_idx_
    for column, categories0 in enumerate(encoder.categories_):
        categories = np.asarray(categories0, dtype=np.int64)
        size = int(categories.max()) + 1 if len(categories) else 1
        table = np.zeros(size, dtype=np.float64)
        dropped = None if drop is None else drop[column]
        for category_position, category in enumerate(categories):
            if dropped is not None and category_position == int(dropped):
                continue
            table[int(category)] = coef[position]
            position += 1
        out.append(table)
    if position != len(coef):
        raise RuntimeError(f"coefficient layout mismatch: {position} != {len(coef)}")
    return out

def _reshape_lookup(table: np.ndarray, rows: int, cols: int) -> np.ndarray:
    table = np.asarray(table, dtype=np.float64)
    expected = rows * cols
    if len(table) < expected:
        padded = np.zeros(expected, dtype=np.float64)
        padded[: len(table)] = table
        table = padded
    return table[:expected].reshape(rows, cols)


def build_raw_graph(model) -> _RawGraph:
    """Lower a fitted HybridQuotientBlockCERM into a semantic DAG-like form."""

    base = model.base_
    feature_idx = np.asarray(base.feature_idx_, dtype=np.int64)
    thresholds = [
        np.asarray(base.encoder_.thresholds_[int(raw_j)], dtype=np.float64)
        for raw_j in feature_idx
    ]
    direct_mask_all = np.asarray(getattr(base.encoder_, "direct_state_mask_", np.zeros(base.encoder_.n_features_in_, dtype=bool)), dtype=bool)
    direct_cards_all = np.asarray(getattr(base.encoder_, "direct_state_cardinalities_", np.zeros(base.encoder_.n_features_in_, dtype=np.int64)), dtype=np.int64)
    direct_state_mask = direct_mask_all[feature_idx]
    threshold_cards = np.asarray([len(x) + 1 for x in thresholds], dtype=np.int64)
    direct_state_cardinalities = np.where(direct_state_mask, direct_cards_all[feature_idx], threshold_cards)
    fine_cards = np.where(direct_state_mask, direct_state_cardinalities, threshold_cards)
    levels = tuple(int(level) for level in base.encoder_.levels)
    maps = {
        level: [
            np.asarray(base.encoder_.maps_[level][int(raw_j)], dtype=np.int64)
            for raw_j in feature_idx
        ]
        for level in levels
    }
    coarse_level = levels[0]
    block_level = int(getattr(model, "block_level", levels[-1]))

    unary: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {
        j: [] for j in range(len(feature_idx))
    }
    pairs: dict[tuple[int, int], list[PairComponent]] = {}
    base_lookup = _compile_lookup(base.oh_, model.base_coef_)
    lookup_pos = 0
    raw_lookup_ops = 0
    raw_map_refs: set[tuple[int, int]] = set()
    raw_table_bytes = 0

    def add_unary(feature: int, state_map: np.ndarray, table: np.ndarray, source: str):
        nonlocal raw_lookup_ops, raw_table_bytes
        table = np.asarray(table, dtype=np.float64)
        unary[feature].append((np.asarray(state_map, dtype=np.int64), table, source))
        raw_lookup_ops += 1
        raw_table_bytes += table.nbytes

    def add_pair(
        left: int,
        right: int,
        left_map: np.ndarray,
        right_map: np.ndarray,
        table: np.ndarray,
        source: str,
    ):
        nonlocal raw_lookup_ops, raw_table_bytes
        scope, lm, rm, tb = _canonical_pair(
            left,
            right,
            np.asarray(left_map, dtype=np.int64),
            np.asarray(right_map, dtype=np.int64),
            np.asarray(table, dtype=np.float64),
        )
        pairs.setdefault(scope, []).append(PairComponent(lm, rm, tb, [source]))
        raw_lookup_ops += 1
        raw_table_bytes += tb.nbytes

    # Main effects.
    main_levels = [
        level for level in levels if level <= int(base.config_.max_main_level)
    ]
    for j in range(len(feature_idx)):
        for position, level in enumerate(main_levels):
            if position == 0:
                state_map = maps[level][j]
                source = f"main{level}:{j}"
            else:
                state_map = base.main_residuals_[level][j][maps[level][j]]
                source = f"main{level}res:{j}"
            add_unary(j, state_map, base_lookup[lookup_pos], source)
            raw_map_refs.add((j, level))
            lookup_pos += 1

    # Hierarchical pair effects.
    fine_pairs = set(base.fine_pairs_)
    for j, k in base.pairs_:
        j, k = int(j), int(k)
        for position, level in enumerate(levels):
            if level > 8 and (j, k) not in fine_pairs:
                continue
            card_j = int(base.encoder_.cardinalities_[level][int(feature_idx[j])])
            card_k = int(base.encoder_.cardinalities_[level][int(feature_idx[k])])
            if position == 0:
                table = _reshape_lookup(base_lookup[lookup_pos], card_j, card_k)
                source = f"pair{level}:{j},{k}"
            else:
                residual = np.asarray(
                    base.pair_residuals_[level][(j, k)], dtype=np.int64
                )
                lut = np.asarray(base_lookup[lookup_pos], dtype=np.float64)
                table = _safe_take(lut, residual).reshape(card_j, card_k)
                source = f"pair{level}res:{j},{k}"
            add_pair(j, k, maps[level][j], maps[level][k], table, source)
            raw_map_refs.update({(j, level), (k, level)})
            lookup_pos += 1

    if lookup_pos != len(base_lookup):
        raise RuntimeError(
            f"base lookup layout mismatch: consumed {lookup_pos}, have {len(base_lookup)}"
        )

    # Conditional block execution groups.
    for group_index, group in enumerate(model.execution_groups_):
        kind = group["kind"]
        gate_j = int(group["gate_j"])
        if kind == "block":
            target_k = int(group["target_k"])
            c4 = int(maps[4][gate_j].max()) + 1
            target_card = int(maps[block_level][target_k].max()) + 1
            table = np.zeros((c4, target_card), dtype=np.float64)
            source = np.asarray(group["table"], dtype=np.float64)
            table[int(group["gate_state"]), : len(source)] = source
            add_pair(
                gate_j,
                target_k,
                maps[4][gate_j],
                maps[block_level][target_k],
                table,
                f"block:{group_index}",
            )
            raw_map_refs.update({(gate_j, 4), (target_k, block_level)})
        elif kind == "fused_pair":
            target_k = int(group["target_k"])
            table = np.asarray(group["table"], dtype=np.float64)
            add_pair(
                gate_j,
                target_k,
                maps[4][gate_j],
                maps[block_level][target_k],
                table,
                f"fused_block_pair:{group_index}",
            )
            raw_map_refs.update({(gate_j, 4), (target_k, block_level)})
        elif kind == "gate_group":
            gate_state = int(group["gate_state"])
            by_target: dict[int, np.ndarray] = {}
            for target_k, target_table in group["targets"]:
                target_k = int(target_k)
                target_table = np.asarray(target_table, dtype=np.float64)
                c4 = int(maps[4][gate_j].max()) + 1
                target_card = int(maps[block_level][target_k].max()) + 1
                table = by_target.setdefault(
                    target_k, np.zeros((c4, target_card), dtype=np.float64)
                )
                table[gate_state, : len(target_table)] += target_table
            for target_k, table in by_target.items():
                add_pair(
                    gate_j,
                    target_k,
                    maps[4][gate_j],
                    maps[block_level][target_k],
                    table,
                    f"gate_group:{group_index}",
                )
                raw_map_refs.update({(gate_j, 4), (target_k, block_level)})
        else:
            raise ValueError(f"unknown execution group: {kind}")

    raw_map_ops = len(raw_map_refs)
    # Quantize + state maps + pair encodings/lookups + AddN + sigmoid.
    term_count = raw_lookup_ops
    raw_nodes = (
        len(feature_idx)
        + raw_map_ops
        + raw_lookup_ops
        + max(0, term_count - 1)
        + 1
    )
    return _RawGraph(
        feature_idx=feature_idx,
        thresholds=thresholds,
        direct_state_mask=direct_state_mask,
        direct_state_cardinalities=direct_state_cardinalities,
        unary_components=unary,
        pair_components=pairs,
        intercept=float(model.intercept_),
        raw_nodes=int(raw_nodes),
        raw_lookup_ops=int(raw_lookup_ops),
        raw_map_ops=int(raw_map_ops),
        raw_table_bytes=int(raw_table_bytes),
        metadata={
            "source_model": type(model).__name__,
            "source_model_bytes": int(getattr(model, "model_bytes_estimate_", 0)),
            "source_pair_scopes": int(len(pairs)),
            "source_base_pairs": int(len(base.pairs_)),
            "source_execution_groups": int(len(model.execution_groups_)),
            "fine_cardinalities": fine_cards.tolist(),
        },
    )


def _merge_pair_components(components: list[PairComponent]) -> tuple[list[PairComponent], int]:
    grouped: dict[tuple[Any, ...], PairComponent] = {}
    fused = 0
    for comp in components:
        key = (
            _array_key(comp.left_map),
            _array_key(comp.right_map),
            tuple(comp.table.shape),
        )
        if key not in grouped:
            grouped[key] = PairComponent(
                comp.left_map.copy(),
                comp.right_map.copy(),
                comp.table.copy(),
                list(comp.sources),
            )
        else:
            grouped[key].table += comp.table
            grouped[key].sources.extend(comp.sources)
            fused += 1
    return list(grouped.values()), fused


def _dense_pair_table(
    components: Iterable[PairComponent],
    left_card: int,
    right_card: int,
) -> np.ndarray:
    out = np.zeros((left_card, right_card), dtype=np.float64)
    left_states = np.arange(left_card, dtype=np.int64)
    right_states = np.arange(right_card, dtype=np.int64)
    for comp in components:
        out += comp.table[
            comp.left_map[left_states][:, None],
            comp.right_map[right_states][None, :],
        ]
    return out


def _factor_pair_table(table: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Exact ANOVA-like decomposition using row/column zero as references."""
    table = np.asarray(table, dtype=np.float64)
    constant = float(table[0, 0])
    left = table[:, 0] - constant
    right = table[0, :] - constant
    residual = table - constant - left[:, None] - right[None, :]
    residual[np.abs(residual) < 1e-15] = 0.0
    return constant, left, right, residual


def _unique_initializer_bytes(program: CERMGraphProgram) -> int:
    unique: dict[tuple[str, tuple[int, ...], str], int] = {}
    for threshold in program.thresholds:
        unique.setdefault(_array_key(threshold), threshold.nbytes)
    for table in program.unary_tables.values():
        unique.setdefault(_array_key(np.asarray(table)), np.asarray(table).nbytes)
    for pair in program.pair_programs:
        if pair.representation == "dense":
            table = np.asarray(pair.dense_table)
            unique.setdefault(_array_key(table), table.nbytes)
        else:
            for comp in pair.components:
                for array in (comp.left_map, comp.right_map, comp.table):
                    a = np.asarray(array)
                    unique.setdefault(_array_key(a), a.nbytes)
    return int(sum(unique.values()))


def optimize_hybrid_graph(
    model,
    objective: str = "balanced",
    op_cost_bytes: float = 96.0,
    dense_expansion_limit: float = 4.0,
    zero_atol: float = 0.0,
) -> CERMGraphProgram:
    """Apply exact ONNX-like graph rewrites to a fitted Hybrid CERM.

    Objectives:
      memory   - minimize initializer bytes;
      latency  - strongly prefer fewer lookups, subject to expansion limit;
      balanced - bytes + op_cost_bytes * lookup count.
    """
    if objective not in {"memory", "latency", "balanced"}:
        raise ValueError("objective must be memory, latency, or balanced")
    raw = build_raw_graph(model)
    cards = np.where(raw.direct_state_mask, raw.direct_state_cardinalities, np.asarray([len(t) + 1 for t in raw.thresholds], dtype=np.int64))
    intercept = float(raw.intercept)
    unary_tables: dict[int, np.ndarray] = {}
    unary_fusions = 0
    zero_removed = 0

    # Compose Map/ResidualMap/Gather chains and fuse all unary lookups per feature.
    for feature, components in raw.unary_components.items():
        table = np.zeros(int(cards[feature]), dtype=np.float64)
        nonzero_components = 0
        for state_map, lut, _source in components:
            expanded = _safe_take(lut, state_map)
            if _is_zero(expanded, zero_atol):
                zero_removed += 1
                continue
            table += expanded
            nonzero_components += 1
        if nonzero_components > 1:
            unary_fusions += nonzero_components - 1
        constant = float(table[0])
        intercept += constant
        table -= constant
        table[np.abs(table) < 1e-15] = 0.0
        if _is_zero(table, zero_atol):
            if nonzero_components:
                zero_removed += 1
            continue
        unary_tables[feature] = table

    pair_programs: list[PairProgram] = []
    pair_component_fusions = 0
    dense_pair_fusions = 0
    additive_factorizations = 0

    for (left, right), original_components in sorted(raw.pair_components.items()):
        components, fused = _merge_pair_components(original_components)
        pair_component_fusions += fused
        kept_components = []
        for comp in components:
            if _is_zero(comp.table, zero_atol):
                zero_removed += 1
            else:
                kept_components.append(comp)
        components = kept_components
        if not components:
            continue

        component_bytes = sum(c.bytes for c in components)
        component_ops = len(components)
        dense = _dense_pair_table(components, int(cards[left]), int(cards[right]))
        constant, left_unary, right_unary, residual = _factor_pair_table(dense)
        residual_zero = _is_zero(residual, zero_atol)
        dense_bytes = 0 if residual_zero else residual.nbytes
        # Unary additions do not add lookup ops when a feature already has a unary
        # program; otherwise they create at most one fused unary lookup.
        extra_unary_ops = int(
            (not _is_zero(left_unary, zero_atol) and left not in unary_tables)
            + (not _is_zero(right_unary, zero_atol) and right not in unary_tables)
        )
        dense_ops = (0 if residual_zero else 1) + extra_unary_ops
        component_score = component_bytes + op_cost_bytes * component_ops
        dense_score = dense_bytes + op_cost_bytes * dense_ops

        if objective == "memory":
            choose_dense = dense_bytes <= component_bytes
        elif objective == "latency":
            choose_dense = (
                dense_ops < component_ops
                and dense_bytes <= dense_expansion_limit * max(component_bytes, 1)
            ) or residual_zero
        else:
            choose_dense = (
                dense_score <= component_score
                and dense_bytes <= dense_expansion_limit * max(component_bytes, 1)
            ) or residual_zero

        if choose_dense:
            intercept += constant
            if not _is_zero(left_unary, zero_atol):
                unary_tables[left] = unary_tables.get(
                    left, np.zeros(int(cards[left]), dtype=np.float64)
                ) + left_unary
            if not _is_zero(right_unary, zero_atol):
                unary_tables[right] = unary_tables.get(
                    right, np.zeros(int(cards[right]), dtype=np.float64)
                ) + right_unary
            if residual_zero:
                additive_factorizations += 1
                zero_removed += 1
                continue
            pair_programs.append(
                PairProgram(
                    left,
                    right,
                    "dense",
                    dense_table=residual,
                    source_count=len(original_components),
                    factorized=not (
                        _is_zero(left_unary, zero_atol)
                        and _is_zero(right_unary, zero_atol)
                        and constant == 0.0
                    ),
                )
            )
            dense_pair_fusions += max(0, len(original_components) - 1)
        else:
            pair_programs.append(
                PairProgram(
                    left,
                    right,
                    "components",
                    components=components,
                    source_count=len(original_components),
                )
            )

    # Final unary normalization after pair factorization.
    final_unary: dict[int, np.ndarray] = {}
    for feature, table in sorted(unary_tables.items()):
        table = np.asarray(table, dtype=np.float64)
        constant = float(table[0])
        intercept += constant
        table = table - constant
        table[np.abs(table) < 1e-15] = 0.0
        if _is_zero(table, zero_atol):
            zero_removed += 1
        else:
            final_unary[feature] = table

    used_features = set(final_unary)
    for pair in pair_programs:
        used_features.update({pair.left, pair.right})
    used_sorted = sorted(used_features)
    features_pruned = len(raw.feature_idx) - len(used_sorted)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}
    if features_pruned:
        final_unary = {old_to_new[k]: v for k, v in final_unary.items()}
        for pair in pair_programs:
            pair.left = old_to_new[pair.left]
            pair.right = old_to_new[pair.right]
        program_feature_idx = raw.feature_idx[used_sorted]
        program_thresholds = [raw.thresholds[j] for j in used_sorted]
        program_direct_mask = raw.direct_state_mask[used_sorted]
        program_direct_cards = raw.direct_state_cardinalities[used_sorted]
    else:
        program_feature_idx = raw.feature_idx
        program_thresholds = raw.thresholds
        program_direct_mask = raw.direct_state_mask
        program_direct_cards = raw.direct_state_cardinalities

    optimized_lookup_ops = len(final_unary) + sum(p.lookup_ops for p in pair_programs)
    map_keys: set[tuple[str, tuple[int, ...], str]] = set()
    for pair in pair_programs:
        if pair.representation == "components":
            for comp in pair.components:
                map_keys.add(_array_key(comp.left_map))
                map_keys.add(_array_key(comp.right_map))
    optimized_map_ops = len(map_keys)
    term_count = optimized_lookup_ops
    optimized_nodes = (
        len(used_features)
        + optimized_map_ops
        + optimized_lookup_ops
        + max(0, term_count - 1)
        + 1
    )
    optimized_table_bytes = int(
        sum(t.nbytes for t in final_unary.values())
        + sum(
            np.asarray(p.dense_table).nbytes
            if p.representation == "dense"
            else sum(c.table.nbytes for c in p.components)
            for p in pair_programs
        )
    )

    program = CERMGraphProgram(
        feature_idx=program_feature_idx,
        thresholds=program_thresholds,
        unary_tables=final_unary,
        pair_programs=pair_programs,
        intercept=intercept,
        direct_state_mask=program_direct_mask,
        direct_state_cardinalities=program_direct_cards,
        metadata={
            **raw.metadata,
            "objective": objective,
            "op_cost_bytes": float(op_cost_bytes),
            "dense_expansion_limit": float(dense_expansion_limit),
            "retained_local_features": used_sorted,
            "passes": [
                "Map-Gather composition",
                "unary Gather-Add fusion",
                "pair component CSE",
                "scope canonicalization",
                "conditional block-to-pair lowering",
                "exact pair ANOVA factorization",
                "constant folding",
                "zero-term elimination",
                "state-map DCE",
                "initializer deduplication",
                "bounded state-index dtype",
            ],
        },
    )
    unique_bytes = _unique_initializer_bytes(program)
    program.stats = GraphOptimizationStats(
        objective=objective,
        raw_nodes=raw.raw_nodes,
        optimized_nodes=int(optimized_nodes),
        raw_lookup_ops=raw.raw_lookup_ops,
        optimized_lookup_ops=int(optimized_lookup_ops),
        raw_map_ops=raw.raw_map_ops,
        optimized_map_ops=int(optimized_map_ops),
        raw_table_bytes=raw.raw_table_bytes,
        optimized_table_bytes=optimized_table_bytes,
        optimized_unique_initializer_bytes=unique_bytes,
        unary_fusions=int(unary_fusions),
        pair_component_fusions=int(pair_component_fusions),
        dense_pair_fusions=int(dense_pair_fusions),
        additive_pair_factorizations=int(additive_factorizations),
        zero_terms_removed=int(zero_removed),
        features_pruned=int(features_pruned),
        float_dtype="float64",
    )
    return program


def validate_program_exactness(
    model,
    program: CERMGraphProgram,
    X: np.ndarray,
) -> dict[str, float]:
    reference_logit = np.asarray(model.decision_function(X), dtype=np.float64)
    program_logit = program.decision_function(X)
    reference_prob = _sigmoid(reference_logit)
    program_prob = _sigmoid(program_logit)
    return {
        "max_logit_error": float(np.max(np.abs(reference_logit - program_logit), initial=0.0)),
        "max_probability_error": float(
            np.max(np.abs(reference_prob - program_prob), initial=0.0)
        ),
        "mean_probability_error": float(np.mean(np.abs(reference_prob - program_prob))),
    }
