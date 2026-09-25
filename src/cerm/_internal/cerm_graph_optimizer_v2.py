from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cerm_graph_optimizer import (
    CERMGraphProgram,
    GraphOptimizationStats,
    PairComponent,
    PairProgram,
    _array_key,
    _dense_pair_table,
    _factor_pair_table,
    _is_zero,
    _state_index_dtype,
    optimize_hybrid_graph,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _safe_zero_small(array: np.ndarray, atol: float = 1e-15) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64).copy()
    out[np.abs(out) < atol] = 0.0
    return out


def _canonicalize_component(component: PairComponent) -> PairComponent:
    """Compress unreachable labels and canonicalize arbitrary mapped-state labels.

    Component state labels are internal names, unlike finest states tied to raw
    thresholds.  They may therefore be permuted without changing the function.
    Iterative row/column lexicographic sorting gives a stable representation for
    many isomorphic tables and improves initializer CSE.
    """

    left_map = np.asarray(component.left_map, dtype=np.int64)
    right_map = np.asarray(component.right_map, dtype=np.int64)
    table = np.asarray(component.table, dtype=np.float64)

    used_left = np.unique(left_map)
    used_right = np.unique(right_map)
    left_remap = np.full(table.shape[0], -1, dtype=np.int64)
    right_remap = np.full(table.shape[1], -1, dtype=np.int64)
    left_remap[used_left] = np.arange(len(used_left), dtype=np.int64)
    right_remap[used_right] = np.arange(len(used_right), dtype=np.int64)
    left_map = left_remap[left_map]
    right_map = right_remap[right_map]
    table = table[np.ix_(used_left, used_right)]

    # Internal labels can be sorted. Repeat because column ordering changes row
    # signatures and vice versa. The small cardinality keeps this inexpensive.
    for _ in range(8):
        changed = False
        row_order = np.asarray(
            sorted(range(table.shape[0]), key=lambda i: tuple(table[i].tolist())),
            dtype=np.int64,
        )
        if not np.array_equal(row_order, np.arange(table.shape[0])):
            inverse = np.empty_like(row_order)
            inverse[row_order] = np.arange(len(row_order))
            table = table[row_order]
            left_map = inverse[left_map]
            changed = True
        col_order = np.asarray(
            sorted(range(table.shape[1]), key=lambda j: tuple(table[:, j].tolist())),
            dtype=np.int64,
        )
        if not np.array_equal(col_order, np.arange(table.shape[1])):
            inverse = np.empty_like(col_order)
            inverse[col_order] = np.arange(len(col_order))
            table = table[:, col_order]
            right_map = inverse[right_map]
            changed = True
        if not changed:
            break

    return PairComponent(left_map, right_map, table, list(component.sources))


@dataclass
class SparseRowPair:
    """Exact pair table storing only active rows.

    `row_slot[state]` is -1 for an all-zero row and otherwise indexes
    `row_table`.  `left` and `right` may be swapped relative to the original
    scope to exploit column sparsity as row sparsity.
    """

    left: int
    right: int
    row_slot: np.ndarray
    row_table: np.ndarray
    source_count: int = 0
    transposed_from_source: bool = False

    def __post_init__(self):
        self.row_slot = np.asarray(self.row_slot, dtype=np.int16)
        self.row_table = np.asarray(self.row_table)
        if self.row_slot.ndim != 1 or self.row_table.ndim != 2:
            raise ValueError("invalid sparse-row pair")
        active = self.row_slot[self.row_slot >= 0]
        if active.size and int(active.max()) >= self.row_table.shape[0]:
            raise ValueError("row slot exceeds sparse table")

    @property
    def lookup_ops(self) -> int:
        return 1

    @property
    def bytes(self) -> int:
        return int(self.row_slot.nbytes + self.row_table.nbytes)


@dataclass
class PairProgramV2:
    left: int
    right: int
    representation: str  # dense | components | sparse_rows
    dense_table: np.ndarray | None = None
    components: list[PairComponent] = field(default_factory=list)
    sparse: SparseRowPair | None = None
    source_count: int = 0
    factorized: bool = False

    @property
    def lookup_ops(self) -> int:
        if self.representation == "dense":
            return 1
        if self.representation == "sparse_rows":
            return 1
        return len(self.components)

    @property
    def bytes(self) -> int:
        if self.representation == "dense":
            return int(np.asarray(self.dense_table).nbytes)
        if self.representation == "sparse_rows":
            assert self.sparse is not None
            return self.sparse.bytes
        return int(sum(c.bytes for c in self.components))


@dataclass
class GraphOptimizationStatsV2:
    base_stats: GraphOptimizationStats
    objective: str
    sparse_row_pairs: int
    sparse_transposes: int
    component_label_compressions: int
    pair_representation_changes: int
    exact_table_bytes: int
    exact_unique_initializer_bytes: int
    lookup_ops: int
    branch_ops: int
    graph_ir_version: str = "cerm-onnx-like-graph-ir-v2"


@dataclass
class CERMGraphProgramV2:
    feature_idx: np.ndarray
    thresholds: list[np.ndarray]
    unary_tables: dict[int, np.ndarray]
    pair_programs: list[PairProgramV2]
    intercept: float
    direct_state_mask: np.ndarray | None = None
    direct_state_cardinalities: np.ndarray | None = None
    table_dtype: str = "float64"
    base_stats: GraphOptimizationStats | None = None
    v2_stats: GraphOptimizationStatsV2 | None = None
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
            raise ValueError("X has fewer columns than required")
        state_dtype = _state_index_dtype(
            int(np.max(self.fine_cardinalities, initial=1)) - 1
        )
        out = np.empty((len(X), len(self.feature_idx)), dtype=state_dtype)
        for j, raw_j in enumerate(self.feature_idx):
            values = X[:, int(raw_j)]
            if self.direct_state_mask[j]:
                rounded = np.rint(values).astype(np.int64)
                card = int(self.direct_state_cardinalities[j])
                valid = np.isfinite(values) & (np.abs(values - rounded) <= 1e-9) & (rounded >= 0) & (rounded < card)
                out[:, j] = np.where(valid, rounded, 0).astype(state_dtype)
            else:
                out[:, j] = np.searchsorted(
                    self.thresholds[j], values, side="right"
                )
        return out

    def decision_function_from_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        score = np.full(len(states), self.intercept, dtype=np.float64)
        for feature, table in self.unary_tables.items():
            score += np.asarray(table)[states[:, feature]]
        for pair in self.pair_programs:
            left = states[:, pair.left]
            right = states[:, pair.right]
            if pair.representation == "dense":
                score += np.asarray(pair.dense_table)[left, right]
            elif pair.representation == "components":
                for component in pair.components:
                    score += np.asarray(component.table)[
                        component.left_map[left], component.right_map[right]
                    ]
            else:
                sparse = pair.sparse
                assert sparse is not None
                slot = sparse.row_slot[left]
                active = slot >= 0
                if np.any(active):
                    score[active] += sparse.row_table[slot[active], right[active]]
        return score

    def states_projected(self, X_selected: np.ndarray) -> np.ndarray:
        """Build states from columns already in ``feature_idx`` order."""
        X_selected = np.asarray(X_selected, dtype=np.float64)
        if X_selected.ndim != 2 or X_selected.shape[1] != len(self.feature_idx):
            raise ValueError("projected graph input width mismatch")
        state_dtype = _state_index_dtype(
            int(np.max(self.fine_cardinalities, initial=1)) - 1
        )
        out = np.empty(X_selected.shape, dtype=state_dtype)
        for j in range(len(self.feature_idx)):
            values = X_selected[:, j]
            if self.direct_state_mask[j]:
                rounded = np.rint(values).astype(np.int64)
                card = int(self.direct_state_cardinalities[j])
                valid = (
                    np.isfinite(values)
                    & (np.abs(values - rounded) <= 1e-9)
                    & (rounded >= 0)
                    & (rounded < card)
                )
                out[:, j] = np.where(valid, rounded, 0).astype(state_dtype)
            else:
                out[:, j] = np.searchsorted(
                    self.thresholds[j], values, side="right"
                )
        return out

    def decision_function_projected(self, X_selected: np.ndarray) -> np.ndarray:
        return self.decision_function_from_states(self.states_projected(X_selected))

    def predict_proba_projected(self, X_selected: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function_projected(X_selected))
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function_from_states(self.states(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def float_copy(self, dtype: str) -> "CERMGraphProgramV2":
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        dt = np.dtype(dtype)
        pairs: list[PairProgramV2] = []
        for pair in self.pair_programs:
            if pair.representation == "dense":
                pairs.append(
                    PairProgramV2(
                        pair.left,
                        pair.right,
                        "dense",
                        dense_table=np.asarray(pair.dense_table, dtype=dt),
                        source_count=pair.source_count,
                        factorized=pair.factorized,
                    )
                )
            elif pair.representation == "components":
                pairs.append(
                    PairProgramV2(
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
            else:
                assert pair.sparse is not None
                pairs.append(
                    PairProgramV2(
                        pair.left,
                        pair.right,
                        "sparse_rows",
                        sparse=SparseRowPair(
                            pair.left,
                            pair.right,
                            pair.sparse.row_slot,
                            np.asarray(pair.sparse.row_table, dtype=dt),
                            source_count=pair.sparse.source_count,
                            transposed_from_source=pair.sparse.transposed_from_source,
                        ),
                        source_count=pair.source_count,
                        factorized=pair.factorized,
                    )
                )
        return CERMGraphProgramV2(
            feature_idx=self.feature_idx.copy(),
            thresholds=[x.copy() for x in self.thresholds],
            unary_tables={k: np.asarray(v, dtype=dt) for k, v in self.unary_tables.items()},
            pair_programs=pairs,
            intercept=float(np.asarray(self.intercept, dtype=dt)),
            direct_state_mask=self.direct_state_mask.copy(),
            direct_state_cardinalities=self.direct_state_cardinalities.copy(),
            table_dtype=dtype,
            base_stats=self.base_stats,
            v2_stats=self.v2_stats,
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
        pairs_manifest = []
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
            elif pair.representation == "components":
                components = []
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
                    arrays[tb] = np.asarray(component.table)
                    components.append(
                        {
                            "left_map": lm,
                            "right_map": rm,
                            "table": tb,
                            "sources": list(component.sources),
                        }
                    )
                entry["components"] = components
            else:
                assert pair.sparse is not None
                slots = f"pair_{i}_row_slot"
                table = f"pair_{i}_row_table"
                arrays[slots] = pair.sparse.row_slot.astype(np.int16)
                arrays[table] = np.asarray(pair.sparse.row_table)
                entry.update(
                    {
                        "row_slot": slots,
                        "row_table": table,
                        "transposed_from_source": bool(
                            pair.sparse.transposed_from_source
                        ),
                    }
                )
            pairs_manifest.append(entry)
        np.savez_compressed(npz_path, **arrays)
        manifest = {
            "format": "cerm-onnx-like-graph-ir-v2",
            "input": "dense_numeric_matrix",
            "state_index_dtype": _state_index_dtype(
                int(np.max(self.fine_cardinalities, initial=1)) - 1
            ).name,
            "table_dtype": self.table_dtype,
            "feature_count": int(len(self.feature_idx)),
            "unary": unary_manifest,
            "pairs": pairs_manifest,
            "metadata": self.metadata,
            "base_stats": None if self.base_stats is None else self.base_stats.__dict__,
            "v2_stats": None if self.v2_stats is None else asdict(self.v2_stats),
        }
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return npz_path, json_path


def _dense_from_pair(pair: PairProgram, cards: np.ndarray) -> np.ndarray:
    if pair.representation == "dense":
        return np.asarray(pair.dense_table, dtype=np.float64)
    return _dense_pair_table(pair.components, int(cards[pair.left]), int(cards[pair.right]))


def _sparse_rows_from_dense(
    left: int,
    right: int,
    table: np.ndarray,
    source_count: int,
    zero_atol: float,
) -> SparseRowPair | None:
    table = _safe_zero_small(table, max(zero_atol, 1e-15))
    candidates = []
    for transposed, current in ((False, table), (True, table.T)):
        active_rows = np.flatnonzero(np.any(np.abs(current) > zero_atol, axis=1))
        if len(active_rows) == 0 or len(active_rows) == current.shape[0]:
            continue
        row_slot = np.full(current.shape[0], -1, dtype=np.int16)
        row_slot[active_rows] = np.arange(len(active_rows), dtype=np.int16)
        row_table = current[active_rows]
        candidate = SparseRowPair(
            right if transposed else left,
            left if transposed else right,
            row_slot,
            row_table,
            source_count=source_count,
            transposed_from_source=transposed,
        )
        candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda x: (x.bytes, x.row_table.shape[0]))


def _unique_initializer_bytes_v2(program: CERMGraphProgramV2) -> int:
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    total = 0

    def add(array: np.ndarray):
        nonlocal total
        key = _array_key(np.asarray(array))
        if key not in seen:
            seen.add(key)
            total += np.asarray(array).nbytes

    for threshold in program.thresholds:
        add(np.asarray(threshold, dtype=np.float64))
    for table in program.unary_tables.values():
        add(table)
    for pair in program.pair_programs:
        if pair.representation == "dense":
            add(np.asarray(pair.dense_table))
        elif pair.representation == "components":
            for component in pair.components:
                map_dtype = np.result_type(
                    _state_index_dtype(int(np.max(component.left_map, initial=0))),
                    _state_index_dtype(int(np.max(component.right_map, initial=0))),
                )
                add(component.left_map.astype(map_dtype))
                add(component.right_map.astype(map_dtype))
                add(component.table)
        else:
            assert pair.sparse is not None
            add(pair.sparse.row_slot)
            add(pair.sparse.row_table)
    return int(total)


def optimize_hybrid_graph_v2(
    model,
    objective: str = "balanced",
    *,
    op_cost_bytes: float = 96.0,
    branch_cost_bytes: float = 24.0,
    dense_expansion_limit: float = 4.0,
    sparse_expansion_limit: float = 1.25,
    zero_atol: float = 0.0,
) -> CERMGraphProgramV2:
    """Add sparse-row and mapped-label canonicalization to the v1 optimizer."""

    if objective not in {"memory", "balanced", "latency"}:
        raise ValueError("invalid objective")
    base = optimize_hybrid_graph(
        model,
        objective=objective,
        op_cost_bytes=op_cost_bytes,
        dense_expansion_limit=dense_expansion_limit,
        zero_atol=zero_atol,
    )
    cards = base.fine_cardinalities
    intercept = float(base.intercept)
    unary = {k: np.asarray(v, dtype=np.float64).copy() for k, v in base.unary_tables.items()}
    pairs: list[PairProgramV2] = []
    sparse_count = 0
    sparse_transposes = 0
    label_compressions = 0
    representation_changes = 0

    for source_pair in base.pair_programs:
        original_pair = source_pair
        original_bytes = (
            np.asarray(source_pair.dense_table).nbytes
            if source_pair.representation == "dense"
            else sum(c.bytes for c in source_pair.components)
        )
        original_ops = source_pair.lookup_ops
        original_score = original_bytes + op_cost_bytes * original_ops

        canonical_components: list[PairComponent] = []
        if source_pair.representation == "components":
            for component in source_pair.components:
                canonical = _canonicalize_component(component)
                if (
                    canonical.table.shape != component.table.shape
                    or not np.array_equal(canonical.left_map, component.left_map)
                    or not np.array_equal(canonical.right_map, component.right_map)
                ):
                    label_compressions += 1
                canonical_components.append(canonical)
            original_bytes = sum(c.bytes for c in canonical_components)
            original_score = original_bytes + op_cost_bytes * len(canonical_components)

        dense_full = _dense_from_pair(source_pair, cards)
        constant, left_unary, right_unary, residual = _factor_pair_table(dense_full)
        residual = _safe_zero_small(residual)
        residual_zero = _is_zero(residual, zero_atol)
        extra_unary_bytes = 0
        if not _is_zero(left_unary, zero_atol) and source_pair.left not in unary:
            extra_unary_bytes += left_unary.nbytes
        if not _is_zero(right_unary, zero_atol) and source_pair.right not in unary:
            extra_unary_bytes += right_unary.nbytes

        dense_bytes = (0 if residual_zero else residual.nbytes) + extra_unary_bytes
        dense_ops = 0 if residual_zero else 1
        dense_score = dense_bytes + op_cost_bytes * dense_ops

        sparse = None if residual_zero else _sparse_rows_from_dense(
            source_pair.left,
            source_pair.right,
            residual,
            source_pair.source_count,
            zero_atol,
        )
        sparse_bytes = np.inf
        sparse_score = np.inf
        if sparse is not None:
            sparse_bytes = sparse.bytes + extra_unary_bytes
            sparse_score = sparse_bytes + op_cost_bytes + branch_cost_bytes

        if objective == "memory":
            choices = [
                (original_bytes, "original"),
                (dense_bytes, "dense"),
                (sparse_bytes, "sparse"),
            ]
        elif objective == "latency":
            # Avoid a large dense expansion merely to remove a branch.
            dense_ok = dense_bytes <= dense_expansion_limit * max(original_bytes, 1)
            sparse_ok = sparse_bytes <= sparse_expansion_limit * max(original_bytes, 1)
            choices = [(op_cost_bytes * original_ops, "original")]
            if dense_ok:
                choices.append((op_cost_bytes * dense_ops + 0.02 * dense_bytes, "dense"))
            if sparse_ok:
                choices.append((op_cost_bytes + branch_cost_bytes + 0.02 * sparse_bytes, "sparse"))
        else:
            choices = [(original_score, "original"), (dense_score, "dense")]
            if sparse_bytes <= sparse_expansion_limit * max(original_bytes, 1):
                choices.append((sparse_score, "sparse"))

        _, choice = min(choices, key=lambda x: (x[0], x[1]))
        if choice == "original":
            if source_pair.representation == "dense":
                pairs.append(
                    PairProgramV2(
                        source_pair.left,
                        source_pair.right,
                        "dense",
                        dense_table=np.asarray(source_pair.dense_table).copy(),
                        source_count=source_pair.source_count,
                        factorized=source_pair.factorized,
                    )
                )
            else:
                pairs.append(
                    PairProgramV2(
                        source_pair.left,
                        source_pair.right,
                        "components",
                        components=canonical_components,
                        source_count=source_pair.source_count,
                        factorized=source_pair.factorized,
                    )
                )
            continue

        representation_changes += 1
        intercept += float(constant)
        if not _is_zero(left_unary, zero_atol):
            unary[source_pair.left] = unary.get(
                source_pair.left,
                np.zeros(int(cards[source_pair.left]), dtype=np.float64),
            ) + left_unary
        if not _is_zero(right_unary, zero_atol):
            unary[source_pair.right] = unary.get(
                source_pair.right,
                np.zeros(int(cards[source_pair.right]), dtype=np.float64),
            ) + right_unary
        if residual_zero:
            continue
        if choice == "dense":
            pairs.append(
                PairProgramV2(
                    source_pair.left,
                    source_pair.right,
                    "dense",
                    dense_table=residual,
                    source_count=source_pair.source_count,
                    factorized=True,
                )
            )
        else:
            assert sparse is not None
            pairs.append(
                PairProgramV2(
                    sparse.left,
                    sparse.right,
                    "sparse_rows",
                    sparse=sparse,
                    source_count=source_pair.source_count,
                    factorized=True,
                )
            )
            sparse_count += 1
            sparse_transposes += int(sparse.transposed_from_source)

    # Normalize unary reference states after factorization.
    normalized_unary: dict[int, np.ndarray] = {}
    for feature, table in unary.items():
        table = np.asarray(table, dtype=np.float64)
        intercept += float(table[0])
        table = _safe_zero_small(table - float(table[0]))
        if not _is_zero(table, zero_atol):
            normalized_unary[feature] = table

    used = set(normalized_unary)
    for pair in pairs:
        used.update((pair.left, pair.right))
    used_sorted = sorted(used)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}
    if len(used_sorted) != len(base.feature_idx):
        normalized_unary = {old_to_new[k]: v for k, v in normalized_unary.items()}
        for pair in pairs:
            pair.left = old_to_new[pair.left]
            pair.right = old_to_new[pair.right]
            if pair.sparse is not None:
                pair.sparse.left = pair.left
                pair.sparse.right = pair.right
        feature_idx = base.feature_idx[used_sorted]
        thresholds = [base.thresholds[j] for j in used_sorted]
        direct_state_mask = base.direct_state_mask[used_sorted]
        direct_state_cardinalities = base.direct_state_cardinalities[used_sorted]
    else:
        feature_idx = base.feature_idx.copy()
        thresholds = [x.copy() for x in base.thresholds]
        direct_state_mask = base.direct_state_mask.copy()
        direct_state_cardinalities = base.direct_state_cardinalities.copy()

    program = CERMGraphProgramV2(
        feature_idx=feature_idx,
        thresholds=thresholds,
        unary_tables=normalized_unary,
        pair_programs=pairs,
        intercept=intercept,
        direct_state_mask=direct_state_mask,
        direct_state_cardinalities=direct_state_cardinalities,
        base_stats=base.stats,
        metadata={
            **base.metadata,
            "graph_ir_version": "v2",
            "v2_passes": [
                "mapped-state label compression",
                "mapped-table permutation canonicalization",
                "sparse-row pair selection",
                "row/column orientation selection",
            ],
            "branch_cost_bytes": float(branch_cost_bytes),
            "sparse_expansion_limit": float(sparse_expansion_limit),
        },
    )
    exact_bytes = int(
        sum(np.asarray(t).nbytes for t in normalized_unary.values())
        + sum(pair.bytes for pair in pairs)
    )
    program.v2_stats = GraphOptimizationStatsV2(
        base_stats=base.stats,
        objective=objective,
        sparse_row_pairs=sparse_count,
        sparse_transposes=sparse_transposes,
        component_label_compressions=label_compressions,
        pair_representation_changes=representation_changes,
        exact_table_bytes=exact_bytes,
        exact_unique_initializer_bytes=_unique_initializer_bytes_v2(program),
        lookup_ops=len(normalized_unary) + sum(pair.lookup_ops for pair in pairs),
        branch_ops=sparse_count,
    )
    return program


def validate_v2_exactness(
    reference: CERMGraphProgram | Any,
    program: CERMGraphProgramV2,
    X: np.ndarray,
) -> dict[str, float]:
    if hasattr(reference, "decision_function"):
        ref_logit = np.asarray(reference.decision_function(X), dtype=np.float64)
    else:
        raise TypeError("reference must expose decision_function")
    out_logit = program.decision_function(X)
    ref_prob = _sigmoid(ref_logit)
    out_prob = _sigmoid(out_logit)
    return {
        "max_logit_error": float(np.max(np.abs(ref_logit - out_logit), initial=0.0)),
        "max_probability_error": float(np.max(np.abs(ref_prob - out_prob), initial=0.0)),
        "mean_probability_error": float(np.mean(np.abs(ref_prob - out_prob))),
    }
