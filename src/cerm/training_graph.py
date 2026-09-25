from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
from typing import Sequence

import numpy as np
from joblib import effective_n_jobs
from scipy import sparse

from ._compat.sklearn import PrivateSklearnAPIUnavailable, train_binary_liblinear


@dataclass(frozen=True)
class BlockColumnView:
    """A prefix view of a maximal conditional-block column bank."""

    train: sparse.csr_matrix
    valid: sparse.csr_matrix
    specs: tuple[dict, ...]
    requested_terms: int
    realized_terms: int


@dataclass
class BlockColumnBank:
    """Exact reusable design columns for one ranked block dictionary.

    A bank is keyed by ``(rank_mode, shrinkage)``.  It builds every column for
    the largest requested block prefix once.  Smaller candidates are exact
    column-prefix views, because block terms and the within-block effect-coded
    columns are emitted in ranked order.
    """

    train: sparse.csr_matrix
    valid: sparse.csr_matrix
    boundaries: np.ndarray
    spec_boundaries: np.ndarray
    specs: tuple[dict, ...]
    max_terms: int

    def view(self, n_terms: int) -> BlockColumnView:
        n = max(0, min(int(n_terms), self.max_terms))
        col_end = int(self.boundaries[n])
        spec_end = int(self.spec_boundaries[n])
        return BlockColumnView(
            train=self.train[:, :col_end],
            valid=self.valid[:, :col_end],
            specs=self.specs[:spec_end],
            requested_terms=n,
            realized_terms=spec_end,
        )

    @classmethod
    def build(
        cls,
        C4_train: np.ndarray,
        C16_train: np.ndarray,
        C4_valid: np.ndarray,
        C16_valid: np.ndarray,
        p_base: np.ndarray,
        terms: Sequence[tuple],
        shrinkage: float,
        sample_weight: np.ndarray | None = None,
    ) -> "BlockColumnBank":
        C4_train = np.asarray(C4_train)
        C16_train = np.asarray(C16_train)
        C4_valid = np.asarray(C4_valid)
        C16_valid = np.asarray(C16_valid)
        p_base = np.asarray(p_base, dtype=float)
        terms = list(terms)

        h = np.maximum(p_base * (1.0 - p_base), 1e-8)
        if sample_weight is not None:
            h = h * np.asarray(sample_weight, dtype=np.float64)

        # Columns are generated one at a time in their final ranked order.
        # Preserve that natural column-major layout instead of materializing a
        # dense zero matrix or redundant COO column-index arrays.  The CSC
        # constructor consumes exactly the generated row/value slices; one
        # final conversion supplies the CSR layout expected by the solvers.
        train_rows: list[np.ndarray] = []
        train_values: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        valid_values: list[np.ndarray] = []
        specs: list[dict] = []
        boundaries = np.zeros(len(terms) + 1, dtype=np.int32)
        spec_boundaries = np.zeros(len(terms) + 1, dtype=np.int32)
        column_index = 0

        for term_idx, (gate_j, gate_state, target_k, target_card, gain) in enumerate(terms):
            train_gate_rows = np.flatnonzero(
                C4_train[:, gate_j] == gate_state
            )
            H = np.bincount(
                C16_train[train_gate_rows, target_k],
                weights=h[train_gate_rows],
                minlength=target_card,
            )
            total_H = float(H.sum())
            states: list[int] = []
            centers: list[float] = []
            scales: list[float] = []

            if total_H > 1e-12 and target_card > 1:
                valid_gate_rows = np.flatnonzero(
                    C4_valid[:, gate_j] == gate_state
                )
                train_target = C16_train[train_gate_rows, target_k]
                valid_target = C16_valid[valid_gate_rows, target_k]

                train_gate_rows_i32 = train_gate_rows.astype(
                    np.int32, copy=False
                )
                valid_gate_rows_i32 = valid_gate_rows.astype(
                    np.int32, copy=False
                )
                for state in range(target_card - 1):
                    pi = float(H[state] / total_H)
                    variance_mass = total_H * pi * (1.0 - pi)
                    if variance_mass <= 1e-12:
                        continue
                    scale = math.sqrt(
                        variance_mass / (variance_mass + shrinkage)
                    )
                    train_rows.append(train_gate_rows_i32)
                    train_values.append(
                        ((train_target == state).astype(float) - pi) * scale
                    )
                    valid_rows.append(valid_gate_rows_i32)
                    valid_values.append(
                        ((valid_target == state).astype(float) - pi) * scale
                    )
                    states.append(int(state))
                    centers.append(pi)
                    scales.append(scale)
                    column_index += 1

            if states:
                specs.append(
                    {
                        "gate_j": int(gate_j),
                        "gate_state": int(gate_state),
                        "target_k": int(target_k),
                        "target_card": int(target_card),
                        "states": states,
                        "centers": centers,
                        "scales": scales,
                        "gain": float(gain),
                    }
                )

            boundaries[term_idx + 1] = column_index
            spec_boundaries[term_idx + 1] = len(specs)

        def assemble_columns(rows, values, n_rows):
            n_columns = len(rows)
            if n_columns == 0:
                return sparse.csr_matrix((n_rows, 0), dtype=float)

            lengths = np.fromiter(
                (len(row_index) for row_index in rows),
                dtype=np.int64,
                count=n_columns,
            )
            indptr = np.empty(n_columns + 1, dtype=np.int64)
            indptr[0] = 0
            indptr[1:] = np.cumsum(lengths, dtype=np.int64)
            indices = np.concatenate(rows)
            data = np.concatenate(values)
            return sparse.csc_matrix(
                (data, indices, indptr),
                shape=(n_rows, n_columns),
                dtype=float,
            ).tocsr()

        train = assemble_columns(train_rows, train_values, len(C4_train))
        valid = assemble_columns(valid_rows, valid_values, len(C4_valid))
        return cls(
            train=train,
            valid=valid,
            boundaries=boundaries,
            spec_boundaries=spec_boundaries,
            specs=tuple(specs),
            max_terms=len(terms),
        )



@dataclass(frozen=True)
class SemanticBlockDictionary:
    """Finite-state metadata for conditional block columns without a CSR suffix."""

    block_target: np.ndarray
    coef_offsets: np.ndarray
    state_ids: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    train_row_offsets: np.ndarray
    train_rows: np.ndarray
    valid_row_offsets: np.ndarray
    valid_rows: np.ndarray
    specs: tuple[dict, ...]

    @property
    def n_columns(self) -> int:
        return int(self.coef_offsets[-1]) if len(self.coef_offsets) else 0

    @classmethod
    def build(
        cls,
        C4_train: np.ndarray,
        C16_train: np.ndarray,
        C4_valid: np.ndarray,
        p_base: np.ndarray,
        terms: Sequence[tuple],
        shrinkage: float,
        sample_weight: np.ndarray | None = None,
    ) -> "SemanticBlockDictionary":
        """Build semantic block metadata directly from ranked terms."""
        C4_train = np.asarray(C4_train)
        C16_train = np.asarray(C16_train)
        C4_valid = np.asarray(C4_valid)
        p_base = np.asarray(p_base, dtype=np.float64).reshape(-1)
        terms = list(terms)
        if (
            C4_train.ndim != 2
            or C16_train.ndim != 2
            or C4_valid.ndim != 2
            or len(C4_train) != len(C16_train)
            or len(C4_train) != len(p_base)
        ):
            raise ValueError("semantic block training shape mismatch")

        hessian = np.maximum(p_base * (1.0 - p_base), 1e-8)
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(weight) != len(C4_train):
                raise ValueError("sample_weight length mismatch")
            hessian = hessian * weight

        specs: list[dict] = []
        for gate_j, gate_state, target_k, target_card, gain in terms:
            gate_j = int(gate_j)
            gate_state = int(gate_state)
            target_k = int(target_k)
            target_card = int(target_card)
            train_rows = np.flatnonzero(
                C4_train[:, gate_j] == gate_state
            )
            H = np.bincount(
                C16_train[train_rows, target_k],
                weights=hessian[train_rows],
                minlength=target_card,
            )
            total_H = float(H.sum())
            states: list[int] = []
            centers: list[float] = []
            scales: list[float] = []
            if total_H > 1e-12 and target_card > 1:
                for state in range(target_card - 1):
                    pi = float(H[state] / total_H)
                    variance_mass = total_H * pi * (1.0 - pi)
                    if variance_mass <= 1e-12:
                        continue
                    states.append(int(state))
                    centers.append(pi)
                    scales.append(
                        math.sqrt(
                            variance_mass / (variance_mass + float(shrinkage))
                        )
                    )
            if states:
                specs.append(
                    {
                        "gate_j": gate_j,
                        "gate_state": gate_state,
                        "target_k": target_k,
                        "target_card": target_card,
                        "states": states,
                        "centers": centers,
                        "scales": scales,
                        "gain": float(gain),
                    }
                )
        return cls.from_specs(C4_train, C4_valid, specs)

    @classmethod
    def from_specs(
        cls,
        C4_train: np.ndarray,
        C4_valid: np.ndarray,
        specs: Sequence[dict],
    ) -> "SemanticBlockDictionary":
        C4_train = np.asarray(C4_train)
        C4_valid = np.asarray(C4_valid)
        specs = tuple(dict(spec) for spec in specs)

        targets = np.empty(len(specs), dtype=np.int32)
        coef_offsets = np.zeros(len(specs) + 1, dtype=np.int32)
        train_row_offsets = np.zeros(len(specs) + 1, dtype=np.int32)
        valid_row_offsets = np.zeros(len(specs) + 1, dtype=np.int32)
        state_pieces = []
        center_pieces = []
        scale_pieces = []
        train_rows = []
        valid_rows = []

        for index, spec in enumerate(specs):
            gate_j = int(spec["gate_j"])
            gate_state = int(spec["gate_state"])
            targets[index] = int(spec["target_k"])

            states = np.asarray(spec["states"], dtype=np.uint8)
            centers = np.asarray(spec["centers"], dtype=np.float64)
            scales = np.asarray(spec["scales"], dtype=np.float64)
            if not (len(states) == len(centers) == len(scales)):
                raise ValueError("semantic block specification length mismatch")
            if len(states) and len(np.unique(states)) != len(states):
                raise ValueError("semantic block state ids must be unique")

            tr = np.flatnonzero(C4_train[:, gate_j] == gate_state).astype(
                np.int32, copy=False
            )
            vr = np.flatnonzero(C4_valid[:, gate_j] == gate_state).astype(
                np.int32, copy=False
            )
            state_pieces.append(states)
            center_pieces.append(centers)
            scale_pieces.append(scales)
            train_rows.append(tr)
            valid_rows.append(vr)

            coef_offsets[index + 1] = coef_offsets[index] + len(states)
            train_row_offsets[index + 1] = train_row_offsets[index] + len(tr)
            valid_row_offsets[index + 1] = valid_row_offsets[index] + len(vr)

        def concatenate(pieces, dtype):
            if not pieces:
                return np.empty(0, dtype=dtype)
            return np.ascontiguousarray(np.concatenate(pieces), dtype=dtype)

        return cls(
            block_target=targets,
            coef_offsets=coef_offsets,
            state_ids=concatenate(state_pieces, np.uint8),
            centers=concatenate(center_pieces, np.float64),
            scales=concatenate(scale_pieces, np.float64),
            train_row_offsets=train_row_offsets,
            train_rows=concatenate(train_rows, np.int32),
            valid_row_offsets=valid_row_offsets,
            valid_rows=concatenate(valid_rows, np.int32),
            specs=specs,
        )

    def valid_linear_response(
        self,
        base_valid,
        target_states_valid: np.ndarray,
        coefficient: np.ndarray,
        intercept: float,
    ) -> np.ndarray:
        """Evaluate the exact semantic columns without materializing their CSR."""
        base_valid = sparse.csr_matrix(base_valid, dtype=np.float64, copy=False)
        states = np.asarray(target_states_valid)
        coef = np.asarray(coefficient, dtype=np.float64).reshape(-1)
        base_dim = int(base_valid.shape[1])
        if len(coef) != base_dim + self.n_columns:
            raise ValueError("semantic coefficient dimension mismatch")
        if states.ndim != 2 or states.shape[0] != base_valid.shape[0]:
            raise ValueError("semantic validation state shape mismatch")

        score = np.asarray(base_valid @ coef[:base_dim]).ravel()
        score = score + float(intercept)
        block_coef = coef[base_dim:]

        for block in range(len(self.block_target)):
            begin = int(self.coef_offsets[block])
            end = int(self.coef_offsets[block + 1])
            row_begin = int(self.valid_row_offsets[block])
            row_end = int(self.valid_row_offsets[block + 1])
            rows = self.valid_rows[row_begin:row_end]
            if begin == end or len(rows) == 0:
                continue

            weights = block_coef[begin:end]
            centers = self.centers[begin:end]
            scales = self.scales[begin:end]
            lookup = np.full(256, -float(np.sum(weights * centers * scales)))
            lookup[self.state_ids[begin:end]] += weights * scales
            target = int(self.block_target[block])
            score[rows] += lookup[states[rows, target].astype(np.uint8)]

        return score



@dataclass(frozen=True)
class SemanticBlockBank:
    """Maximal semantic dictionary with O(1)-ish nested prefix views."""

    dictionary: SemanticBlockDictionary
    term_spec_boundaries: np.ndarray
    max_terms: int

    @classmethod
    def build(
        cls,
        C4_train: np.ndarray,
        C16_train: np.ndarray,
        C4_valid: np.ndarray,
        p_base: np.ndarray,
        terms: Sequence[tuple],
        shrinkage: float,
        sample_weight: np.ndarray | None = None,
    ) -> "SemanticBlockBank":
        terms = list(terms)
        dictionary = SemanticBlockDictionary.build(
            C4_train,
            C16_train,
            C4_valid,
            p_base,
            terms,
            shrinkage,
            sample_weight=sample_weight,
        )
        boundaries = np.zeros(len(terms) + 1, dtype=np.int32)
        spec_index = 0
        specs = dictionary.specs
        for term_index, term in enumerate(terms):
            if spec_index < len(specs):
                spec = specs[spec_index]
                structural = (
                    int(spec["gate_j"]),
                    int(spec["gate_state"]),
                    int(spec["target_k"]),
                    int(spec["target_card"]),
                )
                requested = (
                    int(term[0]),
                    int(term[1]),
                    int(term[2]),
                    int(term[3]),
                )
                if structural == requested:
                    spec_index += 1
            boundaries[term_index + 1] = spec_index
        if spec_index != len(specs):
            raise RuntimeError("semantic term/spec prefix mapping mismatch")
        return cls(
            dictionary=dictionary,
            term_spec_boundaries=boundaries,
            max_terms=len(terms),
        )

    def view(self, n_terms: int) -> SemanticBlockDictionary:
        n = max(0, min(int(n_terms), self.max_terms))
        spec_end = int(self.term_spec_boundaries[n])
        source = self.dictionary
        coef_end = int(source.coef_offsets[spec_end])
        train_row_end = int(source.train_row_offsets[spec_end])
        valid_row_end = int(source.valid_row_offsets[spec_end])
        return SemanticBlockDictionary(
            block_target=source.block_target[:spec_end],
            coef_offsets=source.coef_offsets[: spec_end + 1],
            state_ids=source.state_ids[:coef_end],
            centers=source.centers[:coef_end],
            scales=source.scales[:coef_end],
            train_row_offsets=source.train_row_offsets[: spec_end + 1],
            train_rows=source.train_rows[:train_row_end],
            valid_row_offsets=source.valid_row_offsets[: spec_end + 1],
            valid_rows=source.valid_rows[:valid_row_end],
            specs=source.specs[:spec_end],
        )


@dataclass
class EncodedColumnBank:
    """One-hot encoded maximal design with exact code-column block views."""

    train: sparse.csr_matrix
    valid: sparse.csr_matrix
    cardinalities: np.ndarray
    offsets: np.ndarray

    @classmethod
    def build(cls, train_codes: np.ndarray, valid_codes: np.ndarray, encoder_cls):
        encoder = encoder_cls()
        train = encoder.fit_transform(train_codes)
        valid = encoder.transform(valid_codes)
        cards = np.asarray(encoder.cardinalities_, dtype=np.int32)
        widths = np.maximum(cards - 1, 0)
        offsets = np.concatenate([[0], np.cumsum(widths, dtype=np.int64)])
        return cls(train=train, valid=valid, cardinalities=cards, offsets=offsets)

    def encoded_indices(self, code_columns: Sequence[int]) -> np.ndarray:
        pieces = [
            np.arange(self.offsets[j], self.offsets[j + 1], dtype=np.int64)
            for j in code_columns
            if self.offsets[j + 1] > self.offsets[j]
        ]
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(pieces)

    def view(self, code_columns: Sequence[int]):
        idx = self.encoded_indices(code_columns)
        return self.train[:, idx], self.valid[:, idx]

    def lookup_bytes(self, code_columns: Sequence[int], itemsize: int = 8) -> int:
        cols = np.asarray(list(code_columns), dtype=np.int64)
        if cols.size == 0:
            return 0
        return int(np.maximum(self.cardinalities[cols], 1).sum() * itemsize)


@dataclass(frozen=True)
class LogisticPathSolution:
    C: float
    coefficient: np.ndarray
    intercept: float
    valid_probability: np.ndarray
    n_iter: int


@lru_cache(maxsize=1)
def _semantic_block_native_core():
    from ._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    if getattr(core, "_binary_logistic_semantic_blocks", None) is None:
        raise RuntimeError(
            "native training core lacks semantic-block logistic extension"
        )
    return core


def semantic_block_solver_available() -> bool:
    try:
        _semantic_block_native_core()
        return True
    except Exception:
        return False


def solve_binary_logistic_semantic_blocks(
    base_train,
    target_states_train: np.ndarray,
    semantic: SemanticBlockDictionary,
    y,
    base_valid,
    target_states_valid: np.ndarray,
    *,
    C: float,
    random_state: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
    _seed_override: int | None = None,
) -> LogisticPathSolution:
    """Solve one CERM conditional-block design through the semantic TRON path.

    This is an experimental internal operator. It preserves the existing
    liblinear/TRON objective and stopping rule while replacing only the block
    CSR suffix with finite-state Xv/XTv operations.
    """
    train = sparse.csr_matrix(base_train, dtype=np.float64, copy=False)
    valid = sparse.csr_matrix(base_valid, dtype=np.float64, copy=False)
    if train.indices.dtype.itemsize > 4 or train.indptr.dtype.itemsize > 4:
        raise ValueError("semantic solver requires 32-bit CSR indices")
    if train.indices.dtype != np.int32 or train.indptr.dtype != np.int32:
        train = train.copy()
        train.indices = train.indices.astype(np.int32)
        train.indptr = train.indptr.astype(np.int32)
    if valid.shape[1] != train.shape[1]:
        raise ValueError("semantic train/valid base dimension mismatch")

    states_train = np.ascontiguousarray(target_states_train, dtype=np.uint8)
    states_valid = np.ascontiguousarray(target_states_valid, dtype=np.uint8)
    if states_train.ndim != 2 or len(states_train) != train.shape[0]:
        raise ValueError("semantic train state shape mismatch")
    if states_valid.ndim != 2 or len(states_valid) != valid.shape[0]:
        raise ValueError("semantic validation state shape mismatch")

    target = np.asarray(y)
    if target.ndim != 1 or len(target) != train.shape[0]:
        raise ValueError("binary target shape mismatch")
    if not np.array_equal(np.unique(target), np.asarray([0, 1])):
        raise ValueError("semantic solver requires validated 0/1 targets")

    core = _semantic_block_native_core()

    if _seed_override is None:
        rng = np.random.RandomState(int(random_state))
        seed = int(rng.randint(np.iinfo("i").max))
    else:
        seed = int(_seed_override)
    raw_coef, n_iter = core.binary_logistic_semantic_blocks(
        train,
        states_train,
        semantic.block_target,
        semantic.coef_offsets,
        semantic.state_ids,
        semantic.centers,
        semantic.scales,
        semantic.train_row_offsets,
        semantic.train_rows,
        target,
        C=float(C),
        seed=seed,
        max_iter=int(max_iter),
        sample_weight=sample_weight,
    )
    row = np.asarray(raw_coef[0], dtype=np.float64)
    coefficient = row[:-1].copy()
    intercept = float(row[-1])
    logit = semantic.valid_linear_response(
        valid,
        states_valid,
        coefficient,
        intercept,
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    return LogisticPathSolution(
        C=float(C),
        coefficient=coefficient,
        intercept=intercept,
        valid_probability=probability,
        n_iter=int(np.asarray(n_iter).ravel()[0]),
    )



def fit_binary_logistic_semantic_exact(
    base_train,
    target_states_train: np.ndarray,
    semantic: SemanticBlockDictionary,
    y,
    *,
    C: float,
    random_state: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
):
    """Return a LogisticRegression-compatible model from semantic block TRON."""
    from sklearn.linear_model import LogisticRegression

    empty_base = sparse.csr_matrix(
        (0, int(sparse.csr_matrix(base_train).shape[1])),
        dtype=np.float64,
    )
    empty_states = np.empty(
        (0, int(np.asarray(target_states_train).shape[1])),
        dtype=np.uint8,
    )
    empty_semantic = SemanticBlockDictionary(
        block_target=semantic.block_target,
        coef_offsets=semantic.coef_offsets,
        state_ids=semantic.state_ids,
        centers=semantic.centers,
        scales=semantic.scales,
        train_row_offsets=semantic.train_row_offsets,
        train_rows=semantic.train_rows,
        valid_row_offsets=np.zeros(
            len(semantic.block_target) + 1, dtype=np.int32
        ),
        valid_rows=np.empty(0, dtype=np.int32),
        specs=semantic.specs,
    )
    solution = solve_binary_logistic_semantic_blocks(
        base_train,
        target_states_train,
        empty_semantic,
        y,
        empty_base,
        empty_states,
        C=float(C),
        random_state=int(random_state),
        max_iter=int(max_iter),
        sample_weight=sample_weight,
    )
    model = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model.classes_ = np.asarray([0, 1], dtype=np.asarray(y).dtype)
    model.coef_ = solution.coefficient.reshape(1, -1)
    model.intercept_ = np.asarray([solution.intercept], dtype=np.float64)
    model.n_iter_ = np.asarray([solution.n_iter], dtype=np.int32)
    model.n_features_in_ = int(len(solution.coefficient))
    return model


def solve_binary_logistic_semantic_path(
    base_train,
    target_states_train: np.ndarray,
    semantic: SemanticBlockDictionary,
    y,
    base_valid,
    target_states_valid: np.ndarray,
    C_values: Sequence[float],
    *,
    random_state: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
    n_jobs: int | None = 1,
) -> dict[float, LogisticPathSolution]:
    """Solve multiple C values with the historical per-C seed sequence."""
    Cs = np.asarray(sorted({float(C) for C in C_values}), dtype=np.float64)
    if Cs.size == 0:
        return {}

    rng = np.random.RandomState(int(random_state))
    seeds = [int(rng.randint(np.iinfo("i").max)) for _ in Cs]

    def solve_one(item):
        C, seed = item
        return solve_binary_logistic_semantic_blocks(
            base_train,
            target_states_train,
            semantic,
            y,
            base_valid,
            target_states_valid,
            C=float(C),
            random_state=int(random_state),
            max_iter=int(max_iter),
            sample_weight=sample_weight,
            _seed_override=int(seed),
        )

    workers = min(max(1, effective_n_jobs(n_jobs)), len(Cs))
    items = list(zip(Cs.tolist(), seeds))
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            solutions = list(pool.map(solve_one, items))
    else:
        solutions = [solve_one(item) for item in items]
    return {float(solution.C): solution for solution in solutions}

def solve_binary_logistic_path(
    train,
    y,
    valid,
    C_values: Sequence[float],
    *,
    random_state: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
    n_jobs: int | None = 1,
) -> dict[float, LogisticPathSolution]:
    """Solve exact binary L2-logistic candidates on one validated design.

    This is a specialized TrainingGraph operator for the package's internal
    binary/liblinear case.  It bypasses repeated estimator validation and label
    encoding, but calls the same scikit-learn vendored LIBLINEAR kernel with the
    same solver type, tolerance, intercept scaling, sample weights, and random
    seed sequence as ``LogisticRegression(solver="liblinear")``.  The resulting
    coefficients are bitwise identical on supported sklearn versions.

    A conservative public-estimator fallback keeps the package functional when
    the low-level ABI changes or an unsupported matrix representation is used.
    """
    Cs = np.asarray(sorted({float(C) for C in C_values}), dtype=np.float64)
    if Cs.size == 0:
        return {}

    try:
        target = np.asarray(y)
        if target.ndim != 1 or target.shape[0] != train.shape[0]:
            raise ValueError("binary target shape mismatch")
        if not np.array_equal(np.unique(target), np.asarray([0, 1])):
            raise ValueError("specialized liblinear operator requires 0/1 targets")

        if sparse.issparse(train):
            design = sparse.csr_matrix(train, dtype=np.float64, copy=False)
            # LIBLINEAR's sparse ABI accepts only 32-bit index arrays.
            if design.indices.dtype.itemsize > 4 or design.indptr.dtype.itemsize > 4:
                raise ValueError("64-bit sparse indices are not supported")
        else:
            design = np.asarray(train, dtype=np.float64, order="C")

        if design.ndim != 2 or design.shape[0] != target.shape[0]:
            raise ValueError("binary design shape mismatch")
        target64 = np.require(target, dtype=np.float64, requirements="W").ravel()
        rng = np.random.RandomState(int(random_state))
        seeds = [int(rng.randint(np.iinfo("i").max)) for _ in Cs]

        def solve_one(item):
            C, seed = item
            raw_coef, n_iter = train_binary_liblinear(
                design,
                target64,
                C=float(C),
                seed=int(seed),
                max_iter=int(max_iter),
                sample_weight=sample_weight,
            )
            row = np.asarray(raw_coef[0], dtype=np.float64)
            coef = row[:-1].copy()
            intercept = float(row[-1])
            logit = np.asarray(valid @ coef).ravel() + intercept
            probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
            return LogisticPathSolution(
                C=float(C),
                coefficient=coef,
                intercept=intercept,
                valid_probability=probability,
                n_iter=int(np.asarray(n_iter).ravel()[0]),
            )

        workers = min(max(1, effective_n_jobs(n_jobs)), len(Cs))
        items = list(zip(Cs.tolist(), seeds))
        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                solutions = list(pool.map(solve_one, items))
        else:
            solutions = [solve_one(item) for item in items]
        return {float(solution.C): solution for solution in solutions}
    except (PrivateSklearnAPIUnavailable, TypeError, ValueError, OverflowError):
        from sklearn.linear_model import LogisticRegression

        result = {}
        for C in Cs:
            model = LogisticRegression(
                C=float(C),
                solver="liblinear",
                max_iter=int(max_iter),
                random_state=int(random_state),
            ).fit(train, y, sample_weight=sample_weight)
            result[float(C)] = LogisticPathSolution(
                C=float(C),
                coefficient=model.coef_.ravel().copy(),
                intercept=float(model.intercept_[0]),
                valid_probability=model.predict_proba(valid)[:, 1],
                n_iter=int(np.asarray(model.n_iter_).ravel()[0]),
            )
        return result


def fit_binary_logistic_exact(
    train,
    y,
    *,
    C: float,
    random_state: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
):
    """Return a fitted LogisticRegression-compatible exact binary model.

    The numerical solve is delegated to :func:`solve_binary_logistic_path`,
    hence it uses the same LIBLINEAR objective and seed sequence.  Populating
    the public fitted attributes preserves internal reference-prediction paths
    without paying repeated estimator validation and label encoding.
    """
    from sklearn.linear_model import LogisticRegression

    empty_valid = train[:0]
    solution = solve_binary_logistic_path(
        train, y, empty_valid, [float(C)],
        random_state=int(random_state), max_iter=int(max_iter),
        sample_weight=sample_weight,
    )[float(C)]
    model = LogisticRegression(
        C=float(C), solver="liblinear", max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model.classes_ = np.asarray([0, 1], dtype=np.asarray(y).dtype)
    model.coef_ = solution.coefficient.reshape(1, -1)
    model.intercept_ = np.asarray([solution.intercept], dtype=np.float64)
    model.n_iter_ = np.asarray([solution.n_iter], dtype=np.int32)
    model.n_features_in_ = int(train.shape[1])
    return model


def binary_log_loss(y, probability, eps: float = 1e-10, sample_weight=None) -> float:
    """Fast binary log loss for validated 0/1 targets."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    losses = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    if sample_weight is None:
        return float(np.mean(losses))
    weights = np.asarray(sample_weight, dtype=float)
    return float(np.average(losses, weights=weights))
