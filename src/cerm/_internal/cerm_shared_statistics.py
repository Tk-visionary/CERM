from __future__ import annotations

"""Shared nested sufficient-statistics engine for hybrid block ranking."""

import math
import os

import numpy as np


def nested_gate_maps(
    C4: np.ndarray,
    C16: np.ndarray,
    rows: np.ndarray,
    feature_limit: int,
) -> tuple[np.ndarray, ...] | None:
    """Infer observed fine->coarse maps or return None for non-nested states.

    CERM's fitted 4/16 state hierarchy is a deterministic quotient.  Keeping
    this check local lets synthetic/internal callers that provide unrelated
    state matrices fall back to the historical ordered-pair implementation.
    Unobserved fine states map to zero because they carry zero histogram mass.
    """
    d = min(int(C4.shape[1]), int(feature_limit))
    coarse_rows = np.asarray(C4[rows, :d], dtype=np.int64, order="F")
    fine_rows = np.asarray(C16[rows, :d], dtype=np.int64, order="F")
    maps: list[np.ndarray] = []

    for feature in range(d):
        fine = fine_rows[:, feature]
        coarse = coarse_rows[:, feature]
        fine_card = int(fine.max(initial=0)) + 1
        lower = np.full(
            fine_card,
            np.iinfo(np.int64).max,
            dtype=np.int64,
        )
        upper = np.full(fine_card, -1, dtype=np.int64)
        np.minimum.at(lower, fine, coarse)
        np.maximum.at(upper, fine, coarse)
        observed = upper >= 0
        if np.any(lower[observed] != upper[observed]):
            return None
        mapping = np.zeros(fine_card, dtype=np.int64)
        mapping[observed] = upper[observed]
        maps.append(mapping)

    return tuple(maps)


def _collapse_gate_axis(
    table: np.ndarray,
    mapping: np.ndarray,
    gate_card: int,
) -> np.ndarray:
    """Collapse the fine gate axis of a two-fold pair histogram."""
    output = np.zeros(
        (table.shape[0], int(gate_card), table.shape[2]),
        dtype=np.float64,
    )
    for fine_state, coarse_state in enumerate(mapping):
        output[:, int(coarse_state), :] += table[:, fine_state, :]
    return output


def _resolve_native_threads(n_jobs: int | None, feature_limit: int) -> int:
    if n_jobs is None:
        return min(4, max(1, os.cpu_count() or 1), max(1, int(feature_limit)))
    value = int(n_jobs)
    if value == 0:
        return 1
    if value < 0:
        value = max(1, (os.cpu_count() or 1) + 1 + value)
    return min(max(1, value), max(1, int(feature_limit)))


def _iter_fine_pair_fold_tables(
    fine_keys: np.ndarray,
    target_cards: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    fold_a_size: int,
    *,
    n_jobs: int | None,
    backend_info: dict | None,
):
    """Yield unordered fine-pair fold G/H tables.

    Prefer the existing native multi-value pair-histogram ABI when a packaged
    training core is already available. Source/editable installs do not trigger
    a new compiler invocation solely for this optimization; they retain the
    NumPy fallback unless the core was already loaded by another training path.
    """
    d = int(fine_keys.shape[1])
    pairs = np.asarray(
        [(left, right) for left in range(d) for right in range(left + 1, d)],
        dtype=np.int32,
    )

    core = None
    if (
        len(pairs)
        and int(np.max(target_cards, initial=0)) <= 256
        and int(np.max(fine_keys, initial=0)) <= 255
    ):
        try:
            from .cerm_training_core_runtime import (
                _prebuilt_library_path,
                load_native_training_core,
            )

            already_loaded = load_native_training_core.cache_info().currsize > 0
            prebuilt = _prebuilt_library_path()
            if already_loaded or (prebuilt is not None and prebuilt.is_file()):
                core = load_native_training_core()
        except Exception:
            core = None

    if core is not None:
        values = np.zeros((len(fine_keys), 4), dtype=np.float64)
        split = int(fold_a_size)
        values[:split, 0] = g[:split]
        values[:split, 1] = h[:split]
        values[split:, 2] = g[split:]
        values[split:, 3] = h[split:]
        cards = np.ascontiguousarray(target_cards, dtype=np.int64)
        states_u8 = np.ascontiguousarray(fine_keys, dtype=np.uint8)
        n_threads = _resolve_native_threads(n_jobs, d)

        # Bound temporary native histogram memory. pair_histogram allocates one
        # mass value plus four requested sums per fine cell, so budget on
        # 5 * sizeof(double) per cell. Chunking also improves cache behavior in
        # the wide-feature regimes that motivated this shared-statistics path.
        max_batch_bytes = 2 * 1024 * 1024
        bytes_per_cell = 5 * np.dtype(np.float64).itemsize
        widths = (
            cards[pairs[:, 0]].astype(np.int64)
            * cards[pairs[:, 1]].astype(np.int64)
        )
        batches = []
        begin = 0
        cells = 0
        for index, width in enumerate(widths):
            width = int(width)
            if index > begin and (cells + width) * bytes_per_cell > max_batch_bytes:
                batches.append((begin, index))
                begin = index
                cells = 0
            cells += width
        if begin < len(pairs):
            batches.append((begin, len(pairs)))

        if backend_info is not None:
            backend_info["pair_histogram"] = (
                "native" if len(batches) == 1 else "native_chunked"
            )
            backend_info["native_batches"] = len(batches)
            backend_info["native_batch_budget_bytes"] = max_batch_bytes

        for begin, end in batches:
            pair_batch = np.ascontiguousarray(pairs[begin:end])
            pair_offsets, pair_sums = core.pair_histogram(
                states_u8,
                cards,
                pair_batch,
                values,
                n_threads=n_threads,
            )
            for local_index, (left, right) in enumerate(pair_batch):
                left = int(left)
                right = int(right)
                left_card = int(target_cards[left])
                right_card = int(target_cards[right])
                start = int(pair_offsets[local_index])
                stop = int(pair_offsets[local_index + 1])
                table = pair_sums[start:stop].reshape(
                    left_card, right_card, 4
                )
                G_fine = np.stack(
                    [table[:, :, 0], table[:, :, 2]],
                    axis=0,
                )
                H_fine = np.stack(
                    [table[:, :, 1], table[:, :, 3]],
                    axis=0,
                )
                yield left, right, G_fine, H_fine
        return

    if backend_info is not None:
        backend_info["pair_histogram"] = "python"
    pair_fold_code = np.empty(len(fine_keys), dtype=np.int64)
    split = int(fold_a_size)
    for left in range(d):
        left_card = int(target_cards[left])
        for right in range(left + 1, d):
            right_card = int(target_cards[right])
            np.multiply(
                fine_keys[:, left],
                right_card,
                out=pair_fold_code,
            )
            np.add(
                pair_fold_code,
                fine_keys[:, right],
                out=pair_fold_code,
            )
            base_card = left_card * right_card
            pair_fold_code[split:] += base_card
            total_card = base_card * 2
            G_fine = np.bincount(
                pair_fold_code,
                weights=g,
                minlength=total_card,
            ).reshape(2, left_card, right_card)
            H_fine = np.bincount(
                pair_fold_code,
                weights=h,
                minlength=total_card,
            ).reshape(2, left_card, right_card)
            yield left, right, G_fine, H_fine



def _iter_directional_pair_fold_tables(
    fine_keys: np.ndarray,
    target_cards: np.ndarray,
    maps: tuple[np.ndarray, ...],
    gate_cards: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    fold_a_size: int,
    *,
    n_jobs: int | None,
    backend_info: dict | None,
):
    """Yield both coarse-gate -> fine-target directions for each unordered pair.

    New native cores can fuse fine-pair accumulation with quotient aggregation.
    Older ABI-v2 cores and Python-only installations fall back to the existing
    fine histogram plus Python collapse path while preserving this interface.
    """
    d = int(fine_keys.shape[1])
    pairs = np.asarray(
        [(left, right) for left in range(d) for right in range(left + 1, d)],
        dtype=np.int32,
    )

    core = None
    if (
        len(pairs)
        and int(np.max(target_cards, initial=0)) <= 256
        and int(np.max(fine_keys, initial=0)) <= 255
    ):
        try:
            from .cerm_training_core_runtime import (
                _prebuilt_library_path,
                load_native_training_core,
            )

            already_loaded = load_native_training_core.cache_info().currsize > 0
            prebuilt = _prebuilt_library_path()
            if already_loaded or (prebuilt is not None and prebuilt.is_file()):
                core = load_native_training_core()
        except Exception:
            core = None

    quotient_symbol = (
        core is not None
        and getattr(core, "_quotient_pair_histogram", None) is not None
    )
    if quotient_symbol and len(pairs):
        values = np.zeros((len(fine_keys), 4), dtype=np.float64)
        split = int(fold_a_size)
        values[:split, 0] = g[:split]
        values[:split, 1] = h[:split]
        values[split:, 2] = g[split:]
        values[split:, 3] = h[split:]
        cards = np.ascontiguousarray(target_cards, dtype=np.int64)
        states_u8 = np.ascontiguousarray(fine_keys, dtype=np.uint8)
        n_threads = _resolve_native_threads(n_jobs, d)

        widths = (
            gate_cards[pairs[:, 0]].astype(np.int64)
            * cards[pairs[:, 1]].astype(np.int64)
            + gate_cards[pairs[:, 1]].astype(np.int64)
            * cards[pairs[:, 0]].astype(np.int64)
        )
        bytes_per_cell = 5 * np.dtype(np.float64).itemsize
        max_batch_bytes = 2 * 1024 * 1024
        batches = []
        begin = 0
        cells = 0
        for index, width in enumerate(widths):
            width = int(width)
            if index > begin and (cells + width) * bytes_per_cell > max_batch_bytes:
                batches.append((begin, index))
                begin = index
                cells = 0
            cells += width
        if begin < len(pairs):
            batches.append((begin, len(pairs)))

        first = True
        for begin, end in batches:
            pair_batch = np.ascontiguousarray(pairs[begin:end])
            pair_offsets, pair_sums = core.quotient_pair_histogram(
                states_u8,
                cards,
                maps,
                pair_batch,
                values,
                n_threads=n_threads,
            )

            if first and backend_info is not None:
                backend_info["pair_histogram"] = (
                    "native_quotient"
                    if len(batches) == 1
                    else "native_quotient_chunked"
                )
                backend_info["native_batches"] = len(batches)
                backend_info["native_batch_budget_bytes"] = max_batch_bytes
                first = False

            for local_index, (left, right) in enumerate(pair_batch):
                left = int(left)
                right = int(right)
                left_card = int(target_cards[left])
                right_card = int(target_cards[right])
                left_gate = int(gate_cards[left])
                right_gate = int(gate_cards[right])
                start = int(pair_offsets[local_index])
                stop = int(pair_offsets[local_index + 1])
                midpoint = start + left_gate * right_card
                left_table = pair_sums[start:midpoint].reshape(
                    left_gate, right_card, 4
                )
                right_table = pair_sums[midpoint:stop].reshape(
                    right_gate, left_card, 4
                )
                yield (
                    left,
                    right,
                    right_card,
                    np.stack([left_table[:, :, 0], left_table[:, :, 2]], axis=0),
                    np.stack([left_table[:, :, 1], left_table[:, :, 3]], axis=0),
                )
                yield (
                    right,
                    left,
                    left_card,
                    np.stack([right_table[:, :, 0], right_table[:, :, 2]], axis=0),
                    np.stack([right_table[:, :, 1], right_table[:, :, 3]], axis=0),
                )
        return

    for left, right, G_fine, H_fine in _iter_fine_pair_fold_tables(
        fine_keys,
        target_cards,
        g,
        h,
        fold_a_size,
        n_jobs=n_jobs,
        backend_info=backend_info,
    ):
        left_card = int(target_cards[left])
        right_card = int(target_cards[right])
        yield (
            left,
            right,
            right_card,
            _collapse_gate_axis(
                G_fine,
                maps[left],
                int(gate_cards[left]),
            ),
            _collapse_gate_axis(
                H_fine,
                maps[left],
                int(gate_cards[left]),
            ),
        )
        yield (
            right,
            left,
            left_card,
            _collapse_gate_axis(
                G_fine.transpose(0, 2, 1),
                maps[right],
                int(gate_cards[right]),
            ),
            _collapse_gate_axis(
                H_fine.transpose(0, 2, 1),
                maps[right],
                int(gate_cards[right]),
            ),
        )



def _native_pair_gain_bank_prepared(
    fine_keys: np.ndarray,
    target_cards: np.ndarray,
    maps: tuple[np.ndarray, ...],
    gate_cards: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray | None,
    fold_a_size: int,
    stable_min_support: np.ndarray,
    full_min_support: float,
    *,
    gain_l2: float,
    min_hessian: float,
    n_jobs: int | None,
    backend_info: dict | None,
):
    """Return a compact native candidate bank, or None without the extension."""
    d = int(fine_keys.shape[1])
    pairs = np.asarray(
        [(left, right) for left in range(d) for right in range(left + 1, d)],
        dtype=np.int32,
    )
    if not len(pairs):
        return {
            "gate_j": np.empty(0, dtype=np.int32),
            "gate_state": np.empty(0, dtype=np.int16),
            "target_k": np.empty(0, dtype=np.int32),
            "target_card": np.empty(0, dtype=np.int16),
            "full": np.empty(0, dtype=np.float64),
            "gain_a": np.empty(0, dtype=np.float64),
            "gain_b": np.empty(0, dtype=np.float64),
            "flags": np.empty(0, dtype=np.uint8),
        }

    core = None
    if (
        int(np.max(target_cards, initial=0)) <= 256
        and int(np.max(fine_keys, initial=0)) <= 255
    ):
        try:
            from .cerm_training_core_runtime import (
                _prebuilt_library_path,
                load_native_training_core,
            )

            already_loaded = load_native_training_core.cache_info().currsize > 0
            prebuilt = _prebuilt_library_path()
            if already_loaded or (prebuilt is not None and prebuilt.is_file()):
                core = load_native_training_core()
        except Exception:
            core = None

    if core is None or getattr(core, "_quotient_pair_gain", None) is None:
        return None

    values = np.zeros((len(fine_keys), 6), dtype=np.float64)
    split = int(fold_a_size)
    mass = (
        np.ones(len(fine_keys), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    values[:split, 0] = mass[:split]
    values[:split, 1] = g[:split]
    values[:split, 2] = h[:split]
    values[split:, 3] = mass[split:]
    values[split:, 4] = g[split:]
    values[split:, 5] = h[split:]

    n_threads = _resolve_native_threads(n_jobs, d)
    candidate_offsets, full, gain_a, gain_b, flags = core.quotient_pair_gains(
        np.ascontiguousarray(fine_keys, dtype=np.uint8),
        np.ascontiguousarray(target_cards, dtype=np.int64),
        maps,
        pairs,
        values,
        n_threads=n_threads,
        gain_l2=float(gain_l2),
        min_hessian=float(min_hessian),
        min_support_a=float(stable_min_support[0]),
        min_support_b=float(stable_min_support[1]),
        min_support_full=float(full_min_support),
    )

    total = len(full)
    gate_j = np.empty(total, dtype=np.int32)
    gate_state = np.empty(total, dtype=np.int16)
    target_k = np.empty(total, dtype=np.int32)
    target_card = np.empty(total, dtype=np.int16)
    for pair_index, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        left_gate = int(gate_cards[left])
        right_gate = int(gate_cards[right])
        left_card = int(target_cards[left])
        right_card = int(target_cards[right])
        start = int(candidate_offsets[pair_index])
        middle = start + left_gate
        stop = int(candidate_offsets[pair_index + 1])
        if stop != middle + right_gate:
            raise RuntimeError("native quotient candidate layout mismatch")

        gate_j[start:middle] = left
        gate_state[start:middle] = np.arange(left_gate, dtype=np.int16)
        target_k[start:middle] = right
        target_card[start:middle] = right_card

        gate_j[middle:stop] = right
        gate_state[middle:stop] = np.arange(right_gate, dtype=np.int16)
        target_k[middle:stop] = left
        target_card[middle:stop] = left_card

    if backend_info is not None:
        backend_info["pair_histogram"] = "native_gain"
        backend_info["candidate_evaluator"] = "native_gain"
        backend_info["native_candidate_count"] = int(total)

    return {
        "gate_j": gate_j,
        "gate_state": gate_state,
        "target_k": target_k,
        "target_card": target_card,
        "full": full,
        "gain_a": gain_a,
        "gain_b": gain_b,
        "flags": flags,
    }


def native_paired_fold_block_gain_bank(
    C4: np.ndarray,
    C16: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    fold_a: np.ndarray,
    fold_b: np.ndarray,
    *,
    feature_limit: int,
    full_min_support: int,
    gain_l2: float = 5.0,
    min_hessian: float = 1.0,
    sample_weight: np.ndarray | None = None,
    maps: tuple[np.ndarray, ...] | None = None,
    n_jobs: int | None = 1,
    backend_info: dict | None = None,
):
    """Return native full/stable candidate arrays without Python object rows."""
    fold_a = np.asarray(fold_a, dtype=np.int64)
    fold_b = np.asarray(fold_b, dtype=np.int64)
    rows = np.concatenate([fold_a, fold_b])
    if maps is None:
        maps = nested_gate_maps(C4, C16, rows, feature_limit)
    if maps is None:
        return None

    y_sub = np.asarray(y)[rows]
    p_sub = np.asarray(p, dtype=np.float64)[rows]
    g = np.asarray(y_sub, dtype=np.float64) - p_sub
    h = np.maximum(p_sub * (1.0 - p_sub), 1e-8)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)[rows]
        g = g * weights
        h = h * weights

    d = min(int(C16.shape[1]), int(feature_limit))
    fine_keys = np.asarray(C16[rows, :d], dtype=np.int64, order="F")
    target_cards = np.asarray([len(mapping) for mapping in maps], dtype=np.int64)
    gate_cards = np.asarray(
        [int(mapping.max(initial=0)) + 1 for mapping in maps],
        dtype=np.int64,
    )
    stable_min_support = np.asarray(
        [
            max(10, int(math.ceil(0.025 * len(fold_a) * 2))),
            max(10, int(math.ceil(0.025 * len(fold_b) * 2))),
        ],
        dtype=np.float64,
    )
    return _native_pair_gain_bank_prepared(
        fine_keys,
        target_cards,
        maps,
        gate_cards,
        g,
        h,
        weights,
        len(fold_a),
        stable_min_support,
        float(full_min_support),
        gain_l2=gain_l2,
        min_hessian=min_hessian,
        n_jobs=n_jobs,
        backend_info=backend_info,
    )


def _iter_native_gain_bank_rows(bank, *, include_full: bool):
    """Compatibility materializer for callers that still consume row tuples."""
    gate_j = bank["gate_j"]
    gate_state = bank["gate_state"]
    target_k = bank["target_k"]
    target_card = bank["target_card"]
    full = bank["full"]
    gain_a = bank["gain_a"]
    gain_b = bank["gain_b"]
    flags = bank["flags"]
    for offset in range(len(flags)):
        flag = int(flags[offset])
        full_value = (
            float(full[offset])
            if include_full and (flag & 4)
            else None
        )
        if (flag & 3) == 3:
            stable_a = float(gain_a[offset])
            stable_b = float(gain_b[offset])
        else:
            stable_a = None
            stable_b = None
        if full_value is not None or stable_a is not None:
            yield (
                (
                    int(gate_j[offset]),
                    int(gate_state[offset]),
                    int(target_k[offset]),
                    int(target_card[offset]),
                ),
                full_value,
                stable_a,
                stable_b,
            )

def iter_paired_fold_block_gains_shared(
    C4: np.ndarray,
    C16: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    fold_a: np.ndarray,
    fold_b: np.ndarray,
    *,
    feature_limit: int,
    full_min_support: int,
    gain_l2: float = 5.0,
    min_hessian: float = 1.0,
    sample_weight: np.ndarray | None = None,
    maps: tuple[np.ndarray, ...] | None = None,
    n_jobs: int | None = 1,
    backend_info: dict | None = None,
    include_full: bool = True,
):
    """Yield full and stable gains from one fine histogram per unordered pair.

    Each yielded row is (key, full_gain, stable_a, stable_b). full_gain is
    None when the candidate fails the full-data support/Hessian checks; the
    stable values are both None unless the candidate passes in both folds.

    The full statistic is obtained by summing the two fold-specific collapsed
    G/H tables. This changes floating-point summation order relative to a
    direct full-data row scan, but does not change the candidate family.
    """
    fold_a = np.asarray(fold_a, dtype=np.int64)
    fold_b = np.asarray(fold_b, dtype=np.int64)
    rows = np.concatenate([fold_a, fold_b])
    if maps is None:
        maps = nested_gate_maps(C4, C16, rows, feature_limit)
    if maps is None:
        raise ValueError("C4 must be a deterministic quotient of C16")

    y_sub = np.asarray(y)[rows]
    p_sub = np.asarray(p, dtype=np.float64)[rows]
    g = np.asarray(y_sub, dtype=np.float64) - p_sub
    h = np.maximum(p_sub * (1.0 - p_sub), 1e-8)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)[rows]
        g = g * weights
        h = h * weights

    d = min(int(C16.shape[1]), int(feature_limit))
    fine_keys = np.asarray(C16[rows, :d], dtype=np.int64, order="F")
    target_cards = np.asarray([len(mapping) for mapping in maps], dtype=np.int64)
    gate_cards = np.asarray(
        [int(mapping.max(initial=0)) + 1 for mapping in maps],
        dtype=np.int64,
    )
    stable_min_support = np.asarray(
        [
            max(10, int(math.ceil(0.025 * len(fold_a) * 2))),
            max(10, int(math.ceil(0.025 * len(fold_b) * 2))),
        ],
        dtype=np.float64,
    )

    native_bank = _native_pair_gain_bank_prepared(
        fine_keys,
        target_cards,
        maps,
        gate_cards,
        g,
        h,
        weights,
        len(fold_a),
        stable_min_support,
        float(full_min_support),
        gain_l2=gain_l2,
        min_hessian=min_hessian,
        n_jobs=n_jobs,
        backend_info=backend_info,
    )
    if native_bank is not None:
        yield from _iter_native_gain_bank_rows(
            native_bank,
            include_full=include_full,
        )
        return

    supports: list[np.ndarray] = []
    support_code = np.empty(len(rows), dtype=np.int64)
    for gate_j in range(d):
        gate_card = int(gate_cards[gate_j])
        gate_values = maps[gate_j][fine_keys[:, gate_j]]
        np.copyto(support_code, gate_values, casting="unsafe")
        support_code[len(fold_a):] += gate_card
        if weights is None:
            support = np.bincount(
                support_code,
                minlength=gate_card * 2,
            ).reshape(2, gate_card)
        else:
            support = np.bincount(
                support_code,
                weights=weights,
                minlength=gate_card * 2,
            ).reshape(2, gate_card)
        supports.append(support)

    def evaluate(
        G: np.ndarray,
        H: np.ndarray,
        support: np.ndarray,
        minimum_support: float,
    ):
        total_G = G.sum(axis=1)
        total_H = H.sum(axis=1)
        active = H > 1e-12
        valid = (
            (support >= float(minimum_support))
            & (total_H >= float(min_hessian))
            & (active.sum(axis=1) > 1)
        )
        parent_gain = total_G * total_G / (total_H + gain_l2)
        child_gain = np.sum(
            np.where(
                active,
                (G * G) / (H + gain_l2),
                0.0,
            ),
            axis=1,
        )
        return valid, 0.5 * np.maximum(0.0, child_gain - parent_gain)

    for gate_j, target_k, target_card, G_dir, H_dir in (
        _iter_directional_pair_fold_tables(
            fine_keys,
            target_cards,
            maps,
            gate_cards,
            g,
            h,
            len(fold_a),
            n_jobs=n_jobs,
            backend_info=backend_info,
        )
    ):
        support = supports[gate_j]
        valid_a, gain_a = evaluate(
            G_dir[0],
            H_dir[0],
            support[0],
            stable_min_support[0],
        )
        valid_b, gain_b = evaluate(
            G_dir[1],
            H_dir[1],
            support[1],
            stable_min_support[1],
        )
        stable_valid = valid_a & valid_b

        if include_full:
            G_full = G_dir[0] + G_dir[1]
            H_full = H_dir[0] + H_dir[1]
            support_full = support[0] + support[1]
            full_valid, full_gain = evaluate(
                G_full,
                H_full,
                support_full,
                float(full_min_support),
            )
        else:
            full_valid = np.zeros_like(stable_valid, dtype=bool)
            full_gain = np.zeros_like(gain_a, dtype=np.float64)

        for gate_state in np.flatnonzero(full_valid | stable_valid):
            key = (
                int(gate_j),
                int(gate_state),
                int(target_k),
                int(target_card),
            )
            full_value = (
                float(full_gain[gate_state])
                if full_valid[gate_state]
                else None
            )
            if stable_valid[gate_state]:
                stable_a = float(gain_a[gate_state])
                stable_b = float(gain_b[gate_state])
            else:
                stable_a = None
                stable_b = None
            yield key, full_value, stable_a, stable_b
