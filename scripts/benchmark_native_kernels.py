#!/usr/bin/env python3
"""Durable research benchmark harness for XGBoost-inspired CERM native kernels.

Covers:
1. state histogram
2. quotient pair gain
3. triad histogram / stage1
4. conditional cross-Gram
5. semantic vs CSR TRON

Provides:
- Deterministic synthetic workload generators (small, medium/large)
- Decoupled correctness/parity checks vs timing measurements
- RSS / memory usage tracking
- Rich environment metadata (git, OS, BLAS/OpenMP, Python, CERM config)
- JSON / CSV report exports
- Manual two-revision comparison mode (--compare report1.json report2.json)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


def get_rss_bytes() -> int:
    """Return max RSS or memory footprint in bytes."""
    try:
        import resource

        # On Linux, ru_maxrss is in kilobytes; on macOS in bytes.
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = ru.ru_maxrss
        if platform.system() != "Darwin":
            rss *= 1024
        return int(rss)
    except Exception:
        return 0


def collect_environment_metadata() -> Dict[str, Any]:
    """Collect rich system and software environment metadata."""
    meta: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "env_vars": {
            "CERM_ROW_TILE_SIZE": os.getenv("CERM_ROW_TILE_SIZE"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
        },
    }

    # Git metadata
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        meta["git_commit"] = git_hash
        meta["git_branch"] = git_branch
    except Exception:
        meta["git_commit"] = "unknown"
        meta["git_branch"] = "unknown"

    # Package versions
    try:
        import cerm

        meta["cerm_version"] = getattr(cerm, "__version__", "unknown")
    except Exception:
        meta["cerm_version"] = "unavailable"

    meta["numpy_version"] = np.__version__
    import scipy

    meta["scipy_version"] = scipy.__version__

    # Native training core metadata
    try:
        from cerm._internal import cerm_training_core_runtime as runtime

        supported = runtime.native_training_core_supported()
        meta["native_core_supported"] = supported
        if supported:
            core = runtime.load_native_training_core()
            abi_fn = getattr(core, "_abi_version", None) or getattr(core, "abi_version", None)
            meta["native_core_abi_version"] = abi_fn() if callable(abi_fn) else (abi_fn if abi_fn is not None else "unknown")
            meta["native_core_compiler"] = runtime._compiler()
            meta[
                "native_has_triad"
            ] = getattr(core, "_triad_histogram", None) is not None
            meta[
                "native_has_quotient_gain"
            ] = getattr(core, "_quotient_pair_gain", None) is not None
            meta[
                "native_has_semantic_tron"
            ] = getattr(core, "_binary_logistic_semantic_blocks", None) is not None
    except Exception as exc:
        meta["native_core_supported"] = False
        meta["native_core_error"] = str(exc)

    return meta


# -----------------------------------------------------------------------------
# 1. State Histogram Workload
# -----------------------------------------------------------------------------


def generate_state_histogram_workload(scale: str, seed: int = 42):
    """Generate synthetic state histogram inputs."""
    rng = np.random.default_rng(seed)
    if scale == "small":
        n_rows, n_cols, n_stats = 10_000, 20, 2
    else:  # medium / large
        n_rows, n_cols, n_stats = 250_000, 50, 2

    cards = rng.integers(4, 17, size=n_cols, dtype=np.int32)
    states = np.empty((n_rows, n_cols), dtype=np.uint8)
    for j, card in enumerate(cards):
        states[:, j] = rng.integers(0, int(card), size=n_rows, dtype=np.uint8)

    stats = rng.normal(size=(n_rows, n_stats)).astype(np.float64)
    return states, cards, stats


def check_state_histogram_parity(states, cards, stats, n_threads: int = 1):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    offsets, sums = core.state_histogram(states, cards, stats, n_threads=n_threads)

    # Reference implementation
    total_states = int(offsets[-1])
    ref_sums = np.zeros((total_states, stats.shape[1]), dtype=np.float64)
    for j in range(len(cards)):
        start = int(offsets[j])
        col_states = states[:, j]
        for s in range(int(cards[j])):
            mask = col_states == s
            if np.any(mask):
                ref_sums[start + s] = np.sum(stats[mask], axis=0)

    max_abs_diff = float(np.max(np.abs(sums - ref_sums)))
    denom = np.maximum(np.abs(ref_sums), 1e-12)
    max_rel_diff = float(np.max(np.abs(sums - ref_sums) / denom))
    passed = max_abs_diff <= 1e-10

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance": 1e-10,
    }


def run_state_histogram_timing(states, cards, stats, n_threads: int = 1):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    return core.state_histogram(states, cards, stats, n_threads=n_threads)


# -----------------------------------------------------------------------------
# 2. Quotient Pair Gain Workload
# -----------------------------------------------------------------------------


def generate_quotient_pair_gain_workload(scale: str, seed: int = 42):
    """Generate synthetic quotient pair gain inputs."""
    rng = np.random.default_rng(seed)
    if scale == "small":
        n_rows, n_cols, n_pairs = 10_000, 16, 30
    else:  # medium / large
        n_rows, n_cols, n_pairs = 200_000, 32, 120

    cards = np.full(n_cols, 16, dtype=np.int32)
    states = rng.integers(0, 16, size=(n_rows, n_cols), dtype=np.uint8)

    # Deterministic pairs
    all_pairs = [
        (j, k) for j in range(n_cols) for k in range(j + 1, n_cols)
    ]
    if len(all_pairs) > n_pairs:
        indices = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        indices.sort()
        pairs = np.array([all_pairs[i] for i in indices], dtype=np.int32)
    else:
        pairs = np.array(all_pairs, dtype=np.int32)

    values6 = rng.normal(scale=1.0, size=(n_rows, 6)).astype(np.float64)
    # Ensure positive Hessians
    values6[:, 2] = np.abs(values6[:, 2]) + 0.1
    values6[:, 5] = np.abs(values6[:, 5]) + 0.1

    coarse_maps = [
        np.minimum((np.arange(16) * 4) // 16, 3).astype(np.int64)
        for _ in range(n_cols)
    ]

    return states, cards, coarse_maps, pairs, values6


def check_quotient_pair_gain_parity(
    states, cards, coarse_maps, pairs, values6, n_threads: int = 1
):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()

    gain_l2 = 20.0
    min_hessian = 1e-3
    min_a = 1
    min_b = 1
    min_full = 1

    candidate_offsets, full_n, ga_n, gb_n, fl_n = core.quotient_pair_gains(
        states,
        cards,
        coarse_maps,
        pairs,
        values6,
        n_threads=n_threads,
        gain_l2=gain_l2,
        min_hessian=min_hessian,
        min_support_a=min_a,
        min_support_b=min_b,
        min_support_full=min_full,
    )

    stat_offsets, directional = core.quotient_pair_histogram(
        states,
        cards,
        coarse_maps,
        pairs,
        values6,
        n_threads=n_threads,
    )

    reference_full = np.zeros_like(full_n)

    for pair_index, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        left_gate = int(coarse_maps[left].max(initial=0)) + 1
        right_gate = int(coarse_maps[right].max(initial=0)) + 1
        left_card = int(cards[left])
        right_card = int(cards[right])

        stat_start = int(stat_offsets[pair_index])
        stat_mid = stat_start + left_gate * right_card
        stat_stop = int(stat_offsets[pair_index + 1])
        candidate_start = int(candidate_offsets[pair_index])

        tables = (
            directional[stat_start:stat_mid].reshape(
                left_gate, right_card, 6
            ),
            directional[stat_mid:stat_stop].reshape(
                right_gate, left_card, 6
            ),
        )
        position = candidate_start
        for table in tables:
            mass_a = table[:, :, 0].sum(axis=1)
            grad_a = table[:, :, 1].sum(axis=1)
            hess_a = table[:, :, 2].sum(axis=1)
            mass_b = table[:, :, 3].sum(axis=1)
            grad_b = table[:, :, 4].sum(axis=1)
            hess_b = table[:, :, 5].sum(axis=1)

            active_full = (table[:, :, 2] + table[:, :, 5]) > 1e-12
            grad_full = table[:, :, 1] + table[:, :, 4]
            hess_full_cells = table[:, :, 2] + table[:, :, 5]

            child_full = np.sum(
                np.where(
                    active_full,
                    grad_full ** 2 / (hess_full_cells + gain_l2),
                    0.0,
                ),
                axis=1,
            )

            valid_full = (
                (mass_a + mass_b >= min_full)
                & (hess_a + hess_b >= min_hessian)
                & (active_full.sum(axis=1) > 1)
            )

            size = len(mass_a)
            sl = slice(position, position + size)
            tot_g = grad_a + grad_b
            tot_h = hess_a + hess_b
            reference_full[sl] = 0.5 * np.maximum(
                0.0,
                child_full - tot_g * tot_g / (tot_h + gain_l2),
            )
            reference_full[sl] = np.where(valid_full, reference_full[sl], 0.0)
            position += size

    max_abs_diff = float(np.max(np.abs(full_n - reference_full)))
    denom = np.maximum(np.abs(reference_full), 1e-12)
    max_rel_diff = float(np.max(np.abs(full_n - reference_full) / denom))
    passed = max_abs_diff <= 1e-8

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance": 1e-8,
    }


def run_quotient_pair_gain_timing(
    states, cards, coarse_maps, pairs, values6, n_threads: int = 1
):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    return core.quotient_pair_gains(
        states,
        cards,
        coarse_maps,
        pairs,
        values6,
        n_threads=n_threads,
        gain_l2=20.0,
        min_hessian=1e-3,
        min_support_a=1,
        min_support_b=1,
        min_support_full=1,
    )


# -----------------------------------------------------------------------------
# 3. Triad Histogram Workload
# -----------------------------------------------------------------------------


def generate_triad_histogram_workload(scale: str, seed: int = 42):
    """Generate synthetic triad histogram inputs."""
    rng = np.random.default_rng(seed)
    if scale == "small":
        n_rows, n_cols, n_triads = 10_000, 12, 20
    else:  # medium / large
        n_rows, n_cols, n_triads = 150_000, 24, 80

    states_q4 = rng.integers(0, 4, size=(n_rows, n_cols), dtype=np.uint8)

    # Generate distinct triad column index triplets
    triads_list = []
    while len(triads_list) < n_triads:
        triplet = tuple(
            sorted(rng.choice(n_cols, size=3, replace=False).tolist())
        )
        if triplet not in triads_list:
            triads_list.append(triplet)

    triads = np.array(triads_list, dtype=np.int32)
    stats = rng.normal(size=(n_rows, 2)).astype(np.float64)

    return states_q4, triads, stats


def check_triad_histogram_parity(states_q4, triads, stats, n_threads: int = 1):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    actual = core.triad_histogram(states_q4, triads, stats, n_threads=n_threads)

    # Reference NumPy bitwise implementation
    n_triads = len(triads)
    n_stats = stats.shape[1]
    expected = np.zeros((n_triads, 64, n_stats), dtype=np.float64)

    for t, (a, b, c) in enumerate(triads):
        code = (
            states_q4[:, a].astype(np.int64) * 16
            + states_q4[:, b].astype(np.int64) * 4
            + states_q4[:, c].astype(np.int64)
        )
        for stat_idx in range(n_stats):
            expected[t, :, stat_idx] = np.bincount(
                code, weights=stats[:, stat_idx], minlength=64
            )

    max_abs_diff = float(np.max(np.abs(actual - expected)))
    denom = np.maximum(np.abs(expected), 1e-12)
    max_rel_diff = float(np.max(np.abs(actual - expected) / denom))
    passed = max_abs_diff <= 1e-10

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance": 1e-10,
    }


def run_triad_histogram_timing(states_q4, triads, stats, n_threads: int = 1):
    from cerm._internal.cerm_training_core_runtime import load_native_training_core

    core = load_native_training_core()
    return core.triad_histogram(states_q4, triads, stats, n_threads=n_threads)


# -----------------------------------------------------------------------------
# 4. Conditional Cross-Gram Workload
# -----------------------------------------------------------------------------


def generate_conditional_cross_gram_workload(scale: str, seed: int = 42):
    """Generate synthetic conditional cross-Gram inputs."""
    from cerm import _experimental_undr_geometry as geom

    rng = np.random.default_rng(seed)
    if scale == "small":
        n_rows, n_cols, n_sel, n_cand = 10_000, 12, 4, 16
    else:  # medium / large
        n_rows, n_cols, n_sel, n_cand = 150_000, 24, 8, 32

    states16 = rng.integers(0, 16, size=(n_rows, n_cols), dtype=np.uint8)
    score = rng.normal(size=n_rows)
    y = rng.integers(0, 2, size=n_rows, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)

    sel_cols = rng.choice(n_cols, size=min(n_sel, n_cols), replace=False)
    cand_cols = rng.choice(n_cols, size=min(n_cand, n_cols), replace=True)

    selected = [
        geom.resolution_basis(
            states16, y, score, int(col), 4, 16, name=f"sel_{i}"
        )
        for i, col in enumerate(sel_cols)
    ]
    candidates = [
        geom.resolution_basis(
            states16, y, score, int(col), 4, 16, name=f"cand_{i}"
        )
        for i, col in enumerate(cand_cols)
    ]

    return selected, candidates, h


def check_conditional_cross_gram_parity(selected, candidates, h):
    from cerm import _experimental_undr_geometry as geom
    from cerm.experimental_undr_v21 import _native_conditional_cross_grams

    actual_blocks = _native_conditional_cross_grams(selected, candidates, h)
    if actual_blocks is None:
        return {
            "passed": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "tolerance": 1e-10,
            "note": "native cross-Gram returned None (fallback)",
        }

    expected_blocks = [
        np.vstack([geom.cross_gram(left, cand, h) for left in selected])
        for cand in candidates
    ]

    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for act, exp in zip(actual_blocks, expected_blocks):
        diff = np.abs(act - exp)
        m_abs = float(np.max(diff))
        denom = np.maximum(np.abs(exp), 1e-12)
        m_rel = float(np.max(diff / denom))
        max_abs_diff = max(max_abs_diff, m_abs)
        max_rel_diff = max(max_rel_diff, m_rel)

    passed = max_abs_diff <= 1e-10

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "tolerance": 1e-10,
    }


def run_conditional_cross_gram_timing(selected, candidates, h):
    from cerm.experimental_undr_v21 import _native_conditional_cross_grams

    return _native_conditional_cross_grams(selected, candidates, h)


# -----------------------------------------------------------------------------
# 5. Semantic vs CSR TRON Workload
# -----------------------------------------------------------------------------


def generate_semantic_vs_csr_tron_workload(scale: str, seed: int = 42):
    """Generate synthetic semantic vs CSR TRON inputs."""
    from cerm.training_graph import BlockColumnBank, SemanticBlockDictionary

    rng = np.random.default_rng(seed)
    if scale == "small":
        n_train, n_valid, d, n_terms = 5_000, 1_000, 10, 12
    else:  # medium / large
        n_train, n_valid, d, n_terms = 50_000, 10_000, 20, 30

    C4t = rng.integers(0, 4, size=(n_train, d), dtype=np.uint8)
    C4v = rng.integers(0, 4, size=(n_valid, d), dtype=np.uint8)
    C16t = rng.integers(0, 12, size=(n_train, d), dtype=np.uint8)
    C16v = rng.integers(0, 12, size=(n_valid, d), dtype=np.uint8)

    p_base = np.clip(rng.uniform(0.1, 0.9, size=n_train), 1e-4, 1.0 - 1e-4)

    terms = []
    for t in range(n_terms):
        gate_j = t % d
        gate_st = t % 4
        target_k = (t + 1) % d
        target_card = 12
        gain = float(rng.uniform(1.0, 5.0))
        terms.append((gate_j, gate_st, target_k, target_card, gain))

    bank = BlockColumnBank.build(
        C4t, C16t, C4v, C16v, p_base, terms, 10.0
    )
    view = bank.view(len(terms))
    semantic = SemanticBlockDictionary.from_specs(C4t, C4v, view.specs)

    base_train = sparse.random(
        n_train, 15, density=0.15, format="csr", random_state=seed, dtype=np.float64
    )
    base_valid = sparse.random(
        n_valid, 15, density=0.15, format="csr", random_state=seed + 1, dtype=np.float64
    )
    for m in (base_train, base_valid):
        m.indices = m.indices.astype(np.int32, copy=False)
        m.indptr = m.indptr.astype(np.int32, copy=False)

    full_train = sparse.hstack([base_train, view.train], format="csr")
    full_valid = sparse.hstack([base_valid, view.valid], format="csr")

    y = rng.integers(0, 2, size=n_train, dtype=np.int32)

    return (
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        full_train,
        full_valid,
    )


def check_semantic_vs_csr_tron_parity(
    base_train,
    C16t,
    semantic,
    y,
    base_valid,
    C16v,
    full_train,
    full_valid,
    C: float = 0.2,
    seed: int = 42,
):
    from cerm.training_graph import (
        solve_binary_logistic_path,
        solve_binary_logistic_semantic_blocks,
    )

    # Reference CSR TRON solver
    ref_res = solve_binary_logistic_path(
        full_train,
        y,
        full_valid,
        [C],
        random_state=seed,
        max_iter=1000,
    )[C]

    # Semantic block TRON solver
    act_res = solve_binary_logistic_semantic_blocks(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        C=C,
        random_state=seed,
        max_iter=1000,
    )

    coef_diff = np.abs(act_res.coefficient - ref_res.coefficient)
    max_abs_diff = float(np.max(coef_diff))
    denom = np.maximum(np.abs(ref_res.coefficient), 1e-12)
    max_rel_diff = float(np.max(coef_diff / denom))

    prob_diff = float(np.max(np.abs(act_res.valid_probability - ref_res.valid_probability)))

    passed = max_abs_diff <= 1e-3 and prob_diff <= 5e-4

    return {
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "valid_prob_max_diff": prob_diff,
        "reference_n_iter": int(ref_res.n_iter),
        "semantic_n_iter": int(act_res.n_iter),
        "tolerance": 1e-3,
    }


def run_semantic_tron_timing(
    base_train, C16t, semantic, y, base_valid, C16v, C: float = 0.2, seed: int = 42
):
    from cerm.training_graph import solve_binary_logistic_semantic_blocks

    return solve_binary_logistic_semantic_blocks(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        C=C,
        random_state=seed,
        max_iter=1000,
    )


def run_csr_tron_timing(full_train, y, full_valid, C: float = 0.2, seed: int = 42):
    from cerm.training_graph import solve_binary_logistic_path

    return solve_binary_logistic_path(
        full_train, y, full_valid, [C], random_state=seed, max_iter=1000
    )


# -----------------------------------------------------------------------------
# Benchmark Measurement Runner
# -----------------------------------------------------------------------------


def measure_kernel_performance(
    fn: Callable[[], Any],
    warmup: int = 2,
    repetitions: int = 5,
) -> Dict[str, float]:
    """Execute warmup and timed repetitions while capturing memory RSS."""
    gc.collect()
    rss_start = get_rss_bytes()

    # Warmup runs
    for _ in range(warmup):
        fn()

    times: List[float] = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    gc.collect()
    rss_end = get_rss_bytes()

    arr = np.array(times)
    return {
        "mean_sec": float(np.mean(arr)),
        "std_sec": float(np.std(arr)),
        "min_sec": float(np.min(arr)),
        "median_sec": float(np.median(arr)),
        "max_sec": float(np.max(arr)),
        "iterations": repetitions,
        "warmup": warmup,
        "rss_end_mb": float(rss_end / (1024 * 1024)),
        "rss_delta_mb": float((rss_end - rss_start) / (1024 * 1024)),
    }


# -----------------------------------------------------------------------------
# Main Benchmark Suite Runner
# -----------------------------------------------------------------------------


def run_benchmark_suite(
    scales: List[str],
    kernels: List[str],
    warmup: int = 2,
    repetitions: int = 5,
    seed: int = 42,
    n_threads: int = 1,
) -> Dict[str, Any]:
    """Execute the full benchmark suite across selected scales and kernels."""
    env_meta = collect_environment_metadata()
    results: List[Dict[str, Any]] = []

    all_kernel_keys = [
        "state_histogram",
        "quotient_pair_gain",
        "triad_histogram",
        "conditional_cross_gram",
        "semantic_vs_csr_tron",
    ]

    target_kernels = (
        all_kernel_keys if "all" in kernels else [k for k in kernels if k in all_kernel_keys]
    )

    for scale in scales:
        # 1. State Histogram
        if "state_histogram" in target_kernels:
            print(f"--> Running state_histogram [{scale}]...")
            states, cards, stats = generate_state_histogram_workload(scale, seed)
            parity = check_state_histogram_parity(states, cards, stats, n_threads=n_threads)
            timing = measure_kernel_performance(
                lambda: run_state_histogram_timing(states, cards, stats, n_threads=n_threads),
                warmup=warmup,
                repetitions=repetitions,
            )
            results.append(
                {
                    "kernel": "state_histogram",
                    "scale": scale,
                    "workload_params": {
                        "n_rows": len(states),
                        "n_cols": int(states.shape[1]),
                        "n_stats": int(stats.shape[1]),
                    },
                    "parity": parity,
                    "timing": timing,
                }
            )

        # 2. Quotient Pair Gain
        if "quotient_pair_gain" in target_kernels:
            print(f"--> Running quotient_pair_gain [{scale}]...")
            states, cards, coarse_maps, pairs, values6 = generate_quotient_pair_gain_workload(
                scale, seed
            )
            parity = check_quotient_pair_gain_parity(
                states, cards, coarse_maps, pairs, values6, n_threads=n_threads
            )
            timing = measure_kernel_performance(
                lambda: run_quotient_pair_gain_timing(
                    states, cards, coarse_maps, pairs, values6, n_threads=n_threads
                ),
                warmup=warmup,
                repetitions=repetitions,
            )
            results.append(
                {
                    "kernel": "quotient_pair_gain",
                    "scale": scale,
                    "workload_params": {
                        "n_rows": len(states),
                        "n_cols": int(states.shape[1]),
                        "n_pairs": len(pairs),
                    },
                    "parity": parity,
                    "timing": timing,
                }
            )

        # 3. Triad Histogram
        if "triad_histogram" in target_kernels:
            print(f"--> Running triad_histogram [{scale}]...")
            states_q4, triads, stats = generate_triad_histogram_workload(scale, seed)
            parity = check_triad_histogram_parity(
                states_q4, triads, stats, n_threads=n_threads
            )
            timing = measure_kernel_performance(
                lambda: run_triad_histogram_timing(
                    states_q4, triads, stats, n_threads=n_threads
                ),
                warmup=warmup,
                repetitions=repetitions,
            )
            results.append(
                {
                    "kernel": "triad_histogram",
                    "scale": scale,
                    "workload_params": {
                        "n_rows": len(states_q4),
                        "n_triads": len(triads),
                        "n_stats": int(stats.shape[1]),
                    },
                    "parity": parity,
                    "timing": timing,
                }
            )

        # 4. Conditional Cross-Gram
        if "conditional_cross_gram" in target_kernels:
            print(f"--> Running conditional_cross_gram [{scale}]...")
            selected, candidates, h = generate_conditional_cross_gram_workload(scale, seed)
            parity = check_conditional_cross_gram_parity(selected, candidates, h)
            timing = measure_kernel_performance(
                lambda: run_conditional_cross_gram_timing(selected, candidates, h),
                warmup=warmup,
                repetitions=repetitions,
            )
            results.append(
                {
                    "kernel": "conditional_cross_gram",
                    "scale": scale,
                    "workload_params": {
                        "n_rows": len(h),
                        "n_selected": len(selected),
                        "n_candidates": len(candidates),
                    },
                    "parity": parity,
                    "timing": timing,
                }
            )

        # 5. Semantic vs CSR TRON
        if "semantic_vs_csr_tron" in target_kernels:
            print(f"--> Running semantic_vs_csr_tron [{scale}]...")
            (
                base_train,
                C16t,
                semantic,
                y,
                base_valid,
                C16v,
                full_train,
                full_valid,
            ) = generate_semantic_vs_csr_tron_workload(scale, seed)

            parity = check_semantic_vs_csr_tron_parity(
                base_train,
                C16t,
                semantic,
                y,
                base_valid,
                C16v,
                full_train,
                full_valid,
                seed=seed,
            )
            semantic_timing = measure_kernel_performance(
                lambda: run_semantic_tron_timing(
                    base_train, C16t, semantic, y, base_valid, C16v, seed=seed
                ),
                warmup=warmup,
                repetitions=repetitions,
            )
            csr_timing = measure_kernel_performance(
                lambda: run_csr_tron_timing(full_train, y, full_valid, seed=seed),
                warmup=warmup,
                repetitions=repetitions,
            )

            speedup = (
                csr_timing["median_sec"] / semantic_timing["median_sec"]
                if semantic_timing["median_sec"] > 0
                else 0.0
            )

            results.append(
                {
                    "kernel": "semantic_vs_csr_tron",
                    "scale": scale,
                    "workload_params": {
                        "n_train": len(y),
                        "n_valid": len(base_valid.indices) if hasattr(base_valid, "indices") else 0,
                        "n_terms": semantic.n_columns,
                    },
                    "parity": parity,
                    "timing": semantic_timing,
                    "csr_reference_timing": csr_timing,
                    "semantic_speedup_vs_csr": speedup,
                }
            )

    return {
        "environment": env_meta,
        "benchmarks": results,
    }


# -----------------------------------------------------------------------------
# Report Exporters & Formatters
# -----------------------------------------------------------------------------


def format_ascii_table(report: Dict[str, Any]) -> str:
    """Format benchmark results into a clean ASCII table."""
    lines = []
    env = report.get("environment", {})
    lines.append("=" * 88)
    lines.append(" CERM NATIVE KERNELS RESEARCH BENCHMARK REPORT")
    lines.append("=" * 88)
    lines.append(f" Git Commit  : {env.get('git_commit', 'unknown')} ({env.get('git_branch', 'unknown')})")
    lines.append(f" Platform    : {env.get('platform', 'unknown')} | Python {env.get('python_version', 'unknown')}")
    lines.append(f" NumPy/SciPy : {env.get('numpy_version', '?')} / {env.get('scipy_version', '?')}")
    lines.append(f" Native Core : ABI v{env.get('native_core_abi_version', '?')} ({env.get('native_core_compiler', '?')})")
    lines.append("-" * 88)
    header = f"{'Kernel':<24} {'Scale':<8} {'Parity':<8} {'Median (s)':<12} {'Mean (s)':<12} {'Std (s)':<10} {'RSS (MB)':<10}"
    lines.append(header)
    lines.append("-" * 88)

    for item in report.get("benchmarks", []):
        name = item["kernel"]
        scale = item["scale"]
        passed = "PASS" if item["parity"]["passed"] else "FAIL"
        timing = item["timing"]
        med = f"{timing['median_sec']:.5f}"
        mn = f"{timing['mean_sec']:.5f}"
        st = f"{timing['std_sec']:.5f}"
        rss = f"{timing['rss_end_mb']:.1f}"
        lines.append(f"{name:<24} {scale:<8} {passed:<8} {med:<12} {mn:<12} {st:<10} {rss:<10}")

        if "semantic_speedup_vs_csr" in item:
            csr_med = item["csr_reference_timing"]["median_sec"]
            sp = item["semantic_speedup_vs_csr"]
            lines.append(
                f"  └─ CSR Reference TRON: median={csr_med:.5f}s | Semantic Speedup: {sp:.2f}x"
            )

    lines.append("=" * 88)
    return "\n".join(lines)


def export_json(report: Dict[str, Any], filepath: str) -> None:
    """Export benchmark report to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def export_csv(report: Dict[str, Any], filepath: str) -> None:
    """Export benchmark summary metrics to CSV."""
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kernel",
                "scale",
                "parity_passed",
                "max_abs_diff",
                "median_sec",
                "mean_sec",
                "std_sec",
                "min_sec",
                "max_sec",
                "rss_end_mb",
                "git_commit",
            ]
        )
        commit = report.get("environment", {}).get("git_commit", "unknown")
        for item in report.get("benchmarks", []):
            t = item["timing"]
            p = item["parity"]
            writer.writerow(
                [
                    item["kernel"],
                    item["scale"],
                    p["passed"],
                    p["max_abs_diff"],
                    t["median_sec"],
                    t["mean_sec"],
                    t["std_sec"],
                    t["min_sec"],
                    t["max_sec"],
                    t["rss_end_mb"],
                    commit,
                ]
            )


def compare_reports(file1: str, file2: str) -> str:
    """Compare two benchmark JSON report files side-by-side."""
    with open(file1, "r", encoding="utf-8") as f:
        r1 = json.load(f)
    with open(file2, "r", encoding="utf-8") as f:
        r2 = json.load(f)

    e1, e2 = r1.get("environment", {}), r2.get("environment", {})
    lines = []
    lines.append("=" * 90)
    lines.append(" CERM NATIVE KERNELS COMPARISON REPORT")
    lines.append("=" * 90)
    lines.append(f" File 1 (Baseline) : {file1} | Commit: {e1.get('git_commit', '?')[:8]}")
    lines.append(f" File 2 (Target)   : {file2} | Commit: {e2.get('git_commit', '?')[:8]}")
    lines.append("-" * 90)

    header = f"{'Kernel':<24} {'Scale':<8} {'Base Median(s)':<15} {'Tgt Median(s)':<15} {'Speedup':<10} {'Parity1/2':<10}"
    lines.append(header)
    lines.append("-" * 90)

    b1_map = {(b["kernel"], b["scale"]): b for b in r1.get("benchmarks", [])}
    b2_map = {(b["kernel"], b["scale"]): b for b in r2.get("benchmarks", [])}

    all_keys = list(dict.fromkeys(list(b1_map.keys()) + list(b2_map.keys())))

    for k, scale in all_keys:
        item1 = b1_map.get((k, scale))
        item2 = b2_map.get((k, scale))

        if not item1 or not item2:
            lines.append(f"{k:<24} {scale:<8} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'N/A':<10}")
            continue

        t1 = item1["timing"]["median_sec"]
        t2 = item2["timing"]["median_sec"]
        p1 = "PASS" if item1["parity"]["passed"] else "FAIL"
        p2 = "PASS" if item2["parity"]["passed"] else "FAIL"

        speedup = t1 / t2 if t2 > 0 else 0.0
        sp_str = f"{speedup:.2f}x"
        lines.append(
            f"{k:<24} {scale:<8} {t1:<15.5f} {t2:<15.5f} {sp_str:<10} {p1+'/'+p2:<10}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CERM Native Kernels Research Benchmark Harness"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "all"],
        default="all",
        help="Workload scale size to benchmark (default: all)",
    )
    parser.add_argument(
        "--kernel",
        choices=[
            "all",
            "state_histogram",
            "quotient_pair_gain",
            "triad_histogram",
            "conditional_cross_gram",
            "semantic_vs_csr_tron",
        ],
        default="all",
        help="Kernel to benchmark (default: all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timed repetitions per kernel (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of untimed warmup iterations per kernel (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset generation (default: 42)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count for multithreaded native kernels (default: 1)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full JSON report",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save CSV report",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare two benchmark JSON report files side-by-side",
    )

    args = parser.parse_args()

    if args.compare:
        print(compare_reports(args.compare[0], args.compare[1]))
        return

    scales = ["small", "medium"] if args.scale == "all" else [args.scale]
    kernels = [args.kernel]

    print("Starting CERM Native Kernels Research Benchmark...")
    report = run_benchmark_suite(
        scales=scales,
        kernels=kernels,
        warmup=args.warmup,
        repetitions=args.iterations,
        seed=args.seed,
        n_threads=args.threads,
    )

    print("\n" + format_ascii_table(report))

    if args.output_json:
        export_json(report, args.output_json)
        print(f"Saved JSON report to {args.output_json}")

    if args.output_csv:
        export_csv(report, args.output_csv)
        print(f"Saved CSV report to {args.output_csv}")


if __name__ == "__main__":
    main()
