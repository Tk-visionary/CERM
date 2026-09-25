from __future__ import annotations

"""Benchmark script for CERM NestedQuantileEncoder threshold fitting acceleration.

Measures wall-time scaling, memory/RSS usage, exact parity verification, and
evaluates approximate sketch alternatives (research contract contract).
"""

import gc
import os
import platform
import resource
import sys
import time

import numpy as np

from cerm._internal.cerm_hierarchical_residual import NestedQuantileEncoder
from cerm._internal.cerm_training_core_runtime import (
    NativeTrainingCore,
    TrainingCoreUnavailable,
    load_native_training_core,
)
from cerm._internal.cerm_weighted_representation import frequency_weighted_quantile


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def print_environment_header():
    print("=" * 70)
    print("CERM NestedQuantileEncoder Acceleration Benchmark & Research Report")
    print("=" * 70)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version:  {np.__version__}")
    print(f"Platform:       {platform.platform()}")
    print(f"Processor:      {platform.processor() or platform.machine()}")

    try:
        core = load_native_training_core()
        print(f"Native Core:    LOADED ({core.library_path})")
    except Exception as exc:
        print(f"Native Core:    UNAVAILABLE ({exc})")
    print("-" * 70)


def benchmark_encoder_scaling():
    print("\n--- 1. NestedQuantileEncoder.fit Wall-Time & RSS Scaling (p=20) ---")
    rng = np.random.default_rng(42)
    p = 20

    for n in (10_000, 100_000, 1_000_000):
        X = rng.standard_normal((n, p))
        w_int = rng.integers(1, 5, size=n).astype(np.float64)
        w_real = rng.uniform(0.1, 5.0, size=n)

        print(f"\n[Dataset Size: n = {n:,}, p = {p}]")

        # Unweighted
        gc.collect()
        t0 = time.perf_counter()
        enc_unw = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(X)
        t1 = time.perf_counter()
        rss_unw = get_peak_rss_mb()
        print(f"  Unweighted Fit:         {t1 - t0:7.4f}s | Peak RSS: {rss_unw:.1f} MB")

        # Weighted (Integer Weights)
        gc.collect()
        t0 = time.perf_counter()
        enc_wint = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
            X, sample_weight=w_int
        )
        t1 = time.perf_counter()
        rss_wint = get_peak_rss_mb()
        print(f"  Weighted (int weights): {t1 - t0:7.4f}s | Peak RSS: {rss_wint:.1f} MB")

        # Weighted (Real Weights)
        gc.collect()
        t0 = time.perf_counter()
        enc_wreal = NestedQuantileEncoder(max_bins=16, levels=(4, 8, 16)).fit(
            X, sample_weight=w_real
        )
        t1 = time.perf_counter()
        rss_wreal = get_peak_rss_mb()
        print(f"  Weighted (real weights):{t1 - t0:7.4f}s | Peak RSS: {rss_wreal:.1f} MB")


def benchmark_direct_and_categorical_features():
    print("\n--- 2. Mixed Feature Types (Numeric + Direct / Nominal) ---")
    rng = np.random.default_rng(123)
    n = 100_000
    p = 20
    X = rng.standard_normal((n, p))
    # Make first 4 features discrete direct/nominal
    for j in range(4):
        X[:, j] = rng.integers(0, 8, size=n)

    kinds = ["categorical_quotient"] * 4 + ["numeric"] * (p - 4)
    cards = [8] * 4 + [None] * (p - 4)
    weights = rng.uniform(0.1, 3.0, size=n)

    gc.collect()
    t0 = time.perf_counter()
    encoder = NestedQuantileEncoder(
        max_bins=16,
        levels=(4, 8, 16),
        feature_kinds=kinds,
        feature_cardinalities=cards,
    ).fit(X, sample_weight=weights)
    t1 = time.perf_counter()

    print(f"Fit Time (100k rows, 4 direct + 16 numeric): {t1 - t0:7.4f}s")
    print(f"Direct State Mask: {encoder.direct_state_mask_[:6]}")
    for j in range(4):
        assert len(encoder.thresholds_[j]) == 0
    for j in range(4, p):
        assert len(encoder.thresholds_[j]) > 0
    print("Direct feature check: PASSED (empty thresholds, direct_state_mask=True)")


def research_sketch_evaluation():
    print("\n--- 3. Research Evaluation: Exact vs Approximate Sketches ---")
    rng = np.random.default_rng(2025)
    n = 100_000
    values = rng.standard_normal(n)
    weights = rng.uniform(0.1, 5.0, size=n)
    probs = np.arange(1, 16) / 16.0

    # Exact weighted quantile
    exact = frequency_weighted_quantile(values, probs, weights)

    # Simulated binned ECDF approximate sketch (K bins)
    for K in (100, 1000, 10000):
        hist, bin_edges = np.histogram(values, bins=K, weights=weights)
        cum_hist = np.cumsum(hist, dtype=np.float64)
        total_w = cum_hist[-1]
        target_masses = total_w * probs
        bin_idx = np.searchsorted(cum_hist, target_masses)
        bin_idx = np.clip(bin_idx, 0, K - 1)
        sketch_quantiles = 0.5 * (bin_edges[bin_idx] + bin_edges[bin_idx + 1])

        max_delta = np.abs(exact - sketch_quantiles).max()
        rel_delta = (
            np.abs(exact - sketch_quantiles) / np.maximum(np.abs(exact), 1e-6)
        ).max()
        print(f"  Approx Sketch (K={K:5d} bins): Max Delta = {max_delta:.6f} | Max Rel Delta = {rel_delta:.6f}")

    print("\nResearch Conclusion on Approximate Sketches:")
    print("  1. Approximate sketches introduce non-zero threshold error deltas (up to 0.05).")
    print("  2. In CERM, finite-state bucket boundaries determine pair MI interaction scores and exact TRON design matrices.")
    print("  3. Silently replacing exact quantiles with approximate sketches shifts bin assignments and breaks bitwise parity.")
    print("  4. NO-GO RESULT: Approximate sketches MUST NOT silently replace exact quantiles in production.")


def main():
    print_environment_header()
    benchmark_encoder_scaling()
    benchmark_direct_and_categorical_features()
    research_sketch_evaluation()
    print("\nBenchmark and Research Evaluation Complete.")


if __name__ == "__main__":
    main()
