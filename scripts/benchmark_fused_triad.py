from math import comb
import time
import numpy as np
from scipy.stats import chi2

from cerm._experimental_undr_geometry import logistic_gh
from cerm.experimental_undr_v21 import (
    UNDRConfig,
    _native_fused_triad_stage1,
    _native_triad_histogram_bank,
    _vectorized_native_triad_stage1,
    fit_frozen_undr,
)
from cerm._internal.cerm_training_core_runtime import load_native_training_core
from tests.test_undr_v21_integration import make, split, MockEstimator


def run_benchmark(n_samples: int, d_features: int, seeds: int = 5):
    print(f"=== Benchmark Case: n_samples={n_samples}, d_features={d_features} (triads={comb(d_features, 3)}) ===")

    # 1. Direct Kernel Benchmark with realistic logistic gradients and Hessians
    rng = np.random.default_rng(42)
    S4 = rng.integers(0, 4, size=(n_samples, d_features), dtype=np.uint8)
    y = rng.integers(0, 2, size=n_samples, dtype=np.int32)
    score = rng.normal(scale=1.2, size=n_samples)
    g, h = logistic_gh(y, score)

    triads = np.asarray(
        [
            (a, b, c)
            for a in range(d_features)
            for b in range(a + 1, d_features)
            for c in range(b + 1, d_features)
        ],
        dtype=np.int32,
    )
    critical = float(chi2.ppf(1.0 - 0.001, 27))

    # Warmup
    _native_fused_triad_stage1(S4, g, h, triads, critical)
    _native_triad_histogram_bank(S4, g, h, triads)

    # Benchmark Fused Path
    t0 = time.perf_counter()
    for _ in range(seeds):
        fused_res = _native_fused_triad_stage1(S4, g, h, triads, critical)
    t_fused = (time.perf_counter() - t0) / seeds

    # Benchmark Unfused Path
    t0 = time.perf_counter()
    for _ in range(seeds):
        bank, threads, bytes_unfused = _native_triad_histogram_bank(S4, g, h, triads)
        clear_reject, ambiguous, cheap = _vectorized_native_triad_stage1(bank, critical)
    t_unfused = (time.perf_counter() - t0) / seeds

    flags, cheap_fused, bank_fused, threads_fused, bytes_fused = fused_res

    fused_rejects = np.count_nonzero((flags & 1) != 0)
    unfused_rejects = np.count_nonzero(clear_reject)
    survivors = len(triads) - fused_rejects

    print("Direct Kernel Wall Time:")
    print(f"  Unfused Native : {t_unfused * 1000:.3f} ms (Bank Memory Traffic: {bytes_unfused / 1024:.1f} KiB)")
    print(f"  Fused Native   : {t_fused * 1000:.3f} ms (Selective Bank Traffic: {bytes_fused / 1024:.1f} KiB)")
    print(f"  Speedup        : {t_unfused / t_fused:.2f}x")
    print(f"  Memory Reduction: {(1.0 - bytes_fused / max(bytes_unfused, 1)) * 100:.1f}%")
    print(f"  Survivor Parity : Total Triads = {len(triads)}, Fused Rejects = {fused_rejects}, Unfused Rejects = {unfused_rejects} (Exact Match: {np.array_equal((flags & 1) != 0, clear_reject)})\n")

    # 2. End-to-End UNDR Fit Benchmark
    X, y = make("triad", 123, n=n_samples + 2000, d=d_features, strength=1.4)
    Xt, yt, Xh, yh = split(X, y)
    cfg = UNDRConfig(maximum_features=d_features, budget=2, maximum_stage2_triads=16)
    est = MockEstimator(Xt)

    t0 = time.perf_counter()
    for _ in range(seeds):
        prog_fused = fit_frozen_undr(est, Xt, yt, Xh, yh, config=cfg)
    t_undr_fused = (time.perf_counter() - t0) / seeds

    # Run unfused by disabling fused symbol temporarily
    core = load_native_training_core()
    saved = core._triad_fused_stage1
    core._triad_fused_stage1 = None
    try:
        t0 = time.perf_counter()
        for _ in range(seeds):
            prog_unfused = fit_frozen_undr(est, Xt, yt, Xh, yh, config=cfg)
        t_undr_unfused = (time.perf_counter() - t0) / seeds
    finally:
        core._triad_fused_stage1 = saved

    print("End-to-End UNDR Fit Wall Time:")
    print(f"  Unfused UNDR Fit : {t_undr_unfused * 1000:.3f} ms")
    print(f"  Fused UNDR Fit   : {t_undr_fused * 1000:.3f} ms")
    print(f"  Speedup          : {t_undr_unfused / t_undr_fused:.2f}x")
    print(f"  Selected Names Match: {prog_fused.audit.selected_names == prog_unfused.audit.selected_names} ({prog_fused.audit.selected_names})")
    print(f"  Holdout Gain Match  : {prog_fused.audit.holdout_gain:.6f} vs {prog_unfused.audit.holdout_gain:.6f}\n\n")


if __name__ == "__main__":
    import platform
    print(f"Runtime Context: Python {platform.python_version()} on {platform.machine()} {platform.system()}")
    run_benchmark(n_samples=3000, d_features=8, seeds=10)
    run_benchmark(n_samples=20000, d_features=12, seeds=5)
    run_benchmark(n_samples=50000, d_features=16, seeds=5)
