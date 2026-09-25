from __future__ import annotations

import json
from pathlib import Path
import pytest

from cerm._internal.cerm_training_core_runtime import native_training_core_supported
from scripts.benchmark_native_kernels import (
    collect_environment_metadata,
    generate_state_histogram_workload,
    check_state_histogram_parity,
    generate_quotient_pair_gain_workload,
    check_quotient_pair_gain_parity,
    generate_triad_histogram_workload,
    check_triad_histogram_parity,
    generate_conditional_cross_gram_workload,
    check_conditional_cross_gram_parity,
    generate_semantic_vs_csr_tron_workload,
    check_semantic_vs_csr_tron_parity,
    measure_kernel_performance,
    run_benchmark_suite,
    format_ascii_table,
    export_json,
    export_csv,
    compare_reports,
)

requires_native_training_core = pytest.mark.skipif(
    not native_training_core_supported(),
    reason="native training core is unavailable on this platform",
)


def test_collect_environment_metadata_structure():
    meta = collect_environment_metadata()
    assert "python_version" in meta
    assert "platform" in meta
    assert "git_commit" in meta
    assert "numpy_version" in meta
    assert "scipy_version" in meta
    assert "env_vars" in meta


@requires_native_training_core
def test_state_histogram_generator_and_parity():
    states, cards, stats = generate_state_histogram_workload("small", seed=123)
    assert states.shape[0] == 10_000
    assert len(cards) == 20
    assert stats.shape == (10_000, 2)

    parity = check_state_histogram_parity(states, cards, stats, n_threads=1)
    assert parity["passed"]
    assert parity["max_abs_diff"] <= 1e-10


@requires_native_training_core
def test_quotient_pair_gain_generator_and_parity():
    states, cards, maps, pairs, values6 = generate_quotient_pair_gain_workload(
        "small", seed=123
    )
    assert states.shape == (10_000, 16)
    assert len(pairs) == 30
    assert values6.shape == (10_000, 6)

    parity = check_quotient_pair_gain_parity(
        states, cards, maps, pairs, values6, n_threads=1
    )
    assert parity["passed"]
    assert parity["max_abs_diff"] <= 1e-8


@requires_native_training_core
def test_triad_histogram_generator_and_parity():
    states_q4, triads, stats = generate_triad_histogram_workload("small", seed=123)
    assert states_q4.shape == (10_000, 12)
    assert len(triads) == 20
    assert stats.shape == (10_000, 2)

    parity = check_triad_histogram_parity(states_q4, triads, stats, n_threads=1)
    assert parity["passed"]
    assert parity["max_abs_diff"] <= 1e-10


@requires_native_training_core
def test_conditional_cross_gram_generator_and_parity():
    selected, candidates, h = generate_conditional_cross_gram_workload(
        "small", seed=123
    )
    assert len(selected) == 4
    assert len(candidates) == 12
    assert len(h) == 10_000

    parity = check_conditional_cross_gram_parity(selected, candidates, h)
    assert parity["passed"]
    assert parity["max_abs_diff"] <= 1e-10


@requires_native_training_core
def test_semantic_vs_csr_tron_generator_and_parity():
    (
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        full_train,
        full_valid,
    ) = generate_semantic_vs_csr_tron_workload("small", seed=123)

    assert base_train.shape[0] == 5_000
    assert base_valid.shape[0] == 1_000
    assert len(y) == 5_000

    parity = check_semantic_vs_csr_tron_parity(
        base_train,
        C16t,
        semantic,
        y,
        base_valid,
        C16v,
        full_train,
        full_valid,
        seed=123,
    )
    assert parity["passed"]
    assert parity["max_abs_diff"] <= 1e-3


def test_measure_kernel_performance_structure():
    counter = 0

    def dummy():
        nonlocal counter
        counter += 1

    res = measure_kernel_performance(dummy, warmup=1, repetitions=3)
    assert counter == 4
    assert res["iterations"] == 3
    assert res["warmup"] == 1
    assert "median_sec" in res
    assert "rss_end_mb" in res


@requires_native_training_core
def test_run_benchmark_suite_and_export(tmp_path: Path):
    report = run_benchmark_suite(
        scales=["small"],
        kernels=["state_histogram"],
        warmup=1,
        repetitions=1,
        seed=42,
        n_threads=1,
    )

    assert "environment" in report
    assert len(report["benchmarks"]) == 1
    assert report["benchmarks"][0]["kernel"] == "state_histogram"

    ascii_out = format_ascii_table(report)
    assert "CERM NATIVE KERNELS RESEARCH BENCHMARK REPORT" in ascii_out
    assert "state_histogram" in ascii_out

    json_path = str(tmp_path / "report.json")
    csv_path = str(tmp_path / "report.csv")

    export_json(report, json_path)
    export_csv(report, csv_path)

    assert Path(json_path).is_file()
    assert Path(csv_path).is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["benchmarks"][0]["kernel"] == "state_histogram"

    cmp_out = compare_reports(json_path, json_path)
    assert "CERM NATIVE KERNELS COMPARISON REPORT" in cmp_out
    assert "1.00x" in cmp_out
