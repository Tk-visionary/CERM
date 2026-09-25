from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest

from cerm.training_cache import NewtonHistogramCache
from cerm._internal.cerm_training_core_runtime import native_training_core_supported


def _native_or_skip(states, y, prediction, **kwargs):
    if not native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        return NewtonHistogramCache(
            states,
            y,
            prediction,
            backend="native",
            n_jobs=4,
            **kwargs,
        )
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")


def test_native_newton_cache_matches_python_bitwise():
    rng = np.random.default_rng(20260818)
    states = rng.integers(0, 8, size=(2500, 14), dtype=np.int16)
    y = rng.integers(0, 2, size=len(states))
    prediction = np.clip(0.05 + 0.9 * rng.random(len(states)), 1e-6, 1 - 1e-6)

    reference = NewtonHistogramCache(
        states,
        y,
        prediction,
        feature_limit=12,
        backend="python",
    )
    native = _native_or_skip(
        states,
        y,
        prediction,
        feature_limit=12,
    )

    assert native.backend_ == "native"
    assert native.feature_names == reference.feature_names
    assert native.nbytes == reference.nbytes
    for left, right in zip(native.main, reference.main):
        np.testing.assert_array_equal(left.G, right.G)
        np.testing.assert_array_equal(left.H, right.H)
    assert tuple(native.pairs) == tuple(reference.pairs)
    for key in reference.pairs:
        np.testing.assert_array_equal(native.pairs[key].G, reference.pairs[key].G)
        np.testing.assert_array_equal(native.pairs[key].H, reference.pairs[key].H)
    assert native.rank_pairs(20, 5.0) == reference.rank_pairs(20, 5.0)


def test_auto_backend_falls_back_exactly_when_native_load_fails(monkeypatch):
    from cerm._internal import cerm_training_core_runtime as runtime

    rng = np.random.default_rng(20260819)
    states = rng.integers(0, 8, size=(700, 9), dtype=np.int16)
    y = rng.integers(0, 2, size=len(states))
    prediction = np.clip(0.1 + 0.8 * rng.random(len(states)), 1e-6, 1 - 1e-6)
    reference = NewtonHistogramCache(states, y, prediction, backend="python")

    def fail_load():
        raise runtime.TrainingCoreUnavailable("forced native load failure")

    monkeypatch.setattr(runtime, "load_native_training_core", fail_load)
    fallback = NewtonHistogramCache(states, y, prediction, backend="auto")

    assert fallback.backend_ == "python"
    assert "forced native load failure" in fallback.native_error_
    for left, right in zip(fallback.main, reference.main):
        np.testing.assert_array_equal(left.G, right.G)
        np.testing.assert_array_equal(left.H, right.H)
    assert tuple(fallback.pairs) == tuple(reference.pairs)
    for key in reference.pairs:
        np.testing.assert_array_equal(fallback.pairs[key].G, reference.pairs[key].G)
        np.testing.assert_array_equal(fallback.pairs[key].H, reference.pairs[key].H)
    assert fallback.rank_pairs(20, 5.0) == reference.rank_pairs(20, 5.0)


def test_packaged_prebuilt_is_preferred_over_source_build(monkeypatch):
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime._source_build_supported():
        pytest.skip("source-build training core unavailable")
    compiled = runtime._compile_cached_library()
    runtime.load_native_training_core.cache_clear()
    monkeypatch.setattr(runtime, "_prebuilt_library_path", lambda: compiled)

    def fail_compile():
        raise AssertionError("source build must not run when a prebuilt core exists")

    monkeypatch.setattr(runtime, "_compile_cached_library", fail_compile)
    core = runtime.load_native_training_core()
    assert core.library_path == compiled
    assert core._abi_version() == runtime.ABI_VERSION
    runtime.load_native_training_core.cache_clear()


def test_auto_backend_falls_back_for_uint16_states():
    states = np.array(
        [[0, 0], [1, 256], [2, 300], [3, 1]],
        dtype=np.int64,
    )
    y = np.array([0, 1, 0, 1])
    cache = NewtonHistogramCache(states, y, backend="auto")
    assert cache.states.array.dtype == np.uint16
    assert cache.backend_ == "python"
    assert cache.native_error_ is None


def test_native_backend_rejects_uint16_states():
    states = np.array([[0], [256], [300]], dtype=np.int64)
    y = np.array([0, 1, 0])
    with pytest.raises(NotImplementedError, match="uint8"):
        NewtonHistogramCache(states, y, backend="native")


def test_invalid_training_cache_backend_is_rejected():
    states = np.array([[0], [1]], dtype=np.int16)
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="backend"):
        NewtonHistogramCache(states, y, backend="other")


def test_linux_ci_has_a_native_compiler_when_running_supported_path():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux CI contract only")
    assert shutil.which("g++") or shutil.which("clang++")



def test_native_quotient_pair_histogram_matches_fine_collapse_bitwise():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    rng = np.random.default_rng(20260925)
    n, d = 1800, 9
    cards = rng.integers(9, 17, size=d, dtype=np.int64)
    states = np.empty((n, d), dtype=np.uint8)
    maps = []
    for j, card in enumerate(cards):
        states[:, j] = rng.integers(0, int(card), size=n, dtype=np.uint8)
        fine = np.arange(int(card), dtype=np.int64)
        mapping = np.minimum(
            np.floor(fine.astype(float) * min(4, int(card)) / int(card)).astype(
                np.int64
            ),
            min(4, int(card)) - 1,
        )
        maps.append(mapping)

    pairs = np.asarray(
        [(j, k) for j in range(d) for k in range(j + 1, d)],
        dtype=np.int32,
    )
    values = rng.normal(size=(n, 4)).astype(np.float64)

    fine_offsets, fine_sums = core.pair_histogram(
        states,
        cards,
        pairs,
        values,
        n_threads=4,
    )
    expected = []
    for pair_index, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        left_card = int(cards[left])
        right_card = int(cards[right])
        start = int(fine_offsets[pair_index])
        stop = int(fine_offsets[pair_index + 1])
        table = fine_sums[start:stop].reshape(left_card, right_card, 4)

        left_gate = int(maps[left].max(initial=0)) + 1
        forward = np.zeros((left_gate, right_card, 4), dtype=np.float64)
        for fine_state, coarse_state in enumerate(maps[left]):
            forward[int(coarse_state)] += table[fine_state]
        expected.append(forward.reshape(-1, 4))

        right_gate = int(maps[right].max(initial=0)) + 1
        reverse = np.zeros((right_gate, left_card, 4), dtype=np.float64)
        transposed = table.transpose(1, 0, 2)
        for fine_state, coarse_state in enumerate(maps[right]):
            reverse[int(coarse_state)] += transposed[fine_state]
        expected.append(reverse.reshape(-1, 4))

    expected = np.concatenate(expected, axis=0)
    quotient_offsets, quotient_sums = core.quotient_pair_histogram(
        states,
        cards,
        maps,
        pairs,
        values,
        n_threads=4,
    )

    assert int(quotient_offsets[-1]) == len(expected)
    np.testing.assert_array_equal(quotient_sums, expected)


def test_optional_quotient_symbol_has_explicit_fallback_error():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    saved = core._quotient_pair_histogram
    core._quotient_pair_histogram = None
    try:
        with pytest.raises(
            runtime.TrainingCoreUnavailable,
            match="quotient_pair_histogram",
        ):
            core.quotient_pair_histogram(
                np.zeros((4, 2), dtype=np.uint8),
                np.asarray([1, 1]),
                (np.asarray([0]), np.asarray([0])),
                np.asarray([[0, 1]], dtype=np.int32),
                np.ones((4, 1)),
                n_threads=1,
            )
    finally:
        core._quotient_pair_histogram = saved



def test_native_quotient_pair_gains_match_python_directional_evaluation():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    rng = np.random.default_rng(20260926)
    n, d = 2400, 8
    cards = rng.integers(9, 17, size=d, dtype=np.int64)
    states = np.empty((n, d), dtype=np.uint8)
    maps = []
    for j, card in enumerate(cards):
        states[:, j] = rng.integers(0, int(card), size=n, dtype=np.uint8)
        fine = np.arange(int(card), dtype=np.int64)
        target = min(4, int(card))
        mapping = np.minimum(
            np.floor(fine.astype(float) * target / int(card)).astype(np.int64),
            target - 1,
        )
        maps.append(mapping)

    y = rng.integers(0, 2, size=n, dtype=np.int64)
    prediction = np.clip(
        0.05 + 0.9 * rng.random(n),
        1e-6,
        1.0 - 1e-6,
    )
    weights = np.clip(np.exp(rng.normal(0.0, 0.8, size=n)), 1e-2, 1e2)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    fold_a = np.concatenate([idx0[::2], idx1[::2]])
    fold_b = np.concatenate([idx0[1::2], idx1[1::2]])
    rows = np.concatenate([fold_a, fold_b])
    split = len(fold_a)

    states_sub = np.ascontiguousarray(states[rows])
    p_sub = prediction[rows]
    w_sub = weights[rows]
    g = (y[rows].astype(np.float64) - p_sub) * w_sub
    h = np.maximum(p_sub * (1.0 - p_sub), 1e-8) * w_sub
    values6 = np.zeros((len(rows), 6), dtype=np.float64)
    values6[:split, 0] = w_sub[:split]
    values6[:split, 1] = g[:split]
    values6[:split, 2] = h[:split]
    values6[split:, 3] = w_sub[split:]
    values6[split:, 4] = g[split:]
    values6[split:, 5] = h[split:]

    pairs = np.asarray(
        [(j, k) for j in range(d) for k in range(j + 1, d)],
        dtype=np.int32,
    )
    gain_l2 = 5.0
    min_hessian = 1.0
    min_a = max(10, int(np.ceil(0.025 * len(fold_a) * 2)))
    min_b = max(10, int(np.ceil(0.025 * len(fold_b) * 2)))
    min_full = max(20, int(np.ceil(0.05 * len(rows))))

    stat_offsets, directional = core.quotient_pair_histogram(
        states_sub,
        cards,
        maps,
        pairs,
        values6,
        n_threads=4,
    )
    candidate_offsets, full, gain_a, gain_b, flags = core.quotient_pair_gains(
        states_sub,
        cards,
        maps,
        pairs,
        values6,
        n_threads=4,
        gain_l2=gain_l2,
        min_hessian=min_hessian,
        min_support_a=min_a,
        min_support_b=min_b,
        min_support_full=min_full,
    )

    reference_full = np.zeros_like(full)
    reference_a = np.zeros_like(gain_a)
    reference_b = np.zeros_like(gain_b)
    reference_flags = np.zeros_like(flags)

    for pair_index, (left, right) in enumerate(pairs):
        left = int(left)
        right = int(right)
        left_gate = int(maps[left].max(initial=0)) + 1
        right_gate = int(maps[right].max(initial=0)) + 1
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

            active_a = table[:, :, 2] > 1e-12
            active_b = table[:, :, 5] > 1e-12
            grad_full = table[:, :, 1] + table[:, :, 4]
            hess_full_cells = table[:, :, 2] + table[:, :, 5]
            active_full = hess_full_cells > 1e-12

            child_a = np.sum(
                np.where(
                    active_a,
                    table[:, :, 1] ** 2 / (table[:, :, 2] + gain_l2),
                    0.0,
                ),
                axis=1,
            )
            child_b = np.sum(
                np.where(
                    active_b,
                    table[:, :, 4] ** 2 / (table[:, :, 5] + gain_l2),
                    0.0,
                ),
                axis=1,
            )
            child_full = np.sum(
                np.where(
                    active_full,
                    grad_full ** 2 / (hess_full_cells + gain_l2),
                    0.0,
                ),
                axis=1,
            )

            valid_a = (
                (mass_a >= min_a)
                & (hess_a >= min_hessian)
                & (active_a.sum(axis=1) > 1)
            )
            valid_b = (
                (mass_b >= min_b)
                & (hess_b >= min_hessian)
                & (active_b.sum(axis=1) > 1)
            )
            valid_full = (
                (mass_a + mass_b >= min_full)
                & (hess_a + hess_b >= min_hessian)
                & (active_full.sum(axis=1) > 1)
            )

            size = len(mass_a)
            sl = slice(position, position + size)
            reference_a[sl] = 0.5 * np.maximum(
                0.0,
                child_a - grad_a * grad_a / (hess_a + gain_l2),
            )
            reference_b[sl] = 0.5 * np.maximum(
                0.0,
                child_b - grad_b * grad_b / (hess_b + gain_l2),
            )
            reference_full[sl] = 0.5 * np.maximum(
                0.0,
                child_full
                - (grad_a + grad_b) ** 2
                / (hess_a + hess_b + gain_l2),
            )
            reference_flags[sl] = (
                valid_a.astype(np.uint8)
                | (valid_b.astype(np.uint8) << 1)
                | (valid_full.astype(np.uint8) << 2)
            )
            position += size

    np.testing.assert_array_equal(flags, reference_flags)
    np.testing.assert_allclose(full, reference_full, atol=5e-13, rtol=1e-12)
    np.testing.assert_allclose(gain_a, reference_a, atol=5e-13, rtol=1e-12)
    np.testing.assert_allclose(gain_b, reference_b, atol=5e-13, rtol=1e-12)


def test_optional_quotient_gain_symbol_has_explicit_fallback_error():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    saved = core._quotient_pair_gain
    core._quotient_pair_gain = None
    try:
        with pytest.raises(
            runtime.TrainingCoreUnavailable,
            match="quotient_pair_gain",
        ):
            core.quotient_pair_gains(
                np.zeros((4, 2), dtype=np.uint8),
                np.asarray([1, 1]),
                (np.asarray([0]), np.asarray([0])),
                np.asarray([[0, 1]], dtype=np.int32),
                np.zeros((4, 6), dtype=np.float64),
                n_threads=1,
                gain_l2=5.0,
                min_hessian=1.0,
                min_support_a=1.0,
                min_support_b=1.0,
                min_support_full=2.0,
            )
    finally:
        core._quotient_pair_gain = saved



def test_native_shared_ranking_extensions_preserve_abi_v2():
    from cerm._internal import cerm_training_core_runtime as runtime

    assert runtime.ABI_VERSION == 2
    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    assert core._abi_version() == runtime.ABI_VERSION
    assert core._quotient_pair_histogram is not None
    assert core._quotient_pair_gain is not None



def test_native_triad_histogram_matches_numpy_bitwise():
    from itertools import combinations
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    if core._triad_histogram is None:
        pytest.skip("native core lacks optional triad histogram extension")

    rng = np.random.default_rng(20260930)
    n, d = 2400, 9
    states = rng.integers(0, 4, size=(n, d), dtype=np.uint8)
    triads = np.asarray(
        list(combinations(range(d), 3)),
        dtype=np.int32,
    )
    values = rng.normal(size=(n, 2)).astype(np.float64)

    expected = np.empty((len(triads), 64, 2), dtype=np.float64)
    for index, (a, b, c) in enumerate(triads):
        code = (
            (states[:, a].astype(np.int64) * 4 + states[:, b]) * 4
            + states[:, c]
        )
        expected[index, :, 0] = np.bincount(
            code, weights=values[:, 0], minlength=64
        )
        expected[index, :, 1] = np.bincount(
            code, weights=values[:, 1], minlength=64
        )

    one = core.triad_histogram(
        states, triads, values, n_threads=1
    )
    four = core.triad_histogram(
        states, triads, values, n_threads=4
    )
    np.testing.assert_array_equal(one, expected)
    np.testing.assert_array_equal(four, expected)
    np.testing.assert_array_equal(one, four)


def test_optional_triad_histogram_symbol_has_explicit_fallback_error():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")

    saved = core._triad_histogram
    core._triad_histogram = None
    try:
        with pytest.raises(
            runtime.TrainingCoreUnavailable,
            match="triad_histogram",
        ):
            core.triad_histogram(
                np.zeros((4, 3), dtype=np.uint8),
                np.asarray([[0, 1, 2]], dtype=np.int32),
                np.ones((4, 2), dtype=np.float64),
                n_threads=1,
            )
    finally:
        core._triad_histogram = saved



def test_native_triad_histogram_rejects_precast_wraparound_states():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")
    if core._triad_histogram is None:
        pytest.skip("native core lacks optional triad histogram extension")

    triads = np.asarray([[0, 1, 2]], dtype=np.int32)
    values = np.ones((4, 2), dtype=np.float64)

    wrapped = np.zeros((4, 3), dtype=np.int64)
    wrapped[0, 0] = 256
    with pytest.raises(ValueError, match="q=4 state labels"):
        core.triad_histogram(
            wrapped, triads, values, n_threads=1
        )

    fractional = np.zeros((4, 3), dtype=np.float64)
    fractional[0, 0] = 1.5
    with pytest.raises(ValueError, match="integer q=4 state labels"):
        core.triad_histogram(
            fractional, triads, values, n_threads=1
        )



def test_native_pair_bank_drives_exact_undr_cross_grams():
    from cerm import _experimental_undr_geometry as geom
    from cerm.experimental_undr_v21 import _native_conditional_cross_grams
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core unsupported")
    try:
        runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core unavailable: {exc}")

    rng = np.random.default_rng(20261221)
    n = 2400
    S16 = rng.integers(0, 16, size=(n, 6), dtype=np.uint8)
    score = rng.normal(scale=1.1, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    selected = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"s{feature}"
        )
        for feature in (0, 1)
    ]
    candidates = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"c{feature}"
        )
        for feature in (2, 3, 4, 5)
    ]

    actual = _native_conditional_cross_grams(selected, candidates, h)
    assert actual is not None
    for candidate_index, candidate in enumerate(candidates):
        expected = np.vstack(
            [
                geom.cross_gram(left, candidate, h)
                for left in selected
            ]
        )
        np.testing.assert_array_equal(
            actual[candidate_index],
            expected,
        )



def test_native_conditional_workspace_reuses_one_packed_pool_exactly():
    from cerm import _experimental_undr_geometry as geom
    from cerm.experimental_undr_v21 import (
        DeficitCandidate,
        _ConditionalPairWorkspace,
    )
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core unsupported")
    try:
        runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core unavailable: {exc}")

    rng = np.random.default_rng(20261301)
    n = 2600
    S16 = rng.integers(0, 16, size=(n, 6), dtype=np.uint8)
    score = rng.normal(scale=1.0, size=n)
    y = rng.integers(0, 2, size=n, dtype=np.int32)
    _, h = geom.logistic_gh(y, score)
    bases = [
        geom.resolution_basis(
            S16, y, score, feature, 8, 16, name=f"b{feature}"
        )
        for feature in range(6)
    ]
    candidates = [
        DeficitCandidate(
            family="resolution",
            name=f"b{feature}",
            p_ref=1e-6,
            deficit=1.0,
            df=int(bases[feature].df),
            feature=feature,
            raw_feature=feature,
            source_q=8,
            target_q=16,
        )
        for feature in range(6)
    ]

    workspace = _ConditionalPairWorkspace.build(
        candidates,
        bases,
        h,
    )
    assert workspace is not None
    assert workspace.packed_bytes == n * len(bases)

    actual = workspace.cross_grams(
        candidates[:2],
        bases[:2],
        candidates[2:],
        bases[2:],
    )
    assert actual is not None
    for candidate_index, candidate in enumerate(bases[2:]):
        expected = np.vstack(
            [
                geom.cross_gram(left, candidate, h)
                for left in bases[:2]
            ]
        )
        np.testing.assert_array_equal(
            actual[candidate_index],
            expected,
        )



def test_native_triad_fused_stage1_matches_unfused_vectorized():
    from itertools import combinations
    from cerm._internal import cerm_training_core_runtime as runtime
    from cerm.experimental_undr_v21 import _vectorized_native_triad_stage1

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")
    if getattr(core, "_triad_fused_stage1", None) is None:
        pytest.skip("native core lacks optional triad fused stage1 extension")

    rng = np.random.default_rng(20261021)
    n, d = 3000, 8
    states = rng.integers(0, 4, size=(n, d), dtype=np.uint8)
    triads = np.asarray(list(combinations(range(d), 3)), dtype=np.int32)
    values = rng.normal(size=(n, 2)).astype(np.float64)
    critical = 40.11
    flags, cheap, bank = core.triad_fused_stage1(
        states, triads, values, critical, guard_rel=1e-10, n_threads=4
    )
    unfused_bank = core.triad_histogram(states, triads, values, n_threads=4)
    clear_reject, ambiguous, unfused_cheap = _vectorized_native_triad_stage1(
        unfused_bank, critical, guard_rel=1e-10
    )
    np.testing.assert_array_equal((flags & 1) != 0, clear_reject)
    np.testing.assert_array_equal((flags & 2) != 0, ambiguous)
    np.testing.assert_allclose(cheap, unfused_cheap, atol=1e-12, rtol=1e-12)
    for idx in np.flatnonzero((flags & 1) == 0):
        np.testing.assert_array_equal(bank[idx], unfused_bank[idx])


def test_optional_triad_fused_stage1_symbol_has_explicit_fallback_error():
    from cerm._internal import cerm_training_core_runtime as runtime

    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")
    saved = core._triad_fused_stage1
    core._triad_fused_stage1 = None
    try:
        with pytest.raises(runtime.TrainingCoreUnavailable, match="triad_fused_stage1"):
            core.triad_fused_stage1(
                np.zeros((4, 3), dtype=np.uint8),
                np.asarray([[0, 1, 2]], dtype=np.int32),
                np.ones((4, 2), dtype=np.float64),
                critical=40.0,
                n_threads=1,
            )
    finally:
        core._triad_fused_stage1 = saved
