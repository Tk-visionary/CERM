from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

from cerm import CERMClassifier
from cerm._compat import sklearn as sklearn_compat
from cerm._internal import cerm_training_core_runtime as runtime


def _reference_liblinear(
    design,
    target,
    *,
    C: float,
    seed: int,
    max_iter: int,
    sample_weight=None,
):
    try:
        from sklearn.svm import _liblinear
    except (ImportError, AttributeError):
        pytest.skip("scikit-learn private LIBLINEAR extension unavailable")

    matrix = sparse.csr_matrix(design, dtype=np.float64, copy=False)
    y = np.require(target, dtype=np.float64, requirements="W").ravel()
    if sample_weight is None:
        weights = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    _liblinear.set_verbosity_wrap(0)
    return _liblinear.train_wrap(
        matrix,
        y,
        True,
        0,
        1e-4,
        1.0,
        float(C),
        np.ones(2, dtype=np.float64),
        int(max_iter),
        int(seed),
        0.1,
        weights,
    )


def _native_core_or_skip():
    if not runtime.native_training_core_supported():
        pytest.skip("native training core compiler/platform unavailable")
    try:
        core = runtime.load_native_training_core()
    except RuntimeError as exc:
        pytest.skip(f"native training core could not compile/load: {exc}")
    assert core._abi_version() == runtime.ABI_VERSION == 2
    return core


def _unsorted_csr_with_stored_zeros(seed: int = 20260819):
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(360, 64))
    dense[rng.random(dense.shape) > 0.12] = 0.0
    matrix = sparse.csr_matrix(dense, dtype=np.float64)

    # Preserve the same mathematical matrix while exercising the exact storage
    # order accepted by sklearn/liblinear: unsorted column indices and explicit
    # stored zero values.
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row : row + 2]
        if stop - start > 1:
            order = np.arange(stop - start)[::-1]
            matrix.indices[start:stop] = matrix.indices[start:stop][order]
            matrix.data[start:stop] = matrix.data[start:stop][order]
    if matrix.nnz:
        matrix.data[::37] = 0.0
    matrix.has_sorted_indices = False
    return matrix


@pytest.mark.parametrize("C", [0.05, 0.2, 1.0, 5.0])
@pytest.mark.parametrize("weight_mode", ["none", "nonuniform", "zero_rows"])
def test_native_csr_logistic_matches_liblinear_bitwise(C, weight_mode):
    rng = np.random.default_rng(20260819)
    design = _unsorted_csr_with_stored_zeros()
    target = rng.integers(0, 2, size=design.shape[0], dtype=np.int64)
    target[:4] = np.array([0, 1, 0, 1])

    if weight_mode == "none":
        sample_weight = None
    else:
        sample_weight = 0.25 + 1.75 * rng.random(design.shape[0])
        if weight_mode == "zero_rows":
            sample_weight[5::17] = 0.0
            sample_weight[:4] = 1.0

    seed = 314159
    max_iter = 100
    reference_coef, reference_iter = _reference_liblinear(
        design,
        target,
        C=C,
        seed=seed,
        max_iter=max_iter,
        sample_weight=sample_weight,
    )
    native_coef, native_iter = _native_core_or_skip().binary_logistic_csr(
        design,
        target,
        C=C,
        seed=seed,
        max_iter=max_iter,
        sample_weight=sample_weight,
    )

    np.testing.assert_array_equal(native_coef, reference_coef)
    np.testing.assert_array_equal(native_iter, reference_iter)


def test_small_sparse_source_only_path_does_not_compile_native(monkeypatch):
    rng = np.random.default_rng(29)
    design = sparse.random(
        240,
        35,
        density=0.15,
        format="csr",
        dtype=np.float64,
        random_state=29,
        data_rvs=lambda n: rng.normal(size=n),
    )
    target = rng.integers(0, 2, size=design.shape[0]).astype(np.float64)
    target[:2] = [0.0, 1.0]
    expected = _reference_liblinear(
        design, target, C=0.7, seed=123, max_iter=100
    )

    monkeypatch.setattr(
        sklearn_compat, "_native_training_core_ready_without_compile", lambda: False
    )

    def fail_if_called():
        raise AssertionError("small source-only sparse solve must not compile native")

    monkeypatch.setattr(sklearn_compat, "load_native_training_core", fail_if_called)
    actual = sklearn_compat.train_binary_liblinear(
        design, target, C=0.7, seed=123, max_iter=100
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_train_binary_liblinear_falls_back_exactly_when_native_solver_fails(monkeypatch):
    rng = np.random.default_rng(17)
    design = sparse.random(
        240,
        35,
        density=0.15,
        format="csr",
        dtype=np.float64,
        random_state=17,
        data_rvs=lambda n: rng.normal(size=n),
    )
    target = rng.integers(0, 2, size=design.shape[0])
    target[:2] = [0, 1]
    weights = 0.5 + rng.random(design.shape[0])

    expected = _reference_liblinear(
        design,
        target,
        C=0.7,
        seed=123,
        max_iter=100,
        sample_weight=weights,
    )

    def fail_load():
        raise runtime.TrainingCoreUnavailable("forced native solver failure")

    monkeypatch.setattr(
        sklearn_compat, "_native_training_core_ready_without_compile", lambda: True
    )
    monkeypatch.setattr(sklearn_compat, "load_native_training_core", fail_load)
    actual = sklearn_compat.train_binary_liblinear(
        design,
        np.asarray(target, dtype=np.float64),
        C=0.7,
        seed=123,
        max_iter=100,
        sample_weight=weights,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_dense_binary_liblinear_does_not_enter_native_csr_path(monkeypatch):
    rng = np.random.default_rng(23)
    design = rng.normal(size=(180, 20))
    target = rng.integers(0, 2, size=len(design)).astype(np.float64)
    target[:2] = [0.0, 1.0]

    def fail_if_called():
        raise AssertionError("dense designs must remain on the historical sklearn path")

    monkeypatch.setattr(sklearn_compat, "load_native_training_core", fail_if_called)
    actual = sklearn_compat.train_binary_liblinear(
        design,
        target,
        C=1.0,
        seed=9,
        max_iter=100,
    )

    from sklearn.svm import _liblinear

    expected = _liblinear.train_wrap(
        np.asarray(design, dtype=np.float64, order="C"),
        np.require(target, dtype=np.float64, requirements="W").ravel(),
        False,
        0,
        1e-4,
        1.0,
        1.0,
        np.ones(2, dtype=np.float64),
        100,
        9,
        0.1,
        np.ones(len(design), dtype=np.float64),
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_end_to_end_weighted_fit_matches_forced_liblinear_fallback_bitwise(monkeypatch):
    _native_core_or_skip()
    X, y = make_classification(
        n_samples=900,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=20260819,
    )
    X = np.asarray(X, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(20260819)
    sample_weight = 0.4 + 1.2 * rng.random(len(y))
    sample_weight[11::53] = 0.0

    params = dict(
        search_effort="balanced",
        state_detail="medium",
        feature_budget=12,
        interaction_search_features=10,
        interaction_budget=4,
        interaction_order=2,
        selection_fraction=1.0,
        feature_fraction=1.0,
        calibration="none",
        n_jobs=1,
        random_state=20260819,
    )

    with monkeypatch.context() as reference_patch:
        def fail_load():
            raise runtime.TrainingCoreUnavailable("forced reference fallback")

        reference_patch.setattr(
            sklearn_compat, "_native_training_core_ready_without_compile", lambda: True
        )
        reference_patch.setattr(sklearn_compat, "load_native_training_core", fail_load)
        reference = CERMClassifier(**params).fit(X, y, sample_weight=sample_weight)
        reference_probability = reference.predict_proba(X)
        reference_label = reference.predict(X)

    native = CERMClassifier(**params).fit(X, y, sample_weight=sample_weight)
    native_probability = native.predict_proba(X)

    np.testing.assert_array_equal(native_probability, reference_probability)
    np.testing.assert_array_equal(native.predict(X), reference_label)
    assert type(native.program_) is type(reference.program_)
