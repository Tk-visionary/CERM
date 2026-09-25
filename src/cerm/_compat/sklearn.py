"""Small compatibility boundary for scikit-learn integration.

CERM primarily uses public scikit-learn APIs.  The optional optimized
LIBLINEAR path necessarily touches a private extension module; that access is
isolated here and callers must retain a public-estimator fallback.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from joblib import effective_n_jobs
from scipy import sparse

from .._internal.cerm_training_core_runtime import (
    TrainingCoreUnavailable,
    _prebuilt_library_path,
    load_native_training_core,
)


_NATIVE_LOGISTIC_SOURCE_BUILD_NNZ = 1_000_000


class PrivateSklearnAPIUnavailable(RuntimeError):
    """Raised when an optional private scikit-learn fast path is unavailable."""


def num_samples(value: Any) -> int:
    """Return the number of samples without relying on sklearn private helpers."""

    if isinstance(value, (str, bytes)):
        raise TypeError("Expected a collection or array-like object")
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            raise TypeError("CERM requires an input with a sample dimension")
        first = shape[0]
        if isinstance(first, (int, np.integer)):
            return int(first)
    if hasattr(value, "__array__"):
        array = np.asarray(value)
        if array.ndim == 0:
            raise TypeError("CERM requires an input with a sample dimension")
        return int(array.shape[0])
    try:
        return int(len(value))
    except TypeError as exc:
        raise TypeError("Expected a collection or array-like object") from exc


def safe_indexing(value: Any, indices: Any) -> Any:
    """Select rows from pandas, sparse, NumPy, or sequence inputs.

    Only the integer/boolean row-indexing modes used by CERM cross-validation
    are supported.  This intentionally avoids sklearn's private
    ``_safe_indexing`` API.
    """

    if hasattr(value, "iloc"):
        return value.iloc[indices]
    if sparse.issparse(value):
        return value[indices]
    if isinstance(value, np.ndarray):
        return value[indices]
    try:
        array = np.asarray(value)
        return array[indices]
    except (TypeError, ValueError, IndexError) as exc:
        raise TypeError("value does not support row indexing") from exc


def _native_training_core_ready_without_compile() -> bool:
    if load_native_training_core.cache_info().currsize:
        return True
    prebuilt = _prebuilt_library_path()
    return bool(prebuilt is not None and prebuilt.is_file())


def train_binary_liblinear(
    design: Any,
    target: np.ndarray,
    *,
    C: float,
    seed: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the exact binary LIBLINEAR problem, preferring CERM's CSR-native path.

    Packaged or already-loaded native cores are used for sparse designs without
    a workload threshold.  When a source compile would be required, CERM waits
    until a sufficiently large CSR workload to amortize that one-time cost.

    Returns
    -------
    raw_coef, n_iter
        Outputs from the exact native solver or the private sklearn extension.
        The caller retains a public ``LogisticRegression`` fallback.
    """

    if sparse.issparse(design):
        native_ready = _native_training_core_ready_without_compile()
        large_enough_to_compile = int(getattr(design, "nnz", 0)) >= _NATIVE_LOGISTIC_SOURCE_BUILD_NNZ
        if native_ready or large_enough_to_compile:
            try:
                return load_native_training_core().binary_logistic_csr(
                    design,
                    target,
                    C=float(C),
                    seed=int(seed),
                    max_iter=int(max_iter),
                    sample_weight=sample_weight,
                )
            except (
                TrainingCoreUnavailable,
                AttributeError,
                ImportError,
                TypeError,
                ValueError,
                OverflowError,
                OSError,
            ):
                pass

    try:
        from sklearn.svm import _liblinear
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version dependent
        raise PrivateSklearnAPIUnavailable(
            "scikit-learn's private LIBLINEAR extension is unavailable"
        ) from exc

    _liblinear.set_verbosity_wrap(0)
    is_sparse = bool(sparse.issparse(design))
    class_weight = np.ones(2, dtype=np.float64)
    if sample_weight is None:
        sample_weight_array = np.ones(int(design.shape[0]), dtype=np.float64)
    else:
        sample_weight_array = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if len(sample_weight_array) != int(design.shape[0]):
            raise ValueError("sample_weight length mismatch")
    try:
        return _liblinear.train_wrap(
            design,
            target,
            is_sparse,
            0,  # L2-regularized logistic regression, primal
            1e-4,
            1.0,
            float(C),
            class_weight,
            int(max_iter),
            int(seed),
            0.1,
            sample_weight_array,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PrivateSklearnAPIUnavailable(
            "scikit-learn's private LIBLINEAR ABI is incompatible"
        ) from exc


def fit_ridge_lsqr_alpha_path_exact(
    design: Any,
    target: np.ndarray,
    alphas: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    tol: float = 1e-4,
    max_iter: int | None = None,
    n_jobs: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an exact LSQR Ridge path while sharing invariant preprocessing.

    This is an optional optimization for the repeated-alpha validation path.
    Each LSQR solve receives the same centered/rescaled matrix, right-hand side,
    damping value and stopping criteria as an independent
    ``Ridge(solver="lsqr")`` fit.  The sparse transpose used by the implicit
    centering operator is materialized once instead of once per LSQR matvec.

    If the version-sensitive sklearn helpers are unavailable, the function
    falls back to independent public ``Ridge`` fits.
    """

    alpha_array = np.asarray(alphas, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if alpha_array.size == 0:
        return (
            np.empty((0, int(design.shape[1])), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
        )

    try:
        from scipy.sparse import linalg as sparse_linalg
        from sklearn.linear_model._base import _preprocess_data, _rescale_data
        from sklearn.utils.validation import _check_sample_weight
    except (ImportError, AttributeError):  # pragma: no cover - version dependent
        from sklearn.linear_model import Ridge

        models = [
            Ridge(
                alpha=float(alpha),
                solver="lsqr",
                tol=float(tol),
                max_iter=max_iter,
            ).fit(design, y, sample_weight=sample_weight)
            for alpha in alpha_array
        ]
        return (
            np.vstack([np.asarray(model.coef_, dtype=np.float64) for model in models]),
            np.asarray([float(model.intercept_) for model in models], dtype=np.float64),
            np.asarray(
                [int(np.asarray(model.n_iter_).reshape(-1)[0]) for model in models],
                dtype=np.int32,
            ),
        )

    X = design
    weights = (
        None
        if sample_weight is None
        else _check_sample_weight(sample_weight, X, dtype=X.dtype)
    )
    # ``_preprocess_data`` is private and changed shape between sklearn
    # releases.  Older versions accept ``rescale_with_sw`` and return a
    # sixth work-array; newer versions removed both.  The exact path performs
    # weighted rescaling explicitly below, so the newer five-value call is the
    # same preprocessing contract.
    try:
        processed = _preprocess_data(
            X,
            y,
            fit_intercept=True,
            copy=True,
            sample_weight=weights,
            rescale_with_sw=False,
        )
    except TypeError as exc:
        if "rescale_with_sw" not in str(exc):
            raise
        processed = _preprocess_data(
            X,
            y,
            fit_intercept=True,
            copy=True,
            sample_weight=weights,
        )
    if len(processed) == 6:
        X_processed, y_processed, X_offset, y_offset, X_scale, _ = processed
    elif len(processed) == 5:
        X_processed, y_processed, X_offset, y_offset, X_scale = processed
    else:  # pragma: no cover - defensive boundary for future sklearn changes
        raise PrivateSklearnAPIUnavailable(
            "unsupported sklearn _preprocess_data return shape"
        )
    if weights is None:
        X_rescaled = X_processed
        y_rescaled = y_processed
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)
    else:
        X_rescaled, y_rescaled, sample_weight_sqrt = _rescale_data(
            X_processed, y_processed, weights
        )

    offset_scale = X_offset / X_scale
    transpose = X_rescaled.T

    def matvec(vector):
        return X_rescaled.dot(vector) - sample_weight_sqrt * vector.dot(offset_scale)

    def rmatvec(vector):
        return transpose.dot(vector) - offset_scale * vector.dot(sample_weight_sqrt)

    operator = sparse_linalg.LinearOperator(
        shape=X_rescaled.shape,
        matvec=matvec,
        rmatvec=rmatvec,
    )
    coefficients = np.empty(
        (len(alpha_array), X_rescaled.shape[1]), dtype=X_rescaled.dtype
    )
    iterations = np.empty(len(alpha_array), dtype=np.int32)
    sqrt_alpha = np.sqrt(alpha_array)

    def solve_one(item):
        index, damping = item
        result = sparse_linalg.lsqr(
            operator,
            y_rescaled,
            damp=float(damping),
            atol=float(tol),
            btol=float(tol),
            iter_lim=max_iter,
        )
        return int(index), np.asarray(result[0]), int(result[2])

    workers = min(max(1, effective_n_jobs(n_jobs)), len(sqrt_alpha))
    items = list(enumerate(sqrt_alpha.tolist()))
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            solved = list(pool.map(solve_one, items))
    else:
        solved = [solve_one(item) for item in items]
    for index, coefficient, n_iter in solved:
        coefficients[index] = coefficient
        iterations[index] = n_iter

    coefficients = coefficients / X_scale
    # Compute each intercept separately.  This preserves the one-dimensional
    # dot-product reduction order used by independent Ridge fits.
    intercepts = np.asarray(
        [y_offset - X_offset @ coefficients[index] for index in range(len(alpha_array))],
        dtype=np.float64,
    )
    return coefficients, intercepts, iterations



def fit_multi_target_ridge_lsqr_exact(
    design: Any,
    target: np.ndarray,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one multi-target Ridge LSQR problem with shared sparse invariants.

    Private sklearn preprocessing is intentionally isolated in this compatibility
    module. Dense designs and incompatible sklearn versions retain an exact
    public-Ridge fallback.
    """
    from sklearn.linear_model import Ridge

    values = np.asarray(target, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if not sparse.issparse(design):
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="lsqr",
            max_iter=int(max_iter),
            tol=float(tol),
        ).fit(design, values)
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        return (
            coefficients,
            np.asarray(model.intercept_, dtype=np.float64).reshape(-1),
            np.asarray(model.n_iter_, dtype=np.int32).reshape(-1),
        )

    try:
        from scipy.sparse import linalg as sparse_linalg
        from sklearn.linear_model._base import _preprocess_data
    except (ImportError, AttributeError):  # pragma: no cover - version dependent
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="lsqr",
            max_iter=int(max_iter),
            tol=float(tol),
        ).fit(design, values)
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        return (
            coefficients,
            np.asarray(model.intercept_, dtype=np.float64).reshape(-1),
            np.asarray(model.n_iter_, dtype=np.int32).reshape(-1),
        )

    try:
        processed = _preprocess_data(
            design,
            values,
            fit_intercept=True,
            copy=True,
            sample_weight=None,
            rescale_with_sw=False,
        )
    except TypeError as exc:
        if "rescale_with_sw" not in str(exc):
            raise
        processed = _preprocess_data(
            design,
            values,
            fit_intercept=True,
            copy=True,
            sample_weight=None,
        )
    if len(processed) == 6:
        X_processed, y_processed, X_offset, y_offset, X_scale, _ = processed
    elif len(processed) == 5:
        X_processed, y_processed, X_offset, y_offset, X_scale = processed
    else:  # pragma: no cover - defensive boundary for future sklearn changes
        raise PrivateSklearnAPIUnavailable(
            "unsupported sklearn _preprocess_data return shape"
        )

    sample_weight_sqrt = np.ones(X_processed.shape[0], dtype=X_processed.dtype)
    offset_scale = X_offset / X_scale
    transpose = X_processed.T

    def matvec(vector):
        return X_processed.dot(vector) - sample_weight_sqrt * vector.dot(offset_scale)

    def rmatvec(vector):
        return transpose.dot(vector) - offset_scale * vector.dot(sample_weight_sqrt)

    operator = sparse_linalg.LinearOperator(
        shape=X_processed.shape,
        matvec=matvec,
        rmatvec=rmatvec,
    )
    coefficients = np.empty(
        (y_processed.shape[1], X_processed.shape[1]), dtype=np.float64
    )
    iterations = np.empty(y_processed.shape[1], dtype=np.int32)
    damping = float(np.sqrt(float(alpha)))
    for output in range(y_processed.shape[1]):
        result = sparse_linalg.lsqr(
            operator,
            y_processed[:, output],
            damp=damping,
            atol=float(tol),
            btol=float(tol),
            iter_lim=int(max_iter),
        )
        coefficients[output] = np.asarray(result[0])
        iterations[output] = int(result[2])
    coefficients = coefficients / X_scale
    intercepts = (
        np.asarray(y_offset, dtype=np.float64) - coefficients @ X_offset
    )
    return coefficients, intercepts, iterations
