from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


def load_native_predictor(
    library_path: str | Path,
    *,
    symbol: str = "cerm_graph_predict",
):
    """Load a CERM C ABI predictor from an existing shared library."""

    library_path = Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError(f"native CERM library not found: {library_path}")
    library = ctypes.CDLL(str(library_path))
    try:
        function = getattr(library, symbol)
    except AttributeError as exc:
        raise ValueError(
            f"native CERM symbol {symbol!r} is missing from {library_path}"
        ) from exc
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = None

    def predict(X):
        matrix = np.ascontiguousarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("X must be two-dimensional")
        output = np.empty(len(matrix), dtype=np.float64)
        function(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(matrix),
            matrix.shape[1],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    # Keep the CDLL alive through the closure.
    predict._cerm_library = library
    predict._cerm_symbol = symbol
    return predict


def load_native_matrix_predictor(
    library_path: str | Path,
    *,
    n_outputs: int,
    symbol: str = "cerm_shared_predict",
):
    """Load a C ABI predictor that emits a row-major output matrix."""

    library_path = Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError(f"native CERM library not found: {library_path}")
    if int(n_outputs) < 1:
        raise ValueError("n_outputs must be positive")
    library = ctypes.CDLL(str(library_path))
    try:
        function = getattr(library, symbol)
    except AttributeError as exc:
        raise ValueError(
            f"native CERM symbol {symbol!r} is missing from {library_path}"
        ) from exc
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = None

    def predict(X):
        matrix = np.ascontiguousarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("X must be two-dimensional")
        output = np.empty((len(matrix), int(n_outputs)), dtype=np.float64)
        function(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(matrix),
            matrix.shape[1],
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    predict._cerm_library = library
    predict._cerm_symbol = symbol
    predict._cerm_n_outputs = int(n_outputs)
    return predict
