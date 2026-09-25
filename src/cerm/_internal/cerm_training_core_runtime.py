from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading

import numpy as np

from .cerm_native_build import compile_shared_library

ABI_VERSION = 2
_BLAS_INIT_LOCK = threading.Lock()
_BLAS_CONFIGURED_LIBRARIES: set[str] = set()


class TrainingCoreUnavailable(RuntimeError):
    """Raised when the optional native training core cannot be used."""


class NativeTrainingCore:
    """ctypes wrapper around the exact native training-core ABI."""

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)
        self._library = ctypes.CDLL(str(self.library_path))
        self._blas_configured = False
        self._bind()
        version = int(self._abi_version())
        if version != ABI_VERSION:
            raise TrainingCoreUnavailable(
                f"training-core ABI mismatch: expected {ABI_VERSION}, got {version}"
            )

    def _bind(self) -> None:
        double_p = ctypes.POINTER(ctypes.c_double)
        int32_p = ctypes.POINTER(ctypes.c_int32)
        uint8_p = ctypes.POINTER(ctypes.c_uint8)

        self._abi_version = self._library.cerm_training_core_abi_version
        self._abi_version.argtypes = []
        self._abi_version.restype = ctypes.c_int

        self._state_histogram = self._library.cerm_state_histogram_mt_u8
        self._state_histogram.argtypes = [
            uint8_p,
            ctypes.c_int,
            ctypes.c_int,
            int32_p,
            ctypes.c_int,
            double_p,
            ctypes.c_int,
            double_p,
            ctypes.c_int,
            double_p,
            double_p,
        ]
        self._state_histogram.restype = ctypes.c_int

        self._pair_histogram = self._library.cerm_selected_pair_histogram_mt_u8
        self._pair_histogram.argtypes = [
            uint8_p,
            ctypes.c_int,
            ctypes.c_int,
            int32_p,
            ctypes.c_int,
            int32_p,
            ctypes.c_int,
            int32_p,
            int32_p,
            ctypes.c_int,
            double_p,
            ctypes.c_int,
            double_p,
            ctypes.c_int,
            double_p,
            double_p,
        ]
        self._pair_histogram.restype = ctypes.c_int

        try:
            self._triad_histogram = (
                self._library.cerm_selected_triad_histogram_mt_u8
            )
        except AttributeError:
            self._triad_histogram = None
        if self._triad_histogram is not None:
            self._triad_histogram.argtypes = [
                uint8_p,
                ctypes.c_int,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                ctypes.c_int,
                double_p,
            ]
            self._triad_histogram.restype = ctypes.c_int

        try:
            self._triad_fused_stage1 = (
                self._library.cerm_selected_triad_fused_stage1_mt_u8
            )
        except AttributeError:
            self._triad_fused_stage1 = None
        if self._triad_fused_stage1 is not None:
            self._triad_fused_stage1.argtypes = [
                uint8_p,
                ctypes.c_int,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                uint8_p,
                double_p,
                double_p,
            ]
            self._triad_fused_stage1.restype = ctypes.c_int

        try:
            self._quotient_pair_histogram = (
                self._library.cerm_selected_quotient_pair_histogram_mt_u8
            )
        except AttributeError:
            self._quotient_pair_histogram = None
        if self._quotient_pair_histogram is not None:
            self._quotient_pair_histogram.argtypes = [
                uint8_p,
                ctypes.c_int,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                uint8_p,
                int32_p,
                int32_p,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                double_p,
                double_p,
            ]
            self._quotient_pair_histogram.restype = ctypes.c_int

        try:
            self._quotient_pair_gain = (
                self._library.cerm_selected_quotient_pair_gain_mt_u8
            )
        except AttributeError:
            self._quotient_pair_gain = None
        if self._quotient_pair_gain is not None:
            self._quotient_pair_gain.argtypes = [
                uint8_p,
                ctypes.c_int,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                uint8_p,
                int32_p,
                int32_p,
                ctypes.c_int,
                int32_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                double_p,
                double_p,
                double_p,
                uint8_p,
            ]
            self._quotient_pair_gain.restype = ctypes.c_int

        self._set_blas = self._library.cerm_training_core_set_blas
        self._set_blas.argtypes = [ctypes.c_void_p] * 4
        self._set_blas.restype = ctypes.c_int

        self._binary_logistic_csr = self._library.cerm_liblinear_train_csr_direct_f64
        self._binary_logistic_csr.argtypes = [
            double_p,
            int32_p,
            int32_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            double_p,
            double_p,
            ctypes.c_double,
            ctypes.c_uint32,
            ctypes.c_int32,
            double_p,
            ctypes.c_int32,
            int32_p,
        ]
        self._binary_logistic_csr.restype = ctypes.c_int

        try:
            self._binary_logistic_semantic_blocks = (
                self._library.cerm_liblinear_train_csr_semantic_blocks_f64
            )
        except AttributeError:
            self._binary_logistic_semantic_blocks = None
        if self._binary_logistic_semantic_blocks is not None:
            self._binary_logistic_semantic_blocks.argtypes = [
                double_p,
                int32_p,
                int32_p,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                uint8_p,
                ctypes.c_int32,
                int32_p,
                int32_p,
                uint8_p,
                double_p,
                double_p,
                int32_p,
                int32_p,
                ctypes.c_int32,
                ctypes.c_int32,
                double_p,
                double_p,
                ctypes.c_double,
                ctypes.c_uint32,
                ctypes.c_int32,
                double_p,
                ctypes.c_int32,
                int32_p,
            ]
            self._binary_logistic_semantic_blocks.restype = ctypes.c_int

        try:
            self._weighted_quantile = self._library.cerm_weighted_quantile_f64
        except AttributeError:
            self._weighted_quantile = None
        if self._weighted_quantile is not None:
            self._weighted_quantile.argtypes = [
                double_p,
                double_p,
                ctypes.c_int,
                double_p,
                ctypes.c_int,
                double_p,
            ]
            self._weighted_quantile.restype = ctypes.c_int

    @staticmethod
    def _ptr(array: np.ndarray, ctype):
        return array.ctypes.data_as(ctypes.POINTER(ctype))

    @staticmethod
    def _check_status(status: int, operation: str) -> None:
        if int(status) != 0:
            raise TrainingCoreUnavailable(
                f"native training core {operation} failed with status {int(status)}"
            )

    @staticmethod
    def _cython_blas_pointer(name: str) -> int:
        try:
            from scipy.linalg import cython_blas

            capsule = cython_blas.__pyx_capi__[name]
        except (ImportError, AttributeError, KeyError) as exc:
            raise TrainingCoreUnavailable(
                f"SciPy Cython BLAS capsule {name!r} is unavailable"
            ) from exc
        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.argtypes = [ctypes.py_object]
        get_name.restype = ctypes.c_char_p
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        capsule_name = get_name(capsule)
        pointer = get_pointer(capsule, capsule_name)
        if not pointer:
            raise TrainingCoreUnavailable(
                f"SciPy Cython BLAS capsule {name!r} has no function pointer"
            )
        return int(pointer)

    def _ensure_blas(self) -> None:
        if self._blas_configured:
            return
        library_key = str(self.library_path.resolve())
        with _BLAS_INIT_LOCK:
            if library_key in _BLAS_CONFIGURED_LIBRARIES:
                self._blas_configured = True
                return
            pointers = [
                self._cython_blas_pointer(name)
                for name in ("ddot", "daxpy", "dscal", "dnrm2")
            ]
            status = self._set_blas(*(ctypes.c_void_p(pointer) for pointer in pointers))
            self._check_status(status, "set_blas")
            _BLAS_CONFIGURED_LIBRARIES.add(library_key)
            self._blas_configured = True

    def binary_logistic_csr(
        self,
        design,
        target: np.ndarray,
        *,
        C: float,
        seed: int,
        max_iter: int,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the package's exact sparse L2R_LR problem directly on CSR."""
        from scipy import sparse

        if not sparse.issparse(design):
            raise TypeError("native binary logistic solver requires a sparse design")
        matrix = sparse.csr_matrix(design, dtype=np.float64, copy=False)
        if matrix.indices.dtype != np.int32 or matrix.indptr.dtype != np.int32:
            raise ValueError("native binary logistic solver requires int32 CSR indices")
        y = np.ascontiguousarray(target, dtype=np.float64).reshape(-1)
        if len(y) != matrix.shape[0]:
            raise ValueError("binary target shape mismatch")
        if sample_weight is None:
            weights = np.ones(matrix.shape[0], dtype=np.float64)
        else:
            weights = np.ascontiguousarray(sample_weight, dtype=np.float64).reshape(-1)
            if len(weights) != matrix.shape[0]:
                raise ValueError("sample_weight length mismatch")
        coefficient = np.empty(matrix.shape[1] + 1, dtype=np.float64)
        n_iter = np.zeros(1, dtype=np.int32)
        self._ensure_blas()
        status = self._binary_logistic_csr(
            self._ptr(matrix.data, ctypes.c_double),
            self._ptr(matrix.indices, ctypes.c_int32),
            self._ptr(matrix.indptr, ctypes.c_int32),
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            self._ptr(y, ctypes.c_double),
            self._ptr(weights, ctypes.c_double),
            float(C),
            int(seed),
            int(max_iter),
            self._ptr(coefficient, ctypes.c_double),
            len(coefficient),
            self._ptr(n_iter, ctypes.c_int32),
        )
        self._check_status(status, "binary_logistic_csr")
        return coefficient.reshape(1, -1), n_iter


    def binary_logistic_semantic_blocks(
        self,
        base_design,
        target_states: np.ndarray,
        block_target: np.ndarray,
        block_coef_offsets: np.ndarray,
        block_state_ids: np.ndarray,
        block_centers: np.ndarray,
        block_scales: np.ndarray,
        block_row_offsets: np.ndarray,
        block_rows: np.ndarray,
        target: np.ndarray,
        *,
        C: float,
        seed: int,
        max_iter: int,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve binary L2-logistic regression with semantic block columns.

        The base design remains CSR. Conditional block columns are evaluated
        from finite-state rows and center/scale metadata inside the native TRON
        operator, avoiding a second materialized CSR suffix.
        """
        from scipy import sparse

        if self._binary_logistic_semantic_blocks is None:
            raise TrainingCoreUnavailable(
                "native training core lacks semantic-block logistic extension"
            )
        if not sparse.issparse(base_design):
            raise TypeError("semantic-block solver requires a sparse base design")
        matrix = sparse.csr_matrix(base_design, dtype=np.float64, copy=False)
        if matrix.indices.dtype != np.int32 or matrix.indptr.dtype != np.int32:
            raise ValueError("semantic-block solver requires int32 CSR indices")

        states = np.ascontiguousarray(target_states, dtype=np.uint8)
        if states.ndim != 2 or len(states) != matrix.shape[0]:
            raise ValueError("target_states must have shape (n_samples, n_states)")

        targets = np.ascontiguousarray(block_target, dtype=np.int32).reshape(-1)
        coef_offsets = np.ascontiguousarray(
            block_coef_offsets, dtype=np.int32
        ).reshape(-1)
        state_ids = np.ascontiguousarray(block_state_ids, dtype=np.uint8).reshape(-1)
        centers = np.ascontiguousarray(block_centers, dtype=np.float64).reshape(-1)
        scales = np.ascontiguousarray(block_scales, dtype=np.float64).reshape(-1)
        row_offsets = np.ascontiguousarray(
            block_row_offsets, dtype=np.int32
        ).reshape(-1)
        rows = np.ascontiguousarray(block_rows, dtype=np.int32).reshape(-1)

        n_blocks = len(targets)
        if len(coef_offsets) != n_blocks + 1 or len(row_offsets) != n_blocks + 1:
            raise ValueError("semantic block offsets must have length n_blocks + 1")
        if coef_offsets[0] != 0 or row_offsets[0] != 0:
            raise ValueError("semantic block offsets must start at zero")
        n_block_features = int(coef_offsets[-1])
        if (
            len(state_ids) != n_block_features
            or len(centers) != n_block_features
            or len(scales) != n_block_features
        ):
            raise ValueError("semantic block coefficient metadata mismatch")
        if int(row_offsets[-1]) != len(rows):
            raise ValueError("semantic block row metadata mismatch")
        if n_blocks and (
            np.any(targets < 0) or np.any(targets >= states.shape[1])
        ):
            raise ValueError("semantic block target column is out of range")
        if len(rows) and (np.any(rows < 0) or np.any(rows >= len(states))):
            raise ValueError("semantic block row is out of range")

        y = np.ascontiguousarray(target, dtype=np.float64).reshape(-1)
        if len(y) != matrix.shape[0]:
            raise ValueError("binary target shape mismatch")
        if sample_weight is None:
            weights = np.ones(matrix.shape[0], dtype=np.float64)
        else:
            weights = np.ascontiguousarray(
                sample_weight, dtype=np.float64
            ).reshape(-1)
            if len(weights) != matrix.shape[0]:
                raise ValueError("sample_weight length mismatch")

        coefficient = np.empty(
            matrix.shape[1] + n_block_features + 1, dtype=np.float64
        )
        n_iter = np.zeros(1, dtype=np.int32)
        self._ensure_blas()
        status = self._binary_logistic_semantic_blocks(
            self._ptr(matrix.data, ctypes.c_double),
            self._ptr(matrix.indices, ctypes.c_int32),
            self._ptr(matrix.indptr, ctypes.c_int32),
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            self._ptr(states, ctypes.c_uint8),
            states.shape[1],
            self._ptr(targets, ctypes.c_int32),
            self._ptr(coef_offsets, ctypes.c_int32),
            self._ptr(state_ids, ctypes.c_uint8),
            self._ptr(centers, ctypes.c_double),
            self._ptr(scales, ctypes.c_double),
            self._ptr(row_offsets, ctypes.c_int32),
            self._ptr(rows, ctypes.c_int32),
            n_blocks,
            n_block_features,
            self._ptr(y, ctypes.c_double),
            self._ptr(weights, ctypes.c_double),
            float(C),
            int(seed),
            int(max_iter),
            self._ptr(coefficient, ctypes.c_double),
            len(coefficient),
            self._ptr(n_iter, ctypes.c_int32),
        )
        self._check_status(status, "binary_logistic_semantic_blocks")
        return coefficient.reshape(1, -1), n_iter

    @staticmethod
    def _state_layout(states: np.ndarray, cardinalities: np.ndarray):
        array = np.ascontiguousarray(states, dtype=np.uint8)
        cards = np.ascontiguousarray(cardinalities, dtype=np.int32).reshape(-1)
        if array.ndim != 2 or len(cards) != array.shape[1]:
            raise ValueError("state/cardinality shape mismatch")
        if np.any(cards <= 0) or np.any(cards > 256):
            raise ValueError("native training core requires cardinalities in [1, 256]")
        maxima = array.max(axis=0, initial=0).astype(np.int64)
        if np.any(maxima >= cards):
            raise ValueError("state code exceeds configured cardinality")
        offsets = np.empty(len(cards) + 1, dtype=np.int32)
        offsets[0] = 0
        cumulative = np.cumsum(cards, dtype=np.int64)
        if cumulative[-1] > np.iinfo(np.int32).max:
            raise OverflowError("state bank exceeds int32 native ABI")
        offsets[1:] = cumulative.astype(np.int32)
        return array, cards, offsets

    def state_histogram(
        self,
        states: np.ndarray,
        cardinalities: np.ndarray,
        values: np.ndarray,
        *,
        n_threads: int,
    ):
        array, _cards, offsets = self._state_layout(states, cardinalities)
        stats = np.ascontiguousarray(values, dtype=np.float64)
        if stats.ndim == 1:
            stats = stats[:, None]
        if stats.ndim != 2 or len(stats) != len(array):
            raise ValueError("values must have shape (n_samples, n_values)")
        total_states = int(offsets[-1])
        mass = np.empty(total_states, dtype=np.float64)
        sums = np.empty((total_states, stats.shape[1]), dtype=np.float64)
        status = self._state_histogram(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(offsets, ctypes.c_int32),
            total_states,
            self._ptr(stats, ctypes.c_double),
            stats.shape[1],
            None,
            max(1, int(n_threads)),
            self._ptr(mass, ctypes.c_double),
            self._ptr(sums, ctypes.c_double),
        )
        self._check_status(status, "state_histogram")
        return offsets, sums

    def pair_histogram(
        self,
        states: np.ndarray,
        cardinalities: np.ndarray,
        pairs: np.ndarray,
        values: np.ndarray,
        *,
        n_threads: int,
    ):
        array, cards, state_offsets = self._state_layout(states, cardinalities)
        pair_array = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
        stats = np.ascontiguousarray(values, dtype=np.float64)
        if stats.ndim == 1:
            stats = stats[:, None]
        if stats.ndim != 2 or len(stats) != len(array):
            raise ValueError("values must have shape (n_samples, n_values)")
        if len(pair_array) == 0:
            return np.zeros(1, dtype=np.int32), np.empty((0, stats.shape[1]))
        if np.any(pair_array < 0) or np.any(pair_array >= array.shape[1]):
            raise ValueError("pair state column is out of range")

        right_cards = np.ascontiguousarray(cards[pair_array[:, 1]], dtype=np.int32)
        widths = cards[pair_array[:, 0]].astype(np.int64) * right_cards.astype(np.int64)
        cumulative = np.cumsum(widths, dtype=np.int64)
        if cumulative[-1] > np.iinfo(np.int32).max:
            raise OverflowError("pair state bank exceeds int32 native ABI")
        pair_offsets = np.empty(len(pair_array) + 1, dtype=np.int32)
        pair_offsets[0] = 0
        pair_offsets[1:] = cumulative.astype(np.int32)
        total_pair_states = int(pair_offsets[-1])
        mass = np.empty(total_pair_states, dtype=np.float64)
        sums = np.empty((total_pair_states, stats.shape[1]), dtype=np.float64)
        status = self._pair_histogram(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(state_offsets, ctypes.c_int32),
            int(state_offsets[-1]),
            self._ptr(pair_array, ctypes.c_int32),
            len(pair_array),
            self._ptr(pair_offsets, ctypes.c_int32),
            self._ptr(right_cards, ctypes.c_int32),
            total_pair_states,
            self._ptr(stats, ctypes.c_double),
            stats.shape[1],
            None,
            max(1, int(n_threads)),
            self._ptr(mass, ctypes.c_double),
            self._ptr(sums, ctypes.c_double),
        )
        self._check_status(status, "pair_histogram")
        return pair_offsets, sums

    def triad_histogram(
        self,
        states: np.ndarray,
        triads: np.ndarray,
        values: np.ndarray,
        *,
        n_threads: int,
    ) -> np.ndarray:
        """Aggregate exact q=4 triad statistics with row-order preservation."""
        if self._triad_histogram is None:
            raise TrainingCoreUnavailable(
                "native training core lacks triad_histogram extension"
            )
        source = np.asarray(states)
        if source.ndim != 2:
            raise ValueError("states must be a 2-D matrix")
        if not np.issubdtype(source.dtype, np.integer):
            if not np.issubdtype(source.dtype, np.floating):
                raise ValueError("triad_histogram requires integer q=4 state labels")
            if not np.isfinite(source).all() or not np.array_equal(
                source, np.rint(source)
            ):
                raise ValueError("triad_histogram requires integer q=4 state labels")
        if np.any((source < 0) | (source > 3)):
            raise ValueError("triad_histogram requires q=4 state labels")
        array = np.ascontiguousarray(source, dtype=np.uint8)
        triad_array = np.ascontiguousarray(
            triads, dtype=np.int32
        ).reshape(-1, 3)
        if len(triad_array):
            if np.any(triad_array < 0) or np.any(
                triad_array >= array.shape[1]
            ):
                raise ValueError("triad state column is out of range")
            if np.any(
                (triad_array[:, 0] == triad_array[:, 1])
                | (triad_array[:, 0] == triad_array[:, 2])
                | (triad_array[:, 1] == triad_array[:, 2])
            ):
                raise ValueError("triad columns must be distinct")

        stats = np.ascontiguousarray(values, dtype=np.float64)
        if stats.ndim == 1:
            stats = stats[:, None]
        if stats.ndim != 2 or len(stats) != len(array):
            raise ValueError("values must have shape (n_samples, n_values)")
        output = np.empty(
            (len(triad_array), 64, stats.shape[1]),
            dtype=np.float64,
        )
        if len(triad_array) == 0:
            return output

        status = self._triad_histogram(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(triad_array, ctypes.c_int32),
            len(triad_array),
            self._ptr(stats, ctypes.c_double),
            stats.shape[1],
            max(1, int(n_threads)),
            self._ptr(output, ctypes.c_double),
        )
        self._check_status(status, "triad_histogram")
        return output

    def triad_fused_stage1(
        self,
        states: np.ndarray,
        triads: np.ndarray,
        values: np.ndarray,
        critical: float,
        *,
        guard_rel: float = 1e-10,
        n_threads: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Stage-1 triad screening and selectively materialize G/H banks."""
        if self._triad_fused_stage1 is None:
            raise TrainingCoreUnavailable(
                "native training core lacks triad_fused_stage1 extension"
            )
        source = np.asarray(states)
        if source.ndim != 2:
            raise ValueError("states must be a 2-D matrix")
        if not np.issubdtype(source.dtype, np.integer):
            if not np.issubdtype(source.dtype, np.floating):
                raise ValueError("triad_fused_stage1 requires integer q=4 state labels")
            if not np.isfinite(source).all() or not np.array_equal(
                source, np.rint(source)
            ):
                raise ValueError("triad_fused_stage1 requires integer q=4 state labels")
        if np.any((source < 0) | (source > 3)):
            raise ValueError("triad_fused_stage1 requires q=4 state labels")
        array = np.ascontiguousarray(source, dtype=np.uint8)
        triad_array = np.ascontiguousarray(
            triads, dtype=np.int32
        ).reshape(-1, 3)
        if len(triad_array):
            if np.any(triad_array < 0) or np.any(
                triad_array >= array.shape[1]
            ):
                raise ValueError("triad state column is out of range")
            if np.any(
                (triad_array[:, 0] == triad_array[:, 1])
                | (triad_array[:, 0] == triad_array[:, 2])
                | (triad_array[:, 1] == triad_array[:, 2])
            ):
                raise ValueError("triad columns must be distinct")

        stats = np.ascontiguousarray(values, dtype=np.float64)
        if stats.ndim == 1:
            stats = stats[:, None]
        if stats.ndim != 2 or len(stats) != len(array) or stats.shape[1] < 2:
            raise ValueError("values must have shape (n_samples, 2+)")

        out_flags = np.empty(len(triad_array), dtype=np.uint8)
        out_cheap = np.empty(len(triad_array), dtype=np.float64)
        output = np.empty(
            (len(triad_array), 64, stats.shape[1]),
            dtype=np.float64,
        )
        if len(triad_array) == 0:
            return out_flags, out_cheap, output

        status = self._triad_fused_stage1(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(triad_array, ctypes.c_int32),
            len(triad_array),
            self._ptr(stats, ctypes.c_double),
            stats.shape[1],
            float(critical),
            float(guard_rel),
            max(1, int(n_threads)),
            self._ptr(out_flags, ctypes.c_uint8),
            self._ptr(out_cheap, ctypes.c_double),
            self._ptr(output, ctypes.c_double),
        )
        self._check_status(status, "triad_fused_stage1")
        return out_flags, out_cheap, output


    def quotient_pair_histogram(
        self,
        states: np.ndarray,
        cardinalities: np.ndarray,
        coarse_maps,
        pairs: np.ndarray,
        values: np.ndarray,
        *,
        n_threads: int,
    ):
        """Aggregate both coarse->fine directions from one fine pair histogram.

        This is an additive ABI-v2 extension. Older packaged cores remain
        loadable; callers should catch TrainingCoreUnavailable and fall back to
        pair_histogram + Python quotient aggregation when the optional symbol is
        absent.
        """
        if self._quotient_pair_histogram is None:
            raise TrainingCoreUnavailable(
                "native training core lacks quotient_pair_histogram extension"
            )
        array, cards, state_offsets = self._state_layout(states, cardinalities)
        pair_array = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
        stats = np.ascontiguousarray(values, dtype=np.float64)
        if stats.ndim == 1:
            stats = stats[:, None]
        if stats.ndim != 2 or len(stats) != len(array):
            raise ValueError("values must have shape (n_samples, n_values)")
        if np.any(pair_array < 0) or np.any(pair_array >= array.shape[1]):
            raise ValueError("pair state column is out of range")

        maps = tuple(np.asarray(mapping, dtype=np.int64).reshape(-1) for mapping in coarse_maps)
        if len(maps) != array.shape[1]:
            raise ValueError("coarse_maps length must match state width")
        flat_maps = np.empty(int(state_offsets[-1]), dtype=np.uint8)
        coarse_cards = np.empty(array.shape[1], dtype=np.int32)
        for slot, mapping in enumerate(maps):
            fine_card = int(cards[slot])
            if len(mapping) != fine_card:
                raise ValueError("coarse map cardinality mismatch")
            if np.any(mapping < 0) or np.any(mapping > 255):
                raise ValueError("coarse map values must be in [0, 255]")
            coarse_card = int(mapping.max(initial=0)) + 1
            if coarse_card < 1 or coarse_card > fine_card:
                raise ValueError("invalid coarse map cardinality")
            start, stop = int(state_offsets[slot]), int(state_offsets[slot + 1])
            flat_maps[start:stop] = mapping.astype(np.uint8, copy=False)
            coarse_cards[slot] = coarse_card

        if len(pair_array) == 0:
            return np.zeros(1, dtype=np.int32), np.empty((0, stats.shape[1]))

        left = pair_array[:, 0]
        right = pair_array[:, 1]
        widths = (
            coarse_cards[left].astype(np.int64) * cards[right].astype(np.int64)
            + coarse_cards[right].astype(np.int64) * cards[left].astype(np.int64)
        )
        cumulative = np.cumsum(widths, dtype=np.int64)
        if cumulative[-1] > np.iinfo(np.int32).max:
            raise OverflowError("quotient pair bank exceeds int32 native ABI")
        pair_offsets = np.empty(len(pair_array) + 1, dtype=np.int32)
        pair_offsets[0] = 0
        pair_offsets[1:] = cumulative.astype(np.int32)
        total_pair_states = int(pair_offsets[-1])
        mass = np.empty(total_pair_states, dtype=np.float64)
        sums = np.empty((total_pair_states, stats.shape[1]), dtype=np.float64)
        status = self._quotient_pair_histogram(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(state_offsets, ctypes.c_int32),
            int(state_offsets[-1]),
            self._ptr(flat_maps, ctypes.c_uint8),
            self._ptr(coarse_cards, ctypes.c_int32),
            self._ptr(pair_array, ctypes.c_int32),
            len(pair_array),
            self._ptr(pair_offsets, ctypes.c_int32),
            total_pair_states,
            self._ptr(stats, ctypes.c_double),
            stats.shape[1],
            None,
            max(1, int(n_threads)),
            self._ptr(mass, ctypes.c_double),
            self._ptr(sums, ctypes.c_double),
        )
        self._check_status(status, "quotient_pair_histogram")
        return pair_offsets, sums

    def quotient_pair_gains(
        self,
        states: np.ndarray,
        cardinalities: np.ndarray,
        coarse_maps,
        pairs: np.ndarray,
        fold_values6: np.ndarray,
        *,
        n_threads: int,
        gain_l2: float,
        min_hessian: float,
        min_support_a: float,
        min_support_b: float,
        min_support_full: float,
    ):
        """Evaluate full/stable Newton gains from nested pair statistics."""
        if self._quotient_pair_gain is None:
            raise TrainingCoreUnavailable(
                "native training core lacks quotient_pair_gain extension"
            )
        array, cards, state_offsets = self._state_layout(states, cardinalities)
        pair_array = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
        values = np.ascontiguousarray(fold_values6, dtype=np.float64)
        if values.shape != (len(array), 6):
            raise ValueError("fold_values6 must have shape (n_samples, 6)")
        if np.any(pair_array < 0) or np.any(pair_array >= array.shape[1]):
            raise ValueError("pair state column is out of range")

        maps = tuple(
            np.asarray(mapping, dtype=np.int64).reshape(-1)
            for mapping in coarse_maps
        )
        if len(maps) != array.shape[1]:
            raise ValueError("coarse_maps length must match state width")
        flat_maps = np.empty(int(state_offsets[-1]), dtype=np.uint8)
        coarse_cards = np.empty(array.shape[1], dtype=np.int32)
        for slot, mapping in enumerate(maps):
            fine_card = int(cards[slot])
            if len(mapping) != fine_card:
                raise ValueError("coarse map cardinality mismatch")
            if np.any(mapping < 0) or np.any(mapping > 255):
                raise ValueError("coarse map values must be in [0, 255]")
            coarse_card = int(mapping.max(initial=0)) + 1
            if coarse_card < 1 or coarse_card > fine_card:
                raise ValueError("invalid coarse map cardinality")
            start, stop = int(state_offsets[slot]), int(state_offsets[slot + 1])
            flat_maps[start:stop] = mapping.astype(np.uint8, copy=False)
            coarse_cards[slot] = coarse_card

        if len(pair_array) == 0:
            empty = np.empty(0, dtype=np.float64)
            return (
                np.zeros(1, dtype=np.int32),
                empty.copy(),
                empty.copy(),
                empty.copy(),
                np.empty(0, dtype=np.uint8),
            )

        widths = (
            coarse_cards[pair_array[:, 0]].astype(np.int64)
            + coarse_cards[pair_array[:, 1]].astype(np.int64)
        )
        cumulative = np.cumsum(widths, dtype=np.int64)
        if cumulative[-1] > np.iinfo(np.int32).max:
            raise OverflowError("quotient candidate bank exceeds int32 native ABI")
        candidate_offsets = np.empty(len(pair_array) + 1, dtype=np.int32)
        candidate_offsets[0] = 0
        candidate_offsets[1:] = cumulative.astype(np.int32)
        total_candidates = int(candidate_offsets[-1])
        full = np.empty(total_candidates, dtype=np.float64)
        gain_a = np.empty(total_candidates, dtype=np.float64)
        gain_b = np.empty(total_candidates, dtype=np.float64)
        flags = np.empty(total_candidates, dtype=np.uint8)

        status = self._quotient_pair_gain(
            self._ptr(array, ctypes.c_uint8),
            len(array),
            array.shape[1],
            self._ptr(state_offsets, ctypes.c_int32),
            int(state_offsets[-1]),
            self._ptr(flat_maps, ctypes.c_uint8),
            self._ptr(coarse_cards, ctypes.c_int32),
            self._ptr(pair_array, ctypes.c_int32),
            len(pair_array),
            self._ptr(candidate_offsets, ctypes.c_int32),
            total_candidates,
            self._ptr(values, ctypes.c_double),
            max(1, int(n_threads)),
            float(gain_l2),
            float(min_hessian),
            float(min_support_a),
            float(min_support_b),
            float(min_support_full),
            self._ptr(full, ctypes.c_double),
            self._ptr(gain_a, ctypes.c_double),
            self._ptr(gain_b, ctypes.c_double),
            self._ptr(flags, ctypes.c_uint8),
        )
        self._check_status(status, "quotient_pair_gains")
        return candidate_offsets, full, gain_a, gain_b, flags

    def weighted_quantile(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        quantiles: np.ndarray,
    ) -> np.ndarray:
        if self._weighted_quantile is None:
            raise TrainingCoreUnavailable(
                "native training core lacks weighted_quantile extension"
            )
        v = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
        w = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        q = np.ascontiguousarray(quantiles, dtype=np.float64).reshape(-1)
        if len(v) != len(w):
            raise ValueError("values and weights length mismatch")
        out = np.empty(len(q), dtype=np.float64)
        if len(v) == 0 or len(q) == 0:
            return out
        status = self._weighted_quantile(
            self._ptr(v, ctypes.c_double),
            self._ptr(w, ctypes.c_double),
            len(v),
            self._ptr(q, ctypes.c_double),
            len(q),
            self._ptr(out, ctypes.c_double),
        )
        self._check_status(status, "weighted_quantile")
        return out


def _source_path() -> Path:
    return Path(__file__).with_name("native") / "cerm_training_core.cpp"


def _platform_key() -> str | None:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        machine = "arm64"
    else:
        return None
    if sys.platform.startswith("linux"):
        return f"linux-{machine}"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    if sys.platform == "win32":
        return f"windows-{machine}"
    return None


def _prebuilt_library_path() -> Path | None:
    key = _platform_key()
    if key is None:
        return None
    if sys.platform == "darwin":
        filename = "libcerm_training_core.dylib"
    elif sys.platform == "win32":
        filename = "cerm_training_core.dll"
    else:
        filename = "libcerm_training_core.so"
    return _source_path().parent / "prebuilt" / key / filename


def _cache_root() -> Path:
    configured = os.environ.get("CERM_CACHE_DIR")
    if configured:
        return Path(configured).expanduser() / "training_core"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "cerm" / "training_core"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "CERM" / "training_core"
    return Path.home() / ".cache" / "cerm" / "training_core"


def _source_build_supported() -> bool:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        return False
    return _source_path().is_file() and bool(shutil.which("g++") or shutil.which("clang++"))


def native_training_core_supported() -> bool:
    """Return whether a packaged or source-built training core is usable."""

    prebuilt = _prebuilt_library_path()
    return bool(prebuilt is not None and prebuilt.is_file()) or _source_build_supported()


def _compiler() -> str:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise TrainingCoreUnavailable("no C++17 compiler found for native training core")
    return compiler


def _cached_library_path() -> Path:
    source = _source_path()
    if not source.is_file():
        raise TrainingCoreUnavailable(f"native training-core source is missing: {source}")
    compiler = _compiler()
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[:1]
    identity = "\n".join(
        [
            hashlib.sha256(source.read_bytes()).hexdigest(),
            platform.system(),
            platform.machine(),
            compiler,
            *(version or [""]),
            "-O2",
        ]
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    directory = _cache_root() / key
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"libcerm_training_core{suffix}"


def _compile_cached_library() -> Path:
    if not _source_build_supported():
        raise TrainingCoreUnavailable("native training-core source build is unsupported")
    target = _cached_library_path()
    if target.is_file():
        return target
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        compile_shared_library(
            _source_path(),
            temporary,
            compiler=_compiler(),
            optimization="-O2",
            native_arch=False,
            extra_flags=("-pthread",),
        )
        os.replace(temporary, target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if target.is_file():
            return target
        if isinstance(exc, TrainingCoreUnavailable):
            raise
        raise TrainingCoreUnavailable(str(exc)) from exc
    return target


@lru_cache(maxsize=1)
def load_native_training_core() -> NativeTrainingCore:
    """Prefer a packaged core, otherwise compile once into the user cache."""

    prebuilt = _prebuilt_library_path()
    if prebuilt is not None and prebuilt.is_file():
        try:
            return NativeTrainingCore(prebuilt)
        except Exception:
            if not _source_build_supported():
                raise
    return NativeTrainingCore(_compile_cached_library())
