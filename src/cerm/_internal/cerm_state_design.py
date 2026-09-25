from __future__ import annotations

import numpy as np
from scipy import sparse


class ReferenceStateEncoder:
    """Sparse one-hot encoder for non-negative finite-state columns.

    State 0 is the shared reference state and is omitted. States not observed at
    fit time are ignored during transform, matching OneHotEncoder with
    ``handle_unknown='ignore', drop='first'`` for integer codes.
    """

    def __init__(self, cardinalities: np.ndarray | list[int] | None = None):
        self.fixed_cardinalities = (
            None
            if cardinalities is None
            else np.asarray(cardinalities, dtype=np.int64)
        )

    @staticmethod
    def _validate(codes: np.ndarray) -> np.ndarray:
        array = np.asarray(codes)
        if array.ndim != 2:
            raise ValueError("state code matrix must be two-dimensional")
        if not np.issubdtype(array.dtype, np.integer):
            rounded = np.rint(array)
            if not np.array_equal(array, rounded):
                raise ValueError("state codes must be integers")
            array = rounded.astype(np.int64)
        # Preserve compact integer state banks.  Offset arithmetic below widens
        # only the emitted column indices, not the full n x d code matrix.
        if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
            raise ValueError("state codes must be non-negative")
        return array

    def _fit_validated(self, array: np.ndarray) -> "ReferenceStateEncoder":
        observed = array.max(axis=0, initial=0) + 1
        if self.fixed_cardinalities is None:
            cardinalities = observed
        else:
            if len(self.fixed_cardinalities) != array.shape[1]:
                raise ValueError("fixed cardinality width mismatch")
            if np.any(self.fixed_cardinalities < observed):
                raise ValueError("fixed cardinality is smaller than observed state")
            cardinalities = self.fixed_cardinalities
        self.cardinalities_ = cardinalities.astype(np.int64, copy=True)
        self.offsets_ = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(np.maximum(self.cardinalities_ - 1, 0))]
        )
        self.n_features_in_ = array.shape[1]
        self.n_features_out_ = int(self.offsets_[-1])
        self.categories_ = [
            np.arange(int(cardinality), dtype=np.int64)
            for cardinality in self.cardinalities_
        ]
        self.drop_idx_ = np.zeros(self.n_features_in_, dtype=np.int64)
        return self

    def fit(self, codes: np.ndarray) -> "ReferenceStateEncoder":
        return self._fit_validated(self._validate(codes))

    def _transform_validated(self, array: np.ndarray) -> sparse.csr_matrix:
        if not hasattr(self, "cardinalities_"):
            raise RuntimeError("encoder is not fitted")
        if array.shape[1] != self.n_features_in_:
            raise ValueError("state code width mismatch")

        valid = (array > 0) & (array < self.cardinalities_[None, :])
        flat = np.flatnonzero(valid.ravel(order="C"))
        if flat.size == 0:
            return sparse.csr_matrix(
                (len(array), self.n_features_out_), dtype=np.float64
            )

        # ``flat`` is row-major, and feature offsets are monotone.  The output
        # entries are therefore already CSR-sorted; constructing COO first and
        # converting/sorting it is redundant.
        n_features = self.n_features_in_
        features = flat % n_features
        states = array.ravel(order="C")[flat]
        columns = (
            self.offsets_[features] + states.astype(np.int64, copy=False) - 1
        )
        counts = np.count_nonzero(valid, axis=1)
        indptr = np.empty(len(array) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, dtype=np.int64, out=indptr[1:])
        data = np.ones(flat.size, dtype=np.float64)
        return sparse.csr_matrix(
            (data, columns.astype(np.int64, copy=False), indptr),
            shape=(len(array), self.n_features_out_),
            dtype=np.float64,
        )

    def transform(self, codes: np.ndarray) -> sparse.csr_matrix:
        return self._transform_validated(self._validate(codes))

    def fit_transform(self, codes: np.ndarray) -> sparse.csr_matrix:
        array = self._validate(codes)
        self._fit_validated(array)
        return self._transform_validated(array)
