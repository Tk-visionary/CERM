from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal, Sequence

try:
    from enum import StrEnum
except ImportError:  # Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal stdlib-compatible fallback for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .cerm_hybrid_quotient_block import HybridQuotientBlockCERM


class FeatureKind(StrEnum):
    NUMERIC = "numeric"
    MISSING = "missing_state"
    CATEGORY_IDENTITY = "categorical_identity"
    CATEGORY_IDENTITY_BIT = "categorical_identity_bit"
    CATEGORY_QUOTIENT = "categorical_quotient"
    EMBEDDING_QUOTIENT = "embedding_quotient"
    EMBEDDING_RAW = "embedding_raw"


@dataclass(frozen=True)
class TypedAdapterOutput:
    matrix: np.ndarray
    feature_names: list[str]
    feature_kinds: list[str]
    cardinalities: list[int | None]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_rows: int | None = None) -> "TypedAdapterOutput":
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise ValueError("adapter matrix must be two-dimensional")
        if expected_rows is not None and matrix.shape[0] != expected_rows:
            raise ValueError(f"row mismatch: {matrix.shape[0]} != {expected_rows}")
        width = matrix.shape[1]
        if not (
            len(self.feature_names)
            == len(self.feature_kinds)
            == len(self.cardinalities)
            == width
        ):
            raise ValueError("adapter metadata width does not match matrix width")
        if not np.isfinite(matrix).all():
            raise ValueError("adapter matrix contains non-finite values")
        return self


def _as_binary_target(y: np.ndarray) -> np.ndarray:
    target = np.asarray(y)
    classes = np.unique(target)
    if len(classes) != 2:
        raise ValueError(f"binary target required; got classes={classes!r}")
    if np.array_equal(classes, np.asarray([0, 1])):
        return target.astype(np.int8, copy=False)
    return (target == classes[1]).astype(np.int8)


def _stratified_cv(y: np.ndarray, requested_splits: int, random_state: int):
    counts = np.bincount(_as_binary_target(y), minlength=2)
    n_splits = min(int(requested_splits), int(counts.min()))
    if n_splits < 2:
        raise ValueError("at least two observations per class are required for OOF states")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )


def _quantile_thresholds(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size <= 1 or int(n_bins) <= 1:
        return np.empty(0, dtype=np.float64)
    # If the first n_bins+1 finite values are already distinct, the full
    # column is provably above the identity-state threshold.  Skip the
    # historical full-array unique sort in that common continuous case; if
    # the prefix is inconclusive, fall back to the exact full computation.
    unique = None
    if finite.size > int(n_bins):
        prefix = np.unique(finite[: int(n_bins) + 1])
        if prefix.size <= int(n_bins):
            unique = np.unique(finite)
    else:
        unique = np.unique(finite)
    if unique is not None and unique.size <= n_bins:
        return ((unique[:-1] + unique[1:]) * 0.5).astype(np.float64)
    probabilities = np.arange(1, int(n_bins), dtype=np.float64) / int(n_bins)
    thresholds = np.unique(np.quantile(finite, probabilities))
    return thresholds[(thresholds > finite.min()) & (thresholds < finite.max())]


def _apply_thresholds(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        side="right",
    ).astype(np.int16)


def _stable_category_sort(values: Sequence[Any]) -> list[Any]:
    return sorted(values, key=lambda value: (type(value).__name__, repr(value)))


def _numeric_series_values(values: pd.Series) -> np.ndarray:
    """Exact fast path for already-numeric pandas columns."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        try:
            return values.to_numpy(dtype=np.float64, copy=False, na_value=np.nan)
        except (TypeError, ValueError):
            pass
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


class IdentityCategoricalEncoder:
    """Exact finite-state encoder with an optional missing/unseen state.

    Known categories start at state zero.  When a missing state is reserved,
    missing and unseen values map to the final state; otherwise they map to the
    reference category at state zero.
    """

    def __init__(self, reserve_missing: bool = False):
        self.reserve_missing = bool(reserve_missing)

    def fit(self, values: pd.Series) -> "IdentityCategoricalEncoder":
        series = values.astype("object")
        unique = [value for value in pd.unique(series) if not pd.isna(value)]
        self.mapping_ = {
            value: index
            for index, value in enumerate(_stable_category_sort(unique))
        }
        self.has_missing_ = bool(series.isna().any())
        self.has_reserved_missing_ = self.reserve_missing or self.has_missing_
        self.missing_state_ = len(self.mapping_) if self.has_reserved_missing_ else 0
        self.cardinality_ = len(self.mapping_) + int(self.has_reserved_missing_)
        self.cardinality_ = max(self.cardinality_, 1)
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        fallback = self.missing_state_ if self.has_reserved_missing_ else 0
        categories = list(self.mapping_.keys())
        if categories:
            codes = pd.Categorical(values.astype("object"), categories=categories).codes
            return np.where(codes >= 0, codes, fallback).astype(np.int16, copy=False)
        return np.full(len(values), fallback, dtype=np.int16)

    def fit_transform(self, values: pd.Series) -> np.ndarray:
        return self.fit(values).transform(values)


class OrderedCategoricalQuotient:
    """Cross-fitted smoothed target statistic converted to an ordered state."""

    def __init__(
        self,
        n_bins: int = 8,
        smoothing: float = 20.0,
        n_splits: int = 5,
        random_state: int = 20260803,
    ):
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _fit_map(self, values: pd.Series, y: np.ndarray) -> tuple[dict[Any, float], float]:
        target = _as_binary_target(y)
        frame = pd.DataFrame({"x": values.astype("object"), "y": target})
        global_mean = float(target.mean())
        stats = frame.groupby("x", dropna=False, sort=False)["y"].agg(["sum", "count"])
        score = (stats["sum"] + self.smoothing * global_mean) / (
            stats["count"] + self.smoothing
        )
        return score.to_dict(), global_mean

    @staticmethod
    def _lookup(values: pd.Series, mapping: dict[Any, float], default: float) -> np.ndarray:
        return (
            values.astype("object")
            .map(mapping)
            .fillna(default)
            .to_numpy(dtype=np.float64)
        )

    def fit_transform(self, values: pd.Series, y: np.ndarray) -> np.ndarray:
        target = _as_binary_target(y)
        series = values.reset_index(drop=True)
        oof = np.empty(len(series), dtype=np.float64)
        cv = _stratified_cv(target, self.n_splits, self.random_state)
        for train_idx, valid_idx in cv.split(np.zeros(len(target)), target):
            mapping, default = self._fit_map(series.iloc[train_idx], target[train_idx])
            oof[valid_idx] = self._lookup(series.iloc[valid_idx], mapping, default)
        self.mapping_, self.default_ = self._fit_map(series, target)
        self.thresholds_ = _quantile_thresholds(oof, self.n_bins)
        states = _apply_thresholds(oof, self.thresholds_) + 1
        states[series.isna().to_numpy()] = 0
        self.cardinality_ = int(states.max(initial=0)) + 1
        return states.astype(np.int16, copy=False)

    def transform(self, values: pd.Series) -> np.ndarray:
        series = values
        scores = self._lookup(series, self.mapping_, self.default_)
        states = _apply_thresholds(scores, self.thresholds_) + 1
        states[series.isna().to_numpy()] = 0
        return states.astype(np.int16, copy=False)


class NewtonCategoricalQuotient(OrderedCategoricalQuotient):
    """Cross-fitted regularized Newton category ordering.

    With an intercept-only logistic baseline, category c receives score
    sum(y-p) / (sum(p(1-p)) + l2). The OOF construction prevents a row from
    contributing to its own target-aware state.
    """

    def __init__(self, n_bins=8, l2=10.0, n_splits=5, random_state=20260803):
        super().__init__(n_bins=n_bins, smoothing=0.0, n_splits=n_splits, random_state=random_state)
        self.l2 = float(l2)

    def _fit_map(self, values: pd.Series, y: np.ndarray):
        target = _as_binary_target(y)
        p = np.clip(float(target.mean()), 1e-6, 1 - 1e-6)
        frame = pd.DataFrame({
            "x": values.astype("object"),
            "g": target.astype(float) - p,
            "h": np.full(len(target), p * (1 - p), dtype=float),
        })
        stats = frame.groupby("x", dropna=False, sort=False).agg(G=("g", "sum"), H=("h", "sum"))
        score = stats["G"] / (stats["H"] + self.l2)
        return score.to_dict(), 0.0


class _EmbeddingBase:
    @staticmethod
    def _fit_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(X, dtype=np.float64)
        mean = np.nanmean(matrix, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.nanstd(matrix, axis=0)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, 1.0)
        return mean, scale

    @staticmethod
    def _scale(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, mean)
        return (matrix - mean) / scale

    def _quantize_fit(self, continuous: np.ndarray) -> np.ndarray:
        self.thresholds_ = [
            _quantile_thresholds(continuous[:, index], self.n_bins)
            for index in range(continuous.shape[1])
        ]
        states = np.column_stack(
            [
                _apply_thresholds(continuous[:, index], self.thresholds_[index])
                for index in range(continuous.shape[1])
            ]
        ).astype(np.int16)
        self.cardinalities_ = [
            int(states[:, index].max(initial=0)) + 1
            for index in range(states.shape[1])
        ]
        return states

    def _quantize(self, continuous: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                _apply_thresholds(continuous[:, index], self.thresholds_[index])
                for index in range(continuous.shape[1])
            ]
        ).astype(np.int16)


class EmbeddingPrototypeAdapter(_EmbeddingBase):
    """PCA, shrinkage LDA and fixed class prototypes; native-codegen friendly."""

    def __init__(
        self,
        n_pca: int = 8,
        n_bins: int = 8,
        n_prototypes: int = 64,
        include_pca: bool = True,
        n_splits: int = 5,
        random_state: int = 20260803,
    ):
        self.n_pca = int(n_pca)
        self.n_bins = int(n_bins)
        self.n_prototypes = int(n_prototypes)
        self.include_pca = bool(include_pca)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _fit_prototypes(self, Z: np.ndarray, y: np.ndarray, seed: int) -> list[np.ndarray]:
        from sklearn.cluster import MiniBatchKMeans

        prototypes: list[np.ndarray] = []
        for cls in (0, 1):
            class_rows = Z[y == cls]
            count = max(1, min(self.n_prototypes, len(class_rows)))
            if count == 1:
                centers = class_rows.mean(axis=0, keepdims=True)
            else:
                centers = MiniBatchKMeans(
                    n_clusters=count,
                    batch_size=min(512, len(class_rows)),
                    n_init=3,
                    random_state=seed + cls,
                ).fit(class_rows).cluster_centers_
            prototypes.append(np.asarray(centers, dtype=np.float64))
        return prototypes

    @staticmethod
    def _nearest_distances(Z: np.ndarray, prototypes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        result = []
        for centers in prototypes:
            # Chunking bounds temporary memory for larger embeddings.
            distances = np.empty(len(Z), dtype=np.float64)
            chunk = 2048
            for start in range(0, len(Z), chunk):
                block = Z[start : start + chunk]
                sq = np.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                distances[start : start + len(block)] = sq.min(axis=1)
            result.append(distances)
        return result[0], result[1]

    @classmethod
    def _prototype_features(
        cls,
        Z: np.ndarray,
        prototypes: list[np.ndarray],
        soft_scale: float,
    ) -> np.ndarray:
        d0, d1 = cls._nearest_distances(Z, prototypes)
        scale = max(float(soft_scale), 1e-8)
        soft = 1.0 / (1.0 + np.exp(np.clip((d1 - d0) / scale, -35.0, 35.0)))
        return np.column_stack([d0, d1, d0 - d1, soft])

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        target = _as_binary_target(y)
        self.mean_, self.scale_ = self._fit_scale(X)
        Z = self._scale(X, self.mean_, self.scale_)
        components = min(self.n_pca, Z.shape[1], max(1, len(Z) - 1))
        self.pca_ = PCA(n_components=components, random_state=self.random_state).fit(Z)
        pca_scores = self.pca_.transform(Z)
        extra = np.empty((len(Z), 5), dtype=np.float64)
        cv = _stratified_cv(target, self.n_splits, self.random_state)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(Z, target)):
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                Z[train_idx], target[train_idx]
            )
            extra[valid_idx, 0] = lda.decision_function(Z[valid_idx])
            prototypes = self._fit_prototypes(
                Z[train_idx], target[train_idx], self.random_state + 101 * fold
            )
            train_d0, train_d1 = self._nearest_distances(Z[train_idx], prototypes)
            scale = max(float(np.median(np.concatenate([train_d0, train_d1]))), 1e-8)
            extra[valid_idx, 1:] = self._prototype_features(
                Z[valid_idx], prototypes, scale
            )
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, target)
        self.prototypes_ = self._fit_prototypes(Z, target, self.random_state + 1000)
        train_d0, train_d1 = self._nearest_distances(Z, self.prototypes_)
        self.soft_scale_ = max(
            float(np.median(np.concatenate([train_d0, train_d1]))), 1e-8
        )
        continuous = np.column_stack(
            ([pca_scores] if self.include_pca else []) + [extra]
        )
        return self._quantize_fit(continuous)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._scale(X, self.mean_, self.scale_)
        pca_scores = self.pca_.transform(Z)
        extra = np.column_stack(
            [
                self.lda_.decision_function(Z),
                self._prototype_features(Z, self.prototypes_, self.soft_scale_),
            ]
        )
        continuous = np.column_stack(
            ([pca_scores] if self.include_pca else []) + [extra]
        )
        return self._quantize(continuous)


class EmbeddingKNNAdapter(_EmbeddingBase):
    """Higher-quality but memory-heavy OOF kNN density embedding adapter."""

    def __init__(
        self,
        n_pca: int = 8,
        n_bins: int = 8,
        knn_neighbors: Sequence[int] = (5, 15),
        n_splits: int = 5,
        random_state: int = 20260803,
        n_jobs: int | None = 1,
    ):
        self.n_pca = int(n_pca)
        self.n_bins = int(n_bins)
        self.knn_neighbors = tuple(int(k) for k in knn_neighbors)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.neighbors import KNeighborsClassifier

        target = _as_binary_target(y)
        self.mean_, self.scale_ = self._fit_scale(X)
        Z = self._scale(X, self.mean_, self.scale_)
        components = min(self.n_pca, Z.shape[1], max(1, len(Z) - 1))
        self.pca_ = PCA(n_components=components, random_state=self.random_state).fit(Z)
        pca_scores = self.pca_.transform(Z)
        extra = np.empty((len(Z), 1 + len(self.knn_neighbors)), dtype=np.float64)
        cv = _stratified_cv(target, self.n_splits, self.random_state)
        for train_idx, valid_idx in cv.split(Z, target):
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                Z[train_idx], target[train_idx]
            )
            extra[valid_idx, 0] = lda.decision_function(Z[valid_idx])
            for column, requested in enumerate(self.knn_neighbors, start=1):
                neighbors = max(1, min(requested, len(train_idx)))
                knn = KNeighborsClassifier(
                    n_neighbors=neighbors, weights="distance", n_jobs=self.n_jobs
                ).fit(Z[train_idx], target[train_idx])
                extra[valid_idx, column] = knn.predict_proba(Z[valid_idx])[:, 1]
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, target)
        self.knn_ = []
        for requested in self.knn_neighbors:
            neighbors = max(1, min(requested, len(Z)))
            self.knn_.append(
                KNeighborsClassifier(
                    n_neighbors=neighbors, weights="distance", n_jobs=self.n_jobs
                ).fit(Z, target)
            )
        return self._quantize_fit(np.column_stack([pca_scores, extra]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._scale(X, self.mean_, self.scale_)
        continuous = np.column_stack(
            [
                self.pca_.transform(Z),
                self.lda_.decision_function(Z),
                *[knn.predict_proba(Z)[:, 1] for knn in self.knn_],
            ]
        )
        return self._quantize(continuous)


class BasicEmbeddingAdapter(_EmbeddingBase):
    """Compatibility adapter: PCA plus one class-centroid direction."""

    def __init__(
        self,
        n_pca: int = 8,
        n_bins: int = 8,
        n_splits: int = 5,
        random_state: int = 20260803,
    ):
        self.n_pca = int(n_pca)
        self.n_bins = int(n_bins)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        target = _as_binary_target(y)
        self.mean_, self.scale_ = self._fit_scale(X)
        Z = self._scale(X, self.mean_, self.scale_)
        components = min(self.n_pca, Z.shape[1], max(1, len(Z) - 1))
        self.pca_ = PCA(n_components=components, random_state=self.random_state).fit(Z)
        pca_scores = self.pca_.transform(Z)
        extra = np.empty((len(Z), 2), dtype=np.float64)
        cv = _stratified_cv(target, self.n_splits, self.random_state)
        for train_idx, valid_idx in cv.split(Z, target):
            mu0 = Z[train_idx][target[train_idx] == 0].mean(axis=0)
            mu1 = Z[train_idx][target[train_idx] == 1].mean(axis=0)
            direction = mu1 - mu0
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm >= 1e-10 else np.zeros_like(direction)
            block = Z[valid_idx]
            extra[valid_idx, 0] = block @ direction
            extra[valid_idx, 1] = np.sum((block - mu0) ** 2, axis=1) - np.sum(
                (block - mu1) ** 2, axis=1
            )
        self.mu0_ = Z[target == 0].mean(axis=0)
        self.mu1_ = Z[target == 1].mean(axis=0)
        direction = self.mu1_ - self.mu0_
        norm = np.linalg.norm(direction)
        self.direction_ = direction / norm if norm >= 1e-10 else np.zeros_like(direction)
        return self._quantize_fit(np.column_stack([pca_scores, extra]))

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._scale(X, self.mean_, self.scale_)
        extra = np.column_stack(
            [
                Z @ self.direction_,
                np.sum((Z - self.mu0_) ** 2, axis=1)
                - np.sum((Z - self.mu1_) ** 2, axis=1),
            ]
        )
        return self._quantize(np.column_stack([self.pca_.transform(Z), extra]))


CategoryPolicy = Literal["auto", "identity", "ordered", "newton"]
EmbeddingMode = Literal["prototype", "knn", "basic"]
MissingPolicy = Literal["observed", "always", "none"]
IdentityRepresentation = Literal["state", "binary"]


class TypedQuotientAdapter:
    """Canonical typed front-end for the shared CERM core."""

    VERSION = "4.0"

    def __init__(
        self,
        categorical_columns: Sequence[str] | None = None,
        embedding_columns: Sequence[str] | None = None,
        *,
        category_policy: CategoryPolicy = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_newton_l2: float = 10.0,
        category_identity: IdentityRepresentation = "state",
        embedding_mode: EmbeddingMode = "prototype",
        embedding_pca: int = 8,
        embedding_bins: int = 8,
        embedding_prototypes: int = 64,
        retain_embedding_raw: bool = False,
        missing_policy: MissingPolicy = "observed",
        random_state: int = 20260803,
        n_jobs: int | None = 1,
    ):
        if category_policy not in {"auto", "identity", "ordered", "newton"}:
            raise ValueError("invalid category_policy")
        if embedding_mode not in {"prototype", "knn", "basic"}:
            raise ValueError("invalid embedding_mode")
        if missing_policy not in {"observed", "always", "none"}:
            raise ValueError("invalid missing_policy")
        if category_identity not in {"state", "binary"}:
            raise ValueError("invalid category_identity")
        self.categorical_columns = list(categorical_columns or [])
        self.embedding_columns = list(embedding_columns or [])
        self.category_policy = category_policy
        self.max_identity_categories = int(max_identity_categories)
        self.category_bins = int(category_bins)
        self.category_smoothing = float(category_smoothing)
        self.category_newton_l2 = float(category_newton_l2)
        self.category_identity = category_identity
        self.embedding_mode = embedding_mode
        self.embedding_pca = int(embedding_pca)
        self.embedding_bins = int(embedding_bins)
        self.embedding_prototypes = int(embedding_prototypes)
        self.retain_embedding_raw = bool(retain_embedding_raw)
        self.missing_policy = missing_policy
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    def _validate_frame(self, X: pd.DataFrame, *, fitted: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TypedQuotientAdapter requires a pandas DataFrame")
        if fitted:
            missing = [column for column in self.input_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"missing input columns: {missing}")
            if tuple(X.columns) == tuple(self.input_columns_):
                frame = X
            else:
                frame = X.loc[:, self.input_columns_]
            index = frame.index
            if (
                isinstance(index, pd.RangeIndex)
                and index.start == 0
                and index.stop == len(frame)
                and index.step == 1
            ):
                return frame
            return frame.reset_index(drop=True)
        frame = X.reset_index(drop=True)
        configured = set(self.categorical_columns) | set(self.embedding_columns)
        unknown = configured - set(frame.columns)
        if unknown:
            raise ValueError(f"unknown configured columns: {sorted(unknown)}")
        overlap = set(self.categorical_columns) & set(self.embedding_columns)
        if overlap:
            raise ValueError(f"columns cannot be categorical and embedding: {sorted(overlap)}")
        return frame

    def _category_mode(self, values: pd.Series) -> str:
        cardinality = int(values.nunique(dropna=True))
        if self.category_policy == "identity":
            return "identity"
        if self.category_policy == "ordered":
            return "ordered"
        if self.category_policy == "newton":
            return "newton"
        return "identity" if cardinality <= self.max_identity_categories else "ordered"

    def _embedding_adapter(self):
        if self.embedding_mode == "prototype":
            return EmbeddingPrototypeAdapter(
                n_pca=self.embedding_pca,
                n_bins=self.embedding_bins,
                n_prototypes=self.embedding_prototypes,
                random_state=self.random_state,
            )
        if self.embedding_mode == "knn":
            return EmbeddingKNNAdapter(
                n_pca=self.embedding_pca,
                n_bins=self.embedding_bins,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        return BasicEmbeddingAdapter(
            n_pca=self.embedding_pca,
            n_bins=self.embedding_bins,
            random_state=self.random_state,
        )

    @staticmethod
    def _append_identity(
        arrays: list[np.ndarray],
        names: list[str],
        kinds: list[str],
        cards: list[int | None],
        column: str,
        states: np.ndarray,
        cardinality: int,
        representation: IdentityRepresentation,
    ) -> None:
        if representation == "state":
            arrays.append(states[:, None])
            names.append(f"catid:{column}")
            kinds.append(FeatureKind.CATEGORY_IDENTITY.value)
            cards.append(cardinality)
            return
        bits = max(1, int(np.ceil(np.log2(max(cardinality, 2)))))
        for bit in range(bits):
            arrays.append(((states >> bit) & 1)[:, None])
            names.append(f"catbit:{column}:{bit}")
            kinds.append(FeatureKind.CATEGORY_IDENTITY_BIT.value)
            cards.append(2)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> TypedAdapterOutput:
        frame = self._validate_frame(X, fitted=False)
        target = _as_binary_target(y)
        if len(frame) != len(target):
            raise ValueError("X and y length mismatch")
        self.input_columns_ = list(frame.columns)
        self.numeric_columns_ = [
            column
            for column in frame.columns
            if column not in set(self.categorical_columns) | set(self.embedding_columns)
        ]
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_missing_columns_: set[str] = set()
        self.category_modes_: dict[str, str] = {}
        self.category_encoders_: dict[str, Any] = {}
        arrays: list[np.ndarray] = []
        names: list[str] = []
        kinds: list[str] = []
        cards: list[int | None] = []

        for column in self.numeric_columns_:
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            self.numeric_medians_[column] = median
            arrays.append(np.where(missing, median, values)[:, None])
            names.append(f"num:{column}")
            kinds.append(FeatureKind.NUMERIC.value)
            cards.append(None)
            emit_missing = self.missing_policy == "always" or (
                self.missing_policy == "observed" and bool(missing.any())
            )
            if emit_missing:
                self.numeric_missing_columns_.add(column)
                arrays.append(missing.astype(np.int16)[:, None])
                names.append(f"missing:{column}")
                kinds.append(FeatureKind.MISSING.value)
                cards.append(2)

        for index, column in enumerate(self.categorical_columns):
            mode = self._category_mode(frame[column])
            self.category_modes_[column] = mode
            if mode == "identity":
                encoder = IdentityCategoricalEncoder(
                    reserve_missing=self.missing_policy == "always"
                )
                states = encoder.fit_transform(frame[column])
                self.category_encoders_[column] = encoder
                self._append_identity(
                    arrays,
                    names,
                    kinds,
                    cards,
                    column,
                    states,
                    encoder.cardinality_,
                    self.category_identity,
                )
            else:
                if mode == "newton":
                    encoder = NewtonCategoricalQuotient(
                        n_bins=self.category_bins,
                        l2=self.category_newton_l2,
                        random_state=self.random_state + index,
                    )
                else:
                    encoder = OrderedCategoricalQuotient(
                        n_bins=self.category_bins,
                        smoothing=self.category_smoothing,
                        random_state=self.random_state + index,
                    )
                states = encoder.fit_transform(frame[column], target)
                self.category_encoders_[column] = encoder
                arrays.append(states[:, None])
                names.append(f"catq:{column}")
                kinds.append(FeatureKind.CATEGORY_QUOTIENT.value)
                cards.append(encoder.cardinality_)

        self.embedding_adapter_ = None
        if self.embedding_columns:
            embedding = frame[self.embedding_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            self.embedding_adapter_ = self._embedding_adapter()
            states = self.embedding_adapter_.fit_transform(embedding, target)
            for index in range(states.shape[1]):
                arrays.append(states[:, index : index + 1])
                names.append(f"embq:{index}")
                kinds.append(FeatureKind.EMBEDDING_QUOTIENT.value)
                cards.append(self.embedding_adapter_.cardinalities_[index])
            if self.retain_embedding_raw:
                filled = np.where(
                    np.isfinite(embedding), embedding, self.embedding_adapter_.mean_
                )
                arrays.append(filled)
                names.extend(f"embraw:{column}" for column in self.embedding_columns)
                kinds.extend([FeatureKind.EMBEDDING_RAW.value] * filled.shape[1])
                cards.extend([None] * filled.shape[1])

        matrix = (
            np.column_stack(arrays).astype(np.float64, copy=False)
            if arrays
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        self.feature_names_ = names
        self.feature_kinds_ = kinds
        self.cardinalities_ = cards
        self.output_dim_ = matrix.shape[1]
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=names,
            feature_kinds=kinds,
            cardinalities=cards,
            metadata={
                "adapter_version": self.VERSION,
                "input_columns": self.input_columns_,
                "output_dim": self.output_dim_,
                "category_modes": self.category_modes_,
                "embedding_mode": self.embedding_mode if self.embedding_columns else None,
            },
        ).validate(len(frame))

    def _output_plan(self) -> list[tuple[str, Any, int | None]]:
        plan: list[tuple[str, Any, int | None]] = []
        for column in self.numeric_columns_:
            plan.append(("numeric", column, None))
            if column in self.numeric_missing_columns_:
                plan.append(("missing", column, None))
        for column in self.categorical_columns:
            encoder = self.category_encoders_[column]
            if self.category_modes_[column] == "identity":
                if self.category_identity == "state":
                    plan.append(("category_state", column, None))
                else:
                    bits = max(1, int(np.ceil(np.log2(max(encoder.cardinality_, 2)))))
                    for bit in range(bits):
                        plan.append(("category_bit", column, bit))
            else:
                plan.append(("category_quotient", column, None))
        if self.embedding_columns:
            for index in range(len(self.embedding_adapter_.cardinalities_)):
                plan.append(("embedding_quotient", index, None))
            if self.retain_embedding_raw:
                for index, _column in enumerate(self.embedding_columns):
                    plan.append(("embedding_raw", index, None))
        return plan

    def transform_columns(
        self, X: pd.DataFrame, output_indices: Sequence[int]
    ) -> TypedAdapterOutput:
        """Transform only requested adapted output columns, exactly.

        This is the typed-adapter analogue of the finite-state projected
        transform: source columns and encoders that cannot contribute to the
        fitted downstream model are never materialized.  Returned columns
        follow ``output_indices`` exactly, including arbitrary order.
        """
        frame = self._validate_frame(X, fitted=True)
        indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        if np.any((indices < 0) | (indices >= int(self.output_dim_))):
            raise IndexError("typed adapter output index out of range")
        if len(indices) == int(self.output_dim_) and np.array_equal(
            indices, np.arange(int(self.output_dim_), dtype=np.int64)
        ):
            return self.transform(frame)

        plan = self._output_plan()
        numeric_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        category_cache: dict[Any, np.ndarray] = {}
        embedding_states = None
        embedding_filled = None
        columns: list[np.ndarray] = []

        for raw_index in indices:
            kind, source, extra = plan[int(raw_index)]
            if kind in {"numeric", "missing"}:
                cached = numeric_cache.get(source)
                if cached is None:
                    values = _numeric_series_values(frame[source])
                    missing = ~np.isfinite(values)
                    filled = np.where(missing, self.numeric_medians_[source], values)
                    cached = (filled, missing)
                    numeric_cache[source] = cached
                columns.append(cached[0] if kind == "numeric" else cached[1].astype(np.int16))
            elif kind in {"category_state", "category_bit", "category_quotient"}:
                states = category_cache.get(source)
                if states is None:
                    states = self.category_encoders_[source].transform(frame[source])
                    category_cache[source] = states
                if kind == "category_bit":
                    columns.append((states >> int(extra)) & 1)
                else:
                    columns.append(states)
            elif kind in {"embedding_quotient", "embedding_raw"}:
                if embedding_states is None:
                    embedding = frame[self.embedding_columns].apply(
                        pd.to_numeric, errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    embedding_states = self.embedding_adapter_.transform(embedding)
                    if self.retain_embedding_raw:
                        embedding_filled = np.where(
                            np.isfinite(embedding), embedding, self.embedding_adapter_.mean_
                        )
                if kind == "embedding_quotient":
                    columns.append(embedding_states[:, int(source)])
                else:
                    columns.append(embedding_filled[:, int(source)])
            else:
                raise RuntimeError(f"unsupported typed adapter projection kind: {kind}")

        matrix = (
            np.column_stack(columns).astype(np.float64, copy=False)
            if columns
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        names = [self.feature_names_[int(index)] for index in indices]
        kinds = [self.feature_kinds_[int(index)] for index in indices]
        cards = [self.cardinalities_[int(index)] for index in indices]
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=names,
            feature_kinds=kinds,
            cardinalities=cards,
            metadata={
                "adapter_version": self.VERSION,
                "output_dim": len(indices),
                "projected": True,
            },
        ).validate(len(frame))

    def transform(self, X: pd.DataFrame) -> TypedAdapterOutput:
        frame = self._validate_frame(X, fitted=True)
        arrays: list[np.ndarray] = []
        for column in self.numeric_columns_:
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            arrays.append(np.where(missing, self.numeric_medians_[column], values)[:, None])
            if column in self.numeric_missing_columns_:
                arrays.append(missing.astype(np.int16)[:, None])
        for column in self.categorical_columns:
            encoder = self.category_encoders_[column]
            states = encoder.transform(frame[column])
            if self.category_modes_[column] == "identity":
                if self.category_identity == "state":
                    arrays.append(states[:, None])
                else:
                    bits = max(1, int(np.ceil(np.log2(max(encoder.cardinality_, 2)))))
                    arrays.extend(((states >> bit) & 1)[:, None] for bit in range(bits))
            else:
                arrays.append(states[:, None])
        if self.embedding_columns:
            embedding = frame[self.embedding_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            arrays.append(self.embedding_adapter_.transform(embedding))
            if self.retain_embedding_raw:
                arrays.append(
                    np.where(
                        np.isfinite(embedding), embedding, self.embedding_adapter_.mean_
                    )
                )
        matrix = (
            np.column_stack(arrays).astype(np.float64, copy=False)
            if arrays
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=self.feature_names_,
            feature_kinds=self.feature_kinds_,
            cardinalities=self.cardinalities_,
            metadata={"adapter_version": self.VERSION, "output_dim": self.output_dim_},
        ).validate(len(frame))


class TypedAdapterHybridCERM:
    def __init__(
        self,
        adapter: TypedQuotientAdapter,
        max_features: int = 64,
        pair_feature_limit: int = 24,
        random_state: int = 20260803,
        replacement_objective: str = "balanced",
    ):
        self.adapter = adapter
        self.max_features = int(max_features)
        self.pair_feature_limit = int(pair_feature_limit)
        self.random_state = int(random_state)
        self.replacement_objective = replacement_objective

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TypedAdapterHybridCERM":
        adapted = self.adapter.fit_transform(X, y)
        self.model_ = HybridQuotientBlockCERM(
            max_features=self.max_features,
            pair_feature_limit=self.pair_feature_limit,
            random_state=self.random_state,
            replacement_objective=self.replacement_objective,
            feature_kinds=adapted.feature_kinds,
            feature_cardinalities=adapted.cardinalities,
        ).fit(adapted.matrix, _as_binary_target(y))
        self.classes_ = self.model_.classes_
        self.adapter_metadata_ = dict(adapted.metadata)
        self.adapter_output_ = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.adapter.transform(X).matrix)

    @property
    def model_bytes_estimate_(self) -> int:
        return int(
            self.model_.model_bytes_estimate_
            + estimate_typed_adapter_bytes(self.adapter)
        )


def estimate_typed_adapter_bytes(adapter: TypedQuotientAdapter) -> int:
    """Estimate resident constant/reference bytes for a fitted typed adapter."""
    total = 0
    for encoder in adapter.category_encoders_.values():
        total += len(getattr(encoder, "mapping_", {})) * 16
        if hasattr(encoder, "thresholds_"):
            total += np.asarray(encoder.thresholds_).nbytes
    embedding = adapter.embedding_adapter_
    if embedding is not None:
        for name in ("mean_", "scale_", "direction_", "mu0_", "mu1_"):
            if hasattr(embedding, name):
                total += np.asarray(getattr(embedding, name)).nbytes
        if hasattr(embedding, "pca_"):
            total += embedding.pca_.components_.nbytes + embedding.pca_.mean_.nbytes
        if hasattr(embedding, "lda_"):
            total += embedding.lda_.coef_.nbytes + embedding.lda_.intercept_.nbytes
        if hasattr(embedding, "prototypes_"):
            total += sum(np.asarray(array).nbytes for array in embedding.prototypes_)
        if hasattr(embedding, "knn_"):
            total += sum(
                knn._fit_X.nbytes + knn._y.nbytes for knn in embedding.knn_
            )
        total += sum(np.asarray(threshold).nbytes for threshold in embedding.thresholds_)
    return int(total)


def _json_scalar(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    return {
        "type": f"python:{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _json_key(value: Any) -> str:
    return f"{type(value).__name__}:{repr(value)}"


def _compact_mapping_payload(
    mapping: dict[Any, Any],
    *,
    value_name: str,
    value_cast: type[int] | type[float],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return a compact portable mapping when category keys share a scalar type.

    Common tabular categories are homogeneous strings or integers.  Storing the
    type tag once, followed by parallel key/value arrays, avoids repeating JSON
    field names for every category.  Heterogeneous or unsupported Python objects
    retain the legacy row-wise representation so old readers remain easy to
    support and portability can still be diagnosed explicitly.
    """
    rows: list[tuple[dict[str, Any], Any]] = []
    for key, value in mapping.items():
        if pd.isna(key):
            continue
        rows.append((_json_scalar(key), value_cast(value)))
    if not rows:
        return {
            "encoding": "parallel-json-v1",
            "key_type": "str",
            "keys": [],
            value_name: [],
        }
    key_types = {payload.get("type") for payload, _ in rows}
    portable_scalar_types = {"str", "bool", "int", "float", "NoneType"}
    if len(key_types) == 1 and key_types <= portable_scalar_types:
        key_type = next(iter(key_types))
        return {
            "encoding": "parallel-json-v1",
            "key_type": key_type,
            "keys": [payload.get("value") for payload, _ in rows],
            value_name: [value for _, value in rows],
        }
    return [
        {"key": payload, value_name[:-1] if value_name.endswith("s") else value_name: value}
        for payload, value in rows
    ]


def export_typed_adapter_ir(
    adapter: TypedQuotientAdapter,
    prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "format": "cerm-typed-adapter-ir-v1",
        "schema_version": 3,
        "canonical_format": "cerm-typed-adapter-ir-v2",
        "adapter_version": adapter.VERSION,
        "input_columns": adapter.input_columns_,
        "output_feature_names": adapter.feature_names_,
        "output_feature_kinds": adapter.feature_kinds_,
        "output_cardinalities": adapter.cardinalities_,
        "category_identity": adapter.category_identity,
        "missing_policy": adapter.missing_policy,
        "portable": adapter.embedding_adapter_ is None,
        "numeric": [],
        "categorical": [],
        "embedding": None,
    }
    for index, column in enumerate(adapter.numeric_columns_):
        key = f"numeric_median_{index}"
        arrays[key] = np.asarray([adapter.numeric_medians_[column]], dtype=np.float64)
        manifest["numeric"].append(
            {
                "column": column,
                "column_index": int(adapter.input_columns_.index(column)),
                "median_array": key,
                "emit_missing_state": column in adapter.numeric_missing_columns_,
            }
        )
    for index, column in enumerate(adapter.categorical_columns):
        encoder = adapter.category_encoders_[column]
        entry: dict[str, Any] = {
            "column": column,
            "column_index": int(adapter.input_columns_.index(column)),
            "mode": adapter.category_modes_[column],
        }
        if isinstance(encoder, IdentityCategoricalEncoder):
            entry["mapping"] = _compact_mapping_payload(
                encoder.mapping_, value_name="states", value_cast=int
            )
            entry["cardinality"] = encoder.cardinality_
            entry["representation"] = adapter.category_identity
            entry["has_missing"] = encoder.has_missing_
            entry["reserved_missing"] = encoder.has_reserved_missing_
            entry["missing_state"] = encoder.missing_state_
            entry["unknown_state"] = (
                encoder.missing_state_ if encoder.has_reserved_missing_ else 0
            )
        else:
            threshold_key = f"category_thresholds_{index}"
            arrays[threshold_key] = encoder.thresholds_
            entry.update(
                {
                    "mapping": _compact_mapping_payload(
                        encoder.mapping_, value_name="scores", value_cast=float
                    ),
                    "default_score": encoder.default_,
                    "thresholds_array": threshold_key,
                    "cardinality": encoder.cardinality_,
                }
            )
        mapping_payload = entry.get("mapping", [])
        if isinstance(mapping_payload, list):
            for item in mapping_payload:
                key_payload = item.get("key", {})
                if str(key_payload.get("type", "")).startswith("python:"):
                    manifest["portable"] = False
        manifest["categorical"].append(entry)
    embedding = adapter.embedding_adapter_
    if embedding is not None:
        emb: dict[str, Any] = {
            "class": type(embedding).__name__,
            "native_latency_ready": not isinstance(embedding, EmbeddingKNNAdapter),
            "threshold_arrays": [],
        }
        for name in ("mean_", "scale_", "direction_", "mu0_", "mu1_"):
            if hasattr(embedding, name):
                key = f"embedding_{name.rstrip('_')}"
                arrays[key] = np.asarray(getattr(embedding, name), dtype=np.float64)
                emb[f"{name.rstrip('_')}_array"] = key
        if hasattr(embedding, "pca_"):
            arrays["embedding_pca_components"] = embedding.pca_.components_
            arrays["embedding_pca_mean"] = embedding.pca_.mean_
            emb["pca_components_array"] = "embedding_pca_components"
            emb["pca_mean_array"] = "embedding_pca_mean"
        if hasattr(embedding, "lda_"):
            arrays["embedding_lda_coef"] = embedding.lda_.coef_
            arrays["embedding_lda_intercept"] = embedding.lda_.intercept_
            emb["lda_coef_array"] = "embedding_lda_coef"
            emb["lda_intercept_array"] = "embedding_lda_intercept"
        if hasattr(embedding, "prototypes_"):
            prototype_keys = []
            for index, prototype in enumerate(embedding.prototypes_):
                key = f"embedding_prototype_{index}"
                arrays[key] = prototype
                prototype_keys.append(key)
            emb["prototype_arrays"] = prototype_keys
            emb["soft_scale"] = embedding.soft_scale_
        if hasattr(embedding, "knn_"):
            emb["reference_rows"] = [len(knn._fit_X) for knn in embedding.knn_]
        for index, threshold in enumerate(embedding.thresholds_):
            key = f"embedding_threshold_{index}"
            arrays[key] = threshold
            emb["threshold_arrays"].append(key)
        manifest["embedding"] = emb
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(manifest, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, json_path


# Compatibility names used by earlier experiment scripts.
EmbeddingQuotientAdapter = BasicEmbeddingAdapter
EmbeddingQuotientAdapterV2 = EmbeddingKNNAdapter
EmbeddingQuotientAdapterV3 = EmbeddingPrototypeAdapter


class TypedQuotientAdapterV2(TypedQuotientAdapter):
    def __init__(
        self,
        *args,
        retain_category_raw: bool = True,
        max_raw_categories: int = 16,
        embedding_mode: str = "lda_knn",
        **kwargs,
    ):
        canonical_mode = {
            "lda_knn": "knn",
            "lda_proto": "prototype",
            "pca_proto": "basic",
        }.get(embedding_mode, embedding_mode)
        super().__init__(
            *args,
            category_policy="auto" if retain_category_raw else "ordered",
            max_identity_categories=max_raw_categories,
            embedding_mode=canonical_mode,
            **kwargs,
        )


class TypedQuotientAdapterV3(TypedQuotientAdapterV2):
    def __init__(
        self,
        *args,
        category_identity: str = "binary",
        embedding_mode: str = "lda_proto",
        **kwargs,
    ):
        super().__init__(
            *args,
            category_identity=category_identity,
            embedding_mode=embedding_mode,
            **kwargs,
        )
