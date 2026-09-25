from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import platform
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, column_or_1d

from ._compat import fit_ridge_lsqr_alpha_path_exact, num_samples
from ._version import __version__
from .program import _jsonable, _select_features
from .params import (
    explain_semantic_parameters,
    format_semantic_parameter_summary,
    resolve_semantic_alias,
    semantic_parameter_values,
)
from .validation import (
    dense_numeric_matrix,
    reset_dataframe_index_if_needed,
    validate_dataframe_schema,
)
from ._internal.cerm_hierarchical_residual import (
    NestedQuantileEncoder,
    _parent_labels_from_fine,
)
from ._internal.cerm_state_design import ReferenceStateEncoder
from ._internal.cerm_typed_quotient_adapters_v4 import (
    FeatureKind,
    IdentityCategoricalEncoder,
    TypedAdapterOutput,
    _numeric_series_values,
)


def _levels(max_bins: int) -> tuple[int, ...]:
    if max_bins == 4:
        return (4,)
    if max_bins == 8:
        return (4, 8)
    if max_bins == 16:
        return (4, 8, 16)
    raise ValueError("max_bins must be one of 4, 8, or 16")


def _category_key(value: Any) -> str:
    if pd.isna(value):
        return "<CERM_MISSING>"
    return f"{type(value).__name__}:{value!r}"


class RegressionTargetQuotient:
    """Cross-fitted smoothed target-mean quotient for regression categories."""

    def __init__(
        self,
        n_bins: int = 8,
        smoothing: float = 20.0,
        random_state: int = 20260803,
    ):
        self.n_bins = int(n_bins)
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)

    @staticmethod
    def _keys(values: pd.Series) -> np.ndarray:
        return np.asarray([_category_key(value) for value in values], dtype=object)

    def _mapping(
        self,
        keys: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[dict[str, float], float]:
        if sample_weight is None:
            prior = float(np.mean(y))
            frame = pd.DataFrame({"key": keys, "target": y})
            grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
            means = (grouped["sum"] + self.smoothing * prior) / (
                grouped["count"] + self.smoothing
            )
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            total_weight = float(weights.sum())
            prior = float(np.dot(weights, y) / total_weight)
            frame = pd.DataFrame(
                {
                    "key": keys,
                    "weighted_target": weights * y,
                    "weight": weights,
                }
            )
            grouped = frame.groupby("key", sort=False)[
                ["weighted_target", "weight"]
            ].sum()
            means = (
                grouped["weighted_target"] + self.smoothing * prior
            ) / (grouped["weight"] + self.smoothing)
        return {str(key): float(value) for key, value in means.items()}, prior

    def fit_transform(
        self,
        values: pd.Series,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        keys = self._keys(values)
        y = np.asarray(y, dtype=np.float64)
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        n = len(y)
        n_splits = min(5, max(2, n // 8)) if n >= 8 else 0
        encoded = np.empty(n, dtype=np.float64)
        if n_splits >= 2:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            for train, valid in cv.split(keys):
                mapping, prior = self._mapping(
                    keys[train],
                    y[train],
                    None if weights is None else weights[train],
                )
                encoded[valid] = np.asarray(
                    [mapping.get(str(key), prior) for key in keys[valid]], dtype=np.float64
                )
        else:
            mapping, prior = self._mapping(keys, y, weights)
            encoded[:] = [mapping.get(str(key), prior) for key in keys]

        self.mapping_, self.prior_ = self._mapping(keys, y, weights)
        if np.allclose(encoded, encoded[0] if len(encoded) else 0.0):
            self.thresholds_ = np.empty(0, dtype=np.float64)
        else:
            quantiles = np.arange(1, self.n_bins, dtype=np.float64) / self.n_bins
            thresholds = np.unique(np.quantile(encoded, quantiles))
            thresholds = thresholds[(thresholds > encoded.min()) & (thresholds < encoded.max())]
            self.thresholds_ = thresholds.astype(np.float64)
        self.cardinality_ = int(len(self.thresholds_) + 1)
        return np.searchsorted(self.thresholds_, encoded, side="right").astype(np.int16)

    def transform(self, values: pd.Series) -> np.ndarray:
        keys = self._keys(values)
        encoded = np.asarray(
            [self.mapping_.get(str(key), self.prior_) for key in keys], dtype=np.float64
        )
        return np.searchsorted(self.thresholds_, encoded, side="right").astype(np.int16)

    @property
    def nbytes(self) -> int:
        return int(self.thresholds_.nbytes + sum(len(key) + 8 for key in self.mapping_))


class RegressionTypedAdapter:
    """Typed DataFrame adapter with leakage-reduced regression quotients."""

    VERSION = "1.0"

    def __init__(
        self,
        categorical_columns: Sequence[str] | None = None,
        embedding_columns: Sequence[str] | None = None,
        *,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        missing_policy: str = "observed",
        random_state: int = 20260803,
    ):
        if category_policy not in {"auto", "identity", "ordered", "newton"}:
            raise ValueError("invalid category_policy")
        if category_identity not in {"state", "binary"}:
            raise ValueError("invalid category_identity")
        if missing_policy not in {"observed", "always", "none"}:
            raise ValueError("invalid missing_policy")
        self.categorical_columns = list(categorical_columns or [])
        self.embedding_columns = list(embedding_columns or [])
        self.category_policy = category_policy
        self.max_identity_categories = int(max_identity_categories)
        self.category_bins = int(category_bins)
        self.category_smoothing = float(category_smoothing)
        self.category_identity = category_identity
        self.missing_policy = missing_policy
        self.random_state = int(random_state)

    def _category_mode(self, values: pd.Series) -> str:
        if self.category_policy == "identity":
            return "identity"
        if self.category_policy in {"ordered", "newton"}:
            return "quotient"
        return "identity" if values.nunique(dropna=True) <= self.max_identity_categories else "quotient"

    @staticmethod
    def _append_identity(arrays, names, kinds, cards, column, states, cardinality, representation):
        if representation == "state":
            arrays.append(states[:, None])
            names.append(f"catid:{column}")
            kinds.append(FeatureKind.CATEGORY_IDENTITY.value)
            cards.append(int(cardinality))
            return
        bits = max(1, int(np.ceil(np.log2(max(cardinality, 2)))))
        for bit in range(bits):
            arrays.append(((states >> bit) & 1)[:, None])
            names.append(f"catbit:{column}:{bit}")
            kinds.append(FeatureKind.CATEGORY_IDENTITY_BIT.value)
            cards.append(2)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> TypedAdapterOutput:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RegressionTypedAdapter requires a pandas DataFrame")
        frame = X.reset_index(drop=True)
        unknown = (set(self.categorical_columns) | set(self.embedding_columns)) - set(frame.columns)
        if unknown:
            raise ValueError(f"unknown configured columns: {sorted(unknown)}")
        overlap = set(self.categorical_columns) & set(self.embedding_columns)
        if overlap:
            raise ValueError(f"columns cannot be categorical and embedding: {sorted(overlap)}")
        self.input_columns_ = list(frame.columns)
        categorical = set(self.categorical_columns)
        self.numeric_columns_ = [column for column in frame.columns if column not in categorical]
        self.numeric_medians_ = {}
        self.numeric_missing_columns_ = set()
        self.category_modes_ = {}
        self.category_encoders_ = {}
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
            emit = self.missing_policy == "always" or (
                self.missing_policy == "observed" and bool(missing.any())
            )
            if emit:
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
                self._append_identity(
                    arrays, names, kinds, cards, column, states,
                    encoder.cardinality_, self.category_identity,
                )
            else:
                encoder = RegressionTargetQuotient(
                    n_bins=self.category_bins,
                    smoothing=self.category_smoothing,
                    random_state=self.random_state + index,
                )
                states = encoder.fit_transform(
                    frame[column], y, sample_weight=sample_weight
                )
                arrays.append(states[:, None])
                names.append(f"catq:{column}")
                kinds.append(FeatureKind.CATEGORY_QUOTIENT.value)
                cards.append(encoder.cardinality_)
            self.category_encoders_[column] = encoder

        matrix = np.column_stack(arrays).astype(np.float64, copy=False) if arrays else np.empty((len(frame), 0))
        self.feature_names_ = names
        self.feature_kinds_ = kinds
        self.cardinalities_ = cards
        self.output_dim_ = int(matrix.shape[1])
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=names,
            feature_kinds=kinds,
            cardinalities=cards,
            metadata={
                "adapter_version": self.VERSION,
                "task_type": "regression",
                "input_columns": self.input_columns_,
                "output_dim": self.output_dim_,
                "category_modes": self.category_modes_,
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
            if self.category_modes_[column] == "identity" and self.category_identity == "binary":
                bits = max(1, int(np.ceil(np.log2(max(encoder.cardinality_, 2)))))
                for bit in range(bits):
                    plan.append(("category_bit", column, bit))
            else:
                plan.append(("category_state", column, None))
        return plan

    def transform_columns(
        self, X: pd.DataFrame, output_indices: Sequence[int]
    ) -> TypedAdapterOutput:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, tuple(self.input_columns_)))
        indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        if np.any((indices < 0) | (indices >= int(self.output_dim_))):
            raise IndexError("regression adapter output index out of range")
        if len(indices) == int(self.output_dim_) and np.array_equal(
            indices, np.arange(int(self.output_dim_), dtype=np.int64)
        ):
            return self.transform(frame)
        plan = self._output_plan()
        numeric_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        category_cache: dict[Any, np.ndarray] = {}
        columns: list[np.ndarray] = []
        for raw_index in indices:
            kind, source, extra = plan[int(raw_index)]
            if kind in {"numeric", "missing"}:
                cached = numeric_cache.get(source)
                if cached is None:
                    values = _numeric_series_values(frame[source])
                    missing = ~np.isfinite(values)
                    cached = (
                        np.where(missing, self.numeric_medians_[source], values),
                        missing,
                    )
                    numeric_cache[source] = cached
                columns.append(cached[0] if kind == "numeric" else cached[1].astype(np.int16))
            else:
                states = category_cache.get(source)
                if states is None:
                    states = self.category_encoders_[source].transform(frame[source])
                    category_cache[source] = states
                if kind == "category_bit":
                    columns.append((states >> int(extra)) & 1)
                else:
                    columns.append(states)
        matrix = (
            np.column_stack(columns).astype(np.float64, copy=False)
            if columns
            else np.empty((len(frame), 0), dtype=np.float64)
        )
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=[self.feature_names_[int(i)] for i in indices],
            feature_kinds=[self.feature_kinds_[int(i)] for i in indices],
            cardinalities=[self.cardinalities_[int(i)] for i in indices],
            metadata={
                "adapter_version": self.VERSION,
                "task_type": "regression",
                "projected": True,
            },
        ).validate(len(frame))

    def transform(self, X: pd.DataFrame) -> TypedAdapterOutput:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, tuple(self.input_columns_)))
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
            if self.category_modes_[column] == "identity" and self.category_identity == "binary":
                bits = max(1, int(np.ceil(np.log2(max(encoder.cardinality_, 2)))))
                arrays.extend(((states >> bit) & 1)[:, None] for bit in range(bits))
            else:
                arrays.append(states[:, None])
        matrix = np.column_stack(arrays).astype(np.float64, copy=False) if arrays else np.empty((len(frame), 0))
        return TypedAdapterOutput(
            matrix=matrix,
            feature_names=self.feature_names_,
            feature_kinds=self.feature_kinds_,
            cardinalities=self.cardinalities_,
            metadata={"adapter_version": self.VERSION, "task_type": "regression"},
        ).validate(len(frame))

    @property
    def nbytes(self) -> int:
        total = len(self.numeric_medians_) * 8
        for encoder in self.category_encoders_.values():
            total += int(getattr(encoder, "nbytes", 0))
        return int(total)


def _between_group_score(
    code: np.ndarray,
    y: np.ndarray,
    cardinality: int | None = None,
    sample_weight: np.ndarray | None = None,
    target_stats: _RegressionTargetStats | None = None,
) -> float:
    code = np.asarray(code, dtype=np.int64)
    if code.size == 0:
        return 0.0
    stats = target_stats or _regression_target_stats(y, sample_weight)
    if cardinality is None:
        # np.bincount already determines max(code) in its C loop.  Avoiding a
        # separate NumPy max scan preserves the exact output length and values.
        if stats.sample_weight is None:
            counts = np.bincount(code).astype(np.float64)
            sums = np.bincount(code, weights=y).astype(np.float64)
        else:
            counts = np.bincount(
                code, weights=stats.sample_weight
            ).astype(np.float64)
            sums = np.bincount(
                code, weights=stats.weighted_target
            ).astype(np.float64)
    else:
        cardinality = int(cardinality)
        if stats.sample_weight is None:
            counts = np.bincount(code, minlength=cardinality).astype(np.float64)
            sums = np.bincount(
                code, weights=y, minlength=cardinality
            ).astype(np.float64)
        else:
            counts = np.bincount(
                code, weights=stats.sample_weight, minlength=cardinality
            ).astype(np.float64)
            sums = np.bincount(
                code, weights=stats.weighted_target, minlength=cardinality
            ).astype(np.float64)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    between = float(np.sum(counts * (means - stats.overall) ** 2))
    return between / max(stats.total, 1e-15)


@dataclass(frozen=True)
class RegressionConfig:
    max_main_level: int
    n_pairs: int
    alpha: float


@dataclass(frozen=True)
class _RegressionTargetStats:
    sample_weight: np.ndarray | None
    weighted_target: np.ndarray | None
    overall: float
    total: float


def _regression_target_stats(
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> _RegressionTargetStats:
    if sample_weight is None:
        overall = float(np.mean(y))
        total = float(np.sum((y - overall) ** 2))
        return _RegressionTargetStats(None, None, overall, total)
    weights = np.asarray(sample_weight, dtype=np.float64)
    overall = float(np.dot(weights, y) / weights.sum())
    total = float(np.dot(weights, (y - overall) ** 2))
    return _RegressionTargetStats(weights, weights * y, overall, total)


def _fit_ridge_lsqr_model_exact(
    design: sparse.csr_matrix,
    target: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> Ridge:
    """Return a fitted Ridge object from the exact cached LSQR path."""

    coefficients, intercepts, iterations = fit_ridge_lsqr_alpha_path_exact(
        design,
        target,
        np.asarray([float(alpha)], dtype=np.float64),
        sample_weight=sample_weight,
    )
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.coef_ = coefficients[0]
    model.intercept_ = float(intercepts[0])
    model.n_iter_ = np.asarray([iterations[0]], dtype=np.int32)
    model.n_features_in_ = int(design.shape[1])
    model.solver_ = "lsqr"
    return model


class FiniteStateRidgeRegressor:
    """Finite-state main/pair model with a ridge regression head."""

    def __init__(
        self,
        *,
        max_features: int = 64,
        max_bins: int = 16,
        max_interaction_features: int = 24,
        max_interactions: int = 24,
        interaction_order: int = 2,
        reg_lambda: str | float = "auto",
        preset: str = "accurate",
        subsample: float = 1.0,
        n_jobs: int | None = 1,
        random_state: int = 20260803,
        feature_kinds: Sequence[str] | None = None,
        feature_cardinalities: Sequence[int | None] | None = None,
        include_linear: bool = True,
    ):
        self.max_features = int(max_features)
        self.max_bins = int(max_bins)
        self.max_interaction_features = int(max_interaction_features)
        self.max_interactions = int(max_interactions)
        self.interaction_order = int(interaction_order)
        self.reg_lambda = reg_lambda
        self.preset = str(preset)
        self.subsample = float(subsample)
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self.feature_kinds = feature_kinds
        self.feature_cardinalities = feature_cardinalities
        self.include_linear = bool(include_linear)
        self.levels = _levels(self.max_bins)

    def _selection_sample(self, X, y, sample_weight=None):
        if self.subsample >= 1.0:
            return X, y, sample_weight
        n = max(16, int(np.ceil(len(y) * self.subsample)))
        n = min(n, len(y))
        rng = np.random.default_rng(self.random_state + 49979687)
        indices = np.sort(rng.choice(len(y), size=n, replace=False))
        return (
            X[indices],
            y[indices],
            None if sample_weight is None else sample_weight[indices],
        )

    def _feature_rank(
        self,
        states: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        target_stats: _RegressionTargetStats | None = None,
        *,
        return_scores: bool = False,
    ):
        scores = np.asarray(
            [
                _between_group_score(
                    states[:, j], y, sample_weight=sample_weight, target_stats=target_stats
                )
                for j in range(states.shape[1])
            ],
            dtype=np.float64,
        )
        order = np.lexsort((np.arange(len(scores)), -scores))
        selected = order[: min(self.max_features, len(order))].astype(np.int64)
        if return_scores:
            return selected, scores
        return selected

    def _pair_rank(
        self,
        states: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        target_stats: _RegressionTargetStats | None = None,
        single_scores: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        if self.interaction_order < 2 or self.max_interactions <= 0:
            return []
        d = min(states.shape[1], self.max_interaction_features)
        # Pair scoring repeatedly scans feature columns.  On large candidate
        # banks, a one-time Fortran projection makes those columns contiguous.
        # Keep the historical view on smaller banks so the copy cannot dominate
        # short fits; both branches contain exactly the same state values.
        pair_view = states[:, :d]
        pair_states = (
            np.asarray(pair_view, dtype=states.dtype, order="F")
            if pair_view.size >= 500_000
            else pair_view
        )
        if single_scores is None:
            singles = np.asarray(
                [
                    _between_group_score(
                        pair_states[:, j],
                        y,
                        sample_weight=sample_weight,
                        target_stats=target_stats,
                    )
                    for j in range(d)
                ]
            )
        else:
            singles = np.asarray(single_scores, dtype=np.float64)[:d]
            if len(singles) < d:
                raise ValueError("single_scores is shorter than the pair candidate width")
        pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
        cardinalities = pair_states.max(axis=0, initial=0).astype(np.int64) + 1

        def score(pair):
            j, k = pair
            card_k = int(cardinalities[k])
            joint = pair_states[:, j].astype(np.int64) * card_k + pair_states[:, k]
            value = _between_group_score(
                joint, y, sample_weight=sample_weight, target_stats=target_stats
            ) - singles[j] - singles[k]
            return float(value), j, k

        if self.n_jobs in (None, 1) or len(pairs) < 64:
            # The pair state is a temporary sufficient statistic.  Reusing one
            # int64 buffer avoids allocating an n-row array for every pair and
            # does not alter any state code or reduction order.
            joint = np.empty(len(states), dtype=np.int64)
            scored = []
            for j, k in pairs:
                np.multiply(
                    pair_states[:, j], int(cardinalities[k]), out=joint, dtype=np.int64
                )
                np.add(joint, pair_states[:, k], out=joint)
                value = _between_group_score(
                    joint,
                    y,
                    sample_weight=sample_weight,
                    target_stats=target_stats,
                ) - singles[j] - singles[k]
                scored.append((float(value), j, k))
        else:
            scored = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(score)(pair) for pair in pairs
            )
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        return [(j, k) for _, j, k in scored[: self.max_interactions]]

    def _configs(self, available_pairs: int) -> list[RegressionConfig]:
        if self.reg_lambda == "auto":
            alphas = (0.1, 1.0, 10.0)
        else:
            value = float(self.reg_lambda)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reg_lambda must be 'auto' or a finite positive number")
            alphas = (value,)
        levels = (self.max_bins,) if self.preset == "balanced" else tuple(dict.fromkeys((4, self.max_bins)))
        if self.interaction_order < 2:
            counts = (0,)
        elif self.preset == "balanced":
            counts = tuple(dict.fromkeys((0, min(8, available_pairs))))
        else:
            counts = tuple(dict.fromkeys((0, min(8, available_pairs), available_pairs)))
        configs = [RegressionConfig(0, 0, 1.0)]
        configs.extend(
            RegressionConfig(level, count, alpha)
            for level in levels for count in counts for alpha in alphas
        )
        return list(dict.fromkeys(configs))

    def _codes(
        self,
        states: dict[int, np.ndarray],
        pairs: Sequence[tuple[int, int]],
        config: RegressionConfig,
        pair_cardinalities: Sequence[int] | None = None,
    ) -> np.ndarray:
        if config.max_main_level == 0:
            return np.empty((len(next(iter(states.values()))), 0), dtype=np.int64)
        active_levels = [level for level in self.levels if level <= config.max_main_level]
        pair_level = max(level for level in active_levels if level <= min(config.max_main_level, 8))
        columns: list[np.ndarray] = []
        for feature in range(states[active_levels[0]].shape[1]):
            for level in active_levels:
                columns.append(states[level][:, feature].astype(np.int64, copy=False))
        active_pairs = pairs[: config.n_pairs]
        if pair_cardinalities is None:
            pair_cardinalities = [
                int(states[pair_level][:, k].max(initial=0)) + 1
                for _, k in active_pairs
            ]
        if len(pair_cardinalities) < len(active_pairs):
            raise ValueError("pair_cardinalities is shorter than the active pair list")
        for pair_index, (j, k) in enumerate(active_pairs):
            card_k = int(pair_cardinalities[pair_index])
            right_state = states[pair_level][:, k].astype(np.int64)
            if np.any((right_state < 0) | (right_state >= card_k)):
                raise ValueError("pair state exceeds its fitted cardinality")
            columns.append(
                states[pair_level][:, j].astype(np.int64) * card_k + right_state
            )
        return np.column_stack(columns) if columns else np.zeros((len(next(iter(states.values()))), 1), dtype=np.int64)

    def _linear_positions(self, feature_idx: np.ndarray) -> np.ndarray:
        if not self.include_linear:
            return np.empty(0, dtype=np.int64)
        if self.feature_kinds is None:
            return np.arange(len(feature_idx), dtype=np.int64)
        allowed = {FeatureKind.NUMERIC.value, FeatureKind.EMBEDDING_RAW.value}
        return np.asarray(
            [position for position, original in enumerate(feature_idx) if self.feature_kinds[int(original)] in allowed],
            dtype=np.int64,
        )

    @staticmethod
    def _fit_linear_transform(
        X_selected: np.ndarray,
        positions: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        if len(positions) == 0:
            return sparse.csr_matrix((len(X_selected), 0), dtype=np.float64), np.empty(0), np.empty(0)
        values = np.asarray(X_selected[:, positions], dtype=np.float64)
        if sample_weight is None:
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            mean = np.average(values, axis=0, weights=weights)
            variance = np.average((values - mean) ** 2, axis=0, weights=weights)
            scale = np.sqrt(variance)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return sparse.csr_matrix((values - mean) / scale), mean, scale

    @staticmethod
    def _apply_linear_transform(X_selected: np.ndarray, positions: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        if len(positions) == 0:
            return sparse.csr_matrix((len(X_selected), 0), dtype=np.float64)
        values = np.asarray(X_selected[:, positions], dtype=np.float64)
        return sparse.csr_matrix((values - mean) / scale)

    def _fit_structure(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: RegressionConfig,
        sample_weight: np.ndarray | None = None,
        *,
        fit_head: bool = True,
        cache_training_design: bool = False,
    ):
        self.encoder_ = NestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=self.levels,
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
        )
        target_stats = _regression_target_stats(y, sample_weight)
        if X.shape[1] >= 512:
            # Wide-input exact projection: feature ranking only consumes the
            # finest fitted level.  Avoid materializing every quotient level
            # for hundreds of features that selection will discard; after
            # ranking, build the full hierarchy only for retained features.
            self.encoder_.fit(X)
            ranking_level = max(self.levels)
            ranking_states = self.encoder_.transform_level_columns(
                X, ranking_level, range(X.shape[1])
            )
            self.feature_idx_, all_feature_scores = self._feature_rank(
                ranking_states, y, sample_weight, target_stats,
                return_scores=True,
            )
            states = self.encoder_.transform_columns(X, self.feature_idx_)
        else:
            all_states = self.encoder_.fit_transform(X)
            self.feature_idx_, all_feature_scores = self._feature_rank(
                all_states[max(self.levels)],
                y,
                sample_weight,
                target_stats,
                return_scores=True,
            )
            states = {
                level: matrix[:, self.feature_idx_]
                for level, matrix in all_states.items()
            }
        if config.n_pairs > 0:
            self.pairs_ = self._pair_rank(
                states[max(self.levels)],
                y,
                sample_weight,
                target_stats,
                single_scores=all_feature_scores[self.feature_idx_],
            )[: config.n_pairs]
        else:
            self.pairs_ = []
        self.config_ = config
        active_levels = [level for level in self.levels if level <= config.max_main_level]
        if self.pairs_ and active_levels:
            pair_level = max(
                level for level in active_levels
                if level <= min(config.max_main_level, 8)
            )
            self.pair_cardinalities_ = np.asarray(
                [
                    self.encoder_.cardinalities_[pair_level][
                        int(self.feature_idx_[right])
                    ]
                    for _, right in self.pairs_
                ],
                dtype=np.int32,
            )
        else:
            self.pair_cardinalities_ = np.empty(0, dtype=np.int32)
        codes = self._codes(
            states,
            self.pairs_,
            config,
            self.pair_cardinalities_,
        )
        self.state_encoder_ = None
        state_design = sparse.csr_matrix((len(X), 0), dtype=np.float64)
        if config.max_main_level > 0:
            self.state_encoder_ = ReferenceStateEncoder()
            state_design = self.state_encoder_.fit_transform(codes)
        selected_X = X[:, self.feature_idx_]
        self.linear_positions_ = self._linear_positions(self.feature_idx_)
        linear_design, self.linear_mean_, self.linear_scale_ = self._fit_linear_transform(
            selected_X, self.linear_positions_, sample_weight
        )
        if cache_training_design:
            self._training_state_design_ = state_design
            self._training_linear_design_ = linear_design
        if fit_head:
            if linear_design.shape[1]:
                self.linear_regressor_ = _fit_ridge_lsqr_model_exact(
                    linear_design, y, 1.0, sample_weight
                )
                linear_prediction = self.linear_regressor_.predict(linear_design)
                self.linear_coef_ = np.asarray(self.linear_regressor_.coef_, dtype=np.float64).reshape(-1)
                self.linear_intercept_ = float(self.linear_regressor_.intercept_)
            else:
                self.linear_regressor_ = None
                linear_prediction = np.zeros(len(y), dtype=np.float64)
                self.linear_coef_ = np.empty(0, dtype=np.float64)
                self.linear_intercept_ = 0.0
            residual = y - linear_prediction
            self.lookup_ = []
            if state_design.shape[1]:
                self.regressor_ = _fit_ridge_lsqr_model_exact(
                    state_design, residual, float(config.alpha), sample_weight
                )
                state_coef = np.asarray(self.regressor_.coef_, dtype=np.float64).reshape(-1)
                offset = 0
                for card in self.state_encoder_.cardinalities_:
                    table = np.zeros(int(card), dtype=np.float64)
                    width = max(int(card) - 1, 0)
                    if width:
                        table[1:] = state_coef[offset : offset + width]
                    offset += width
                    self.lookup_.append(table)
                if offset != len(state_coef):
                    raise RuntimeError("regression coefficient layout mismatch")
                self.intercept_ = float(self.regressor_.intercept_)
            else:
                self.regressor_ = None
                self.intercept_ = 0.0
        else:
            self.linear_regressor_ = None
            self.regressor_ = None
            self.linear_coef_ = np.zeros(linear_design.shape[1], dtype=np.float64)
            self.linear_intercept_ = 0.0
            self.lookup_ = [
                np.zeros(int(cardinality), dtype=np.float64)
                for cardinality in (
                    () if self.state_encoder_ is None else self.state_encoder_.cardinalities_
                )
            ]
            self.intercept_ = 0.0
        self.design_dim_ = int(state_design.shape[1] + linear_design.shape[1])
        self.model_bytes_estimate_ = int(
            sum(table.nbytes for table in self.lookup_)
            + self.encoder_.threshold_bytes_
            + self.feature_idx_.nbytes
            + np.asarray(self.pairs_, dtype=np.int32).nbytes
            + self.pair_cardinalities_.nbytes
            + self.encoder_.direct_state_mask_.nbytes
            + self.encoder_.direct_state_cardinalities_.nbytes
            + self.linear_coef_.nbytes + self.linear_mean_.nbytes + self.linear_scale_.nbytes + 8
        )
        self._prepare_execution_maps()
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        _fit_final_head: bool = True,
        _cache_training_design: bool = False,
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = (
            None
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        forced = getattr(self, "_forced_representation_config", None)
        if forced is not None:
            config = RegressionConfig(
                int(forced[0]), int(forced[1]), float(forced[2])
            )
            self.validation_scores_ = []
            self.selection_tolerance_ = 0.0
            return self._fit_structure(
                X,
                y,
                config,
                weights,
                fit_head=_fit_final_head,
                cache_training_design=_cache_training_design,
            )
        X_selection, y_selection, weight_selection = self._selection_sample(
            X, y, weights
        )
        if len(y_selection) < 12:
            config = RegressionConfig(self.max_bins, 0, 1.0 if self.reg_lambda == "auto" else float(self.reg_lambda))
            self.validation_scores_ = []
            return self._fit_structure(
                X, y, config, weights,
                fit_head=_fit_final_head,
                cache_training_design=_cache_training_design,
            )
        if weight_selection is None:
            Xa, Xv, ya, yv = train_test_split(
                X_selection,
                y_selection,
                test_size=0.22,
                random_state=self.random_state,
            )
            wa = wv = None
        else:
            Xa, Xv, ya, yv, wa, wv = train_test_split(
                X_selection,
                y_selection,
                weight_selection,
                test_size=0.22,
                random_state=self.random_state,
            )
        encoder = NestedQuantileEncoder(
            max_bins=self.max_bins,
            levels=self.levels,
            feature_kinds=self.feature_kinds,
            feature_cardinalities=self.feature_cardinalities,
        )
        selection_target_stats = _regression_target_stats(ya, wa)
        if Xa.shape[1] >= 512:
            encoder.fit(Xa)
            ranking_level = max(self.levels)
            Sa_rank = encoder.transform_level_columns(
                Xa, ranking_level, range(Xa.shape[1])
            )
            feature_idx, all_feature_scores = self._feature_rank(
                Sa_rank, ya, wa, selection_target_stats, return_scores=True
            )
            Sa = encoder.transform_columns(Xa, feature_idx)
            Sv = encoder.transform_columns(Xv, feature_idx)
        else:
            Sa_all = encoder.fit_transform(Xa)
            Sv_all = encoder.transform(Xv)
            feature_idx, all_feature_scores = self._feature_rank(
                Sa_all[max(self.levels)],
                ya,
                wa,
                selection_target_stats,
                return_scores=True,
            )
            Sa = {
                level: matrix[:, feature_idx]
                for level, matrix in Sa_all.items()
            }
            Sv = {
                level: matrix[:, feature_idx]
                for level, matrix in Sv_all.items()
            }
        pairs = self._pair_rank(
            Sa[max(self.levels)],
            ya,
            wa,
            selection_target_stats,
            single_scores=all_feature_scores[feature_idx],
        )
        configs = self._configs(len(pairs))
        rows = []
        linear_positions = self._linear_positions(feature_idx)
        train_linear, linear_mean, linear_scale = self._fit_linear_transform(
            Xa[:, feature_idx], linear_positions, wa
        )
        valid_linear = self._apply_linear_transform(
            Xv[:, feature_idx], linear_positions, linear_mean, linear_scale
        )
        if train_linear.shape[1]:
            linear_coef_path, linear_intercepts, _ = (
                fit_ridge_lsqr_alpha_path_exact(
                    train_linear,
                    ya,
                    np.asarray([1.0], dtype=np.float64),
                    sample_weight=wa,
                    n_jobs=self.n_jobs,
                )
            )
            linear_coef = linear_coef_path[0]
            linear_intercept = linear_intercepts[0]
            train_residual = ya - (
                train_linear @ linear_coef + linear_intercept
            )
            valid_baseline = (
                valid_linear @ linear_coef + linear_intercept
            )
        else:
            train_residual = ya
            valid_baseline = np.zeros(len(yv), dtype=np.float64)
        structure_design_cache: dict[
            tuple[int, int], tuple[sparse.csr_matrix, sparse.csr_matrix]
        ] = {}
        positive_configs = [
            config for config in configs if config.max_main_level > 0
        ]
        # Within one quotient level, pair budgets are strict prefixes of the
        # same code bank.  Build the largest design once and take exact CSR
        # prefix views for smaller budgets.  The resulting indptr, indices and
        # data arrays are identical to independent ReferenceStateEncoder fits.
        for main_level in dict.fromkeys(
            config.max_main_level for config in positive_configs
        ):
            level_configs = [
                config
                for config in positive_configs
                if config.max_main_level == main_level
            ]
            max_pair_count = max(config.n_pairs for config in level_configs)
            active_levels = [
                level for level in self.levels if level <= main_level
            ]
            pair_level = max(
                level
                for level in active_levels
                if level <= min(main_level, 8)
            )
            max_pair_cards = np.asarray(
                [
                    encoder.cardinalities_[pair_level][int(feature_idx[right])]
                    for _, right in pairs[:max_pair_count]
                ],
                dtype=np.int32,
            )
            max_config = RegressionConfig(main_level, max_pair_count, 1.0)
            max_train_codes = self._codes(
                Sa, pairs, max_config, max_pair_cards
            )
            max_valid_codes = self._codes(
                Sv, pairs, max_config, max_pair_cards
            )
            max_encoder = ReferenceStateEncoder()
            max_train_design = max_encoder.fit_transform(max_train_codes)
            max_valid_design = max_encoder.transform(max_valid_codes)
            main_code_columns = len(feature_idx) * len(active_levels)
            for pair_count in dict.fromkeys(
                config.n_pairs for config in level_configs
            ):
                code_columns = main_code_columns + int(pair_count)
                design_width = int(max_encoder.offsets_[code_columns])
                if design_width == max_train_design.shape[1]:
                    train_prefix = max_train_design
                    valid_prefix = max_valid_design
                else:
                    train_prefix = max_train_design[:, :design_width].tocsr()
                    valid_prefix = max_valid_design[:, :design_width].tocsr()
                structure_design_cache[(main_level, pair_count)] = (
                    train_prefix,
                    valid_prefix,
                )

        def append_validation_row(config, train_state_design, prediction):
            squared_error = (prediction - yv) ** 2
            if wv is None:
                mse = float(np.mean(squared_error))
                se_mse = (
                    float(
                        np.std(squared_error, ddof=1)
                        / np.sqrt(len(squared_error))
                    )
                    if len(squared_error) > 1
                    else 0.0
                )
            else:
                mse = float(np.average(squared_error, weights=wv))
                centered = squared_error - mse
                variance = float(np.average(centered**2, weights=wv))
                effective_n = float(wv.sum() ** 2 / np.dot(wv, wv))
                se_mse = float(np.sqrt(variance / max(effective_n, 1.0)))
            rows.append(
                {
                    "max_main_level": config.max_main_level,
                    "n_pairs": config.n_pairs,
                    "alpha": config.alpha,
                    "mse": mse,
                    "se_mse": se_mse,
                    "design_dimension": int(
                        train_state_design.shape[1] + train_linear.shape[1]
                    ),
                }
            )

        config_index = 0
        while config_index < len(configs):
            config = configs[config_index]
            if config.max_main_level == 0:
                train_state_design = sparse.csr_matrix(
                    (len(ya), 0), dtype=np.float64
                )
                append_validation_row(config, train_state_design, valid_baseline)
                config_index += 1
                continue

            structure_key = (config.max_main_level, config.n_pairs)
            structure_configs = []
            while config_index < len(configs):
                candidate = configs[config_index]
                if (
                    candidate.max_main_level,
                    candidate.n_pairs,
                ) != structure_key:
                    break
                structure_configs.append(candidate)
                config_index += 1
            train_state_design, valid_state_design = structure_design_cache[
                structure_key
            ]
            coefficients, intercepts, _ = fit_ridge_lsqr_alpha_path_exact(
                train_state_design,
                train_residual,
                np.asarray(
                    [candidate.alpha for candidate in structure_configs],
                    dtype=np.float64,
                ),
                sample_weight=wa,
                n_jobs=self.n_jobs,
            )
            for path_index, candidate in enumerate(structure_configs):
                state_prediction = (
                    valid_state_design @ coefficients[path_index]
                    + intercepts[path_index]
                )
                prediction = valid_baseline + state_prediction
                append_validation_row(
                    candidate, train_state_design, prediction
                )
        rows.sort(key=lambda row: (row["mse"], row["design_dimension"], row["alpha"]))
        best_risk = rows[0]
        tolerance = best_risk["se_mse"]
        eligible = [row for row in rows if row["mse"] <= best_risk["mse"] + tolerance]
        best = min(eligible, key=lambda row: (row["design_dimension"], row["mse"], row["alpha"]))
        self.validation_scores_ = rows
        self.selection_tolerance_ = float(tolerance)
        config = RegressionConfig(best["max_main_level"], best["n_pairs"], best["alpha"])
        return self._fit_structure(
            X, y, config, weights,
            fit_head=_fit_final_head,
            cache_training_design=_cache_training_design,
        )

    def _states(self, X: np.ndarray) -> dict[int, np.ndarray]:
        # Exact projected transform: inference only references ``feature_idx_``.
        # Transforming those raw columns directly is bitwise identical to
        # ``encoder_.transform(X)[level][:, feature_idx_]`` while avoiding work
        # for every feature discarded by representation selection.
        return self.encoder_.transform_columns(
            np.asarray(X, dtype=np.float64), self.feature_idx_
        )

    def _prepare_execution_maps(self) -> None:
        """Compile exact parent maps from one execution quotient level.

        The historical regression executor materialized one state matrix for
        every active quotient level.  Nested quotient labels are deterministic
        functions of the finest active level, so inference can generate that
        level once and recover every lower state through tiny integer maps.
        This changes no lookup ordering or floating-point accumulation order.
        """
        active_levels = tuple(
            level for level in self.levels if level <= int(self.config_.max_main_level)
        )
        self.execution_active_levels_ = active_levels
        if not active_levels:
            self.execution_level_ = 0
            self.execution_parent_maps_ = tuple()
            self.execution_pair_level_ = 0
            self.execution_pair_plans_ = tuple()
            self.execution_code_maxima_ = tuple()
            return
        execution_level = int(max(active_levels))
        self.execution_level_ = execution_level
        parent_maps = []
        code_maxima = []
        for raw_feature in np.asarray(self.feature_idx_, dtype=np.int64):
            raw_feature = int(raw_feature)
            child_card = int(
                self.encoder_.cardinalities_[execution_level][raw_feature]
            )
            feature_maps = []
            for level in active_levels:
                if int(level) == execution_level:
                    mapping = np.arange(child_card, dtype=np.int32)
                else:
                    mapping = _parent_labels_from_fine(
                        self.encoder_.maps_[execution_level][raw_feature],
                        self.encoder_.maps_[int(level)][raw_feature],
                        child_card,
                    ).astype(np.int32, copy=False)
                feature_maps.append(mapping)
                code_maxima.append(int(np.max(mapping, initial=0)))
            parent_maps.append(tuple(feature_maps))
        self.execution_parent_maps_ = tuple(parent_maps)
        self.execution_pair_level_ = int(
            max(
                level
                for level in active_levels
                if level <= min(int(self.config_.max_main_level), 8)
            )
        )
        self.execution_pair_level_index_ = active_levels.index(
            self.execution_pair_level_
        )
        pair_map_index = int(self.execution_pair_level_index_)
        pair_plans = []
        for pair_index, (left, right) in enumerate(self.pairs_):
            left = int(left)
            right = int(right)
            left_map = self.execution_parent_maps_[left][pair_map_index]
            right_map = self.execution_parent_maps_[right][pair_map_index]
            exec_right_cardinality = int(len(right_map))
            pair_code_map = (
                left_map[:, None].astype(np.int64, copy=False)
                * int(self.pair_cardinalities_[pair_index])
                + right_map[None, :]
            ).reshape(-1)
            pair_plans.append(
                (left, right, exec_right_cardinality, pair_code_map)
            )
            code_maxima.append(int(np.max(pair_code_map, initial=0)))
        self.execution_pair_plans_ = tuple(pair_plans)
        self.execution_code_maxima_ = tuple(code_maxima)

    def _ensure_execution_maps(self) -> None:
        if not hasattr(self, "execution_parent_maps_"):
            self._prepare_execution_maps()

    def _execution_states(self, X: np.ndarray, *, projected: bool = False) -> np.ndarray:
        self._ensure_execution_maps()
        if not self.execution_active_levels_:
            return np.empty((len(X), len(self.feature_idx_)), dtype=np.int16)
        if projected:
            return self.encoder_.transform_level_projected(
                X, self.execution_level_, self.feature_idx_
            )
        return self.encoder_.transform_level_columns(
            X, self.execution_level_, self.feature_idx_
        )

    def _decision_from_execution_states_with_lookup(
        self,
        execution_states: np.ndarray,
        lookup,
        intercept,
    ):
        """Execute the historical lookup stream from one quotient state bank.

        Full-domain operators reuse one gather scratch buffer.  ``np.take``
        copies the exact table values and ``np.add`` performs the same ordered
        floating-point additions while avoiding one temporary allocation per
        operator.  Operators whose future state domain can exceed the fitted
        table retain the historical validity-mask path.
        """
        self._ensure_execution_maps()
        intercept_array = np.asarray(intercept, dtype=np.float64)
        if intercept_array.ndim == 0:
            prediction = np.full(
                len(execution_states), float(intercept_array), dtype=np.float64
            )
            multi_head = False
            scratch = np.empty(len(execution_states), dtype=np.float64)
        else:
            prediction = np.broadcast_to(
                intercept_array, (len(execution_states), len(intercept_array))
            ).copy()
            multi_head = True
            scratch = np.empty_like(prediction)
        lookup_index = 0

        def accumulate(state):
            nonlocal lookup_index, prediction
            table = lookup[lookup_index]
            full_domain = (
                lookup_index < len(self.execution_code_maxima_)
                and self.execution_code_maxima_[lookup_index] < len(table)
            )
            if full_domain:
                if multi_head:
                    np.take(table, state, axis=0, out=scratch)
                else:
                    np.take(table, state, out=scratch)
                np.add(prediction, scratch, out=prediction)
            else:
                valid = (state >= 0) & (state < len(table))
                if multi_head:
                    prediction[valid, :] += table[state[valid], :]
                else:
                    prediction[valid] += table[state[valid]]
            lookup_index += 1

        for feature, feature_maps in enumerate(self.execution_parent_maps_):
            execution_state = execution_states[:, feature]
            for parent_map in feature_maps:
                accumulate(parent_map[execution_state])
        if self.pairs_:
            joint_scratch = np.empty(len(execution_states), dtype=np.int64)
            code_scratch = np.empty(len(execution_states), dtype=np.int64)
            for left, right, exec_right_cardinality, pair_code_map in self.execution_pair_plans_:
                np.multiply(
                    execution_states[:, int(left)],
                    int(exec_right_cardinality),
                    out=joint_scratch,
                    dtype=np.int64,
                )
                np.add(
                    joint_scratch, execution_states[:, int(right)], out=joint_scratch
                )
                np.take(pair_code_map, joint_scratch, out=code_scratch)
                accumulate(code_scratch)
        if lookup_index != len(lookup):
            raise RuntimeError(
                f"regression execution lookup mismatch: consumed {lookup_index}, "
                f"expected {len(lookup)}"
            )
        return prediction

    def decision_function_projected(self, X_selected: np.ndarray) -> np.ndarray:
        """Predict from columns already projected to ``feature_idx_`` order."""
        selected_X = np.asarray(X_selected, dtype=np.float64)
        if selected_X.ndim != 2 or selected_X.shape[1] != len(self.feature_idx_):
            raise ValueError("projected regression input width mismatch")
        if self.config_.max_main_level > 0:
            execution_states = self._execution_states(selected_X, projected=True)
            prediction = self._decision_from_execution_states_with_lookup(
                execution_states, self.lookup_, self.intercept_
            )
        else:
            prediction = np.full(len(selected_X), self.intercept_, dtype=np.float64)
        if len(self.linear_positions_):
            linear = (
                selected_X[:, self.linear_positions_] - self.linear_mean_
            ) / self.linear_scale_
            prediction += self.linear_intercept_ + linear @ self.linear_coef_
        return prediction

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        selected_X = None
        if self.config_.max_main_level > 0:
            execution_states = self._execution_states(array, projected=False)
            prediction = self._decision_from_execution_states_with_lookup(
                execution_states, self.lookup_, self.intercept_
            )
        else:
            prediction = np.full(len(array), self.intercept_, dtype=np.float64)
        if len(self.linear_positions_):
            selected_X = array[:, self.feature_idx_]
            linear = (
                selected_X[:, self.linear_positions_] - self.linear_mean_
            ) / self.linear_scale_
            prediction += self.linear_intercept_ + linear @ self.linear_coef_
        return prediction

    predict = decision_function

    def export_ir(self, prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        lookup_offsets = np.concatenate([[0], np.cumsum([len(table) for table in self.lookup_])]).astype(np.int64)
        lookup_values = np.concatenate(self.lookup_) if self.lookup_ else np.empty(0)
        arrays: dict[str, Any] = {
            "feature_indices": self.feature_idx_.astype(np.int32),
            "pairs": np.asarray(self.pairs_, dtype=np.int32).reshape(-1, 2),
            "pair_cardinalities": self.pair_cardinalities_.astype(np.int32),
            "direct_state_mask": self.encoder_.direct_state_mask_.astype(np.uint8),
            "direct_state_cardinalities": self.encoder_.direct_state_cardinalities_.astype(np.int32),
            "lookup_offsets": lookup_offsets,
            "lookup_values": lookup_values,
            "intercept": np.asarray([self.intercept_]),
            "linear_positions": self.linear_positions_.astype(np.int32),
            "linear_mean": self.linear_mean_,
            "linear_scale": self.linear_scale_,
            "linear_coef": self.linear_coef_,
            "linear_intercept": np.asarray([self.linear_intercept_]),
        }
        for index, thresholds in enumerate(self.encoder_.thresholds_):
            arrays[f"thresholds_{index}"] = thresholds
        for level in self.levels:
            for index, mapping in enumerate(self.encoder_.maps_[level]):
                arrays[f"map_{level}_{index}"] = mapping
        np.savez_compressed(npz_path, **arrays)
        metadata = {
            "format": "cerm-finite-state-regression-v2",
            "task_type": "regression",
            "levels": list(self.levels),
            "max_main_level": self.config_.max_main_level,
            "n_pairs": self.config_.n_pairs,
            "alpha": self.config_.alpha,
            "design_dimension": self.design_dim_,
            "model_bytes_estimate": self.model_bytes_estimate_,
            "array_file": npz_path.name,
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return npz_path, json_path


@dataclass
class CompiledRegressionProgram:
    predict_raw: Any
    adapter: RegressionTypedAdapter | None
    input_columns: tuple[str, ...] | None
    source_path: Path
    library_path: Path
    feature_indices: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _matrix(self, X):
        if self.adapter is None:
            matrix = dense_numeric_matrix(X, self.input_columns)
            return _select_features(matrix, self.feature_indices)
        frame = validate_dataframe_schema(X, self.input_columns or ())
        if self.feature_indices is not None and hasattr(self.adapter, "transform_columns"):
            transformed = self.adapter.transform_columns(frame, self.feature_indices)
            return transformed.matrix if hasattr(transformed, "matrix") else transformed
        transformed = self.adapter.transform(frame)
        matrix = transformed.matrix if hasattr(transformed, "matrix") else transformed
        return _select_features(matrix, self.feature_indices)

    def decision_function(self, X):
        return np.asarray(self.predict_raw(self._matrix(X)), dtype=np.float64)

    predict = decision_function

    @property
    def model_bytes_estimate(self) -> int:
        adapter_bytes = 0 if self.adapter is None else self.adapter.nbytes
        library_bytes = self.library_path.stat().st_size if self.library_path.is_file() else 0
        return int(adapter_bytes + library_bytes)

    @staticmethod
    def _record(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def save(self, directory: str | Path, *, include_source: bool = False) -> Path:
        """Save a platform-specific predictor without pickle state."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / self.library_path.name
        shutil.copy2(self.library_path, library)
        source_record = None
        if include_source and self.source_path.is_file():
            source = directory / self.source_path.name
            shutil.copy2(self.source_path, source)
            source_record = self._record(source)

        adapter_record = None
        if self.adapter is not None:
            if isinstance(self.adapter, PortableRegressionAdapter):
                adapter_npz, adapter_json = self.adapter.export(directory / "adapter")
                portable = bool(self.adapter.metadata.get("portable", False))
            else:
                adapter_npz, adapter_json = export_regression_adapter_ir(
                    self.adapter, directory / "adapter"
                )
                adapter_metadata = json.loads(
                    adapter_json.read_text(encoding="utf-8")
                )
                portable = bool(adapter_metadata.get("portable", False))
            if not portable:
                raise ValueError(
                    "compiled regression adapter contains non-portable category values"
                )
            adapter_record = {
                "format": "cerm-regression-adapter-ir-v1",
                "portable": True,
                "npz": self._record(adapter_npz),
                "json": self._record(adapter_json),
            }

        runtime_path = directory / "runtime.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "input_columns": _jsonable(self.input_columns),
                    "feature_indices": _jsonable(self.feature_indices),
                    "metadata": _jsonable(self.metadata),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest = {
            "format": "cerm-compiled-regression-v2",
            "library": self._record(library),
            "source": source_record,
            "runtime": self._record(runtime_path),
            "adapter": adapter_record,
            "symbol": self.metadata.get("symbol", "cerm_regression_predict"),
            "machine": platform.machine(),
            "system": platform.system(),
            "metadata": _jsonable(self.metadata),
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "CompiledRegressionProgram":
        from .native_runtime import load_native_predictor

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_name = manifest.get("format")
        if format_name not in {
            "cerm-compiled-regression-v1",
            "cerm-compiled-regression-v2",
        }:
            raise ValueError(f"unsupported compiled regression format: {format_name!r}")
        if manifest.get("machine") != platform.machine() or manifest.get("system") != platform.system():
            raise RuntimeError("compiled regression artifact targets a different platform")

        def verify(record):
            artifact = manifest_path.parent / record["file"]
            payload = artifact.read_bytes()
            if len(payload) != int(record["bytes"]):
                raise ValueError(f"byte-count mismatch for {artifact.name}")
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {artifact.name}")
            return artifact

        library = verify(manifest["library"])
        source_record = manifest.get("source")
        source = verify(source_record) if source_record else manifest_path.parent / "unavailable.cpp"
        symbol = manifest.get("symbol", "cerm_regression_predict")

        if format_name == "cerm-compiled-regression-v1":
            state = joblib.load(verify(manifest["state"]))
            adapter = state.get("adapter")
            input_columns = state.get("input_columns")
            feature_indices = state.get("feature_indices")
            metadata = dict(state.get("metadata") or {})
        else:
            runtime = json.loads(verify(manifest["runtime"]).read_text(encoding="utf-8"))
            adapter_record = manifest.get("adapter")
            adapter = None
            if adapter_record is not None:
                if not adapter_record.get("portable", False):
                    raise ValueError("compiled regression adapter is not portable")
                adapter = PortableRegressionAdapter.load(
                    verify(adapter_record["json"]),
                    verify(adapter_record["npz"]),
                )
            input_columns = runtime.get("input_columns")
            feature_indices = runtime.get("feature_indices")
            metadata = dict(runtime.get("metadata") or {})

        return cls(
            predict_raw=load_native_predictor(library, symbol=symbol),
            adapter=adapter,
            input_columns=tuple(input_columns) if input_columns is not None else None,
            source_path=source,
            library_path=library,
            feature_indices=tuple(feature_indices) if feature_indices is not None else None,
            metadata={**metadata, "reloaded": True, "symbol": symbol},
        )


@dataclass
class RegressionSemanticProgram:
    model: FiniteStateRidgeRegressor
    adapter: RegressionTypedAdapter | None
    input_columns: tuple[str, ...] | None
    adapted_feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...] | None
    library_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def _matrix(self, X):
        if self.adapter is None:
            matrix = dense_numeric_matrix(X, self.input_columns)
            return _select_features(matrix, self.feature_indices)
        frame = validate_dataframe_schema(X, self.input_columns or ())
        if self.feature_indices is not None and hasattr(self.adapter, "transform_columns"):
            return self.adapter.transform_columns(frame, self.feature_indices).matrix
        matrix = self.adapter.transform(frame).matrix
        return _select_features(matrix, self.feature_indices)

    def decision_function(self, X):
        if (
            self.adapter is not None
            and self.feature_indices is not None
            and hasattr(self.adapter, "transform_columns")
            and hasattr(self.model, "feature_idx_")
            and hasattr(self.model, "decision_function_projected")
        ):
            outer = np.asarray(self.feature_indices, dtype=np.int64)
            core = np.asarray(self.model.feature_idx_, dtype=np.int64)
            frame = validate_dataframe_schema(X, self.input_columns or ())
            projected = self.adapter.transform_columns(frame, outer[core]).matrix
            return self.model.decision_function_projected(projected)
        return self.model.decision_function(self._matrix(X))

    predict = decision_function

    @property
    def model_bytes_estimate(self) -> int:
        adapter_bytes = 0 if self.adapter is None else self.adapter.nbytes
        return int(self.model.model_bytes_estimate_ + adapter_bytes)

    def optimize(self, target: str = "balanced") -> "RegressionSemanticProgram":
        if target not in {"memory", "balanced", "latency"}:
            raise ValueError("target must be memory, balanced, or latency")
        self.metadata["optimized_for"] = target
        self.metadata["lookup_fused"] = True
        return self

    def compile_native(self, prefix: str | Path) -> CompiledRegressionProgram:
        from ._internal.cerm_regression_native_codegen import compile_regression_native

        project_adapter = self.adapter is not None and self.feature_indices is not None
        predict, source, library = compile_regression_native(
            self.model, prefix, projected_input=project_adapter
        )
        compiled_feature_indices = self.feature_indices
        if project_adapter:
            outer = np.asarray(self.feature_indices, dtype=np.int64)
            compiled_feature_indices = tuple(
                int(index)
                for index in outer[np.asarray(self.model.feature_idx_, dtype=np.int64)]
            )
        return CompiledRegressionProgram(
            predict_raw=predict,
            adapter=self.adapter,
            input_columns=self.input_columns,
            source_path=Path(source),
            library_path=Path(library),
            feature_indices=compiled_feature_indices,
            metadata={
                "backend": "finite-state-regression-native",
                "symbol": "cerm_regression_predict",
                "task_type": "regression",
                "projected_adapter_input": bool(project_adapter),
            },
        )

    def export(self, directory: str | Path, *, config: dict[str, Any] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_npz, model_json = self.model.export_ir(directory / "model")
        adapter_record = None
        if self.adapter is not None:
            adapter_npz, adapter_json = export_regression_adapter_ir(
                self.adapter, directory / "adapter"
            )
            adapter_metadata = json.loads(adapter_json.read_text(encoding="utf-8"))
            adapter_record = {
                "npz": None,
                "json": None,
                "portable": bool(adapter_metadata.get("portable", False)),
                "format": "cerm-regression-adapter-ir-v1",
            }
        def record(path: Path):
            payload = path.read_bytes()
            return {"file": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        if adapter_record is not None:
            adapter_record["npz"] = record(adapter_npz)
            adapter_record["json"] = record(adapter_json)
        manifest = {
            "format": "cerm-python-package-v3",
            "task_type": "regression",
            "n_outputs": 1,
            "library_version": self.library_version,
            "input_columns": _jsonable(self.input_columns),
            "adapted_feature_names": _jsonable(self.adapted_feature_names),
            "feature_indices": _jsonable(self.feature_indices),
            "model": {"npz": record(model_npz), "json": record(model_json)},
            "adapter": adapter_record,
            "config": _jsonable(config or {}),
            "metadata": _jsonable(self.metadata),
            "model_bytes_estimate": self.model_bytes_estimate,
        }
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


def _encode_portable_scalar(value: Any) -> dict[str, Any]:
    """Encode common categorical scalar values without pickle execution."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, int):
        return {"type": "int", "value": int(value)}
    if isinstance(value, float):
        if not np.isfinite(value):
            return {"type": "float-repr", "value": repr(float(value))}
        return {"type": "float", "value": float(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes-hex", "value": value.hex()}
    if isinstance(value, tuple):
        items = [_encode_portable_scalar(item) for item in value]
        return {
            "type": "tuple",
            "items": items,
            "portable": all(item.get("portable", True) for item in items),
        }
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        return {"type": "timestamp", "value": timestamp.isoformat()}
    return {
        "type": "unsupported",
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
        "portable": False,
    }


def _decode_portable_scalar(payload: dict[str, Any]) -> Any:
    kind = payload.get("type")
    if kind == "none":
        return None
    if kind == "bool":
        return bool(payload["value"])
    if kind == "int":
        return int(payload["value"])
    if kind == "float":
        return float(payload["value"])
    if kind == "float-repr":
        return float(payload["value"])
    if kind == "str":
        return str(payload["value"])
    if kind == "bytes-hex":
        return bytes.fromhex(payload["value"])
    if kind == "tuple":
        return tuple(_decode_portable_scalar(item) for item in payload.get("items", []))
    if kind == "timestamp":
        return pd.Timestamp(payload["value"])
    raise ValueError(
        "regression adapter contains an unsupported categorical value: "
        f"{payload.get('python_type', kind)!r}"
    )


def export_regression_adapter_ir(
    adapter: RegressionTypedAdapter, prefix: str | Path
) -> tuple[Path, Path]:
    """Export a fitted regression adapter without Python pickle state."""

    prefix = Path(prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "format": "cerm-regression-adapter-ir-v1",
        "adapter_version": adapter.VERSION,
        "task_type": "regression",
        "input_columns": list(adapter.input_columns_),
        "output_feature_names": list(adapter.feature_names_),
        "output_feature_kinds": list(adapter.feature_kinds_),
        "output_cardinalities": _jsonable(adapter.cardinalities_),
        "category_identity": adapter.category_identity,
        "missing_policy": adapter.missing_policy,
        "portable": True,
        "numeric": [],
        "categorical": [],
    }
    for index, column in enumerate(adapter.numeric_columns_):
        key = f"numeric_median_{index}"
        arrays[key] = np.asarray([adapter.numeric_medians_[column]], dtype=np.float64)
        manifest["numeric"].append(
            {
                "column": str(column),
                "column_index": int(adapter.input_columns_.index(column)),
                "median_array": key,
                "emit_missing_state": column in adapter.numeric_missing_columns_,
            }
        )
    for index, column in enumerate(adapter.categorical_columns):
        encoder = adapter.category_encoders_[column]
        mode = adapter.category_modes_[column]
        entry: dict[str, Any] = {
            "column": str(column),
            "column_index": int(adapter.input_columns_.index(column)),
            "mode": mode,
        }
        if mode == "identity":
            mapping = [
                {"key": _encode_portable_scalar(key), "state": int(value)}
                for key, value in encoder.mapping_.items()
            ]
            if not all(item["key"].get("portable", True) for item in mapping):
                manifest["portable"] = False
            entry.update(
                {
                    "mapping": mapping,
                    "cardinality": int(encoder.cardinality_),
                    "representation": adapter.category_identity,
                    "missing_state": int(encoder.missing_state_),
                    "unknown_state": int(
                        encoder.missing_state_
                        if getattr(encoder, "has_reserved_missing_", False)
                        else 0
                    ),
                }
            )
        else:
            threshold_key = f"category_thresholds_{index}"
            arrays[threshold_key] = np.asarray(encoder.thresholds_, dtype=np.float64)
            entry.update(
                {
                    "mapping": {str(key): float(value) for key, value in encoder.mapping_.items()},
                    "prior": float(encoder.prior_),
                    "threshold_array": threshold_key,
                    "cardinality": int(encoder.cardinality_),
                    "key_format": "type-name-and-repr",
                }
            )
        manifest["categorical"].append(entry)
    np.savez_compressed(npz_path, **arrays)
    manifest["array_file"] = npz_path.name
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return npz_path, json_path


@dataclass
class PortableRegressionAdapter:
    """Execution-only adapter reconstructed from JSON/NPZ regression IR."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, json_path: str | Path, npz_path: str | Path):
        metadata = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if metadata.get("format") != "cerm-regression-adapter-ir-v1":
            raise ValueError(f"unsupported regression adapter format: {metadata.get('format')!r}")
        if not metadata.get("portable", False):
            raise ValueError("regression adapter contains non-portable categorical values")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return cls(metadata=metadata, arrays=arrays)

    def export(self, prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(prefix)
        npz_path = prefix.with_suffix(".npz")
        json_path = prefix.with_suffix(".json")
        np.savez_compressed(npz_path, **self.arrays)
        metadata = dict(self.metadata)
        metadata["array_file"] = npz_path.name
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return npz_path, json_path

    @property
    def input_columns(self) -> tuple[Any, ...]:
        return tuple(self.metadata["input_columns"])

    @property
    def nbytes(self) -> int:
        metadata_bytes = len(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))
        return int(metadata_bytes + sum(array.nbytes for array in self.arrays.values()))

    def transform_columns(self, X: pd.DataFrame, output_indices) -> np.ndarray:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, self.input_columns))
        indices = np.asarray(output_indices, dtype=np.int64).reshape(-1)
        output_dim = self.metadata.get("output_dim")
        if output_dim is None:
            output_dim = 0
            for entry in self.metadata.get("numeric", []):
                output_dim += 1 + int(bool(entry.get("emit_missing_state", False)))
            representation = self.metadata.get("category_identity", "state")
            for entry in self.metadata.get("categorical", []):
                if entry["mode"] == "identity" and representation == "binary":
                    output_dim += max(
                        1, int(np.ceil(np.log2(max(int(entry["cardinality"]), 2))))
                    )
                else:
                    output_dim += 1
        output_dim = int(output_dim)
        if np.any((indices < 0) | (indices >= output_dim)):
            raise IndexError("portable regression adapter output index out of range")
        if len(indices) == output_dim and np.array_equal(
            indices, np.arange(output_dim, dtype=np.int64)
        ):
            return self.transform(frame)
        wanted = set(int(i) for i in indices)
        generated: dict[int, np.ndarray] = {}
        out_index = 0
        for entry in self.metadata.get("numeric", []):
            column = self.input_columns[int(entry["column_index"])]
            slots = [(out_index, "numeric")]
            out_index += 1
            if entry.get("emit_missing_state", False):
                slots.append((out_index, "missing"))
                out_index += 1
            if not any(slot in wanted for slot, _ in slots):
                continue
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            median = float(self.arrays[entry["median_array"]][0])
            filled = np.where(missing, median, values)
            for slot, kind in slots:
                if slot in wanted:
                    generated[slot] = filled if kind == "numeric" else missing.astype(np.int16)

        representation = self.metadata.get("category_identity", "state")
        for entry in self.metadata.get("categorical", []):
            column = self.input_columns[int(entry["column_index"])]
            if entry["mode"] == "identity" and representation == "binary":
                width = max(1, int(np.ceil(np.log2(max(int(entry["cardinality"]), 2)))))
            else:
                width = 1
            slots = list(range(out_index, out_index + width))
            out_index += width
            if not any(slot in wanted for slot in slots):
                continue
            series = frame[column]
            if entry["mode"] == "identity":
                mapping = {
                    _decode_portable_scalar(item["key"]): int(item["state"])
                    for item in entry.get("mapping", [])
                }
                mapped = series.astype("object").map(mapping)
                fallback = int(entry.get("unknown_state", entry.get("missing_state", 0)))
                states = mapped.fillna(fallback).to_numpy(dtype=np.int16)
                if representation == "binary":
                    for bit, slot in enumerate(slots):
                        if slot in wanted:
                            generated[slot] = (states >> bit) & 1
                else:
                    generated[slots[0]] = states
            else:
                keys = np.asarray([_category_key(value) for value in series], dtype=object)
                mapping = {str(key): float(value) for key, value in entry["mapping"].items()}
                prior = float(entry["prior"])
                scores = np.asarray(
                    [mapping.get(str(key), prior) for key in keys], dtype=np.float64
                )
                thresholds = self.arrays[entry["threshold_array"]]
                generated[slots[0]] = np.searchsorted(
                    thresholds, scores, side="right"
                )
        if not len(indices):
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.column_stack([generated[int(index)] for index in indices]).astype(
            np.float64, copy=False
        )

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = reset_dataframe_index_if_needed(validate_dataframe_schema(X, self.input_columns))
        arrays: list[np.ndarray] = []
        for entry in self.metadata.get("numeric", []):
            column = self.input_columns[int(entry["column_index"])]
            values = _numeric_series_values(frame[column])
            missing = ~np.isfinite(values)
            median = float(self.arrays[entry["median_array"]][0])
            arrays.append(np.where(missing, median, values)[:, None])
            if entry.get("emit_missing_state", False):
                arrays.append(missing.astype(np.int16)[:, None])

        representation = self.metadata.get("category_identity", "state")
        for entry in self.metadata.get("categorical", []):
            column = self.input_columns[int(entry["column_index"])]
            series = frame[column]
            if entry["mode"] == "identity":
                mapping = {
                    _decode_portable_scalar(item["key"]): int(item["state"])
                    for item in entry.get("mapping", [])
                }
                mapped = series.astype("object").map(mapping)
                fallback = int(entry.get("unknown_state", entry.get("missing_state", 0)))
                states = mapped.fillna(fallback).to_numpy(dtype=np.int16)
                if representation == "binary":
                    bits = max(1, int(np.ceil(np.log2(max(int(entry["cardinality"]), 2)))))
                    arrays.extend(((states >> bit) & 1)[:, None] for bit in range(bits))
                else:
                    arrays.append(states[:, None])
            else:
                keys = np.asarray([_category_key(value) for value in series], dtype=object)
                mapping = {str(key): float(value) for key, value in entry["mapping"].items()}
                prior = float(entry["prior"])
                scores = np.asarray([mapping.get(str(key), prior) for key in keys], dtype=np.float64)
                thresholds = self.arrays[entry["threshold_array"]]
                arrays.append(np.searchsorted(thresholds, scores, side="right")[:, None])
        if not arrays:
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.column_stack(arrays).astype(np.float64, copy=False)


@dataclass
class PortableRegressionProgram:
    """Pure NumPy/Pandas execution of a versioned regression export.

    Unlike joblib artifacts, this loader reconstructs inference only from the
    checked JSON and NPZ files in ``cerm-python-package-v3``.
    """

    manifest: dict[str, Any]
    model_metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    adapter: PortableRegressionAdapter | None

    @classmethod
    def load(cls, directory: str | Path) -> "PortableRegressionProgram":
        from .io import verify_export

        directory = Path(directory)
        manifest_path = directory / "manifest.json" if directory.is_dir() else directory
        manifest = verify_export(manifest_path)
        if manifest.get("format") != "cerm-python-package-v3" or manifest.get("task_type") != "regression":
            raise ValueError("PortableRegressionProgram requires a regression package-v3 export")
        base = manifest_path.parent
        model_json = base / manifest["model"]["json"]["file"]
        model_npz = base / manifest["model"]["npz"]["file"]
        model_metadata = json.loads(model_json.read_text(encoding="utf-8"))
        if model_metadata.get("format") not in {
            "cerm-finite-state-regression-v1",
            "cerm-finite-state-regression-v2",
        }:
            raise ValueError(f"unsupported finite-state regression format: {model_metadata.get('format')!r}")
        with np.load(model_npz, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        adapter = None
        adapter_record = manifest.get("adapter")
        if adapter_record is not None:
            if not adapter_record.get("portable", False):
                raise ValueError("exported regression adapter is not portable")
            adapter = PortableRegressionAdapter.load(
                base / adapter_record["json"]["file"],
                base / adapter_record["npz"]["file"],
            )
        return cls(
            manifest=manifest,
            model_metadata=model_metadata,
            arrays=arrays,
            adapter=adapter,
        )

    def _matrix(self, X) -> np.ndarray:
        input_columns = self.manifest.get("input_columns")
        feature_indices = self.manifest.get("feature_indices")
        if self.adapter is None:
            matrix = dense_numeric_matrix(X, tuple(input_columns) if input_columns is not None else None)
            return _select_features(matrix, feature_indices)
        if feature_indices is not None and hasattr(self.adapter, "transform_columns"):
            return self.adapter.transform_columns(X, feature_indices)
        matrix = self.adapter.transform(X)
        return _select_features(matrix, feature_indices)

    def _prepare_execution_maps(self) -> None:
        levels = tuple(int(level) for level in self.model_metadata["levels"])
        max_main_level = int(self.model_metadata["max_main_level"])
        active_levels = tuple(level for level in levels if level <= max_main_level)
        self._execution_active_levels = active_levels
        if not active_levels:
            self._execution_level = 0
            self._execution_parent_maps = tuple()
            self._execution_pair_level = 0
            self._execution_pair_level_index = 0
            self._execution_pair_plans = tuple()
            self._execution_code_maxima = tuple()
            return
        execution_level = int(max(active_levels))
        selected = self.arrays["feature_indices"].astype(np.int64, copy=False)
        parent_maps = []
        code_maxima = []
        for raw_feature in selected:
            raw_feature = int(raw_feature)
            child_map = self.arrays[f"map_{execution_level}_{raw_feature}"]
            child_card = int(np.max(child_map, initial=0)) + 1
            feature_maps = []
            for level in active_levels:
                if level == execution_level:
                    mapping = np.arange(child_card, dtype=np.int32)
                else:
                    mapping = _parent_labels_from_fine(
                        child_map,
                        self.arrays[f"map_{level}_{raw_feature}"],
                        child_card,
                    ).astype(np.int32, copy=False)
                feature_maps.append(mapping)
                code_maxima.append(int(np.max(mapping, initial=0)))
            parent_maps.append(tuple(feature_maps))
        self._execution_level = execution_level
        self._execution_parent_maps = tuple(parent_maps)
        pair_level = int(
            max(level for level in active_levels if level <= min(max_main_level, 8))
        )
        self._execution_pair_level = pair_level
        self._execution_pair_level_index = active_levels.index(pair_level)
        pair_map_index = int(self._execution_pair_level_index)
        pairs = self.arrays["pairs"].astype(np.int64, copy=False).reshape(-1, 2)
        pair_cards = self.arrays["pair_cardinalities"].astype(np.int64, copy=False)
        pair_plans = []
        for pair_index, (left, right) in enumerate(pairs):
            left = int(left)
            right = int(right)
            left_map = self._execution_parent_maps[left][pair_map_index]
            right_map = self._execution_parent_maps[right][pair_map_index]
            exec_right_cardinality = int(len(right_map))
            pair_code_map = (
                left_map[:, None].astype(np.int64, copy=False)
                * int(pair_cards[pair_index])
                + right_map[None, :]
            ).reshape(-1)
            pair_plans.append((left, right, exec_right_cardinality, pair_code_map))
            code_maxima.append(int(np.max(pair_code_map, initial=0)))
        self._execution_pair_plans = tuple(pair_plans)
        self._execution_code_maxima = tuple(code_maxima)

    def _ensure_execution_maps(self) -> None:
        if not hasattr(self, "_execution_parent_maps"):
            self._prepare_execution_maps()

    def _direct_level_thresholds(self, raw_feature: int, level: int):
        cache = getattr(self, "_direct_level_threshold_cache", None)
        if cache is None:
            cache = {}
            self._direct_level_threshold_cache = cache
        key = (int(raw_feature), int(level))
        if key in cache:
            return cache[key]
        direct_mask = self.arrays["direct_state_mask"].astype(bool, copy=False)
        if direct_mask[int(raw_feature)]:
            cache[key] = None
            return None
        mapping = np.asarray(
            self.arrays[f"map_{int(level)}_{int(raw_feature)}"],
            dtype=np.int64,
        )
        diff = np.diff(mapping)
        cardinality = int(np.max(mapping, initial=0)) + 1
        if (
            len(mapping)
            and int(mapping[0]) == 0
            and np.all((diff >= 0) & (diff <= 1))
            and int(mapping[-1]) + 1 == cardinality
        ):
            boundaries = np.flatnonzero(diff != 0)
            thresholds = np.asarray(
                self.arrays[f"thresholds_{int(raw_feature)}"], dtype=np.float64
            )[boundaries]
            if len(thresholds) + 1 == cardinality:
                cache[key] = thresholds
                return thresholds
        cache[key] = None
        return None

    def _execution_states(
        self, matrix: np.ndarray, selected_indices: np.ndarray
    ) -> np.ndarray:
        self._ensure_execution_maps()
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        direct_mask = self.arrays["direct_state_mask"].astype(bool, copy=False)
        direct_cards = self.arrays["direct_state_cardinalities"].astype(
            np.int64, copy=False
        )
        out = np.empty((len(matrix), len(selected_indices)), dtype=np.int16)
        for position, raw_feature in enumerate(selected_indices):
            raw_feature = int(raw_feature)
            values = matrix[:, raw_feature]
            mapping = self.arrays[
                f"map_{int(self._execution_level)}_{raw_feature}"
            ]
            if direct_mask[raw_feature]:
                state = np.rint(values).astype(np.int64)
                if np.any(~np.isfinite(values)) or np.any(
                    (state < 0) | (state >= direct_cards[raw_feature])
                ):
                    raise ValueError(
                        f"feature {raw_feature} contains an unknown finite state"
                    )
                out[:, position] = mapping[state]
                continue
            direct_thresholds = self._direct_level_thresholds(
                raw_feature, self._execution_level
            )
            if direct_thresholds is not None:
                out[:, position] = np.searchsorted(
                    direct_thresholds, values, side="right"
                )
            else:
                fine = np.searchsorted(
                    self.arrays[f"thresholds_{raw_feature}"],
                    values,
                    side="right",
                )
                out[:, position] = mapping[fine]
        return out

    def _states_columns(
        self, matrix: np.ndarray, columns: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Exact projected finite-state transform for portable regression."""
        columns = np.asarray(columns, dtype=np.int64)
        direct_mask = self.arrays["direct_state_mask"].astype(bool, copy=False)
        direct_cards = self.arrays["direct_state_cardinalities"].astype(
            np.int64, copy=False
        )
        fine = np.empty((len(matrix), len(columns)), dtype=np.int16)
        for out_feature, raw_feature in enumerate(columns):
            raw_feature = int(raw_feature)
            values = matrix[:, raw_feature]
            if direct_mask[raw_feature]:
                state = np.rint(values).astype(np.int64)
                if np.any(~np.isfinite(values)) or np.any(
                    (state < 0) | (state >= direct_cards[raw_feature])
                ):
                    raise ValueError(
                        f"feature {raw_feature} contains an unknown finite state"
                    )
                fine[:, out_feature] = state
            else:
                thresholds = self.arrays[f"thresholds_{raw_feature}"]
                fine[:, out_feature] = np.searchsorted(
                    thresholds, values, side="right"
                )
        states: dict[int, np.ndarray] = {}
        for level in self.model_metadata["levels"]:
            transformed = np.empty_like(fine)
            for out_feature, raw_feature in enumerate(columns):
                mapping = self.arrays[f"map_{int(level)}_{int(raw_feature)}"]
                transformed[:, out_feature] = mapping[fine[:, out_feature]]
            states[int(level)] = transformed
        return states

    def predict(self, X) -> np.ndarray:
        matrix = self._matrix(X)
        selected_indices = self.arrays["feature_indices"].astype(np.int64)
        selected = matrix[:, selected_indices]
        max_main_level = int(self.model_metadata["max_main_level"])
        if max_main_level > 0:
            execution_states = self._execution_states(matrix, selected_indices)
            offsets = self.arrays["lookup_offsets"].astype(np.int64, copy=False)
            values = self.arrays["lookup_values"].astype(np.float64, copy=False)
            prediction = np.full(
                len(matrix), float(self.arrays["intercept"][0]), dtype=np.float64
            )
            lookup_index = 0
            scratch = np.empty(len(matrix), dtype=np.float64)

            def accumulate(state):
                nonlocal lookup_index, prediction
                table = values[
                    offsets[lookup_index] : offsets[lookup_index + 1]
                ]
                full_domain = (
                    lookup_index < len(self._execution_code_maxima)
                    and self._execution_code_maxima[lookup_index] < len(table)
                )
                if full_domain:
                    np.take(table, state, out=scratch)
                    np.add(prediction, scratch, out=prediction)
                else:
                    valid = (state >= 0) & (state < len(table))
                    prediction[valid] += table[state[valid]]
                lookup_index += 1

            for feature, feature_maps in enumerate(self._execution_parent_maps):
                execution_state = execution_states[:, feature]
                for parent_map in feature_maps:
                    accumulate(parent_map[execution_state])
            if self._execution_pair_plans:
                joint_scratch = np.empty(len(matrix), dtype=np.int64)
                code_scratch = np.empty(len(matrix), dtype=np.int64)
                for left, right, exec_right_cardinality, pair_code_map in self._execution_pair_plans:
                    np.multiply(
                        execution_states[:, int(left)],
                        int(exec_right_cardinality),
                        out=joint_scratch,
                        dtype=np.int64,
                    )
                    np.add(
                        joint_scratch, execution_states[:, int(right)], out=joint_scratch
                    )
                    np.take(pair_code_map, joint_scratch, out=code_scratch)
                    accumulate(code_scratch)
            if lookup_index != len(offsets) - 1:
                raise RuntimeError(
                    f"portable regression lookup layout mismatch: consumed "
                    f"{lookup_index}, expected {len(offsets) - 1}"
                )
        else:
            prediction = np.full(
                len(matrix), float(self.arrays["intercept"][0]), dtype=np.float64
            )
        linear_positions = self.arrays["linear_positions"].astype(
            np.int64, copy=False
        )
        if len(linear_positions):
            linear = (
                selected[:, linear_positions]
                - self.arrays["linear_mean"]
            ) / self.arrays["linear_scale"]
            prediction += float(self.arrays["linear_intercept"][0])
            prediction += linear @ self.arrays["linear_coef"]
        return prediction

    decision_function = predict


class CERMRegressor(RegressorMixin, BaseEstimator):
    """Finite-state regressor with main and selected pair interactions."""

    VERSION = __version__

    def __init__(
        self,
        *,
        max_features: int = 64,
        max_bins: int = 16,
        subsample: float = 1.0,
        colsample: float = 1.0,
        n_jobs: int | None = 1,
        search_effort: str | None = None,
        state_detail: str | None = None,
        feature_budget: int | None = None,
        interaction_search_features: int | None = None,
        interaction_budget: int | None = None,
        selection_fraction: float | None = None,
        feature_fraction: float | None = None,
        l2_regularization: str | float | None = None,
        preset: str = "accurate",
        max_interaction_features: int = 24,
        max_interactions: int | None = None,
        interaction_order: int = 2,
        reg_lambda: str | float = "auto",
        random_state: int = 20260803,
        categorical_features: Sequence[str] | str | None = "auto",
        embedding_features: Sequence[str] | None = None,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        missing_policy: str = "observed",
        include_linear: bool = True,
    ):
        self.search_effort = search_effort
        self.state_detail = state_detail
        self.feature_budget = feature_budget
        self.interaction_search_features = interaction_search_features
        self.interaction_budget = interaction_budget
        self.selection_fraction = selection_fraction
        self.feature_fraction = feature_fraction
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.n_jobs = n_jobs
        self.preset = preset
        self.max_interaction_features = max_interaction_features
        self.max_interactions = max_interactions
        self.interaction_order = interaction_order
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.embedding_features = embedding_features
        self.category_policy = category_policy
        self.max_identity_categories = max_identity_categories
        self.category_bins = category_bins
        self.category_smoothing = category_smoothing
        self.category_identity = category_identity
        self.missing_policy = missing_policy
        self.include_linear = include_linear
        self._semantic_legacy_inputs = {
            "preset": preset,
            "max_bins": max_bins,
            "max_features": max_features,
            "max_interaction_features": max_interaction_features,
            "max_interactions": max_interactions,
            "subsample": subsample,
            "colsample": colsample,
            "reg_lambda": reg_lambda,
        }
        self._refresh_semantic_aliases()

    def _refresh_semantic_aliases(self) -> None:
        conflicts: list[str] = []
        specs = (
            ("search_effort", "preset", "accurate", {"thorough": "accurate", "balanced": "balanced"}),
            ("state_detail", "max_bins", 16, {"coarse": 4, "medium": 8, "fine": 16}),
            ("feature_budget", "max_features", 64, None),
            ("interaction_search_features", "max_interaction_features", 24, None),
            ("interaction_budget", "max_interactions", None, None),
            ("selection_fraction", "subsample", 1.0, None),
            ("feature_fraction", "colsample", 1.0, None),
            ("l2_regularization", "reg_lambda", "auto", None),
        )
        for semantic_name, legacy_name, legacy_default, mapping in specs:
            legacy_value = self._semantic_legacy_inputs[legacy_name]
            try:
                resolved = resolve_semantic_alias(
                    semantic_name=semantic_name,
                    semantic_value=getattr(self, semantic_name),
                    legacy_name=legacy_name,
                    legacy_value=legacy_value,
                    legacy_default=legacy_default,
                    mapping=mapping,
                )
            except ValueError as exc:
                conflicts.append(str(exc))
                resolved = resolve_semantic_alias(
                    semantic_name=semantic_name,
                    semantic_value=getattr(self, semantic_name),
                    legacy_name=legacy_name,
                    legacy_value=legacy_value,
                    legacy_default=legacy_default,
                    mapping=mapping,
                    strict=False,
                )
            setattr(self, legacy_name, resolved)
        self._semantic_conflicts = tuple(conflicts)

    def set_params(self, **params):
        legacy_names = set(self._semantic_legacy_inputs)
        result = super().set_params(**params)
        for name in legacy_names & set(params):
            self._semantic_legacy_inputs[name] = params[name]
        return result

    def get_user_params(self) -> dict[str, object]:
        """Return the compact, recommended parameter view."""
        return semantic_parameter_values(self, task="regressor")

    def explain_params(self) -> dict[str, dict[str, object]]:
        """Return effective values together with plain-language meanings."""
        return explain_semantic_parameters(self, task="regressor")

    def parameter_summary(self) -> str:
        """Return a human-readable summary of the effective model choices."""
        return format_semantic_parameter_summary(self, task="regressor")

    @staticmethod
    def _infer_categorical(frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column]))

    def _validate_params(self):
        self._refresh_semantic_aliases()
        if self._semantic_conflicts:
            raise ValueError("; ".join(self._semantic_conflicts))
        if self.max_bins not in {4, 8, 16}:
            raise ValueError("max_bins must be one of 4, 8, or 16")
        if not isinstance(self.max_features, (int, np.integer)) or self.max_features < 1:
            raise ValueError("max_features must be a positive integer")
        if not 0 < float(self.subsample) <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if not 0 < float(self.colsample) <= 1:
            raise ValueError("colsample must be in (0, 1]")
        if self.preset not in {"accurate", "balanced"}:
            raise ValueError("preset must be 'accurate' or 'balanced'")
        if self.interaction_order not in {1, 2}:
            raise ValueError("interaction_order must be 1 or 2")
        if self.max_interactions is not None and int(self.max_interactions) < 0:
            raise ValueError("max_interactions must be None or non-negative")
        if self.max_interaction_features < 1:
            raise ValueError("max_interaction_features must be positive")
        if self.category_policy not in {"auto", "identity", "ordered", "newton"}:
            raise ValueError("invalid category_policy")

    def fit(
        self,
        X,
        y: Iterable,
        sample_weight: Iterable[float] | None = None,
    ):
        start = time.perf_counter()
        self._validate_params()
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError("sparse input is not supported; provide a dense ndarray or pandas DataFrame")
        n_samples = num_samples(X)
        y_array = column_or_1d(y, warn=True).astype(np.float64)
        if len(y_array) != n_samples:
            raise ValueError(f"X and y have inconsistent lengths: {n_samples} and {len(y_array)}")
        if not np.isfinite(y_array).all():
            raise ValueError("y contains NaN or infinity")
        if sample_weight is None:
            weight_array = None
        else:
            weight_array = column_or_1d(sample_weight, warn=True).astype(np.float64)
            if len(weight_array) != n_samples:
                raise ValueError(
                    "X and sample_weight have inconsistent lengths: "
                    f"{n_samples} and {len(weight_array)}"
                )
            if not np.isfinite(weight_array).all():
                raise ValueError("sample_weight contains NaN or infinity")
            if np.any(weight_array < 0):
                raise ValueError("sample_weight cannot contain negative values")
            if not np.any(weight_array > 0):
                raise ValueError(
                    "sample_weight cannot be all zero; it must contain at least one positive value"
                )
        target_type = type_of_target(y_array, input_name="y", raise_unknown=True)
        if target_type not in {"continuous", "binary", "multiclass"}:
            raise ValueError(f"CERMRegressor requires a one-dimensional numeric target, got {target_type}")
        if n_samples < 8:
            raise ValueError(f"CERMRegressor requires at least 8 samples; got n_samples={n_samples}")

        feature_kinds = None
        feature_cardinalities = None
        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates:
                raise ValueError("DataFrame columns must be unique")
            frame = X
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            self.n_features_in_ = int(frame.shape[1])
            self.input_columns_ = tuple(frame.columns)
            categorical = self._infer_categorical(frame) if self.categorical_features == "auto" else tuple(self.categorical_features or ())
            self.categorical_features_ = categorical
            self.embedding_features_ = tuple(self.embedding_features or ())
            if categorical or self.embedding_features_ or frame.isna().any().any():
                self.adapter_ = RegressionTypedAdapter(
                    categorical_columns=categorical,
                    embedding_columns=self.embedding_features_,
                    category_policy=self.category_policy,
                    max_identity_categories=int(self.max_identity_categories),
                    category_bins=int(self.category_bins),
                    category_smoothing=float(self.category_smoothing),
                    category_identity=self.category_identity,
                    missing_policy=self.missing_policy,
                    random_state=int(self.random_state),
                )
                adapted = self.adapter_.fit_transform(
                    frame, y_array, sample_weight=weight_array
                )
                matrix = adapted.matrix
                names = tuple(adapted.feature_names)
                feature_kinds = tuple(adapted.feature_kinds)
                feature_cardinalities = tuple(adapted.cardinalities)
            else:
                self.adapter_ = None
                matrix = dense_numeric_matrix(frame)
                names = tuple(map(str, frame.columns))
        else:
            matrix = dense_numeric_matrix(X)
            self.feature_names_in_ = None
            self.n_features_in_ = int(matrix.shape[1])
            self.input_columns_ = None
            self.adapter_ = None
            self.categorical_features_ = ()
            self.embedding_features_ = ()
            names = tuple(f"x{i}" for i in range(matrix.shape[1]))

        self.n_full_adapted_features_ = int(matrix.shape[1])
        keep = max(1, min(self.n_full_adapted_features_, int(np.ceil(float(self.colsample) * self.n_full_adapted_features_))))
        if keep < self.n_full_adapted_features_:
            rng = np.random.default_rng(int(self.random_state) + 32452843)
            indices = np.sort(rng.choice(self.n_full_adapted_features_, size=keep, replace=False)).astype(np.int64)
            matrix = np.ascontiguousarray(matrix[:, indices])
            names = tuple(names[index] for index in indices)
            if feature_kinds is not None:
                feature_kinds = tuple(feature_kinds[index] for index in indices)
                feature_cardinalities = tuple(feature_cardinalities[index] for index in indices)
        else:
            indices = np.arange(self.n_full_adapted_features_, dtype=np.int64)
        self.adapted_feature_indices_ = indices
        self.n_adapted_features_ = int(matrix.shape[1])
        self.adapted_feature_names_ = np.asarray(names, dtype=object)

        max_interactions = (
            24 if self.max_interactions is None and self.preset == "accurate"
            else 12 if self.max_interactions is None
            else int(self.max_interactions)
        )
        self.model_ = FiniteStateRidgeRegressor(
            max_features=int(self.max_features),
            max_bins=int(self.max_bins),
            max_interaction_features=int(self.max_interaction_features),
            max_interactions=max_interactions,
            interaction_order=int(self.interaction_order),
            reg_lambda=self.reg_lambda,
            preset=self.preset,
            subsample=float(self.subsample),
            n_jobs=self.n_jobs,
            random_state=int(self.random_state),
            feature_kinds=feature_kinds,
            feature_cardinalities=feature_cardinalities,
            include_linear=bool(self.include_linear),
        )
        forced_config = getattr(self, "_forced_representation_config", None)
        if forced_config is not None:
            self.model_._forced_representation_config = forced_config
        self.model_.fit(
            matrix,
            y_array,
            sample_weight=weight_array,
            _fit_final_head=not bool(getattr(self, "_representation_only_fit", False)),
            _cache_training_design=(
                bool(getattr(self, "_representation_only_fit", False))
                and self.adapter_ is None
            ),
        )
        if bool(getattr(self, "_representation_only_fit", False)):
            if self.adapter_ is None:
                self._fit_matrix_cache_ = matrix
                self._fit_training_design_is_final_ = True
            else:
                final_adapted = self.adapter_.transform(frame).matrix
                self._fit_matrix_cache_ = np.ascontiguousarray(
                    final_adapted[:, self.adapted_feature_indices_]
                )
                self._fit_training_design_is_final_ = False
        self.program_ = RegressionSemanticProgram(
            model=self.model_,
            adapter=self.adapter_,
            input_columns=self.input_columns_,
            adapted_feature_names=names,
            feature_indices=tuple(int(index) for index in self.adapted_feature_indices_),
            library_version=self.VERSION,
            metadata={"task_type": "regression", "engine": "finite-state-ridge"},
        )
        self._prediction_program_ = self.program_
        self.fit_seconds_ = float(time.perf_counter() - start)
        self.fit_diagnostics_ = {
            "task_type": "regression",
            "engine": "finite-state-ridge",
            "selected_features": int(len(self.model_.feature_idx_)),
            "pair_count": int(len(self.model_.pairs_)),
            "design_dimension": int(self.model_.design_dim_),
            "model_bytes_estimate": int(self.model_bytes_estimate_),
            "selected_config": {
                "max_main_level": self.model_.config_.max_main_level,
                "n_pairs": self.model_.config_.n_pairs,
                "alpha": self.model_.config_.alpha,
            },
            "fit_seconds": self.fit_seconds_,
            "sample_weighted": weight_array is not None,
            "sample_weight_sum": (
                None if weight_array is None else float(weight_array.sum())
            ),
        }
        return self

    def _check_X_schema(self, X):
        if self.input_columns_ is None:
            shape = getattr(X, "shape", np.asarray(X).shape)
            if len(shape) != 2:
                raise ValueError("Expected 2D array. Reshape your data")
            if int(shape[1]) != self.n_features_in_:
                raise ValueError(
                    f"X has {shape[1]} features, but CERMRegressor is expecting "
                    f"{self.n_features_in_} features as input"
                )
        else:
            validate_dataframe_schema(X, self.input_columns_)

    def predict(self, X):
        check_is_fitted(self, "program_")
        self._check_X_schema(X)
        return self._prediction_program_.predict(X)


    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "program_")
        return self.adapted_feature_names_.copy()

    @property
    def model_bytes_estimate_(self):
        check_is_fitted(self, "program_")
        return self.program_.model_bytes_estimate

    def optimize(self, target: str = "balanced"):
        check_is_fitted(self, "program_")
        self._prediction_program_ = self.program_.optimize(target)
        return self._prediction_program_

    def compile_native(self, prefix: str | Path) -> CompiledRegressionProgram:
        check_is_fitted(self, "program_")
        return self.program_.compile_native(prefix)

    def export(self, directory: str | Path) -> Path:
        check_is_fitted(self, "program_")
        return self.program_.export(directory, config=self.get_params(deep=False))

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "program_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CERMRegressor":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError("serialized object is not a CERMRegressor")
        return estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.one_d_labels = True
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags
