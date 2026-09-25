from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .errors import (
    CERMResourceLimitError,
    CERMResourceWarning,
    CERMParameterError,
    format_resource_limit,
)
from .params import resolve_estimator_parameters


_VALID_RESOURCE_POLICIES = {"ignore", "warn", "raise"}


@dataclass(frozen=True)
class FitResourcePlan:
    raw_rows: int
    raw_features: int
    estimated_full_adapted_features: int
    estimated_adapted_features: int
    effective_selection_rows: int
    max_bins: int
    subsample: float
    colsample: float
    n_jobs: int | None
    effective_pair_features: int
    pair_candidates_per_ranking: int
    directed_block_pairs_per_ranking: int
    estimated_pair_evaluations: int
    estimated_block_evaluations: int
    effective_internal_fit_units: int
    estimated_solver_calls: int
    estimated_input_bytes: int
    estimated_adapted_matrix_bytes: int
    estimated_state_bank_bytes: int
    estimated_design_bytes: int
    estimated_peak_memory_bytes: int
    estimated_knn_distance_evaluations: int
    risk: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataframe_bytes(frame: pd.DataFrame) -> int:
    try:
        return int(frame.memory_usage(index=True, deep=True).sum())
    except Exception:
        return int(frame.shape[0] * max(frame.shape[1], 1) * 8)


def _ndarray_bytes(X) -> int:
    array = np.asarray(X)
    return int(getattr(array, "nbytes", array.size * max(array.dtype.itemsize, 8)))


def _embedding_width(estimator, n_rows: int, n_embedding_features: int) -> int:
    if n_embedding_features <= 0:
        return 0
    pca = min(int(estimator.embedding_pca), n_embedding_features, max(1, n_rows - 1))
    if estimator.embedding_mode == "prototype":
        quotient = pca + 5
    elif estimator.embedding_mode == "knn":
        quotient = pca + 3
    else:
        quotient = pca + 2
    if estimator.retain_embedding_raw:
        quotient += n_embedding_features
    return int(quotient)


def _estimate_dataframe_width(estimator, frame: pd.DataFrame) -> tuple[int, int]:
    categorical = (
        tuple(
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        )
        if estimator.categorical_features == "auto"
        else tuple(estimator.categorical_features or ())
    )
    embedding = tuple(estimator.embedding_features or ())
    categorical_set = set(categorical)
    embedding_set = set(embedding)
    numeric = [
        column
        for column in frame.columns
        if column not in categorical_set and column not in embedding_set
    ]

    width = 0
    for column in numeric:
        width += 1
        if estimator.missing_policy == "always":
            width += 1
        elif estimator.missing_policy == "observed" and bool(frame[column].isna().any()):
            width += 1

    for column in categorical:
        if estimator.category_identity == "binary":
            unique = int(frame[column].nunique(dropna=True))
            reserve = 1 if estimator.missing_policy == "always" else 0
            card = max(2, unique + reserve)
            identity = (
                estimator.category_policy == "identity"
                or (
                    estimator.category_policy == "auto"
                    and unique <= int(estimator.max_identity_categories)
                )
            )
            width += max(1, int(math.ceil(math.log2(card)))) if identity else 1
        else:
            width += 1

    width += _embedding_width(estimator, len(frame), len(embedding))
    return max(int(width), 1), len(embedding)


def _candidate_count(
    search_profile: str, max_interactions: int, fixed_C, n_rows: int
) -> int:
    natural_limit = max(8, min(40, max(1, int(n_rows * 0.78)) // 12))
    max_blocks = min(int(max_interactions), natural_limit)
    if max_blocks <= 0:
        return 1
    base = (
        (8, 16, 32)
        if search_profile in {"practical", "aggressive"}
        else (4, 8, 16, 24, 32)
    )
    prefixes = list(dict.fromkeys(min(int(value), max_blocks) for value in base))
    prefixes = [value for value in prefixes if value > 0]
    per_mode = len(prefixes)
    if search_profile not in {"practical", "aggressive"}:
        per_mode += 1
    if fixed_C is None:
        per_mode += 1
    return 1 + 2 * per_mode


def estimate_fit_resources(estimator, X) -> FitResourcePlan:
    """Conservatively estimate fit work before target-dependent training begins."""

    shape = getattr(X, "shape", None)
    if shape is None or len(shape) < 2:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("Expected 2D array, got 1D array instead. Reshape your data")
        n_rows, n_raw = map(int, array.shape)
    else:
        n_rows, n_raw = int(shape[0]), int(shape[1])

    if isinstance(X, pd.DataFrame):
        adapted, embedding_features = _estimate_dataframe_width(estimator, X)
        input_bytes = _dataframe_bytes(X)
    else:
        adapted = n_raw
        embedding_features = 0
        input_bytes = _ndarray_bytes(X)

    resolved = resolve_estimator_parameters(estimator)
    full_adapted = int(adapted)
    adapted = max(
        1,
        min(full_adapted, int(math.ceil(full_adapted * resolved.colsample))),
    )
    selection_rows = max(16, int(math.ceil(n_rows * resolved.subsample)))
    selection_rows = min(n_rows, selection_rows)
    selected = max(1, min(int(resolved.max_features), adapted))
    pair_features = max(1, min(int(resolved.max_interaction_features), selected))
    pair_candidates = pair_features * (pair_features - 1) // 2
    directed_blocks = (
        pair_features * max(pair_features - 1, 0)
        if resolved.max_interactions > 0
        else 0
    )

    selection_units = (
        int(estimator.selection_folds) + 1
        if estimator.selection_strategy == "cross_fitted"
        else 3
    )
    calibration_units = (
        int(estimator.calibration_folds) if estimator.calibration != "none" else 0
    )
    internal_fit_units = selection_units + calibration_units

    pair_evaluations = pair_candidates * internal_fit_units
    block_evaluations = directed_blocks * 3 * internal_fit_units
    candidates_per_selection = _candidate_count(
        resolved.search_profile,
        resolved.max_interactions,
        resolved.fixed_C,
        n_rows,
    )
    solver_calls = candidates_per_selection * selection_units + calibration_units

    full_adapted_bytes = int(n_rows * full_adapted * 8)
    adapted_bytes = int(n_rows * adapted * 8)
    level_count = 1 + int(resolved.max_bins >= 8) + int(resolved.max_bins >= 16)
    state_bytes = int(
        n_rows * selected * level_count * (1 if resolved.max_bins <= 255 else 2)
    )
    active_multiplier = 1 + int(resolved.max_bins >= 8) + int(resolved.max_bins >= 16)
    active_per_row = max(1, min(active_multiplier * 3 * selected, 4096))
    design_bytes = int(n_rows * active_per_row * 12 + (n_rows + 1) * 4)
    peak = int(
        input_bytes
        + full_adapted_bytes
        + adapted_bytes
        + state_bytes
        + 2 * design_bytes
    )

    knn_evaluations = 0
    if embedding_features and estimator.embedding_mode == "knn":
        folds = min(5, max(2, n_rows))
        knn_evaluations = int(
            2 * n_rows * n_rows * (folds - 1) / folds + 2 * n_rows * n_rows
        )

    reasons: list[str] = []
    if pair_candidates >= 10_000:
        reasons.append("quadratic pair search is large")
    if block_evaluations >= 1_000_000:
        reasons.append("directed block histogram work is large")
    if peak >= 2 * 1024**3:
        reasons.append("estimated peak memory exceeds 2 GiB")
    if internal_fit_units >= 8:
        reasons.append("selection and calibration multiply internal fits")
    if knn_evaluations >= 100_000_000:
        reasons.append("exact kNN embedding has near-quadratic distance work")

    if any(
        value
        for value in (
            peak >= 8 * 1024**3,
            pair_evaluations >= 10_000_000,
            block_evaluations >= 20_000_000,
            knn_evaluations >= 1_000_000_000,
        )
    ):
        risk = "critical"
    elif reasons:
        risk = "high"
    elif pair_evaluations >= 100_000 or peak >= 512 * 1024**2:
        risk = "moderate"
    else:
        risk = "low"

    return FitResourcePlan(
        raw_rows=n_rows,
        raw_features=n_raw,
        estimated_full_adapted_features=full_adapted,
        estimated_adapted_features=adapted,
        effective_selection_rows=selection_rows,
        max_bins=int(resolved.max_bins),
        subsample=float(resolved.subsample),
        colsample=float(resolved.colsample),
        n_jobs=resolved.n_jobs,
        effective_pair_features=pair_features,
        pair_candidates_per_ranking=pair_candidates,
        directed_block_pairs_per_ranking=directed_blocks,
        estimated_pair_evaluations=int(pair_evaluations),
        estimated_block_evaluations=int(block_evaluations),
        effective_internal_fit_units=int(internal_fit_units),
        estimated_solver_calls=int(solver_calls),
        estimated_input_bytes=int(input_bytes),
        estimated_adapted_matrix_bytes=adapted_bytes,
        estimated_state_bank_bytes=state_bytes,
        estimated_design_bytes=design_bytes,
        estimated_peak_memory_bytes=peak,
        estimated_knn_distance_evaluations=knn_evaluations,
        risk=risk,
        reasons=tuple(reasons),
    )


def enforce_resource_plan(estimator, plan: FitResourcePlan) -> None:
    policy = str(estimator.resource_policy)
    if policy not in _VALID_RESOURCE_POLICIES:
        raise CERMParameterError("resource_policy must be one of: ignore, raise, warn")
    if policy == "ignore":
        return

    violations: list[str] = []
    limits = (
        (
            "estimated pair evaluations",
            plan.estimated_pair_evaluations,
            estimator.max_pair_evaluations,
        ),
        (
            "estimated block evaluations",
            plan.estimated_block_evaluations,
            estimator.max_block_evaluations,
        ),
        (
            "estimated kNN distance evaluations",
            plan.estimated_knn_distance_evaluations,
            estimator.max_knn_distance_evaluations,
        ),
    )
    for name, value, limit in limits:
        if limit is not None and int(value) > int(limit):
            violations.append(f"{name} {value:,} exceeds limit {int(limit):,}")
    memory_limit = resolve_estimator_parameters(estimator).max_memory_mb
    if memory_limit is not None:
        bytes_limit = float(memory_limit) * 1024**2
        if plan.estimated_peak_memory_bytes > bytes_limit:
            violations.append(
                "estimated peak memory "
                f"{plan.estimated_peak_memory_bytes / 1024**2:.1f} MiB exceeds "
                f"limit {float(memory_limit):.1f} MiB"
            )

    if not violations:
        return
    message = format_resource_limit(violations)
    if policy == "raise":
        raise CERMResourceLimitError(message)
    warnings.warn(message, CERMResourceWarning, stacklevel=3)
