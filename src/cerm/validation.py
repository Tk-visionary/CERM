from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, column_or_1d

from .errors import CERMDataSchemaError, CERMParameterError, format_schema_mismatch


_VALID_REPLACEMENT_OBJECTIVES = {"memory", "balanced", "latency"}
_VALID_CATEGORY_POLICIES = {"auto", "identity", "ordered", "newton"}
_VALID_CATEGORY_IDENTITIES = {"state", "binary"}
_VALID_EMBEDDING_MODES = {"prototype", "knn", "basic"}
_VALID_ENCODERS = {"quantile", "newton"}
_VALID_RANKINGS = {
    "mi",
    "newton",
    "residual_newton",
    "newton_prefilter_mi",
    "diversified",
}
_VALID_MISSING_POLICIES = {"observed", "always", "none"}
_VALID_SELECTION_STRATEGIES = {"two_holdout", "cross_fitted"}
_VALID_SEARCH_PROFILES = {"full_exact", "practical", "aggressive"}
_VALID_PRESETS = {"accurate", "balanced"}
_VALID_PREDICTION_BACKENDS = {"optimized", "semantic"}
_VALID_CALIBRATION = {"none", "intercept", "affine"}
_VALID_RESOURCE_POLICIES = {"ignore", "warn", "raise"}


def validate_estimator_parameters(estimator: Any) -> None:
    """Validate public estimator parameters before any fitted state is created."""

    integer_positive = {
        "max_features": estimator.max_features,
        "max_bins": estimator.max_bins,
        "pair_feature_limit": estimator.pair_feature_limit,
        "max_identity_categories": estimator.max_identity_categories,
        "category_bins": estimator.category_bins,
        "embedding_pca": estimator.embedding_pca,
        "embedding_bins": estimator.embedding_bins,
        "embedding_prototypes": estimator.embedding_prototypes,
        "newton_prebins": estimator.newton_prebins,
        "ranking_prefilter_multiplier": estimator.ranking_prefilter_multiplier,
        "selection_folds": estimator.selection_folds,
        "calibration_folds": estimator.calibration_folds,
    }
    for name, value in integer_positive.items():
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise CERMParameterError(f"{name} must be a positive integer")

    if int(estimator.max_bins) not in {4, 8, 16}:
        raise CERMParameterError("max_bins must be one of: 4, 8, 16")
    for name in ("subsample", "colsample"):
        value = getattr(estimator, name)
        if not np.isfinite(value) or not (0.0 < float(value) <= 1.0):
            raise CERMParameterError(f"{name} must be in (0, 1]")
    if estimator.n_jobs is not None:
        if not isinstance(estimator.n_jobs, (int, np.integer)) or int(estimator.n_jobs) == 0:
            raise CERMParameterError("n_jobs must be None or a non-zero integer")

    if estimator.preset not in _VALID_PRESETS:
        options = ", ".join(sorted(_VALID_PRESETS))
        raise CERMParameterError(f"preset must be one of: {options}")
    if estimator.max_interaction_features is not None:
        value = estimator.max_interaction_features
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise CERMParameterError(
                "max_interaction_features must be a positive integer or None"
            )
    if estimator.max_interactions is not None:
        value = estimator.max_interactions
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise CERMParameterError(
                "max_interactions must be a non-negative integer or None"
            )
    if (
        not isinstance(estimator.class_specific_budget, (int, np.integer))
        or int(estimator.class_specific_budget) < 0
    ):
        raise CERMParameterError("class_specific_budget must be a non-negative integer")
    if (
        not isinstance(estimator.interaction_order, (int, np.integer))
        or int(estimator.interaction_order) not in {1, 2}
    ):
        raise CERMParameterError("interaction_order must be 1 or 2")
    if estimator.reg_lambda != "auto":
        try:
            reg_lambda = float(estimator.reg_lambda)
        except (TypeError, ValueError) as exc:
            raise CERMParameterError(
                "reg_lambda must be 'auto' or a finite positive number"
            ) from exc
        if not np.isfinite(reg_lambda) or reg_lambda <= 0:
            raise CERMParameterError(
                "reg_lambda must be 'auto' or a finite positive number"
            )

    if estimator.selection_folds < 2:
        raise CERMParameterError("selection_folds must be at least 2")
    if estimator.category_bins < 2 or estimator.embedding_bins < 2:
        raise CERMParameterError("category_bins and embedding_bins must be at least 2")
    if estimator.newton_prebins < 4:
        raise CERMParameterError("newton_prebins must be at least 4")

    nonnegative = {
        "category_smoothing": estimator.category_smoothing,
        "category_newton_l2": estimator.category_newton_l2,
        "newton_gain_l2": estimator.newton_gain_l2,
        "newton_min_hessian": estimator.newton_min_hessian,
        "ranking_l2": estimator.ranking_l2,
        "cost_per_byte": estimator.cost_per_byte,
        "cost_per_operator": estimator.cost_per_operator,
        "block_cost_per_byte": estimator.block_cost_per_byte,
        "block_cost_per_eval": estimator.block_cost_per_eval,
        "selection_near_tie": estimator.selection_near_tie,
        "selection_min_improvement": estimator.selection_min_improvement,
        "calibration_l2": estimator.calibration_l2,
        "calibration_min_improvement": estimator.calibration_min_improvement,
        "calibration_min_signal": estimator.calibration_min_signal,
    }
    for name, value in nonnegative.items():
        if not np.isfinite(value) or float(value) < 0:
            raise CERMParameterError(f"{name} must be finite and non-negative")

    choices = {
        "replacement_objective": (
            estimator.replacement_objective,
            _VALID_REPLACEMENT_OBJECTIVES,
        ),
        "prediction_backend": (estimator.prediction_backend, _VALID_PREDICTION_BACKENDS),
        "category_policy": (estimator.category_policy, _VALID_CATEGORY_POLICIES),
        "category_identity": (estimator.category_identity, _VALID_CATEGORY_IDENTITIES),
        "embedding_mode": (estimator.embedding_mode, _VALID_EMBEDDING_MODES),
        "encoder_kind": (estimator.encoder_kind, _VALID_ENCODERS),
        "ranking_kind": (estimator.ranking_kind, _VALID_RANKINGS),
        "missing_policy": (estimator.missing_policy, _VALID_MISSING_POLICIES),
        "selection_strategy": (
            estimator.selection_strategy,
            _VALID_SELECTION_STRATEGIES,
        ),
        "multiclass_strategy": (
            estimator.multiclass_strategy,
            {"ovr", "shared", "error"},
        ),
        "shared_multiclass_objective": (
            estimator.shared_multiclass_objective,
            {"auto", "ovr", "multinomial"},
        ),
        "representation_strategy": (
            estimator.representation_strategy,
            {"baseline", "adaptive"},
        ),
        "search_profile": (estimator.search_profile, _VALID_SEARCH_PROFILES),
        "calibration": (estimator.calibration, _VALID_CALIBRATION),
        "resource_policy": (estimator.resource_policy, _VALID_RESOURCE_POLICIES),
    }
    for name, (value, allowed) in choices.items():
        if name == "search_profile" and value is None:
            continue
        if value not in allowed:
            options = ", ".join(sorted(allowed))
            raise CERMParameterError(f"{name} must be one of: {options}")

    if not isinstance(estimator.cache_training_statistics, (bool, np.bool_)):
        raise TypeError("cache_training_statistics must be boolean")

    optional_positive = {
        "max_memory_mb": estimator.max_memory_mb,
        "max_estimated_peak_memory_mb": estimator.max_estimated_peak_memory_mb,
        "max_pair_evaluations": estimator.max_pair_evaluations,
        "max_block_evaluations": estimator.max_block_evaluations,
        "max_knn_distance_evaluations": estimator.max_knn_distance_evaluations,
    }
    for name, value in optional_positive.items():
        if value is None:
            continue
        if not np.isfinite(value) or float(value) <= 0:
            raise CERMParameterError(f"{name} must be None or a finite positive number")
        if name not in {"max_memory_mb", "max_estimated_peak_memory_mb"} and not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be None or an integer")

    categorical = estimator.categorical_features
    if categorical not in (None, "auto") and isinstance(categorical, str):
        raise TypeError(
            "categorical_features must be 'auto', None, or a sequence of column names"
        )
    if isinstance(estimator.embedding_features, str):
        raise TypeError("embedding_features must be None or a sequence of column names")

    if categorical not in (None, "auto"):
        categorical_set = set(categorical)
        embedding_set = set(estimator.embedding_features or ())
        overlap = sorted(categorical_set & embedding_set)
        if overlap:
            raise CERMParameterError(
                f"features cannot be both categorical and embedding: {overlap}"
            )


def validate_binary_target(
    y: Iterable, *, n_samples: int, n_features: int | None = None
) -> np.ndarray:
    if y is None:
        raise ValueError("CERMClassifier requires y to be passed, but the target y is None")
    y_array = column_or_1d(y, warn=True)
    if len(y_array) != n_samples:
        raise ValueError(
            f"X and y have inconsistent lengths: {n_samples} and {len(y_array)}"
        )
    if pd.isna(y_array).any():
        raise ValueError("y contains missing values")

    target_type = type_of_target(y_array, input_name="y", raise_unknown=True)
    if target_type == "continuous":
        raise ValueError("continuous target is not supported by CERMClassifier")
    if target_type != "binary":
        raise ValueError(
            "Only binary classification is supported. "
            f"The type of the target is {target_type}."
        )

    classes, counts = np.unique(y_array, return_counts=True)
    if len(classes) != 2:
        if n_samples == 1:
            raise ValueError("CERMClassifier cannot fit 1 sample or one class")
        raise ValueError("Only binary classification is supported. The target has one class.")
    if counts.min() < 4:
        feature_note = " For 1 feature(s)," if n_features == 1 else ""
        raise ValueError(
            f"CERMClassifier requires at least 4 samples in each class for its "
            f"stratified selection splits.{feature_note}"
        )
    return np.asarray(y_array)


def dense_numeric_matrix(
    X: pd.DataFrame | np.ndarray,
    input_columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    if sparse.issparse(X):
        raise TypeError(
            "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
        )
    if isinstance(X, pd.DataFrame):
        frame = X
        if frame.columns.has_duplicates:
            raise CERMDataSchemaError("DataFrame columns must be unique")
        if input_columns is not None:
            missing = [column for column in input_columns if column not in frame.columns]
            extra = [column for column in frame.columns if column not in input_columns]
            if missing or extra:
                raise CERMDataSchemaError(format_schema_mismatch(missing, extra))
            frame = frame.loc[:, input_columns]
        try:
            if all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
                matrix = frame.to_numpy(dtype=np.float64, copy=False)
            else:
                matrix = frame.apply(pd.to_numeric, errors="raise").to_numpy(
                    dtype=np.float64
                )
        except (TypeError, ValueError) as exc:
            raise CERMDataSchemaError(
                "numeric CERM input contains non-numeric values; configure "
                "categorical_features or use categorical_features='auto'"
            ) from exc
        if not np.isfinite(matrix).all():
            raise CERMDataSchemaError("numeric CERM input contains NaN or infinity")
    else:
        matrix = check_array(
            X,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            allow_nd=False,
            ensure_all_finite=True,
            copy=False,
        )
    if matrix.shape[1] < 1:
        raise ValueError("X must contain at least one feature")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def validate_dataframe_schema(
    X: pd.DataFrame,
    input_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("a fitted DataFrame model requires pandas DataFrame input")
    if X.columns.has_duplicates:
        raise CERMDataSchemaError("DataFrame columns must be unique")
    if tuple(X.columns) == tuple(input_columns):
        return X
    missing = [column for column in input_columns if column not in X.columns]
    extra = [column for column in X.columns if column not in input_columns]
    if missing or extra:
        raise CERMDataSchemaError(format_schema_mismatch(missing, extra))
    return X.loc[:, input_columns]


def reset_dataframe_index_if_needed(X: pd.DataFrame) -> pd.DataFrame:
    """Match ``reset_index(drop=True)`` without copying a default RangeIndex."""
    index = X.index
    if (
        isinstance(index, pd.RangeIndex)
        and index.start == 0
        and index.stop == len(X)
        and index.step == 1
    ):
        return X
    return X.reset_index(drop=True)
