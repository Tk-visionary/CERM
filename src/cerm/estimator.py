from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d
from sklearn.utils.validation import check_is_fitted

from ._compat import num_samples
from ._version import __version__
from .config import CERMConfig, TypedAdapterConfig
from .diagnostics import collect_fit_diagnostics
from .calibration import (
    CalibrationResult,
    apply_affine_calibration,
    cross_fitted_logits,
    fit_affine_calibrator,
)
from .training_cache import NewtonHistogramCache
from .program import ProgramBundle, SemanticProgram
from .shared_multitask import SharedFiniteStateModel, SharedFiniteStateProgram
from .params import (
    ResolvedCERMParams,
    explain_semantic_parameters,
    format_semantic_parameter_summary,
    resolve_estimator_parameters,
    resolve_semantic_alias,
    semantic_parameter_values,
)
from .resources import estimate_fit_resources, enforce_resource_plan
from .validation import (
    dense_numeric_matrix,
    validate_binary_target,
    validate_dataframe_schema,
    validate_estimator_parameters,
)
from ._internal.cerm_hybrid_quotient_block import HybridQuotientBlockCERM
from .selection import CrossFittedHybridCERM
from ._internal.cerm_typed_quotient_adapters_v4 import TypedQuotientAdapter


def _validated_classification_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return None
    weights = column_or_1d(sample_weight, warn=True).astype(np.float64)
    if len(weights) != int(n_samples):
        raise ValueError(
            "X and sample_weight have inconsistent lengths: "
            f"{n_samples} and {len(weights)}"
        )
    if not np.isfinite(weights).all():
        raise ValueError("sample_weight contains NaN or infinity")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values")
    if not np.any(weights > 0):
        raise ValueError(
            "sample_weight cannot be all zero; it must contain at least one positive value"
        )
    return weights


class CERMClassifier(ClassifierMixin, BaseEstimator):
    """Binary finite-state classifier with a sklearn-compatible public API.

    Parameters are intentionally explicit so the estimator can be cloned and
    tuned with ``GridSearchCV``.  The semantic learner and program compiler are
    separate: fitting creates ``program_``; ``optimize`` and ``compile`` produce
    deployment representations without changing the fitted statistical model.
    """

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
        memory_limit_mb: float | None = None,
        preset: str = "accurate",
        max_interaction_features: int | None = None,
        max_interactions: int | None = None,
        interaction_order: int = 2,
        reg_lambda: str | float = "auto",
        max_memory_mb: float | None = None,
        pair_feature_limit: int = 24,
        search_profile: str | None = None,
        replacement_objective: str = "balanced",
        prediction_backend: str = "optimized",
        random_state: int = 20260803,
        categorical_features: Sequence[str] | str | None = "auto",
        embedding_features: Sequence[str] | None = None,
        category_policy: str = "auto",
        max_identity_categories: int = 16,
        category_bins: int = 8,
        category_smoothing: float = 20.0,
        category_identity: str = "state",
        embedding_mode: str = "prototype",
        embedding_pca: int = 8,
        embedding_bins: int = 8,
        embedding_prototypes: int = 64,
        retain_embedding_raw: bool = False,
        missing_policy: str = "observed",
        category_newton_l2: float = 10.0,
        encoder_kind: str = "quantile",
        newton_prebins: int = 64,
        newton_gain_l2: float = 5.0,
        newton_min_hessian: float = 1.0,
        ranking_kind: str = "mi",
        ranking_l2: float = 5.0,
        ranking_prefilter_multiplier: int = 4,
        cost_per_byte: float = 0.0,
        cost_per_operator: float = 0.0,
        block_cost_per_byte: float = 0.0,
        block_cost_per_eval: float = 0.0,
        selection_strategy: str = "two_holdout",
        selection_folds: int = 3,
        selection_near_tie: float = 0.0015,
        selection_min_improvement: float = 0.0010,
        calibration: str = "none",
        calibration_folds: int = 3,
        calibration_l2: float = 1e-3,
        calibration_min_improvement: float = 1e-4,
        calibration_min_signal: float = 1.0,
        cache_training_statistics: bool = False,
        resource_policy: str = "raise",
        max_estimated_peak_memory_mb: float | None = 4096.0,
        max_pair_evaluations: int | None = 2_000_000,
        max_block_evaluations: int | None = 5_000_000,
        max_knn_distance_evaluations: int | None = 100_000_000,
        multiclass_strategy: str = "ovr",
        shared_multiclass_objective: str = "ovr",
        representation_strategy: str = "baseline",
        class_specific_budget: int = 12,
    ):
        self.search_effort = search_effort
        self.state_detail = state_detail
        self.feature_budget = feature_budget
        self.interaction_search_features = interaction_search_features
        self.interaction_budget = interaction_budget
        self.selection_fraction = selection_fraction
        self.feature_fraction = feature_fraction
        self.l2_regularization = l2_regularization
        self.memory_limit_mb = memory_limit_mb
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
        self.max_memory_mb = max_memory_mb
        self.pair_feature_limit = pair_feature_limit
        self.search_profile = search_profile
        self.replacement_objective = replacement_objective
        self.prediction_backend = prediction_backend
        self.random_state = random_state
        self.categorical_features = categorical_features
        self.embedding_features = embedding_features
        self.category_policy = category_policy
        self.max_identity_categories = max_identity_categories
        self.category_bins = category_bins
        self.category_smoothing = category_smoothing
        self.category_identity = category_identity
        self.embedding_mode = embedding_mode
        self.embedding_pca = embedding_pca
        self.embedding_bins = embedding_bins
        self.embedding_prototypes = embedding_prototypes
        self.retain_embedding_raw = retain_embedding_raw
        self.missing_policy = missing_policy
        self.category_newton_l2 = category_newton_l2
        self.encoder_kind = encoder_kind
        self.newton_prebins = newton_prebins
        self.newton_gain_l2 = newton_gain_l2
        self.newton_min_hessian = newton_min_hessian
        self.ranking_kind = ranking_kind
        self.ranking_l2 = ranking_l2
        self.ranking_prefilter_multiplier = ranking_prefilter_multiplier
        self.cost_per_byte = cost_per_byte
        self.cost_per_operator = cost_per_operator
        self.block_cost_per_byte = block_cost_per_byte
        self.block_cost_per_eval = block_cost_per_eval
        self.selection_strategy = selection_strategy
        self.selection_folds = selection_folds
        self.selection_near_tie = selection_near_tie
        self.selection_min_improvement = selection_min_improvement
        self.calibration = calibration
        self.calibration_folds = calibration_folds
        self.calibration_l2 = calibration_l2
        self.calibration_min_improvement = calibration_min_improvement
        self.calibration_min_signal = calibration_min_signal
        self.cache_training_statistics = cache_training_statistics
        self.resource_policy = resource_policy
        self.max_estimated_peak_memory_mb = max_estimated_peak_memory_mb
        self.max_pair_evaluations = max_pair_evaluations
        self.max_block_evaluations = max_block_evaluations
        self.max_knn_distance_evaluations = max_knn_distance_evaluations
        self.multiclass_strategy = multiclass_strategy
        self.shared_multiclass_objective = shared_multiclass_objective
        self.representation_strategy = representation_strategy
        self.class_specific_budget = class_specific_budget
        self._semantic_legacy_inputs = {
            "preset": preset,
            "max_bins": max_bins,
            "max_features": max_features,
            "max_interaction_features": max_interaction_features,
            "max_interactions": max_interactions,
            "subsample": subsample,
            "colsample": colsample,
            "reg_lambda": reg_lambda,
            "max_memory_mb": max_memory_mb,
        }
        self._refresh_semantic_aliases()

    def _refresh_semantic_aliases(self) -> None:
        conflicts: list[str] = []
        specs = (
            ("search_effort", "preset", "accurate", {"thorough": "accurate", "balanced": "balanced"}),
            ("state_detail", "max_bins", 16, {"coarse": 4, "medium": 8, "fine": 16}),
            ("feature_budget", "max_features", 64, None),
            ("interaction_search_features", "max_interaction_features", None, None),
            ("interaction_budget", "max_interactions", None, None),
            ("selection_fraction", "subsample", 1.0, None),
            ("feature_fraction", "colsample", 1.0, None),
            ("l2_regularization", "reg_lambda", "auto", None),
            ("memory_limit_mb", "max_memory_mb", None, None),
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
        if self.search_effort is not None and self.search_profile is not None:
            conflicts.append(
                "search_effort and search_profile cannot be specified together"
            )
        self._semantic_conflicts = tuple(conflicts)

    def set_params(self, **params):
        parameter_names = set(self.get_params(deep=False))
        legacy_names = set(getattr(self, "_semantic_legacy_inputs", {}))
        bulk_parameter_reset = bool(parameter_names) and parameter_names.issubset(params)
        result = super().set_params(**params)
        if not hasattr(self, "_semantic_legacy_inputs"):
            self._semantic_legacy_inputs = {
                name: getattr(self, name) for name in legacy_names
            }
        for name in legacy_names & set(params):
            self._semantic_legacy_inputs[name] = params[name]
        # sklearn's estimator contract requires a bulk set_params(**get_params())
        # round-trip to preserve the exact objects supplied by the caller.  A
        # normal partial update still refreshes CERM's convenience aliases
        # immediately; bulk resets defer alias resolution until validation/fit.
        if not bulk_parameter_reset:
            self._refresh_semantic_aliases()
        return result

    @classmethod
    def from_config(cls, config: CERMConfig) -> "CERMClassifier":
        adapter = config.adapter
        return cls(
            max_features=config.max_features,
            max_bins=config.max_bins,
            subsample=config.subsample,
            colsample=config.colsample,
            n_jobs=config.n_jobs,
            search_effort=config.search_effort,
            state_detail=config.state_detail,
            feature_budget=config.feature_budget,
            interaction_search_features=config.interaction_search_features,
            interaction_budget=config.interaction_budget,
            selection_fraction=config.selection_fraction,
            feature_fraction=config.feature_fraction,
            l2_regularization=config.l2_regularization,
            memory_limit_mb=config.memory_limit_mb,
            preset=config.preset,
            max_interaction_features=config.max_interaction_features,
            max_interactions=config.max_interactions,
            interaction_order=config.interaction_order,
            reg_lambda=config.reg_lambda,
            max_memory_mb=config.max_memory_mb,
            pair_feature_limit=config.pair_feature_limit,
            search_profile=config.search_profile,
            replacement_objective=config.replacement_objective,
            prediction_backend=config.prediction_backend,
            random_state=config.random_state,
            categorical_features=adapter.categorical_features,
            embedding_features=adapter.embedding_features,
            category_policy=adapter.category_policy,
            max_identity_categories=adapter.max_identity_categories,
            category_bins=adapter.category_bins,
            category_smoothing=adapter.category_smoothing,
            category_identity=adapter.category_identity,
            embedding_mode=adapter.embedding_mode,
            embedding_pca=config.adapter.embedding_pca,
            embedding_bins=adapter.embedding_bins,
            embedding_prototypes=adapter.embedding_prototypes,
            retain_embedding_raw=adapter.retain_embedding_raw,
            missing_policy=adapter.missing_policy,
            category_newton_l2=adapter.category_newton_l2,
            encoder_kind=config.encoder_kind,
            newton_prebins=config.newton_prebins,
            newton_gain_l2=config.newton_gain_l2,
            newton_min_hessian=config.newton_min_hessian,
            ranking_kind=config.ranking_kind,
            ranking_l2=config.ranking_l2,
            ranking_prefilter_multiplier=config.ranking_prefilter_multiplier,
            cost_per_byte=config.cost_per_byte,
            cost_per_operator=config.cost_per_operator,
            block_cost_per_byte=config.block_cost_per_byte,
            block_cost_per_eval=config.block_cost_per_eval,
            selection_strategy=config.selection_strategy,
            selection_folds=config.selection_folds,
            selection_near_tie=config.selection_near_tie,
            selection_min_improvement=config.selection_min_improvement,
            calibration=config.calibration,
            calibration_folds=config.calibration_folds,
            calibration_l2=config.calibration_l2,
            calibration_min_improvement=config.calibration_min_improvement,
            calibration_min_signal=config.calibration_min_signal,
            cache_training_statistics=config.cache_training_statistics,
            resource_policy=config.resource_policy,
            max_estimated_peak_memory_mb=config.max_estimated_peak_memory_mb,
            max_pair_evaluations=config.max_pair_evaluations,
            max_block_evaluations=config.max_block_evaluations,
            max_knn_distance_evaluations=config.max_knn_distance_evaluations,
            multiclass_strategy=config.multiclass_strategy,
            shared_multiclass_objective=config.shared_multiclass_objective,
            representation_strategy=config.representation_strategy,
            class_specific_budget=config.class_specific_budget,
        )

    def to_config(self) -> CERMConfig:
        categorical = self.categorical_features
        if categorical in (None, "auto"):
            categorical_config = categorical
        else:
            categorical_config = tuple(categorical)
        return CERMConfig(
            max_features=int(self.max_features),
            max_bins=int(self.max_bins),
            subsample=float(self.subsample),
            colsample=float(self.colsample),
            n_jobs=None if self.n_jobs is None else int(self.n_jobs),
            search_effort=self.search_effort,
            state_detail=self.state_detail,
            feature_budget=(None if self.feature_budget is None else int(self.feature_budget)),
            interaction_search_features=(
                None if self.interaction_search_features is None
                else int(self.interaction_search_features)
            ),
            interaction_budget=(
                None if self.interaction_budget is None else int(self.interaction_budget)
            ),
            selection_fraction=(
                None if self.selection_fraction is None else float(self.selection_fraction)
            ),
            feature_fraction=(
                None if self.feature_fraction is None else float(self.feature_fraction)
            ),
            l2_regularization=self.l2_regularization,
            memory_limit_mb=(
                None if self.memory_limit_mb is None else float(self.memory_limit_mb)
            ),
            preset=str(self.preset),
            max_interaction_features=(
                None if self.max_interaction_features is None
                else int(self.max_interaction_features)
            ),
            max_interactions=(
                None if self.max_interactions is None else int(self.max_interactions)
            ),
            interaction_order=int(self.interaction_order),
            reg_lambda=self.reg_lambda,
            max_memory_mb=(
                None if self.max_memory_mb is None else float(self.max_memory_mb)
            ),
            pair_feature_limit=int(self.pair_feature_limit),
            search_profile=(
                None if self.search_profile is None else str(self.search_profile)
            ),
            replacement_objective=str(self.replacement_objective),
            prediction_backend=str(self.prediction_backend),
            random_state=int(self.random_state),
            encoder_kind=str(self.encoder_kind),
            newton_prebins=int(self.newton_prebins),
            newton_gain_l2=float(self.newton_gain_l2),
            newton_min_hessian=float(self.newton_min_hessian),
            ranking_kind=str(self.ranking_kind),
            ranking_l2=float(self.ranking_l2),
            ranking_prefilter_multiplier=int(self.ranking_prefilter_multiplier),
            cost_per_byte=float(self.cost_per_byte),
            cost_per_operator=float(self.cost_per_operator),
            block_cost_per_byte=float(self.block_cost_per_byte),
            block_cost_per_eval=float(self.block_cost_per_eval),
            selection_strategy=str(self.selection_strategy),
            selection_folds=int(self.selection_folds),
            selection_near_tie=float(self.selection_near_tie),
            selection_min_improvement=float(self.selection_min_improvement),
            calibration=str(self.calibration),
            calibration_folds=int(self.calibration_folds),
            calibration_l2=float(self.calibration_l2),
            calibration_min_improvement=float(self.calibration_min_improvement),
            calibration_min_signal=float(self.calibration_min_signal),
            cache_training_statistics=bool(self.cache_training_statistics),
            resource_policy=str(self.resource_policy),
            max_estimated_peak_memory_mb=(
                None if self.max_estimated_peak_memory_mb is None
                else float(self.max_estimated_peak_memory_mb)
            ),
            max_pair_evaluations=(
                None if self.max_pair_evaluations is None else int(self.max_pair_evaluations)
            ),
            max_block_evaluations=(
                None if self.max_block_evaluations is None else int(self.max_block_evaluations)
            ),
            max_knn_distance_evaluations=(
                None if self.max_knn_distance_evaluations is None
                else int(self.max_knn_distance_evaluations)
            ),
            multiclass_strategy=str(self.multiclass_strategy),
            shared_multiclass_objective=str(self.shared_multiclass_objective),
            representation_strategy=str(self.representation_strategy),
            class_specific_budget=int(self.class_specific_budget),
            adapter=TypedAdapterConfig(
                categorical_features=categorical_config,
                embedding_features=(
                    None if self.embedding_features is None
                    else tuple(self.embedding_features)
                ),
                category_policy=str(self.category_policy),
                max_identity_categories=int(self.max_identity_categories),
                category_bins=int(self.category_bins),
                category_smoothing=float(self.category_smoothing),
                category_newton_l2=float(self.category_newton_l2),
                category_identity=str(self.category_identity),
                embedding_mode=str(self.embedding_mode),
                embedding_pca=int(self.embedding_pca),
                embedding_bins=int(self.embedding_bins),
                embedding_prototypes=int(self.embedding_prototypes),
                retain_embedding_raw=bool(self.retain_embedding_raw),
                missing_policy=str(self.missing_policy),
            ),
        )

    def _validate_params(self) -> None:
        self._refresh_semantic_aliases()
        if self._semantic_conflicts:
            raise ValueError("; ".join(self._semantic_conflicts))
        validate_estimator_parameters(self)

    def resolve_params(self) -> ResolvedCERMParams:
        """Resolve presets and compatibility aliases without fitting."""
        self._validate_params()
        return resolve_estimator_parameters(self)

    def get_user_params(self) -> dict[str, object]:
        """Return the compact, recommended parameter view."""
        return semantic_parameter_values(self, task="classifier")

    def explain_params(self) -> dict[str, dict[str, object]]:
        """Return effective values together with plain-language meanings."""
        return explain_semantic_parameters(self, task="classifier")

    def parameter_summary(self) -> str:
        """Return a human-readable summary of the effective model choices."""
        return format_semantic_parameter_summary(self, task="classifier")

    @staticmethod
    def _infer_categorical(frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        )

    def _resolved_categorical(self, X: pd.DataFrame) -> tuple[str, ...]:
        if self.categorical_features == "auto":
            return self._infer_categorical(X)
        return tuple(self.categorical_features or ())

    def _make_adapter(self, X: pd.DataFrame) -> TypedQuotientAdapter | None:
        categorical = self._resolved_categorical(X)
        embedding = tuple(self.embedding_features or ())
        has_missing = bool(X.isna().any().any())
        has_non_numeric = bool(categorical)
        if not (categorical or embedding or has_missing or has_non_numeric):
            self.categorical_features_ = ()
            self.embedding_features_ = ()
            return None
        self.categorical_features_ = categorical
        self.embedding_features_ = embedding
        return TypedQuotientAdapter(
            categorical_columns=categorical,
            embedding_columns=embedding,
            category_policy=self.category_policy,
            max_identity_categories=int(self.max_identity_categories),
            category_bins=int(self.category_bins),
            category_smoothing=float(self.category_smoothing),
            category_newton_l2=float(self.category_newton_l2),
            category_identity=self.category_identity,
            embedding_mode=self.embedding_mode,
            embedding_pca=int(self.embedding_pca),
            embedding_bins=int(self.embedding_bins),
            embedding_prototypes=int(self.embedding_prototypes),
            retain_embedding_raw=bool(self.retain_embedding_raw),
            missing_policy=self.missing_policy,
            random_state=int(self.random_state),
            n_jobs=self.n_jobs,
        )

    def _core_kwargs(self, feature_kinds, feature_cardinalities) -> dict:
        return {
            "feature_kinds": feature_kinds,
            "max_bins": int(self.max_bins),
            "selection_subsample": float(self.subsample),
            "n_jobs": self.n_jobs,
            "feature_cardinalities": feature_cardinalities,
            "encoder_kind": self.encoder_kind,
            "newton_prebins": int(self.newton_prebins),
            "newton_gain_l2": float(self.newton_gain_l2),
            "newton_min_hessian": float(self.newton_min_hessian),
            "ranking_kind": self.ranking_kind,
            "ranking_l2": float(self.ranking_l2),
            "ranking_prefilter_multiplier": int(self.ranking_prefilter_multiplier),
            "cost_per_byte": float(self.cost_per_byte),
            "cost_per_operator": float(self.cost_per_operator),
            "block_cost_per_byte": float(self.block_cost_per_byte),
            "block_cost_per_eval": float(self.block_cost_per_eval),
        }

    def _effective_pair_feature_limit(self) -> int:
        return int(resolve_estimator_parameters(self).max_interaction_features)

    def _make_learner(
        self,
        *,
        random_state: int,
        core_kwargs: dict,
        strategy: str | None = None,
    ):
        strategy = self.selection_strategy if strategy is None else strategy
        resolved = resolve_estimator_parameters(self)
        common = dict(
            max_features=int(resolved.max_features),
            pair_feature_limit=int(resolved.max_interaction_features),
            search_profile=str(resolved.search_profile),
            max_interactions=int(resolved.max_interactions),
            fixed_C=resolved.fixed_C,
            random_state=int(random_state),
            replacement_objective=self.replacement_objective,
            **core_kwargs,
        )
        if strategy == "cross_fitted":
            return CrossFittedHybridCERM(
                cv_folds=int(self.selection_folds),
                near_tie=float(self.selection_near_tie),
                min_improvement=float(self.selection_min_improvement),
                **common,
            )
        return HybridQuotientBlockCERM(**common)

    def _fit_optional_calibration(
        self,
        matrix: np.ndarray,
        y_binary: np.ndarray,
        core_kwargs: dict,
    ) -> None:
        if self.calibration == "none":
            self.calibration_result_ = CalibrationResult(
                method="none",
                accepted=False,
                scale=1.0,
                offset=0.0,
                raw_logloss=float("nan"),
                calibrated_logloss=float("nan"),
                improvement=0.0,
                improvement_se=0.0,
                signal_to_noise=0.0,
                folds=0,
                converged=True,
            )
            self.model_.calibration_result_ = self.calibration_result_
            return

        selected = self.model_.selected_hybrid_config_

        def learner_factory(seed: int):
            return self._make_learner(
                random_state=seed,
                core_kwargs=core_kwargs,
                strategy="cross_fitted",
            )

        logits, folds = cross_fitted_logits(
            matrix,
            y_binary,
            learner_factory=learner_factory,
            selected_config=selected,
            n_splits=int(self.calibration_folds),
            random_state=int(self.random_state) + 15485863,
        )
        result = fit_affine_calibrator(
            logits,
            y_binary,
            method=self.calibration,
            l2=float(self.calibration_l2),
            min_improvement=float(self.calibration_min_improvement),
            min_signal=float(self.calibration_min_signal),
            folds=folds,
        )
        self.calibration_result_ = result
        if result.accepted:
            apply_affine_calibration(
                self.model_,
                result.scale,
                result.offset,
                matrix,
            )
        self.model_.calibration_result_ = result

    def estimate_fit_resources(self, X: pd.DataFrame | np.ndarray):
        """Return a structural fit-cost and memory plan without fitting."""
        self._validate_params()
        return estimate_fit_resources(self, X)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable, sample_weight=None) -> "CERMClassifier":
        if y is None:
            return self._fit_binary(X, y, sample_weight=sample_weight)
        target_type = type_of_target(y, input_name="y", raise_unknown=True)
        if target_type == "binary":
            return self._fit_binary(X, y, sample_weight=sample_weight)
        if target_type == "multiclass":
            if (
                self.representation_strategy == "adaptive"
                and self.multiclass_strategy != "shared"
            ):
                raise ValueError(
                    "representation_strategy='adaptive' requires "
                    "multiclass_strategy='shared'"
                )
            if self.multiclass_strategy == "error":
                raise ValueError("multiclass target is disabled by multiclass_strategy='error'")
            if self.multiclass_strategy == "shared":
                return self._fit_multiclass_shared(X, y, sample_weight=sample_weight)
            if self.multiclass_strategy != "ovr":
                raise ValueError("multiclass_strategy must be 'ovr', 'shared', or 'error'")
            return self._fit_multiclass(X, y, sample_weight=sample_weight)
        if target_type == "multilabel-indicator":
            raise ValueError(
                "Use CERMMultiLabelClassifier for multilabel indicator targets"
            )
        return self._fit_binary(X, y, sample_weight=sample_weight)

    def _prepare_target_neutral_input(self, X, reference_binary: np.ndarray):
        if isinstance(X, pd.DataFrame):
            frame = X
            if frame.columns.has_duplicates:
                raise ValueError("DataFrame columns must be unique")
            categorical = self._resolved_categorical(frame)
            embedding = tuple(self.embedding_features or ())
            if embedding:
                return None
            for column in categorical:
                if self.category_policy in {"ordered", "newton"}:
                    return None
                if self.category_policy == "auto" and int(frame[column].nunique(dropna=True)) > int(self.max_identity_categories):
                    return None
            adapter = self._make_adapter(frame)
            if adapter is None:
                matrix = dense_numeric_matrix(frame)
                names = tuple(map(str, frame.columns))
                kinds = None
                cards = None
                metadata = None
            else:
                adapted = adapter.fit_transform(frame, reference_binary)
                matrix = adapted.matrix
                names = tuple(adapted.feature_names)
                kinds = tuple(adapted.feature_kinds)
                cards = tuple(adapted.cardinalities)
                metadata = dict(adapted.metadata)
            return {
                "matrix": matrix,
                "adapter": adapter,
                "input_columns": tuple(frame.columns),
                "feature_names_in": np.asarray(frame.columns, dtype=object),
                "n_features_in": int(frame.shape[1]),
                "adapted_names": names,
                "feature_kinds": kinds,
                "feature_cardinalities": cards,
                "adapter_metadata": metadata,
                "categorical_features": tuple(categorical),
                "embedding_features": (),
            }

        matrix = dense_numeric_matrix(X)
        return {
            "matrix": matrix,
            "adapter": None,
            "input_columns": None,
            "feature_names_in": None,
            "n_features_in": int(matrix.shape[1]),
            "adapted_names": tuple(f"x{i}" for i in range(matrix.shape[1])),
            "feature_kinds": None,
            "feature_cardinalities": None,
            "adapter_metadata": None,
            "categorical_features": (),
            "embedding_features": (),
        }

    def _fit_binary_from_prepared(
        self,
        prepared: dict,
        y_binary: np.ndarray,
        *,
        classes: tuple[Any, Any] = (0, 1),
        sample_weight=None,
    ) -> "CERMClassifier":
        self.task_type_ = "binary"
        fit_start = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        start = time.perf_counter()
        self._validate_params()
        resolved = resolve_estimator_parameters(self)
        self.resolved_params_ = resolved.to_dict()
        matrix_full = np.asarray(prepared["matrix"], dtype=np.float64)
        self.fit_resource_plan_ = estimate_fit_resources(self, matrix_full)
        enforce_resource_plan(self, self.fit_resource_plan_)
        self._label_encoder_ = LabelEncoder().fit(np.asarray(classes))
        self.classes_ = np.asarray(classes)
        self.feature_names_in_ = prepared["feature_names_in"]
        self.n_features_in_ = int(prepared["n_features_in"])
        self.input_columns_ = prepared["input_columns"]
        self.adapter_ = prepared["adapter"]
        self.adapter_output_ = None
        self.adapter_metadata_ = prepared["adapter_metadata"]
        self.categorical_features_ = prepared["categorical_features"]
        self.embedding_features_ = prepared["embedding_features"]
        stage_seconds["validation"] = time.perf_counter() - start

        start = time.perf_counter()
        names = tuple(prepared["adapted_names"])
        feature_kinds = prepared["feature_kinds"]
        feature_cardinalities = prepared["feature_cardinalities"]
        self.n_full_adapted_features_ = int(matrix_full.shape[1])
        keep = max(1, min(
            self.n_full_adapted_features_,
            int(np.ceil(float(resolved.colsample) * self.n_full_adapted_features_)),
        ))
        if keep < self.n_full_adapted_features_:
            rng = np.random.default_rng(int(self.random_state) + 32452843)
            feature_indices = np.sort(
                rng.choice(self.n_full_adapted_features_, size=keep, replace=False)
            ).astype(np.int64)
            matrix = np.ascontiguousarray(matrix_full[:, feature_indices])
            names = tuple(names[index] for index in feature_indices)
            if feature_kinds is not None:
                feature_kinds = tuple(feature_kinds[index] for index in feature_indices)
            if feature_cardinalities is not None:
                feature_cardinalities = tuple(feature_cardinalities[index] for index in feature_indices)
        else:
            feature_indices = np.arange(self.n_full_adapted_features_, dtype=np.int64)
            matrix = matrix_full
        self.adapted_feature_indices_ = feature_indices
        self.n_adapted_features_ = int(matrix.shape[1])
        self.adapted_feature_names_ = np.asarray(names, dtype=object)
        self.transient_adapted_matrix_bytes_ = int(matrix.nbytes)
        self.retained_adapter_output_bytes_ = 0
        stage_seconds["adapter"] = time.perf_counter() - start

        core_kwargs = self._core_kwargs(feature_kinds, feature_cardinalities)
        start = time.perf_counter()
        learner = self._make_learner(
            random_state=int(self.random_state), core_kwargs=core_kwargs
        )
        if sample_weight is None:
            self.model_ = learner.fit(matrix, np.asarray(y_binary, dtype=np.int32))
        else:
            self.model_ = learner.fit(
                matrix, np.asarray(y_binary, dtype=np.int32),
                sample_weight=sample_weight,
            )
        stage_seconds["learner"] = time.perf_counter() - start

        start = time.perf_counter()
        self._fit_optional_calibration(matrix, np.asarray(y_binary, dtype=np.int32), core_kwargs)
        stage_seconds["calibration"] = time.perf_counter() - start

        start = time.perf_counter()
        self.program_ = SemanticProgram(
            model=self.model_,
            adapter=self.adapter_,
            input_columns=self.input_columns_,
            adapted_feature_names=tuple(names),
            feature_indices=tuple(int(i) for i in self.adapted_feature_indices_),
            classes=tuple(self.classes_),
            library_version=self.VERSION,
        )
        self._prediction_program_ = (
            self.program_.optimize(self.replacement_objective)
            if self.prediction_backend == "optimized"
            else self.program_
        )
        if self.cache_training_statistics:
            self.training_cache_ = self._build_newton_cache_from_matrix(
                matrix, np.asarray(y_binary, dtype=np.int32), prediction="base",
                level=int(resolved.max_bins),
                feature_limit=int(self._effective_pair_feature_limit()),
            )
        stage_seconds["program_and_cache"] = time.perf_counter() - start
        self.fit_stage_seconds_ = stage_seconds
        self.fit_seconds_ = time.perf_counter() - fit_start
        self.fit_diagnostics_ = collect_fit_diagnostics(self)
        self.sample_weighted_ = sample_weight is not None
        self.sample_weight_sum_ = (
            None if sample_weight is None else float(np.asarray(sample_weight).sum())
        )
        return self

    def _fit_multiclass_shared(self, X, y, sample_weight=None) -> "CERMClassifier":
        fit_start = time.perf_counter()
        self._validate_params()
        if self.calibration != "none":
            raise ValueError("shared multiclass currently requires calibration='none'")
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is currently supported for binary and OVR multiclass, "
                "not shared multiclass"
            )
        if self.encoder_kind != "quantile":
            raise ValueError("shared multiclass currently requires encoder_kind='quantile'")
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError(
                "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
            )
        n_samples = num_samples(X)
        weights = _validated_classification_sample_weight(sample_weight, n_samples)
        y_array = column_or_1d(y, warn=True)
        if len(y_array) != n_samples:
            raise ValueError(
                f"X and y have inconsistent lengths: {n_samples} and {len(y_array)}"
            )
        if pd.isna(y_array).any():
            raise ValueError("y contains missing values")
        self._label_encoder_ = LabelEncoder().fit(y_array)
        encoded = self._label_encoder_.transform(y_array).astype(np.int32)
        self.classes_ = self._label_encoder_.classes_
        if len(self.classes_) < 3:
            return self._fit_binary(X, y_array, sample_weight=sample_weight)
        counts = np.bincount(encoded)
        if counts.min() < 4:
            raise ValueError(
                "shared multiclass requires at least 4 samples in each class"
            )
        shared = self._prepare_target_neutral_input(
            X, (encoded == 0).astype(np.int32)
        )
        if shared is None:
            if self.representation_strategy == "adaptive":
                raise ValueError(
                    "adaptive shared representation requires target-neutral "
                    "numeric or identity-categorical preprocessing"
                )
            original = self.multiclass_strategy
            try:
                self.multiclass_strategy = "ovr"
                fitted = self._fit_multiclass(X, y_array, sample_weight=sample_weight)
                fitted.fit_diagnostics_["requested_strategy"] = original
                fitted.fit_diagnostics_["shared_fallback"] = "target-aware-preprocessing"
                return fitted
            finally:
                self.multiclass_strategy = original

        resolved = resolve_estimator_parameters(self)
        self.resolved_params_ = resolved.to_dict()
        matrix_full = np.asarray(shared["matrix"], dtype=np.float64)
        names = tuple(shared["adapted_names"])
        feature_kinds = shared["feature_kinds"]
        feature_cardinalities = shared["feature_cardinalities"]
        self.n_full_adapted_features_ = int(matrix_full.shape[1])
        keep = max(1, min(
            self.n_full_adapted_features_,
            int(np.ceil(float(resolved.colsample) * self.n_full_adapted_features_)),
        ))
        if keep < self.n_full_adapted_features_:
            rng = np.random.default_rng(int(self.random_state) + 32452843)
            feature_indices = np.sort(
                rng.choice(self.n_full_adapted_features_, size=keep, replace=False)
            ).astype(np.int64)
            matrix = np.ascontiguousarray(matrix_full[:, feature_indices])
            names = tuple(names[index] for index in feature_indices)
            if feature_kinds is not None:
                feature_kinds = tuple(feature_kinds[index] for index in feature_indices)
            if feature_cardinalities is not None:
                feature_cardinalities = tuple(
                    feature_cardinalities[index] for index in feature_indices
                )
        else:
            feature_indices = np.arange(self.n_full_adapted_features_, dtype=np.int64)
            matrix = matrix_full

        model = SharedFiniteStateModel(
            task_type="multiclass",
            max_features=int(resolved.max_features),
            pair_feature_limit=int(resolved.max_interaction_features),
            max_bins=int(resolved.max_bins),
            random_state=int(self.random_state),
            feature_kinds=feature_kinds,
            feature_cardinalities=feature_cardinalities,
            max_pairs=(int(resolved.max_interactions) if int(resolved.interaction_order) >= 2 else 0),
            fixed_C=resolved.fixed_C,
            search_profile=str(resolved.search_profile),
            selection_subsample=float(resolved.subsample),
            n_jobs=self.n_jobs,
            multiclass_objective=str(self.shared_multiclass_objective),
            representation_strategy=str(self.representation_strategy),
            class_specific_budget=int(self.class_specific_budget),
        ).fit(matrix, encoded)
        self.task_type_ = "multiclass"
        self.model_ = model
        self.adapter_ = shared["adapter"]
        self.input_columns_ = shared["input_columns"]
        self.feature_names_in_ = shared["feature_names_in"]
        self.n_features_in_ = int(shared["n_features_in"])
        self.adapter_metadata_ = shared["adapter_metadata"]
        self.categorical_features_ = shared["categorical_features"]
        self.embedding_features_ = shared["embedding_features"]
        self.adapted_feature_indices_ = feature_indices
        self.n_adapted_features_ = int(matrix.shape[1])
        self.adapted_feature_names_ = np.asarray(names, dtype=object)
        self.transient_adapted_matrix_bytes_ = int(matrix.nbytes)
        self.retained_adapter_output_bytes_ = 0
        self.program_ = SharedFiniteStateProgram(
            model=model,
            adapter=self.adapter_,
            input_columns=self.input_columns_,
            adapted_feature_names=tuple(names),
            feature_indices=tuple(int(index) for index in feature_indices),
            task_type="multiclass",
            classes=tuple(self.classes_),
            library_version=self.VERSION,
            metadata={
                "strategy": "shared-finite-state-softmax",
                "representation": "shared-main-pair",
                "multiclass_objective": str(model.multiclass_objective_),
                "multinomial_weight": float(model.config_.multinomial_weight),
                "representation_strategy": str(model.representation_strategy_),
                "class_specific_budget": int(self.class_specific_budget),
                "class_delta_selected_budget": int(
                    model.class_delta_selected_budget_
                ),
            },
        )
        self._prediction_program_ = self.program_
        self.fit_seconds_ = float(time.perf_counter() - fit_start)
        self.fit_stage_seconds_ = {"shared_multiclass": self.fit_seconds_}
        self.fit_diagnostics_ = {
            "task_type": "multiclass",
            "strategy": "shared-finite-state-softmax",
            "multiclass_objective": str(model.multiclass_objective_),
            "multinomial_weight": float(model.config_.multinomial_weight),
            "representation_strategy": str(model.representation_strategy_),
            "class_specific_budget": int(self.class_specific_budget),
            "class_delta_selected_budget": int(
                model.class_delta_selected_budget_
            ),
            "class_delta_validation_improvement": float(
                model.class_delta_validation_improvement_
            ),
            "class_delta_validation_scores": list(
                model.class_delta_validation_scores_
            ),
            "class_delta_fit_diagnostics": list(
                model.class_delta_fit_diagnostics_
            ),
            "pre_guard_multiclass_objective": str(
                model.pre_guard_multiclass_objective_
            ),
            "pre_guard_multinomial_weight": float(
                model.pre_guard_multinomial_weight_
            ),
            "objective_guard_applied": bool(model.objective_guard_applied_),
            "objective_guard_margin": float(model.objective_guard_margin_),
            "best_ovr_validation_loss": float(
                model.best_ovr_validation_loss_
            ),
            "non_ovr_validation_advantage": float(
                model.non_ovr_validation_advantage_
            ),
            "n_classes": int(len(self.classes_)),
            "shared_preprocessing": True,
            "shared_state_encoder": True,
            "selected_config": model.config_.__dict__,
            "design_dim": int(model.design_dim_),
            "pair_count": int(len(model.pairs_)),
            "model_bytes_estimate": int(self.model_bytes_estimate_),
            "fit_seconds": self.fit_seconds_,
        }
        return self

    def _fit_multiclass(self, X, y, sample_weight=None) -> "CERMClassifier":
        fit_start = time.perf_counter()
        self._validate_params()
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError(
                "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
            )
        n_samples = num_samples(X)
        weights = _validated_classification_sample_weight(sample_weight, n_samples)
        y_array = column_or_1d(y, warn=True)
        if len(y_array) != n_samples:
            raise ValueError(
                f"X and y have inconsistent lengths: {n_samples} and {len(y_array)}"
            )
        if pd.isna(y_array).any():
            raise ValueError("y contains missing values")
        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) < 3:
            return self._fit_binary(X, y_array, sample_weight=weights)
        if counts.min() < 4:
            raise ValueError(
                "CERMClassifier requires at least 4 samples in each class for "
                "one-vs-rest selection splits"
            )
        self.task_type_ = "multiclass"
        self._label_encoder_ = LabelEncoder().fit(y_array)
        encoded = self._label_encoder_.transform(y_array).astype(np.int32)
        self.classes_ = self._label_encoder_.classes_

        params = self.get_params(deep=False)
        params["multiclass_strategy"] = "error"
        shared = self._prepare_target_neutral_input(X, (encoded == 0).astype(np.int32))

        def fit_head(index: int):
            child_params = dict(params)
            child_params["random_state"] = int(self.random_state) + 104729 * index
            child = CERMClassifier(**child_params)
            binary = (encoded == index).astype(np.int32)
            if shared is None:
                child.fit(X, binary, sample_weight=weights)
            else:
                child._fit_binary_from_prepared(shared, binary, sample_weight=weights)
            return child

        if self.n_jobs in (None, 1):
            estimators = [fit_head(index) for index in range(len(self.classes_))]
        else:
            estimators = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")(
                joblib.delayed(fit_head)(index) for index in range(len(self.classes_))
            )
        self.estimators_ = estimators
        self.program_ = ProgramBundle(
            programs=tuple(estimator.program_ for estimator in estimators),
            task_type="multiclass",
            classes=tuple(self.classes_),
            library_version=self.VERSION,
            metadata={
                "strategy": "one-vs-rest",
                "probability_coupling": "softmax-of-ovr-logits",
                "preprocessing": "shared-target-neutral" if shared is not None else "independent-target-aware",
            },
        )
        self._prediction_program_ = (
            self.program_.optimize(self.replacement_objective)
            if self.prediction_backend == "optimized"
            else self.program_
        )
        first = estimators[0]
        self.n_features_in_ = first.n_features_in_
        self.feature_names_in_ = getattr(first, "feature_names_in_", None)
        self.input_columns_ = first.input_columns_
        self.adapter_ = shared["adapter"] if shared is not None else None
        self.categorical_features_ = getattr(first, "categorical_features_", ())
        self.embedding_features_ = getattr(first, "embedding_features_", ())
        self.adapted_feature_names_ = first.adapted_feature_names_.copy()
        self.adapted_feature_indices_ = first.adapted_feature_indices_.copy()
        self.n_full_adapted_features_ = first.n_full_adapted_features_
        self.n_adapted_features_ = first.n_adapted_features_
        self.fit_seconds_ = float(time.perf_counter() - fit_start)
        self.fit_stage_seconds_ = {
            "independent_binary_heads": float(
                sum(getattr(estimator, "fit_seconds_", 0.0) for estimator in estimators)
            )
        }
        self.fit_diagnostics_ = {
            "task_type": "multiclass",
            "strategy": "one-vs-rest",
            "n_classes": int(len(self.classes_)),
            "model_bytes_estimate": int(self.model_bytes_estimate_),
            "shared_preprocessing": bool(shared is not None),
            "fit_seconds": self.fit_seconds_,
            "sample_weighted": weights is not None,
            "sample_weight_sum": None if weights is None else float(weights.sum()),
            "head_diagnostics": [
                estimator.fit_diagnostics_.to_dict()
                if hasattr(estimator.fit_diagnostics_, "to_dict")
                else estimator.fit_diagnostics_
                for estimator in estimators
            ],
        }
        return self

    def _fit_binary(self, X: pd.DataFrame | np.ndarray, y: Iterable, sample_weight=None) -> "CERMClassifier":
        self.task_type_ = "binary"
        fit_start = time.perf_counter()
        stage_seconds: dict[str, float] = {}

        start = time.perf_counter()
        self._validate_params()
        resolved = resolve_estimator_parameters(self)
        self.resolved_params_ = resolved.to_dict()
        if X is None:
            raise ValueError("X cannot be None")
        if sparse.issparse(X):
            raise TypeError(
                "sparse input is not supported; provide a dense ndarray or pandas DataFrame"
            )
        self.fit_resource_plan_ = estimate_fit_resources(self, X)
        enforce_resource_plan(self, self.fit_resource_plan_)
        try:
            n_samples = num_samples(X)
        except TypeError as exc:
            raise TypeError("X must be a two-dimensional array-like object") from exc
        weights = _validated_classification_sample_weight(sample_weight, n_samples)
        if weights is not None and self.selection_strategy == "cross_fitted":
            raise ValueError(
                "sample_weight is not yet supported with selection_strategy='cross_fitted'"
            )
        if weights is not None and self.calibration != "none":
            raise ValueError("sample_weight currently requires calibration='none'")
        shape = getattr(X, "shape", None)
        n_features = int(shape[1]) if shape is not None and len(shape) >= 2 else None
        y_array = validate_binary_target(
            y, n_samples=n_samples, n_features=n_features
        )
        if weights is not None and np.any(weights == 0.0):
            positive = weights > 0.0
            if isinstance(X, pd.DataFrame):
                X = X.iloc[np.flatnonzero(positive)]
            else:
                X = np.asarray(X)[positive]
            y_array = y_array[positive]
            weights = weights[positive]
            n_samples = int(len(y_array))
            y_array = validate_binary_target(
                y_array, n_samples=n_samples, n_features=n_features
            )
        self._label_encoder_ = LabelEncoder().fit(y_array)
        y_binary = self._label_encoder_.transform(y_array).astype(np.int32)
        self.classes_ = self._label_encoder_.classes_
        stage_seconds["validation"] = time.perf_counter() - start

        start = time.perf_counter()
        feature_kinds = None
        feature_cardinalities = None
        self.adapter_output_ = None
        self.adapter_metadata_ = None
        if isinstance(X, pd.DataFrame):
            frame = X
            if frame.columns.has_duplicates:
                raise ValueError("DataFrame columns must be unique")
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            self.n_features_in_ = frame.shape[1]
            self.input_columns_ = tuple(frame.columns)
            self.adapter_ = self._make_adapter(frame)
            if self.adapter_ is None:
                matrix = dense_numeric_matrix(frame)
                adapted_names = tuple(map(str, frame.columns))
            else:
                if weights is not None:
                    target_aware_categories = tuple(
                        column
                        for column in self.categorical_features_
                        if self.adapter_._category_mode(frame[column]) != "identity"
                    )
                    if target_aware_categories or self.embedding_features_:
                        raise ValueError(
                            "sample_weight is not supported with target-aware typed preprocessing; "
                            "use category_policy='identity' without embedding features, or numeric input"
                        )
                adapted = self.adapter_.fit_transform(frame, y_binary)
                matrix = adapted.matrix
                adapted_names = tuple(adapted.feature_names)
                feature_kinds = tuple(adapted.feature_kinds)
                feature_cardinalities = tuple(adapted.cardinalities)
                self.adapter_metadata_ = dict(adapted.metadata)
                del adapted
        else:
            matrix = dense_numeric_matrix(X)
            self.feature_names_in_ = None
            self.n_features_in_ = matrix.shape[1]
            self.input_columns_ = None
            self.adapter_ = None
            self.categorical_features_ = ()
            self.embedding_features_ = ()
            adapted_names = tuple(f"x{i}" for i in range(matrix.shape[1]))

        self.n_full_adapted_features_ = int(matrix.shape[1])
        keep = max(1, min(
            self.n_full_adapted_features_,
            int(np.ceil(float(resolved.colsample) * self.n_full_adapted_features_)),
        ))
        if keep < self.n_full_adapted_features_:
            rng = np.random.default_rng(int(self.random_state) + 32452843)
            feature_indices = np.sort(
                rng.choice(self.n_full_adapted_features_, size=keep, replace=False)
            ).astype(np.int64)
            matrix = np.ascontiguousarray(matrix[:, feature_indices])
            adapted_names = tuple(adapted_names[index] for index in feature_indices)
            if feature_kinds is not None:
                feature_kinds = tuple(feature_kinds[index] for index in feature_indices)
            if feature_cardinalities is not None:
                feature_cardinalities = tuple(
                    feature_cardinalities[index] for index in feature_indices
                )
        else:
            feature_indices = np.arange(self.n_full_adapted_features_, dtype=np.int64)
        self.adapted_feature_indices_ = feature_indices
        self.n_adapted_features_ = int(matrix.shape[1])
        self.adapted_feature_names_ = np.asarray(adapted_names, dtype=object)
        self.transient_adapted_matrix_bytes_ = int(matrix.nbytes)
        self.retained_adapter_output_bytes_ = 0
        stage_seconds["adapter"] = time.perf_counter() - start

        core_kwargs = self._core_kwargs(feature_kinds, feature_cardinalities)
        start = time.perf_counter()
        learner = self._make_learner(
            random_state=int(self.random_state),
            core_kwargs=core_kwargs,
        )
        if weights is None:
            self.model_ = learner.fit(matrix, y_binary)
        else:
            self.model_ = learner.fit(matrix, y_binary, sample_weight=weights)
        stage_seconds["learner"] = time.perf_counter() - start

        start = time.perf_counter()
        self._fit_optional_calibration(matrix, y_binary, core_kwargs)
        stage_seconds["calibration"] = time.perf_counter() - start

        start = time.perf_counter()
        self.program_ = SemanticProgram(
            model=self.model_,
            adapter=self.adapter_,
            input_columns=self.input_columns_,
            adapted_feature_names=tuple(adapted_names),
            feature_indices=tuple(int(i) for i in self.adapted_feature_indices_),
            classes=tuple(self.classes_),
            library_version=self.VERSION,
        )
        self._prediction_program_ = (
            self.program_.optimize(self.replacement_objective)
            if self.prediction_backend == "optimized"
            else self.program_
        )
        if self.cache_training_statistics:
            self.training_cache_ = self._build_newton_cache_from_matrix(
                matrix,
                y_binary,
                prediction="base",
                level=int(resolved.max_bins),
                feature_limit=int(self._effective_pair_feature_limit()),
            )
        stage_seconds["program_and_cache"] = time.perf_counter() - start
        self.fit_stage_seconds_ = stage_seconds
        self.fit_seconds_ = time.perf_counter() - fit_start
        self.fit_diagnostics_ = collect_fit_diagnostics(self)
        self.sample_weighted_ = weights is not None
        self.sample_weight_sum_ = None if weights is None else float(weights.sum())
        return self

    def _check_X_schema(self, X: pd.DataFrame | np.ndarray) -> None:
        if self.input_columns_ is None:
            shape = getattr(X, "shape", None)
            if shape is None:
                shape = np.asarray(X).shape
            if len(shape) != 2:
                raise ValueError("Expected 2D array, got 1D array instead. Reshape your data")
            if int(shape[1]) != self.n_features_in_:
                raise ValueError(
                    f"X has {shape[1]} features, but CERMClassifier is expecting "
                    f"{self.n_features_in_} features as input"
                )
        else:
            validate_dataframe_schema(X, self.input_columns_)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "program_")
        self._check_X_schema(X)
        return self._prediction_program_.decision_function(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "program_")
        self._check_X_schema(X)
        return self._prediction_program_.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "program_")
        self._check_X_schema(X)
        if getattr(self, "task_type_", "binary") == "multiclass":
            return self._prediction_program_.predict(X)
        binary = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)
        return self._label_encoder_.inverse_transform(binary)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "program_")
        if input_features is not None and self.input_columns_ is not None:
            provided = tuple(input_features)
            if provided != self.input_columns_:
                raise ValueError(
                    "input_features must match feature_names_in_ in fitted order"
                )
        return self.adapted_feature_names_.copy()

    @property
    def model_bytes_estimate_(self) -> int:
        check_is_fitted(self, "program_")
        return self.program_.model_bytes_estimate

    def _build_newton_cache_from_matrix(
        self,
        matrix: np.ndarray,
        y_binary: np.ndarray,
        *,
        prediction: str | np.ndarray | None,
        level: int | None,
        feature_limit: int | None,
    ) -> NewtonHistogramCache:
        encoder = self.model_.base_.encoder_
        available_levels = tuple(int(value) for value in encoder.levels)
        if level is None:
            level = max(available_levels)
        if int(level) not in available_levels:
            raise ValueError(
                f"level must be one of {sorted(available_levels)}, got {level}"
            )
        selected_idx = np.asarray(self.model_.base_.feature_idx_, dtype=np.int64)
        states = encoder.transform_level_columns(
            matrix, int(level), selected_idx
        )
        names = tuple(
            str(self.adapted_feature_names_[index]) for index in selected_idx
        )
        if prediction == "base":
            probability = self.model_.base_.predict_proba(matrix)[:, 1]
        elif prediction == "model":
            probability = self.model_.predict_proba(matrix)[:, 1]
        elif prediction is None:
            probability = None
        elif isinstance(prediction, str):
            raise ValueError(
                "prediction must be 'base', 'model', None, or an array"
            )
        else:
            probability = np.asarray(prediction, dtype=np.float64)
        return NewtonHistogramCache(
            states,
            y_binary,
            probability,
            feature_names=names,
            feature_limit=feature_limit,
        )

    def build_newton_cache(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable,
        *,
        prediction: str | np.ndarray | None = "base",
        level: int | None = None,
        feature_limit: int | None = None,
        store: bool = True,
    ) -> NewtonHistogramCache:
        """Build reusable finite-state G/H statistics for ranking sweeps."""
        check_is_fitted(self, "program_")
        if getattr(self, "task_type_", "binary") != "binary":
            raise NotImplementedError("Newton histogram cache is available only for binary CERM heads")
        self._check_X_schema(X)
        matrix = self.program_._matrix(X)
        y_array = np.asarray(list(y) if not hasattr(y, "__len__") else y)
        if len(y_array) != len(matrix):
            raise ValueError("X and y must have the same number of rows")
        try:
            y_binary = self._label_encoder_.transform(y_array).astype(np.int32)
        except ValueError as exc:
            raise ValueError("y contains labels not seen during fit") from exc

        cache = self._build_newton_cache_from_matrix(
            matrix,
            y_binary,
            prediction=prediction,
            level=level,
            feature_limit=feature_limit,
        )
        if store:
            self.training_cache_ = cache
        return cache

    def clear_training_cache(self) -> None:
        if hasattr(self, "training_cache_"):
            del self.training_cache_

    def optimize(self, target: str = "balanced"):
        check_is_fitted(self, "program_")
        return self.program_.optimize(target)

    def compile_native(self, prefix: str | Path):
        check_is_fitted(self, "program_")
        return self.program_.compile_native(prefix)

    def compile(
        self,
        calibration_X: pd.DataFrame | np.ndarray,
        prefix: str | Path,
        *,
        target: str = "latency",
        benchmark_rows: int = 50_000,
    ):
        check_is_fitted(self, "program_")
        return self.program_.autotune(
            calibration_X,
            prefix,
            target=target,
            benchmark_rows=benchmark_rows,
        )

    def export(self, directory: str | Path) -> Path:
        check_is_fitted(self, "program_")
        return self.program_.export(directory, config=self.get_params(deep=False))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        tags.target_tags.one_d_labels = True
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "program_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CERMClassifier":
        estimator = joblib.load(path)
        if not isinstance(estimator, cls):
            raise TypeError("serialized object is not a CERMClassifier")
        return estimator
