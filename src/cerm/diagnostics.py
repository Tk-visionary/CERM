from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .params import resolve_estimator_parameters


@dataclass(frozen=True)
class FitDiagnostics:
    preset: str
    search_profile: str
    interaction_order: int
    max_bins: int
    subsample: float
    colsample: float
    n_jobs: int | None
    active_reductions: tuple[str, ...]
    search_semantics: str
    requested_pair_feature_limit: int
    requested_max_interaction_features: int
    effective_pair_feature_limit: int
    requested_max_interactions: int | None
    effective_max_interactions: int
    reg_lambda: str | float
    raw_features: int
    full_adapted_features: int
    adapted_features: int
    selection_rows: int
    selected_features: int
    pair_count: int
    fine_pair_count: int
    block_count: int
    design_dimension: int
    model_bytes_estimate: int
    selected_block_rank_mode: str
    replacement_objective: str
    replacement_ops_before: int
    replacement_ops_after: int
    calibration_method: str
    calibration_accepted: bool
    calibration_scale: float
    calibration_offset: float
    calibration_oof_logloss_improvement: float
    calibration_improvement_se: float
    calibration_signal_to_noise: float
    training_cache_bytes: int
    training_graph_max_code_columns: int
    training_graph_max_design_dimension: int
    training_graph_candidate_count: int
    training_graph_design_count: int
    training_graph_block_bank_count: int
    fit_seconds: float
    fit_stage_seconds: dict[str, float]
    estimated_peak_memory_bytes: int
    estimated_pair_evaluations: int
    estimated_block_evaluations: int
    estimated_solver_calls: int
    effective_internal_fit_units: int
    resource_risk: str
    transient_adapted_matrix_bytes: int
    retained_adapter_output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_fit_diagnostics(estimator) -> FitDiagnostics:
    model = estimator.model_
    base = model.base_
    selected = model.selected_hybrid_config_
    resolved = resolve_estimator_parameters(estimator)
    requested_alias = (
        int(estimator.pair_feature_limit)
        if estimator.max_interaction_features is None
        else int(estimator.max_interaction_features)
    )
    return FitDiagnostics(
        preset=str(resolved.preset),
        search_profile=str(resolved.search_profile),
        interaction_order=int(resolved.interaction_order),
        max_bins=int(resolved.max_bins),
        subsample=float(resolved.subsample),
        colsample=float(resolved.colsample),
        n_jobs=resolved.n_jobs,
        active_reductions=tuple(resolved.active_reductions),
        search_semantics=str(resolved.search_semantics),
        requested_pair_feature_limit=int(estimator.pair_feature_limit),
        requested_max_interaction_features=requested_alias,
        effective_pair_feature_limit=int(model.pair_feature_limit),
        requested_max_interactions=(
            None if estimator.max_interactions is None else int(estimator.max_interactions)
        ),
        effective_max_interactions=int(resolved.max_interactions),
        reg_lambda=resolved.reg_lambda,
        raw_features=int(estimator.n_features_in_),
        full_adapted_features=int(estimator.n_full_adapted_features_),
        adapted_features=int(estimator.n_adapted_features_),
        selection_rows=int(getattr(model, "selection_rows_", len(getattr(model, "classes_", ())))),
        selected_features=int(len(base.feature_idx_)),
        pair_count=int(len(base.pairs_)),
        fine_pair_count=int(len(base.fine_pairs_)),
        block_count=int(len(model.block_tables_)),
        design_dimension=int(model.design_dim_),
        model_bytes_estimate=int(estimator.model_bytes_estimate_),
        selected_block_rank_mode=str(selected.rank_mode),
        replacement_objective=str(model.replacement_objective),
        replacement_ops_before=int(model.replacement_original_ops_),
        replacement_ops_after=int(model.replacement_ops_),
        calibration_method=str(estimator.calibration_result_.method),
        calibration_accepted=bool(estimator.calibration_result_.accepted),
        calibration_scale=float(estimator.calibration_result_.scale),
        calibration_offset=float(estimator.calibration_result_.offset),
        calibration_oof_logloss_improvement=float(
            estimator.calibration_result_.improvement
        ),
        calibration_improvement_se=float(
            estimator.calibration_result_.improvement_se
        ),
        calibration_signal_to_noise=float(
            estimator.calibration_result_.signal_to_noise
        ),
        training_cache_bytes=int(
            getattr(getattr(estimator, "training_cache_", None), "nbytes", 0)
        ),
        training_graph_max_code_columns=int(
            getattr(base, "training_graph_columns_", 0)
        ),
        training_graph_max_design_dimension=int(
            getattr(base, "training_graph_design_dim_", 0)
        ),
        training_graph_candidate_count=int(
            getattr(model, "primary_candidate_count_", 0)
        ),
        training_graph_design_count=int(
            getattr(model, "primary_design_count_", 0)
        ),
        training_graph_block_bank_count=int(
            getattr(model, "primary_column_bank_count_", 0)
        ),
        fit_seconds=float(getattr(estimator, "fit_seconds_", 0.0)),
        fit_stage_seconds=dict(getattr(estimator, "fit_stage_seconds_", {})),
        estimated_peak_memory_bytes=int(
            getattr(getattr(estimator, "fit_resource_plan_", None), "estimated_peak_memory_bytes", 0)
        ),
        estimated_pair_evaluations=int(
            getattr(getattr(estimator, "fit_resource_plan_", None), "estimated_pair_evaluations", 0)
        ),
        estimated_block_evaluations=int(
            getattr(getattr(estimator, "fit_resource_plan_", None), "estimated_block_evaluations", 0)
        ),
        estimated_solver_calls=int(
            getattr(getattr(estimator, "fit_resource_plan_", None), "estimated_solver_calls", 0)
        ),
        effective_internal_fit_units=int(
            getattr(getattr(estimator, "fit_resource_plan_", None), "effective_internal_fit_units", 0)
        ),
        resource_risk=str(
            getattr(getattr(estimator, "fit_resource_plan_", None), "risk", "unknown")
        ),
        transient_adapted_matrix_bytes=int(
            getattr(estimator, "transient_adapted_matrix_bytes_", 0)
        ),
        retained_adapter_output_bytes=int(
            getattr(estimator, "retained_adapter_output_bytes_", 0)
        ),
    )
