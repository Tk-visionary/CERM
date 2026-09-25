from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TypedAdapterConfig:
    categorical_features: tuple[str, ...] | str | None = "auto"
    embedding_features: tuple[str, ...] | None = None
    category_policy: str = "auto"
    max_identity_categories: int = 16
    category_bins: int = 8
    category_smoothing: float = 20.0
    category_newton_l2: float = 10.0
    category_identity: str = "state"
    embedding_mode: str = "prototype"
    embedding_pca: int = 8
    embedding_bins: int = 8
    embedding_prototypes: int = 64
    retain_embedding_raw: bool = False
    missing_policy: str = "observed"


@dataclass(frozen=True)
class CERMConfig:
    max_features: int = 64
    max_bins: int = 16
    subsample: float = 1.0
    colsample: float = 1.0
    n_jobs: int | None = 1
    search_effort: str | None = None
    state_detail: str | None = None
    feature_budget: int | None = None
    interaction_search_features: int | None = None
    interaction_budget: int | None = None
    selection_fraction: float | None = None
    feature_fraction: float | None = None
    l2_regularization: str | float | None = None
    memory_limit_mb: float | None = None
    preset: str = "accurate"
    max_interaction_features: int | None = None
    max_interactions: int | None = None
    interaction_order: int = 2
    reg_lambda: str | float = "auto"
    max_memory_mb: float | None = None
    pair_feature_limit: int = 24
    search_profile: str | None = None
    replacement_objective: str = "balanced"
    prediction_backend: str = "optimized"
    random_state: int = 20260803
    encoder_kind: str = "quantile"
    newton_prebins: int = 64
    newton_gain_l2: float = 5.0
    newton_min_hessian: float = 1.0
    ranking_kind: str = "mi"
    ranking_l2: float = 5.0
    ranking_prefilter_multiplier: int = 4
    cost_per_byte: float = 0.0
    cost_per_operator: float = 0.0
    block_cost_per_byte: float = 0.0
    block_cost_per_eval: float = 0.0
    selection_strategy: str = "two_holdout"
    selection_folds: int = 3
    selection_near_tie: float = 0.0015
    selection_min_improvement: float = 0.0010
    calibration: str = "none"
    calibration_folds: int = 3
    calibration_l2: float = 1e-3
    calibration_min_improvement: float = 1e-4
    calibration_min_signal: float = 1.0
    cache_training_statistics: bool = False
    resource_policy: str = "raise"
    max_estimated_peak_memory_mb: float | None = 4096.0
    max_pair_evaluations: int | None = 2_000_000
    max_block_evaluations: int | None = 5_000_000
    max_knn_distance_evaluations: int | None = 100_000_000
    multiclass_strategy: str = "ovr"
    shared_multiclass_objective: str = "ovr"
    representation_strategy: str = "baseline"
    class_specific_budget: int = 12
    adapter: TypedAdapterConfig = field(default_factory=TypedAdapterConfig)
