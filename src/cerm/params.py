from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .errors import CERMAliasConflictError, CERMParameterError, format_alias_conflict


_PRESET_TO_PROFILE = {
    "accurate": "full_exact",
    "balanced": "practical",
}
_SEARCH_EFFORT_TO_PRESET = {
    "thorough": "accurate",
    "balanced": "balanced",
}
_STATE_DETAIL_TO_BINS = {
    "coarse": 4,
    "medium": 8,
    "fine": 16,
}
_BINS_TO_STATE_DETAIL = {value: key for key, value in _STATE_DETAIL_TO_BINS.items()}
_PRESET_TO_SEARCH_EFFORT = {value: key for key, value in _SEARCH_EFFORT_TO_PRESET.items()}


def _is_fused_regression_surface(estimator: Any) -> bool:
    return getattr(estimator, "_parameter_surface_kind", None) == "fused_regression"


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return float(left) == float(right)
        except (TypeError, ValueError):
            return False
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def resolve_semantic_alias(
    *,
    semantic_name: str,
    semantic_value: Any,
    legacy_name: str,
    legacy_value: Any,
    legacy_default: Any,
    mapping: Mapping[Any, Any] | None = None,
    strict: bool = True,
) -> Any:
    """Resolve one user-facing alias while rejecting ambiguous specifications."""

    if semantic_value is None:
        return legacy_value
    if mapping is None:
        resolved = semantic_value
    else:
        try:
            valid = semantic_value in mapping
        except TypeError:
            valid = False
        if not valid:
            options = ", ".join(map(str, mapping))
            if strict:
                raise CERMParameterError(
                    f"{semantic_name} must be one of: {options}. "
                    f"Use {semantic_name!r} in new code or the historical "
                    f"{legacy_name!r}, but do not mix conflicting naming styles."
                )
            return legacy_value
        resolved = mapping[semantic_value]
    if not _same_value(legacy_value, legacy_default) and not _same_value(
        legacy_value, resolved
    ):
        message = format_alias_conflict(
            semantic_name=semantic_name,
            semantic_value=semantic_value,
            legacy_name=legacy_name,
            legacy_value=legacy_value,
        )
        if strict:
            raise CERMAliasConflictError(message)
    return resolved


@dataclass(frozen=True)
class ResolvedCERMParams:
    """Concrete training parameters after resolving presets and aliases."""

    preset: str
    search_profile: str
    max_features: int
    max_bins: int
    subsample: float
    colsample: float
    n_jobs: int | None
    max_interaction_features: int
    max_interactions: int
    interaction_order: int
    reg_lambda: str | float
    fixed_C: float | None
    max_memory_mb: float | None
    active_reductions: tuple[str, ...]
    search_semantics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_estimator_parameters(estimator: Any) -> ResolvedCERMParams:
    preset = str(getattr(estimator, "preset", "accurate"))
    if preset not in _PRESET_TO_PROFILE:
        options = ", ".join(sorted(_PRESET_TO_PROFILE))
        raise CERMParameterError(f"preset must be one of: {options}")

    legacy_profile = getattr(estimator, "search_profile", None)
    search_profile = (
        _PRESET_TO_PROFILE[preset]
        if legacy_profile is None
        else str(legacy_profile)
    )
    resolved_preset = preset if legacy_profile is None else "custom"

    alias_limit = getattr(estimator, "max_interaction_features", None)
    if alias_limit is None:
        requested_interaction_features = int(estimator.pair_feature_limit)
    else:
        requested_interaction_features = int(alias_limit)

    interaction_order = int(getattr(estimator, "interaction_order", 2))
    if interaction_order == 1:
        effective_interaction_features = 1
        effective_max_interactions = 0
    else:
        effective_interaction_features = requested_interaction_features
        if search_profile == "aggressive":
            effective_interaction_features = min(effective_interaction_features, 20)
        requested_max_interactions = getattr(estimator, "max_interactions", None)
        effective_max_interactions = (
            32 if requested_max_interactions is None else int(requested_max_interactions)
        )

    reg_lambda = getattr(estimator, "reg_lambda", "auto")
    if reg_lambda == "auto":
        fixed_C = None
    else:
        value = float(reg_lambda)
        if not np.isfinite(value) or value <= 0:
            raise CERMParameterError(
                "reg_lambda must be 'auto' or a finite positive number"
            )
        fixed_C = 1.0 / value
        reg_lambda = value

    max_memory_alias = getattr(estimator, "max_memory_mb", None)
    legacy_memory = getattr(estimator, "max_estimated_peak_memory_mb", None)
    max_memory_mb = legacy_memory if max_memory_alias is None else max_memory_alias
    if max_memory_mb is not None:
        max_memory_mb = float(max_memory_mb)

    active_reductions = []
    if int(estimator.max_bins) < 16:
        active_reductions.append("reduced_state_resolution")
    if float(estimator.subsample) < 1.0:
        active_reductions.append("selection_row_subsample")
    if float(estimator.colsample) < 1.0:
        active_reductions.append("feature_subspace")
    if search_profile != "full_exact":
        active_reductions.append("reduced_candidate_family")
    if interaction_order < 2:
        active_reductions.append("main_effects_only")
    if effective_max_interactions < 32:
        active_reductions.append("interaction_cap")
    if fixed_C is not None:
        active_reductions.append("fixed_regularization")
    active_reductions = tuple(active_reductions)
    search_semantics = "full" if not active_reductions else "+".join(active_reductions)

    return ResolvedCERMParams(
        preset=resolved_preset,
        search_profile=search_profile,
        max_features=int(estimator.max_features),
        max_bins=int(estimator.max_bins),
        subsample=float(estimator.subsample),
        colsample=float(estimator.colsample),
        n_jobs=None if estimator.n_jobs is None else int(estimator.n_jobs),
        max_interaction_features=int(effective_interaction_features),
        max_interactions=int(effective_max_interactions),
        interaction_order=interaction_order,
        reg_lambda=reg_lambda,
        fixed_C=fixed_C,
        max_memory_mb=max_memory_mb,
        active_reductions=active_reductions,
        search_semantics=search_semantics,
    )


def _default_interaction_budget(estimator: Any, *, task: str) -> int:
    value = getattr(estimator, "max_interactions", None)
    if value is not None:
        return int(value)
    if task == "classifier":
        return 32
    return 24 if getattr(estimator, "preset", "accurate") == "accurate" else 12


def semantic_parameter_values(estimator: Any, *, task: str) -> dict[str, Any]:
    """Return the compact, recommended view of an estimator configuration."""

    if _is_fused_regression_surface(estimator):
        max_bins = int(estimator.max_bins)
        return {
            "state_detail": _BINS_TO_STATE_DETAIL.get(max_bins, max_bins),
            "feature_budget": int(estimator.max_features),
            "interaction_search_features": int(estimator.max_interaction_features),
            "interaction_budget": int(estimator.max_pairs),
            "survival_bins": int(estimator.n_bins),
        }

    preset = str(getattr(estimator, "preset", "accurate"))
    search_profile = getattr(estimator, "search_profile", None)
    if search_profile is None:
        search_effort = _PRESET_TO_SEARCH_EFFORT.get(preset, "custom")
    else:
        search_effort = "custom"
    max_bins = int(getattr(estimator, "max_bins", 16))
    interaction_features = getattr(estimator, "max_interaction_features", None)
    if interaction_features is None:
        interaction_features = getattr(estimator, "pair_feature_limit", 24)
    reg_lambda = getattr(estimator, "reg_lambda", "auto")
    values = {
        "search_effort": search_effort,
        "state_detail": _BINS_TO_STATE_DETAIL.get(max_bins, max_bins),
        "feature_budget": int(getattr(estimator, "max_features", 64)),
        "interaction_order": int(getattr(estimator, "interaction_order", 2)),
        "interaction_search_features": int(interaction_features),
        "interaction_budget": _default_interaction_budget(estimator, task=task),
        "selection_fraction": float(getattr(estimator, "subsample", 1.0)),
        "feature_fraction": float(getattr(estimator, "colsample", 1.0)),
        "l2_regularization": reg_lambda,
        "n_jobs": getattr(estimator, "n_jobs", 1),
    }
    if hasattr(estimator, "max_memory_mb") or hasattr(
        estimator, "max_estimated_peak_memory_mb"
    ):
        memory = getattr(estimator, "max_memory_mb", None)
        if memory is None:
            memory = getattr(estimator, "max_estimated_peak_memory_mb", None)
        values["memory_limit_mb"] = memory
    if hasattr(estimator, "representation_mode"):
        values["representation_mode"] = estimator.representation_mode
    if hasattr(estimator, "loss"):
        values["loss"] = estimator.loss
    return values


def tunable_parameter_values(estimator: Any, *, task: str) -> dict[str, Any]:
    """Return the HPO-oriented model parameters without convenience aliases.

    These values directly control the fitted model or representation space and
    are the preferred surface for GridSearchCV/Optuna-style tuning. Presets,
    human-language aliases, and execution resource limits are deliberately
    excluded.
    """
    if _is_fused_regression_surface(estimator):
        return {
            "n_bins": int(estimator.n_bins),
            "max_bins": int(estimator.max_bins),
            "max_features": int(estimator.max_features),
            "max_interaction_features": int(estimator.max_interaction_features),
            "max_pairs": int(estimator.max_pairs),
        }

    interaction_features = getattr(estimator, "max_interaction_features", None)
    if interaction_features is None:
        interaction_features = getattr(estimator, "pair_feature_limit", 24)
    values = {
        "max_bins": int(getattr(estimator, "max_bins", 16)),
        "max_features": int(getattr(estimator, "max_features", 64)),
        "max_interaction_features": int(interaction_features),
        "max_interactions": _default_interaction_budget(estimator, task=task),
        "interaction_order": int(getattr(estimator, "interaction_order", 2)),
        "reg_lambda": getattr(estimator, "reg_lambda", "auto"),
        "subsample": float(getattr(estimator, "subsample", 1.0)),
        "colsample": float(getattr(estimator, "colsample", 1.0)),
    }
    if hasattr(estimator, "head_alpha"):
        values["head_alpha"] = float(estimator.head_alpha)
    return values


def search_parameter_values(estimator: Any) -> dict[str, Any]:
    """Return algorithm/search-family controls, separate from model HPO."""
    if _is_fused_regression_surface(estimator):
        # The public estimator fixes the V2 search family. Engine selection is
        # deliberately not represented as a numeric/categorical HPO knob.
        return {}
    values = {
        "preset": getattr(estimator, "preset", "accurate"),
        "search_profile": getattr(estimator, "search_profile", None),
    }
    for name in (
        "ranking_kind",
        "selection_strategy",
        "multiclass_strategy",
        "shared_multiclass_objective",
    ):
        if hasattr(estimator, name):
            values[name] = getattr(estimator, name)
    return values


def convenience_parameter_values(estimator: Any, *, task: str) -> dict[str, Any]:
    """Return human-language compatibility aliases. Not the preferred HPO surface."""
    return semantic_parameter_values(estimator, task=task)


def resource_parameter_values(estimator: Any) -> dict[str, Any]:
    """Return execution/resource controls that should not be tuned for quality."""
    if _is_fused_regression_surface(estimator):
        return {}
    values = {"n_jobs": getattr(estimator, "n_jobs", 1)}
    memory = getattr(estimator, "max_memory_mb", None)
    if memory is None:
        memory = getattr(estimator, "max_estimated_peak_memory_mb", None)
    if hasattr(estimator, "max_memory_mb") or hasattr(
        estimator, "max_estimated_peak_memory_mb"
    ):
        values["max_memory_mb"] = memory
    for name in (
        "resource_policy",
        "max_pair_evaluations",
        "max_block_evaluations",
        "max_knn_distance_evaluations",
    ):
        if hasattr(estimator, name):
            values[name] = getattr(estimator, name)
    return values


def _parameter_task(estimator: Any) -> str:
    name = type(estimator).__name__.lower()
    return "classifier" if "classifier" in name else "regressor"


def _prepare_parameter_view(estimator: Any, *, require_model_surface: bool = True) -> None:
    if require_model_surface and not (
        hasattr(estimator, "max_bins") and hasattr(estimator, "max_features")
    ):
        raise CERMParameterError(
            "parameter-layer model helpers require CERMClassifier, CERMRegressor, "
            "CERMFusedRegressor, or CERMGeneralizedRegressor; for multi-output "
            "wrappers inspect/tune their base estimator instead"
        )
    refresh = getattr(estimator, "_refresh_semantic_aliases", None)
    if callable(refresh):
        refresh()
    conflicts = getattr(estimator, "_semantic_conflicts", ())
    if conflicts:
        raise CERMParameterError("; ".join(map(str, conflicts)))


def get_tunable_params(estimator: Any) -> dict[str, Any]:
    """Return direct model parameters recommended for HPO."""
    _prepare_parameter_view(estimator)
    return tunable_parameter_values(estimator, task=_parameter_task(estimator))


def get_search_params(estimator: Any) -> dict[str, Any]:
    """Return search-family/categorical algorithm controls."""
    _prepare_parameter_view(estimator)
    return search_parameter_values(estimator)


def get_resource_params(estimator: Any) -> dict[str, Any]:
    """Return execution/resource controls, separate from predictive HPO."""
    _prepare_parameter_view(estimator, require_model_surface=False)
    return resource_parameter_values(estimator)


def get_convenience_params(estimator: Any) -> dict[str, Any]:
    """Return human-language compatibility aliases."""
    _prepare_parameter_view(estimator)
    return convenience_parameter_values(estimator, task=_parameter_task(estimator))


_PARAMETER_EXPLANATIONS = {
    "search_effort": "How broadly CERM compares candidate representations.",
    "state_detail": "How finely continuous features are divided into finite states.",
    "feature_budget": "Maximum number of features retained for main effects.",
    "interaction_order": "1 uses main effects only; 2 also allows two-feature interactions.",
    "interaction_search_features": "Number of strongest features allowed into interaction search.",
    "interaction_budget": "Maximum number of interaction terms retained by the task learner.",
    "survival_bins": "Number of residual-survival threshold bins used by fused regression.",
    "selection_fraction": "Fraction of training rows used to choose the representation; final fitting uses all rows.",
    "feature_fraction": "Fraction of adapted input columns retained in the fitted model.",
    "l2_regularization": "L2 shrinkage; 'auto' compares the validated candidates.",
    "memory_limit_mb": "Pre-fit structural memory budget in megabytes.",
    "n_jobs": "Parallel worker count used by supported selection and adapter operations.",
    "representation_mode": "Whether generalized regression searches CERM states ('auto') or directly fits a linear representation.",
    "loss": "Prediction objective used by the generalized regression head.",
}


def explain_semantic_parameters(
    estimator: Any, *, task: str
) -> dict[str, dict[str, Any]]:
    values = semantic_parameter_values(estimator, task=task)
    return {
        name: {"value": value, "meaning": _PARAMETER_EXPLANATIONS[name]}
        for name, value in values.items()
    }


def format_semantic_parameter_summary(estimator: Any, *, task: str) -> str:
    explained = explain_semantic_parameters(estimator, task=task)
    lines = ["CERM parameter summary"]
    for name, payload in explained.items():
        lines.append(f"- {name}={payload['value']!r}: {payload['meaning']}")
    return "\n".join(lines)