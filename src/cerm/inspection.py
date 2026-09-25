from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _diagnostics_dict(estimator: Any) -> dict[str, Any]:
    diagnostics = getattr(estimator, "fit_diagnostics_", None)
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "to_dict"):
        return dict(diagnostics.to_dict())
    if isinstance(diagnostics, dict):
        return dict(diagnostics)
    return {}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024.0 or unit == "GiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GiB"


def _structure_model(estimator: Any) -> Any | None:
    structure = getattr(estimator, "representation_model_", None)
    if structure is not None:
        return structure
    structure = getattr(estimator, "model_", None)
    if structure is None:
        return None
    base = getattr(structure, "base_", None)
    return base if base is not None else structure


def _selected_feature_names(estimator: Any, structure: Any) -> list[str]:
    names = getattr(estimator, "adapted_feature_names_", None)
    indices = getattr(structure, "feature_idx_", None)
    if names is None or indices is None:
        return []
    names_array = np.asarray(names, dtype=object)
    selected = np.asarray(indices, dtype=np.int64)
    if np.any((selected < 0) | (selected >= len(names_array))):
        return []
    return [str(names_array[int(index)]) for index in selected]


def _block_specs(estimator: Any) -> list[dict[str, Any]]:
    model = getattr(estimator, "model_", None)
    specs = getattr(model, "block_specs_", None)
    if specs is None:
        return []
    return [dict(spec) for spec in specs]


@dataclass(frozen=True)
class CERMModelSummary:
    """Compact, task-neutral summary of a CERM estimator."""

    estimator: str
    fitted: bool
    task: str | None
    n_features_in: int | None
    n_adapted_features: int | None
    selected_features: int | None
    pair_count: int | None
    block_count: int | None
    fit_seconds: float | None
    model_bytes: int | None
    resource_risk: str | None
    search_effort: Any
    state_detail: Any
    interaction_order: int | None
    active_reductions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        status = "fitted" if self.fitted else "unfitted"
        task = "" if self.task is None else f", {self.task}"
        lines = [f"{self.estimator} ({status}{task})"]
        lines.append(
            "  configuration: "
            f"search_effort={self.search_effort!r}, "
            f"state_detail={self.state_detail!r}, "
            f"interaction_order={self.interaction_order!r}"
        )
        if not self.fitted:
            return "\n".join(lines)
        lines.append(
            "  features: "
            f"input={self.n_features_in}, adapted={self.n_adapted_features}, "
            f"selected={self.selected_features}"
        )
        lines.append(
            "  interactions: "
            f"pairs={self.pair_count}, blocks={self.block_count}"
        )
        fit_text = "unknown" if self.fit_seconds is None else f"{self.fit_seconds:.3f} s"
        lines.append(
            "  resources: "
            f"fit={fit_text}, model={_format_bytes(self.model_bytes)}, "
            f"risk={self.resource_risk or 'unknown'}"
        )
        if self.active_reductions:
            lines.append("  active reductions: " + ", ".join(self.active_reductions))
        return "\n".join(lines)


@dataclass(frozen=True)
class CERMModelInspection:
    """Notebook-friendly fitted-model inspection bundle."""

    summary: CERMModelSummary
    features: pd.DataFrame
    interactions: pd.DataFrame
    structure: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "features": self.features.to_dict(orient="records"),
            "interactions": self.interactions.to_dict(orient="records"),
            "structure": self.structure.to_dict(orient="records"),
        }

    def __repr__(self) -> str:
        return (
            f"CERMModelInspection({self.summary.estimator}, "
            f"features={len(self.features)}, interactions={len(self.interactions)})"
        )

    def _repr_html_(self) -> str:
        heading = f"<pre>{str(self.summary)}</pre>"
        features = self.features.to_html(index=False, escape=True)
        structure = self.structure.to_html(index=False, escape=True)
        return (
            heading
            + "<h4>Selected features</h4>"
            + features
            + "<h4>Model structure</h4>"
            + structure
        )


def model_summary(estimator: Any) -> CERMModelSummary:
    """Return a compact summary for a fitted or unfitted CERM estimator."""

    user_params: dict[str, Any] = {}
    getter = getattr(estimator, "get_user_params", None)
    if callable(getter):
        user_params = dict(getter())

    fitted = hasattr(estimator, "n_features_in_")
    diagnostics = _diagnostics_dict(estimator) if fitted else {}
    task = getattr(estimator, "task_type_", None) if fitted else None
    if task is None:
        task = diagnostics.get("task_type")

    selected_features = diagnostics.get("selected_features")
    pair_count = diagnostics.get("pair_count")
    block_count = diagnostics.get("block_count")
    fit_seconds = diagnostics.get("fit_seconds", getattr(estimator, "fit_seconds_", None))
    model_bytes = diagnostics.get(
        "model_bytes_estimate", getattr(estimator, "model_bytes_estimate_", None)
    )
    resource_risk = diagnostics.get("resource_risk")
    active_reductions = tuple(diagnostics.get("active_reductions", ()))

    return CERMModelSummary(
        estimator=type(estimator).__name__,
        fitted=bool(fitted),
        task=None if task is None else str(task),
        n_features_in=_optional_int(getattr(estimator, "n_features_in_", None)),
        n_adapted_features=_optional_int(getattr(estimator, "n_adapted_features_", None)),
        selected_features=_optional_int(selected_features),
        pair_count=_optional_int(pair_count),
        block_count=_optional_int(block_count),
        fit_seconds=_optional_float(fit_seconds),
        model_bytes=_optional_int(model_bytes),
        resource_risk=None if resource_risk is None else str(resource_risk),
        search_effort=user_params.get("search_effort"),
        state_detail=user_params.get("state_detail"),
        interaction_order=_optional_int(user_params.get("interaction_order")),
        active_reductions=active_reductions,
    )


def feature_table(estimator: Any) -> pd.DataFrame:
    """Return selected finite-state features as a small notebook table."""

    structure = _structure_model(estimator)
    columns = ["rank", "adapted_index", "feature_name", "used_in_pair", "used_in_block"]
    if structure is None:
        return pd.DataFrame(columns=columns)
    indices = getattr(structure, "feature_idx_", None)
    names = _selected_feature_names(estimator, structure)
    if indices is None or not names:
        return pd.DataFrame(columns=columns)

    pairs = [tuple(map(int, pair)) for pair in getattr(structure, "pairs_", ())]
    pair_positions = {index for pair in pairs for index in pair}
    blocks = _block_specs(estimator)
    block_positions = {
        int(value)
        for spec in blocks
        for key in ("gate_j", "target_k")
        if (value := spec.get(key)) is not None
    }
    rows = []
    for rank, (adapted_index, name) in enumerate(zip(indices, names)):
        rows.append(
            {
                "rank": int(rank),
                "adapted_index": int(adapted_index),
                "feature_name": name,
                "used_in_pair": rank in pair_positions,
                "used_in_block": rank in block_positions,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def interaction_table(estimator: Any) -> pd.DataFrame:
    """Return selected pair/fine-pair/block interactions in feature-name form."""

    columns = [
        "kind",
        "left_rank",
        "left_feature",
        "right_rank",
        "right_feature",
        "detail",
    ]
    structure = _structure_model(estimator)
    if structure is None:
        return pd.DataFrame(columns=columns)
    names = _selected_feature_names(estimator, structure)
    if not names:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    def append_pair(kind: str, left: Any, right: Any, detail: str = "") -> None:
        left_i = int(left)
        right_i = int(right)
        if not (0 <= left_i < len(names) and 0 <= right_i < len(names)):
            return
        rows.append(
            {
                "kind": kind,
                "left_rank": left_i,
                "left_feature": names[left_i],
                "right_rank": right_i,
                "right_feature": names[right_i],
                "detail": detail,
            }
        )

    for left, right in getattr(structure, "pairs_", ()):
        append_pair("pair", left, right)
    for left, right in getattr(structure, "fine_pairs_", ()):
        append_pair("fine_pair", left, right)
    for spec in _block_specs(estimator):
        if "gate_j" in spec and "target_k" in spec:
            append_pair(
                "block",
                spec["gate_j"],
                spec["target_k"],
                f"gate_state={spec.get('gate_state')}",
            )
    return pd.DataFrame(rows, columns=columns)


def structure_table(estimator: Any) -> pd.DataFrame:
    """Return main and interaction terms in one compact structure table."""

    columns = ["kind", "term", "left_feature", "right_feature", "detail"]
    rows: list[dict[str, Any]] = []
    for record in feature_table(estimator).to_dict(orient="records"):
        name = str(record["feature_name"])
        rows.append(
            {
                "kind": "main",
                "term": name,
                "left_feature": name,
                "right_feature": None,
                "detail": f"selected_rank={int(record['rank'])}",
            }
        )
    for record in interaction_table(estimator).to_dict(orient="records"):
        left = str(record["left_feature"])
        right = str(record["right_feature"])
        operator = " -> " if record["kind"] == "block" else " × "
        rows.append(
            {
                "kind": record["kind"],
                "term": f"{left}{operator}{right}",
                "left_feature": left,
                "right_feature": right,
                "detail": record["detail"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def inspect_model(estimator: Any) -> CERMModelInspection:
    """Return compact summary plus selected feature/interaction/structure tables."""

    return CERMModelInspection(
        summary=model_summary(estimator),
        features=feature_table(estimator),
        interactions=interaction_table(estimator),
        structure=structure_table(estimator),
    )
