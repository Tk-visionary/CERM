from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CERMValidationError(ValueError):
    """Base class for user-facing CERM validation failures."""


class CERMDataSchemaError(CERMValidationError):
    """Raised when prediction input does not match the fitted table schema."""


class CERMParameterError(CERMValidationError):
    """Raised for invalid public CERM parameter values or combinations."""


class CERMAliasConflictError(CERMParameterError):
    """Raised when semantic and historical parameter names disagree."""


class CERMResourceWarning(UserWarning):
    """Warning emitted when a fit plan exceeds a configured resource budget."""


class CERMResourceLimitError(RuntimeError):
    """Raised before fit when a configured resource budget would be exceeded."""


def format_schema_mismatch(
    missing: Sequence[Any],
    extra: Sequence[Any],
) -> str:
    """Return the canonical DataFrame schema-mismatch message."""

    return (
        f"DataFrame schema mismatch; missing={list(missing)}, extra={list(extra)}. "
        "Pass the same columns used during fit; column order may differ."
    )


def format_alias_conflict(
    *,
    semantic_name: str,
    semantic_value: Any,
    legacy_name: str,
    legacy_value: Any,
) -> str:
    """Return the canonical semantic/historical alias conflict message."""

    return (
        f"{semantic_name}={semantic_value!r} conflicts with "
        f"{legacy_name}={legacy_value!r}; specify only one naming style. "
        f"Prefer the user-facing parameter {semantic_name!r} in new code."
    )


def format_resource_limit(violations: Sequence[str]) -> str:
    """Return the canonical resource-budget message with a recovery hint."""

    return (
        "CERM fit resource budget exceeded: "
        + "; ".join(violations)
        + ". Call estimate_fit_resources(X) to inspect the plan before fitting, "
        "or change the explicit budget/search controls after task-level validation."
    )
