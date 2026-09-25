"""Check that public estimator signatures, documentation, aliases, and release metadata agree."""

from __future__ import annotations

import inspect
from pathlib import Path
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - CI runs this check on modern Python
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cerm import (  # noqa: E402
    CERMClassifier,
    CERMFusedRegressor,
    CERMGeneralizedRegressor,
    CERMRegressor,
    CERMRidgeRegressor,
)
from cerm._version import __version__  # noqa: E402


ESTIMATORS = {
    "CERMClassifier": CERMClassifier,
    "CERMRegressor": CERMRegressor,
    "CERMFusedRegressor": CERMFusedRegressor,
    "CERMRidgeRegressor": CERMRidgeRegressor,
    "CERMGeneralizedRegressor": CERMGeneralizedRegressor,
}

ALIAS_CASES = {
    "search_effort": ("preset", "balanced", "balanced"),
    "state_detail": ("max_bins", "medium", 8),
    "feature_budget": ("max_features", 7, 7),
    "interaction_search_features": ("max_interaction_features", 5, 5),
    "interaction_budget": ("max_interactions", 3, 3),
    "selection_fraction": ("subsample", 0.75, 0.75),
    "feature_fraction": ("colsample", 0.75, 0.75),
    "l2_regularization": ("reg_lambda", 2.0, 2.0),
    "memory_limit_mb": ("max_memory_mb", 512.0, 512.0),
}

REQUIRED_PROJECT_URLS = {"Homepage", "Documentation", "Repository", "Issues", "Changelog"}


def fail(message: str) -> None:
    raise SystemExit(f"public API check failed: {message}")


def constructor_parameters(estimator) -> set[str]:
    return {
        name
        for name, parameter in inspect.signature(estimator.__init__).parameters.items()
        if name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


def inventory_parameters(text: str, estimator_name: str) -> set[str]:
    pattern = re.compile(
        rf"<!-- CERM-PARAMETERS:{re.escape(estimator_name)}:START -->(.*?)"
        rf"<!-- CERM-PARAMETERS:{re.escape(estimator_name)}:END -->",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        fail(f"parameter inventory block missing for {estimator_name}")
    return set(re.findall(r"`([A-Za-z][A-Za-z0-9_]*)`", match.group(1)))


def documented_aliases(text: str) -> dict[str, str]:
    pattern = re.compile(
        r"<!-- CERM-ALIASES:START -->(.*?)<!-- CERM-ALIASES:END -->",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        fail("semantic alias inventory block missing")
    pairs = re.findall(
        r"\|\s*`([A-Za-z][A-Za-z0-9_]*)`\s*\|\s*`([A-Za-z][A-Za-z0-9_]*)`\s*\|",
        match.group(1),
    )
    return dict(pairs)


def check_parameter_inventory() -> None:
    text = (ROOT / "docs" / "parameter_inventory.md").read_text(encoding="utf-8")
    for name, estimator in ESTIMATORS.items():
        actual = constructor_parameters(estimator)
        documented = inventory_parameters(text, name)
        missing = sorted(actual - documented)
        stale = sorted(documented - actual)
        if missing or stale:
            fail(f"{name} inventory mismatch; missing={missing}, stale={stale}")

    documented = documented_aliases(text)
    expected = {semantic: legacy for semantic, (legacy, _, _) in ALIAS_CASES.items()}
    if documented != expected:
        fail(f"semantic alias inventory mismatch; documented={documented}, expected={expected}")

    for estimator_name, estimator in ESTIMATORS.items():
        parameters = constructor_parameters(estimator)
        for semantic, (legacy, sample, expected_value) in ALIAS_CASES.items():
            if semantic not in parameters:
                continue
            if legacy not in parameters:
                fail(f"{estimator_name} exposes {semantic} without historical {legacy}")
            instance = estimator(**{semantic: sample})
            actual_value = getattr(instance, legacy)
            if actual_value != expected_value:
                fail(
                    f"{estimator_name} alias {semantic}->{legacy} resolves to "
                    f"{actual_value!r}, expected {expected_value!r}"
                )


def check_release_metadata() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    urls = pyproject.get("project", {}).get("urls", {})
    missing_urls = sorted(REQUIRED_PROJECT_URLS - set(urls))
    if missing_urls:
        fail(f"pyproject.toml [project.urls] missing {missing_urls}")
    if not all(str(value).startswith("https://") for value in urls.values()):
        fail("all [project.urls] values must be HTTPS URLs")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"(?m)^#+\s+{re.escape(__version__)}\s*$", changelog) is None:
        fail(f"CHANGELOG.md has no heading for current version {__version__}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for field in ("repository-code", "url"):
        if re.search(rf"(?m)^{re.escape(field)}\s*:", citation) is None:
            fail(f"CITATION.cff is missing {field}")


def main() -> None:
    check_parameter_inventory()
    check_release_metadata()
    print(
        "public API checks passed: "
        f"{len(ESTIMATORS)} estimator signatures, {len(ALIAS_CASES)} aliases, release metadata"
    )


if __name__ == "__main__":
    main()
