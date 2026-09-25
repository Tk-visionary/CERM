"""Validate only CERM's declared runtime dependencies in the current environment."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    failures: list[str] = []
    for raw in project["dependencies"]:
        requirement = Requirement(raw)
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            failures.append(f"{requirement.name}: not installed")
            continue
        if installed not in requirement.specifier:
            failures.append(
                f"{requirement.name}: installed {installed}, required {requirement.specifier}"
            )
        else:
            print(f"{requirement.name} {installed}: OK")
    if failures:
        raise SystemExit("CERM dependency check failed:\n- " + "\n- ".join(failures))
    print("CERM runtime dependency check passed")


if __name__ == "__main__":
    main()
