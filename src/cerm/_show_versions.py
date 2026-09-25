"""Environment reporting helpers for reproducible bug reports."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import platform
import sys

from ._version import __version__


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - required runtime deps are installed
        return "not installed"


def show_versions() -> dict[str, str]:
    """Return CERM, Python, platform, and runtime dependency versions."""

    return {
        "cerm": __version__,
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "pandas": _distribution_version("pandas"),
        "scikit-learn": _distribution_version("scikit-learn"),
        "joblib": _distribution_version("joblib"),
    }
