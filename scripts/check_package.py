"""Repository and distribution hygiene checks for CERM."""

from __future__ import annotations

import ast
import py_compile
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cerm"
REQUIRED = (
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "SECURITY.md",
    "SUPPORT.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "THIRD_PARTY_NOTICES.md",
    "AI_ASSISTANCE.md",
    "PROVENANCE.md",
    "CITATION.cff",
    "pyproject.toml",
)
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
}
FORBIDDEN_NAMES = {".coverage", ".DS_Store"}
REQUIRED_PROJECT_URLS = {"Homepage", "Documentation", "Repository", "Issues", "Changelog"}


def fail(message: str) -> None:
    raise SystemExit(f"package check failed: {message}")


def read_version() -> str:
    module = ast.parse((SRC / "_version.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    fail("could not read __version__ from src/cerm/_version.py")
    raise AssertionError


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    if project.get("name") != "CERM":
        fail("unexpected project name")
    if project.get("dynamic") != ["version"]:
        fail("version must be dynamically sourced")
    if project.get("license") != "Apache-2.0":
        fail("unexpected license expression")
    authors = project.get("authors", [])
    if not any(author.get("name") == "Taishi Kawahara" for author in authors):
        fail("Taishi Kawahara must be listed as an author")
    if (SRC / "_vendor").exists() or not (SRC / "_internal" / "__init__.py").is_file():
        fail("historical _vendor package must be replaced by _internal")
    urls = set(project.get("urls", {}))
    missing_urls = sorted(REQUIRED_PROJECT_URLS - urls)
    if missing_urls:
        fail(f"missing project URLs: {missing_urls}")
    if pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] != "cerm._version.__version__":
        fail("setuptools version source is inconsistent")

    version = read_version()
    if version != "1.0.0a1":
        fail(f"unexpected release version: {version}")

    for research_dir in (ROOT / "benchmarks", ROOT / "undr", ROOT / "docs" / "research"):
        if research_dir.exists():
            fail(f"research-only tree must live in the separate research workspace: {research_dir.relative_to(ROOT)}")

    offenders = []
    for path in ROOT.rglob("*"):
        generated_part = any(
            part in FORBIDDEN_PARTS or part.endswith(".egg-info")
            for part in path.parts
        )
        if generated_part or path.name in FORBIDDEN_NAMES:
            offenders.append(path)
    if offenders:
        fail(f"generated artifacts are present: {offenders[0]}")

    private_import_offenders = []
    for path in SRC.rglob("*.py"):
        if "_compat" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
                if any(alias.name.startswith("_") for alias in node.names):
                    private_import_offenders.append(path)
                    break
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("sklearn.") and "._" in alias.name for alias in node.names):
                    private_import_offenders.append(path)
                    break
    if private_import_offenders:
        fail(f"private sklearn imports escaped compatibility layer: {private_import_offenders}")

    with tempfile.TemporaryDirectory(prefix="cerm-compile-") as tmp:
        target = Path(tmp)
        for index, path in enumerate(sorted(SRC.rglob("*.py"))):
            try:
                py_compile.compile(
                    str(path),
                    cfile=str(target / f"{index}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                fail(f"compile failed for {path}: {exc}")

    print(f"CERM package checks passed for version {version}")


if __name__ == "__main__":
    main()
