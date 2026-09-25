"""Validate local documentation links and publication metadata."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:")


def fail(message: str) -> None:
    raise SystemExit(f"documentation check failed: {message}")


def check_markdown_links() -> int:
    checked = 0
    roots = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md"]
    roots.extend(sorted((ROOT / "docs").rglob("*.md")))
    for document in roots:
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split()[0].strip("<>")
            if not target or target.startswith("#") or target.startswith(SKIP_SCHEMES):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            resolved = (document.parent / path_text).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                fail(f"link escapes repository: {document.relative_to(ROOT)} -> {target}")
            if not resolved.exists():
                fail(f"broken local link: {document.relative_to(ROOT)} -> {target}")
            checked += 1
    return checked


def require_fields(path: Path, fields: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [
        field
        for field in fields
        if not re.search(rf"(?m)^{re.escape(field)}\s*:", text)
    ]
    if missing:
        fail(f"{path.relative_to(ROOT)} is missing fields: {missing}")


def main() -> None:
    links = check_markdown_links()
    citation = ROOT / "CITATION.cff"
    require_fields(
        citation,
        (
            "cff-version",
            "message",
            "title",
            "type",
            "authors",
            "version",
            "license",
            "repository-code",
            "url",
        ),
    )
    citation_text = citation.read_text(encoding="utf-8")
    version_text = (ROOT / "src" / "cerm" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_text)
    if match is None or f'version: "{match.group(1)}"' not in citation_text:
        fail("CITATION.cff version does not match cerm._version")
    if 'license: "Apache-2.0"' not in citation_text:
        fail("CITATION.cff must declare Apache-2.0")
    for path in sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml")):
        if path.name == "config.yml":
            continue
        require_fields(path, ("name", "description", "title", "body"))
    print(f"documentation checks passed: {links} local links")


if __name__ == "__main__":
    main()
