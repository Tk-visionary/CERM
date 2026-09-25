"""Validate CERM wheel metadata/RECORD and source-distribution contents."""

from __future__ import annotations

import base64
import csv
import hashlib
from io import StringIO
from pathlib import Path
import sys
import tarfile
import zipfile

from packaging.metadata import Metadata

EXPECTED_NAME = "CERM"
EXPECTED_VERSION = "1.0.0a1"
THIRD_PARTY_LICENSE_SUFFIXES = {
    "/THIRD_PARTY_NOTICES.md",
    "/LIBLINEAR-BSD-3-Clause.txt",
    "/SCIKIT-LEARN-BSD-3-Clause.txt",
}


def _require_suffixes(names: set[str], suffixes: set[str], kind: str) -> None:
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise RuntimeError(f"{kind} is missing required files: {missing}")


def verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(metadata_names) != 1 or len(record_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA and RECORD")

        metadata = Metadata.from_email(archive.read(metadata_names[0]), validate=True)
        if metadata.name != EXPECTED_NAME or str(metadata.version) != EXPECTED_VERSION:
            raise RuntimeError(f"unexpected wheel identity: {metadata.name} {metadata.version}")
        if metadata.license_expression != "Apache-2.0":
            raise RuntimeError("unexpected license expression")
        if metadata.author != "Taishi Kawahara":
            raise RuntimeError("unexpected or missing author metadata")
        if any(name.startswith("tests/") for name in names):
            raise RuntimeError("tests must not be installed in the wheel")
        if any(name.startswith("cerm/_vendor/") for name in names):
            raise RuntimeError("historical cerm._vendor package must not be installed")
        if "cerm/_internal/__init__.py" not in names:
            raise RuntimeError("wheel is missing cerm._internal")
        _require_suffixes(names, THIRD_PARTY_LICENSE_SUFFIXES, "wheel")

        rows = csv.reader(StringIO(archive.read(record_names[0]).decode("utf-8")))
        for filename, digest, size in rows:
            if filename == record_names[0]:
                continue
            payload = archive.read(filename)
            if size and int(size) != len(payload):
                raise RuntimeError(f"RECORD size mismatch for {filename}")
            if digest:
                algorithm, encoded = digest.split("=", 1)
                if algorithm != "sha256":
                    raise RuntimeError(f"unsupported RECORD algorithm: {algorithm}")
                actual = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
                if actual != encoded:
                    raise RuntimeError(f"RECORD hash mismatch for {filename}")
    print(f"wheel verified: {path.name}")


def verify_sdist(path: Path) -> None:
    required_suffixes = {
        "/pyproject.toml",
        "/README.md",
        "/CHANGELOG.md",
        "/LICENSE",
        "/THIRD_PARTY_NOTICES.md",
        "/LICENSES/LIBLINEAR-BSD-3-Clause.txt",
        "/LICENSES/SCIKIT-LEARN-BSD-3-Clause.txt",
        "/SECURITY.md",
        "/AI_ASSISTANCE.md",
        "/PROVENANCE.md",
        "/CITATION.cff",
        "/docs/benchmarking.md",
        "/docs/installation.md",
        "/examples/quickstart.py",
        "/src/cerm/__init__.py",
        "/src/cerm/_version.py",
        "/src/cerm/adaptive_representation.py",
    }
    forbidden_fragments = (
        "/artifacts/",
        "/evidence/",
        "/reports/",
        "/provenance/",
        "/patches/",
        "/benchmarks/",
        "/undr/",
        "/docs/research/",
        "/.github/",
        "/tests/",
    )
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        _require_suffixes(names, required_suffixes, "sdist")
        if any("/__pycache__/" in name or name.endswith(".pyc") for name in names):
            raise RuntimeError("sdist contains generated Python cache files")
        leaked = [name for name in names if any(fragment in f"/{name}" for fragment in forbidden_fragments)]
        if leaked:
            raise RuntimeError(f"sdist contains repository-only files: {leaked[0]}")
    print(f"sdist verified: {path.name}")


def main(arguments: list[str]) -> None:
    if not arguments:
        raise SystemExit("usage: verify_distributions.py DIST [DIST ...]")
    for raw in arguments:
        path = Path(raw)
        if path.suffix == ".whl":
            verify_wheel(path)
        elif path.name.endswith(".tar.gz"):
            verify_sdist(path)
        else:
            raise RuntimeError(f"unsupported distribution type: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
