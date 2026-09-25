from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _verify_record(base: Path, record: dict[str, Any]) -> Path:
    if not isinstance(record, dict) or "file" not in record:
        raise ValueError("invalid artifact record")
    artifact = base / record["file"]
    if not artifact.is_file():
        raise FileNotFoundError(f"referenced artifact not found: {artifact}")
    payload = artifact.read_bytes()
    if "bytes" in record and len(payload) != int(record["bytes"]):
        raise ValueError(f"byte-count mismatch for {artifact.name}")
    if "sha256" in record:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {artifact.name}")
    return artifact


def _verify_model_adapter_sections(manifest_path: Path, manifest: dict[str, Any]) -> None:
    base = manifest_path.parent
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("invalid model section")
    for record in model.values():
        _verify_record(base, record)
    adapter = manifest.get("adapter")
    if adapter is None:
        return
    if adapter.get("source") == "bundle-shared-adapter":
        return
    if "file" in adapter:
        _verify_record(base, adapter)
        return
    if not isinstance(adapter, dict):
        raise ValueError("invalid adapter section")
    artifact_records = {
        key: record
        for key, record in adapter.items()
        if isinstance(record, dict) and "file" in record
    }
    if not artifact_records:
        raise ValueError("adapter section does not contain artifact records")
    for record in artifact_records.values():
        _verify_record(base, record)


def verify_export(path: str | Path) -> dict[str, Any]:
    """Verify binary, regression, and multi-output CERM export manifests.

    Bundle manifests are traversed recursively, so one call validates every
    referenced binary/constant head and the head-manifest hashes recorded by the
    task-neutral bundle.
    """

    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CERM manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    format_name = manifest.get("format")

    if format_name in {
        "cerm-python-package-v2",
        "cerm-python-package-v3",
        "cerm-shared-program-v1",
        "cerm-shared-program-v2",
    }:
        _verify_model_adapter_sections(manifest_path, manifest)
        return manifest

    if format_name == "cerm-compiled-shared-program-v1":
        base = manifest_path.parent
        _verify_record(base, manifest["library"])
        _verify_record(base, manifest["runtime"])
        if manifest.get("source") is not None:
            _verify_record(base, manifest["source"])
        adapter = manifest.get("adapter")
        if adapter is not None:
            _verify_record(base, adapter["json"])
            _verify_record(base, adapter["npz"])
        return manifest

    if format_name in {"cerm-compiled-program-v1", "cerm-compiled-program-v2"}:
        base = manifest_path.parent
        _verify_record(base, manifest["library"])
        if manifest.get("source") is not None:
            _verify_record(base, manifest["source"])
        if format_name == "cerm-compiled-program-v1":
            _verify_record(base, manifest["state"])
        else:
            _verify_record(base, manifest["runtime"])
            adapter = manifest.get("adapter")
            if adapter is not None:
                if not isinstance(adapter, dict) or not adapter.get("portable", False):
                    raise ValueError("invalid compiled binary adapter section")
                _verify_record(base, adapter["json"])
                _verify_record(base, adapter["npz"])
        return manifest

    if format_name == "cerm-compiled-bundle-v2":
        base = manifest_path.parent
        shared = manifest.get("shared_adapter")
        if shared is not None:
            if not isinstance(shared, dict) or not shared.get("portable", False):
                raise ValueError("invalid compiled bundle shared adapter")
            _verify_record(base, shared["json"])
            _verify_record(base, shared["npz"])
        heads = manifest.get("heads")
        if not isinstance(heads, list):
            raise ValueError("invalid compiled bundle head list")
        for expected_index, head in enumerate(heads):
            if int(head.get("index", -1)) != expected_index:
                raise ValueError("compiled bundle head indices must be contiguous")
            child_manifest = base / head["directory"] / head["manifest"]
            payload = child_manifest.read_bytes()
            if hashlib.sha256(payload).hexdigest() != head["manifest_sha256"]:
                raise ValueError(
                    f"compiled head manifest SHA-256 mismatch for head {expected_index}"
                )
            verify_export(child_manifest)
        return manifest

    if format_name in {"cerm-compiled-regression-v1", "cerm-compiled-regression-v2"}:
        base = manifest_path.parent
        _verify_record(base, manifest["library"])
        if manifest.get("source") is not None:
            _verify_record(base, manifest["source"])
        if format_name == "cerm-compiled-regression-v1":
            _verify_record(base, manifest["state"])
        else:
            _verify_record(base, manifest["runtime"])
            adapter = manifest.get("adapter")
            if adapter is not None:
                if not isinstance(adapter, dict) or not adapter.get("portable", False):
                    raise ValueError("invalid compiled regression adapter section")
                _verify_record(base, adapter["json"])
                _verify_record(base, adapter["npz"])
        return manifest

    if format_name == "cerm-constant-binary-v1":
        probability = float(manifest.get("probability"))
        if not 0.0 <= probability <= 1.0:
            raise ValueError("constant probability must be in [0, 1]")
        return manifest

    if format_name in {"cerm-program-bundle-v1", "cerm-program-bundle-v2"}:
        if format_name == "cerm-program-bundle-v2":
            shared = manifest.get("shared_adapter")
            if not isinstance(shared, dict):
                raise ValueError("bundle v2 requires a shared_adapter section")
            for record in shared.values():
                _verify_record(manifest_path.parent, record)
        heads = manifest.get("heads")
        if not isinstance(heads, list) or len(heads) != int(manifest.get("n_outputs", -1)):
            raise ValueError("invalid program bundle head list")
        for expected_index, head in enumerate(heads):
            if int(head.get("index", -1)) != expected_index:
                raise ValueError("program bundle head indices must be contiguous")
            child_manifest = manifest_path.parent / head["directory"] / head["manifest"]
            payload = child_manifest.read_bytes()
            if hashlib.sha256(payload).hexdigest() != head["manifest_sha256"]:
                raise ValueError(f"head manifest SHA-256 mismatch for head {expected_index}")
            verify_export(child_manifest)
        return manifest

    raise ValueError(f"unsupported CERM export format: {format_name!r}")
