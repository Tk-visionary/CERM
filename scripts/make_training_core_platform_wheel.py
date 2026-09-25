"""Retag a CERM wheel that contains a packaged training-core shared library.

This keeps the project backend simple: build the normal wheel after generating
one platform-matching prebuilt training core, then convert that wheel into a
``py3-none-<platform>`` wheel with ``Root-Is-Purelib: false`` and a refreshed
RECORD.  The output is intended for validation/release tooling, not runtime use.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from io import StringIO
from pathlib import Path
import re
import sysconfig
import zipfile

from packaging.utils import canonicalize_name, parse_wheel_filename


def _normalize_platform_tag(value: str) -> str:
    return re.sub(r"[-.]", "_", value)


def _default_platform_tag() -> str:
    # Use the interpreter's local platform name by default.  Release tooling may
    # override this only after a compatibility audit (e.g. manylinux policy or
    # an explicit macOS deployment target).
    return _normalize_platform_tag(sysconfig.get_platform())


def _digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _wheel_filename(path: Path, platform_tag: str) -> str:
    name, version, build, _tags = parse_wheel_filename(path.name)
    distribution = canonicalize_name(name).replace("-", "_")
    parts = [distribution, str(version)]
    if build:
        parts.append("".join(str(value) for value in build))
    parts.extend(["py3", "none", platform_tag])
    return "-".join(parts) + ".whl"


def convert(input_wheel: Path, output_dir: Path, platform_tag: str) -> Path:
    platform_tag = _normalize_platform_tag(platform_tag)
    if platform_tag == "any":
        raise ValueError("native training-core wheel cannot use platform tag 'any'")

    with zipfile.ZipFile(input_wheel) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}

    wheel_files = [name for name in payloads if name.endswith(".dist-info/WHEEL")]
    record_files = [name for name in payloads if name.endswith(".dist-info/RECORD")]
    if len(wheel_files) != 1 or len(record_files) != 1:
        raise RuntimeError("wheel must contain exactly one WHEEL and one RECORD")

    native_names = [
        name
        for name in payloads
        if "/_internal/native/prebuilt/" in f"/{name}"
        and name.endswith((".so", ".dylib", ".dll"))
    ]
    if len(native_names) != 1:
        raise RuntimeError(
            f"platform wheel must contain exactly one prebuilt training core, found {native_names}"
        )

    wheel_name = wheel_files[0]
    lines = payloads[wheel_name].decode("utf-8").splitlines()
    rewritten: list[str] = []
    saw_purelib = False
    for line in lines:
        # WHEEL is parsed as RFC-style headers.  Do not preserve a terminating
        # blank line before appending Tag: or pip will treat the tag as body text.
        if not line.strip():
            continue
        if line.startswith("Root-Is-Purelib:"):
            rewritten.append("Root-Is-Purelib: false")
            saw_purelib = True
        elif not line.startswith("Tag:"):
            rewritten.append(line)
    if not saw_purelib:
        raise RuntimeError("WHEEL metadata is missing Root-Is-Purelib")
    rewritten.append(f"Tag: py3-none-{platform_tag}")
    payloads[wheel_name] = ("\n".join(rewritten) + "\n").encode("utf-8")

    record_name = record_files[0]
    rows = StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for name in sorted(payloads):
        if name == record_name:
            continue
        payload = payloads[name]
        writer.writerow([name, _digest(payload), str(len(payload))])
    writer.writerow([record_name, "", ""])
    payloads[record_name] = rows.getvalue().encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / _wheel_filename(input_wheel, platform_tag)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(payloads):
            archive.writestr(name, payloads[name])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dist-native"))
    parser.add_argument("--platform-tag", default=_default_platform_tag())
    args = parser.parse_args()
    output = convert(args.input_wheel, args.output_dir, args.platform_tag)
    print(output)


if __name__ == "__main__":
    main()
