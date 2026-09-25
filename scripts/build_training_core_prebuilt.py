"""Build the portable CERM training-core shared library for this platform."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "cerm" / "_internal" / "native" / "cerm_training_core.cpp"
ABI_VERSION = 2


def platform_key() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        machine = "arm64"
    else:
        raise RuntimeError(f"unsupported architecture: {machine}")
    if sys.platform.startswith("linux"):
        return f"linux-{machine}"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    raise RuntimeError(f"prebuilt training-core builder is unsupported on {sys.platform}")


def library_name() -> str:
    if sys.platform == "darwin":
        return "libcerm_training_core.dylib"
    if sys.platform.startswith("linux"):
        return "libcerm_training_core.so"
    raise RuntimeError(f"unsupported platform: {sys.platform}")


def main() -> None:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("no C++17 compiler found")
    output = (
        ROOT
        / "src"
        / "cerm"
        / "_internal"
        / "native"
        / "prebuilt"
        / platform_key()
        / library_name()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-ffunction-sections",
        "-fdata-sections",
        "-pthread",
    ]
    if sys.platform == "darwin":
        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
        command.append(f"-mmacosx-version-min={deployment_target}")
    command.extend(
        [
            str(SOURCE),
            "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
            "-o",
            str(output),
        ]
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "training-core prebuilt compilation failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    library = ctypes.CDLL(str(output))
    abi = library.cerm_training_core_abi_version
    abi.argtypes = []
    abi.restype = ctypes.c_int
    version = int(abi())
    if version != ABI_VERSION:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"training-core ABI mismatch: expected {ABI_VERSION}, got {version}")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
