from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class NativeBuildArtifact:
    source_path: Path
    library_path: Path
    command: tuple[str, ...]
    compiler_stdout: str
    compiler_stderr: str


def compile_shared_library(
    source_path: str | Path,
    library_path: str | Path,
    *,
    compiler: str = "g++",
    optimization: str = "-O3",
    native_arch: bool = True,
    extra_flags: Sequence[str] = (),
) -> NativeBuildArtifact:
    source = Path(source_path)
    library = Path(library_path)
    if not source.exists():
        raise FileNotFoundError(source)
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError(f"C++ compiler not found: {compiler}")
    command = [
        executable,
        optimization,
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    if native_arch:
        command.append("-march=native")
    command.extend(extra_flags)
    dead_code_flag = (
        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections"
    )
    command.extend([str(source), dead_code_flag, "-o", str(library)])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "native compilation failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return NativeBuildArtifact(
        source_path=source,
        library_path=library,
        command=tuple(command),
        compiler_stdout=result.stdout,
        compiler_stderr=result.stderr,
    )
