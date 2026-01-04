from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_paths() -> list[Path]:
    """Return candidate locations for the CUDA shared library.

    Search order:
      1) DIMREDUCE4GPU_LIB_PATH env var (file or directory)
      2) Package directory: dimreduce4gpu/lib/
      3) Legacy locations used by older versions:
         - <sys.prefix>/dimreduce4gpu/
         - <package_dir>/../lib/
    """
    candidates: list[Path] = []

    env = os.environ.get("DIMREDUCE4GPU_LIB_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            candidates.append(p)
        else:
            candidates.append(p.parent)

    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / "lib")
    candidates.append(Path(sys.prefix) / "dimreduce4gpu")
    candidates.append(pkg_dir.parent / "lib")  # legacy build output

    return candidates


def get_library_path() -> str | None:
    """Return the path to libdimreduce4gpu if it exists, otherwise None."""
    libname = "dimreduce4gpu.dll" if os.name == "nt" else "libdimreduce4gpu.so"

    for base in _candidate_paths():
        try:
            candidate = base / libname
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        except OSError:
            continue

    return None


def native_available() -> bool:
    return get_library_path() is not None


def require_native() -> str:
    """Return the shared library path, or raise a friendly error."""
    path = get_library_path()
    if path is None:
        searched = [str(p) for p in _candidate_paths()]
        msg = (
            "dimreduce4gpu native CUDA library is not available. "
            "This is expected on CPU-only machines/CI.\n\n"
            "To enable GPU functionality:\n"
            "  1) Build the CUDA library via CMake (see README)\n"
            "  2) Ensure libdimreduce4gpu is located in dimreduce4gpu/lib/\n"
            "     (or set DIMREDUCE4GPU_LIB_PATH to its directory)\n\n"
            f"Searched: {searched}"
        )
        raise RuntimeError(msg)
    return path
