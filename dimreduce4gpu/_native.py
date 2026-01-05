from __future__ import annotations

import ctypes
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
        p = Path(env).expanduser().resolve()
        if p.is_file():
            candidates.append(p.parent)
        else:
            candidates.append(p)

    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / "lib")
    candidates.append(Path(sys.prefix) / "dimreduce4gpu")
    candidates.append(pkg_dir.parent / "lib")  # legacy build output

    return candidates


def get_library_path() -> str | None:
    """Return the full path to libdimreduce4gpu if it exists, otherwise None."""
    libname = "dimreduce4gpu.dll" if os.name == "nt" else "libdimreduce4gpu.so"

    for base in _candidate_paths():
        try:
            candidate = base / libname
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        except OSError:
            continue

    return None


def native_built() -> bool:
    """True if the native shared library exists and can be dlopen()'d.

    This does NOT guarantee GPU execution is possible (that requires an NVIDIA driver).
    """
    path = get_library_path()
    if path is None:
        return False
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


def native_runnable() -> bool:
    """True if the native library is built AND the NVIDIA driver library is present.

    This is a best-effort signal that GPU execution is possible. On CI build containers,
    compilation may succeed but the driver (libcuda.so.1) is typically absent.
    """
    if not native_built():
        return False
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


# Backwards-compatible name (historically used by earlier patches/tests).
# "available" here means "built & loadable", not "GPU runnable".
def native_available() -> bool:
    return native_built()


def require_native_built() -> str:
    """Return the shared library path, or raise a friendly error.

    Raises if the library is missing OR cannot be loaded due to missing deps.
    """
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

    try:
        ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(
            "dimreduce4gpu found libdimreduce4gpu, but it could not be loaded. "
            "This usually means a required shared library dependency is missing.\n\n"
            f"Path: {path}\n"
            f"Load error: {e}"
        ) from e

    return path


def require_native_runnable() -> str:
    """Return the shared library path, or raise a friendly error if GPU can't run."""
    path = require_native_built()

    # If the driver library is missing, GPU execution can't work.
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise RuntimeError(
            "dimreduce4gpu native library is present, but the NVIDIA driver runtime "
            "(libcuda.so.1) is not available in this environment.\n\n"
            "This environment can compile the CUDA library, but cannot execute GPU code. "
            "To run GPU computations, use a machine/runner with NVIDIA drivers and a GPU."
        ) from e

    return path


# Backwards-compatible alias
def require_native() -> str:
    return require_native_built()
