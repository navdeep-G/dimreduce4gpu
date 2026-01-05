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


def _cuda_driver_device_count() -> tuple[bool, int, str]:
    """Return (ok, device_count, reason) using the CUDA Driver API.

    This checks for:
      - NVIDIA driver runtime availability (libcuda.so.1)
      - cuInit success
      - at least one CUDA device

    It does not execute kernels, but it is a strong signal the environment is capable of GPU execution.
    """
    if not native_built():
        return False, 0, "native library is not built"

    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return False, 0, f"NVIDIA driver runtime missing (libcuda.so.1): {e}"

    # int cuInit(unsigned int Flags);
    cuInit = getattr(libcuda, "cuInit", None)
    cuDeviceGetCount = getattr(libcuda, "cuDeviceGetCount", None)
    if cuInit is None or cuDeviceGetCount is None:
        return False, 0, "CUDA Driver API symbols missing (cuInit/cuDeviceGetCount)"

    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int

    cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuDeviceGetCount.restype = ctypes.c_int

    CUDA_SUCCESS = 0
    CUDA_ERROR_NO_DEVICE = 100

    rc = int(cuInit(0))
    if rc == CUDA_ERROR_NO_DEVICE:
        return False, 0, "no CUDA devices detected (driver present, but no GPU available)"
    if rc != CUDA_SUCCESS:
        return False, 0, f"cuInit failed with error code {rc}"

    count = ctypes.c_int(0)
    rc2 = int(cuDeviceGetCount(ctypes.byref(count)))
    if rc2 != CUDA_SUCCESS:
        return False, 0, f"cuDeviceGetCount failed with error code {rc2}"

    if count.value <= 0:
        return False, 0, "no CUDA devices detected (device count is 0)"

    return True, int(count.value), ""


def native_runnable() -> bool:
    """True if the native library is built, loadable, and a CUDA device is available.

    This is stricter than native_built(): it requires the NVIDIA driver runtime and at least
    one CUDA device (via CUDA Driver API).
    """
    ok, _count, _reason = _cuda_driver_device_count()
    return ok


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

    ok, _count, reason = _cuda_driver_device_count()
    if not ok:
        raise RuntimeError(
            "dimreduce4gpu native library is present, but this environment is not able to run GPU code.

"
            f"Reason: {reason}

"
            "This environment may be able to compile the CUDA library, but to execute GPU computations you need:
"
            "  - NVIDIA drivers installed (libcuda.so.1 available)
"
            "  - At least one CUDA-capable GPU device
"
            "  - A compatible CUDA runtime/toolkit for your driver
"
        )

    return path




# Backwards-compatible alias
def require_native() -> str:
    return require_native_built()
