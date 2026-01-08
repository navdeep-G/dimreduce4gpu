from __future__ import annotations

import ctypes
import os
import sys

from .lib_dimreduce4gpu import params


def _candidate_paths() -> list[str]:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    env_path = os.environ.get("DIMREDUCE4GPU_CPU_LIB_PATH")
    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)

    candidates.append(os.path.join(curr_path, "lib", "libdimreduce4cpu.so"))
    candidates.append(os.path.join(curr_path, "../lib/libdimreduce4cpu.so"))
    candidates.append(os.path.join(sys.prefix, "dimreduce4gpu", "libdimreduce4cpu.so"))

    out: list[str] = []
    for p in candidates:
        p = os.path.abspath(os.path.expanduser(p))
        if p not in out:
            out.append(p)
    return out


def cpu_built() -> bool:
    for p in _candidate_paths():
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p)
                return True
            except OSError:
                return False
    return False


def require_cpu_built() -> str:
    paths = _candidate_paths()
    for p in paths:
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p)
                return p
            except OSError as e:
                raise RuntimeError(
                    f"CPU native library was found but could not be loaded. Path={p}. Error={e}"
                ) from e
    raise RuntimeError(
        "CPU native library (libdimreduce4cpu.so) not found. "
        "Build it with:\n"
        "  cmake -S . -B build -DDIMREDUCE4GPU_BUILD_CPU=ON -DDIMREDUCE4GPU_BUILD_CUDA=OFF\n"
        "  cmake --build build -j\n"
        "Or set DIMREDUCE4GPU_CPU_LIB_PATH to the full path of libdimreduce4cpu.so.\n"
        f"Searched: {paths}"
    )


def _load_tsvd_cpu_lib():
    lib_path = require_cpu_built()
    mod = ctypes.cdll.LoadLibrary(lib_path)
    fn = mod.truncated_svd_float
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        params,
    ]
    return fn


def _load_pca_cpu_lib():
    lib_path = require_cpu_built()
    mod = ctypes.cdll.LoadLibrary(lib_path)
    fn = mod.pca_float
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        params,
    ]
    return fn
