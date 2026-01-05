from __future__ import annotations

import ctypes
from typing import Literal

from .lib_dimreduce4cpu import cpu_built
from .lib_dimreduce4gpu import _load_pca_lib, _load_tsvd_lib

Backend = Literal["auto", "gpu", "cpu"]


def _cuda_device_count() -> int:
    """Return CUDA device count using the driver API (libcuda)."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0

    cu_init = libcuda.cuInit
    cu_init.argtypes = [ctypes.c_uint]
    cu_init.restype = ctypes.c_int

    cu_device_get_count = libcuda.cuDeviceGetCount
    cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_device_get_count.restype = ctypes.c_int

    if cu_init(0) != 0:
        return 0
    count = ctypes.c_int(0)
    if cu_device_get_count(ctypes.byref(count)) != 0:
        return 0
    return int(count.value)


def gpu_runnable() -> bool:
    """True if GPU native library can be loaded and a CUDA device is available."""
    try:
        _load_tsvd_lib()
        _load_pca_lib()
    except Exception:
        return False
    return _cuda_device_count() > 0


def select_backend(requested: Backend) -> Backend:
    if requested == "auto":
        if gpu_runnable():
            return "gpu"
        if cpu_built():
            return "cpu"
        return "gpu"
    return requested
