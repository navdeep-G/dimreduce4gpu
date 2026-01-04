from __future__ import annotations

import ctypes
from typing import Callable

from ._native import require_native


class params(ctypes.Structure):
    _fields_ = [
        ("X_n", ctypes.c_int),
        ("X_m", ctypes.c_int),
        ("k", ctypes.c_int),
        ("algorithm", ctypes.c_char_p),
        ("n_iter", ctypes.c_int),
        ("random_state", ctypes.c_int),
        ("tol", ctypes.c_float),
        ("verbose", ctypes.c_int),
        ("gpu_id", ctypes.c_int),
        ("whiten", ctypes.c_bool),
    ]


def _load_shared() -> ctypes.CDLL:
    """Load the CUDA shared library with a friendlier error message."""
    lib_path = require_native()

    # Fix for GOMP weirdness (historical). Harmless if not present.
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    return ctypes.cdll.LoadLibrary(lib_path)


def _load_tsvd_lib() -> Callable:
    """Return the TruncatedSVD entry point (float32)."""
    mod = _load_shared()
    fn = mod.truncated_svd_float
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # w
        ctypes.POINTER(ctypes.c_float),  # U
        ctypes.POINTER(ctypes.c_float),  # X_transformed
        ctypes.POINTER(ctypes.c_float),  # explained_variance
        ctypes.POINTER(ctypes.c_float),  # explained_variance_ratio
        params,
    ]
    return fn


def _load_pca_lib() -> Callable:
    """Return the PCA entry point (float32)."""
    mod = _load_shared()
    fn = mod.pca_float
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.POINTER(ctypes.c_float),  # Q
        ctypes.POINTER(ctypes.c_float),  # w
        ctypes.POINTER(ctypes.c_float),  # U
        ctypes.POINTER(ctypes.c_float),  # X_transformed
        ctypes.POINTER(ctypes.c_float),  # explained_variance
        ctypes.POINTER(ctypes.c_float),  # explained_variance_ratio
        ctypes.POINTER(ctypes.c_float),  # mean
        params,
    ]
    return fn
