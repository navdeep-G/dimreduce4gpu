import ctypes
import os
import sys


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


def _load_tsvd_lib():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [
        os.path.join(sys.prefix, "dimreduce4gpu"),
        curr_path,
        os.path.join(curr_path, "lib"),
    ]

    if os.name == "nt":
        dll_path = [os.path.join(p, "dimreduce4gpu.dll") for p in dll_path]
    else:
        dll_path = [os.path.join(p, "libdimreduce4gpu.so") for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if len(lib_path) == 0:
        raise RuntimeError(
            "Could not find CUDA native library 'libdimreduce4gpu'. Looked in: "
            + ", ".join(dll_path)
            + ". Build the CUDA backend (CMake) or set DIMREDUCE4GPU_LIB_PATH to point to the .so."
        )

    # Fix for GOMP weirdness with CUDA 8.0
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass
    _mod = ctypes.cdll.LoadLibrary(lib_path[0])
    _tsvd_code = _mod.truncated_svd_float
    _tsvd_code.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        params,
    ]

    return _tsvd_code


def _load_pca_lib():
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [
        os.path.join(sys.prefix, "dimreduce4gpu"),
        os.path.join(curr_path, "lib"),
        os.path.join(curr_path, "../lib/"),
    ]

    if os.name == "nt":
        dll_path = [os.path.join(p, "dimreduce4gpu.dll") for p in dll_path]
    else:
        dll_path = [os.path.join(p, "libdimreduce4gpu.so") for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if len(lib_path) == 0:
        raise RuntimeError(
            "Could not find CUDA native library 'libdimreduce4gpu'. Looked in: "
            + ", ".join(dll_path)
            + ". Build the CUDA backend (CMake) or set DIMREDUCE4GPU_LIB_PATH to point to the .so."
        )

    # Fix for GOMP weirdness with CUDA 8.0
    try:
        ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass
    _mod = ctypes.cdll.LoadLibrary(lib_path[0])
    _pca_code = _mod.pca_float
    _pca_code.argtypes = [
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

    return _pca_code
