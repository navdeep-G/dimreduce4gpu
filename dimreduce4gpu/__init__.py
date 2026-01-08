from __future__ import annotations

from ._backend import gpu_runnable, select_backend
from .lib_dimreduce4cpu import cpu_built, require_cpu_built
from .lib_dimreduce4gpu import params
from .pca import PCA
from .truncated_svd import TruncatedSVD


def native_built() -> bool:
    """Backward-compatible alias for "CUDA .so can be loaded".

    Historically the project exposed `native_built()` / `native_runnable()`.
    In the newer backend-selection implementation we expose `gpu_runnable()`.
    This keeps older tests and user code working.
    """

    try:
        # Import lazily to avoid importing ctypes on module import.
        from .lib_dimreduce4gpu import _load_pca_lib, _load_tsvd_lib

        _load_pca_lib()
        _load_tsvd_lib()
        return True
    except Exception:
        return False


def native_runnable() -> bool:
    """Backward-compatible alias for GPU runnable status."""

    return gpu_runnable()


def require_native() -> None:
    """Backward-compatible helper that requires the CUDA native library.

    Note: this checks for the CUDA backend specifically. It does not consider
    the CPU backend.
    """

    try:
        # Import lazily to avoid importing ctypes on module import.
        from .lib_dimreduce4gpu import _load_pca_lib, _load_tsvd_lib

        _load_pca_lib()
        _load_tsvd_lib()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CUDA native library (libdimreduce4gpu) is not available. "
            "Install/build the CUDA backend and ensure CUDA runtime libraries are on the loader path."
        ) from e


__all__ = [
    "PCA",
    "TruncatedSVD",
    "gpu_runnable",
    "native_built",
    "native_runnable",
    "require_native",
    "cpu_built",
    "require_cpu_built",
    "select_backend",
    "params",
]
