from __future__ import annotations

from ._native import (
    native_available,
    native_built,
    native_library_path,
    native_runnable,
    require_native,
    require_native_built,
    require_native_runnable,
)
from .pca import PCA
from .truncated_svd import TruncatedSVD

__version__ = "0.1.0"

__all__ = [
    "PCA",
    "TruncatedSVD",
    "native_available",  # backwards-compat: built & dlopen'able
    "native_built",
    "native_library_path",
    "native_runnable",
    "require_native",  # backwards-compat: built & dlopen'able
    "require_native_built",
    "require_native_runnable",
    "__version__",
]
