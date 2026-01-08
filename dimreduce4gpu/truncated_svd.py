from __future__ import annotations

import ctypes
from typing import Literal, Optional

import numpy as np

from ._backend import select_backend
from .lib_dimreduce4cpu import _load_tsvd_cpu_lib
from .lib_dimreduce4gpu import _load_tsvd_lib, params

Backend = Literal["auto", "gpu", "cpu"]


def _as_fptr(x: np.ndarray):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class TruncatedSVD:
    """Truncated SVD with GPU (CUDA) or CPU native backend."""

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "power",
        n_iter: int = 5,
        random_state: Optional[int] = None,
        tol: float = 1e-5,
        verbose: bool = False,
        gpu_id: int = 0,
        backend: Backend = "auto",
    ) -> None:
        self.n_components = int(n_components)
        self.algorithm = str(algorithm)
        self.n_iter = int(n_iter)
        self.random_state = (
            int(random_state) if random_state is not None else int(np.random.randint(0, 2**31 - 1))
        )
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.gpu_id = int(gpu_id)
        self.backend: Backend = backend

        self._Q: Optional[np.ndarray] = None
        self._w: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    @property
    def components_(self) -> np.ndarray:
        if self._Q is None:
            raise AttributeError("components_ is not available before fit/fit_transform.")
        return self._Q

    @property
    def singular_values_(self) -> np.ndarray:
        if self._w is None:
            raise AttributeError("singular_values_ is not available before fit/fit_transform.")
        return self._w

    def fit(self, X: np.ndarray, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import scipy

        if isinstance(X, scipy.sparse.csr_matrix):
            X = X.toarray()

        X = np.ascontiguousarray(X, dtype=np.float32)
        n, m = X.shape
        k = min(self.n_components, n, m)

        Q = np.zeros((k, m), dtype=np.float32)
        w = np.zeros((k,), dtype=np.float32)
        U = np.zeros((n, k), dtype=np.float32)
        X_transformed = np.zeros((n, k), dtype=np.float32)
        explained_variance = np.zeros((k,), dtype=np.float32)
        explained_variance_ratio = np.zeros((k,), dtype=np.float32)

        p = params()
        p.X_n = n
        p.X_m = m
        p.k = k
        p.algorithm = self.algorithm.encode("utf-8")
        p.n_iter = self.n_iter
        p.random_state = self.random_state
        p.tol = float(self.tol)
        p.verbose = 1 if self.verbose else 0
        p.gpu_id = self.gpu_id
        p.whiten = False

        backend = select_backend(self.backend)
        fn = _load_tsvd_cpu_lib() if backend == "cpu" else _load_tsvd_lib()

        fn(
            _as_fptr(X),
            _as_fptr(Q),
            _as_fptr(w),
            _as_fptr(U),
            _as_fptr(X_transformed),
            _as_fptr(explained_variance),
            _as_fptr(explained_variance_ratio),
            p,
        )

        self._Q = Q
        self._w = w
        self._U = U
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        return X_transformed

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        return X @ self.components_.T
