from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ._backend import select_backend
from .lib_dimreduce4cpu import _load_pca_cpu_lib
from .lib_dimreduce4gpu import _load_pca_lib, params
from .truncated_svd import TruncatedSVD, _as_fptr

Backend = Literal["auto", "gpu", "cpu"]


class PCA(TruncatedSVD):
    """PCA implemented via SVD with native GPU or CPU backend."""

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "cusolver",
        n_iter: int = 5,
        random_state: Optional[int] = None,
        tol: float = 1e-5,
        verbose: bool = False,
        gpu_id: int = 0,
        whiten: bool = False,
        backend: Backend = "auto",
    ) -> None:
        super().__init__(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            gpu_id=gpu_id,
            backend=backend,
        )
        self.whiten = bool(whiten)
        self.mean_: Optional[np.ndarray] = None

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
        mean = np.zeros((m,), dtype=np.float32)

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
        p.whiten = bool(self.whiten)

        backend = select_backend(self.backend)
        fn = _load_pca_cpu_lib() if backend == "cpu" else _load_pca_lib()

        fn(
            _as_fptr(X),
            _as_fptr(Q),
            _as_fptr(w),
            _as_fptr(U),
            _as_fptr(X_transformed),
            _as_fptr(explained_variance),
            _as_fptr(explained_variance_ratio),
            _as_fptr(mean),
            p,
        )

        self._Q = Q
        self._w = w
        self._U = U
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.mean_ = mean
        return X_transformed
