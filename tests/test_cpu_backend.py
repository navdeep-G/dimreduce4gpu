import numpy as np
import pytest

from dimreduce4gpu import PCA, TruncatedSVD
from dimreduce4gpu.lib_dimreduce4cpu import cpu_built


def _require_cpu_built():
    if not cpu_built():
        pytest.skip("CPU native library is not built or cannot be loaded.")


def _svd_trunc_reconstruct(X: np.ndarray, k: int, center: bool):
    if center:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    u = u[:, :k]
    s = s[:k]
    vt = vt[:k, :]
    return (u * s) @ vt


def test_auto_uses_cpu_when_gpu_unavailable():
    _require_cpu_built()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    pca = PCA(n_components=2, backend="auto")
    Z = pca.fit_transform(X)
    assert Z.shape == (4, 2)
    assert np.isfinite(Z).all()


def test_pca_cpu_reconstruction_matches_reference():
    _require_cpu_built()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    pca = PCA(n_components=2, backend="cpu", algorithm="cusolver")
    Z = pca.fit_transform(X)
    comps = pca.components_
    Xc_hat = Z @ comps
    Xc_ref = _svd_trunc_reconstruct(X.astype(np.float64), 2, center=True)
    # relative Frobenius error
    err = np.linalg.norm(Xc_hat - Xc_ref) / max(1e-12, np.linalg.norm(Xc_ref))
    assert err < 1e-4


def test_tsvd_cpu_reconstruction_matches_reference():
    _require_cpu_built()
    X = np.array(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 2.0, 0.0], [0.0, 0.25, 2.0]],
        dtype=np.float32,
    )
    tsvd = TruncatedSVD(n_components=2, backend="cpu", algorithm="cusolver")
    Z = tsvd.fit_transform(X)
    comps = tsvd.components_
    X_hat = Z @ comps
    X_ref = _svd_trunc_reconstruct(X.astype(np.float64), 2, center=False)
    err = np.linalg.norm(X_hat - X_ref) / max(1e-12, np.linalg.norm(X_ref))
    assert err < 1e-4
