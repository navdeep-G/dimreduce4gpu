import pytest
import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import TruncatedSVD as SkTSVD

from dimreduce4gpu import PCA, TruncatedSVD
from dimreduce4gpu.lib_dimreduce4cpu import cpu_built


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def test_pca_cpu_close_to_sklearn_full_svd():
    if not cpu_built():
        pytest.skip("CPU native library is not built or cannot be loaded.")

    # Small-but-nontrivial matrix to keep test fast and deterministic.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 32)).astype(np.float32)

    ours = PCA(n_components=8, backend="cpu", algorithm="cusolver")
    Z_ours = ours.fit_transform(X)
    C_ours = np.asarray(ours.components_, dtype=np.float64)

    sk = SkPCA(n_components=8, svd_solver="full", random_state=0)
    Z_sk = sk.fit_transform(X)
    C_sk = np.asarray(sk.components_, dtype=np.float64)

    assert Z_ours.shape == Z_sk.shape == (128, 8)
    assert C_ours.shape == C_sk.shape == (8, 32)

    # Compare component score vectors by absolute correlation (sign may flip).
    for j in range(8):
        assert _corr_abs(Z_ours[:, j], Z_sk[:, j]) > 0.995

    # Compare components similarly (up to sign).
    for j in range(8):
        assert _corr_abs(C_ours[j, :], C_sk[j, :]) > 0.995


def test_tsvd_cpu_close_to_sklearn_randomized():
    if not cpu_built():
        pytest.skip("CPU native library is not built or cannot be loaded.")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(160, 40)).astype(np.float32)

    ours = TruncatedSVD(n_components=10, backend="cpu", algorithm="cusolver")
    Z_ours = ours.fit_transform(X)
    C_ours = np.asarray(ours.components_, dtype=np.float64)

    sk = SkTSVD(n_components=10, algorithm="randomized", n_iter=7, random_state=0)
    Z_sk = sk.fit_transform(X)
    C_sk = np.asarray(sk.components_, dtype=np.float64)

    assert Z_ours.shape == Z_sk.shape == (160, 10)
    assert C_ours.shape == C_sk.shape == (10, 40)

    for j in range(10):
        assert _corr_abs(Z_ours[:, j], Z_sk[:, j]) > 0.99
        assert _corr_abs(C_ours[j, :], C_sk[j, :]) > 0.99
