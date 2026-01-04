import numpy as np
import pytest

from dimreduce4gpu import PCA, TruncatedSVD, native_available


def _svd_reference_pca(X: np.ndarray, n_components: int):
    """CPU reference for PCA: center then SVD."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = vt[:n_components]
    scores = X_centered @ components.T
    return scores, components


def _svd_reference_tsvd(X: np.ndarray, n_components: int):
    """CPU reference for TruncatedSVD: SVD without centering."""
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    components = vt[:n_components]
    scores = X @ components.T
    return scores, components


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation magnitude between two 1D vectors (sign-invariant)."""
    a = a.ravel()
    b = b.ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _assert_missing_native_raises(callable_):
    # Always executed in CI if native library isn't present:
    # ensures failures are *clean* and *actionable*.
    with pytest.raises(RuntimeError) as excinfo:
        callable_()
    msg = str(excinfo.value).lower()
    assert ("native" in msg) or ("libdimreduce4gpu" in msg) or ("cuda" in msg)


def test_pca_fit_transform_matches_cpu_svd_up_to_sign():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    if not native_available():
        _assert_missing_native_raises(lambda: PCA(n_components=2).fit_transform(X))
        return

    pca = PCA(n_components=2, algorithm="cusolver")
    X_gpu = pca.fit_transform(X)

    X_ref, _ = _svd_reference_pca(X.astype(np.float64), 2)

    assert X_gpu.shape == X_ref.shape

    # Compare each component score vector by absolute correlation (sign may flip).
    for j in range(2):
        assert _corr_abs(X_gpu[:, j], X_ref[:, j]) > 0.99

    # Components should be approximately orthonormal.
    components = np.array(pca.components_, dtype=np.float64)
    assert components.shape == (2, 3)
    gram = components @ components.T
    assert np.allclose(gram, np.eye(2), atol=1e-3)


def test_truncated_svd_fit_transform_matches_cpu_svd_up_to_sign():
    X = np.array(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 2.0, 0.0], [0.0, 0.25, 2.0]],
        dtype=np.float32,
    )

    if not native_available():
        _assert_missing_native_raises(
            lambda: TruncatedSVD(n_components=2, algorithm="power").fit_transform(X)
        )
        return

    tsvd = TruncatedSVD(n_components=2, algorithm="power", n_iter=5)
    X_gpu = tsvd.fit_transform(X)

    X_ref, _ = _svd_reference_tsvd(X.astype(np.float64), 2)

    assert X_gpu.shape == X_ref.shape
    for j in range(2):
        assert _corr_abs(X_gpu[:, j], X_ref[:, j]) > 0.99

    components = np.array(tsvd.components_, dtype=np.float64)
    assert components.shape == (2, 3)
    gram = components @ components.T
    assert np.allclose(gram, np.eye(2), atol=1e-2)
