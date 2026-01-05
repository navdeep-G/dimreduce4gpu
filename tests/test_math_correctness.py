from __future__ import annotations

import numpy as np

import dimreduce4gpu
from dimreduce4gpu import PCA, TruncatedSVD


def _svd_reference_pca(X: np.ndarray, n_components: int):
    X_centered = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = vt[:n_components]
    scores = X_centered @ components.T
    return scores, components


def _svd_reference_tsvd(X: np.ndarray, n_components: int):
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    components = vt[:n_components]
    scores = X @ components.T
    return scores, components


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def test_pca_cpu_backend_matches_cpu_svd_reference_up_to_sign() -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    # CPU backend should exist in CI (built by workflow), but handle local dev gracefully.
    if not dimreduce4gpu.cpu_built() and not dimreduce4gpu.native_runnable():
        # Nothing can run; ensure error is clear.
        try:
            PCA(n_components=2, backend="cpu").fit_transform(X)
        except RuntimeError as e:
            assert "cpu" in str(e).lower() or "libdimreduce4cpu" in str(e).lower()
        else:
            raise AssertionError("Expected CPU backend to be unavailable")
        return

    pca = PCA(n_components=2, backend="cpu")
    X_cpu = pca.fit_transform(X)

    X_ref, _ = _svd_reference_pca(X.astype(np.float64), 2)
    assert X_cpu.shape == X_ref.shape

    for j in range(2):
        assert _corr_abs(X_cpu[:, j], X_ref[:, j]) > 0.99

    components = np.array(pca.components_, dtype=np.float64)
    assert components.shape == (2, 3)
    gram = components @ components.T
    assert np.allclose(gram, np.eye(2), atol=1e-3)


def test_truncated_svd_cpu_backend_matches_cpu_svd_reference_up_to_sign() -> None:
    X = np.array(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 2.0, 0.0], [0.0, 0.25, 2.0]],
        dtype=np.float32,
    )

    if not dimreduce4gpu.cpu_built() and not dimreduce4gpu.native_runnable():
        try:
            TruncatedSVD(n_components=2, algorithm="power", backend="cpu").fit_transform(X)
        except RuntimeError as e:
            assert "cpu" in str(e).lower() or "libdimreduce4cpu" in str(e).lower()
        else:
            raise AssertionError("Expected CPU backend to be unavailable")
        return

    tsvd = TruncatedSVD(n_components=2, algorithm="power", n_iter=5, backend="cpu")
    X_cpu = tsvd.fit_transform(X)

    X_ref, _ = _svd_reference_tsvd(X.astype(np.float64), 2)
    assert X_cpu.shape == X_ref.shape
    for j in range(2):
        assert _corr_abs(X_cpu[:, j], X_ref[:, j]) > 0.99

    components = np.array(tsvd.components_, dtype=np.float64)
    assert components.shape == (2, 3)
    gram = components @ components.T
    assert np.allclose(gram, np.eye(2), atol=1e-2)
