import numpy as np

from dimreduce4gpu import PCA, TruncatedSVD, native_available


def _svd_reference_pca(X: np.ndarray, n_components: int):
    # Center X for PCA reference
    Xc = X - X.mean(axis=0, keepdims=True)
    # Full SVD on CPU
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # PCA components are Vt[:k]
    comps = Vt[:n_components]
    # Transformed scores are Xc @ comps.T (up to sign)
    X_t = Xc @ comps.T
    return X_t, comps


def _svd_reference_tsvd(X: np.ndarray, n_components: int):
    # For TruncatedSVD (no centering): X ≈ U S Vt
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:n_components]
    X_t = X @ comps.T
    return X_t, comps


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    # Correlation magnitude between two 1D vectors
    a = a.ravel()
    b = b.ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 1.0 if np.allclose(a, b) else 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def test_pca_fit_transform_matches_cpu_svd_up_to_sign():
    X = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32
    )

    if not native_available():
        # CI/CPU-only path: test that we fail cleanly with an actionable message
        try:
            PCA(n_components=2).fit_transform(X)
            assert False, "Expected PCA.fit_transform to fail when native library is unavailable."
        except RuntimeError as e:
            msg = str(e).lower()
            assert "native" in msg or "libdimreduce4gpu" in msg or "cuda" in msg
        return

    # GPU/native path: validate numerics up to sign ambiguity
    pca = PCA(n_components=2, algorithm="cusolver")
    X_gpu = pca.fit_transform(X)

    X_ref, comps_ref = _svd_reference_pca(X.astype(np.float64), 2)

    assert X_gpu.shape == X_ref.shape

    # Compare each component score vector by absolute correlation (sign may flip)
    for j in range(2):
        assert _corr_abs(X_gpu[:, j], X_ref[:, j]) > 0.99

    # Components should be approximately orthonormal
    C = np.array(pca.components_, dtype=np.float64)
    assert C.shape == (2, 3)
    I = C @ C.T
    assert np.allclose(I, np.eye(2), atol=1e-3)


def test_truncated_svd_fit_transform_matches_cpu_svd_up_to_sign():
    X = np.array(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 2.0, 0.0], [0.0, 0.25, 2.0]],
        dtype=np.float32,
    )

    if not native_available():
        try:
            TruncatedSVD(n_components=2, algorithm="power").fit_transform(X)
            assert False, "Expected TruncatedSVD.fit_transform to fail when native library is unavailable."
        except RuntimeError as e:
            msg = str(e).lower()
            assert "native" in msg or "libdimreduce4gpu" in msg or "cuda" in msg
        return

    tsvd = TruncatedSVD(n_components=2, algorithm="power", n_iter=5)
    X_gpu = tsvd.fit_transform(X)

    X_ref, comps_ref = _svd_reference_tsvd(X.astype(np.float64), 2)

    assert X_gpu.shape == X_ref.shape
    for j in range(2):
        assert _corr_abs(X_gpu[:, j], X_ref[:, j]) > 0.99

    C = np.array(tsvd.components_, dtype=np.float64)
    assert C.shape == (2, 3)
    I = C @ C.T
    assert np.allclose(I, np.eye(2), atol=1e-2)
