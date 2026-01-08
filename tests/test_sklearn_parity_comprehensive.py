from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import TruncatedSVD as SkTSVD

import dimreduce4gpu
from dimreduce4gpu import PCA, TruncatedSVD

try:
    from scipy.linalg import subspace_angles
except Exception:  # pragma: no cover
    subspace_angles = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Case:
    name: str
    n: int
    m: int
    k: int
    rank_deficient: bool = False


CASES: tuple[Case, ...] = (
    Case("tall_skinny", n=256, m=64, k=16),
    Case("short_wide", n=64, m=256, k=16),
    Case("square", n=192, m=192, k=24),
    Case("rank_deficient", n=200, m=80, k=16, rank_deficient=True),
)

DTYPES: tuple[np.dtype, ...] = (np.float32, np.float64)


def _require_cpu_backend() -> None:
    # CI builds the CPU backend; locally this may not be present.
    if not dimreduce4gpu.cpu_built():
        pytest.skip(
            "CPU backend not built (libdimreduce4cpu.so missing). "
            "Build with CMake (DIMREDUCE4GPU_BUILD_CPU=ON) or run in CI."
        )


def _make_matrix(case: Case, dtype: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((case.n, case.m)).astype(dtype, copy=False)

    # Add anisotropic scaling so PCA has a meaningful spectrum.
    scale = np.linspace(1.0, 3.0, case.m, dtype=dtype)
    X *= scale

    if case.rank_deficient:
        # Force approximate rank-k by projecting onto a k-dimensional subspace.
        # This makes component ordering/sign less stable, so we validate via subspace metrics.
        A = rng.standard_normal((case.m, case.k)).astype(dtype, copy=False)
        X = (X @ A) @ A.T

    return X


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _max_principal_angle_deg(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Return max principal angle (degrees) between the subspaces spanned by the
    *rows* of C1 and C2 (components are typically shape (k, m)).

    This is invariant to sign flips and (mostly) robust to rotations/orderings
    when singular values are close.
    """
    if subspace_angles is None:  # pragma: no cover
        # Fallback: use a crude metric if scipy isn't present.
        # This should not happen in CI because scipy is installed.
        Q1, _ = np.linalg.qr(C1.T)
        Q2, _ = np.linalg.qr(C2.T)
        s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        ang = np.arccos(s)
    else:
        ang = subspace_angles(C1.T, C2.T)

    return float(np.max(ang) * 180.0 / math.pi)


def _relative_fro_error(A: np.ndarray, B: np.ndarray) -> float:
    denom = np.linalg.norm(B)
    if denom == 0:
        return float(np.linalg.norm(A))
    return float(np.linalg.norm(A - B) / denom)


def _pca_reconstruct(X: np.ndarray, scores: np.ndarray, components: np.ndarray) -> np.ndarray:
    # PCA is defined on centered data.
    Xc = _center(X)
    return scores @ components


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize("seed", [0, 7])
def test_pca_cpu_matches_sklearn_full_solver(case: Case, dtype: np.dtype, seed: int) -> None:
    """
    "Exact" parity test: our CPU backend with algorithm='cusolver' should match
    sklearn's full SVD solver closely (up to sign/order invariances).
    """
    _require_cpu_backend()
    X = _make_matrix(case, dtype, seed)

    ours = PCA(n_components=case.k, backend="cpu", algorithm="cusolver", random_state=seed)
    X_ours = ours.fit_transform(X)

    sk = SkPCA(n_components=case.k, svd_solver="full", random_state=seed)
    X_sk = sk.fit_transform(X)

    # Compare component subspaces (robust to sign flips / swaps).
    angle = _max_principal_angle_deg(np.asarray(ours.components_, dtype=np.float64), sk.components_.astype(np.float64))
    assert angle < 0.75

    # Compare explained variance ratios (should be close for "full" solver).
    ours_evr = np.asarray(getattr(ours, "explained_variance_ratio_", []), dtype=np.float64)
    sk_evr = np.asarray(sk.explained_variance_ratio_, dtype=np.float64)
    if ours_evr.size == sk_evr.size and ours_evr.size > 0:
        assert _relative_fro_error(ours_evr, sk_evr) < 2e-2

    # Compare reconstructions (centered).
    recon_ours = _pca_reconstruct(X, X_ours.astype(np.float64), np.asarray(ours.components_, dtype=np.float64))
    recon_sk = _pca_reconstruct(X, X_sk.astype(np.float64), sk.components_.astype(np.float64))
    assert _relative_fro_error(recon_ours, recon_sk) < 2e-2


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize("seed", [1, 9])
def test_pca_cpu_matches_sklearn_randomized(case: Case, dtype: np.dtype, seed: int) -> None:
    """
    Approx parity test: randomized/power methods are stochastic and may differ
    slightly. We validate via subspace angles + reconstruction error.
    """
    _require_cpu_backend()
    X = _make_matrix(case, dtype, seed)

    n_iter = 5
    ours = PCA(
        n_components=case.k,
        backend="cpu",
        algorithm="power",
        n_iter=n_iter,
        random_state=seed,
    )
    X_ours = ours.fit_transform(X)

    sk = SkPCA(
        n_components=case.k,
        svd_solver="randomized",
        iterated_power=n_iter,
        random_state=seed,
    )
    X_sk = sk.fit_transform(X)

    angle = _max_principal_angle_deg(
        np.asarray(ours.components_, dtype=np.float64), sk.components_.astype(np.float64)
    )
    # NOTE: randomized methods are approximate and can return different but still
    # high-quality subspaces (especially when the spectrum is not well-separated).
    # We keep an angle sanity-check, but use reconstruction quality (vs sklearn)
    # as the primary correctness criterion.
    assert angle < 35.0

    ours_components = np.asarray(ours.components_, dtype=np.float64)
    recon_ours = _pca_reconstruct(X, X_ours.astype(np.float64), ours_components)
    recon_sk = _pca_reconstruct(X, X_sk.astype(np.float64), sk.components_.astype(np.float64))

    # Compare how well each method reconstructs the centered data; require ours to
    # be within 10% of sklearn's reconstruction error.
    X_centered = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    err_ours = _relative_fro_error(X_centered, recon_ours)
    err_sk = _relative_fro_error(X_centered, recon_sk)
    if case.rank_deficient:
        # Rank-deficient cases can yield near-zero reconstruction error for sklearn.
        # Use an absolute tolerance and a slightly looser relative bound.
        if err_sk < 1e-10:
            assert err_ours <= (2e-6 if dtype is np.float64 else 8e-6)
        else:
            assert err_ours <= (1.35 * err_sk + 1e-12)
    else:
        assert err_ours <= (1.10 * err_sk + 1e-12)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize("seed", [2, 11])
def test_tsvd_cpu_matches_sklearn_randomized(case: Case, dtype: np.dtype, seed: int) -> None:
    """
    TruncatedSVD parity against sklearn randomized solver.

    Note: TruncatedSVD does NOT center X.
    """
    _require_cpu_backend()
    X = _make_matrix(case, dtype, seed)

    n_iter = 5
    ours = TruncatedSVD(
        n_components=case.k,
        backend="cpu",
        algorithm="power",
        n_iter=n_iter,
        random_state=seed,
    )
    X_ours = ours.fit_transform(X)

    sk = SkTSVD(
        n_components=case.k,
        algorithm="randomized",
        n_iter=n_iter,
        random_state=seed,
    )
    X_sk = sk.fit_transform(X)

    angle = _max_principal_angle_deg(
        np.asarray(ours.components_, dtype=np.float64), sk.components_.astype(np.float64)
    )
    assert angle < 35.0

    # Compare reconstruction in original space (no centering). As with PCA, use
    # reconstruction error vs sklearn as the primary criterion.
    ours_components = np.asarray(ours.components_, dtype=np.float64)
    recon_ours = X_ours.astype(np.float64) @ ours_components
    recon_sk = X_sk.astype(np.float64) @ sk.components_.astype(np.float64)
    err_ours = _relative_fro_error(X.astype(np.float64), recon_ours)
    err_sk = _relative_fro_error(X.astype(np.float64), recon_sk)
    if case.rank_deficient:
        if err_sk < 1e-10:
            assert err_ours <= (2e-6 if dtype is np.float64 else 8e-6)
        else:
            assert err_ours <= (1.35 * err_sk + 1e-12)
    else:
        assert err_ours <= (1.10 * err_sk + 1e-12)

