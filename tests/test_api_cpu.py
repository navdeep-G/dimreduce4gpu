from __future__ import annotations

import numpy as np
import pytest

import dimreduce4gpu
from dimreduce4gpu import PCA, TruncatedSVD


def test_auto_backend_runs_on_cpu_when_gpu_unavailable() -> None:
    """AUTO should pick CPU on CPU-only machines (common CI case)."""

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    X_pca = PCA(n_components=2, backend="auto").fit_transform(X)
    assert X_pca.shape == (4, 2)

    X_tsvd = TruncatedSVD(n_components=2, algorithm="power", backend="auto").fit_transform(X)
    assert X_tsvd.shape == (4, 2)


def test_force_gpu_raises_when_not_runnable() -> None:
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    if dimreduce4gpu.native_runnable():
        X_pca = PCA(n_components=2, backend="gpu").fit_transform(X)
        assert X_pca.shape == (4, 2)
        return

    with pytest.raises(RuntimeError) as excinfo:
        PCA(n_components=2, backend="gpu").fit_transform(X)

    msg = str(excinfo.value).lower()
    assert ("cuda" in msg) or ("native" in msg) or ("libdimreduce4gpu" in msg)
