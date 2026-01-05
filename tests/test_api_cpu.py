import numpy as np
import pytest

import dimreduce4gpu


def test_native_helpers_are_available():
    # Always runs (CPU-only CI included)
    assert isinstance(dimreduce4gpu.native_built(), bool)
    assert isinstance(dimreduce4gpu.native_runnable(), bool)


def test_pca_and_tsvd_behavior_without_native_or_driver():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

    if not dimreduce4gpu.native_built():
        with pytest.raises(RuntimeError) as e1:
            dimreduce4gpu.PCA(n_components=2).fit_transform(X)
        msg1 = str(e1.value).lower()
        assert ("native" in msg1) or ("libdimreduce4gpu" in msg1) or ("cuda" in msg1)

        with pytest.raises(RuntimeError) as e2:
            dimreduce4gpu.TruncatedSVD(n_components=2).fit_transform(X)
        msg2 = str(e2.value).lower()
        assert ("native" in msg2) or ("libdimreduce4gpu" in msg2) or ("cuda" in msg2)
        return

    if not dimreduce4gpu.native_runnable():
        # Library can be dlopen()'d, but NVIDIA driver runtime isn't available.
        with pytest.raises(RuntimeError) as e1:
            dimreduce4gpu.PCA(n_components=2).fit_transform(X)
        msg1 = str(e1.value).lower()
        assert ("driver" in msg1) or ("libcuda" in msg1)

        with pytest.raises(RuntimeError) as e2:
            dimreduce4gpu.TruncatedSVD(n_components=2).fit_transform(X)
        msg2 = str(e2.value).lower()
        assert ("driver" in msg2) or ("libcuda" in msg2)
        return

    # Runnable path: construction and execution should work.
    pca = dimreduce4gpu.PCA(n_components=2)
    out = pca.fit_transform(X)
    assert out.shape[1] == 2

    tsvd = dimreduce4gpu.TruncatedSVD(n_components=2)
    out2 = tsvd.fit_transform(X)
    assert out2.shape[1] == 2
