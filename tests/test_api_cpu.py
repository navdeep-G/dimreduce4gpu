import pytest

import dimreduce4gpu


def test_native_helpers_are_available():
    # Always runs (CPU-only CI included)
    assert isinstance(dimreduce4gpu.native_available(), bool)


def test_pca_and_tsvd_behavior_without_native():
    # These assertions run in all environments. They do not skip.
    if dimreduce4gpu.native_available():
        # If the native library is available, basic construction should work.
        pca = dimreduce4gpu.PCA(n_components=2)
        tsvd = dimreduce4gpu.TruncatedSVD(n_components=2)
        assert pca.n_components == 2
        assert tsvd.n_components == 2
    else:
        # On CPU-only machines, *using* the estimators should raise a clear error.
        # Construction can be pure-Python; the native library is required at fit/transform time.
        import numpy as np

        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)

        with pytest.raises(RuntimeError) as e1:
            dimreduce4gpu.PCA(n_components=2).fit_transform(X)
        msg1 = str(e1.value).lower()
        assert "native extension" in msg1 or "shared library" in msg1 or "cuda" in msg1

        with pytest.raises(RuntimeError) as e2:
            dimreduce4gpu.TruncatedSVD(n_components=2).fit_transform(X)
        msg2 = str(e2.value).lower()
        assert "native extension" in msg2 or "shared library" in msg2 or "cuda" in msg2
