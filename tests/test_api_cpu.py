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
        # On CPU-only machines, construction should raise a clear error.
        with pytest.raises(RuntimeError) as e1:
            dimreduce4gpu.PCA(n_components=2)
        msg1 = str(e1.value).lower()
        assert "native extension" in msg1 or "shared library" in msg1 or "cuda" in msg1

        with pytest.raises(RuntimeError) as e2:
            dimreduce4gpu.TruncatedSVD(n_components=2)
        msg2 = str(e2.value).lower()
        assert "native extension" in msg2 or "shared library" in msg2 or "cuda" in msg2
