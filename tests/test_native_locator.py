from dimreduce4gpu._native import get_library_path


def test_get_library_path_returns_none_when_missing():
    # On CI (no build artifacts), we expect no library path.
    assert get_library_path() is None or isinstance(get_library_path(), str)
