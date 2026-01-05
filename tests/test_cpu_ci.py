import dimreduce4gpu


def test_import_works_on_cpu_ci():
    # Package should import even when the CUDA shared library isn't built.
    assert hasattr(dimreduce4gpu, "native_built")
    assert isinstance(dimreduce4gpu.native_built(), bool)


def test_require_native_message_is_clear():
    if dimreduce4gpu.native_built():
        return

    try:
        dimreduce4gpu.require_native()
        raise AssertionError("require_native() should raise when native library is missing")
    except RuntimeError as e:
        msg = str(e).lower()
        assert "native" in msg
        assert "cuda" in msg
