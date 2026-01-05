from __future__ import annotations

import dimreduce4gpu


def test_require_native_message_is_clear() -> None:
    """require_native should exist for backwards compatibility.

    In CPU-only CI, CUDA native library will not be present, so it should raise
    a clear RuntimeError (not AttributeError).
    """

    if dimreduce4gpu.native_built():
        # If CUDA library is present, require_native should not raise.
        dimreduce4gpu.require_native()
        return

    try:
        dimreduce4gpu.require_native()
    except RuntimeError as e:
        msg = str(e).lower()
        assert ("cuda" in msg) or ("libdimreduce4gpu" in msg) or ("native" in msg)
    else:
        raise AssertionError("Expected require_native() to raise when CUDA backend is unavailable")
