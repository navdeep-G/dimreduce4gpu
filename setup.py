"""Setuptools entrypoint.

Most configuration lives in ``setup.cfg`` / ``pyproject.toml``.

This project ships a native CUDA shared library (``libdimreduce4gpu.so``) as
*package data* under ``dimreduce4gpu/lib/``. Wheels must therefore be marked as
*non-pure* so installers don't treat them as universal.
"""

from __future__ import annotations

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that contains platform-specific binaries."""

    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
