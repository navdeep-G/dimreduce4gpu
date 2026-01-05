from __future__ import annotations

from setuptools import setup

# ---- Wheel tagging ----
# This package ships a prebuilt native shared library (libdimreduce4gpu.so) as package data.
# Without this override, setuptools/wheel may incorrectly mark the wheel as "pure" (py3-none-any).
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    _bdist_wheel = None


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

    cmdclass = {"bdist_wheel": bdist_wheel}
else:
    cmdclass = {}


setup(cmdclass=cmdclass)
