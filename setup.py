from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Build the native CPU shared library as part of the Python build.

    This ensures `pip install .` and wheel builds produce a usable package
    without requiring users to run CMake manually.
    """

    def run(self):
        # Build the native CPU backend into the build output tree so it ends up
        # inside the wheel.
        build_lib = Path(self.build_lib)
        out_dir = build_lib / "dimreduce4gpu" / "lib"
        out_dir.mkdir(parents=True, exist_ok=True)

        src_dir = Path(__file__).resolve().parent
        build_dir = src_dir / "build" / "cpu"
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DDIMREDUCE4GPU_BUILD_CPU=ON",
            "-DDIMREDUCE4GPU_BUILD_CUDA=OFF",
            f"-DDIMREDUCE4GPU_OUTPUT_DIR={out_dir}",
        ]

        # Allow users to pass additional CMake args.
        extra = os.environ.get("DIMREDUCE4GPU_CMAKE_ARGS")
        if extra:
            cmake_args.extend(extra.split())

        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])

        super().run()


setup(
    name="dimreduce4gpu",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"dimreduce4gpu": ["lib/*.so", "lib/*.dll", "lib/*.dylib"]},
    cmdclass={"build_py": build_py},
)
