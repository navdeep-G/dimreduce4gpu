# dimreduce4gpu

`dimreduce4gpu` provides **GPU-accelerated** dimensionality reduction primitives:
- **PCA** (centers data)
- **TruncatedSVD** (does not center data)

The Python API calls into a CUDA shared library (`libdimreduce4gpu.so`). The repository includes a CMake
build for that native library.

## What CI covers

The default GitHub Actions workflow validates as much as possible without a GPU:

- **Linting & unit tests** on CPU-only runners
- **CUDA compilation** of `libdimreduce4gpu.so` (in a CUDA toolkit container)
- **Native artifact verification**: the `.so` exists, is a valid ELF shared library, dependencies resolve,
  required exported symbols are present, and the library can be `dlopen`'d.

GPU kernel execution (numerical correctness on-device) is validated via an **optional GPU-only workflow**
that you can run on a GPU runner.

## Quickstart (Python)

```python
import numpy as np
from dimreduce4gpu import PCA

X = np.random.default_rng(0).normal(size=(10_000, 128)).astype(np.float32)
pca = PCA(n_components=32)

# Requires the CUDA shared library to be built and available.
X2 = pca.fit_transform(X)
print(X2.shape)
```

If the native CUDA library is not available or the environment cannot run GPU code, you can check:

```python
import dimreduce4gpu

if not dimreduce4gpu.native_built():
    print("GPU library missing; build libdimreduce4gpu.so first.")
elif not dimreduce4gpu.native_runnable():
    print("GPU library is present, but CUDA driver/GPU is not available in this environment.")
```

## Install (Python-only)

```bash
python -m pip install -e .
```

This installs the Python wrappers. GPU functionality requires building the CUDA library (next section).

## Wheels (GPU)

This repo includes a release workflow that can build **platform-specific** wheels
containing `libdimreduce4gpu.so` under `dimreduce4gpu/lib/`.

Notes:

- These wheels are **not universal** (they are platform-tagged).
- They expect CUDA runtime libraries (`libcublas`, `libcusolver`, `libcusparse`) to
  be available at runtime (common path: `/usr/local/cuda/lib64`).
- GPU execution requires NVIDIA drivers and a CUDA-capable GPU.

## Build the CUDA library

### Prerequisites
- CUDA toolkit installed (CUDA 11/12 recommended; older versions may work)
- A C++ compiler supported by your CUDA toolkit
- CMake

### Build
```bash
rm -rf build
mkdir build
cd build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build places `libdimreduce4gpu.so` into:

```
dimreduce4gpu/lib/
```

You can also override the lookup path at runtime:

```bash
export DIMREDUCE4GPU_LIB_PATH=/path/to/directory/containing/libdimreduce4gpu.so
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format .
pytest
```

## Benchmarks

- `bench/benchmark_pca.py` is a minimal example.
- `bench/run_benchmarks.py` is a benchmark harness that writes JSON results.

There is an optional GPU-runner workflow (`.github/workflows/benchmarks.yml`) that runs benchmarks on a schedule.

## Releases and wheels

Tagging a release (e.g. `v0.1.0`) triggers a workflow that builds platform wheels containing the `.so` and attaches
them to the GitHub Release.

These wheels are **Linux x86_64** and require a compatible CUDA runtime environment (driver + cuBLAS/cuSOLVER/cuSPARSE).

## Project hygiene

- CI: `.github/workflows/ci.yml`
- Contributing guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## Credits

The CUDA implementations are based on ideas from the `h2o4gpu` project:
- PCA module: https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca
