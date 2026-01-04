# dimreduce4gpu

`dimreduce4gpu` provides **GPU-accelerated** dimensionality reduction primitives:
- **PCA** (centers data)
- **TruncatedSVD** (does not center data)

The Python API calls into a CUDA shared library (`libdimreduce4gpu.so`). The repository includes a CMake
build for that native library.

> **Note on CI:** the default GitHub Actions workflow runs on CPU-only runners, so it validates **linting**
> and **CPU-safe import/tests**. GPU build/runtime is intentionally not executed in CI.

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

If the native CUDA library is not available, you can check:

```python
import dimreduce4gpu

if not dimreduce4gpu.native_available():
    print("GPU library missing; build the CUDA library first.")
```

## Install (Python-only)

```bash
python -m pip install -e .
```

This installs the Python wrappers. GPU functionality requires building the CUDA library (next section).

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
cmake ..
make -j
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

See `bench/benchmark_pca.py` for a simple benchmark harness. It will skip if the native library is not built.

## Project hygiene

- CI: `.github/workflows/ci.yml`
- Contributing guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## Credits

The CUDA implementations are based on ideas from the `h2o4gpu` project:
- PCA module: https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca
