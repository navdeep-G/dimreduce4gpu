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

### Diagnose native/GPU availability

This repo includes a small CLI to help debug native library loading and GPU runtime availability:

```bash
dimreduce4gpu-diagnose
dimreduce4gpu-diagnose --json
```

The two key checks are:
- `native_built()`: `.so` exists and can be `dlopen()`'d
- `native_runnable()`: NVIDIA driver initializes and at least one CUDA device is available

### GPU testing without GitHub GPU runners

GitHub-hosted runners do not provide GPUs by default. This repo therefore:
- compiles and verifies the `.so` on CPU-only CI (across multiple CUDA toolkit containers)
- runs true GPU correctness/benchmark jobs only on GPU-capable environments

See `docs/GPU_TESTING.md` for reproducible options (Docker, one-off cloud VM).

## Benchmarks

Benchmarks require a GPU-capable environment (where `native_runnable()` is true).

```bash
python bench/run_benchmarks.py --out bench-results.json
```

See `docs/GPU_TESTING.md` for suggested ways to run GPU tests/benchmarks without GitHub GPU runners.

## GPU testing without GitHub GPU runners

GitHub-hosted runners do not provide GPUs by default. This repo:

- builds the CUDA `.so` in CI using CUDA toolkit containers
- verifies it is structurally sound (deps resolve, exports exist, `dlopen` works)

For runtime GPU tests (`pytest` correctness checks, benchmarks), run on any machine with an NVIDIA GPU.
See `docs/GPU_TESTING.md`.

## Project hygiene

- CI: `.github/workflows/ci.yml`
- Contributing guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## Credits

The CUDA implementations are based on ideas from the `h2o4gpu` project:
- PCA module: https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca
