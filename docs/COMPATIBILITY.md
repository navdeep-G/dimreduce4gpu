# CUDA compatibility

## Toolkit vs driver

- Building the native library requires the CUDA **toolkit** (nvcc).
- Running the native library requires the NVIDIA **driver runtime** (`libcuda.so.1`) and a CUDA-capable GPU.

`dimreduce4gpu` exposes two helpful checks:

- `native_built()` — the `.so` exists and can be `dlopen()`'d.
- `native_runnable()` — the driver initializes and at least one CUDA device is available.

## CUDA versions in CI

This repo compiles (and verifies) the `.so` in CI across multiple CUDA *toolkit* containers to catch
breaking changes in CUDA/Thrust/header behavior early.

## GPU architectures

The build uses a configurable architecture list:

- CMake cache variable: `DIMREDUCE4GPU_CUDA_ARCHES`
- If `CMAKE_CUDA_ARCHITECTURES` is not set, it defaults to `DIMREDUCE4GPU_CUDA_ARCHES`.

Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDIMREDUCE4GPU_CUDA_ARCHES="70;75;80;86"
```

Choose architectures based on your target GPUs (e.g., 75 for T4, 86 for RTX 30xx, 89 for L4, 90 for H100).
