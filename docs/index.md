# dimreduce4gpu

`dimreduce4gpu` provides GPU-accelerated dimensionality reduction primitives:

- PCA
- TruncatedSVD

The Python API calls into a CUDA shared library (`libdimreduce4gpu.so`).

## CI coverage

The default CI validates:

- Linting (Ruff)
- Python unit tests (CPU-only)
- CUDA compilation of `libdimreduce4gpu.so` in a CUDA toolkit container
- Verification that the native library is structurally valid (ELF, deps resolve, expected symbols exist, `dlopen` works)

GPU runtime numerical correctness is validated via an optional GPU-runner workflow.
