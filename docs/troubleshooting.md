# Troubleshooting

## "Native library missing" / `native_built()` is false

- Build the shared library with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- Confirm the file exists:

```bash
ls -lh dimreduce4gpu/lib/libdimreduce4gpu.so
```

- If the library is somewhere else, set:

```bash
export DIMREDUCE4GPU_LIB_PATH=/path/to/directory/containing/libdimreduce4gpu.so
```

## "GPU not runnable" / `native_runnable()` is false

This means the `.so` loads, but CUDA can't initialize a device (common causes: no NVIDIA driver, no GPU).

- On Linux, ensure the NVIDIA driver is installed and `nvidia-smi` works.
- Ensure `libcuda.so.1` is available on the loader path.

## Missing CUDA runtime libraries (`libcublas`, `libcusolver`, `libcusparse`)

The CUDA shared library links against cuBLAS/cuSOLVER/cuSPARSE. Ensure your runtime environment has these libraries and they are discoverable by the dynamic loader:

- Common path: `/usr/local/cuda/lib64`
- Or set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```
