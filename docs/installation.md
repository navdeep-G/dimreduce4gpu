# Installation

## Python package

For development:

```bash
python -m pip install -e .
```

This installs the Python wrappers. GPU functionality requires building the native CUDA library.

## Native CUDA library

### Prerequisites

- CUDA toolkit (CUDA 11/12 recommended)
- A compiler supported by your CUDA toolkit
- CMake >= 3.18

### Build

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build places:

- `dimreduce4gpu/lib/libdimreduce4gpu.so`

### Runtime library discovery

By default the package loads the shared library from `dimreduce4gpu/lib/`.
You can override the search path using:

```bash
export DIMREDUCE4GPU_LIB_PATH=/path/to/directory/containing/libdimreduce4gpu.so
```
