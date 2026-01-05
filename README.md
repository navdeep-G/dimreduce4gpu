# `dimreduce4gpu`

**`dimreduce4gpu`** is a GPU-accelerated dimensionality reduction library built with CUDA, designed for fast and efficient large-scale data reduction. It provides implementations of popular algorithms like Principal Component Analysis (PCA) and Truncated Singular Value Decomposition (SVD), optimized to harness GPU power—making it ideal for high-performance applications in data science and machine learning.

---

## 🚀 Features

- **GPU-Accelerated**: Leverages CUDA to achieve significant speedups on large datasets.
- **Optimized Implementations**: Includes PCA and Truncated SVD tailored for high throughput and scale.
- **Python Integration**: Easily integrates into Python-based data workflows.

## ✅ Modern builds and CI

- CPU-only installs are supported via a native C++ backend (`libdimreduce4cpu.*`).
- GPU acceleration uses the CUDA backend (`libdimreduce4gpu.*`) when available.
- GitHub Actions runs unit tests on CPU runners, and includes a build+verify job for
  the native libraries.
- A dedicated workflow builds manylinux CPU wheels: `.github/workflows/wheels.yml`.

### Backend selection

Both `PCA` and `TruncatedSVD` accept `backend`:

- `backend="auto"` (default): GPU if runnable, else CPU
- `backend="cpu"`: force CPU backend
- `backend="gpu"`: force GPU backend

---

## 📌 Supported Algorithms

- **Principal Component Analysis (PCA)**  
  Reduces dimensionality by transforming variables into a set of linearly uncorrelated principal components.

- **Truncated Singular Value Decomposition (SVD)**  
  Approximates SVD by retaining only the most significant singular values, making it suitable for sparse and large-scale datasets.

---

## 🛠 Build Instructions

### 📋 Requirements

- **Python**: 3.9+
- **Build tools**: CMake 3.18+, a C++17 compiler
- **CPU backend**: BLAS + LAPACK development headers (e.g., OpenBLAS)
- **GPU backend (optional)**: CUDA toolkit + NVIDIA driver/runtime

### Quickstart (CPU)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
pytest -q
```

### Building the native libraries (developers)

CPU-only build:

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release -DDIMREDUCE4GPU_BUILD_CPU=ON -DDIMREDUCE4GPU_BUILD_CUDA=OFF
cmake --build build/cpu -j
```

CUDA build (requires CUDA toolkit):

```bash
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release -DDIMREDUCE4GPU_BUILD_CPU=ON -DDIMREDUCE4GPU_BUILD_CUDA=ON
cmake --build build/cuda -j
```

## 📦 Integration in Other Projects

`dimreduce4gpu` is also part of other GPU-optimized machine learning ecosystems:

- **[H2O4GPU](https://github.com/h2oai/h2o4gpu)** by [H2O.ai](https://www.h2o.ai/)
  - 🔹 [Truncated SVD Module](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
  - 🔹 [PCA Module](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca)

---

## 🤝 Contributing

We welcome contributions! Feel free to:

- 🐛 [Open an issue](https://github.com/navdeep-G/dimreduce4gpu/issues) for bugs or feature requests
- 💬 Ask questions or share ideas
- 🔧 Submit pull requests to improve the project

Thank you for using **`dimreduce4gpu`**!



## CPU backend implementation

See `docs/CPU_BACKEND.md` for a detailed explanation of the CPU PCA/TruncatedSVD algorithms and how parity is tested against scikit-learn.

## Benchmarks

See `docs/BENCHMARKS.md` and `bench/benchmark_cpu_vs_sklearn.py` for CPU performance comparisons against scikit-learn.
