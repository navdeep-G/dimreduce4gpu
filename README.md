# `dimreduce4gpu`

**`dimreduce4gpu`** is a GPU-accelerated dimensionality reduction library built with CUDA, designed for fast and efficient large-scale data reduction. It provides implementations of popular algorithms like Principal Component Analysis (PCA) and Truncated Singular Value Decomposition (SVD), optimized to harness GPU power—making it ideal for high-performance applications in data science and machine learning.

---

## 🚀 Features

- **GPU-Accelerated**: Leverages CUDA to achieve significant speedups on large datasets.
- **Optimized Implementations**: Includes PCA and Truncated SVD tailored for high throughput and scale.
- **Python Integration**: Easily integrates into Python-based data workflows.

---

## 📌 Supported Algorithms

- **Principal Component Analysis (PCA)**  
  Reduces dimensionality by transforming variables into a set of linearly uncorrelated principal components.

- **Truncated Singular Value Decomposition (SVD)**  
  Approximates SVD by retaining only the most significant singular values, making it suitable for sparse and large-scale datasets.

---

## 🛠 Build Instructions

### 📋 Requirements

- **CUDA**: Up to version 9.0  
- **OS**: Linux (tested on Ubuntu 16.04)  
- **Compiler**: GCC 4.9+ and CMake  
- **Python**: Version 3.6

---

### 🐧 Setup on Ubuntu 16.04 (with `virtualenv`)

1. **Install Dependencies**:
   ```bash
   sudo apt-get -y --no-install-recommends install \
       python3.6 \
       python3.6-dev \
       virtualenv \
       python3-pip
   ```

2. **Create virtual environment**:
    ```bash
    virtualenv --python=python3.6 .venv
    source .venv/bin/activate
    pip install setuptools --no-cache-dir
    ```

3. **Configure CUDA Environment**:

   Add the following to your .bashrc or shell configuration file:
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH_MORE=/home/$USER/lib/:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH_MORE
    export CUDADIR=/usr/local/cuda/include/
    ```

5. **Clone and Build the Project:e**:
    ```bash
    git clone --recursive git@github.com:navdeep-G/dimreduce4gpu.git
    cd dimreduce4gpu
    virtualenv -p python3.6 env
    make
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

