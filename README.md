# dimreduce4gpu

`dimreduce4gpu` is a dimensionality reduction library optimized for GPUs, enabling fast and efficient data reduction through CUDA. This library includes popular algorithms, such as Principal Component Analysis (PCA) and Truncated Singular Value Decomposition (SVD), specifically adapted to leverage GPU acceleration, making it suitable for high-performance applications in data science and machine learning.

## Current Algorithms

- **Principal Component Analysis (PCA)**: A technique that reduces data dimensionality by transforming variables into a set of uncorrelated principal components.
- **Truncated SVD**: An approximation of SVD that focuses on the most significant singular values, useful for sparse and large-scale datasets.

## Building `dimreduce4gpu`

### Build Environment Requirements

To successfully build and run `dimreduce4gpu`, ensure your environment meets the following requirements:

- **CUDA Version**: Compatible up to CUDA 9.0
- **Operating System**: Linux
- **Compiler**: GCC 4.9+ with CMake
- **Python Version**: Python 3.6

#### Setup for Ubuntu 16.04 with `virtualenv`

1. **Install Python and required packages**:
    ```bash
    apt-get -y --no-install-recommends install \
        python3.6 \
        python3.6-dev \
        virtualenv \
        python3-pip
    virtualenv --python=python3.6 .venv
    pip install setuptools --no-cache-dir
    . .venv/bin/activate
    ```

2. **Configure environment variables**:
    Add the following lines to `.bashrc` or your environment configuration file to set up CUDA paths:
    ```bash
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH_MORE=/home/$USER/lib/:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH_MORE
    export CUDADIR=/usr/local/cuda/include/
    ```

3. **Compile the project and install the Python interface**:
    ```bash
    git clone --recursive git@github.com:navdeep-G/dimreduce4gpu.git
    cd dimreduce4gpu
    virtualenv -p python3.6 env
    make
    ```

## Usage of `dimreduce4gpu` in Other Projects

The `dimreduce4gpu` library is integrated into other open-source projects for GPU-accelerated data science solutions. For example:

- **[H2O4GPU](https://github.com/h2oai/h2o4gpu/tree/master)**: A GPU-optimized library of solvers by [H2O.ai](https://www.h2o.ai/) which includes implementations of dimensionality reduction methods:
  - **Truncated SVD**: [H2O4GPU SVD Implementation](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
  - **Principal Component Analysis**: [H2O4GPU PCA Implementation](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca)

---

For additional information or to contribute, feel free to submit an issue or a pull request on GitHub. Thank you for your interest in `dimreduce4gpu`!
