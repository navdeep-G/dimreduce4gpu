# dimreduce4gpu

Dimensionality reduction (`dimreduce`) on GPUs (`4gpu`) 

## Building

### Build Environment

* **NOTE:** `dimreduce4gpu` is tested up to CUDA 9.2

* Linux machine w/ GCC4.9+ and CMake installed.

* Python 3.6.

For `virtualenv` and ubuntu 16.04:

```arma.header
apt-get -y --no-install-recommends  install \
    python3.6 \
    python3.6-dev \
    virtualenv \
    python3-pip
virtualenv --python=python3.6 .venv
pip install setuptools --no-cache-dir
. .venv/bin/activate
```

- Add to `.bashrc` or your own environment (e.g.):

```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH_MORE=/home/$USER/lib/:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH_MORE
export CUDADIR=/usr/local/cuda/include/
```

- To compile everything, install the Python interface, and run a simple test:

```
git clone --recursive git@github.com:navdeep-G/dimreduce4gpu.git
cd dimreduce4gpu
virtualenv -p python3.6 env
pip install -r requirements.txt
bash run_cmake.sh
```

## `dimreduce4gpu` usage in other projects
* This library (or a variant of it) is used in the following open source projects:
    * [H2O4GPU](https://github.com/h2oai/h2o4gpu/tree/master)(A collection of GPU solvers by [H2O.ai](https://www.h2o.ai/) )
        * Usage:
            * [Truncated SVD](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
            * [Principal Components Analysis](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca)

