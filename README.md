# tsvd4gpu

Singular Value Decomposition(SVD) & Truncated SVD written in CUDA based on the following eigenvalue solvers:
[cusolverDnSsyevd dense eigenvalue solver](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-syevd)
 and the [power method](https://en.wikipedia.org/wiki/Power_iteration)

## Usage in other projects
* This library (or a variant of it) is used in the following projects:
    * [H2O4GPU](https://github.com/h2oai/h2o4gpu/tree/master)(A collection of GPU solvers by [H2O.ai](https://www.h2o.ai/) )
        * Usage:
            * [Truncated SVD](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
            * [Principal Components Analysis](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca) (Uses Truncated SVD for               most of the computational work)

## Simple install & test

`bash run_cmake.sh`

