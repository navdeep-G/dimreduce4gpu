# truncatedSvd

Singular Value Decomposition(SVD) & Truncated SVD written in CUDA based on the following eigenvalue solvers:
[cusolverDnSsyevd dense eigenvalue solver](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-syevd)
 & the [power method](https://en.wikipedia.org/wiki/Power_iteration)

## Usage in other projects
* This library is used in the following projects:
    * [h2o4gpu](https://github.com/h2oai/h2o4gpu/tree/master)
        * Usage:
            * [Truncated SVD](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
            * [Principal Components Analysis](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca) (Uses Truncated SVD for               most of the computational work)

## Simple install & test

`bash run_cmake.sh`

