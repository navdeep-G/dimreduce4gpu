# truncatedSvd

SVD & Truncated SVD written in CUDA based on the following eigenvalue solvers:
[cusolverDnSsyevd dense eigenvalue solver](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-syevd)
 & the [power method](https://en.wikipedia.org/wiki/Power_iteration)

## Usage in other projects
* This library is currently the `Truncated SVD` implementation in the [h2o4gpu project](https://github.com/h2oai/h2o4gpu/tree/master), which I actively contribute to.
    - [Link to Truncated SVD implementation in h2o4gpu](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd)
    - This library is also the computational backend for h2o4gpu's [PCA implementation](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca)

## Simple install & test

`bash run_cmake.sh`

