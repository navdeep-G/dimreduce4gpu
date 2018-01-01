# truncatedSvd

SVD & Truncated SVD written in CUDA based on the [cusolverDnSsyevd dense eigenvalue solver](http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-syevd) and the [power method](https://en.wikipedia.org/wiki/Power_iteration)

# Usage in other projects
* This implementation is currently being used by the [h2o4gpu project](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/tsvd), which I actively contribute to.
* In addition, it is the backbone of the [PCA implementation](https://github.com/h2oai/h2o4gpu/tree/master/src/gpu/pca) in the h2o4gpu project.

# Simple install & test

`bash run_cmake.sh`

# Benchmarks coming soon...

