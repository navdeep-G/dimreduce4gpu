# CPU backend implementation

`dimreduce4gpu` provides a high-performance **native CPU backend** (`libdimreduce4cpu.so`) so the library works even when CUDA/GPU execution isn't available.

This page explains what the CPU implementation is doing at a high level, and how it relates to the CUDA implementation.

## Backend selection

Both `PCA` and `TruncatedSVD` accept a `backend` argument:

- `backend="auto"` (default): use GPU if runnable, otherwise CPU
- `backend="cpu"`: force CPU backend
- `backend="gpu"`: force GPU backend (raises a clear error if unavailable)

## PCA on CPU

PCA is defined on **centered** data.

1. Center the input matrix:

   \[
   X_c = X - \mu, \quad \mu = \frac{1}{n}\sum_i X_i
   \]

2. Compute a truncated SVD of the centered matrix:

   \[
   X_c \approx U_k \Sigma_k V_k^T
   \]

3. The principal axes (components) are:

   \[
   \text{components} = V_k^T
   \]

4. The projected data (scores) returned by `fit_transform` is:

   \[
   Z = X_c V_k
   \]

### Solvers

The CPU backend supports two solver styles:

- **`algorithm="cusolver"`**: an *exact* dense SVD path (via LAPACK) for accuracy and for small/medium problems.
- **`algorithm="power"`**: a fast *approximate* solver based on randomized/power-iteration SVD.

The randomized/power approach is similar in spirit to GPU power-method solvers: it is usually much faster when `n_components << min(n_samples, n_features)` and the spectrum is well-behaved, while remaining very close to the exact solution.

## TruncatedSVD on CPU

`TruncatedSVD` matches scikit-learn semantics: **no centering** is performed.

It computes a truncated SVD:

\[
X \approx U_k \Sigma_k V_k^T
\]

and returns projected data:

\[
Z = X V_k
\]

## Why results can differ from CUDA / scikit-learn

Even when implementations are “mathematically the same”, you should expect small differences because of:

- different floating-point reduction orders
- different BLAS/LAPACK implementations (OpenBLAS vs MKL vs cuBLAS/cuSOLVER)
- sign ambiguity in singular vectors (\(v\) and \(-v\) are equivalent)
- component re-ordering when singular values are very close

For that reason, parity tests compare **subspaces** (principal angles) and reconstruction error rather than relying on exact equality.

## How correctness is tested

The test suite uses scikit-learn as the source of truth and validates:

- **subspace similarity** via principal angles (invariant to sign flips and many reorderings)
- **reconstruction closeness** (relative Frobenius error)
- explained-variance ratios when available

See: `tests/test_sklearn_parity_comprehensive.py`.
