#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <cblas.h>

extern "C" {
// LAPACK (Fortran) symbols
void sgesdd_(char* jobz, int* m, int* n, float* a, int* lda, float* s, float* u, int* ldu,
            float* vt, int* ldvt, float* work, int* lwork, int* iwork, int* info);

void sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* a, int* lda, float* s, float* u, int* ldu,
            float* vt, int* ldvt, float* work, int* lwork, int* info);

void sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
void sorgqr_(int* m, int* n, int* k, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
}

namespace {

inline bool str_eq(const char* a, const char* b) {
  if (!a || !b) return false;
  return std::string(a) == std::string(b);
}

struct SVDResult {
  // Column-major:
  // U: n x k (ldu=n), S: k, VT: k x m (ldvt=k)
  std::vector<float> U;
  std::vector<float> S;
  std::vector<float> VT;
  int n = 0;
  int m = 0;
  int k = 0;
};

std::vector<float> to_col_major(const float* X_row, int n, int m) {
  std::vector<float> X_col(static_cast<size_t>(n) * static_cast<size_t>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      X_col[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] = X_row[i * m + j];
    }
  }
  return X_col;
}

void compute_mean_center_colmajor(const float* X_row, int n, int m, float* mean_out,
                                 std::vector<float>& Xc_col) {
  std::vector<double> mean_d(static_cast<size_t>(m), 0.0);
  for (int j = 0; j < m; ++j) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(X_row[i * m + j]);
    mean_d[j] = acc / static_cast<double>(n);
    mean_out[j] = static_cast<float>(mean_d[j]);
  }

  Xc_col.assign(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0f);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      Xc_col[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
          static_cast<float>(static_cast<double>(X_row[i * m + j]) - mean_d[j]);
    }
  }
}

// Exact SVD on X_col (column-major, lda=n). Returns top-k.
SVDResult exact_svd_topk_colmajor(const float* X_col_in, int n, int m, int k) {
  const int min_nm = std::min(n, m);
  const int kk = std::min(k, min_nm);

  std::vector<float> A(static_cast<size_t>(n) * static_cast<size_t>(m));
  std::copy(X_col_in, X_col_in + static_cast<size_t>(n) * static_cast<size_t>(m), A.begin());

  std::vector<float> s(static_cast<size_t>(min_nm));
  std::vector<float> Ufull(static_cast<size_t>(n) * static_cast<size_t>(min_nm));
  std::vector<float> VTfull(static_cast<size_t>(min_nm) * static_cast<size_t>(m));

  // Workspace query for sgesdd
  char jobz = 'S';
  int M = n, N = m, lda = n, ldu = n, ldvt = min_nm, info = 0;
  int lwork = -1;
  float wkopt = 0.0f;
  std::vector<int> iwork(static_cast<size_t>(8) * static_cast<size_t>(min_nm));
  sgesdd_(&jobz, &M, &N, A.data(), &lda, s.data(), Ufull.data(), &ldu, VTfull.data(), &ldvt,
          &wkopt, &lwork, iwork.data(), &info);
  lwork = static_cast<int>(wkopt);
  std::vector<float> work(static_cast<size_t>(std::max(1, lwork)));

  sgesdd_(&jobz, &M, &N, A.data(), &lda, s.data(), Ufull.data(), &ldu, VTfull.data(), &ldvt,
          work.data(), &lwork, iwork.data(), &info);

  if (info != 0) {
    // Fall back to sgesvd
    char jobu = 'S';
    char jobvt = 'S';
    int lwork2 = -1;
    float wkopt2 = 0.0f;
    sgesvd_(&jobu, &jobvt, &M, &N, A.data(), &lda, s.data(), Ufull.data(), &ldu, VTfull.data(),
            &ldvt, &wkopt2, &lwork2, &info);
    lwork2 = static_cast<int>(wkopt2);
    std::vector<float> work2(static_cast<size_t>(std::max(1, lwork2)));
    sgesvd_(&jobu, &jobvt, &M, &N, A.data(), &lda, s.data(), Ufull.data(), &ldu, VTfull.data(),
            &ldvt, work2.data(), &lwork2, &info);
    if (info != 0) return {};
  }

  SVDResult out;
  out.n = n;
  out.m = m;
  out.k = kk;
  out.S.assign(s.begin(), s.begin() + kk);

  // Copy U (n x kk) column-major (ldu=n)
  out.U.assign(static_cast<size_t>(n) * static_cast<size_t>(kk), 0.0f);
  for (int j = 0; j < kk; ++j) {
    std::copy(Ufull.begin() + static_cast<size_t>(j) * static_cast<size_t>(n),
              Ufull.begin() + static_cast<size_t>(j + 1) * static_cast<size_t>(n),
              out.U.begin() + static_cast<size_t>(j) * static_cast<size_t>(n));
  }

  // Copy VT (kk x m) column-major (ldvt=min_nm)
  out.VT.assign(static_cast<size_t>(kk) * static_cast<size_t>(m), 0.0f);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < kk; ++i) {
      out.VT[static_cast<size_t>(j) * static_cast<size_t>(kk) + static_cast<size_t>(i)] =
          VTfull[static_cast<size_t>(j) * static_cast<size_t>(min_nm) + static_cast<size_t>(i)];
    }
  }
  return out;
}

// Randomized SVD on X_col (column-major, lda=n). Returns top-k.

static bool ortho_qr_inplace(std::vector<float>& A, int n, int l) {
  // Orthonormalize A (n x l, column-major) in-place using QR.
  int M = n;
  int N = l;
  int K = std::min(M, N);
  int lda = n;
  int info = 0;
  std::vector<float> tau(static_cast<size_t>(std::max(1, K)));
  int lwork = -1;
  float wkopt = 0.0f;
  sgeqrf_(&M, &N, A.data(), &lda, tau.data(), &wkopt, &lwork, &info);
  if (info != 0) return false;
  lwork = static_cast<int>(wkopt);
  std::vector<float> work(static_cast<size_t>(std::max(1, lwork)));
  sgeqrf_(&M, &N, A.data(), &lda, tau.data(), work.data(), &lwork, &info);
  if (info != 0) return false;

  int lwork2 = -1;
  float wkopt2 = 0.0f;
  sorgqr_(&M, &N, &K, A.data(), &lda, tau.data(), &wkopt2, &lwork2, &info);
  if (info != 0) return false;
  lwork2 = static_cast<int>(wkopt2);
  work.assign(static_cast<size_t>(std::max(1, lwork2)), 0.0f);
  sorgqr_(&M, &N, &K, A.data(), &lda, tau.data(), work.data(), &lwork2, &info);
  if (info != 0) return false;
  return true;
}

SVDResult randomized_svd_topk_colmajor(const float* X_col, int n, int m, int k, int n_iter, int random_state) {
  const int min_nm = std::min(n, m);
  const int kk = std::min(k, min_nm);
  const int oversample = 10;
  const int l = std::min(kk + oversample, min_nm);

  std::mt19937 rng(static_cast<uint32_t>(random_state <= 0 ? 12345 : random_state));
  std::normal_distribution<float> nd(0.0f, 1.0f);

  // Omega: m x l (column-major, ld=m)
  std::vector<float> Omega(static_cast<size_t>(m) * static_cast<size_t>(l));
  for (auto& v : Omega) v = nd(rng);

  // Y = X * Omega => n x l (column-major, ld=n)
  std::vector<float> Y(static_cast<size_t>(n) * static_cast<size_t>(l), 0.0f);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, l, m, 1.0f, X_col, n, Omega.data(), m, 0.0f, Y.data(), n);

  // Power iterations: Y = (X X^T)^q X Omega
  for (int it = 0; it < std::max(0, n_iter); ++it) {
    // Z = X^T Y => m x l
    std::vector<float> Z(static_cast<size_t>(m) * static_cast<size_t>(l), 0.0f);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, l, n, 1.0f, X_col, n, Y.data(), n, 0.0f, Z.data(), m);
    // Y = X Z => n x l
    std::fill(Y.begin(), Y.end(), 0.0f);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, l, m, 1.0f, X_col, n, Z.data(), m, 0.0f, Y.data(), n);
  
    // Normalize to improve numerical stability (similar to sklearn's power_iteration_normalizer).
    if (!ortho_qr_inplace(Y, n, l)) return {};
}

  // QR factorization of Y to get Q (n x l) in Y
  std::vector<float> tau(static_cast<size_t>(l));
  int M = n, N = l, lda = n, info = 0;
  int lwork = -1;
  float wkopt = 0.0f;
  sgeqrf_(&M, &N, Y.data(), &lda, tau.data(), &wkopt, &lwork, &info);
  lwork = static_cast<int>(wkopt);
  std::vector<float> work(static_cast<size_t>(std::max(1, lwork)));
  sgeqrf_(&M, &N, Y.data(), &lda, tau.data(), work.data(), &lwork, &info);
  if (info != 0) return {};

  int K = l;
  lwork = -1;
  wkopt = 0.0f;
  sorgqr_(&M, &N, &K, Y.data(), &lda, tau.data(), &wkopt, &lwork, &info);
  lwork = static_cast<int>(wkopt);
  work.assign(static_cast<size_t>(std::max(1, lwork)), 0.0f);
  sorgqr_(&M, &N, &K, Y.data(), &lda, tau.data(), work.data(), &lwork, &info);
  if (info != 0) return {};
  // Q is now in Y (n x l), ld=n

  // B = Q^T X => l x m (column-major, ld=l)
  std::vector<float> B(static_cast<size_t>(l) * static_cast<size_t>(m), 0.0f);
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, l, m, n, 1.0f, Y.data(), n, X_col, n, 0.0f, B.data(), l);

  // SVD of B (l x m), get Uhat (l x l), VT (l x m)
  std::vector<float> s(static_cast<size_t>(l));
  std::vector<float> Uhat(static_cast<size_t>(l) * static_cast<size_t>(l));
  std::vector<float> VTfull(static_cast<size_t>(l) * static_cast<size_t>(m));

  char jobz = 'S';
  int Mb = l, Nb = m, ldab = l, ldu = l, ldvt = l, info2 = 0;
  int lwork2 = -1;
  float wkopt2 = 0.0f;
  std::vector<int> iwork(static_cast<size_t>(8) * static_cast<size_t>(l));
  sgesdd_(&jobz, &Mb, &Nb, B.data(), &ldab, s.data(), Uhat.data(), &ldu, VTfull.data(), &ldvt,
          &wkopt2, &lwork2, iwork.data(), &info2);
  lwork2 = static_cast<int>(wkopt2);
  std::vector<float> work2(static_cast<size_t>(std::max(1, lwork2)));
  sgesdd_(&jobz, &Mb, &Nb, B.data(), &ldab, s.data(), Uhat.data(), &ldu, VTfull.data(), &ldvt,
          work2.data(), &lwork2, iwork.data(), &info2);

  if (info2 != 0) {
    // fall back to sgesvd
    char jobu = 'S';
    char jobvt = 'S';
    int lwork3 = -1;
    float wkopt3 = 0.0f;
    sgesvd_(&jobu, &jobvt, &Mb, &Nb, B.data(), &ldab, s.data(), Uhat.data(), &ldu, VTfull.data(),
            &ldvt, &wkopt3, &lwork3, &info2);
    lwork3 = static_cast<int>(wkopt3);
    std::vector<float> work3(static_cast<size_t>(std::max(1, lwork3)));
    sgesvd_(&jobu, &jobvt, &Mb, &Nb, B.data(), &ldab, s.data(), Uhat.data(), &ldu, VTfull.data(),
            &ldvt, work3.data(), &lwork3, &info2);
    if (info2 != 0) return {};
  }

  // Uapprox = Q * Uhat_k => n x kk
  std::vector<float> Uapprox(static_cast<size_t>(n) * static_cast<size_t>(kk), 0.0f);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, kk, l, 1.0f, Y.data(), n, Uhat.data(), l, 0.0f, Uapprox.data(), n);

  SVDResult out;
  out.n = n;
  out.m = m;
  out.k = kk;
  out.U = std::move(Uapprox);
  out.S.assign(s.begin(), s.begin() + kk);

  // VTfull is l x m (ldvt=l). Copy first kk rows into out.VT (kk x m, ldvt=kk)
  out.VT.assign(static_cast<size_t>(kk) * static_cast<size_t>(m), 0.0f);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < kk; ++i) {
      out.VT[static_cast<size_t>(j) * static_cast<size_t>(kk) + static_cast<size_t>(i)] =
          VTfull[static_cast<size_t>(j) * static_cast<size_t>(l) + static_cast<size_t>(i)];
    }
  }

  return out;
}

void compute_explained_variance_rowmajor(const float* X_row, int n, int m, const float* s, int k,
                                        float* explained_variance, float* explained_variance_ratio) {
  const double denom = std::max(1, n - 1);
  for (int i = 0; i < k; ++i) {
    explained_variance[i] = static_cast<float>((static_cast<double>(s[i]) * static_cast<double>(s[i])) / denom);
  }
  double total_var = 0.0;
  for (int j = 0; j < m; ++j) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(X_row[i * m + j]);
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
      const double d = static_cast<double>(X_row[i * m + j]) - mean;
      var += d * d;
    }
    var /= denom;
    total_var += var;
  }
  if (total_var <= 0.0) total_var = 1.0;
  for (int i = 0; i < k; ++i) {
    explained_variance_ratio[i] = static_cast<float>(static_cast<double>(explained_variance[i]) / total_var);
  }
}

void fill_outputs_rowmajor(const SVDResult& svd, float* Q_row, float* w_out, float* U_row, float* X_transformed_row) {
  const int n = svd.n;
  const int m = svd.m;
  const int k = svd.k;

  // w
  std::copy(svd.S.begin(), svd.S.end(), w_out);

  // Q (components) in row-major: k x m.
  // svd.VT is column-major (ldvt=k), shape k x m.
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < m; ++j) {
      Q_row[i * m + j] = svd.VT[static_cast<size_t>(j) * static_cast<size_t>(k) + static_cast<size_t>(i)];
    }
  }

  // U in row-major: n x k.
  // svd.U is column-major (ldu=n), shape n x k.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      U_row[i * k + j] = svd.U[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)];
    }
  }

  // X_transformed = U * diag(w) (row-major n x k)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      X_transformed_row[i * k + j] = U_row[i * k + j] * w_out[j];
    }
  }
}

}  // namespace

extern "C" {

void truncated_svd_float(const float* X, float* Q, float* w, float* U, float* X_transformed,
                         float* explained_variance, float* explained_variance_ratio, params p) {
  const int n = p.X_n;
  const int m = p.X_m;
  const int k = std::min(p.k, std::min(n, m));
  if (!X || !Q || !w || !U || !X_transformed) return;

  std::vector<float> X_col = to_col_major(X, n, m);

  const bool use_exact = str_eq(p.algorithm, "cusolver") || (std::min(n, m) <= 256);
  SVDResult svd = use_exact ? exact_svd_topk_colmajor(X_col.data(), n, m, k)
                            : randomized_svd_topk_colmajor(X_col.data(), n, m, k, p.n_iter, p.random_state);
  if (svd.U.empty() || svd.S.empty() || svd.VT.empty()) return;

  fill_outputs_rowmajor(svd, Q, w, U, X_transformed);

  if (explained_variance && explained_variance_ratio) {
    compute_explained_variance_rowmajor(X, n, m, w, k, explained_variance, explained_variance_ratio);
  }
}

void pca_float(const float* X, float* Q, float* w, float* U, float* X_transformed,
               float* explained_variance, float* explained_variance_ratio, float* mean, params p) {
  const int n = p.X_n;
  const int m = p.X_m;
  const int k = std::min(p.k, std::min(n, m));
  if (!X || !Q || !w || !U || !X_transformed || !mean) return;

  std::vector<float> Xc_col;
  compute_mean_center_colmajor(X, n, m, mean, Xc_col);

  const bool use_exact = str_eq(p.algorithm, "cusolver") || (std::min(n, m) <= 256);
  SVDResult svd = use_exact ? exact_svd_topk_colmajor(Xc_col.data(), n, m, k)
                            : randomized_svd_topk_colmajor(Xc_col.data(), n, m, k, p.n_iter, p.random_state);
  if (svd.U.empty() || svd.S.empty() || svd.VT.empty()) return;

  fill_outputs_rowmajor(svd, Q, w, U, X_transformed);

  if (explained_variance && explained_variance_ratio) {
    // PCA uses centered data for variance
    // Build a centered row-major view on the fly for total variance calculation
    std::vector<float> Xc_row(static_cast<size_t>(n) * static_cast<size_t>(m));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        Xc_row[i * m + j] = X[i * m + j] - mean[j];
      }
    }
    compute_explained_variance_rowmajor(Xc_row.data(), n, m, w, k, explained_variance, explained_variance_ratio);
  }
}

}  // extern "C"
