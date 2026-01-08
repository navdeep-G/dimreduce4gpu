#pragma once

#include <cstdbool>
#include <cstdint>

#ifdef _WIN32
  #ifdef DIMREDUCE4CPU_EXPORTS
    #define DIMREDUCE4CPU_API __declspec(dllexport)
  #else
    #define DIMREDUCE4CPU_API __declspec(dllimport)
  #endif
#else
  #define DIMREDUCE4CPU_API
#endif

extern "C" {

// Mirror of Python-side params struct (dimreduce4gpu/lib_dimreduce4gpu.py).
struct params {
  int32_t X_n;
  int32_t X_m;
  int32_t k;
  const char* algorithm;
  int32_t n_iter;
  int32_t random_state;
  float tol;
  int32_t verbose;
  int32_t gpu_id;
  bool whiten;
};

DIMREDUCE4CPU_API void truncated_svd_float(
    const float* X,
    float* Q,
    float* w,
    float* U,
    float* X_transformed,
    float* explained_variance,
    float* explained_variance_ratio,
    params p);

DIMREDUCE4CPU_API void pca_float(
    const float* X,
    float* Q,
    float* w,
    float* U,
    float* X_transformed,
    float* explained_variance,
    float* explained_variance_ratio,
    float* mean,
    params p);

}  // extern "C"
