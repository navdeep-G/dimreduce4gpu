#pragma once
#include "cublas_v2.h"
#include "utils.cuh"
#include <cusparse.h>
#include <cusolverDn.h>

namespace device
{
	class DeviceContext
	{
	public:
		cublasHandle_t cublas_handle;
		cusolverDnHandle_t cusolver_handle;
		cusparseHandle_t cusparse_handle;
		util::CubMemory cub_mem;

		DeviceContext()
		{
			util::safe_cublas(cublasCreate(&cublas_handle));
			util::safe_cusolver(cusolverDnCreate(&cusolver_handle));
			util::safe_cusparse(cusparseCreate(&cusparse_handle));
		}

		~DeviceContext()
		{
			util::safe_cublas(cublasDestroy(cublas_handle));
			util::safe_cusolver(cusolverDnDestroy(cusolver_handle));
			util::safe_cusparse(cusparseDestroy(cusparse_handle));
		}
	};
}
