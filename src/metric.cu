#include "metric.cuh"
#include "utils.cuh"
#include "matrix.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

namespace scl
{

	scl_float pyksvd_metric(const Matrix<scl_float>& R)
	{
		auto dptr = thrust::device_pointer_cast(R.data());

		scl_float sum_square = thrust::transform_reduce(dptr, dptr + R.size(), sqr_op(),
		                                            0.0, thrust::plus<scl_float>());
		scl_float f_norm = std::sqrt(sum_square);
		return f_norm / R.columns();
	}

	scl_float rmse_metric(const Matrix<scl_float>& R)
	{
		auto dptr = thrust::device_pointer_cast(R.data());

		scl_float MSE = thrust::transform_reduce(dptr, dptr + R.size(), sqr_op(),
		                                     0.0, thrust::plus<scl_float>()) / R.size();
		return std::sqrt(MSE);
	}
}
