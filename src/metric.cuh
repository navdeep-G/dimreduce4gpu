#pragma once
#include "matrix.cuh"
#include "scl.h"


namespace scl
{
	scl_float pyksvd_metric(const Matrix<scl_float>& R);

	scl_float rmse_metric(const Matrix<scl_float>& R);
}
