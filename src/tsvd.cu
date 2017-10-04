#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "matrix.cuh"
#include "device_context.cuh"
#include <tsvd.h>
#include <ctime>
#include <thrust/iterator/counting_iterator.h>
#include<algorithm>

namespace tsvd
{

//Calculate U, which is:
// U = A*V/sigma where A is our X Matrix, V is Qt, and sigma is 1/w_i
void calculate_u(const Matrix<float> &X, const Matrix<float> &Qt, const Matrix<float> &w, Matrix<float> &U, DeviceContext &context){

	multiply(X, Qt, U, context, false, true, 1.0f); //A*V
	auto d_u = U.data();
	auto d_sigma = w.data();
	auto column_size = U.rows();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+U.size(), [=]__device__(int idx){
		int column = idx/column_size;
		float sigma = d_sigma[column];
		float u = d_u[idx];
		if(sigma != 0.0){
			d_u[idx] = u * 1.0/sigma;
		} else{
			d_u[idx] = 0.0;
		}
	} );

}


void truncated_svd(const double* _X, double* _Q, double* _w, double* _U, params _param)
{
	try
	{
		Matrix<float>X(_param.X_m, _param.X_n);
		X.copy(_X);

		Matrix<float>XtX(_param.X_n, _param.X_n);

		//create context
		DeviceContext context;
		multiply(X, X, XtX, context, true, false, 1.0f);

		Matrix<float>Q(XtX.rows(), XtX.columns());
		Matrix<float>w(Q.rows(), 1);

		calculate_eigen_pairs_exact(XtX, Q, w, context);
		Matrix<float>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context);
		Qt.copy_to_host(_Q); //Send to host

		w.transform([=]__device__(float elem){
			if(elem > 0.0){
				return std::sqrt(elem);
			}else{
				return 0.0f;
			}
		}
		);
		std::vector<double> w_temp(w.size());
		w.copy_to_host(w_temp.data()); //Send to host
		std::reverse(w_temp.begin(), w_temp.end());
		std::copy(w_temp.begin(), w_temp.begin() + _param.k, _w);

		//Get U matrix
		Matrix<float>U(X.columns(), X.columns());
		calculate_u(X, Qt, w, U, context);
		U.copy_to_host(_U); //Send to host

		}
		catch (std::exception e)
		{
			std::cerr << "tsvd error: " << e.what() << "\n";
		}
		catch (std::string e)
		{
			std::cerr << "tsvd error: " << e << "\n";
		}
		catch (...)
		{
			std::cerr << "tsvd error\n";
		}
	}

}
