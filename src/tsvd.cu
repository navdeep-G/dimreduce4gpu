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

void divide(const Matrix<float> &XVar, const Matrix<float> &XVarSum, Matrix<float> &ExplainedVarRatio, DeviceContext &context){
	auto d_x_var = XVar.data();
	auto d_x_var_sum = XVarSum.data();
	auto d_expl_var_ratio = ExplainedVarRatio.data();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+ExplainedVarRatio.size(), [=]__device__(int idx){
		float div_val = d_x_var[idx] / d_x_var_sum[0];
		d_expl_var_ratio[idx] = div_val;
	} );
}

void square_val(const Matrix<float> &UmultSigma, Matrix<float> &UmultSigmaSquare, DeviceContext &context){
	auto n = UmultSigma.columns();
	auto m = UmultSigma.rows();
	auto k = UmultSigmaSquare.rows();
	auto d_u_mult_sigma = UmultSigma.data();
	auto d_u_mult_sigma_square = UmultSigmaSquare.data();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+UmultSigmaSquare.size(), [=]__device__(int idx){
		float square_val = std::pow(d_u_mult_sigma[idx],2);
		d_u_mult_sigma_square[idx] = square_val;
	} );
}

void calc_var(const Matrix<float>UmultSigma, Matrix<float> &UmultSigmaVar, int k, DeviceContext &context){
	//Set aside matrix of 1's for getting columnar sums(t(UmultSima) * UmultOnes)
	Matrix<float>UmultOnes(UmultSigma.rows(), 1);
	UmultOnes.fill(1.0f);

	//Allocate matrices for variance calculation
	Matrix<float>UmultSigmaSquare(UmultSigma.rows(), UmultSigma.columns());
	Matrix<float>UmultSigmaSum(k, 1);
	Matrix<float>UmultSigmaSumSquare(k, 1);
	Matrix<float>UmultSigmaSumOfSquare(k, 1);
	Matrix<float>UmultSigmaVarNum(k, 1);

	//Calculate Variance
	square_val(UmultSigma, UmultSigmaSquare, context);
	multiply(UmultSigmaSquare, UmultOnes, UmultSigmaSumOfSquare, context, true, false, 1.0f);
	multiply(UmultSigma, UmultOnes, UmultSigmaSum, context, true, false, 1.0f);
	square_val(UmultSigmaSum, UmultSigmaSumSquare, context);
	//Get rows
	auto m = UmultSigma.rows();
	multiply(UmultSigmaSumOfSquare, m, context);
	subtract(UmultSigmaSumOfSquare, UmultSigmaSumSquare, UmultSigmaVarNum, context);
	auto d_u_sigma_var_num = UmultSigmaVarNum.data();
	auto d_u_sigma_var = UmultSigmaVar.data();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+UmultSigmaVar.size(), [=]__device__(int idx){
		float div_val = d_u_sigma_var_num[idx]/(std::pow(m,2));
		d_u_sigma_var[idx] = div_val;
	} );
}

void col_reverse_q(const Matrix<float> &Q, Matrix<float> &QReversed, DeviceContext &context){
	auto n = Q.columns();
	auto m = Q.rows();
	auto k = QReversed.rows();
	auto d_q = Q.data();
	auto d_q_reversed = QReversed.data();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+QReversed.size(), [=]__device__(int idx){
		int dest_row = idx % m;
		int dest_col = idx/m;
		int src_row = dest_row;
		int src_col = (n - dest_col) - 1;
		d_q_reversed[idx] = d_q[src_col * m + src_row];
	} );
}

// Truncated Q to k vectors (truncated svd)
void row_reverse_trunc_q(const Matrix<float> &Qt, Matrix<float> &QtTrunc, DeviceContext &context){
	auto m = Qt.rows();
	auto k = QtTrunc.rows();
	auto d_q = Qt.data();
	auto d_q_trunc = QtTrunc.data();
	auto counting = thrust::make_counting_iterator <int>(0);
	thrust::for_each(counting, counting+QtTrunc.size(), [=]__device__(int idx){
		int dest_row = idx % k;
		int dest_col = idx / k;
		int src_row = (m - dest_row) - 1;
		int src_col = dest_col;
		float q = d_q[src_col * m + src_row];
		d_q_trunc[idx] = q;
	} );
}

// Calculate U, which is:
// U = A*V/sigma where A is our X Matrix, V is Q, and sigma is 1/w_i
void calculate_u(const Matrix<float> &X, const Matrix<float> &Q, const Matrix<float> &w, Matrix<float> &U, DeviceContext &context){
	multiply(X, Q, U, context, false, false, 1.0f); //A*V
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

void truncated_svd(const double* _X, double* _Q, double* _w, double* _U, double* _explained_variance, double* _explained_variance_ratio, params _param)
{
	try
	{
		//Take in X matrix and allocate for X^TX
		Matrix<float>X(_param.X_m, _param.X_n);
		X.copy(_X);
		Matrix<float>XtX(_param.X_n, _param.X_n);

		//create context
		DeviceContext context;

		//Multiply X and Xt and output result to XtX
		multiply(X, X, XtX, context, true, false, 1.0f);

		//Set up Q (V^T) and w (singular value) matrices (w is a matrix of size Q.rows() by 1; really just a vector
		Matrix<float>Q(XtX.rows(), XtX.columns()); // n X n -> V^T
		Matrix<float>w(Q.rows(), 1);
		calculate_eigen_pairs_exact(XtX, Q, w, context);

		//Obtain Q^T to obtain vector as row major order
		Matrix<float>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context); //Needed for calculate_u()
		Matrix<float>QtTrunc(_param.k, Qt.columns());
		row_reverse_trunc_q(Qt, QtTrunc, context);
		QtTrunc.copy_to_host(_Q); //Send to host

		//Obtain square root of eigenvalues, which are singular values
		w.transform([=]__device__(float elem){
			if(elem > 0.0){
				return std::sqrt(elem);
			}else{
				return 0.0f;
			}
		}
		);

		//Sort from biggest singular value to smallest
		std::vector<double> w_temp(w.size());
		w.copy_to_host(w_temp.data()); //Send to host
		std::reverse(w_temp.begin(), w_temp.end());
		std::copy(w_temp.begin(), w_temp.begin() + _param.k, _w);
		Matrix<float>sigma(_param.k, 1);
		sigma.copy(w_temp.data());

		//Get U matrix
		Matrix<float>U(X.rows(), _param.k);
		Matrix<float>QReversed(Q.rows(), Q.columns());
		col_reverse_q(Q, QReversed, context);
		calculate_u(X, QReversed, sigma, U, context);
		U.copy_to_host(_U); //Send to host

		//Explained Variance
		Matrix<float>UmultSigma(U.rows(), U.columns());
		//U * Sigma
		multiply_diag(U, sigma, UmultSigma, context, false);
		Matrix<float>UmultSigmaVar(_param.k, 1);
		calc_var(UmultSigma, UmultSigmaVar, _param.k, context);
		UmultSigmaVar.copy_to_host(_explained_variance);

		//Explained Variance Ratio
		//Set aside matrix of 1's for getting sum of columnar variances
		Matrix<float>XmultOnes(X.rows(), 1);
		XmultOnes.fill(1.0f);
		Matrix<float>XVar(1, X.columns());
		calc_var(X, XVar, X.columns(), context);
		Matrix<float>XVarSum(1,1);
		multiply(XVar, XmultOnes, XVarSum, context, false, false, 1.0f);
		Matrix<float>ExplainedVarRatio(_param.k, 1);
		divide(UmultSigmaVar, XVarSum, ExplainedVarRatio, context);
		ExplainedVarRatio.copy_to_host(_explained_variance_ratio);

		}
		catch (const std::exception &e)
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
