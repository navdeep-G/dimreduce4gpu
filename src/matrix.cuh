#pragma once
#include "utils.cuh"
#include "device_context.cuh"
#include "cusolverDn.h"
#include <../cub/cub/cub.cuh>
#include "scl.h"

namespace scl
{
	/**
	 * \class	Matrix
	 *
	 * \brief	Matrix type. Stores data internally in column major forscl.
	 *
	 * \author	Rory
	 * \date	2/20/2017
	 */

	template <typename T>
	class Matrix
	{
		int _m;
		int _n;

		T* _data;

	public:

		/**
		 * \fn	Matrix()
		 *
		 * \brief	Default constructor.
		 *
		 * \author	Rory
		 * \date	3/15/2017
		 */

		Matrix() : _m(0), _n(0), _data(nullptr)
		{
		}

		/**
		 * \fn	Matrix(int m, int n)
		 *
		 * \brief	Constructor. Initialize sclrix with m rows and n columns in device memory.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \param	m	Matrix rows.
		 * \param	n	Matrix columns.
		 */

		Matrix(int m, int n) : _m(m), _n(n)
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
		}

		/**
		 * \fn	Matrix(const Matrix<T>& M)
		 *
		 * \brief	Constructor. Initialise sclrix by copying existing sclrix.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \param	M	The Matrix&lt;T&gt; to copy.
		 */

		Matrix(const Matrix<T>& M) : _n(M.columns()), _m(M.rows())
		{
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
			this->copy(M);
		}

		~Matrix()
		{
			safe_cuda(cudaFree(_data));
		}

		/**
		 * \fn	void resize(int m, int n)
		 *
		 * \brief	Resizes.
		 *
		 * \author	Rory
		 * \date	3/15/2017
		 *
		 * \param	m	Matrix rows.
		 * \param	n	Matrix columns.
		 */

		void resize(int m, int n)
		{
			_m = m;
			_n = n;
			if (_data != nullptr)
			{
				safe_cuda(cudaFree(_data));
			}
			safe_cuda(cudaMalloc(&_data, _n*_m* sizeof(T)));
		}

		/**
		 * \fn	T* data()
		 *
		 * \brief Return raw pointer to data. Data is allocated on device.	
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	Raw pointer to Matrix data.
		 */

		T* data()
		{
			return _data;
		}

		/**
		 * \fn	T* data()
		 *
		 * \brief Return const raw pointer to data. Data is allocated on device.	
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	Raw pointer to Matrix data.
		 */
		const T* data() const
		{
			return _data;
		}

		/**
		 * \fn	thrust::device_ptr<T> dptr()
		 *
		 * \brief	Get thrust device pointer to sclrix data. Useful for invoking thrust functions.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	A thrust::device_ptr&lt;T&gt;
		 */

		thrust::device_ptr<T> dptr()
		{
			return thrust::device_pointer_cast(_data);
		}

		/**
		 * \fn	thrust::device_ptr<T> dptr()
		 *
		 * \brief	Get const thrust device pointer to sclrix data. Useful for invoking thrust functions.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	A thrust::device_ptr&lt;T&gt;
		 */

		thrust::device_ptr<const T> dptr() const
		{
			return thrust::device_pointer_cast(_data);
		}

		/**
		 * \fn	int rows() const
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	Number of sclrix rows.
		 */

		int rows() const
		{
			return _m;
		}

		/**
		 * \fn	int columns() const
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return	Number of sclrix columns.
		 */

		int columns() const
		{
			return _n;
		}

		/**
		 * \fn	int size() const
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \return Number of sclrix elements (m*n).
		 */

		int size() const
		{
			return _n * _m;
		}

		/**
		 * \fn	void zero()
		 *
		 * \brief	Zeroes sclrix elements.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 */

		void zero()
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, 0);
		}

		/**
		 * \fn	void fill(T val)
		 *
		 * \brief	Fills sclrix with given value.
		 *
		 * \author	Rory
		 * \date	3/15/2017
		 *
		 * \param	val	The value.
		 */

		void fill(T val)
		{
			thrust::fill(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data) + _n * _m, val);
		}

		/**
		 * \fn	void random(int random_seed = 0)
		 *
		 * \brief	Fills sclrix elements with uniformly distributed numbers between 0-1.0
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \param	random_seed	(Optional) The random seed.
		 */

		void random(int random_seed = 0)
		{
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + _m * _n,
			                  thrust::device_ptr<T>(_data),
			                  [=]__device__(int idx)
			                  {
				                  thrust::default_random_engine randEng(random_seed);
				                  thrust::uniform_real_distribution<scl_float> uniDist;
				                  randEng.discard(idx);
				                  return uniDist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void random_normal(int random_seed = 0)
		 *
		 * \brief	Fill sclrix with normally distributed random numbers between zero and one.
		 *
		 * \author	Rory
		 * \date	3/27/2017
		 *
		 * \param	random_seed	(Optional) The random seed.
		 */

		void random_normal(int random_seed = 0)
		{
			auto counting = thrust::make_counting_iterator(0);
			thrust::transform(counting, counting + _m * _n,
			                  thrust::device_ptr<T>(_data),
			                  [=]__device__(int idx)
			                  {
				                  thrust::default_random_engine randEng(random_seed);
				                  thrust::normal_distribution<scl_float> dist;
				                  randEng.discard(idx);
				                  return dist(randEng);
			                  }
			);
		}

		/**
		 * \fn	void copy(const T*hptr)
		 *
		 * \brief	Copies from host pointer to sclrix. Assumes host pointer contains array of same size as sclrix.
		 *
		 * \author	Rory
		 * \date	2/27/2017
		 *
		 * \param	hptr	Host pointer.
		 */

		template <typename HostT>
		void copy(const HostT* hptr)
		{
			thrust::copy(hptr, hptr + this->size(), this->dptr());
		}

		/**
		 * \fn	void copy(const Matrix<T>& M)
		 *
		 * \brief	Copies the given M.
		 *
		 * \author	Rory
		 * \date	3/6/2017
		 *
		 * \param	M	The Matrix&lt;T&gt; to process.
		 */

		void copy(const Matrix<T>& M)
		{
			scl_check(M.rows() == this->rows()&&M.columns() == this->columns(), "Cannot copy sclrix. Dimensions are different.");
			thrust::copy(M.dptr(), M.dptr() + M.size(), this->dptr());
		}


		void print() const
		{
			thrust::host_vector<T> h_scl(thrust::device_ptr<T>(_data), thrust::device_ptr<T>(_data + _n * _m));
			for (int i = 0; i < _m; i++)
			{
				for (int j = 0; j < _n; j++)
				{
					printf("%1.2f ", h_scl[j * _m + i]);
				}
				printf("\n");
			}
		}

		template<typename function_t>
		void transform(function_t f)
		{
			thrust::transform(this->dptr(), this->dptr() + this->size(), this->dptr(), f);
		}
	};

	/**
	 * \fn	void multiply(const Matrix<scl_float>& A, const Matrix<scl_float>& B, Matrix<scl_float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false, scl_float alpha=1.0f);
	 *
	 * \brief	Matrix multiplication. ABa = C. A or B may be transposed. a is a scalar.
	 *
	 * \author	Rory
	 * \date	2/21/2017
	 *
	 * \param 		  	A		   	The Matrix&lt;float&gt; to process.
	 * \param 		  	B		   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	C		   	The Matrix&lt;float&gt; to process.
	 * \param [in,out]	context	   	The context.
	 * \param 		  	transpose_a	(Optional) True to transpose a.
	 * \param 		  	transpose_b	(Optional) True to transpose b.
	 * \param 		  	alpha	   	(Optional) The alpha.
	 */

	void multiply(const Matrix<scl_float>& A, const Matrix<scl_float>& B, Matrix<scl_float>& C, DeviceContext& context, bool transpose_a = false, bool transpose_b = false, scl_float alpha = 1.0f);

	/**
	 * \fn	void multiply(Matrix<scl_float>& A, const scl_float a ,DeviceContext& context);
	 *
	 * \brief	Matrix scalar multiplication.
	 *
	 * \author	Rory
	 * \date	3/6/2017
	 *
	 * \param [in,out]	A	   	The Matrix&lt;float&gt; to process.
	 * \param 		  	a	   	The scalar.
	 * \param [in,out]	context	The context.
	 */

	void multiply(Matrix<scl_float>& A, const scl_float a, DeviceContext& context);

	/**
	 * \fn	void sclrix_sub(const Matrix<scl_float>& A, const Matrix<float>& B, Matrix<float>& C, DeviceContext& context)
	 *
	 * \brief	Matrix subtraction. A - B = C.
	 *
	 * \author	Rory
	 ned* \date	2/21/2017
	 *
	 */

	void subtract(const Matrix<scl_float>& A, const Matrix<scl_float>& B, Matrix<scl_float>& C, DeviceContext& context);

	/**
	 * \fn	void add(const Matrix<scl_float>& A, const Matrix<scl_float>& B, Matrix<scl_float>& C, DeviceContext& context);
	 *
	 * \brief	Matrix addition. A + B = C	
	 *
	 * \author	Rory
	 * \date	3/6/2017
	 *
	 * \param 		  	A	   	The Matrix&lt;scl_float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	C	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void add(const Matrix<scl_float>& A, const Matrix<scl_float>& B, Matrix<scl_float>& C, DeviceContext& context);
	/**
	 * \fn	void transpose(const Matrix<scl_float >&A, Matrix<scl_float >&B, DeviceContext& context)
	 *
	 * \brief	Transposes sclrix A into sclrix B.
	 *
	 * \author	Rory
	 * \date	2/27/2017
	 *
	 * \param 		  	A	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	B	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void transpose(const Matrix<scl_float>& A, Matrix<scl_float>& B, DeviceContext& context);

	/**
	 * \fn	void linear_solve(const Matrix<scl_float>& A, Matrix<scl_float>& X, const Matrix<scl_float>& B, DeviceContext& context)
	 *
	 * \brief	Solve linear system AX=B to find B.
	 *
	 * \author	Rory
	 * \date	2/26/2017
	 *
	 * \param 		  	A	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	X	   	The Matrix&lt;scl_float&gt; to process.
	 * \param 		  	B	   	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	context	The context.
	 */

	void linear_solve(const Matrix<scl_float>& A, Matrix<scl_float>& X, const Matrix<scl_float>& B, DeviceContext& context);

	/**
	 * \fn	void pseudoinverse(const Matrix<scl_float>& A, Matrix<scl_float>& pinvA, DeviceContext& context)
	 *
	 * \brief	Calculate Moore-Penrose seudoinverse using the singular value decomposition method.
	 *
	 * \author	Rory
	 * \date	2/26/2017
	 *
	 * \param 		  	A	   	Input sclrix.
	 * \param [in,out]	pinvA  	The pseudoinverse out.
	 * \param [in,out]	context	Device context.
	 */

	void pseudoinverse(const Matrix<scl_float>& A, Matrix<scl_float>& pinvA, DeviceContext& context);

	/**
	 * \fn	void normalize_columns(Matrix<scl_float>& M, Matrix<scl_float>& M_temp, Matrix<scl_float>& column_length, Matrix<scl_float>& ones, DeviceContext& context);
	 *
	 * \brief	Normalize sclrix columns.
	 *
	 * \author	Rory
	 * \date	3/6/2017
	 *
	 * \param [in,out]	M			 	The Matrix&lt;scl_float&gt; to process.
	 * \param [in,out]	M_temp		 	Temporary storage sclrix of size >= M.
	 * \param [in,out]	column_length	Temporary storage sclrix with one element per column.
	 * \param [in,out]	ones		 	Matrix of ones of length M.columns().
	 * \param [in,out]	context		 	The context.
	 */

	void normalize_columns(Matrix<scl_float>& M, Matrix<scl_float>& M_temp, Matrix<scl_float>& column_length, const Matrix<scl_float>& ones, DeviceContext& context);

	void normalize_columns(Matrix<scl_float>& M, DeviceContext& context);

	void f_normalize(Matrix<scl_float>& M, DeviceContext& context);

	void gradient_descent_solve(const Matrix<scl_float>& A, Matrix<scl_float>& X, const Matrix<scl_float>& B, Matrix<scl_float>& R, DeviceContext& context, scl_float eps = 0.1, scl_float min_rmse_change = 1e-5);

	void test_linear_solve();

	/**
	 * \fn	void residual(const Matrix<scl_float >&X, const Matrix<scl_float >&D, const Matrix<scl_float >&S, Matrix<scl_float >&R, DeviceContext & context);
	 *
	 * \brief	Calculate residual R = X - DS
	 *
	 * \author	Rory
	 * \date	3/16/2017
	 *
	 */

	void residual(const Matrix<scl_float>& X, const Matrix<scl_float>& D, const Matrix<scl_float>& S, Matrix<scl_float>& R, DeviceContext& context);

	void calculate_eigen_pairs_exact(const Matrix<scl_float>& X, Matrix<scl_float>& Q, Matrix<scl_float>& w, DeviceContext& context);


}
