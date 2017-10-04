#pragma once

#ifdef WIN32
#define tsvd_export __declspec(dllexport)
#else
#define tsvd_export
#endif

namespace tsvd
{
	extern "C"
	{

		typedef float  tsvd_float;

		struct params
		{
			int X_n;
			int X_m;
			int k;
		};

		/**
		 *
		 * \param 		  	_X
		 * \param [in,out]	_Q
		 * \param [in,out]	_w
		 * \param [in,out]  _U
		 * \param 		  	_param
		 */

		tsvd_export void truncated_svd(const double * _X, double * _Q, double * _w, double* _U, params _param);
	}
}
