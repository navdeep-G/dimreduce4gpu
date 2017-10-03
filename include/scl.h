#pragma once

#ifdef WIN32
#define scl_export __declspec(dllexport)
#else
#define scl_export
#endif

namespace scl
{
	extern "C"
	{

		typedef float  scl_float;

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
		 * \param 		  	_param
		 */

		scl_export void truncated_svd(const double * _X, double * _Q, double * _w, params _param);
	}
}
