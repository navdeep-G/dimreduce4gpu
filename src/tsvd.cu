#include <cstdio>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "matrix.cuh"
#include "device_context.cuh"
#include <tsvd.h>
#include <ctime>


namespace tsvd
{

void truncated_svd(const double* _X, double* _Q, double* _w, params _param)
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
		normalize_columns(Q, context);
		Matrix<float>Qt(Q.columns(), Q.rows());
		transpose(Q, Qt, context);
		Qt.print();
		w.transform([=]__device__(float elem){
			if(elem > 0.0){
				return std::sqrt(elem);
			}else{
				return 0.0f;
			}
		}
		);
		w.print();
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
