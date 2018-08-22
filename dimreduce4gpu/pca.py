import ctypes
import numpy as np
from .truncated_svd import TruncatedSVD
from .lib_dimreduce4gpu import _load_pca_lib
from .lib_dimreduce4gpu import params

class PCA(TruncatedSVD):
    """Principal Component Analysis (PCA)

    Dimensionality reduction using truncated Singular Value Decomposition
    for GPU

    This implementation uses the Cusolver implementation of the truncated SVD.
    Contrary to SVD, this estimator does center the data before computing
    the singular value decomposition.

    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data

    whiten : bool, optional
        When True (False by default) the `components_` vectors are multiplied
        by the square root of (n_samples) and divided by the singular values to
        ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    verbose: bool
        Verbose or not

    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.
    """

    def __init__(self, n_components=2, whiten=False,
                 verbose=0, gpu_id=0):
        super().__init__(n_components)
        self.whiten = whiten
        self.n_components_ = n_components
        self.mean_ = None
        self.noise_variance_ = None
        self.algorithm = "cusolver"
        self.verbose = verbose
        self.gpu_id = gpu_id

    def fit(self, X):
        """Fit Principal Components Analysis on matrix X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :returns self : object

        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit Principal Components Analysis on matrix X and perform dimensionality reduction on X.
        :param X : {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.
        :param y : Ignored
               For ScikitLearn compatibility
        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always be a
                         dense array.
        """
        import scipy
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            X = scipy.sparse.csr_matrix.todense(X)

        X = self._check_double(X)
        matrix_type = np.float64 if self.double_precision == 1 else np.float32
        X = np.asfortranarray(X, dtype=matrix_type)
        Q = np.empty(
            (self.n_components, X.shape[1]), dtype=matrix_type)
        U = np.empty(
            (X.shape[0], self.n_components), dtype=matrix_type)
        w = np.empty(self.n_components, dtype=matrix_type)
        explained_variance = np.empty(self.n_components, dtype=matrix_type)
        explained_variance_ratio = np.empty(self.n_components, dtype=matrix_type)
        mean = np.empty(X.shape[1], dtype=matrix_type)
        X_transformed = np.empty((U.shape[0], self.n_components), dtype=matrix_type)

        param = params()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.algorithm = self.algorithm.encode('utf-8')
        param.n_iter = self.n_iter
        param.random_state = self.random_state
        param.tol = self.tol
        param.verbose = 1 if self.verbose else 0
        param.gpu_id = self.gpu_id
        param.whiten = self.whiten

        _pca_code = _load_pca_lib()
        _pca_code(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U), _as_fptr(X_transformed),
                   _as_fptr(explained_variance), _as_fptr(explained_variance_ratio), _as_fptr(mean), param)

        self._w = w
        self._U = U
        self._Q = Q
        self._X = X

        n = X.shape[0]
        # To match sci-kit #TODO Port to cuda?
        self.explained_variance = self.singular_values_ ** 2 / (n - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio = \
            self.explained_variance / total_var.sum()
        # self.explained_variance_ratio = explained_variance_ratio
        self.mean_ = mean

        # TODO noise_variance_ calculation
        # can be done inside lib.pca if a bottleneck
        n_samples, n_features = X.shape
        total_var = np.var(X, ddof=1, axis=0)
        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = \
                (total_var.sum() - self.explained_variance_.sum())
            self.noise_variance_ /= \
                min(n_features, n_samples) - self.n_components
        else:
            self.noise_variance_ = 0.

        return X_transformed

    def _check_double(self, data, convert=True):
        """Transform input data into a type which can be passed into C land."""
        if convert and data.dtype != np.float64 and data.dtype != np.float32:
            self._print_verbose(0, "Detected numeric data format which is not "
                                   "supported. Casting to np.float32.")
            data = np.ascontiguousarray(data, dtype=np.floa32)
        if data.dtype == np.float64:
            self._print_verbose(0, "Detected np.float64 data")
            self.double_precision = 1
            data = np.ascontiguousarray(data, dtype=np.float64)
        elif data.dtype == np.float32:
            self._print_verbose(0, "Detected np.float32 data")
            self.double_precision = 0
            data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported data type %s, "
                "should be either np.float32 or np.float64" % data.dtype)
        return data

def _as_dptr(x):
    '''

    :param x:
    :return:
    '''
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def _as_fptr(x):
    '''

    :param x:
    :return:
    '''
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

