import ctypes
import sys

import numpy as np

from .lib_dimreduce4gpu import _load_tsvd_lib, params


class TruncatedSVD:
    """Dimensionality reduction using truncated SVD for GPUs
    Perform linear dimensionality reduction by means of truncated singular value decomposition (SVD).
    Contrary to PCA, this estimator does not center the data before computing the singular value decomposition.
    Parameters
    ----------
    n_components: int, Default=2
        Desired dimensionality of output data
    algorithm: string, Default="power"
        SVD solver to use.
        Either "cusolver" (similar to ARPACK)
        or "power" for the power method.
    n_iter: int, Default=100
        number of iterations (only relevant for power method)
        Should be at most 2147483647 due to INT_MAX in C++ backend.
    int random_state: seed (None for auto-generated)
    float tol: float, Default=1E-5
        Tolerance for "power" method. Ignored by "cusolver".
        Should be > 0.0 to ensure convergence.
        Should be 0.0 to effectively ignore
        and only base convergence upon n_iter
    verbose: bool
        Verbose or not
    n_gpus : int, optional, default: 1
        How many gpus to use.  If 0, use CPU backup method.
        Currently SVD only uses 1 GPU, so >1 has no effect compared to 1.
    gpu_id : int, optional, default: 0
        ID of the GPU on which the algorithm should run.
    """

    def __init__(
        self,
        n_components=2,
        algorithm="power",
        n_iter=100,
        random_state=None,
        tol=1e-5,
        verbose=0,
        n_gpus=1,
        gpu_id=0,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.randint(0, 2**31 - 1)
        self.tol = tol
        self.verbose = verbose
        self.n_gpus = n_gpus
        self.gpu_id = gpu_id

    def fit(self, X):
        """Fit Truncated SVD on matrix X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :returns self : object

        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit Truncated SVD on matrix X and perform dimensionality reduction on X.
        :param X : {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.
        :param y : Ignored
               For ScikitLearn compatibility
        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always be a
                         dense array.
        """
        import scipy

        # SciPy 1.11+ deprecates the nested namespace `scipy.sparse.csr.csr_matrix`.
        if isinstance(X, scipy.sparse.csr_matrix):
            X = X.toarray()

        X = self._check_double(X)
        matrix_type = np.float64 if self.double_precision == 1 else np.float32

        X = np.asfortranarray(X, dtype=matrix_type)
        Q = np.empty((self.n_components, X.shape[1]), dtype=matrix_type)
        U = np.empty((X.shape[0], self.n_components), dtype=matrix_type)
        w = np.empty(self.n_components, dtype=matrix_type)
        explained_variance = np.empty(self.n_components, dtype=matrix_type)
        explained_variance_ratio = np.empty(self.n_components, dtype=matrix_type)
        X_transformed = np.empty((U.shape[0], self.n_components), dtype=matrix_type)

        param = params()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.algorithm = self.algorithm.encode('utf-8')
        param.tol = self.tol
        param.n_iter = self.n_iter
        param.random_state = self.random_state
        param.verbose = self.verbose
        param.gpu_id = self.gpu_id
        param.whiten = False  # Whitening is not exposed for tsvd yet

        if param.tol < 0.0:
            raise ValueError("The `tol` parameter must be >= 0.0 but got " + str(param.tol))
        if param.n_iter < 1:
            raise ValueError("The `n_iter` parameter must be > 1 but got " + str(param.n_iter))
        if param.n_iter > 2147483647:
            raise ValueError(
                "The `n_iter parameter cannot exceed "
                "the value for "
                "C++ INT_MAX (2147483647) "
                "but got`" + str(self.n_iter)
            )

        _tsvd_code = _load_tsvd_lib()
        _tsvd_code(
            _as_fptr(X),
            _as_fptr(Q),
            _as_fptr(w),
            _as_fptr(U),
            _as_fptr(X_transformed),
            _as_fptr(explained_variance),
            _as_fptr(explained_variance_ratio),
            param,
        )

        self._w = w
        self._X = X
        self._U = U
        self._Q = Q
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.
        :param X : {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.
        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always
                         be a dense array.
        """
        fit = self.fit(X)
        X_new = fit.U * fit.singular_values_
        return X_new

    def inverse_transform(self, X):
        """Transform X back to its original space.
        :param X : array-like, shape (n_samples, n_components)
                Data to transform back to original space
        :returns X_original : array, shape (n_samples, n_features)
                              Note that this is always a dense array.
        """
        return np.dot(X, self.components_)

    def _check_double(self, data, convert=True):
        """Transform input data into a type which can be passed into C land."""
        if convert and data.dtype != np.float64 and data.dtype != np.float32:
            self._print_verbose(
                0, "Detected numeric data format which is not supported. Casting to np.float32."
            )
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
                f"Unsupported data type {data.dtype}, should be either np.float32 or np.float64"
            )
        return data

    def _print_verbose(self, level, msg):
        if self.verbose > level:
            print(msg)
            sys.stdout.flush()

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        from sklearn.utils.fixes import signature

        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != 'self' and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    f" {cls} with constructor {init_signature} doesn't follow this convention."
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        :param deep : bool
            If True, will return the parameters for this
            estimator and contained subobjects that are estimators.
        :returns params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils / __init__.py but it gets overwritten
            # when running under python3 somehow.
            import warnings

            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if w and w[0].category is DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

                # XXX : should we rather test if instance of estimator ?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this solver.
        :returns self : self
            Returns self
        """
        if not params:
            # Simple optimization to gain speed(inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        from sklearn.externals import six

        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {name} for estimator {self}. Check the list of available parameters with `estimator.get_params().keys()`."
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        f"Invalid parameter {key} for estimator {self.__class__.__name__}. Check the list of available parameters with `estimator.get_params().keys()`."
                    )
                setattr(self, key, value)
        return self

    @property
    def components_(self):
        """
        Components
        """
        return self._Q

    @property
    def explained_variance_(self):
        """
        The variance of the training samples transformed by a projection to
        each component.
        """
        return self.explained_variance

    @property
    def explained_variance_ratio_(self):
        """
        Percentage of variance explained by each of the selected components.
        """
        return self.explained_variance_ratio

    @property
    def singular_values_(self):
        """
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.
        """
        return self._w

    @property
    def U(self):
        """
        U Matrix
        """
        return self._U


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
