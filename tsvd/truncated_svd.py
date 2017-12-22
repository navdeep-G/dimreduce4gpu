import ctypes
import numpy as np
from .lib_tsvd import _load_tsvd_lib
from .lib_tsvd import params

class TruncatedSVD(object):
    """Dimensionality reduction using truncated SVD for GPUs

    Perform linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition.

    :param: int n_components: Desired dimensionality of output data

    :param: str algorithm: SVD solver to use.
                           Either “cusolver” (similar to ARPACK)
                           or “power” for the power method.

    :param: float tol: Tolerance for "power" method. Ignored by "cusolver".
                       Should be > 0.0 to ensure convergence.

    """

    def __init__(self, n_components=2, algorithm="cusolver", tol=1e-5):
        self.n_components = n_components
        self.algorithm = algorithm
        self.tol = tol

    def fit(self, X):
        """Fit Truncated SVD on matrix X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :returns self : object

        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit Truncated SVD on matrix X and perform dimensionality reduction
           on X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
                  Training data.

        :returns X_new : array, shape (n_samples, n_components)
                         Reduced version of X. This will always be a
                         dense array.

        """
        X = np.asfortranarray(X, dtype=np.float64)
        Q = np.empty(
            (self.n_components, X.shape[1]), dtype=np.float64, order='F')
        U = np.empty(
            (X.shape[0], self.n_components), dtype=np.float64, order='F')
        w = np.empty(self.n_components, dtype=np.float64)
        explained_variance = np.empty(self.n_components, dtype=np.float64)
        explained_variance_ratio = np.empty(self.n_components, dtype=np.float64)
        param = params()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components
        param.algorithm = self.algorithm.encode('utf-8')
        param.tol = self.tol

        if param.tol <= 0.0:
            raise ValueError("The `tol` parameter must be > 0.0")
	
        _tsvd_code = _load_tsvd_lib()
        _tsvd_code(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U), _as_fptr(explained_variance), _as_fptr(explained_variance_ratio), param)

        self._Q = Q
        self._w = w
        self._U = U
        self._X = X
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio

        X_transformed = U * w
        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        :param: X {array-like, sparse matrix}, shape (n_samples, n_features)
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

        :param: X array-like, shape (n_samples, n_components)

        :returns X_original : array, shape (n_samples, n_features)
                              Note that this is always a dense array.

        """
        return np.dot(X, self.components_)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        :param bool deep : If True, will return the parameters for this
            estimator and contained subobjects that are estimators.

        :returns dict params : Parameter names mapped to their values.
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
                if w and w[0].category == DeprecationWarning:
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

        :return: self
        """
        if not params:
            # Simple optimization to gain speed(inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        import six
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    @property
    def components_(self):
        return self._Q

    @property
    def explained_variance_(self):
        return self.explained_variance

    @property
    def explained_variance_ratio_(self):
        return self.explained_variance_ratio

    @property
    def singular_values_(self):
        return self._w

    @property
    def U(self):
        return self._U

# Util to send pointers to backend
def _as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

