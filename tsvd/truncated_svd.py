import ctypes
import numpy as np
from .lib_tsvd import _load_tsvd_lib
from .lib_tsvd import params

class TruncatedSVD(object):

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asfortranarray(X, dtype=np.float64)
        Q = np.empty((self.n_components, X.shape[1]), dtype=np.float64, order='F')
        U = np.empty((X.shape[0], self.n_components), dtype=np.float64, order='F')
        w = np.empty(self.n_components, dtype=np.float64)
        param = params()
        param.X_m = X.shape[0]
        param.X_n = X.shape[1]
        param.k = self.n_components

        _tsvd_code = _load_tsvd_lib()
        _tsvd_code(_as_fptr(X), _as_fptr(Q), _as_fptr(w), _as_fptr(U), param)

        self._Q = Q
        self._w = w
        self._U = U
        self._X = X

        return self

    @property
    def components_(self):
        return self._Q

    @property
    def singular_values_(self):
        return self._w

    @property
    def X(self):
        return self._X

    @property
    def U(self):
        return self._U

def _as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
