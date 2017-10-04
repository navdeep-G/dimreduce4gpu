import os, sys, ctypes
import numpy as np

class params(ctypes.Structure):
    _fields_  = [('X_n', ctypes.c_int),
            ('X_m', ctypes.c_int),
            ('k', ctypes.c_int)
    ]

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
dll_path = [os.path.join(sys.prefix, 'tsvd'), os.path.join(curr_path, '../lib/')]

if os.name == 'nt':
    dll_path = [os.path.join(p, 'tsvd.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'libtsvd.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

if len(lib_path) is 0:
    print('Could not find shared library path at the following locations:')
    print(dll_path)

# Fix for GOMP weirdness with CUDA 8.0
try:
    ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
except:
    pass
_mod = ctypes.cdll.LoadLibrary(lib_path[0])
_tsvd_code = _mod.truncated_svd
_tsvd_code.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), params]

def as_fptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def truncated_svd(X, k):
    X = np.asfortranarray(X, dtype=np.float64)
    Q = np.empty((X.shape), dtype=np.float64, order='F')
    U = np.empty((X.shape[1], X.shape[1]), dtype=np.float64, order='F')
    w = np.empty(k, dtype=np.float64)
    param = params()
    param.X_m = X.shape[0]
    param.X_n = X.shape[1]
    param.k = k

    _tsvd_code(as_fptr(X), as_fptr(Q), as_fptr(w),as_fptr(U), param)

    return Q, U, w
