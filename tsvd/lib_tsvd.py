import os, sys, ctypes

class params(ctypes.Structure):
    _fields_  = [('X_n', ctypes.c_int),
                ('X_m', ctypes.c_int),
                ('k', ctypes.c_int),
                ('algorithm', ctypes.c_char_p)]

def _load_tsvd_lib():
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
    _tsvd_code.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), params]

    return _tsvd_code