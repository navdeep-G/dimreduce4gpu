from distutils.core import setup
import os

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
dll_path = [os.path.join(curr_path, '../lib/'),
                os.path.join(curr_path, './lib/')]

if os.name == 'nt':
    dll_path = [os.path.join(p, 'scl.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'libscl.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

setup(name='scl', 
    py_modules=['scl'],
    data_files=[('scl', lib_path)])

