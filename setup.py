from distutils.core import setup
import os

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
dll_path = [os.path.join(curr_path, '../lib/'),
                os.path.join(curr_path, './lib/')]

if os.name == 'nt':
    dll_path = [os.path.join(p, 'dimreduce4gpu.dll') for p in dll_path]
else:
    dll_path = [os.path.join(p, 'libdimreduce4gpu.so') for p in dll_path]

lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

setup(name='dimreduce4gpu',
      version='0.1.0',
      description='Dimensionality Reduction on GPUs',
      author='Navdeep Gill',
      author_email='mr.navdeepgill@gmail.com',
      url='https://github.com/navdeep-G/dimreduce4gpu',
      packages=['dimreduce4gpu',],
      data_files=[('dimreduce4gpu', lib_path)])

