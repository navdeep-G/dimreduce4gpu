#!/bin/sh
cd build
cmake ..
make
cd ../python
python setup.py install
cd ../test
python test_small.py
