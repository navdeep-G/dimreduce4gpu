#!/bin/sh
dir=build
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi
cd build
cmake ..
make
cd ../python
python setup.py install
cd ../test
python test_small.py
