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
cd ..
python setup.py install
cd test
echo "TSVD TEST"
python test_tsvd_small.py
echo "PCA TEST"
python test_pca_small.py
