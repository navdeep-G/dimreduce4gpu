
default: build_install_all

clean:
	rm -rf dimreduce4gpu.egg-info
	rm -rf dist
	rm -rf build

cmake:
	rm -rf build
	mkdir build
	cd build && cmake .. && make

setup:
	pip install -r requirements.txt

build: setup
	python setup.py bdist_wheel

build_all: cmake build

build_install: build
	pip install dist/dimreduce4gpu-0.1.0-py3-none-any.whl

build_install_all: clean cmake build_install

