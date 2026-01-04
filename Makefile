default: build_install_all

clean:
	rm -rf dimreduce4gpu.egg-info
	rm -rf dist
	rm -rf build

cmake:
	rm -rf build
	mkdir -p build
	cd build && cmake .. && make -j

setup:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -e ".[dev]"

build:
	python -m pip wheel . -w dist

build_all: cmake build

build_install: build
	python -m pip install --upgrade --force-reinstall dist/*.whl

build_install_all: clean cmake build_install
