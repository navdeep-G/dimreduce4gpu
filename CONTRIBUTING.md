# Contributing

Thanks for considering a contribution!

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
pytest
```

## Running formatting & lint

```bash
ruff check . --fix
ruff format .
```

## GPU build (CUDA)

This project ships Python wrappers that call a CUDA shared library (`libdimreduce4gpu.so`).
To build the CUDA library locally:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j
```

The resulting shared library should be placed in `dimreduce4gpu/lib/` (the CMake config does this by default).
