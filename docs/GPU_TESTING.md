# GPU testing (without GitHub GPU runners)

This project uses a CUDA native library (`libdimreduce4gpu.so`). GitHub-hosted runners do **not** provide
GPUs by default, so runtime GPU tests (`native_runnable() == True`) won't run in standard CI.

You can still validate the GPU path in a few reproducible ways.

## Option A: Run in Docker on a machine with an NVIDIA GPU

Prereqs:
- Linux host with NVIDIA drivers installed
- Docker + NVIDIA Container Toolkit configured (`docker run --gpus all ...` works)

Build and run:

```bash
docker build -t dimreduce4gpu:dev -f Dockerfile.gpu .
docker run --rm -it --gpus all -v "$PWD":/work -w /work dimreduce4gpu:dev \
  bash -lc "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && \
           bash ci/verify_native_so.sh dimreduce4gpu/lib/libdimreduce4gpu.so && \
           pytest -q"
```

## Option B: Quick cloud validation (one-off GPU VM)

You can spin up a GPU VM for a short time, run the same commands, and tear it down.

### Generic steps
1. Create a GPU VM with an NVIDIA GPU (e.g., T4/L4/A10). Use Ubuntu 22.04.
2. Install NVIDIA drivers.
3. Install CUDA toolkit *or* use the CUDA Docker image (recommended).
4. Clone the repo and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
bash ci/verify_native_so.sh dimreduce4gpu/lib/libdimreduce4gpu.so
pytest -q
```

### Notes
- If you run inside `Dockerfile.gpu`, you still need the host NVIDIA driver.
- `dimreduce4gpu-diagnose` can help debug driver/library availability.

## Option C: Manual validation checklist before a release

Run the following on a GPU machine:
- `bash ci/verify_native_so.sh dimreduce4gpu/lib/libdimreduce4gpu.so`
- `pytest -q` (this runs the numeric correctness checks when `native_runnable()` is true)
- Optional: run `bench/run_benchmarks.py` and record results
