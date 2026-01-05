#!/usr/bin/env bash
set -euo pipefail

SO_PATH="${1:-dimreduce4gpu/lib/libdimreduce4gpu.so}"

echo "== Verifying native shared library: ${SO_PATH}"

# 1) Must exist
test -f "${SO_PATH}"
ls -lh "${SO_PATH}"

# 2) Must be a shared object (ELF DYN)
file "${SO_PATH}" | tee /dev/stderr | grep -E "ELF .* shared object"
readelf -h "${SO_PATH}" | grep -E "Type:\s+DYN" >/dev/null

# 3) Dynamic section (deps + rpath/runpath)
echo "== Dynamic section (NEEDED/RPATH/RUNPATH):"
readelf -d "${SO_PATH}" | egrep "NEEDED|RPATH|RUNPATH" || true

# Fail if it accidentally bakes in a CI/build path RPATH/RUNPATH
if readelf -d "${SO_PATH}" | egrep -q "RPATH|RUNPATH"; then
  if readelf -d "${SO_PATH}" | egrep -q "/__w/|/home/runner|/mnt/|/tmp/"; then
    echo "ERROR: .so contains a CI/build-path RPATH/RUNPATH" >&2
    exit 1
  fi
fi

# 4) Dependencies must resolve
echo "== ldd (must not have 'not found'):"
ldd "${SO_PATH}" | tee /dev/stderr
if ldd "${SO_PATH}" | grep -q "not found"; then
  echo "ERROR: Missing runtime dependency (ldd shows 'not found')." >&2
  exit 1
fi

# 5) Relocations/symbol resolution must succeed
echo "== ldd -r (relocations/symbol resolution):"
ldd -r "${SO_PATH}" | tee /dev/stderr
if ldd -r "${SO_PATH}" | egrep -q "undefined symbol"; then
  echo "ERROR: Undefined symbol detected in relocation check." >&2
  exit 1
fi

# 6) dlopen test via ctypes (no GPU required)
echo "== dlopen via ctypes:"
python3 - <<'PY'
import ctypes, os, sys
so = os.environ.get("SO_PATH", "dimreduce4gpu/lib/libdimreduce4gpu.so")
try:
    ctypes.CDLL(os.path.abspath(so))
except OSError as e:
    print(f"ERROR: ctypes.CDLL failed to load {so}: {e}", file=sys.stderr)
    raise
print("OK: dlopen succeeded")
PY

# 7) Report whether the environment is GPU-runnable.
# This should not fail on CPU-only runners; it's informational.
echo "== CUDA driver/device check (informational):"
python3 - <<'PY'
import importlib.util
from pathlib import Path

native_path = Path('dimreduce4gpu') / '_native.py'
spec = importlib.util.spec_from_file_location('dimreduce4gpu_native', native_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {native_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print(f"native_built(): {mod.native_built()}")

ok, count, reason = mod._cuda_driver_device_count()  # intentionally internal for diagnostics
if ok:
    print(f"native_runnable(): True (CUDA devices: {count})")
else:
    print(f"native_runnable(): False ({reason})")
PY

echo "== All checks passed."
