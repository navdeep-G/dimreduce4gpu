#!/usr/bin/env bash
set -euo pipefail

SO_PATH="${1:-dimreduce4gpu/lib/libdimreduce4cpu.so}"

echo "== Verifying CPU shared library: ${SO_PATH}"
test -f "${SO_PATH}"
ls -l "${SO_PATH}"

echo "== file:"
file "${SO_PATH}"

echo "== Dynamic section (NEEDED/RPATH/RUNPATH):"
readelf -d "${SO_PATH}" | egrep 'NEEDED|RPATH|RUNPATH' || true

echo "== ldd (must not have 'not found'):"
ldd "${SO_PATH}" | tee /tmp/ldd_cpu.txt
if grep -q "not found" /tmp/ldd_cpu.txt; then
  echo "ERROR: unresolved shared library dependencies"
  exit 2
fi

echo "== Exported symbol contract:"
while IFS= read -r sym; do
  if ! nm -D "${SO_PATH}" | awk '{print $3}' | grep -qx "${sym}"; then
    echo "ERROR: missing required exported symbol: ${sym}"
    exit 3
  fi
done < "$(dirname "$0")/expected_cpu_symbols.txt"

echo "== dlopen via ctypes:"
python - <<PY
import ctypes
ctypes.CDLL("${SO_PATH}")
print("OK: dlopen succeeded")
PY

echo "OK: CPU shared library looks structurally valid."
