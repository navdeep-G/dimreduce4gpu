"""GPU benchmark harness.

This script is intended to run on a GPU-capable machine where the native library
is runnable (see `dimreduce4gpu.native_runnable()`). It records timing data and
basic environment metadata to a JSON file.

Run:

    python bench/run_benchmarks.py --out bench_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

import dimreduce4gpu
from dimreduce4gpu import PCA, TruncatedSVD


@dataclass
class Case:
    name: str
    kind: str
    n: int
    m: int
    k: int


def _timed(fn, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    return median, times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json", help="Output JSON path")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not dimreduce4gpu.native_runnable():
        raise RuntimeError(
            "Native library is not runnable in this environment. "
            "Run this on a machine with NVIDIA drivers and a CUDA-capable GPU."
        )

    rng = np.random.default_rng(0)

    cases = [
        Case("pca_50k_x_256_k32", "pca", 50_000, 256, 32),
        Case("tsvd_50k_x_256_k32", "tsvd", 50_000, 256, 32),
        Case("pca_200k_x_128_k16", "pca", 200_000, 128, 16),
    ]

    results: dict[str, Any] = {
        "meta": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "commit": os.environ.get("GITHUB_SHA"),
        },
        "cases": [],
    }

    for c in cases:
        X = rng.normal(size=(c.n, c.m)).astype(np.float32)

        if c.kind == "pca":
            model = PCA(n_components=c.k, algorithm="cusolver", verbose=False)

            def run(model=model, X=X):
                model.fit_transform(X)

        elif c.kind == "tsvd":
            model = TruncatedSVD(n_components=c.k, algorithm="power", n_iter=5, verbose=False)

            def run(model=model, X=X):
                model.fit_transform(X)

        else:
            raise ValueError(c.kind)

        median, all_times = _timed(run, warmup=args.warmup, repeats=args.repeats)

        results["cases"].append(
            {
                "case": asdict(c),
                "median_seconds": median,
                "all_seconds": all_times,
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
