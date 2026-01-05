from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict

import numpy as np

from dimreduce4gpu import PCA, TruncatedSVD, native_runnable


@dataclass
class BenchResult:
    name: str
    n_samples: int
    n_features: int
    n_components: int
    seconds: float


def _time_it(fn, warmup: int = 1, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run simple GPU benchmarks for dimreduce4gpu")
    parser.add_argument("--out", default="bench-results.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not native_runnable():
        raise SystemExit(
            "native_runnable() is False: a GPU-capable environment is required to run benchmarks"
        )

    rng = np.random.default_rng(args.seed)

    cases = [
        ("pca_cusolver", PCA, {"algorithm": "cusolver"}),
        ("tsvd_power", TruncatedSVD, {"algorithm": "power", "n_iter": 5}),
    ]

    shapes = [
        (10_000, 128, 32),
        (50_000, 256, 64),
    ]

    results: list[BenchResult] = []
    for (name, cls, kwargs) in cases:
        for (n, d, k) in shapes:
            X = rng.standard_normal((n, d), dtype=np.float32)
            model = cls(n_components=k, **kwargs)

            def run():
                model.fit_transform(X)

            sec = _time_it(run, warmup=1, repeats=3)
            results.append(
                BenchResult(
                    name=name,
                    n_samples=n,
                    n_features=d,
                    n_components=k,
                    seconds=sec,
                )
            )

    out = {
        "results": [asdict(r) for r in results],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
