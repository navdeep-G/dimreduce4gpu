from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import TruncatedSVD as SkTSVD

from dimreduce4gpu import PCA, TruncatedSVD


@dataclass(frozen=True)
class Result:
    name: str
    seconds_median: float
    seconds_runs: list[float]


def _timeit(fn, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()

    runs: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return runs


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = p.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.m), dtype=dtype)

    # Add anisotropy to make the problem non-trivial.
    X *= np.linspace(1.0, 3.0, args.m, dtype=dtype)

    # PCA
    ours_pca = PCA(n_components=args.k, backend="cpu", algorithm="power", n_iter=5, random_state=args.seed)
    sk_pca = SkPCA(n_components=args.k, svd_solver="randomized", iterated_power=5, random_state=args.seed)

    def run_ours_pca(model=ours_pca, X=X):
        model.fit_transform(X)

    def run_sk_pca(model=sk_pca, X=X):
        model.fit_transform(X)

    ours_pca_runs = _timeit(run_ours_pca, warmup=args.warmup, repeats=args.repeats)
    sk_pca_runs = _timeit(run_sk_pca, warmup=args.warmup, repeats=args.repeats)

    # TSVD
    ours_tsvd = TruncatedSVD(n_components=args.k, backend="cpu", algorithm="power", n_iter=5, random_state=args.seed)
    sk_tsvd = SkTSVD(n_components=args.k, algorithm="randomized", n_iter=5, random_state=args.seed)

    def run_ours_tsvd(model=ours_tsvd, X=X):
        model.fit_transform(X)

    def run_sk_tsvd(model=sk_tsvd, X=X):
        model.fit_transform(X)

    ours_tsvd_runs = _timeit(run_ours_tsvd, warmup=args.warmup, repeats=args.repeats)
    sk_tsvd_runs = _timeit(run_sk_tsvd, warmup=args.warmup, repeats=args.repeats)

    results = {
        "params": vars(args),
        "pca": {
            "ours": asdict(Result("dimreduce4gpu_cpu_pca", _median(ours_pca_runs), ours_pca_runs)),
            "sklearn": asdict(Result("sklearn_pca", _median(sk_pca_runs), sk_pca_runs)),
            "speedup": _median(sk_pca_runs) / _median(ours_pca_runs) if _median(ours_pca_runs) > 0 else None,
        },
        "tsvd": {
            "ours": asdict(Result("dimreduce4gpu_cpu_tsvd", _median(ours_tsvd_runs), ours_tsvd_runs)),
            "sklearn": asdict(Result("sklearn_tsvd", _median(sk_tsvd_runs), sk_tsvd_runs)),
            "speedup": _median(sk_tsvd_runs) / _median(ours_tsvd_runs) if _median(ours_tsvd_runs) > 0 else None,
        },
    }

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
