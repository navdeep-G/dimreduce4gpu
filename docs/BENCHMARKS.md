# Benchmarks

This repo includes a benchmark script that compares **dimreduce4gpu CPU backend** performance vs **scikit-learn**.

## Run locally

```bash
python bench/benchmark_cpu_vs_sklearn.py --n 2000 --m 256 --k 64 --repeats 5
```

## GitHub Actions (CPU benchmarks)

There is a `Benchmarks (CPU)` workflow that you can run manually from the Actions tab.
It uploads a JSON artifact with timings and speed ratios.

> Benchmarks are not run on every PR to avoid CI flakiness caused by shared runner noise.
