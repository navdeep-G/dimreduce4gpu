# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**, and this project aims to follow **Semantic Versioning**.

## [0.1.0] - 2026-01-05
### Added
- Modern Python packaging (`pyproject.toml`, `setup.cfg`) and improved developer tooling (ruff, pytest).
- GitHub Actions CI for linting and tests, plus a CUDA build verification job.
- Clear native library detection (`native_built()`, `native_runnable()`) and improved error messages.
- Deterministic unit tests, plus optional GPU-only workflow for full numerical correctness tests.
- Native `.so` verification script (`ci/verify_native_so.sh`) with dependency + symbol checks.
