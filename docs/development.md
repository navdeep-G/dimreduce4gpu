# Development

## Local setup

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Lint and tests

```bash
ruff check .
ruff format .
pytest
```

## Building the native library

See [Installation](installation.md) for the CMake build steps.

## GPU tests in GitHub Actions

This repo provides an optional workflow that runs *on a GPU runner*:

- `.github/workflows/gpu-tests.yml`

Run it via **Actions → GPU Tests → Run workflow** after you configure a self-hosted GPU runner (label `gpu`) or a GitHub hosted GPU runner.
