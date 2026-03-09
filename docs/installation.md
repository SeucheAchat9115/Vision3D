# Installation

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.12 | Required by `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | latest | Recommended package manager |
| CUDA toolkit | optional | Recommended for training; CPU-only runs work but are slow |

## Install core dependencies

```bash
git clone <repository-url>
cd Vision3D
uv sync
```

This creates a virtual environment in `.venv/` and installs all runtime
dependencies declared in `pyproject.toml`.

## Optional dependency groups

Vision3D uses dependency groups to keep the base install lightweight.
Install only the groups you need:

| Group | Command | What it adds |
|-------|---------|--------------|
| `dev` | `uv sync --group dev` | `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit` |
| `viz` | `uv sync --group viz` | `mcap` – required for Foxglove MCAP logging |
| `nuscenes` | `uv sync --group nuscenes` | `nuscenes-devkit` – required for NuScenes conversion |

To install everything at once:

```bash
uv sync --group dev --group viz --group nuscenes
```

## Activate the environment

```bash
# uv run — automatically uses .venv
uv run python tools/train.py

# Or activate manually
source .venv/bin/activate
python tools/train.py
```

## Set up pre-commit hooks (development only)

Pre-commit runs `ruff` (lint + format) and `mypy` (type-check) on every
`git commit`. Install once after cloning:

```bash
uv sync --group dev
pre-commit install
```

To run all hooks against the entire codebase manually:

```bash
pre-commit run --all-files
```

## Verify the installation

Run the test suite to confirm everything is working:

```bash
uv run pytest
```

All tests should pass on a clean install. If any test fails due to missing
optional dependencies, install the relevant group and re-run.

## Runtime dependency summary

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.2.0 | Core deep-learning framework |
| `torchvision` | ≥ 0.17.0 | Pretrained ResNet weights, image utilities |
| `pytorch-lightning` | ≥ 2.2.0 | Training loop, callbacks, checkpointing |
| `hydra-core` | ≥ 1.3.0 | Hierarchical configuration management |
| `omegaconf` | ≥ 2.3.0 | Config parsing (used internally by Hydra) |
| `Pillow` | ≥ 10.0.0 | Image I/O |
| `numpy` | ≥ 1.26.0 | Numerical operations |
| `scipy` | ≥ 1.12.0 | Hungarian matching (`linear_sum_assignment`) |
| `tensorboard` | ≥ 2.20.0 | Training metric logging |
