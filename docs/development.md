# Development Guide

## Repository layout

```
Vision3D/
├── configs/               # Hydra YAML configs (model, dataset, experiment)
├── docs/                  # Specialised documentation (this folder)
├── scripts/               # Offline data-processing utilities
│   ├── convert_nuscenes.py
│   └── generate_dummy_dataset.py
├── src/
│   └── vision3d/          # Installable Python package
│       ├── config/        # Hydra dataclass schema + runtime interfaces
│       ├── core/          # Losses, matchers, evaluators
│       ├── data/          # Dataset, loaders, filters, augmentations
│       ├── engine/        # PyTorch Lightning module
│       ├── models/        # Backbone, neck, encoder, head
│       └── utils/         # Geometry helpers, Foxglove logger
├── tests/                 # Unit, integration, and smoke tests
├── train.py               # Hydra entry point
├── pyproject.toml         # Project metadata & dependencies
├── .pre-commit-config.yaml
└── uv.lock
```

## Environment setup

See [installation.md](installation.md) for the full step-by-step guide.
Quick summary:

```bash
uv sync --group dev        # core + dev tools
pre-commit install         # install git hooks
```

## Code style

The project enforces style automatically via pre-commit hooks:

| Tool | Role | Config |
|------|------|--------|
| `ruff` | Linting + auto-formatting | `[tool.ruff]` in `pyproject.toml` |
| `mypy` | Static type checking | `[tool.mypy]` in `pyproject.toml` |

Key style conventions:
- **Line length:** 100 characters (`ruff` enforces, `mypy` does not check).
- **Type annotations:** required on all public functions and class attributes
  (mypy `strict` mode).
- **Imports:** sorted and organised by `ruff` (isort rules).
- **Python version:** 3.12+ syntax is fine (`target-version = "py312"`).

Run the linters manually at any time:

```bash
# Lint + format check
uv run ruff check .
uv run ruff format --check .

# Auto-fix formatting
uv run ruff format .

# Type checking
uv run mypy src/
```

## Testing

Tests live in `tests/` and are structured to mirror `src/vision3d/`:

```
tests/
├── config/             # Schema and dataclass validation tests
├── core/               # Loss, matcher unit tests
├── data/               # Loader, filter, augmentation unit tests
├── integration/        # End-to-end pipeline + model smoke tests
├── models/             # (model-specific unit tests)
├── scripts/            # Converter tests
└── utils/              # Geometry and foxglove tests
```

### Running tests

```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/data/

# Single file
uv run pytest tests/core/test_losses.py

# With coverage report
uv run pytest --cov=src/vision3d --cov-report=term-missing

# Verbose output
uv run pytest -v
```

### Test categories

| Category | Directory | Description |
|----------|-----------|-------------|
| Unit | `tests/config/`, `tests/core/`, `tests/data/`, `tests/utils/` | Test individual classes in isolation using small synthetic tensors |
| Integration | `tests/integration/` | Test component interactions (data pipeline → model → loss) using dummy data |
| Smoke | `tests/integration/` | Fast end-to-end checks that the full training loop runs without error |

All tests use the **dummy dataset** (generated via
`scripts/generate_dummy_dataset.py`) so they do not require any external data.
The dummy data generator is called automatically by the integration test
fixtures.

### Writing tests

- Follow the existing style in `tests/`.
- Use `pytest` fixtures for shared setup (see `tests/integration/helpers.py`).
- Keep unit tests fast (< 1 s per test) — avoid large tensors or real images.
- Prefer parametrize over copy-pasted test functions.

Example:

```python
import pytest
import torch
from vision3d.core.losses import DetectionLoss

@pytest.mark.parametrize("batch_size", [1, 2])
def test_detection_loss_returns_scalar(batch_size: int) -> None:
    loss_fn = DetectionLoss(num_classes=10, cls_weight=2.0, bbox_weight=0.25)
    ...
    assert loss.ndim == 0
```

## Pre-commit hooks

`.pre-commit-config.yaml` runs `ruff` and `mypy` automatically before each
commit. To bypass in an emergency (not recommended):

```bash
git commit --no-verify -m "wip: skip hooks"
```

## CI / GitHub Actions

The CI pipeline (`.github/workflows/ci.yml`) runs on every push and PR:

1. **Lint** – `ruff check` + `ruff format --check`
2. **Type check** – `mypy src/`
3. **Tests** – `pytest` with coverage

All three jobs must pass for a PR to be mergeable.

## Adding a new model

1. Create a new sub-directory under `src/vision3d/models/` (e.g. `models/centerpoint/`).
2. Implement the `nn.Module` class with a typed `forward` method that accepts
   `BatchData` and returns `BoundingBox3DPrediction`.
3. Add a Hydra config file under `configs/model/<name>.yaml`.
4. Register the class in `src/vision3d/models/__init__.py`.
5. Add unit / integration tests under `tests/models/` and
   `tests/integration/`.

## Adding a new dataset

1. Write a converter script under `scripts/convert_<dataset>.py` that
   produces the Vision3D generic JSON + image layout (see
   [data_format.md](data_format.md)).
2. Add a Hydra dataset config under `configs/dataset/<name>.yaml`.
3. Test with `scripts/generate_dummy_dataset.py` first to validate the
   pipeline before running on real data.

## Visualisation

Vision3D can export `.mcap` files for [Foxglove Studio](https://foxglove.dev/):

1. Install the `viz` group: `uv sync --group viz`
2. Run training normally — `FoxgloveMCAPLogger` writes files to
   `<output_dir>/mcap/` during validation epochs.
3. Open Foxglove Studio and drag-and-drop the `.mcap` file to compare
   ground-truth vs. predicted bounding boxes.

## Dependency management

Dependencies are managed with `uv` and pinned in `uv.lock`.

```bash
# Add a new runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --group dev <package>

# Update all dependencies
uv sync --upgrade

# Regenerate lock file after manual pyproject.toml edits
uv lock
```

Always commit both `pyproject.toml` and `uv.lock` together.
