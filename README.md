# Vision3D

A modular, PyTorch-native 3D object detection framework for autonomous
driving and robotics. Vision3D is designed to be readable and maintainable —
a clean alternative to heavier frameworks like MMDet3D.

## Quick start

```bash
# 1. Clone and install
git clone <repository-url>
cd Vision3D
uv sync

# 2. Generate synthetic test data
python tools/generate_dummy_dataset.py --output-dir data/dummy --num-frames 50

# 3. Train (dummy data, default settings)
python tools/train.py

# 4. Train on NuScenes (after conversion)
python tools/train.py dataset=nuscenes max_epochs=24 batch_size=4
```

## Tech stack

| Concern | Tool |
|---------|------|
| Package management | [uv](https://docs.astral.sh/uv/) |
| Deep learning | PyTorch (pure — no custom CUDA kernels) |
| Training loop | PyTorch Lightning |
| Configuration | Hydra + typed Python dataclasses |
| Linting / formatting | ruff |
| Type checking | mypy |
| Testing | pytest |
| Visualisation | Foxglove Studio (offline `.mcap` files) |

## Architecture

Multi-view camera images pass through a four-stage pipeline:

```
ResNetBackbone → FPNNeck → BEVEncoder → DetectionHead
```

- **ResNetBackbone** — multi-scale 2-D feature extraction (ImageNet pretrained)
- **FPNNeck** — aligns channel dimensions across FPN levels
- **BEVEncoder** — BEVFormer-style encoder with Temporal Self-Attention and
  Spatial Cross-Attention (uses `F.grid_sample`, no custom CUDA)
- **DetectionHead** — DETR-style transformer decoder; outputs 3-D boxes,
  scores, and class labels

Training uses **Hungarian matching** (bipartite assignment) followed by
**Focal + L1 + GIoU loss**. Evaluation reports mAP at configurable
centre-distance thresholds.

## Data format

Each frame is represented by a single JSON file plus one PNG per camera.
Bounding boxes use the 10-DOF ego-centric format:

```
[x, y, z, w, l, h, sin(θ), cos(θ), vx, vy]
```

All coordinate frames are resolved offline during dataset conversion;
the training pipeline only ever reads clean, ego-centric data.

→ Full specification: [docs/data_format.md](docs/data_format.md)

## Documentation

| File | Contents |
|------|----------|
| [docs/installation.md](docs/installation.md) | Prerequisites, uv setup, optional dependency groups |
| [docs/data_format.md](docs/data_format.md) | JSON schema, bounding-box format, coordinate system, NuScenes conversion |
| [docs/configs.md](docs/configs.md) | Hydra config reference, all YAML keys, override examples |
| [docs/models.md](docs/models.md) | Model architecture, each sub-module, constructor arguments |
| [docs/dataloading.md](docs/dataloading.md) | Dataset, loaders, filters, augmentations, dataclass hierarchy |
| [docs/development.md](docs/development.md) | Contributing guide, linting, testing, CI pipeline |

## Repository layout

```
Vision3D/
├── configs/               # Hydra YAML configs (model, dataset, experiment)
├── docs/                  # Specialised documentation
├── tools/                 # Training entry point and offline data-processing utilities
├── src/vision3d/          # Installable Python package
│   ├── config/            # Hydra schema + runtime dataclasses
│   ├── core/              # Losses, matchers, evaluators
│   ├── data/              # Dataset, loaders, filters, augmentations
│   ├── engine/            # PyTorch Lightning module
│   ├── models/            # Backbone, neck, encoder, head
│   └── utils/             # Geometry helpers, Foxglove logger
├── tests/                 # Unit, integration, and smoke tests
└── pyproject.toml
```

## License

MIT
