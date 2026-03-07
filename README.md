# Vision3D: Open-Source 3D Object Detection Framework

## 1. Project Overview & Philosophy
**Vision3D** is a modular, PyTorch-native codebase for 3D object detection, designed for autonomous driving and robotics. Built as a clean alternative to overly complex frameworks (like MMDet3D), it prioritizes readability, community maintainability, and seamless integration with modern tools.

**Core Technical Stack:**
* **Deep Learning:** Pure PyTorch (No custom CUDA/C++ extensions).
* **Training Engine:** PyTorch Lightning.
* **Configuration:** Hydra with strictly typed Python Dataclasses.
* **Visualization:** Foxglove (via offline `.mcap` generation).

## 2. Architectural Decisions for POC
* **Initial Modality:** Multi-view Vision (extensible to LiDAR/Radar later).
* **Baseline Model:** BEVFormer.
* **Lightning Integration:** The PyTorch Lightning module acts strictly as a *container*. Hydra recursively instantiates all sub-components (Backbone, Encoder, Head, Loss, Matcher) and passes them into the Lightning module's `__init__`.
* **Bounding Box Format:** Standard 10-parameter format: `[x, y, z, w, l, h, sin(theta), cos(theta), vx, vy]`.
* **Dataloading:** Standard multi-threaded PyTorch `DataLoader` (no complex caching for the POC).
* **Metrics:** Custom, lean reimplementation of essential metrics in pure PyTorch/NumPy to avoid dependency bloat.

## 3. Data & Coordinate System Strategy
* **Coordinate Frame:** Ego-centric. All transformations are applied *offline* during dataset conversion. 
* **Images:** Assumed to be pre-undistorted.
* **Generic Format:** Data is loaded from a custom format consisting of PNGs and a single JSON file per frame.
    * The JSON contains all bounding boxes, camera intrinsics, and a pointer/tag to past frames for temporal attention.
* **Dataset Conversion:** NuScenes will be the first supported dataset, handled via an offline standalone conversion script.

---

## 4. Proposed File Structure

```text
Vision3D/
├── configs/                     # Hydra configuration files
│   ├── model/                   # Backbone, Encoder, Head, Loss configs
│   ├── dataset/                 # Dataloader and path configs
│   ├── experiment/              # High-level experiment overrides
│   └── train.yaml               # Main entry point config
├── data/                        # Local generic dataset directory (gitignored)
├── scripts/
│   └── convert_nuscenes.py      # Offline script to generate PNGs + JSONs
├── src/
│   └── vision3d/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py       # Main torch.utils.data.Dataset
│       │   ├── loaders.py       # ImageLoader, JsonLoader
│       │   ├── augmentations.py # DataAugmenter, Transforms
│       │   └── filters.py       # BoxFilter, BoundsFilter
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbones/       # e.g., ResNet
│       │   ├── necks/           # e.g., FPN
│       │   ├── encoders/        # BEVFormer temporal/spatial encoders
│       │   └── heads/           # 3D detection heads
│       ├── core/
│       │   ├── __init__.py
│       │   ├── matchers.py      # Bipartite/Hungarian matching
│       │   ├── losses.py        # Box and classification losses
│       │   └── evaluators.py    # Orchestrates evaluation (mAP, NDS)
│       ├── engine/
│       │   ├── __init__.py
│       │   └── lit_module.py    # The PyTorch Lightning container
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py        # Hydra Dataclass definitions
│       └── utils/
│           ├── __init__.py
│           ├── geometry.py      # Camera projectors, MSDA workarounds
│           └── foxglove.py      # MCAP writer callback for visualization
├── tests/                       # Unit tests for core components
├── train.py                     # Main training script (Hydra @main)
├── requirements.txt
└── README.md
```



---

## 5. Required Classes to Implement

### Data Layer
* `Vision3DDataset(Dataset)`: Orchestrates the loading process. Uses loaders, filters, and augmenters to output standardized dictionary/dataclass batches.
* `JsonLoader`: Responsible purely for reading, parsing, and validating the generic JSON schema.
* `ImageLoader`: Handles multi-threaded I/O for reading the 6+ multi-view PNGs efficiently.
* `BoxFilter`: Filters out ground truth boxes that fall outside the perception range, have too few visible points, or are physically invalid.
* `DataAugmenter`: Applies synchronized 3D and 2D augmentations (e.g., global rotation, image flipping). *Crucial: Must update camera extrinsics/intrinsics if the 3D space or 2D image is altered.*
* `NuScenesConverter`: A standalone utility class to parse the `nuscenes-devkit` and output ego-centric coordinates, undistorted images, and the unified JSON.

### Model Architecture (BEVFormer POC)
* `ResNetBackbone(nn.Module)`: Standard 2D feature extractor.
* `FPNNeck(nn.Module)`: Feature Pyramid Network to align scales.
* `BEVEncoder(nn.Module)`: The core BEVFormer module. Must implement Temporal Self-Attention and Spatial Cross-Attention.
    * *Constraint:* Must use native PyTorch (e.g., `F.grid_sample`) instead of custom Deformable Attention CUDA kernels.
* `DetectionHead(nn.Module)`: Takes the BEV grid and uses a transformer decoder to output 3D bounding box predictions and class logits.

### Core Logic
* `HungarianMatcher`: Computes the optimal bipartite matching between predicted queries and ground truth boxes based on classification and box distance costs.
* `DetectionLoss(nn.Module)`: Computes Focal Loss for classification and L1/GIoU loss for the bounding box parameters `[x, y, z, w, l, h, sin(theta), cos(theta), vx, vy]`.
* `CameraProjector`: A geometry utility to project 3D reference points to 2D image coordinates using the provided intrinsics and sensor-to-ego extrinsics.
* `Vision3DEvaluator`: Orchestrates the evaluation loop. Accumulates predictions, formats them, and computes custom mAP/NDS-like metrics at the end of an epoch.

### Engine & Utilities
* `Vision3DLightningModule(pl.LightningModule)`: 
    * `__init__(self, backbone, neck, encoder, head, matcher, loss, evaluator)`
    * Handles `training_step`, `validation_step`, and optimizer configuration.
* `FoxgloveMCAPLogger(pl.Callback)`: A Lightning Callback that hooks into `on_validation_epoch_end` to write ground truth and predicted bounding boxes into an `.mcap` file for easy drag-and-drop debugging in Foxglove Studio.

---

## 6. Generic JSON Dataset Schema

Each frame in the dataset will have a single corresponding JSON file (e.g., `frame_000123.json`). This keeps dataloading simple and localized.

```json
{
  "frame_id": "unique_string_identifier",
  "timestamp": 1600000000.123456,
  "past_frame_ids": [
    "unique_string_identifier_minus_1",
    "unique_string_identifier_minus_2"
  ],
  "cameras": {
    "front": {
      "image_path": "images/front/frame_000123.png",
      "intrinsics": [
        [800.0, 0.0, 400.0],
        [0.0, 800.0, 300.0],
        [0.0, 0.0, 1.0]
      ],
      "sensor2ego_translation": [1.5, 0.0, 1.2],
      "sensor2ego_rotation": [0.0, 0.0, 0.0, 1.0]
    },
    "back": { ... },
    "left": { ... }
  },
  "annotations": [
    {
      "instance_id": "object_track_id_99",
      "class_name": "car",
      "bbox_3d": [
        10.5, -2.1, 0.5,      // x, y, z (center in ego-centric coords)
        4.5, 1.8, 1.5,        // l, w, h
        0.707, 0.707,         // sin(theta), cos(theta)
        5.2, 0.0              // vx, vy
      ]
    }
  ]
}
```

---

## 7. Hydra Configuration (Dataclasses)

Using Python Dataclasses for Hydra ensures strict typing and auto-completion. These are defined in `src/vision3d/config/schema.py`.

```python
from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class BackboneConfig:
    _target_: str = "vision3d.models.backbones.ResNetBackbone"
    depth: int = 50
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3])

@dataclass
class NeckConfig:
    _target_: str = "vision3d.models.necks.FPNNeck"
    in_channels: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    out_channels: int = 256

@dataclass
class EncoderConfig:
    _target_: str = "vision3d.models.encoders.BEVEncoder"
    bev_h: int = 200
    bev_w: int = 200
    embed_dims: int = 256
    num_layers: int = 6

@dataclass
class HeadConfig:
    _target_: str = "vision3d.models.heads.DetectionHead"
    num_classes: int = 10
    in_channels: int = 256

@dataclass
class LossConfig:
    _target_: str = "vision3d.core.losses.DetectionLoss"
    cls_weight: float = 2.0
    bbox_weight: float = 0.25

@dataclass
class MatcherConfig:
    _target_: str = "vision3d.core.matchers.HungarianMatcher"
    cost_class: float = 2.0
    cost_bbox: float = 0.25

@dataclass
class EvaluatorConfig:
    _target_: str = "vision3d.core.evaluators.Vision3DEvaluator"
    eval_range: float = 50.0

@dataclass
class LitModuleConfig:
    _target_: str = "vision3d.engine.lit_module.Vision3DLightningModule"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
```
