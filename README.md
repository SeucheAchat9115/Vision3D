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
│       │   └── dataset.py       # Parses the generic JSON + PNG format
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
│       │   └── metrics.py       # Custom lean NDS/mAP evaluators
│       ├── engine/
│       │   ├── __init__.py
│       │   └── lit_module.py    # The PyTorch Lightning container
│       └── utils/
│           ├── __init__.py
│           ├── geometry.py      # Pure PyTorch MSDA (grid_sample) workarounds
│           └── foxglove.py      # MCAP writer callback for visualization
├── tests/                       # Unit tests for core components
├── train.py                     # Main training script (Hydra @main)
├── requirements.txt
└── README.md
```

---

## 5. Required Classes to Implement

### Data Layer
* `Vision3DDataset(Dataset)`: Reads the generic JSON, loads the corresponding PNGs, structures the historical frame logic, and outputs standardized dictionary/dataclass batches.
* `NuScenesConverter`: A standalone utility class (run via `scripts/convert_nuscenes.py`) to parse the `nuscenes-devkit` and output ego-centric coordinates, undistorted images, and the unified JSON.

### Model Architecture (BEVFormer POC)
* `ResNetBackbone(nn.Module)`: Standard 2D feature extractor.
* `FPNNeck(nn.Module)`: Feature Pyramid Network to align scales.
* `BEVEncoder(nn.Module)`: The core BEVFormer module. Must implement Temporal Self-Attention and Spatial Cross-Attention.
    * *Constraint:* Must use native PyTorch (e.g., `F.grid_sample`) instead of custom Deformable Attention CUDA kernels.
* `DetectionHead(nn.Module)`: Takes the BEV grid and uses a transformer decoder to output 3D bounding box predictions and class logits.

### Core Logic
* `HungarianMatcher`: Computes the optimal bipartite matching between predicted queries and ground truth boxes based on classification and box distance costs.
* `DetectionLoss(nn.Module)`: Computes Focal Loss for classification and L1/GIoU loss for the bounding box parameters `[x, y, z, w, l, h, sin(theta), cos(theta), vx, vy]`.
* `BEVMetrics`: A standalone class to accumulate predictions and calculate mAP/NDS-like metrics at the end of an epoch.

### Engine & Utilities
* `Vision3DLightningModule(pl.LightningModule)`: 
    * `__init__(self, backbone, neck, encoder, head, matcher, loss, metrics)`
    * Handles `training_step`, `validation_step`, and optimizer configuration.
* `FoxgloveMCAPLogger(pl.Callback)`: A Lightning Callback that hooks into `on_validation_epoch_end` to write ground truth and predicted bounding boxes into an `.mcap` file for easy drag-and-drop debugging in Foxglove Studio.
