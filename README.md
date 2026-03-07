# Vision3D 🚙👁️

**Vision3D** is a clean, modular, and PyTorch-native codebase for 3D object detection, built specifically for autonomous driving and robotics. 

Born out of the need for a maintainable alternative to overly complex and dependency-heavy frameworks, Vision3D prioritizes readability, strict separation of concerns, and seamless integration with modern deep learning tools.

## 🚀 Core Philosophy

* **Pure PyTorch:** Zero custom C++/CUDA extensions. We rely on native PyTorch operations to ensure cross-platform compatibility, painless installation, and easy debugging.
* **Modular by Design:** Built on **PyTorch Lightning**, the engineering (training loops, distributed setups) is strictly separated from the science (model architecture).
* **Strict Configuration:** Powered by **Hydra** and strictly typed Python Dataclasses to prevent nested dictionary hell.
* **Pre-computed Geometry:** We enforce an **ego-centric coordinate system** offline. Complex transformations are handled during dataset conversion, keeping the dataloader and training loop incredibly fast and simple.
* **Lean Dependencies:** Custom, lightweight reimplementations of essential 3D metrics (NDS, mAP) to avoid framework bloat.

## 📦 Supported Models & Modalities

**Current Modalities:**
* Multi-view Vision

**Planned Modalities:**
* LiDAR (Point Clouds)
* Radar
* Multi-sensor Fusion

**Baseline Models:**
* **BEVFormer** (Proof of Concept)

## 📂 Architecture & File Structure

The PyTorch Lightning module acts purely as a container. Hydra recursively instantiates the Backbone, Encoder, Head, Loss, and Matcher, injecting them directly into the model.

```text
Vision3D/
├── configs/                     # Hydra configuration files (train.yaml, model/, dataset/)
├── data/                        # Local generic dataset directory (PNGs + JSONs)
├── scripts/
│   └── convert_nuscenes.py      # Offline converter for nuScenes -> Vision3D format
├── src/
│   └── vision3d/
│       ├── data/                # Dataloaders and generic JSON parsers
│       ├── models/              # Backbones, Necks, Encoders (BEVFormer), and Heads
│       ├── core/                # Losses, Hungarian Matchers, and lean Metrics
│       ├── engine/              # PyTorch Lightning container (LitModule)
│       └── utils/               # Pure PyTorch workarounds (e.g., MSDA) & Foxglove loggers
├── tests/                       # Unit tests for core components
├── train.py                     # Main training entry point
└── requirements.txt
