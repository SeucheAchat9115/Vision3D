# Data Loading

This document covers how Vision3D ingests, filters, augments, and batches
data during training and validation.

## Overview

```
disk (JSON + images)
        │
        ▼
  JsonLoader          reads & validates per-frame JSON
        │
        ▼
  ImageLoader         loads & resizes multi-view PNGs concurrently
        │
        ▼
  ImageFilter         rejects out-of-ODD frames
        │
        ▼
  BoxFilter           removes invalid / out-of-range annotations
        │
        ▼
  DataAugmenter       3-D + 2-D synchronized augmentation
        │
        ▼
  Vision3DDataset     __getitem__ → FrameData
        │
  collate_fn
        ▼
  BatchData           passed to LightningModule
```

All intermediate objects are **typed dataclasses** defined in
`src/vision3d/config/schema.py`. There are no plain Python dicts after the
loader stage.

## Runtime dataclass hierarchy

```python
@dataclass
class BatchData:
    batch_size: int
    frames: list[FrameData]

@dataclass
class FrameData:
    frame_id: str
    timestamp: float
    cameras: dict[str, CameraView]
    targets: BoundingBox3DTarget | None
    predictions: BoundingBox3DPrediction | None   # filled by model
    matches: MatchingResult | None                # filled by matcher
    past_frames: list[FrameData]

@dataclass
class CameraView:
    image: torch.Tensor          # (C, H, W)
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    name: str

@dataclass
class CameraIntrinsics:
    matrix: torch.Tensor         # (3, 3)

@dataclass
class CameraExtrinsics:
    translation: torch.Tensor    # (3,)
    rotation: torch.Tensor       # (4,)  quaternion [w, x, y, z]

@dataclass
class BoundingBox3DTarget:
    boxes: torch.Tensor          # (N, 10)  10-DOF format
    labels: torch.Tensor         # (N,)
    instance_ids: list[str]

@dataclass
class BoundingBox3DPrediction:
    boxes: torch.Tensor          # (M, 10)
    scores: torch.Tensor         # (M,)
    labels: torch.Tensor         # (M,)

@dataclass
class MatchingResult:
    pred_indices: torch.Tensor   # (K,)
    gt_indices: torch.Tensor     # (K,)
```

`BatchData.to(device)` recursively moves every nested tensor to the target
device in a single call.

## Vision3DDataset

**File:** `src/vision3d/data/dataset.py`

`torch.utils.data.Dataset` subclass that orchestrates the loading pipeline.

```python
dataset = Vision3DDataset(
    data_root="data/nuscenes_v3d",
    split="train",
    num_past_frames=2,
    image_size=(900, 1600),
    box_filter=BoxFilter(...),
    image_filter=ImageFilter(...),
    augmenter=DataAugmenter(...),
)
```

`__getitem__` performs the following steps:

1. Resolve the frame JSON path from `data_root/<split>/`.
2. Call `JsonLoader` to parse the JSON and validate the schema.
3. Call `ImageLoader` to load all camera images concurrently.
4. Apply `ImageFilter`; if the frame is rejected, skip to the next frame.
5. Apply `BoxFilter` to the annotations.
6. Apply `DataAugmenter` (training split only).
7. Return a fully populated `FrameData` object.

### collate_fn

```python
DataLoader(..., collate_fn=Vision3DDataset.collate_fn)
```

Packs a list of `FrameData` into a single `BatchData`. Tensors are **not**
stacked because frames may have different numbers of annotations.

## JsonLoader

**File:** `src/vision3d/data/loaders.py`

Reads, parses, and validates a single frame JSON file.

- Checks that all required top-level keys are present.
- Validates intrinsic matrix shape (3×3).
- Validates extrinsic translation shape (3,) and quaternion shape (4,).
- Verifies each `bbox_3d` has exactly 10 values.
- Returns a validated `dict` that is further assembled into dataclasses by
  `Vision3DDataset`.

## ImageLoader

**File:** `src/vision3d/data/loaders.py`

Loads multiple camera images concurrently using a `ThreadPoolExecutor`.

```python
loader = ImageLoader(target_size=(900, 1600), num_workers=4)
images: dict[str, torch.Tensor] = loader.load(camera_paths)
```

- Reads PNG or JPG files via `Pillow`.
- Resizes to `target_size` using bilinear interpolation.
- Normalises with ImageNet statistics:
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Returns a `dict[camera_name → Tensor(C, H, W)]`.

## BoxFilter

**File:** `src/vision3d/data/filters.py`

Removes ground-truth boxes that do not meet quality criteria.

| Filter | Config key | Description |
|--------|-----------|-------------|
| Distance | `max_distance` | Discard boxes whose horizontal range `√(x²+y²)` exceeds this value (m); only ground-plane axes are used, matching typical autonomous-driving range conventions |
| Point count | `min_points` | Discard boxes with fewer LiDAR points than this threshold |
| Physical validity | — | Discard boxes where any of `w, l, h ≤ 0` |
| Class allow-list | `allowed_classes` | Retain only listed class names; `null` keeps all |

Returns a `BoundingBox3DTarget` with only the surviving boxes.

## ImageFilter

**File:** `src/vision3d/data/filters.py`

Decides whether an entire frame should be skipped based on per-frame metadata.

| Check | Config key | Description |
|-------|-----------|-------------|
| Weather | `rejected_weather` | Skip frames whose `metadata.weather` is in this list |
| Annotations | `require_annotations` | Skip frames with zero annotations |

Returns a boolean. `Vision3DDataset` moves to the next index when `False`.

## DataAugmenter

**File:** `src/vision3d/data/augmentations.py`

Applies **synchronised** 3-D and 2-D augmentations. All camera geometry is
kept consistent with box transformations — no separate augmentation for images
and boxes.

| Augmentation | Config keys | Effect |
|--------------|------------|--------|
| Global yaw rotation | `global_rot_range` | Rotates box centres, velocities, yaw angles, and camera extrinsics by the same random angle |
| Global scale | `global_scale_range` | Scales box dimensions and translations uniformly; adjusts camera extrinsics accordingly |
| Horizontal flip | `flip_prob` | Mirrors the ego frame and all camera orientations; negates `y` positions and yaw angles |
| Colour jitter | `color_jitter_prob` | Random brightness and contrast adjustment (images only; no geometry change) |

The contract is: after augmentation, projecting ego-frame box centres with
the updated camera extrinsics must yield the same pixel coordinates as before
augmentation with the original extrinsics.

## DataLoader configuration

```python
from torch.utils.data import DataLoader
from vision3d.data.dataset import Vision3DDataset

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    persistent_workers=cfg.num_workers > 0,
    collate_fn=Vision3DDataset.collate_fn,
)
```

Recommended settings:

| Setting | Train | Val |
|---------|-------|-----|
| `shuffle` | `True` | `False` |
| `num_workers` | 4–8 | 4–8 |
| `persistent_workers` | `True` (if workers > 0) | `True` |
| `pin_memory` | `True` (if GPU) | `True` |

## Data lifecycle contract

The following table shows which fields of `FrameData` are populated at each
pipeline stage:

| Stage | `cameras` | `targets` | `predictions` | `matches` |
|-------|-----------|-----------|---------------|-----------|
| After `Vision3DDataset.__getitem__` | ✅ | ✅ | — | — |
| After augmentation | ✅ (updated geometry) | ✅ (updated boxes) | — | — |
| After model forward pass | ✅ | ✅ | ✅ | — |
| After `HungarianMatcher` | ✅ | ✅ | ✅ | ✅ |
