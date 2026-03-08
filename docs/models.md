# Models

Vision3D's default model is a **BEVFormer-based 3D object detector** for
multi-view camera inputs. The architecture is implemented in pure PyTorch
(no custom CUDA kernels) so it runs on any hardware that supports standard
`torch.nn` and `F.grid_sample` operations.

## Overall architecture

```
Multi-view images (B × N_cams × C × H × W)
        │
        ▼
┌───────────────────┐
│   ResNetBackbone  │  Multi-scale 2-D features
└───────────────────┘
        │  list of (B·N_cams, C_i, H_i, W_i)
        ▼
┌───────────────────┐
│     FPNNeck       │  Align channel dimensions & spatial scales
└───────────────────┘
        │  list of (B·N_cams, 256, H_i, W_i)
        ▼
┌───────────────────┐
│    BEVEncoder     │  Temporal Self-Attention + Spatial Cross-Attention
└───────────────────┘
        │  (B, 256, bev_h, bev_w)
        ▼
┌───────────────────┐
│  DetectionHead    │  Transformer decoder → box predictions
└───────────────────┘
        │  BoundingBox3DPrediction
        ▼
   boxes (B, Q, 10) · scores (B, Q) · labels (B, Q)
```

All sub-modules are independently configurable via Hydra; see
[configs.md](configs.md) for the full reference.

## BEVFormerModel

**File:** `src/vision3d/models/bevformer.py`

Top-level `nn.Module` that chains backbone → neck → encoder → head.
Accepts `BatchData` (the typed dataclass) and returns
`BoundingBox3DPrediction`.

```python
model = BEVFormerModel(
    backbone=ResNetBackbone(...),
    neck=FPNNeck(...),
    encoder=BEVEncoder(...),
    head=DetectionHead(...),
)
predictions = model(batch_data)
```

### Constructor arguments

| Argument | Type | Description |
|----------|------|-------------|
| `backbone` | `ResNetBackbone` | Feature extractor |
| `neck` | `FPNNeck` | Multi-scale feature alignment |
| `encoder` | `BEVEncoder` | BEV-space encoder |
| `head` | `DetectionHead` | Object query decoder |

## ResNetBackbone

**File:** `src/vision3d/models/backbones/resnet.py`

A standard ResNet feature extractor. Exposes intermediate feature maps from
configurable stages to allow FPN fusion.

### Constructor arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `depth` | `int` | `50` | ResNet depth: 18, 34, 50, 101, or 152 |
| `out_indices` | `list[int]` | `[2, 3, 4]` | Which backbone stages to expose to the neck (see table below) |
| `pretrained` | `bool` | `True` | Initialise with ImageNet weights |
| `frozen_stages` | `int` | `1` | Number of stages frozen during training |

### Output

`list[torch.Tensor]` – one tensor per entry in `out_indices`, each with
shape `(B·N_cams, C_stage, H_stage, W_stage)`.

`out_indices` values and the corresponding ResNet-50 feature maps:

| `out_indices` value | ResNet layer | Channels | Approx. stride |
|---------------------|-------------|----------|----------------|
| 1 | `layer1` | 256 | 4× |
| 2 | `layer2` | 512 | 8× |
| 3 | `layer3` | 1024 | 16× |
| 4 | `layer4` | 2048 | 32× |

The default `[2, 3, 4]` matches the FPN `in_channels` default of
`[512, 1024, 2048]`.

## FPNNeck

**File:** `src/vision3d/models/necks/fpn.py`

Feature Pyramid Network that aligns the multi-scale backbone outputs to a
common channel dimension and fuses them top-down.

### Constructor arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `in_channels` | `list[int]` | `[512, 1024, 2048]` | Input channel counts (must match `ResNetBackbone.out_indices`) |
| `out_channels` | `int` | `256` | Unified output channels for all levels |
| `num_outs` | `int` | `4` | Number of output feature levels |

### Processing

1. **Lateral convolutions** (1×1): map each backbone level to `out_channels`.
2. **Top-down fusion**: iteratively upsample from coarser to finer levels and
   add features.
3. **Output convolutions** (3×3): smooth each level independently.

### Output

`list[torch.Tensor]` of length `num_outs`, each with shape
`(B·N_cams, out_channels, H_i, W_i)`.

## BEVEncoder

**File:** `src/vision3d/models/encoders/bev_encoder.py`

The core BEVFormer module. Maintains a learnable 2-D BEV query grid and
updates it through stacked `BEVEncoderLayer` blocks, each containing:

- **Temporal Self-Attention (TSA):** multi-head attention between the current
  BEV queries and the previous timestep's BEV features, enabling the model to
  exploit motion cues.
- **Spatial Cross-Attention (SCA):** projects 3-D reference points to 2-D
  image coordinates and samples multi-view features via `F.grid_sample`.
- **Feed-Forward Network (FFN):** two-layer MLP with residual connection.

### Constructor arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `bev_h` | `int` | `200` | BEV grid height in cells |
| `bev_w` | `int` | `200` | BEV grid width in cells |
| `embed_dims` | `int` | `256` | Transformer embedding dimension |
| `num_layers` | `int` | `6` | Number of encoder layers |
| `num_heads` | `int` | `8` | Attention heads |
| `num_points` | `int` | `4` | Sampling points per head (SCA) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `pc_range` | `list[float]` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` | Perception cube `[x_min, y_min, z_min, x_max, y_max, z_max]` (m) |

### Output

`torch.Tensor` of shape `(B, embed_dims, bev_h, bev_w)`.

## DetectionHead

**File:** `src/vision3d/models/heads/detection_head.py`

A DETR-style transformer decoder. Uses learnable object queries to
iteratively attend to the BEV feature map and predict 3-D bounding boxes.

### Constructor arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `num_classes` | `int` | `10` | Number of object categories |
| `in_channels` | `int` | `256` | BEV feature channel count |
| `num_queries` | `int` | `900` | Number of parallel object queries |
| `num_decoder_layers` | `int` | `6` | Transformer decoder layers |
| `num_heads` | `int` | `8` | Attention heads |
| `ffn_dim` | `int` | `1024` | FFN hidden dimension |
| `dropout` | `float` | `0.1` | Dropout probability |

### Output

`BoundingBox3DPrediction` with:
- `boxes`: `(B, num_queries, 10)` – 10-DOF box parameters
- `scores`: `(B, num_queries)` – sigmoid confidence scores
- `labels`: `(B, num_queries)` – predicted class indices

## Core logic

### HungarianMatcher

**File:** `src/vision3d/core/matchers.py`

Computes optimal bipartite matching between model predictions and ground
truth boxes using `scipy.optimize.linear_sum_assignment`. The cost matrix
is a weighted sum of:

- **Classification cost** (`cost_class`): 1 − predicted probability for the
  ground-truth class.
- **Box distance cost** (`cost_bbox`): L1 distance between predicted and
  ground-truth box parameters.

Returns a `MatchingResult` dataclass containing aligned prediction and
ground-truth index tensors.

### DetectionLoss

**File:** `src/vision3d/core/losses.py`

Computes the training loss on matched prediction/ground-truth pairs:

| Loss term | Function | Config key |
|-----------|----------|------------|
| Classification | Focal Loss | `cls_weight`, `focal_alpha`, `focal_gamma` |
| Box regression | L1 loss | `bbox_weight` |
| Box regression | GIoU loss | `giou_weight` |

### Vision3DEvaluator

**File:** `src/vision3d/core/evaluators.py`

Accumulates per-frame predictions and ground truth across a validation epoch,
then computes:

- **mAP** (mean Average Precision) at each `distance_threshold`.
- **NDS**-style composite metric.

Results are logged via the Lightning trainer as `val/mAP`.

## CameraProjector (geometry utility)

**File:** `src/vision3d/utils/geometry.py`

Projects 3-D reference points (ego frame) to 2-D pixel coordinates for each
camera view. Used internally by `BEVEncoder.SCA` to determine which image
regions to sample.

```python
projector = CameraProjector()
pixel_coords = projector.project(
    points_3d,          # (N, 3) in ego frame
    intrinsics_matrix,  # (3, 3)
    sensor2ego_translation,  # (3,)
    sensor2ego_rotation,     # (4,) quaternion
)  # returns (N, 2) pixel coordinates
```

## Lightning module

**File:** `src/vision3d/engine/lit_module.py`  
**Class:** `Vision3DLightningModule`

Wraps `BEVFormerModel` and the training-only components (matcher, loss,
evaluator) in a `pl.LightningModule`. The module is a thin orchestration
layer — it does not implement any model logic itself.

| Method | Responsibility |
|--------|---------------|
| `training_step` | Forward pass → matcher → loss → log `train/loss` |
| `validation_step` | Forward pass → accumulate predictions in evaluator |
| `on_validation_epoch_end` | Compute mAP/NDS → log `val/mAP` |
| `configure_optimizers` | AdamW + cosine LR scheduler |
