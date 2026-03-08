# Configuration

Vision3D uses [Hydra](https://hydra.cc/) for configuration management.
All config values are backed by **strictly typed Python dataclasses** defined
in `src/vision3d/config/schema.py`, which prevents silent key-name typos and
enables IDE auto-completion.

## Config directory layout

```
configs/
├── train.yaml            # Root entry point (merged with defaults below)
├── model/
│   └── bevformer.yaml    # BEVFormer architecture + training hyperparameters
├── dataset/
│   ├── dummy.yaml        # Synthetic dummy dataset
│   └── nuscenes.yaml     # NuScenes (after offline conversion)
└── experiment/
    └── default.yaml      # Placeholder – no overrides applied
```

## Root config — `configs/train.yaml`

```yaml
defaults:
  - model: bevformer
  - dataset: dummy
  - experiment: default
  - _self_

max_epochs: 24
batch_size: 1
num_workers: 4
seed: 42
output_dir: outputs/${now:%Y-%m-%d_%H-%M-%S}
```

| Key | Default | Description |
|-----|---------|-------------|
| `max_epochs` | `24` | Total training epochs |
| `batch_size` | `1` | Samples per GPU per step |
| `num_workers` | `4` | DataLoader worker processes |
| `seed` | `42` | Global random seed (passed to `pl.seed_everything`) |
| `output_dir` | `outputs/<timestamp>` | Root for checkpoints, logs, and MCAP files |

## Model config — `configs/model/bevformer.yaml`

Instantiates `Vision3DLightningModule`, which wraps `BEVFormerModel` together
with the loss, matcher, and evaluator.

```yaml
model:
  _target_: vision3d.engine.lit_module.Vision3DLightningModule

  model:
    _target_: vision3d.models.bevformer.BEVFormerModel

    backbone:
      _target_: vision3d.models.backbones.ResNetBackbone
      depth: 50
      out_indices: [2, 3, 4]
      pretrained: true
      frozen_stages: 1

    neck:
      _target_: vision3d.models.necks.FPNNeck
      in_channels: [512, 1024, 2048]
      out_channels: 256
      num_outs: 4

    encoder:
      _target_: vision3d.models.encoders.BEVEncoder
      bev_h: 200
      bev_w: 200
      embed_dims: 256
      num_layers: 6
      num_heads: 8
      num_points: 4
      dropout: 0.1
      pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    head:
      _target_: vision3d.models.heads.DetectionHead
      num_classes: 10
      in_channels: 256
      num_queries: 900
      num_decoder_layers: 6
      num_heads: 8
      ffn_dim: 1024
      dropout: 0.1

  loss:
    _target_: vision3d.core.losses.DetectionLoss
    num_classes: 10
    cls_weight: 2.0
    bbox_weight: 0.25
    giou_weight: 0.25
    focal_alpha: 0.25
    focal_gamma: 2.0

  matcher:
    _target_: vision3d.core.matchers.HungarianMatcher
    cost_class: 2.0
    cost_bbox: 0.25

  evaluator:
    _target_: vision3d.core.evaluators.Vision3DEvaluator
    num_classes: 10
    eval_range: 50.0
    distance_thresholds: [0.5, 1.0, 2.0, 4.0]

  learning_rate: 2.0e-4
  weight_decay: 1.0e-4
  max_epochs: 24
```

### Backbone keys

| Key | Default | Description |
|-----|---------|-------------|
| `depth` | `50` | ResNet depth: 18, 34, 50, 101, or 152 |
| `out_indices` | `[2, 3, 4]` | Which backbone stages to expose to the neck (0-indexed) |
| `pretrained` | `true` | Load ImageNet weights |
| `frozen_stages` | `1` | Number of early stages to freeze during training |

### Neck keys

| Key | Default | Description |
|-----|---------|-------------|
| `in_channels` | `[512, 1024, 2048]` | Channel counts of backbone feature maps |
| `out_channels` | `256` | Unified channel count for all FPN output levels |
| `num_outs` | `4` | Number of FPN output levels |

### Encoder keys

| Key | Default | Description |
|-----|---------|-------------|
| `bev_h` | `200` | Height of the BEV grid in cells |
| `bev_w` | `200` | Width of the BEV grid in cells |
| `embed_dims` | `256` | Transformer embedding dimension |
| `num_layers` | `6` | Number of BEVFormer encoder layers |
| `num_heads` | `8` | Attention heads per layer |
| `num_points` | `4` | Sampling points per head in spatial cross-attention |
| `dropout` | `0.1` | Dropout probability |
| `pc_range` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` | `[x_min, y_min, z_min, x_max, y_max, z_max]` in metres |

### Head keys

| Key | Default | Description |
|-----|---------|-------------|
| `num_classes` | `10` | Number of object classes |
| `in_channels` | `256` | BEV feature channel count |
| `num_queries` | `900` | Number of learnable object queries |
| `num_decoder_layers` | `6` | Transformer decoder layers |
| `num_heads` | `8` | Attention heads |
| `ffn_dim` | `1024` | Feed-forward network hidden dimension |
| `dropout` | `0.1` | Dropout probability |

### Loss keys

| Key | Default | Description |
|-----|---------|-------------|
| `num_classes` | `10` | Must match `head.num_classes` |
| `cls_weight` | `2.0` | Weight for focal classification loss |
| `bbox_weight` | `0.25` | Weight for L1 box regression loss |
| `giou_weight` | `0.25` | Weight for GIoU box regression loss |
| `focal_alpha` | `0.25` | Focal loss α |
| `focal_gamma` | `2.0` | Focal loss γ |

### Matcher keys

| Key | Default | Description |
|-----|---------|-------------|
| `cost_class` | `2.0` | Classification cost weight in assignment matrix |
| `cost_bbox` | `0.25` | Bounding-box distance cost weight |

### Evaluator keys

| Key | Default | Description |
|-----|---------|-------------|
| `num_classes` | `10` | Number of classes to evaluate |
| `eval_range` | `50.0` | Maximum detection range in metres |
| `distance_thresholds` | `[0.5, 1.0, 2.0, 4.0]` | Centre-distance thresholds for mAP |

## Dataset configs

### `configs/dataset/nuscenes.yaml`

```yaml
dataset:
  _target_: vision3d.data.dataset.Vision3DDataset
  data_root: data/nuscenes_v3d
  split: train
  num_past_frames: 2
  image_size: [900, 1600]

  box_filter:
    _target_: vision3d.data.filters.BoxFilter
    max_distance: 50.0
    min_points: 1
    allowed_classes:
      - car
      - truck
      - bus
      - pedestrian
      - motorcycle
      - bicycle
      - trailer
      - construction_vehicle
      - traffic_cone
      - barrier

  image_filter:
    _target_: vision3d.data.filters.ImageFilter
    rejected_weather: []
    require_annotations: true

  augmenter:
    _target_: vision3d.data.augmentations.DataAugmenter
    global_rot_range: [-0.3925, 0.3925]
    global_scale_range: [0.95, 1.05]
    flip_prob: 0.5
    color_jitter_prob: 0.5
```

### `configs/dataset/dummy.yaml`

Same structure as NuScenes but uses `data/dummy`, disables point-count
filtering (`min_points: 0`), and does not require annotations.

### Dataset keys

| Key | Default | Description |
|-----|---------|-------------|
| `data_root` | — | Path to the converted dataset root |
| `split` | `"train"` | `"train"`, `"val"`, or `"test"` |
| `num_past_frames` | `2` | Past frames to load for temporal attention |
| `image_size` | `[900, 1600]` | Target image size as `[height, width]` |
| `box_filter.max_distance` | `50.0` | Discard boxes beyond this range (m) |
| `box_filter.min_points` | `1` | Minimum LiDAR points (0 disables check) |
| `box_filter.allowed_classes` | `null` | Class allow-list; `null` keeps all classes |
| `image_filter.rejected_weather` | `[]` | Metadata weather values to reject |
| `image_filter.require_annotations` | `true` | Skip frames with no annotations |
| `augmenter.global_rot_range` | `[-0.39, 0.39]` | Yaw rotation range in radians |
| `augmenter.global_scale_range` | `[0.95, 1.05]` | Uniform scale factor range |
| `augmenter.flip_prob` | `0.5` | Horizontal flip probability |
| `augmenter.color_jitter_prob` | `0.5` | Colour jitter probability |

## Experiment configs

Experiment configs in `configs/experiment/` exist purely to override values
from the base model and dataset configs. Create a new file, e.g.
`configs/experiment/exp_01.yaml`, and add only the keys you want to change:

```yaml
# configs/experiment/exp_01.yaml
model:
  encoder:
    num_layers: 3
  learning_rate: 1.0e-4

max_epochs: 12
batch_size: 2
```

Then activate it with:

```bash
python train.py experiment=exp_01
```

## Running training

```bash
# Default: dummy dataset, BEVFormer, 24 epochs
python train.py

# Switch to NuScenes
python train.py dataset=nuscenes

# Full override example
python train.py dataset=nuscenes max_epochs=12 batch_size=4 \
  model.learning_rate=1e-4

# Hyperparameter sweep (Hydra multirun)
python train.py --multirun \
  model.learning_rate=1e-4,2e-4 \
  max_epochs=12,24
```

Outputs are written to `outputs/<timestamp>/` and include:
- TensorBoard event files
- Lightning checkpoints (best `val/mAP`)
- Foxglove `.mcap` files (visualisation)

## Hydra output directory

Hydra automatically changes the working directory to `output_dir` for each
run. Override the output root:

```bash
python train.py output_dir=/mnt/storage/runs/exp_42
```

For multi-run sweeps each sub-run is placed in
`outputs/multirun/<timestamp>/<job_num>/`.
