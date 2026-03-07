"""
Internal data interface dataclasses and Hydra configuration schema for Vision3D.

This module defines two groups of dataclasses:
  1. Hydra config dataclasses (e.g. BackboneConfig) used for strict typed
     configuration of all sub-components via Hydra.
  2. Runtime data interface dataclasses (e.g. FrameData, BatchData) that act
     as strictly typed containers passed through the entire pipeline.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Runtime data interface dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CameraExtrinsics:
    """Sensor-to-ego transformation for a single camera.

    Attributes:
        translation: 3-D translation vector from sensor to ego frame.
            Shape: (3,)
        rotation: Unit quaternion (w, x, y, z) rotating sensor to ego frame.
            Shape: (4,)
    """

    translation: torch.Tensor  # Shape: (3,)
    rotation: torch.Tensor  # Shape: (4,) - quaternion [w, x, y, z]


@dataclass
class CameraIntrinsics:
    """Pinhole camera projection matrix.

    Attributes:
        matrix: 3×3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
            Shape: (3, 3)
    """

    matrix: torch.Tensor  # Shape: (3, 3)


@dataclass
class CameraView:
    """All data associated with a single camera view at one timestamp.

    Attributes:
        image: Pre-undistorted RGB image tensor. Shape: (C, H, W)
        intrinsics: Pinhole camera intrinsics.
        extrinsics: Sensor-to-ego transformation.
        name: Human-readable camera name, e.g. "front" or "back_left".
    """

    image: torch.Tensor  # Shape: (C, H, W)
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    name: str


@dataclass
class BoundingBox3DTarget:
    """Ground-truth 3-D bounding boxes for a single frame.

    Attributes:
        boxes: Box parameters in the 10-DOF ego-centric format
            [x, y, z, w, l, h, sin(θ), cos(θ), vx, vy].
            Shape: (N, 10)
        labels: Integer class index for each box. Shape: (N,)
        instance_ids: Unique tracking / instance identifiers. Length N.
    """

    boxes: torch.Tensor  # Shape: (N, 10)
    labels: torch.Tensor  # Shape: (N,)
    instance_ids: List[str]


@dataclass
class BoundingBox3DPrediction:
    """Model predictions for a single frame.

    Attributes:
        boxes: Predicted box parameters in the 10-DOF format. Shape: (M, 10)
        scores: Confidence score per box. Shape: (M,)
        labels: Predicted integer class index per box. Shape: (M,)
    """

    boxes: torch.Tensor  # Shape: (M, 10)
    scores: torch.Tensor  # Shape: (M,)
    labels: torch.Tensor  # Shape: (M,)


@dataclass
class MatchingResult:
    """Result of bipartite matching between predictions and ground truth.

    Attributes:
        pred_indices: Indices (into predictions) of the matched pairs. Shape: (K,)
        gt_indices: Indices (into ground truth) of the matched pairs. Shape: (K,)
    """

    pred_indices: torch.Tensor  # Shape: (K,)
    gt_indices: torch.Tensor  # Shape: (K,)


@dataclass
class FrameData:
    """Master container for all data associated with a single timestamp.

    Populated progressively through the pipeline:
      - Dataset: cameras + targets
      - Forward pass: predictions
      - Matcher: matches

    Attributes:
        frame_id: Unique string identifier for this frame.
        timestamp: Unix timestamp in seconds.
        cameras: Dict keyed by camera name containing the CameraView objects.
        targets: Ground-truth annotations (None during inference).
        predictions: Model output (populated after forward pass).
        matches: Bipartite matching result (populated during training).
        past_frames: Ordered list of prior FrameData for temporal attention.
    """

    frame_id: str
    timestamp: float
    cameras: Dict[str, CameraView]
    targets: Optional[BoundingBox3DTarget] = None
    predictions: Optional[BoundingBox3DPrediction] = None
    matches: Optional[MatchingResult] = None
    past_frames: List[FrameData] = field(default_factory=list)


@dataclass
class BatchData:
    """Collated output from the DataLoader passed through the full pipeline.

    Attributes:
        batch_size: Number of frames in this batch.
        frames: List of FrameData objects, one per sample in the batch.
    """

    batch_size: int
    frames: List[FrameData]

    def to(self, device: torch.device) -> BatchData:
        """Recursively move all nested tensors to *device*.

        Should traverse the dataclass hierarchy and call `.to(device)` on every
        `torch.Tensor` found inside `frames` (including tensors inside
        CameraView, CameraIntrinsics, CameraExtrinsics, BoundingBox3DTarget,
        BoundingBox3DPrediction, MatchingResult, and past_frames recursively).
        """
        # TODO: implement recursive tensor relocation
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hydra configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    """Hydra configuration for ResNetBackbone.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        depth: ResNet depth (18, 34, 50, 101, …).
        out_indices: Feature map stages to expose to the neck.
    """

    _target_: str = "vision3d.models.backbones.ResNetBackbone"
    depth: int = 50
    out_indices: List[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class NeckConfig:
    """Hydra configuration for FPNNeck.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        in_channels: Channel sizes of each input feature map from the backbone.
        out_channels: Unified output channel count for all FPN levels.
    """

    _target_: str = "vision3d.models.necks.FPNNeck"
    in_channels: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    out_channels: int = 256


@dataclass
class EncoderConfig:
    """Hydra configuration for BEVEncoder.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        bev_h: Height (rows) of the BEV grid in voxels.
        bev_w: Width (columns) of the BEV grid in voxels.
        embed_dims: Embedding dimension used throughout the transformer.
        num_layers: Number of BEVFormer encoder layers.
    """

    _target_: str = "vision3d.models.encoders.BEVEncoder"
    bev_h: int = 200
    bev_w: int = 200
    embed_dims: int = 256
    num_layers: int = 6


@dataclass
class HeadConfig:
    """Hydra configuration for DetectionHead.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        num_classes: Number of object classes to detect.
        in_channels: Channel dimension of the BEV feature map.
    """

    _target_: str = "vision3d.models.heads.DetectionHead"
    num_classes: int = 10
    in_channels: int = 256


@dataclass
class LossConfig:
    """Hydra configuration for DetectionLoss.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        cls_weight: Weighting factor for the classification (focal) loss term.
        bbox_weight: Weighting factor for the bounding-box regression loss term.
    """

    _target_: str = "vision3d.core.losses.DetectionLoss"
    cls_weight: float = 2.0
    bbox_weight: float = 0.25


@dataclass
class MatcherConfig:
    """Hydra configuration for HungarianMatcher.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        cost_class: Weighting of classification cost in the assignment matrix.
        cost_bbox: Weighting of bounding-box distance cost in the assignment matrix.
    """

    _target_: str = "vision3d.core.matchers.HungarianMatcher"
    cost_class: float = 2.0
    cost_bbox: float = 0.25


@dataclass
class EvaluatorConfig:
    """Hydra configuration for Vision3DEvaluator.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        eval_range: Maximum distance (metres) at which detections are evaluated.
    """

    _target_: str = "vision3d.core.evaluators.Vision3DEvaluator"
    eval_range: float = 50.0


@dataclass
class BEVFormerModelConfig:
    """Hydra configuration for BEVFormerModel.

    Bundles the backbone, neck, encoder, and head sub-configs so that Hydra
    can recursively instantiate the full `BEVFormerModel` as a single unit.
    This config is passed as the `model` argument to `LitModuleConfig`.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        backbone: Configuration for the ResNet backbone.
        neck: Configuration for the FPN neck.
        encoder: Configuration for the BEV encoder.
        head: Configuration for the detection head.
    """

    _target_: str = "vision3d.models.bevformer.BEVFormerModel"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: HeadConfig = field(default_factory=HeadConfig)


@dataclass
class LitModuleConfig:
    """Top-level Hydra configuration for Vision3DLightningModule.

    Accepts a pre-assembled `BEVFormerModel` config alongside the
    training-only components (matcher, loss, evaluator). Hydra instantiates
    the model first and passes it to the Lightning module, keeping the two
    concerns cleanly separated.
    """

    _target_: str = "vision3d.engine.lit_module.Vision3DLightningModule"
    model: BEVFormerModelConfig = field(default_factory=BEVFormerModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)


@dataclass
class DatasetConfig:
    """Hydra configuration for Vision3DDataset.

    Attributes:
        _target_: Fully qualified class path for Hydra instantiation.
        data_root: Root directory containing image folders and JSON files.
        split: Dataset split, e.g. "train", "val", or "test".
        num_cameras: Expected number of cameras per frame.
        num_past_frames: How many past frames to load for temporal attention.
        image_size: Target (height, width) to resize images to.
    """

    _target_: str = "vision3d.data.dataset.Vision3DDataset"
    data_root: str = "data/"
    split: str = "train"
    num_cameras: int = 6
    num_past_frames: int = 2
    image_size: List[int] = field(default_factory=lambda: [900, 1600])


@dataclass
class TrainConfig:
    """Root Hydra training configuration.

    Bundles the dataset and model configurations used by the main train.py
    entry point, plus generic Lightning Trainer hyperparameters.
    """

    model: LitModuleConfig = field(default_factory=LitModuleConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    max_epochs: int = 24
    batch_size: int = 1
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    output_dir: str = "outputs/"
