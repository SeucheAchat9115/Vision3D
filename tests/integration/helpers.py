"""Shared helpers for Vision3D integration tests.

All test modules in this package import their fixtures and model factories
from here so that construction logic is defined in exactly one place.
"""

from __future__ import annotations

import torch

from vision3d.config.schema import (
    BatchData,
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
)
from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.bevformer import BEVFormerModel
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.heads.detection_head import DetectionHead
from vision3d.models.necks.fpn import FPNNeck

# Channel counts returned by each ResNet depth for out_indices=[1,2,3]
RESNET_CHANNELS: dict[int, list[int]] = {
    18: [64, 128, 256],
    34: [64, 128, 256],
    50: [256, 512, 1024],
}


def make_camera_view(
    image_h: int = 64,
    image_w: int = 64,
    *,
    name: str = "front",
    seed: int = 0,
) -> CameraView:
    """Return a synthetic CameraView with identity-like extrinsics."""
    torch.manual_seed(seed)
    K = torch.tensor([[400.0, 0.0, image_w / 2.0], [0.0, 400.0, image_h / 2.0], [0.0, 0.0, 1.0]])
    return CameraView(
        image=torch.rand(3, image_h, image_w),
        intrinsics=CameraIntrinsics(matrix=K),
        extrinsics=CameraExtrinsics(
            translation=torch.tensor([1.5, 0.0, 1.2]),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
        ),
        name=name,
    )


def make_frame(
    num_cameras: int = 1,
    num_boxes: int = 3,
    num_classes: int = 5,
    image_h: int = 64,
    image_w: int = 64,
    *,
    frame_id: str = "test_frame",
    seed: int = 0,
    with_targets: bool = True,
) -> FrameData:
    """Return a synthetic FrameData for integration testing."""
    cam_names = [f"cam_{i}" for i in range(num_cameras)]
    cameras = {
        name: make_camera_view(image_h, image_w, name=name, seed=seed + i)
        for i, name in enumerate(cam_names)
    }
    targets: BoundingBox3DTarget | None = None
    if with_targets:
        torch.manual_seed(seed + 100)
        targets = BoundingBox3DTarget(
            boxes=torch.randn(num_boxes, 10),
            labels=torch.randint(0, num_classes, (num_boxes,)),
            instance_ids=[f"{frame_id}_id_{i}" for i in range(num_boxes)],
        )
    return FrameData(
        frame_id=frame_id,
        timestamp=float(seed),
        cameras=cameras,
        targets=targets,
    )


def make_batch(
    batch_size: int = 2,
    num_cameras: int = 1,
    num_boxes: int = 3,
    num_classes: int = 5,
    image_h: int = 64,
    image_w: int = 64,
    *,
    with_targets: bool = True,
) -> BatchData:
    """Return a synthetic BatchData."""
    frames = [
        make_frame(
            num_cameras=num_cameras,
            num_boxes=num_boxes,
            num_classes=num_classes,
            image_h=image_h,
            image_w=image_w,
            frame_id=f"frame_{i}",
            seed=i,
            with_targets=with_targets,
        )
        for i in range(batch_size)
    ]
    return BatchData(batch_size=batch_size, frames=frames)


def make_small_model(
    backbone_depth: int = 18,
    out_indices: list[int] | None = None,
    embed_dims: int = 32,
    bev_h: int = 4,
    bev_w: int = 4,
    num_classes: int = 5,
    num_queries: int = 10,
    num_bev_layers: int = 1,
) -> BEVFormerModel:
    """Return a small BEVFormerModel suitable for fast CPU integration tests."""
    if out_indices is None:
        out_indices = [1, 2, 3]
    all_channels = RESNET_CHANNELS[backbone_depth]
    in_channels = [all_channels[i - 1] for i in out_indices]
    backbone = ResNetBackbone(depth=backbone_depth, pretrained=False, out_indices=out_indices)
    neck = FPNNeck(in_channels=in_channels, out_channels=embed_dims, num_outs=len(out_indices))
    encoder = BEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        num_layers=num_bev_layers,
        num_heads=4,
        num_points=2,
        dropout=0.0,
    )
    head = DetectionHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        num_queries=num_queries,
        num_decoder_layers=1,
        num_heads=4,
        ffn_dim=64,
    )
    return BEVFormerModel(backbone=backbone, neck=neck, encoder=encoder, head=head)
