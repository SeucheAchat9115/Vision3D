"""
Top-level BEVFormer model for Vision3D.

Provides `BEVFormerModel`, which combines backbone, neck, BEV encoder, and
detection head into a single `nn.Module` that is passed to the Lightning module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vision3d.config.schema import BatchData, BoundingBox3DPrediction
from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.heads.detection_head import DetectionHead
from vision3d.models.necks.fpn import FPNNeck
from vision3d.utils.geometry import CameraProjector


class BEVFormerModel(nn.Module):
    """High-level BEVFormer model that encapsulates the full forward pass."""

    def __init__(
        self,
        backbone: ResNetBackbone,
        neck: FPNNeck,
        encoder: BEVEncoder,
        head: DetectionHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: BatchData,
        prev_bev: torch.Tensor | None = None,
    ) -> tuple[BoundingBox3DPrediction, torch.Tensor]:
        """Run the full BEVFormer forward pass on a batch."""
        frames = batch.frames
        all_images = []
        all_intrinsics = []
        all_extrinsics = []
        for frame in frames:
            cam_views = list(frame.cameras.values())
            for cam in cam_views:
                all_images.append(cam.image)
            frame_intrinsics = torch.stack([c.intrinsics.matrix for c in cam_views])
            all_intrinsics.append(frame_intrinsics)
            frame_extrinsics = []
            for c in cam_views:
                R = CameraProjector.quaternion_to_rotation_matrix(
                    c.extrinsics.rotation.unsqueeze(0)
                ).squeeze(0)
                T = torch.eye(4, device=R.device, dtype=R.dtype)
                T[:3, :3] = R
                T[:3, 3] = c.extrinsics.translation
                frame_extrinsics.append(T)
            all_extrinsics.append(torch.stack(frame_extrinsics))
        images = torch.stack(all_images)
        intrinsics = torch.stack(all_intrinsics)
        extrinsics = torch.stack(all_extrinsics)
        features = self.backbone(images)
        features = self.neck(features)
        bev = self.encoder(features, intrinsics, extrinsics, prev_bev)
        predictions = self.head(bev)
        new_bev = bev.flatten(2).permute(2, 0, 1)
        return predictions, new_bev
