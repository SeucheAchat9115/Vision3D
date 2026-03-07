"""
Top-level BEVFormer model for Vision3D.

Provides `BEVFormerModel`, which combines backbone, neck, BEV encoder, and
detection head into a single `nn.Module` that is passed to the Lightning module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vision3d.config.schema import BatchData, BoundingBox3DPrediction
from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.heads.detection_head import DetectionHead
from vision3d.models.necks.fpn import FPNNeck


class BEVFormerModel(nn.Module):
    """High-level BEVFormer model that encapsulates the full forward pass.

    Acts as a pure container: receives sub-modules at construction time and
    composes them into the end-to-end forward pipeline. No training logic lives
    here; that responsibility belongs to `Vision3DLightningModule`.

    Pipeline:
      1. **Backbone**: Extract multi-scale 2-D features from all camera images.
      2. **Neck (FPN)**: Align feature map scales and channel dimensions.
      3. **BEV Encoder**: Lift multi-view features to a BEV grid using
         Temporal Self-Attention and Spatial Cross-Attention.
      4. **Detection Head**: Decode object queries against the BEV grid.

    Args:
        backbone: Instantiated `ResNetBackbone` module.
        neck: Instantiated `FPNNeck` module.
        encoder: Instantiated `BEVEncoder` module.
        head: Instantiated `DetectionHead` module.
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        neck: FPNNeck,
        encoder: BEVEncoder,
        head: DetectionHead,
    ) -> None:
        super().__init__()
        # TODO: assign backbone, neck, encoder, head as sub-modules (self.backbone = ...)
        raise NotImplementedError

    def forward(
        self,
        batch: BatchData,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[BoundingBox3DPrediction, torch.Tensor]:
        """Run the full BEVFormer forward pass on a batch.

        Args:
            batch: `BatchData` with all frames, camera images, and calibration.
            prev_bev: Optional previous BEV feature tensor passed to the encoder
                for temporal attention. Shape: (bev_h*bev_w, B, embed_dims).

        Returns:
            Tuple of:
              - `BoundingBox3DPrediction`: Detection predictions for the batch.
              - `torch.Tensor`: Updated BEV feature map to be used as `prev_bev`
                in the next timestep. Shape: (bev_h*bev_w, B, embed_dims).
        """
        # TODO: stack all camera images from batch.frames into a single tensor
        #       of shape (B * num_cameras, C, H, W)
        # TODO: assemble batched intrinsics (B, num_cameras, 3, 3)
        # TODO: assemble batched extrinsics (B, num_cameras, 4, 4)
        # TODO: pass stacked images through self.backbone to get multi-scale features
        # TODO: pass backbone features through self.neck (FPN)
        # TODO: pass FPN features + calibration + prev_bev to self.encoder
        # TODO: pass BEV features to self.head to get predictions
        # TODO: return (predictions, new_bev) where new_bev feeds the next forward call
        raise NotImplementedError
