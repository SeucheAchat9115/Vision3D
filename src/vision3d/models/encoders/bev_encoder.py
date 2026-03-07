"""
BEVFormer encoder for Vision3D.

Provides `BEVEncoder`, which is the core of the BEVFormer architecture. It
maintains a learnable Bird's-Eye-View (BEV) query grid and iteratively refines
it through stacked encoder layers that each perform:

  1. **Temporal Self-Attention (TSA)**: Cross-attends the current BEV queries
     against the BEV features from the previous timestep (aligned to the
     current ego pose) to capture temporal context.

  2. **Spatial Cross-Attention (SCA)**: Projects each BEV query's corresponding
     3-D reference point onto each camera image using `CameraProjector`, samples
     image features from the FPN feature maps with `F.grid_sample`, and
     aggregates them via multi-head attention to lift 2-D features to BEV.

Constraint: Uses only native PyTorch operations (no custom CUDA kernels).
In particular, deformable attention is approximated via `F.grid_sample` with
learnable offset predictions, avoiding the need for the `mmcv` CUDA extension.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoderLayer(nn.Module):
    """A single BEVFormer encoder layer combining TSA and SCA.

    Args:
        embed_dims: Hidden dimension used throughout the attention layers.
        num_heads: Number of attention heads.
        num_points: Number of reference points sampled per BEV query in SCA.
        dropout: Dropout probability applied after each attention block.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO: create nn.MultiheadAttention for Temporal Self-Attention (TSA)
        # TODO: create a linear projection + norm for TSA output
        # TODO: create an offset MLP for predicting sampling offsets in SCA
        # TODO: create nn.MultiheadAttention for Spatial Cross-Attention (SCA)
        # TODO: create a linear projection + norm for SCA output
        # TODO: create a feed-forward network (two-layer MLP with GELU activation)
        # TODO: create layer normalisation for the FFN residual connection
        raise NotImplementedError

    def forward(
        self,
        bev_queries: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        image_features: List[torch.Tensor],
        reference_points: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Process BEV queries through TSA → SCA → FFN.

        Args:
            bev_queries: Current BEV query tensor of shape (bev_h*bev_w, B, embed_dims).
            prev_bev: Previous timestep BEV features of the same shape, or None
                if this is the first frame.
            image_features: List of FPN feature maps, each of shape
                (B * num_cameras, C, H_l, W_l).
            reference_points: 3-D reference points in ego coordinates for each
                BEV query. Shape: (bev_h*bev_w, 3).
            camera_intrinsics: Batched intrinsic matrices. Shape: (B, num_cameras, 3, 3).
            camera_extrinsics: Batched extrinsic transforms. Shape: (B, num_cameras, 4, 4).
            spatial_shapes: Tensor listing (H_l, W_l) for each FPN level.

        Returns:
            Updated BEV query tensor of shape (bev_h*bev_w, B, embed_dims).
        """
        # TODO: if prev_bev is not None, apply TSA by cross-attending bev_queries to prev_bev
        # TODO: add & norm after TSA (residual connection)
        # TODO: project reference_points to each camera image plane using
        #       CameraProjector (or inline geometry)
        # TODO: sample multi-scale image features at projected 2-D points with F.grid_sample
        # TODO: aggregate sampled features via SCA (multi-head attention)
        # TODO: add & norm after SCA
        # TODO: apply FFN with residual and layer norm
        # TODO: return updated bev_queries
        raise NotImplementedError


class BEVEncoder(nn.Module):
    """BEVFormer encoder: lifts multi-view image features to a BEV grid.

    Maintains a learnable BEV query grid (bev_h × bev_w) and processes it
    through `num_layers` stacked `BEVEncoderLayer` modules. The final output
    is a dense BEV feature map suitable for the `DetectionHead`.

    Args:
        bev_h: Number of BEV grid rows.
        bev_w: Number of BEV grid columns.
        embed_dims: Feature dimension used throughout.
        num_layers: Number of stacked encoder layers.
        num_heads: Number of attention heads per layer.
        num_points: Number of image sampling points per query in SCA.
        dropout: Dropout probability.
        pc_range: Point-cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            that defines the physical extent of the BEV grid in metres.
    """

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        embed_dims: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        pc_range: List[float] = None,
    ) -> None:
        super().__init__()
        # TODO: store all hyperparameters
        # TODO: create learnable BEV query embedding: nn.Embedding(bev_h*bev_w, embed_dims)
        # TODO: create learnable positional embedding for BEV queries
        # TODO: build a grid of 3-D reference points in ego coordinates derived from
        #       pc_range and (bev_h, bev_w); register as a buffer
        # TODO: build nn.ModuleList of BEVEncoderLayer modules (num_layers total)
        raise NotImplementedError

    def forward(
        self,
        image_features: List[torch.Tensor],
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the BEV feature map from multi-view image features.

        Args:
            image_features: Multi-scale FPN feature maps for all cameras.
                Each tensor: (B * num_cameras, C, H_l, W_l).
            camera_intrinsics: Batched intrinsic matrices (B, num_cameras, 3, 3).
            camera_extrinsics: Batched sensor-to-ego transforms (B, num_cameras, 4, 4).
            prev_bev: BEV features from the previous frame for temporal attention.
                Shape: (bev_h*bev_w, B, embed_dims) or None.

        Returns:
            BEV feature tensor of shape (B, embed_dims, bev_h, bev_w).
        """
        # TODO: expand BEV query embeddings to (bev_h*bev_w, B, embed_dims)
        # TODO: add positional embeddings to BEV queries
        # TODO: pass queries through each BEVEncoderLayer sequentially
        # TODO: reshape output from (bev_h*bev_w, B, embed_dims) to (B, embed_dims, bev_h, bev_w)
        # TODO: return the BEV feature map
        raise NotImplementedError
