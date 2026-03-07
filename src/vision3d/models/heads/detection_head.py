"""
3-D detection head for Vision3D.

Provides `DetectionHead`, which takes the BEV feature map produced by
`BEVEncoder` and uses a transformer decoder to regress 3-D bounding box
parameters and class logits for a fixed set of detection queries.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from vision3d.config.schema import BoundingBox3DPrediction


class DetectionHead(nn.Module):
    """Transformer-decoder-based 3-D object detection head.

    Architecture:
      1. A learnable set of `num_queries` object queries.
      2. A stack of `num_decoder_layers` transformer decoder layers, each
         cross-attending to the BEV feature map.
      3. Prediction MLPs applied to the final decoder output:
         - **Box regression MLP**: outputs the 10-DOF box parameters
           [x, y, z, w, l, h, sin(θ), cos(θ), vx, vy].
         - **Classification MLP**: outputs `num_classes` raw logits.

    The head returns a `BoundingBox3DPrediction` dataclass containing all
    `num_queries` predictions (before any NMS or score thresholding), suitable
    for direct input to `HungarianMatcher` and `DetectionLoss`.

    Args:
        num_classes: Number of object categories.
        in_channels: Channel dimension of the incoming BEV feature map.
        num_queries: Number of object detection queries per frame.
        num_decoder_layers: Depth of the transformer decoder.
        num_heads: Number of attention heads in each decoder layer.
        ffn_dim: Hidden dimension of the feed-forward network in each decoder layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 256,
        num_queries: int = 900,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO: create learnable object query embedding: nn.Embedding(num_queries, in_channels)
        # TODO: create learnable object query positional embedding (same shape)
        # TODO: build a TransformerDecoder (nn.TransformerDecoder) with num_decoder_layers layers
        # TODO: create box regression MLP: in_channels → hidden → 10 outputs
        # TODO: create classification MLP: in_channels → hidden → num_classes outputs
        raise NotImplementedError

    def forward(
        self,
        bev_features: torch.Tensor,
    ) -> BoundingBox3DPrediction:
        """Decode object queries against BEV features to produce predictions.

        Args:
            bev_features: BEV feature map of shape (B, in_channels, bev_h, bev_w).

        Returns:
            `BoundingBox3DPrediction` with:
              - `boxes`: shape (B, num_queries, 10)
              - `scores`: shape (B, num_queries) — softmax or sigmoid probabilities
              - `labels`: shape (B, num_queries) — argmax class predictions
        """
        # TODO: flatten BEV spatial dims: (B, C, H, W) → (H*W, B, C) as memory
        # TODO: expand object queries and positional embeddings to batch size
        # TODO: run transformer decoder with queries attending to BEV memory
        # TODO: apply box regression MLP to decoder output → (B, num_queries, 10)
        # TODO: apply classification MLP → (B, num_queries, num_classes)
        # TODO: compute scores via sigmoid (multi-label) or softmax (single-label)
        # TODO: compute labels via argmax over class dimension
        # TODO: wrap results in BoundingBox3DPrediction and return
        raise NotImplementedError
