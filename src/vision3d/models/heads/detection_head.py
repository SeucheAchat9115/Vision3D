"""
3-D detection head for Vision3D.

Provides `DetectionHead`, which takes the BEV feature map produced by
`BEVEncoder` and uses a transformer decoder to regress 3-D bounding box
parameters and class logits for a fixed set of detection queries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vision3d.config.schema import BoundingBox3DPrediction


class DetectionHead(nn.Module):
    """Transformer-decoder-based 3-D object detection head."""

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
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, in_channels)
        self.query_pos = nn.Embedding(num_queries, in_channels)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.box_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 10),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, bev_features: torch.Tensor) -> BoundingBox3DPrediction:
        """Decode object queries against BEV features to produce predictions."""
        B, C, H, W = bev_features.shape
        memory = bev_features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        queries = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        pos = self.query_pos.weight.unsqueeze(1).expand(-1, B, -1)
        queries = queries + pos
        decoded = self.decoder(queries, memory)  # (Q, B, C)
        decoded = decoded.permute(1, 0, 2)  # (B, Q, C)
        boxes = self.box_head(decoded)  # (B, Q, 10)
        logits = self.cls_head(decoded)  # (B, Q, num_classes)
        scores = torch.sigmoid(logits)
        labels = scores.argmax(dim=-1)
        max_scores = scores.max(dim=-1).values
        return BoundingBox3DPrediction(boxes=boxes, scores=max_scores, labels=labels)
