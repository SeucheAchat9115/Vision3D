"""
Feature Pyramid Network (FPN) neck for Vision3D.

Provides `FPNNeck`, which takes the list of multi-scale feature maps produced
by the backbone and outputs a set of feature maps with a unified channel
dimension and spatially aligned scales, ready to be consumed by the BEVEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    """Feature Pyramid Network that aligns backbone feature map scales."""

    def __init__(
        self,
        in_channels: list[int] | None = None,
        out_channels: int = 256,
        num_outs: int = 4,
    ) -> None:
        super().__init__()
        if in_channels is None:
            in_channels = [512, 1024, 2048]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(num_outs)]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Build the feature pyramid and return aligned feature maps."""
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features, strict=True)]
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled
        outs = [self.output_convs[i](laterals[i]) for i in range(len(laterals))]
        for i in range(len(laterals), self.num_outs):
            outs.append(self.output_convs[i](F.max_pool2d(outs[-1], kernel_size=2, stride=2)))
        return outs
