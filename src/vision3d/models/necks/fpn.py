"""
Feature Pyramid Network (FPN) neck for Vision3D.

Provides `FPNNeck`, which takes the list of multi-scale feature maps produced
by the backbone and outputs a set of feature maps with a unified channel
dimension and spatially aligned scales, ready to be consumed by the BEVEncoder.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class FPNNeck(nn.Module):
    """Feature Pyramid Network that aligns backbone feature map scales.

    Takes the multi-scale output of `ResNetBackbone` and produces a list of
    feature maps at the same scales but with a uniform `out_channels` channel
    count. Lateral connections + top-down pathway with nearest-neighbour
    upsampling are used to fuse semantic and spatial information.

    Architecture overview:
      1. Apply a 1×1 lateral convolution to each backbone feature map to
         project it to `out_channels`.
      2. Build a top-down pathway by upsampling each level and adding it to the
         level below.
      3. Apply a 3×3 output convolution to each level to smooth aliasing
         artefacts from the addition.

    Args:
        in_channels: List of channel counts for each input feature map, ordered
            from the largest spatial resolution to the smallest.
            E.g. [512, 1024, 2048] for ResNet-50 stages 1-3.
        out_channels: Unified output channel count for all pyramid levels.
        num_outs: Number of output feature levels. Extra levels beyond those
            provided by the backbone are generated via max-pooling.
    """

    def __init__(
        self,
        in_channels: List[int] = None,
        out_channels: int = 256,
        num_outs: int = 4,
    ) -> None:
        super().__init__()
        # TODO: validate that len(in_channels) >= 1
        # TODO: build nn.ModuleList of 1×1 lateral convolutions (one per in_channels entry)
        # TODO: build nn.ModuleList of 3×3 output convolutions (one per output level)
        # TODO: if num_outs > len(in_channels), build extra max-pooling layers for
        #       the additional output levels
        raise NotImplementedError

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build the feature pyramid and return aligned feature maps.

        Args:
            features: List of backbone feature tensors ordered from the largest
                spatial resolution to the smallest. Each tensor has shape
                (B, C_i, H_i, W_i).

        Returns:
            List of feature tensors all with `out_channels` channels, ordered
            from largest to smallest spatial resolution.
        """
        # TODO: apply lateral convolutions to all input feature maps
        # TODO: run the top-down pathway (iterating from the smallest scale up),
        #       upsampling and adding to the next finer level
        # TODO: apply output 3×3 convolutions to each level
        # TODO: generate any extra levels via max-pooling if num_outs > len(features)
        # TODO: return the list of output feature maps
        raise NotImplementedError
