"""
ResNet backbone for Vision3D multi-view feature extraction.

Provides `ResNetBackbone`, a standard 2-D convolutional feature extractor based
on the ResNet architecture. It accepts batched camera images and returns
multi-scale feature maps that are subsequently processed by the FPN neck.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ResNetBackbone(nn.Module):
    """Standard ResNet 2-D feature extractor.

    Accepts a batch of images (potentially from multiple camera views flattened
    together) and returns a list of feature maps at the stages specified by
    `out_indices`.

    The implementation wraps `torchvision.models.resnet` and extracts
    intermediate feature maps using forward hooks or by overriding the forward
    pass to return intermediate activations.

    Note: Only the feature extraction portion (stem + stages) is used; the
    classification head is discarded.

    Args:
        depth: ResNet variant depth. Supported values: 18, 34, 50, 101, 152.
        out_indices: List of stage indices (0-based) whose output feature maps
            are returned. Stage 0 is the stem output; stages 1-4 correspond to
            layer1–layer4.
        pretrained: If True, initialise weights from ImageNet-pretrained
            torchvision checkpoint.
        frozen_stages: Stages whose parameters are frozen (not updated during
            training). Set to -1 to train all stages.
    """

    def __init__(
        self,
        depth: int = 50,
        out_indices: List[int] = None,
        pretrained: bool = True,
        frozen_stages: int = 1,
    ) -> None:
        super().__init__()
        # TODO: validate that depth is in {18, 34, 50, 101, 152}
        # TODO: instantiate the torchvision ResNet model for the given depth
        # TODO: remove the average-pool and classification head layers
        # TODO: register forward hooks or restructure layers to expose selected stages
        # TODO: load pretrained weights if pretrained=True
        # TODO: freeze parameters for stages <= frozen_stages
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature maps from a batch of images.

        Args:
            images: Float tensor of shape (B * num_cameras, C, H, W), where
                B is the batch size and num_cameras is the number of camera
                views. Images should be normalised to ImageNet statistics.

        Returns:
            List of feature tensors at the stages in `self.out_indices`.
            For ResNet-50 with out_indices=[1, 2, 3], this is:
              - Stage 1: (B * num_cameras, 256, H/4, W/4)
              - Stage 2: (B * num_cameras, 512, H/8, W/8)
              - Stage 3: (B * num_cameras, 1024, H/16, W/16)
        """
        # TODO: pass images through the stem (conv1, bn1, relu, maxpool)
        # TODO: pass through each stage (layer1 … layer4) and collect outputs
        #       at the indices listed in self.out_indices
        # TODO: return the collected feature tensors as a list
        raise NotImplementedError

    def _freeze_stages(self) -> None:
        """Freeze parameters for all stages up to self.frozen_stages."""
        # TODO: iterate over the appropriate layer modules
        # TODO: set requires_grad=False for all parameters in frozen stages
        raise NotImplementedError
