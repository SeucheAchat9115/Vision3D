"""
ResNet backbone for Vision3D multi-view feature extraction.

Provides `ResNetBackbone`, a standard 2-D convolutional feature extractor based
on the ResNet architecture. It accepts batched camera images and returns
multi-scale feature maps that are subsequently processed by the FPN neck.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

RESNET_CONFIGS = {
    18: tv_models.resnet18,
    34: tv_models.resnet34,
    50: tv_models.resnet50,
    101: tv_models.resnet101,
    152: tv_models.resnet152,
}


class ResNetBackbone(nn.Module):
    """Standard ResNet 2-D feature extractor."""

    def __init__(
        self,
        depth: int = 50,
        out_indices: list[int] | None = None,
        pretrained: bool = True,
        frozen_stages: int = 1,
    ) -> None:
        super().__init__()
        if depth not in RESNET_CONFIGS:
            raise ValueError(f"depth must be one of {list(RESNET_CONFIGS)}, got {depth}")
        self.depth = depth
        self.out_indices = out_indices if out_indices is not None else [1, 2, 3]
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        weights = tv_models.get_model_weights(RESNET_CONFIGS[depth]).DEFAULT if pretrained else None
        resnet = RESNET_CONFIGS[depth](weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self._freeze_stages()

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale feature maps from a batch of images."""
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        stage_outputs = [x]
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = stage(x)
            stage_outputs.append(x)
        return [stage_outputs[idx] for idx in self.out_indices]

    def _freeze_stages(self) -> None:
        """Freeze parameters for all stages up to self.frozen_stages."""
        if self.frozen_stages < 0:
            return
        for m in [self.conv1, self.bn1]:
            for param in m.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            if i <= 4:
                stage = [self.layer1, self.layer2, self.layer3, self.layer4][i - 1]
                for param in stage.parameters():
                    param.requires_grad = False
