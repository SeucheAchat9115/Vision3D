"""Integration tests: ResNet backbone ↔ FPN neck and FPN neck ↔ BEV encoder.

Covered scenarios:
  - ResNet18/34 channel compatibility with FPNNeck
  - ResNet50 channel compatibility with FPNNeck
  - Single-scale backbone output wired to FPN
  - Extra FPN levels via max-pooling
  - Gradient flow from FPN output back through the backbone
  - Batch dimension preservation through backbone + neck
  - BEVEncoder output shape when fed FPN features
  - Gradient flow from BEV map all the way back to image pixels
"""

from __future__ import annotations

import pytest
import torch

from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.necks.fpn import FPNNeck

# ---------------------------------------------------------------------------
# 1. Backbone + Neck combinations
# ---------------------------------------------------------------------------


class TestBackboneNeckIntegration:
    """Verify that different backbone depths are wired correctly to FPNNeck."""

    @pytest.mark.parametrize("depth", [18, 34])
    def test_resnet_shallow_fpn_output_shape(self, depth: int) -> None:
        """ResNet18/34 channels [64,128,256] must be compatible with FPNNeck."""
        backbone = ResNetBackbone(depth=depth, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 64, "All FPN outputs must have out_channels=64"

    def test_resnet50_fpn_output_shape(self) -> None:
        """ResNet50 channels [256,512,1024] must be compatible with FPNNeck."""
        backbone = ResNetBackbone(depth=50, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[256, 512, 1024], out_channels=128, num_outs=3)
        x = torch.randn(1, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 128

    def test_single_scale_backbone_fpn(self) -> None:
        """Single-scale backbone output (out_indices=[3]) wired into FPN."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[3])
        neck = FPNNeck(in_channels=[256], out_channels=32, num_outs=1)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        assert len(features) == 1
        outs = neck(features)
        assert len(outs) == 1
        assert outs[0].shape[1] == 32

    def test_extra_fpn_levels_via_maxpool(self) -> None:
        """FPN should add extra max-pooled levels when num_outs > len(in_channels)."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=5)
        x = torch.randn(2, 3, 64, 64)
        features = backbone(x)
        outs = neck(features)
        assert len(outs) == 5
        for out in outs:
            assert out.shape[1] == 64

    def test_backbone_fpn_gradient_flows(self) -> None:
        """Gradients must propagate from FPN output back through the backbone."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        features = backbone(x)
        outs = neck(features)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        # conv1 is frozen (requires_grad=False) but x is not frozen
        assert x.grad is not None

    def test_backbone_fpn_batch_size_preserved(self) -> None:
        """Batch dimension must be preserved through backbone and neck."""
        for B in [1, 3]:
            backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
            neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
            x = torch.randn(B, 3, 64, 64)
            outs = neck(backbone(x))
            for out in outs:
                assert out.shape[0] == B

    def test_fpn_output_spatial_resolution_decreases(self) -> None:
        """Each successive FPN level must have equal or smaller spatial resolution."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
        x = torch.randn(1, 3, 128, 128)
        outs = neck(backbone(x))
        for i in range(len(outs) - 1):
            assert outs[i].shape[-1] >= outs[i + 1].shape[-1], (
                f"FPN level {i} spatial size {outs[i].shape[-1]} "
                f"must be >= level {i + 1} size {outs[i + 1].shape[-1]}"
            )

    def test_fpn_all_levels_finite(self) -> None:
        """All FPN output tensors must be finite (no NaN or Inf)."""
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=32, num_outs=3)
        x = torch.randn(2, 3, 64, 64)
        outs = neck(backbone(x))
        for i, out in enumerate(outs):
            assert torch.isfinite(out).all(), f"FPN level {i} contains non-finite values"


# ---------------------------------------------------------------------------
# 2. Neck → Encoder integration
# ---------------------------------------------------------------------------


class TestNeckEncoderIntegration:
    """Verify that FPNNeck outputs are compatible with BEVEncoder inputs."""

    def test_neck_encoder_output_shape(self) -> None:
        """BEVEncoder must accept neck feature maps and return (B, C, H, W)."""
        B, C, H, W = 2, 32, 4, 4
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=C, num_outs=3)
        encoder = BEVEncoder(
            bev_h=H, bev_w=W, embed_dims=C, num_layers=1, num_heads=4, num_points=2, dropout=0.0
        )
        images = torch.randn(B, 3, 64, 64)
        features = neck(backbone(images))
        K = torch.zeros(B, 1, 3, 3)
        K[:, :, 0, 0] = 400.0
        K[:, :, 1, 1] = 400.0
        K[:, :, 0, 2] = 8.0
        K[:, :, 1, 2] = 8.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4).clone()
        E[:, :, 2, 3] = 5.0
        bev = encoder(features, K, E)
        assert bev.shape == (B, C, H, W)

    def test_neck_encoder_gradient_flows(self) -> None:
        """Gradients must flow from BEV map all the way back to image pixels."""
        B, C = 1, 32
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=C, num_outs=3)
        encoder = BEVEncoder(
            bev_h=4, bev_w=4, embed_dims=C, num_layers=1, num_heads=4, num_points=2, dropout=0.0
        )
        images = torch.randn(B, 3, 64, 64, requires_grad=True)
        features = neck(backbone(images))
        K = torch.zeros(B, 1, 3, 3)
        K[:, :, 0, 0] = 400.0
        K[:, :, 1, 1] = 400.0
        K[:, :, 0, 2] = 8.0
        K[:, :, 1, 2] = 8.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4).clone()
        bev = encoder(features, K, E)
        bev.sum().backward()
        assert images.grad is not None

    def test_neck_encoder_bev_is_finite(self) -> None:
        """BEVEncoder output must contain only finite values."""
        B, C, H, W = 2, 32, 4, 4
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[1, 2, 3])
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=C, num_outs=3)
        encoder = BEVEncoder(
            bev_h=H, bev_w=W, embed_dims=C, num_layers=1, num_heads=4, num_points=2, dropout=0.0
        )
        images = torch.randn(B, 3, 64, 64)
        features = neck(backbone(images))
        K = torch.zeros(B, 1, 3, 3)
        K[:, :, 0, 0] = 400.0
        K[:, :, 1, 1] = 400.0
        K[:, :, 0, 2] = 8.0
        K[:, :, 1, 2] = 8.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4).clone()
        bev = encoder(features, K, E)
        assert torch.isfinite(bev).all()
