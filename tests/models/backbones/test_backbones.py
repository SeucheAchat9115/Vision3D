"""Unit tests for ResNetBackbone."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vision3d.models.backbones.resnet import ResNetBackbone


class TestResNetBackboneInterface:
    """Verify the interface of ResNetBackbone."""

    def test_is_nn_module(self):
        assert issubclass(ResNetBackbone, nn.Module)

    def test_has_forward(self):
        assert hasattr(ResNetBackbone, "forward")

    def test_init_stores_params(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, frozen_stages=0)
        assert backbone.depth == 18
        assert backbone.out_indices == [1, 2, 3]

    def test_custom_out_indices_stored(self):
        backbone = ResNetBackbone(depth=18, out_indices=[2, 3], pretrained=False)
        assert backbone.out_indices == [2, 3]

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be one of"):
            ResNetBackbone(depth=999, pretrained=False)


class TestResNetBackboneForward:
    """Test ResNetBackbone forward pass (no pre-training download needed)."""

    def _make_backbone(
        self,
        depth: int = 18,
        out_indices: list[int] | None = None,
        frozen_stages: int = 0,
    ) -> ResNetBackbone:
        return ResNetBackbone(
            depth=depth,
            out_indices=out_indices,
            pretrained=False,
            frozen_stages=frozen_stages,
        )

    def test_forward_returns_list(self):
        backbone = self._make_backbone()
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        assert isinstance(outs, list)

    def test_output_length_matches_out_indices(self):
        backbone = self._make_backbone(out_indices=[1, 2, 3])
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        assert len(outs) == 3

    def test_single_out_index(self):
        backbone = self._make_backbone(out_indices=[3])
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        assert len(outs) == 1

    def test_batch_size_preserved(self):
        backbone = self._make_backbone()
        B = 2
        x = torch.randn(B, 3, 64, 64)
        outs = backbone(x)
        for out in outs:
            assert out.shape[0] == B

    def test_output_channels_resnet18(self):
        """ResNet18 output channel counts for stages [1,2,3,4] are [64,128,256,512]."""
        expected = {1: 64, 2: 128, 3: 256}
        backbone = self._make_backbone(depth=18, out_indices=[1, 2, 3])
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        for out, idx in zip(outs, [1, 2, 3], strict=True):
            assert out.shape[1] == expected[idx]

    def test_output_spatial_sizes_decrease(self):
        """Later stages should have smaller spatial dimensions."""
        backbone = self._make_backbone(out_indices=[1, 2, 3])
        x = torch.randn(1, 3, 128, 128)
        outs = backbone(x)
        for i in range(len(outs) - 1):
            assert outs[i].shape[2] >= outs[i + 1].shape[2]

    def test_output_dtype(self):
        backbone = self._make_backbone()
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        for out in outs:
            assert out.dtype == torch.float32

    def test_output_requires_grad(self):
        backbone = self._make_backbone()
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        outs = backbone(x)
        assert any(out.requires_grad for out in outs)


class TestResNetBackboneFreezeStages:
    """Test that frozen stages have no-gradient parameters."""

    def test_frozen_stage_1_has_no_grad(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, frozen_stages=1)
        for param in backbone.layer1.parameters():
            assert not param.requires_grad

    def test_unfrozen_stage_has_grad(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, frozen_stages=0)
        for param in backbone.layer1.parameters():
            assert param.requires_grad

    def test_frozen_stages_beyond_1_also_frozen(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, frozen_stages=2)
        for param in backbone.layer1.parameters():
            assert not param.requires_grad
        for param in backbone.layer2.parameters():
            assert not param.requires_grad
        # Stage 3 should still have grad
        assert any(param.requires_grad for param in backbone.layer3.parameters())

    def test_frozen_conv1_and_bn1(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, frozen_stages=0)
        for param in backbone.conv1.parameters():
            assert not param.requires_grad
        for param in backbone.bn1.parameters():
            assert not param.requires_grad


class TestResNetBackboneDepths:
    """Test ResNetBackbone with different depths."""

    def test_resnet18_forward_pass(self):
        backbone = ResNetBackbone(depth=18, pretrained=False, out_indices=[3])
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        assert len(outs) == 1
        assert outs[0].shape[0] == 1

    def test_resnet34_forward_pass(self):
        backbone = ResNetBackbone(depth=34, pretrained=False, out_indices=[3])
        x = torch.randn(1, 3, 64, 64)
        outs = backbone(x)
        assert len(outs) == 1
        assert outs[0].shape[0] == 1
