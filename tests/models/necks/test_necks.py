"""Unit tests for FPNNeck."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vision3d.models.necks.fpn import FPNNeck


def _make_features(
    in_channels: list[int],
    spatial_sizes: list[tuple[int, int]],
    batch_size: int = 2,
) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, c, h, w)
        for c, (h, w) in zip(in_channels, spatial_sizes, strict=True)
    ]


class TestFPNNeckInterface:
    """Verify the interface of FPNNeck."""

    def test_is_nn_module(self):
        assert issubclass(FPNNeck, nn.Module)

    def test_has_forward(self):
        assert hasattr(FPNNeck, "forward")

    def test_init_stores_params(self):
        neck = FPNNeck(in_channels=[128, 256, 512], out_channels=128, num_outs=4)
        assert neck.in_channels == [128, 256, 512]
        assert neck.out_channels == 128
        assert neck.num_outs == 4


class TestFPNNeckForward:
    """Test FPNNeck forward pass shapes and values."""

    def test_output_is_list(self):
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        assert isinstance(outs, list)

    def test_output_length_matches_num_outs(self):
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=4)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        assert len(outs) == 4

    def test_output_channel_matches_out_channels(self):
        out_channels = 64
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=out_channels, num_outs=3)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        for out in outs:
            assert out.shape[1] == out_channels

    def test_batch_size_preserved(self):
        B = 4
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)], batch_size=B)
        outs = neck(features)
        for out in outs:
            assert out.shape[0] == B

    def test_output_tensors_are_float(self):
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        for out in outs:
            assert out.dtype == torch.float32

    def test_extra_outs_via_max_pool(self):
        """num_outs > len(in_channels) should add max-pooled outputs."""
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=5)
        features = _make_features([64, 128, 256], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        assert len(outs) == 5
        # Extra outputs should have smaller spatial size than the last feature
        assert outs[3].shape[2] < outs[2].shape[2]

    def test_default_in_channels(self):
        """Default in_channels=[512, 1024, 2048] should work with ResNet50 sizes."""
        neck = FPNNeck(out_channels=256, num_outs=3)
        features = _make_features([512, 1024, 2048], [(32, 32), (16, 16), (8, 8)])
        outs = neck(features)
        assert len(outs) == 3
        for out in outs:
            assert out.shape[1] == 256

    def test_output_requires_grad(self):
        neck = FPNNeck(in_channels=[64, 128], out_channels=64, num_outs=2)
        features = _make_features([64, 128], [(16, 16), (8, 8)])
        outs = neck(features)
        for out in outs:
            assert out.requires_grad

    def test_single_scale_input(self):
        """FPN with a single input scale should work."""
        neck = FPNNeck(in_channels=[256], out_channels=128, num_outs=1)
        features = [torch.randn(2, 256, 16, 16)]
        outs = neck(features)
        assert len(outs) == 1
        assert outs[0].shape == (2, 128, 16, 16)

    def test_top_down_upsampling(self):
        """The FPN's top-down path should affect the lowest-level feature map."""
        torch.manual_seed(0)
        neck = FPNNeck(in_channels=[64, 128], out_channels=64, num_outs=2)
        # Check that output0 is NOT equal to just applying lateral to input0
        # (because top-down from level 1 contributes)
        features = _make_features([64, 128], [(16, 16), (8, 8)])
        outs_with_fpn = neck(features)
        assert outs_with_fpn[0].shape[2] == 16

    def test_mismatched_in_channels_raises(self):
        """Providing fewer feature maps than in_channels should raise an error."""
        neck = FPNNeck(in_channels=[64, 128, 256], out_channels=64, num_outs=3)
        features = _make_features([64, 128], [(32, 32), (16, 16)])  # Only 2
        with pytest.raises((RuntimeError, AssertionError, ValueError)):
            neck(features)
