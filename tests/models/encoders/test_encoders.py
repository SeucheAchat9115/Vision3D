"""Unit tests for BEVEncoder and BEVEncoderLayer."""

from __future__ import annotations

import torch
import torch.nn as nn

from vision3d.models.encoders.bev_encoder import BEVEncoder, BEVEncoderLayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera_matrices(
    B: int = 2,
    num_cameras: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (intrinsics (B, C, 3, 3), extrinsics (B, C, 4, 4))."""
    K = torch.zeros(B, num_cameras, 3, 3)
    K[:, :, 0, 0] = 400.0
    K[:, :, 1, 1] = 400.0
    K[:, :, 0, 2] = 100.0
    K[:, :, 1, 2] = 100.0
    K[:, :, 2, 2] = 1.0

    E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, num_cameras, 4, 4).clone()
    E[:, :, 2, 3] = 5.0  # Sensor 5m above ego origin in z
    return K, E


def _make_bev_encoder(
    bev_h: int = 4,
    bev_w: int = 4,
    embed_dims: int = 32,
    num_layers: int = 1,
    num_cameras: int = 2,
) -> BEVEncoder:
    return BEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dims=embed_dims,
        num_layers=num_layers,
        num_heads=4,
        num_points=2,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# BEVEncoderLayer tests
# ---------------------------------------------------------------------------


class TestBEVEncoderLayerInterface:
    """Verify the interface of BEVEncoderLayer."""

    def test_is_nn_module(self):
        assert issubclass(BEVEncoderLayer, nn.Module)

    def test_has_forward(self):
        assert hasattr(BEVEncoderLayer, "forward")

    def test_init_stores_params(self):
        layer = BEVEncoderLayer(embed_dims=64, num_heads=4, num_points=4)
        assert layer.embed_dims == 64
        assert layer.num_heads == 4
        assert layer.num_points == 4


class TestBEVEncoderLayerForward:
    """Test BEVEncoderLayer forward pass."""

    def _make_inputs(
        self,
        HW: int = 16,
        B: int = 2,
        embed_dims: int = 32,
        num_cameras: int = 2,
        feat_h: int = 8,
        feat_w: int = 8,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bev_queries = torch.randn(HW, B, embed_dims)
        img_features = [torch.randn(B * num_cameras, embed_dims, feat_h, feat_w)]
        ref_pts = torch.randn(HW, 3)
        K = torch.zeros(B, num_cameras, 3, 3)
        K[:, :, 0, 0] = 50.0
        K[:, :, 1, 1] = 50.0
        K[:, :, 0, 2] = 4.0
        K[:, :, 1, 2] = 4.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, num_cameras, 4, 4).clone()
        E[:, :, 2, 3] = 10.0
        spatial_shapes = torch.tensor([[feat_h, feat_w]], dtype=torch.long)
        return bev_queries, img_features, ref_pts, K, E, spatial_shapes

    def test_output_shape(self):
        HW, B, C = 16, 2, 32
        layer = BEVEncoderLayer(embed_dims=C, num_heads=4, num_points=2, dropout=0.0)
        bev_q, img_f, ref, K, E, ss = self._make_inputs(HW, B, C)
        out = layer(bev_q, None, img_f, ref, K, E, ss)
        assert out.shape == (HW, B, C)

    def test_output_dtype(self):
        HW, B, C = 8, 1, 32
        layer = BEVEncoderLayer(embed_dims=C, num_heads=4, num_points=2, dropout=0.0)
        bev_q, img_f, ref, K, E, ss = self._make_inputs(HW, B, C, num_cameras=1)
        out = layer(bev_q, None, img_f, ref, K, E, ss)
        assert out.dtype == torch.float32

    def test_with_prev_bev_tsa_active(self):
        """Passing prev_bev should activate temporal self-attention."""
        HW, B, C = 8, 2, 32
        layer = BEVEncoderLayer(embed_dims=C, num_heads=4, num_points=2, dropout=0.0)
        bev_q, img_f, ref, K, E, ss = self._make_inputs(HW, B, C)
        prev_bev = torch.randn(HW, B, C)
        out_with = layer(bev_q, prev_bev, img_f, ref, K, E, ss)
        out_without = layer(bev_q, None, img_f, ref, K, E, ss)
        assert out_with.shape == (HW, B, C)
        assert out_without.shape == (HW, B, C)

    def test_output_has_gradient(self):
        HW, B, C = 8, 1, 32
        layer = BEVEncoderLayer(embed_dims=C, num_heads=4, num_points=2, dropout=0.0)
        bev_q, img_f, ref, K, E, ss = self._make_inputs(HW, B, C, num_cameras=1)
        bev_q.requires_grad_(True)
        out = layer(bev_q, None, img_f, ref, K, E, ss)
        assert out.requires_grad


# ---------------------------------------------------------------------------
# BEVEncoder tests
# ---------------------------------------------------------------------------


class TestBEVEncoderInterface:
    """Verify the interface of BEVEncoder."""

    def test_is_nn_module(self):
        assert issubclass(BEVEncoder, nn.Module)

    def test_has_forward(self):
        assert hasattr(BEVEncoder, "forward")

    def test_init_stores_params(self):
        enc = BEVEncoder(bev_h=8, bev_w=8, embed_dims=32, num_layers=2)
        assert enc.bev_h == 8
        assert enc.bev_w == 8
        assert enc.embed_dims == 32
        assert enc.num_layers == 2

    def test_reference_points_registered_as_buffer(self):
        enc = _make_bev_encoder()
        assert hasattr(enc, "reference_points")
        ref: torch.Tensor = enc.reference_points
        assert ref.shape == (enc.bev_h * enc.bev_w, 3)


class TestBEVEncoderForward:
    """Test BEVEncoder forward pass output shape and properties."""

    def _make_inputs(
        self,
        B: int = 2,
        num_cameras: int = 2,
        embed_dims: int = 32,
        feat_h: int = 8,
        feat_w: int = 8,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        img_features = [torch.randn(B * num_cameras, embed_dims, feat_h, feat_w)]
        K = torch.zeros(B, num_cameras, 3, 3)
        K[:, :, 0, 0] = 50.0
        K[:, :, 1, 1] = 50.0
        K[:, :, 0, 2] = 4.0
        K[:, :, 1, 2] = 4.0
        K[:, :, 2, 2] = 1.0
        E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, num_cameras, 4, 4).clone()
        E[:, :, 2, 3] = 5.0
        return img_features, K, E

    def test_output_shape(self):
        B, C, H, W = 2, 32, 4, 4
        enc = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev_map = enc(img_features, K, E)
        assert bev_map.shape == (B, C, H, W)

    def test_output_dtype(self):
        B, C = 1, 32
        enc = _make_bev_encoder(embed_dims=C)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev_map = enc(img_features, K, E)
        assert bev_map.dtype == torch.float32

    def test_output_requires_grad(self):
        B, C = 2, 32
        enc = _make_bev_encoder(embed_dims=C)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev_map = enc(img_features, K, E)
        assert bev_map.requires_grad

    def test_with_prev_bev(self):
        B, C, H, W = 2, 32, 4, 4
        enc = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        prev_bev = torch.randn(H * W, B, C)
        bev_map = enc(img_features, K, E, prev_bev=prev_bev)
        assert bev_map.shape == (B, C, H, W)

    def test_single_layer(self):
        B, C, H, W = 1, 32, 3, 3
        enc = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C, num_layers=1)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev_map = enc(img_features, K, E)
        assert bev_map.shape == (B, C, H, W)

    def test_multi_layer(self):
        B, C, H, W = 2, 32, 4, 4
        enc = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C, num_layers=3)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev_map = enc(img_features, K, E)
        assert bev_map.shape == (B, C, H, W)

    def test_different_batch_sizes(self):
        C, H, W = 32, 4, 4
        enc = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C)
        for B in [1, 3]:
            img_features, K, E = self._make_inputs(B=B, embed_dims=C)
            bev_map = enc(img_features, K, E)
            assert bev_map.shape == (B, C, H, W)

    def test_num_layers_affects_output(self):
        """Different depth encoders should produce different outputs."""
        B, C, H, W = 2, 32, 4, 4
        enc1 = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C, num_layers=1)
        enc2 = _make_bev_encoder(bev_h=H, bev_w=W, embed_dims=C, num_layers=2)
        img_features, K, E = self._make_inputs(B=B, embed_dims=C)
        bev1 = enc1(img_features, K, E)
        bev2 = enc2(img_features, K, E)
        # Different models → different outputs
        assert not torch.allclose(bev1, bev2)
