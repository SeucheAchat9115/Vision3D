"""Unit tests for DetectionHead."""

from __future__ import annotations

import torch
import torch.nn as nn

from vision3d.config.schema import BoundingBox3DPrediction
from vision3d.models.heads.detection_head import DetectionHead


class TestDetectionHeadInterface:
    """Verify the interface of DetectionHead."""

    def test_is_nn_module(self):
        assert issubclass(DetectionHead, nn.Module)

    def test_has_forward(self):
        assert hasattr(DetectionHead, "forward")

    def test_init_stores_params(self):
        head = DetectionHead(num_classes=5, in_channels=128, num_queries=100)
        assert head.num_classes == 5
        assert head.in_channels == 128
        assert head.num_queries == 100


class TestDetectionHeadForward:
    """Test DetectionHead forward pass."""

    def _make_head(
        self,
        num_classes: int = 5,
        in_channels: int = 32,
        num_queries: int = 10,
    ) -> DetectionHead:
        return DetectionHead(
            num_classes=num_classes,
            in_channels=in_channels,
            num_queries=num_queries,
            num_decoder_layers=1,
            num_heads=4,
            ffn_dim=64,
        )

    def test_forward_returns_prediction(self):
        head = self._make_head()
        bev = torch.randn(2, 32, 4, 4)
        result = head(bev)
        assert isinstance(result, BoundingBox3DPrediction)

    def test_boxes_shape(self):
        B, Q, C = 2, 10, 32
        head = self._make_head(in_channels=C, num_queries=Q)
        bev = torch.randn(B, C, 4, 4)
        result = head(bev)
        assert result.boxes.shape == (B, Q, 10)

    def test_scores_shape(self):
        B, Q, C = 3, 8, 32
        head = self._make_head(in_channels=C, num_queries=Q)
        bev = torch.randn(B, C, 4, 4)
        result = head(bev)
        assert result.scores.shape == (B, Q)

    def test_labels_shape(self):
        B, Q, C = 2, 6, 32
        head = self._make_head(in_channels=C, num_queries=Q)
        bev = torch.randn(B, C, 4, 4)
        result = head(bev)
        assert result.labels.shape == (B, Q)

    def test_scores_are_non_negative(self):
        """Scores are produced via sigmoid, so they should be in [0, 1]."""
        head = self._make_head()
        bev = torch.randn(2, 32, 4, 4)
        result = head(bev)
        assert result.scores.min().item() >= 0.0
        assert result.scores.max().item() <= 1.0

    def test_labels_within_class_range(self):
        num_classes = 5
        head = self._make_head(num_classes=num_classes)
        bev = torch.randn(2, 32, 4, 4)
        result = head(bev)
        assert result.labels.min().item() >= 0
        assert result.labels.max().item() < num_classes

    def test_output_dtype_float32(self):
        head = self._make_head()
        bev = torch.randn(2, 32, 4, 4)
        result = head(bev)
        assert result.boxes.dtype == torch.float32
        assert result.scores.dtype == torch.float32

    def test_boxes_have_gradient(self):
        head = self._make_head()
        bev = torch.randn(2, 32, 4, 4, requires_grad=True)
        result = head(bev)
        assert result.boxes.requires_grad

    def test_single_sample_batch(self):
        head = self._make_head(num_queries=5)
        bev = torch.randn(1, 32, 4, 4)
        result = head(bev)
        assert result.boxes.shape == (1, 5, 10)

    def test_larger_spatial_bev(self):
        """Head should work with larger H/W BEV maps."""
        head = self._make_head(num_queries=4)
        bev = torch.randn(1, 32, 8, 8)
        result = head(bev)
        assert result.boxes.shape == (1, 4, 10)

    def test_single_class(self):
        head = DetectionHead(
            num_classes=1,
            in_channels=32,
            num_queries=6,
            num_decoder_layers=1,
            num_heads=4,
            ffn_dim=64,
        )
        bev = torch.randn(2, 32, 4, 4)
        result = head(bev)
        assert result.labels.max().item() == 0  # Only class 0


class TestDetectionHeadMultipleClasses:
    """Test DetectionHead with varied class counts."""

    def test_ten_classes(self):
        head = DetectionHead(
            num_classes=10,
            in_channels=64,
            num_queries=20,
            num_decoder_layers=1,
            num_heads=4,
            ffn_dim=128,
        )
        bev = torch.randn(2, 64, 4, 4)
        result = head(bev)
        assert result.labels.min().item() >= 0
        assert result.labels.max().item() < 10
