"""Integration tests: Full BEVFormerModel pipeline.

Covered scenarios:
  - Output types and shapes
  - Temporal prev_bev handling
  - Inference without ground-truth targets
  - Multi-camera support (1, 3, 6 cameras)
  - Confidence scores in [0, 1] range
  - Label predictions within class range
  - Gradient flow through full model
  - Parameterized tests for backbone depths, query counts, BEV grid sizes
  - Model determinism in eval mode
"""

from __future__ import annotations

import pytest
import torch

from tests.integration.helpers import make_batch, make_small_model
from vision3d.config.schema import BoundingBox3DPrediction


class TestBEVFormerModelIntegration:
    """End-to-end BEVFormerModel integration tests."""

    def test_forward_returns_correct_types(self) -> None:
        """Model must return (BoundingBox3DPrediction, Tensor)."""
        model = make_small_model()
        batch = make_batch()
        preds, new_bev = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)
        assert isinstance(new_bev, torch.Tensor)

    def test_forward_prediction_shapes(self) -> None:
        """Prediction tensors must have correct shapes for batch_size=2, num_queries=10."""
        B, Q, num_classes = 2, 10, 5
        model = make_small_model(num_classes=num_classes, num_queries=Q)
        batch = make_batch(batch_size=B)
        preds, _ = model(batch)
        assert preds.boxes.shape == (B, Q, 10)
        assert preds.scores.shape == (B, Q)
        assert preds.labels.shape == (B, Q)

    def test_forward_new_bev_shape(self) -> None:
        """new_bev must be shaped (H*W, B, C) for temporal attention."""
        B, C, H, W = 2, 32, 4, 4
        model = make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        batch = make_batch(batch_size=B)
        _, new_bev = model(batch)
        assert new_bev.shape == (H * W, B, C)

    def test_forward_single_frame(self) -> None:
        """Model must work with batch_size=1."""
        model = make_small_model()
        batch = make_batch(batch_size=1)
        preds, new_bev = model(batch)
        assert preds.boxes.shape[0] == 1

    def test_forward_with_prev_bev(self) -> None:
        """Temporal path (prev_bev) must not raise and must yield correct shapes."""
        B, C, H, W, Q = 2, 32, 4, 4, 10
        model = make_small_model(embed_dims=C, bev_h=H, bev_w=W, num_queries=Q)
        batch = make_batch(batch_size=B)
        _, new_bev = model(batch, prev_bev=None)
        preds2, _ = model(batch, prev_bev=new_bev)
        assert preds2.boxes.shape == (B, Q, 10)

    def test_forward_without_targets(self) -> None:
        """Inference without targets must not raise."""
        model = make_small_model()
        batch = make_batch(with_targets=False)
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_forward_multi_camera(self) -> None:
        """Model must handle multiple cameras per frame."""
        model = make_small_model()
        batch = make_batch(num_cameras=3)
        preds, _ = model(batch)
        assert preds.boxes.shape[0] == 2  # batch_size=2 by default

    def test_forward_six_cameras(self) -> None:
        """Full autonomous-driving scenario: 6 cameras per frame."""
        model = make_small_model()
        batch = make_batch(num_cameras=6)
        preds, _ = model(batch)
        assert preds.boxes.shape[0] == 2

    def test_forward_scores_in_unit_interval(self) -> None:
        """Confidence scores must lie in [0, 1] (sigmoid activation)."""
        model = make_small_model()
        batch = make_batch()
        preds, _ = model(batch)
        assert preds.scores.min().item() >= 0.0
        assert preds.scores.max().item() <= 1.0

    def test_forward_labels_within_class_range(self) -> None:
        """Predicted class labels must be in [0, num_classes)."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        preds, _ = model(batch)
        assert preds.labels.min().item() >= 0
        assert preds.labels.max().item() < num_classes

    def test_forward_gradients_flow_through_full_model(self) -> None:
        """A loss on the predictions must produce non-None gradients for the head."""
        model = make_small_model()
        batch = make_batch()
        preds, _ = model(batch)
        loss = preds.boxes.sum()
        loss.backward()
        head_grad = next(
            (p.grad for p in model.head.parameters() if p.grad is not None),
            None,
        )
        assert head_grad is not None

    def test_forward_bev_output_is_finite(self) -> None:
        """The new_bev tensor returned by the model must contain only finite values."""
        model = make_small_model()
        batch = make_batch()
        _, new_bev = model(batch)
        assert torch.isfinite(new_bev).all()

    def test_forward_deterministic_in_eval_mode(self) -> None:
        """Two forward passes with the same input must produce identical results in eval mode."""
        model = make_small_model()
        model.eval()
        batch = make_batch(batch_size=1)
        with torch.no_grad():
            preds1, _ = model(batch)
            preds2, _ = model(batch)
        assert torch.allclose(preds1.boxes, preds2.boxes)
        assert torch.allclose(preds1.scores, preds2.scores)

    @pytest.mark.parametrize("depth", [18, 34])
    def test_different_backbone_depths(self, depth: int) -> None:
        """Models with different backbone depths must produce compatible outputs."""
        model = make_small_model(backbone_depth=depth)
        batch = make_batch()
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    @pytest.mark.parametrize("num_queries", [5, 20, 50])
    def test_different_query_counts(self, num_queries: int) -> None:
        """Models with different query counts must produce shapes matching num_queries."""
        model = make_small_model(num_queries=num_queries)
        batch = make_batch()
        preds, _ = model(batch)
        assert preds.boxes.shape[1] == num_queries

    @pytest.mark.parametrize("bev_size", [(4, 4), (6, 8)])
    def test_different_bev_grid_sizes(self, bev_size: tuple[int, int]) -> None:
        """Models with different BEV grid sizes must produce correctly-sized new_bev."""
        H, W = bev_size
        C = 32
        model = make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        batch = make_batch(batch_size=1)
        _, new_bev = model(batch)
        assert new_bev.shape == (H * W, 1, C)
