"""Integration tests: BEVFormerModel → HungarianMatcher → DetectionLoss pipeline.

Covered scenarios:
  - Loss is finite and differentiable
  - Backward pass populates gradients
  - Edge cases: empty targets, single GT, more GT than queries
  - Mixed batch (some frames with targets, some without)
  - Various loss weight configurations
  - Optimizer weight update after backward
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest
import torch

from tests.integration.helpers import make_batch, make_frame, make_small_model
from vision3d.config.schema import BatchData, BoundingBox3DPrediction
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.models.bevformer import BEVFormerModel

F = TypeVar("F", bound=Callable[..., object])


def typed_parametrize(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper around pytest.mark.parametrize for mypy compatibility."""
    return cast(Callable[[F], F], pytest.mark.parametrize(*args, **kwargs))


class TestModelMatcherLossIntegration:
    """Integration tests for the BEVFormerModel → HungarianMatcher → DetectionLoss pipeline."""

    def _run_training_pass(
        self,
        batch: BatchData,
        model: BEVFormerModel,
        matcher: HungarianMatcher,
        loss_fn: DetectionLoss,
    ) -> torch.Tensor:
        """Helper: forward, match, compute loss; return total loss scalar."""
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        return cast(torch.Tensor, total)

    def test_training_pass_loss_is_finite(self) -> None:
        """End-to-end training pass must produce a finite scalar loss."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_loss_requires_grad(self) -> None:
        """Loss must be differentiable with respect to model parameters."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert total.requires_grad

    def test_training_pass_backward(self) -> None:
        """Backward pass must not raise and must populate gradients."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        total.backward()  # type: ignore[no-untyped-call]
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        assert n_params_with_grad > 0

    def test_training_pass_with_empty_targets(self) -> None:
        """Training step must handle frames with zero ground-truth boxes."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = make_frame(num_boxes=0, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_single_frame_single_gt(self) -> None:
        """Training step must work with exactly one GT box."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes, num_queries=10)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = make_frame(num_boxes=1, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_more_gt_than_queries(self) -> None:
        """When GT boxes exceed queries the matcher clips matches; loss must be finite."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes, num_queries=3)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = make_frame(num_boxes=8, num_classes=num_classes)
        batch = BatchData(batch_size=1, frames=[frame])
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_training_pass_without_targets_skipped(self) -> None:
        """Frames with no targets should be excluded; remaining frames still produce a loss."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame_with = make_frame(num_boxes=2, num_classes=num_classes, frame_id="has_targets")
        frame_without = make_frame(
            num_boxes=0, num_classes=num_classes, frame_id="no_targets", with_targets=False
        )
        batch = BatchData(batch_size=2, frames=[frame_with, frame_without])
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        assert len(pred_list) == 1  # Only the frame_with was included
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        assert torch.isfinite(total)

    @typed_parametrize(
        ("cls_w", "bbox_w", "giou_w"),
        [(2.0, 0.25, 0.1), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)],
    )
    def test_training_pass_different_loss_weights(
        self, cls_w: float, bbox_w: float, giou_w: float
    ) -> None:
        """Training pass must produce finite loss for various loss weight combinations."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(
            num_classes=num_classes,
            cls_weight=cls_w,
            bbox_weight=bbox_w,
            giou_weight=giou_w,
        )
        batch = make_batch(num_classes=num_classes)
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        assert torch.isfinite(total)

    def test_optimizer_step_updates_parameters(self) -> None:
        """An SGD optimizer step must change at least one parameter value."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        batch = make_batch(num_classes=num_classes)

        # Capture a snapshot of parameters before the step
        params_before = [p.data.clone() for p in model.parameters() if p.requires_grad]

        optimizer.zero_grad()
        total = self._run_training_pass(batch, model, matcher, loss_fn)
        total.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        params_after = [p.data for p in model.parameters() if p.requires_grad]
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, params_after, strict=True)
        )
        assert changed, "No parameter was updated after optimizer.step()"

    def test_loss_breakdown_keys_present(self) -> None:
        """DetectionLoss must return a breakdown dict containing expected keys."""
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        matches = matcher.match_batch(pred_list, targets)
        _, breakdown = loss_fn(pred_list, targets, matches)
        for key in ("loss_cls", "loss_bbox", "loss_giou"):
            assert key in breakdown
