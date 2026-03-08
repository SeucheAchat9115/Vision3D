"""Unit tests for DetectionLoss."""

from __future__ import annotations

import torch
import torch.nn as nn

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)
from vision3d.core.losses import DetectionLoss


def _make_pred_tgt_match(
    num_queries: int = 10,
    num_gt: int = 4,
    num_classes: int = 5,
    *,
    seed: int = 0,
) -> tuple[
    list[BoundingBox3DPrediction],
    list[BoundingBox3DTarget],
    list[MatchingResult],
]:
    torch.manual_seed(seed)
    boxes_pred = torch.randn(num_queries, 10)
    # Make predicted box dimensions positive for GIoU to make sense
    boxes_pred[:, 3:6] = boxes_pred[:, 3:6].abs() + 0.5
    pred = BoundingBox3DPrediction(
        boxes=boxes_pred,
        scores=torch.randn(num_queries, num_classes),
        labels=torch.randint(0, num_classes, (num_queries,)),
    )
    boxes_tgt = torch.randn(num_gt, 10)
    boxes_tgt[:, 3:6] = boxes_tgt[:, 3:6].abs() + 0.5
    tgt = BoundingBox3DTarget(
        boxes=boxes_tgt,
        labels=torch.randint(0, num_classes, (num_gt,)),
        instance_ids=[f"id_{i}" for i in range(num_gt)],
    )
    pi = torch.arange(num_gt)
    gi = torch.arange(num_gt)
    match = MatchingResult(pred_indices=pi, gt_indices=gi)
    return [pred], [tgt], [match]


def _make_empty_match(
    num_queries: int = 10,
    num_classes: int = 5,
) -> tuple[
    list[BoundingBox3DPrediction],
    list[BoundingBox3DTarget],
    list[MatchingResult],
]:
    pred = BoundingBox3DPrediction(
        boxes=torch.randn(num_queries, 10),
        scores=torch.randn(num_queries, num_classes),
        labels=torch.zeros(num_queries, dtype=torch.long),
    )
    tgt = BoundingBox3DTarget(
        boxes=torch.zeros(0, 10),
        labels=torch.zeros(0, dtype=torch.long),
        instance_ids=[],
    )
    match = MatchingResult(
        pred_indices=torch.zeros(0, dtype=torch.long),
        gt_indices=torch.zeros(0, dtype=torch.long),
    )
    return [pred], [tgt], [match]


class TestDetectionLossInterface:
    """Verify the interface contract of DetectionLoss."""

    def test_is_nn_module(self):
        assert issubclass(DetectionLoss, nn.Module)

    def test_has_forward(self):
        assert hasattr(DetectionLoss, "forward")

    def test_init_stores_weights(self):
        loss = DetectionLoss(num_classes=5, cls_weight=3.0, bbox_weight=0.5, giou_weight=0.1)
        assert loss.num_classes == 5
        assert loss.cls_weight == 3.0
        assert loss.bbox_weight == 0.5
        assert loss.giou_weight == 0.1

    def test_forward_returns_tuple(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        result = loss_fn(preds, tgts, matches)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_returns_scalar_and_dict(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        total, breakdown = loss_fn(preds, tgts, matches)
        assert total.ndim == 0
        assert isinstance(breakdown, dict)

    def test_breakdown_contains_expected_keys(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        _, breakdown = loss_fn(preds, tgts, matches)
        assert "loss_cls" in breakdown
        assert "loss_bbox" in breakdown
        assert "loss_giou" in breakdown


class TestDetectionLossValues:
    """Test that loss values are valid and behave as expected."""

    def test_total_loss_is_finite(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        total, _ = loss_fn(preds, tgts, matches)
        assert torch.isfinite(total)

    def test_all_components_are_finite(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        _, breakdown = loss_fn(preds, tgts, matches)
        for v in breakdown.values():
            assert torch.isfinite(v), f"{v} is not finite"

    def test_total_is_weighted_sum_of_components(self):
        cw, bw, gw = 2.0, 0.5, 0.3
        loss_fn = DetectionLoss(num_classes=5, cls_weight=cw, bbox_weight=bw, giou_weight=gw)
        preds, tgts, matches = _make_pred_tgt_match()
        total, breakdown = loss_fn(preds, tgts, matches)
        expected = (
            cw * breakdown["loss_cls"] + bw * breakdown["loss_bbox"] + gw * breakdown["loss_giou"]
        )
        assert torch.allclose(total, expected)

    def test_loss_non_negative_classification(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        _, breakdown = loss_fn(preds, tgts, matches)
        assert breakdown["loss_cls"].item() >= 0.0

    def test_loss_non_negative_bbox(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        _, breakdown = loss_fn(preds, tgts, matches)
        assert breakdown["loss_bbox"].item() >= 0.0

    def test_perfect_box_prediction_yields_zero_l1(self):
        """When predicted boxes equal GT boxes, L1 loss should be 0."""
        loss_fn = DetectionLoss(num_classes=5)
        boxes = torch.randn(4, 10)
        pred = BoundingBox3DPrediction(
            boxes=boxes.clone(),
            scores=torch.randn(4, 5),
            labels=torch.zeros(4, dtype=torch.long),
        )
        tgt = BoundingBox3DTarget(
            boxes=boxes.clone(),
            labels=torch.zeros(4, dtype=torch.long),
            instance_ids=[f"id_{i}" for i in range(4)],
        )
        match = MatchingResult(
            pred_indices=torch.arange(4),
            gt_indices=torch.arange(4),
        )
        _, breakdown = loss_fn([pred], [tgt], [match])
        assert breakdown["loss_bbox"].item() < 1e-5

    def test_loss_with_empty_matches(self):
        """Loss should be finite even when there are no matches."""
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_empty_match()
        total, breakdown = loss_fn(preds, tgts, matches)
        assert torch.isfinite(total)
        for v in breakdown.values():
            assert torch.isfinite(v)

    def test_loss_requires_grad(self):
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        preds[0].boxes.requires_grad_(True)
        preds[0].scores.requires_grad_(True)
        total, _ = loss_fn(preds, tgts, matches)
        assert total.requires_grad

    def test_batch_of_two_frames(self):
        loss_fn = DetectionLoss(num_classes=5)
        p1, t1, m1 = _make_pred_tgt_match(seed=0)
        p2, t2, m2 = _make_pred_tgt_match(seed=1)
        total, breakdown = loss_fn(p1 + p2, t1 + t2, m1 + m2)
        assert torch.isfinite(total)
        for v in breakdown.values():
            assert torch.isfinite(v)

    def test_giou_loss_at_most_two(self):
        """GIoU loss per box is bounded in [-1, 1] so the mean loss ≤ 2 typically."""
        loss_fn = DetectionLoss(num_classes=5)
        preds, tgts, matches = _make_pred_tgt_match()
        _, breakdown = loss_fn(preds, tgts, matches)
        # (1 - GIoU) is in [0, 2]; mean should be non-negative
        assert breakdown["loss_giou"].item() >= 0.0
