"""Unit tests for Vision3DEvaluator."""

from __future__ import annotations

import pytest
import torch

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
)
from vision3d.core.evaluators import Vision3DEvaluator


def _make_pred(
    num: int = 5,
    num_classes: int = 3,
    *,
    cls_idx: int = 0,
    score: float = 0.9,
    seed: int = 0,
) -> BoundingBox3DPrediction:
    torch.manual_seed(seed)
    boxes = torch.randn(num, 10)
    boxes[:, 2] = 0.0  # z=0
    boxes[:, 8:10] = 0.0  # no velocity
    labels = torch.full((num,), cls_idx, dtype=torch.long)
    scores = torch.full((num,), score)
    return BoundingBox3DPrediction(boxes=boxes, scores=scores, labels=labels)


def _make_tgt(
    num: int = 5,
    num_classes: int = 3,
    *,
    cls_idx: int = 0,
    seed: int = 1,
) -> BoundingBox3DTarget:
    torch.manual_seed(seed)
    boxes = torch.randn(num, 10)
    boxes[:, 2] = 0.0
    boxes[:, 8:10] = 0.0
    labels = torch.full((num,), cls_idx, dtype=torch.long)
    return BoundingBox3DTarget(
        boxes=boxes,
        labels=labels,
        instance_ids=[f"id_{i}" for i in range(num)],
    )


class TestVision3DEvaluatorInterface:
    """Verify the public interface of Vision3DEvaluator."""

    def test_has_reset(self):
        assert hasattr(Vision3DEvaluator, "reset")

    def test_has_update(self):
        assert hasattr(Vision3DEvaluator, "update")

    def test_has_compute(self):
        assert hasattr(Vision3DEvaluator, "compute")

    def test_init_defaults(self):
        ev = Vision3DEvaluator()
        assert ev.num_classes == 10
        assert ev.eval_range == 50.0
        assert len(ev.distance_thresholds) == 4

    def test_init_custom(self):
        ev = Vision3DEvaluator(
            num_classes=3,
            class_names=["car", "ped", "cycle"],
            eval_range=25.0,
            distance_thresholds=[0.5, 1.0],
        )
        assert ev.num_classes == 3
        assert ev.class_names == ["car", "ped", "cycle"]
        assert ev.eval_range == 25.0
        assert ev.distance_thresholds == [0.5, 1.0]

    def test_auto_class_names_generated(self):
        ev = Vision3DEvaluator(num_classes=3)
        assert ev.class_names == ["class_0", "class_1", "class_2"]

    def test_compute_returns_dict(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        result = ev.compute()
        assert isinstance(result, dict)


class TestVision3DEvaluatorReset:
    """Test that reset() properly clears state."""

    def test_reset_clears_predictions(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        ev.reset()
        assert len(ev._all_predictions) == 0
        assert len(ev._all_targets) == 0

    def test_compute_after_reset_returns_zeros(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        ev.reset()
        metrics = ev.compute()
        assert metrics["mAP"] == pytest.approx(0.0)


class TestVision3DEvaluatorUpdate:
    """Test update() accumulation logic."""

    def test_update_accumulates_multiple_calls(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred(seed=0)], [_make_tgt(seed=1)])
        ev.update([_make_pred(seed=2)], [_make_tgt(seed=3)])
        assert len(ev._all_predictions) == 2
        assert len(ev._all_targets) == 2

    def test_update_filters_by_eval_range(self):
        """Boxes far outside eval_range should be removed."""
        ev = Vision3DEvaluator(num_classes=3, eval_range=10.0)
        far_boxes = torch.zeros(3, 10)
        far_boxes[:, :2] = 100.0  # Well outside 10m range
        tgt = BoundingBox3DTarget(
            boxes=far_boxes,
            labels=torch.zeros(3, dtype=torch.long),
            instance_ids=["a", "b", "c"],
        )
        pred = BoundingBox3DPrediction(
            boxes=far_boxes.clone(),
            scores=torch.ones(3) * 0.9,
            labels=torch.zeros(3, dtype=torch.long),
        )
        ev.update([pred], [tgt])
        # After filtering, the stored targets should have 0 boxes
        assert ev._all_targets[0].boxes.shape[0] == 0

    def test_update_keeps_near_boxes(self):
        ev = Vision3DEvaluator(num_classes=3, eval_range=50.0)
        near_boxes = torch.zeros(2, 10)
        near_boxes[:, :2] = 1.0
        tgt = BoundingBox3DTarget(
            boxes=near_boxes,
            labels=torch.zeros(2, dtype=torch.long),
            instance_ids=["a", "b"],
        )
        pred = BoundingBox3DPrediction(
            boxes=near_boxes.clone(),
            scores=torch.ones(2),
            labels=torch.zeros(2, dtype=torch.long),
        )
        ev.update([pred], [tgt])
        assert ev._all_targets[0].boxes.shape[0] == 2

    def test_update_handles_empty_predictions(self):
        ev = Vision3DEvaluator(num_classes=3)
        pred = BoundingBox3DPrediction(
            boxes=torch.zeros(0, 10),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )
        ev.update([pred], [_make_tgt()])
        assert len(ev._all_predictions) == 1


class TestVision3DEvaluatorCompute:
    """Test compute() metric calculations."""

    def test_compute_has_map_key(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        assert "mAP" in metrics

    def test_compute_has_nds_key(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        assert "NDS" in metrics

    def test_compute_has_per_class_ap(self):
        class_names = ["car", "ped", "cycle"]
        ev = Vision3DEvaluator(num_classes=3, class_names=class_names)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        for name in class_names:
            assert f"AP/{name}" in metrics

    def test_compute_has_tp_metrics(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        for key in ["ATE", "ASE", "AOE", "AVE"]:
            assert key in metrics

    def test_map_between_zero_and_one(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        assert 0.0 <= metrics["mAP"] <= 1.0

    def test_nds_between_zero_and_one(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred()], [_make_tgt()])
        metrics = ev.compute()
        assert 0.0 <= metrics["NDS"] <= 1.0

    def test_no_predictions_yields_zero_map(self):
        ev = Vision3DEvaluator(num_classes=3)
        pred = BoundingBox3DPrediction(
            boxes=torch.zeros(0, 10),
            scores=torch.zeros(0),
            labels=torch.zeros(0, dtype=torch.long),
        )
        ev.update([pred], [_make_tgt()])
        metrics = ev.compute()
        assert metrics["mAP"] == pytest.approx(0.0)

    def test_perfect_nearby_prediction_raises_ap(self):
        """Predictions that exactly match GT centres should yield AP > 0."""
        ev = Vision3DEvaluator(num_classes=1, distance_thresholds=[2.0])
        gt_boxes = torch.zeros(4, 10)
        gt_boxes[:, :2] = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        tgt = BoundingBox3DTarget(
            boxes=gt_boxes,
            labels=torch.zeros(4, dtype=torch.long),
            instance_ids=[f"id_{i}" for i in range(4)],
        )
        pred_boxes = gt_boxes.clone()
        pred = BoundingBox3DPrediction(
            boxes=pred_boxes,
            scores=torch.ones(4) * 0.99,
            labels=torch.zeros(4, dtype=torch.long),
        )
        ev.update([pred], [tgt])
        metrics = ev.compute()
        assert metrics["mAP"] > 0.0

    def test_compute_tp_metrics_all_finite(self):
        ev = Vision3DEvaluator(num_classes=3)
        ev.update([_make_pred(5)], [_make_tgt(5)])
        metrics = ev.compute()
        for key in ["ATE", "ASE", "AOE", "AVE"]:
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0.0

    def test_compute_after_multiple_updates(self):
        ev = Vision3DEvaluator(num_classes=3)
        for i in range(5):
            ev.update([_make_pred(seed=i)], [_make_tgt(seed=i + 5)])
        metrics = ev.compute()
        assert torch.isfinite(torch.tensor(metrics["mAP"]))

    def test_compute_ap_for_class_empty_returns_zero(self):
        """_compute_ap_for_class should return 0.0 when there are no GT boxes."""
        ev = Vision3DEvaluator(num_classes=1)
        result = ev._compute_ap_for_class(0, 1.0)
        assert result == pytest.approx(0.0)
