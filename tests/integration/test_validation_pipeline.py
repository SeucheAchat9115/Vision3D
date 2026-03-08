"""Integration tests: BEVFormerModel → Vision3DEvaluator pipeline.

Covered scenarios:
  - mAP metric is finite and in [0, 1]
  - All expected metric keys are present
  - Accumulation across multiple batches
  - State reset between evaluation epochs
  - No crash with zero ground-truth boxes
  - Per-class AP keys exposed when class names are provided
"""

from __future__ import annotations

import pytest
import torch

from vision3d.config.schema import BoundingBox3DPrediction
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.models.bevformer import BEVFormerModel
from vision3d.config.schema import BatchData
from tests.integration.helpers import make_batch, make_small_model


class TestModelEvaluatorIntegration:
    """Integration tests for the BEVFormerModel → Vision3DEvaluator pipeline."""

    def _run_validation_pass(
        self,
        batch: BatchData,
        model: BEVFormerModel,
        evaluator: Vision3DEvaluator,
    ) -> None:
        """Helper: forward then accumulate into evaluator."""
        with torch.no_grad():
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
        evaluator.update(pred_list, targets)

    def test_validation_pass_produces_finite_map(self) -> None:
        """Validation pipeline must produce a finite mAP."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        assert isinstance(metrics["mAP"], float)
        assert 0.0 <= metrics["mAP"] <= 1.0

    def test_validation_pass_metrics_keys(self) -> None:
        """All expected metric keys must be present in compute() output."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        for key in ["mAP", "NDS", "ATE", "ASE", "AOE", "AVE"]:
            assert key in metrics

    def test_validation_multiple_batches(self) -> None:
        """Evaluator must correctly accumulate across multiple validation batches."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        for _i in range(3):
            batch = make_batch(batch_size=2, num_classes=num_classes)
            self._run_validation_pass(batch, model, evaluator)
        assert len(evaluator._all_predictions) == 6
        metrics = evaluator.compute()
        assert torch.isfinite(torch.tensor(metrics["mAP"]))

    def test_validation_reset_clears_state(self) -> None:
        """reset() must clear accumulated state so re-evaluation starts fresh."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        evaluator.reset()
        assert len(evaluator._all_predictions) == 0
        assert evaluator.compute()["mAP"] == pytest.approx(0.0)

    def test_validation_with_no_gt_boxes(self) -> None:
        """Validation must not crash when frames contain no GT annotations."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes)
        batch = make_batch(num_boxes=0, num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        assert isinstance(metrics["mAP"], float)

    def test_validation_per_class_ap_keys_present(self) -> None:
        """Vision3DEvaluator must expose AP/{class_name} for every named class."""
        num_classes = 3
        class_names = ["car", "pedestrian", "cyclist"]
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes, class_names=class_names)
        batch = make_batch(num_classes=num_classes)
        self._run_validation_pass(batch, model, evaluator)
        metrics = evaluator.compute()
        for name in class_names:
            assert f"AP/{name}" in metrics

    def test_validation_two_epochs_independent(self) -> None:
        """Two successive evaluation epochs (with reset in between) must give independent mAP."""
        num_classes = 3
        model = make_small_model(num_classes=num_classes)
        model.eval()  # deterministic behaviour required for reproducible mAP
        evaluator = Vision3DEvaluator(num_classes=num_classes)

        # Epoch 1
        batch1 = make_batch(batch_size=2, num_classes=num_classes)
        self._run_validation_pass(batch1, model, evaluator)
        metrics1 = evaluator.compute()

        evaluator.reset()

        # Epoch 2 – identical data so mAP should be the same
        self._run_validation_pass(batch1, model, evaluator)
        metrics2 = evaluator.compute()

        assert metrics1["mAP"] == pytest.approx(metrics2["mAP"])
