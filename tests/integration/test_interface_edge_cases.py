"""Integration tests: interface edge-cases and cross-component contracts.

Covered scenarios:
  - BatchData.to() on CPU is a no-op that returns self
  - All image tensors reside on the target device after .to()
  - FrameData with targets=None works in model forward
  - Output shapes are consistent with model constructor config
  - MatchingResult pred_indices and gt_indices have equal length
  - Loss breakdown weighted sum equals total loss
  - Per-class AP keys exposed for named classes
  - Temporal past_frames in FrameData don't break forward pass
  - prev_bev=None vs. real tensor yields different model outputs
  - Full pipeline with num_classes=1 (minimum)
"""

from __future__ import annotations

import pytest
import torch

from vision3d.config.schema import BatchData, BoundingBox3DPrediction
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from tests.integration.helpers import make_batch, make_frame, make_small_model


class TestInterfaceEdgeCases:
    """Tests that verify interface contracts are satisfied even for edge cases."""

    def test_batch_data_to_device_is_noop_on_cpu(self) -> None:
        """BatchData.to('cpu') must succeed and return self."""
        batch = make_batch()
        returned = batch.to(torch.device("cpu"))
        assert returned is batch

    def test_batch_data_to_moves_image_tensors(self) -> None:
        """All image tensors in BatchData must reside on the target device after .to()."""
        batch = make_batch()
        batch.to(torch.device("cpu"))
        for frame in batch.frames:
            for cam in frame.cameras.values():
                assert cam.image.device.type == "cpu"

    def test_frame_data_with_all_optional_fields_none(self) -> None:
        """FrameData with targets=None must not raise in model forward pass."""
        model = make_small_model()
        frame = make_frame(with_targets=False)
        batch = BatchData(batch_size=1, frames=[frame])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_model_output_shapes_consistent_with_config(self) -> None:
        """All model output shapes must match the constructor configuration."""
        B, Q, C_cls = 3, 7, 4
        model = make_small_model(num_classes=C_cls, num_queries=Q)
        batch = make_batch(batch_size=B, num_classes=C_cls)
        preds, _ = model(batch)
        assert preds.boxes.shape == (B, Q, 10)
        assert preds.scores.shape == (B, Q)
        assert preds.labels.shape == (B, Q)

    def test_matching_result_consistent_length(self) -> None:
        """pred_indices and gt_indices in MatchingResult must have equal length."""
        model = make_small_model(num_classes=5)
        matcher = HungarianMatcher()
        batch = make_batch(num_classes=5)
        preds_batch, _ = model(batch)
        for i, frame in enumerate(batch.frames):
            if frame.targets is None:
                continue
            pred = BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            result = matcher.match(pred, frame.targets)
            assert result.pred_indices.shape == result.gt_indices.shape

    def test_loss_breakdown_sums_to_total(self) -> None:
        """The weighted sum of breakdown losses must equal the total loss."""
        num_classes = 5
        cw, bw, gw = 2.0, 0.5, 0.3
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(
            num_classes=num_classes, cls_weight=cw, bbox_weight=bw, giou_weight=gw
        )
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
        total, breakdown = loss_fn(pred_list, targets, matches)
        expected = (
            cw * breakdown["loss_cls"] + bw * breakdown["loss_bbox"] + gw * breakdown["loss_giou"]
        )
        assert torch.allclose(total, expected, atol=1e-5)

    def test_evaluator_per_class_ap_keys_present(self) -> None:
        """Vision3DEvaluator must expose AP/{class_name} for every class."""
        num_classes = 3
        class_names = ["car", "pedestrian", "cyclist"]
        model = make_small_model(num_classes=num_classes)
        evaluator = Vision3DEvaluator(num_classes=num_classes, class_names=class_names)
        batch = make_batch(num_classes=num_classes)
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
        metrics = evaluator.compute()
        for name in class_names:
            assert f"AP/{name}" in metrics

    def test_model_with_temporal_frames_in_batch(self) -> None:
        """FrameData.past_frames must not interfere with model forward pass."""
        model = make_small_model()
        past = make_frame(frame_id="past_frame", seed=99)
        current = make_frame(frame_id="current_frame", seed=0)
        current.past_frames = [past]
        batch = BatchData(batch_size=1, frames=[current])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_bev_encoder_prev_bev_none_vs_provided(self) -> None:
        """Model output must differ when prev_bev is None vs. a real tensor."""
        B, C, H, W = 1, 32, 4, 4
        model = make_small_model(embed_dims=C, bev_h=H, bev_w=W)
        model.eval()
        batch = make_batch(batch_size=B)
        with torch.no_grad():
            preds_no_prev, bev = model(batch, prev_bev=None)
            preds_with_prev, _ = model(batch, prev_bev=bev)
        assert not torch.allclose(preds_no_prev.boxes, preds_with_prev.boxes)

    def test_full_pipeline_with_single_class(self) -> None:
        """num_classes=1 (minimum) must work end-to-end."""
        num_classes = 1
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = make_frame(num_classes=num_classes, num_boxes=2)
        batch = BatchData(batch_size=1, frames=[frame])
        preds_batch, _ = model(batch)
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[0],
                scores=preds_batch.scores[0],
                labels=preds_batch.labels[0],
            )
        ]
        targets = [frame.targets]
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        assert torch.isfinite(total)

    def test_predictions_are_detachable(self) -> None:
        """BoundingBox3DPrediction tensors must be detachable without raising."""
        model = make_small_model()
        batch = make_batch()
        preds, _ = model(batch)
        # Detach should not raise
        boxes_np = preds.boxes.detach().cpu().numpy()
        scores_np = preds.scores.detach().cpu().numpy()
        assert boxes_np.shape == preds.boxes.shape
        assert scores_np.shape == preds.scores.shape
