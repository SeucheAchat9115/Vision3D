"""Integration tests: data augmentation and filtering pipeline.

Covered scenarios:
  - DataAugmenter output still passes through BEVFormerModel
  - Augmented data produces finite model predictions
  - DataAugmenter preserves image spatial dimensions
  - DataAugmenter preserves camera intrinsic matrix shape
  - BoxFilter removes out-of-range boxes and model still forward-passes
  - BoxFilter + DataAugmenter combined pipeline ends in finite loss
  - ImageFilter accepts/rejects frames based on metadata
  - Deterministic augmentation (fixed seed) yields identical results
  - Augmentation with all transforms disabled is identity-like
  - BoxFilter with no boxes remaining still yields valid FrameData
"""

from __future__ import annotations

import torch

from tests.integration.helpers import make_frame, make_small_model
from vision3d.config.schema import BatchData, BoundingBox3DPrediction, BoundingBox3DTarget
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.data.augmentations import DataAugmenter
from vision3d.data.filters import BoxFilter, ImageFilter


class TestAugmentationModelIntegration:
    """Verify that DataAugmenter output is consumable by BEVFormerModel."""

    def test_augmented_frame_passes_model_forward(self) -> None:
        """Augmented frames must pass through the model without error."""
        augmenter = DataAugmenter(seed=42)
        model = make_small_model()
        frame = make_frame(num_boxes=3, num_classes=5)
        augmented = augmenter(frame)
        batch = BatchData(batch_size=1, frames=[augmented])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_augmented_batch_predictions_finite(self) -> None:
        """Model predictions on augmented data must be finite."""
        augmenter = DataAugmenter(seed=7)
        model = make_small_model()
        frames = [augmenter(make_frame(num_boxes=2, num_classes=5, seed=i)) for i in range(2)]
        batch = BatchData(batch_size=2, frames=frames)
        preds, _ = model(batch)
        assert torch.isfinite(preds.boxes).all()
        assert torch.isfinite(preds.scores).all()

    def test_augmentation_preserves_image_shape(self) -> None:
        """DataAugmenter must not change the spatial dimensions of camera images."""
        augmenter = DataAugmenter(seed=0)
        frame = make_frame(image_h=64, image_w=64)
        original_shapes = {name: cam.image.shape for name, cam in frame.cameras.items()}
        augmented = augmenter(frame)
        for name, shape in original_shapes.items():
            assert augmented.cameras[name].image.shape == shape

    def test_augmentation_preserves_intrinsic_matrix_shape(self) -> None:
        """DataAugmenter must preserve the 3×3 shape of the intrinsic matrix."""
        augmenter = DataAugmenter(seed=1)
        frame = make_frame()
        augmented = augmenter(frame)
        for cam in augmented.cameras.values():
            assert cam.intrinsics.matrix.shape == (3, 3)

    def test_augmentation_with_flip_training_pass(self) -> None:
        """Flipped frames must produce a finite training loss."""
        augmenter = DataAugmenter(
            flip_prob=1.0,
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            color_jitter_prob=0.0,
            seed=0,
        )
        num_classes = 5
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        frame = augmenter(make_frame(num_boxes=3, num_classes=num_classes))
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

    def test_deterministic_augmentation_same_output(self) -> None:
        """Two DataAugmenters with the same seed must produce identical outputs."""
        frame1 = make_frame(num_boxes=3, seed=0)
        frame2 = make_frame(num_boxes=3, seed=0)
        aug1 = DataAugmenter(seed=123)
        aug2 = DataAugmenter(seed=123)
        result1 = aug1(frame1)
        result2 = aug2(frame2)
        for name in result1.cameras:
            assert torch.allclose(result1.cameras[name].image, result2.cameras[name].image)

    def test_augmentation_disabled_preserves_boxes(self) -> None:
        """With all augmentations disabled the box values must remain unchanged."""
        augmenter = DataAugmenter(
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            flip_prob=0.0,
            color_jitter_prob=0.0,
            crop_scale_range=None,
            seed=0,
        )
        frame = make_frame(num_boxes=3, num_classes=5, seed=42)
        original_boxes = frame.targets.boxes.clone()
        augmented = augmenter(frame)
        assert torch.allclose(augmented.targets.boxes, original_boxes)


class TestBoxFilterModelIntegration:
    """Verify that BoxFilter output is correctly consumed by BEVFormerModel."""

    def test_filtered_targets_pass_model_forward(self) -> None:
        """Filtered annotations must not prevent the model from running a forward pass."""
        box_filter = BoxFilter(max_distance=50.0)
        model = make_small_model(num_classes=5)
        frame = make_frame(num_boxes=5, num_classes=5)
        frame.targets = box_filter.filter(frame.targets)
        batch = BatchData(batch_size=1, frames=[frame])
        preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_filter_removes_distant_boxes(self) -> None:
        """BoxFilter must remove boxes whose (x, y) distance exceeds max_distance."""
        box_filter = BoxFilter(max_distance=5.0)
        targets = BoundingBox3DTarget(
            boxes=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # dist=1 → keep
                    [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # dist=10 → remove
                ]
            ),
            labels=torch.tensor([0, 1]),
            instance_ids=["a", "b"],
        )
        filtered = box_filter.filter(targets)
        assert filtered.boxes.shape[0] == 1
        assert filtered.instance_ids == ["a"]

    def test_filter_removes_invalid_dimension_boxes(self) -> None:
        """BoxFilter must remove boxes with zero or negative width/length/height."""
        box_filter = BoxFilter(max_distance=100.0)
        targets = BoundingBox3DTarget(
            boxes=torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # valid → keep
                    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # w=0 → remove
                    [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # l<0 → remove
                ]
            ),
            labels=torch.tensor([0, 1, 2]),
            instance_ids=["a", "b", "c"],
        )
        filtered = box_filter.filter(targets)
        assert filtered.boxes.shape[0] == 1
        assert filtered.instance_ids == ["a"]

    def test_filter_empty_targets_unchanged(self) -> None:
        """BoxFilter on empty targets must return an empty BoundingBox3DTarget."""
        box_filter = BoxFilter(max_distance=50.0)
        targets = BoundingBox3DTarget(
            boxes=torch.zeros(0, 10),
            labels=torch.zeros(0, dtype=torch.long),
            instance_ids=[],
        )
        filtered = box_filter.filter(targets)
        assert filtered.boxes.shape[0] == 0

    def test_filter_plus_augmentation_training_pass(self) -> None:
        """BoxFilter → DataAugmenter → Model → Loss must produce finite loss."""
        num_classes = 5
        box_filter = BoxFilter(max_distance=50.0)
        augmenter = DataAugmenter(seed=99)
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)

        frame = make_frame(num_boxes=4, num_classes=num_classes)
        frame.targets = box_filter.filter(frame.targets)
        frame = augmenter(frame)

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


class TestImageFilterIntegration:
    """Verify that ImageFilter correctly accepts or rejects frames."""

    def test_image_filter_accepts_clear_weather(self) -> None:
        """ImageFilter must keep a frame with acceptable weather."""
        image_filter = ImageFilter(rejected_weather=["rain", "fog"])
        metadata = {"weather": "clear", "time_of_day": "day"}
        assert image_filter.should_keep(metadata, num_boxes_after_filter=2)

    def test_image_filter_rejects_bad_weather(self) -> None:
        """ImageFilter must discard a frame with rejected weather."""
        image_filter = ImageFilter(rejected_weather=["rain", "fog"])
        metadata = {"weather": "rain"}
        assert not image_filter.should_keep(metadata, num_boxes_after_filter=5)

    def test_image_filter_requires_annotations(self) -> None:
        """With require_annotations=True, frames with zero boxes must be rejected."""
        image_filter = ImageFilter(require_annotations=True)
        metadata = {"weather": "clear"}
        assert not image_filter.should_keep(metadata, num_boxes_after_filter=0)
        assert image_filter.should_keep(metadata, num_boxes_after_filter=1)

    def test_image_filter_rejects_custom_metadata(self) -> None:
        """ImageFilter must discard frames matching rejected_metadata key-value pairs."""
        image_filter = ImageFilter(rejected_metadata={"time_of_day": "night"})
        assert not image_filter.should_keep({"time_of_day": "night"}, num_boxes_after_filter=3)
        assert image_filter.should_keep({"time_of_day": "day"}, num_boxes_after_filter=3)
