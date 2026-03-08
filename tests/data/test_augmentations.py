"""Unit tests for DataAugmenter."""

from __future__ import annotations

import math

import pytest
import torch

from vision3d.config.schema import (
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
)
from vision3d.data.augmentations import DataAugmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    image_h: int = 32,
    image_w: int = 64,
    *,
    seed: int = 0,
) -> CameraView:
    torch.manual_seed(seed)
    return CameraView(
        image=torch.rand(3, image_h, image_w),
        intrinsics=CameraIntrinsics(
            matrix=torch.tensor(
                [[400.0, 0.0, image_w / 2], [0.0, 400.0, image_h / 2], [0.0, 0.0, 1.0]]
            )
        ),
        extrinsics=CameraExtrinsics(
            translation=torch.tensor([1.5, 0.0, 1.2]),
            rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ),
        name="front",
    )


def _make_frame(
    num_boxes: int = 4,
    image_h: int = 32,
    image_w: int = 64,
    *,
    seed: int = 0,
) -> FrameData:
    torch.manual_seed(seed)
    boxes = torch.randn(num_boxes, 10)
    boxes[:, 3:6] = boxes[:, 3:6].abs() + 0.5  # Positive dimensions
    labels = torch.zeros(num_boxes, dtype=torch.long)
    return FrameData(
        frame_id="test_frame",
        timestamp=0.0,
        cameras={"front": _make_camera(image_h, image_w, seed=seed)},
        targets=BoundingBox3DTarget(
            boxes=boxes,
            labels=labels,
            instance_ids=[f"id_{i}" for i in range(num_boxes)],
        ),
    )


def _make_frame_no_targets(
    image_h: int = 32,
    image_w: int = 64,
) -> FrameData:
    return FrameData(
        frame_id="test_frame",
        timestamp=0.0,
        cameras={"front": _make_camera(image_h, image_w)},
        targets=None,
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class TestDataAugmenterInterface:
    """Verify the interface of DataAugmenter."""

    def test_is_callable(self):
        assert callable(DataAugmenter)

    def test_call_returns_frame_data(self):
        aug = DataAugmenter(seed=42)
        frame = _make_frame()
        result = aug(frame)
        assert isinstance(result, FrameData)

    def test_call_returns_same_object(self):
        """Augmentation is done in-place; the same FrameData object is returned."""
        aug = DataAugmenter(seed=42)
        frame = _make_frame()
        result = aug(frame)
        assert result is frame


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotate3D:
    """Tests for _rotate_3d."""

    def test_box_count_preserved(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=5)
        n_before = frame.targets.boxes.shape[0]
        aug._rotate_3d(frame, math.pi / 4)
        assert frame.targets.boxes.shape[0] == n_before

    def test_box_xy_rotated(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=1)
        original_xy = frame.targets.boxes[:, :2].clone()
        aug._rotate_3d(frame, math.pi / 2)  # 90° rotation
        rotated_xy = frame.targets.boxes[:, :2]
        assert not torch.allclose(original_xy, rotated_xy, atol=1e-4)

    def test_rotation_preserves_xy_norm(self):
        """Rotation should preserve the distance of boxes from origin."""
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=4)
        norms_before = torch.norm(frame.targets.boxes[:, :2], dim=1)
        aug._rotate_3d(frame, math.pi / 3)
        norms_after = torch.norm(frame.targets.boxes[:, :2], dim=1)
        assert torch.allclose(norms_before, norms_after, atol=1e-5)

    def test_extrinsics_rotation_updated(self):
        aug = DataAugmenter()
        frame = _make_frame()
        q_before = frame.cameras["front"].extrinsics.rotation.clone()
        aug._rotate_3d(frame, math.pi / 4)
        q_after = frame.cameras["front"].extrinsics.rotation
        assert not torch.allclose(q_before, q_after)

    def test_extrinsics_rotation_stays_unit_quaternion(self):
        aug = DataAugmenter()
        frame = _make_frame()
        aug._rotate_3d(frame, math.pi / 5)
        q = frame.cameras["front"].extrinsics.rotation
        assert abs(torch.norm(q).item() - 1.0) < 1e-5

    def test_zero_angle_is_noop(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=3)
        boxes_before = frame.targets.boxes.clone()
        aug._rotate_3d(frame, 0.0)
        assert torch.allclose(frame.targets.boxes, boxes_before, atol=1e-5)

    def test_no_targets_does_not_raise(self):
        aug = DataAugmenter()
        frame = _make_frame_no_targets()
        aug._rotate_3d(frame, math.pi / 4)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScale3D:
    """Tests for _scale_3d."""

    def test_box_dims_scaled(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=4)
        boxes_before = frame.targets.boxes[:, :6].clone()
        aug._scale_3d(frame, 2.0)
        assert torch.allclose(frame.targets.boxes[:, :6], boxes_before * 2.0, atol=1e-5)

    def test_velocity_scaled(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=3)
        vel_before = frame.targets.boxes[:, 8:10].clone()
        aug._scale_3d(frame, 2.0)
        assert torch.allclose(frame.targets.boxes[:, 8:10], vel_before * 2.0, atol=1e-5)

    def test_translation_scaled(self):
        aug = DataAugmenter()
        frame = _make_frame()
        trans_before = frame.cameras["front"].extrinsics.translation.clone()
        aug._scale_3d(frame, 2.0)
        assert torch.allclose(
            frame.cameras["front"].extrinsics.translation, trans_before * 2.0, atol=1e-5
        )

    def test_unit_scale_is_noop(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=3)
        boxes_before = frame.targets.boxes.clone()
        aug._scale_3d(frame, 1.0)
        assert torch.allclose(frame.targets.boxes, boxes_before, atol=1e-5)

    def test_no_targets_does_not_raise(self):
        aug = DataAugmenter()
        frame = _make_frame_no_targets()
        aug._scale_3d(frame, 1.5)


# ---------------------------------------------------------------------------
# Horizontal flip
# ---------------------------------------------------------------------------


class TestFlipHorizontal:
    """Tests for _flip_horizontal."""

    def test_box_y_negated(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=4)
        y_before = frame.targets.boxes[:, 1].clone()
        aug._flip_horizontal(frame)
        assert torch.allclose(frame.targets.boxes[:, 1], -y_before, atol=1e-5)

    def test_double_flip_is_identity(self):
        aug = DataAugmenter()
        frame = _make_frame(num_boxes=3)
        boxes_before = frame.targets.boxes.clone()
        aug._flip_horizontal(frame)
        aug._flip_horizontal(frame)
        assert torch.allclose(frame.targets.boxes, boxes_before, atol=1e-5)

    def test_intrinsics_cx_updated(self):
        aug = DataAugmenter()
        frame = _make_frame()
        cam = frame.cameras["front"]
        W = cam.image.shape[2]
        cx_before = cam.intrinsics.matrix[0, 2].item()
        aug._flip_horizontal(frame)
        cx_after = cam.intrinsics.matrix[0, 2].item()
        assert cx_after == pytest.approx(W - cx_before)

    def test_extrinsics_translation_y_negated(self):
        aug = DataAugmenter()
        frame = _make_frame()
        ty_before = frame.cameras["front"].extrinsics.translation[1].item()
        aug._flip_horizontal(frame)
        ty_after = frame.cameras["front"].extrinsics.translation[1].item()
        assert ty_after == pytest.approx(-ty_before)

    def test_image_flipped(self):
        aug = DataAugmenter()
        frame = _make_frame()
        img_before = frame.cameras["front"].image.clone()
        aug._flip_horizontal(frame)
        img_after = frame.cameras["front"].image
        expected = torch.flip(img_before, dims=[2])
        assert torch.allclose(img_after, expected)

    def test_no_targets_does_not_raise(self):
        aug = DataAugmenter()
        frame = _make_frame_no_targets()
        aug._flip_horizontal(frame)


# ---------------------------------------------------------------------------
# Color jitter
# ---------------------------------------------------------------------------


class TestJitterColors:
    """Tests for _jitter_colors."""

    def test_images_remain_non_negative(self):
        aug = DataAugmenter(color_jitter_prob=1.0, seed=0)
        frame = _make_frame()
        # Start from clamped images
        frame.cameras["front"].image = frame.cameras["front"].image.clamp(0, 1)
        aug._jitter_colors(frame.cameras, prob=1.0)
        # After jitter + clamp, still in [0, 1]
        assert frame.cameras["front"].image.min().item() >= 0.0
        assert frame.cameras["front"].image.max().item() <= 1.0

    def test_zero_prob_does_not_change_image(self):
        aug = DataAugmenter(seed=0)
        frame = _make_frame()
        img_before = frame.cameras["front"].image.clone()
        aug._jitter_colors(frame.cameras, prob=0.0)
        assert torch.allclose(frame.cameras["front"].image, img_before)


# ---------------------------------------------------------------------------
# Crop & resize
# ---------------------------------------------------------------------------


class TestCropResize:
    """Tests for _crop_resize."""

    def test_image_shape_unchanged(self):
        aug = DataAugmenter(seed=0)
        frame = _make_frame(image_h=32, image_w=64)
        H_before = frame.cameras["front"].image.shape[1]
        W_before = frame.cameras["front"].image.shape[2]
        aug._crop_resize(frame.cameras, scale=0.8)
        assert frame.cameras["front"].image.shape == (3, H_before, W_before)

    def test_intrinsics_focal_lengths_updated(self):
        aug = DataAugmenter(seed=0)
        frame = _make_frame(image_h=32, image_w=64)
        cam = frame.cameras["front"]
        fx_before = cam.intrinsics.matrix[0, 0].item()
        aug._crop_resize(frame.cameras, scale=0.5)
        fx_after = cam.intrinsics.matrix[0, 0].item()
        # With scale=0.5 the crop is half size → focal lengths scale by 1/0.5=2
        assert fx_after > fx_before

    def test_scale_one_gives_near_identity(self):
        """Scale = 1.0 means no crop, so intrinsics should be unchanged."""
        aug = DataAugmenter(seed=0)
        frame = _make_frame(image_h=32, image_w=64)
        cam = frame.cameras["front"]
        fx_before = cam.intrinsics.matrix[0, 0].item()
        aug._crop_resize(frame.cameras, scale=1.0)
        fx_after = cam.intrinsics.matrix[0, 0].item()
        assert fx_before == pytest.approx(fx_after, rel=0.01)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestDataAugmenterPipeline:
    """Test the full __call__ pipeline."""

    def test_full_pipeline_returns_frame(self):
        aug = DataAugmenter(
            global_rot_range=(-0.1, 0.1),
            global_scale_range=(0.95, 1.05),
            flip_prob=0.5,
            color_jitter_prob=0.5,
            seed=42,
        )
        frame = _make_frame(num_boxes=5)
        result = aug(frame)
        assert isinstance(result, FrameData)

    def test_no_augmentation_with_equal_ranges(self):
        """When rot and scale ranges are equal (no variation), those ops are skipped."""
        aug = DataAugmenter(
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            flip_prob=0.0,
            color_jitter_prob=0.0,
        )
        frame = _make_frame(num_boxes=3)
        boxes_before = frame.targets.boxes.clone()
        aug(frame)
        assert torch.allclose(frame.targets.boxes, boxes_before, atol=1e-5)

    def test_seeded_augmenter_is_deterministic_flip(self):
        """Two augmenters with the same seed should produce the same flip decision."""
        frame1 = _make_frame(num_boxes=3, seed=0)
        frame2 = _make_frame(num_boxes=3, seed=0)
        aug1 = DataAugmenter(
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            flip_prob=0.5,
            color_jitter_prob=0.0,
            seed=7,
        )
        aug2 = DataAugmenter(
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            flip_prob=0.5,
            color_jitter_prob=0.0,
            seed=7,
        )
        aug1(frame1)
        aug2(frame2)
        assert torch.allclose(frame1.targets.boxes, frame2.targets.boxes, atol=1e-5)
