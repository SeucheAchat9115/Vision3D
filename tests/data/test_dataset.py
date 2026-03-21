"""Unit tests for the dataset module (Vision3DDataset)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vision3d.config.schema import BatchData, FrameData
from vision3d.data.augmentations import DataAugmenter
from vision3d.data.dataset import Vision3DDataset
from vision3d.data.filters import BoxFilter, ImageFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(directory: Path, filename: str, h: int = 16, w: int = 16) -> Path:
    """Write a small random PNG image."""
    import numpy as np
    from PIL import Image as PILImage

    directory.mkdir(parents=True, exist_ok=True)
    p = directory / filename
    arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
    PILImage.fromarray(arr).save(str(p))
    return p


def _make_frame_json(
    root: Path,
    frame_id: str,
    split: str = "train",
    num_boxes: int = 2,
    past_frame_ids: list[str] | None = None,
    weather: str = "clear",
    image_h: int = 16,
    image_w: int = 16,
) -> Path:
    """Create a JSON frame file and dummy image files."""
    cam_name = "front"
    img_rel_path = f"images/{cam_name}/{frame_id}.png"
    _make_image(root / "images" / cam_name, f"{frame_id}.png", h=image_h, w=image_w)

    boxes = []
    for i in range(num_boxes):
        boxes.append(
            {
                "instance_id": f"obj_{i}",
                "class_name": "car",
                "bbox_3d": [float(i), 0.0, 0.0, 2.0, 1.5, 1.5, 0.0, 1.0, 0.0, 0.0],
            }
        )

    frame_data = {
        "frame_id": frame_id,
        "timestamp": 1600000000.0,
        "past_frame_ids": past_frame_ids or [],
        "cameras": {
            cam_name: {
                "image_path": img_rel_path,
                "intrinsics": [
                    [400.0, 0.0, image_w / 2],
                    [0.0, 400.0, image_h / 2],
                    [0.0, 0.0, 1.0],
                ],
                "sensor2ego_translation": [1.5, 0.0, 1.2],
                "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            }
        },
        "annotations": boxes,
        "metadata": {"weather": weather, "time_of_day": "day"},
    }

    (root / split).mkdir(parents=True, exist_ok=True)
    json_path = root / split / f"{frame_id}.json"
    with open(json_path, "w") as f:
        json.dump(frame_data, f)
    return json_path


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestVision3DDatasetInterface:
    """Verify Vision3DDataset interface."""

    def test_is_torch_dataset(self):
        from torch.utils.data import Dataset

        assert issubclass(Vision3DDataset, Dataset)

    def test_has_len(self):
        assert hasattr(Vision3DDataset, "__len__")

    def test_has_getitem(self):
        assert hasattr(Vision3DDataset, "__getitem__")

    def test_has_collate_fn(self):
        assert hasattr(Vision3DDataset, "collate_fn")


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestVision3DDatasetLoading:
    """Test loading behaviour of Vision3DDataset."""

    def test_len_reflects_frame_count(self, tmp_path: Path):
        for i in range(3):
            _make_frame_json(tmp_path, f"frame_{i:03d}")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        assert len(ds) == 3

    def test_getitem_returns_frame_data(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        assert isinstance(item, FrameData)

    def test_getitem_frame_id_matches(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        assert item.frame_id == "frame_000"

    def test_getitem_cameras_populated(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        assert "front" in item.cameras
        cam = item.cameras["front"]
        assert isinstance(cam.image, torch.Tensor)
        assert cam.image.shape == (3, 16, 16)

    def test_getitem_targets_populated(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", num_boxes=3)
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        assert item.targets is not None
        assert item.targets.boxes.shape == (3, 10)

    def test_getitem_empty_annotations(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", num_boxes=0)
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        assert item.targets is not None
        assert item.targets.boxes.shape[0] == 0

    def test_empty_split_dir_gives_zero_length(self, tmp_path: Path):
        ds = Vision3DDataset(str(tmp_path), split="nonexistent", image_size=(16, 16))
        assert len(ds) == 0

    def test_image_resized_to_target_size(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", image_h=64, image_w=64)
        target_h, target_w = 8, 8
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(target_h, target_w))
        item = ds[0]
        assert item.cameras["front"].image.shape == (3, target_h, target_w)


class TestVision3DDatasetFiltering:
    """Test filtering logic in Vision3DDataset."""

    def test_box_filter_applied(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", num_boxes=4)
        bf = BoxFilter(max_distance=-1.0)  # Negative range → all boxes filtered
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16), box_filter=bf)
        item = ds[0]
        assert item.targets is not None
        assert item.targets.boxes.shape[0] == 0

    def test_image_filter_rejected_weather(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", num_boxes=2, weather="fog")
        imf = ImageFilter(rejected_weather=["fog"])
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16), image_filter=imf)
        item = ds[0]
        # When image filter rejects a frame, targets are set to empty
        assert item.targets is not None
        assert item.targets.boxes.shape[0] == 0


class TestVision3DDatasetAugmentation:
    """Test augmentation integration in Vision3DDataset."""

    def test_augmenter_applied(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", num_boxes=2)
        aug = DataAugmenter(
            global_rot_range=(0.0, 0.0),
            global_scale_range=(1.0, 1.0),
            flip_prob=1.0,  # Always flip
            color_jitter_prob=0.0,
            seed=0,
        )
        ds_aug = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16), augmenter=aug)
        ds_no_aug = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item_no_aug = ds_no_aug[0]
        item_aug = ds_aug[0]
        # With flip_prob=1.0, the image should be horizontally flipped
        aug_img = item_aug.cameras["front"].image
        no_aug_img = item_no_aug.cameras["front"].image
        # The flipped image should differ from the original (unless perfectly symmetric)
        flipped = torch.flip(no_aug_img, dims=[2])
        assert torch.allclose(aug_img, flipped, atol=1e-4)


class TestVision3DDatasetTemporalFrames:
    """Test temporal past frame loading."""

    def test_past_frames_loaded(self, tmp_path: Path):
        _make_frame_json(tmp_path, "past_000")
        _make_frame_json(tmp_path, "frame_001", past_frame_ids=["past_000"])
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16), num_past_frames=1)
        # frame_001 references past_000
        frames_by_id = {ds[i].frame_id: ds[i] for i in range(len(ds))}
        item = frames_by_id["frame_001"]
        assert len(item.past_frames) == 1
        assert item.past_frames[0].frame_id == "past_000"
        assert item.past_frames[0].cameras == {}

    def test_past_frame_images_loaded_when_enabled(self, tmp_path: Path):
        _make_frame_json(tmp_path, "past_000")
        _make_frame_json(tmp_path, "frame_001", past_frame_ids=["past_000"])
        ds = Vision3DDataset(
            str(tmp_path),
            split="train",
            image_size=(16, 16),
            num_past_frames=1,
            load_past_images=True,
        )
        frames_by_id = {ds[i].frame_id: ds[i] for i in range(len(ds))}
        item = frames_by_id["frame_001"]
        assert len(item.past_frames) == 1
        assert "front" in item.past_frames[0].cameras
        assert item.past_frames[0].cameras["front"].image.shape == (3, 16, 16)

    def test_missing_past_frame_gracefully_skipped(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000", past_frame_ids=["nonexistent_id"])
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        item = ds[0]
        # Nonexistent past frame should be silently skipped
        assert len(item.past_frames) == 0


# ---------------------------------------------------------------------------
# CollateFunc tests
# ---------------------------------------------------------------------------


class TestCollateFunction:
    """Test the static collate_fn method."""

    def test_collate_returns_batch_data(self, tmp_path: Path):
        for i in range(2):
            _make_frame_json(tmp_path, f"frame_{i:03d}")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        frames = [ds[0], ds[1]]
        batch = Vision3DDataset.collate_fn(frames)
        assert isinstance(batch, BatchData)

    def test_collate_batch_size_correct(self, tmp_path: Path):
        for i in range(3):
            _make_frame_json(tmp_path, f"frame_{i:03d}")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        frames = [ds[i] for i in range(3)]
        batch = Vision3DDataset.collate_fn(frames)
        assert batch.batch_size == 3

    def test_collate_preserves_frames(self, tmp_path: Path):
        for i in range(2):
            _make_frame_json(tmp_path, f"frame_{i:03d}")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        frames = [ds[0], ds[1]]
        batch = Vision3DDataset.collate_fn(frames)
        assert len(batch.frames) == 2

    def test_collate_single_frame(self, tmp_path: Path):
        _make_frame_json(tmp_path, "frame_000")
        ds = Vision3DDataset(str(tmp_path), split="train", image_size=(16, 16))
        batch = Vision3DDataset.collate_fn([ds[0]])
        assert batch.batch_size == 1
