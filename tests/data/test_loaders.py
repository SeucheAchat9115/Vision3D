"""Unit tests for JsonLoader and ImageLoader."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image as PILImage

from vision3d.data.loaders import ImageLoader, JsonLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_FRAME: dict[str, Any] = {
    "frame_id": "frame_001",
    "timestamp": 1600000000.0,
    "past_frame_ids": [],
    "cameras": {
        "front": {
            "image_path": "images/front/frame_001.png",
            "intrinsics": [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]],
            "sensor2ego_translation": [1.5, 0.0, 1.2],
            "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        }
    },
    "annotations": [
        {
            "instance_id": "obj_001",
            "class_name": "car",
            "bbox_3d": [10.0, -2.0, 0.5, 4.5, 1.8, 1.5, 0.0, 1.0, 0.0, 0.0],
        }
    ],
    "metadata": {"weather": "clear", "time_of_day": "day"},
}


def _write_valid_json(tmp_path: Path) -> Path:
    p = tmp_path / "frame_001.json"
    p.write_text(json.dumps(_VALID_FRAME))
    return p


def _write_image(directory: Path, filename: str, size: tuple[int, int] = (64, 64)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / filename
    img = PILImage.new("RGB", size, color=(128, 64, 32))
    img.save(str(p))
    return p


# ---------------------------------------------------------------------------
# JsonLoader tests
# ---------------------------------------------------------------------------


class TestJsonLoaderInterface:
    """Verify the interface of JsonLoader."""

    def test_has_load_method(self):
        assert hasattr(JsonLoader, "load")

    def test_returns_dict(self, tmp_path: Path):
        loader = JsonLoader()
        p = _write_valid_json(tmp_path)
        data = loader.load(p)
        assert isinstance(data, dict)


class TestJsonLoaderFileHandling:
    """Test file I/O of JsonLoader."""

    def test_load_valid_file(self, tmp_path: Path):
        loader = JsonLoader()
        p = _write_valid_json(tmp_path)
        data = loader.load(p)
        assert data["frame_id"] == "frame_001"
        assert data["timestamp"] == pytest.approx(1600000000.0)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        loader = JsonLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.json")

    def test_load_returns_all_required_keys(self, tmp_path: Path):
        loader = JsonLoader()
        p = _write_valid_json(tmp_path)
        data = loader.load(p)
        for key in ("frame_id", "timestamp", "cameras", "annotations", "metadata"):
            assert key in data


class TestJsonLoaderValidation:
    """Test schema validation in JsonLoader."""

    def test_validation_disabled_accepts_minimal_json(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=False)
        p = tmp_path / "minimal.json"
        p.write_text(json.dumps({"only": "this"}))
        data = loader.load(p)
        assert data == {"only": "this"}

    def test_missing_top_level_key_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        del bad["annotations"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="Missing required key"):
            loader.load(p)

    def test_missing_camera_key_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        del bad["cameras"]["front"]["intrinsics"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="intrinsics"):
            loader.load(p)

    def test_invalid_intrinsics_shape_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        bad["cameras"]["front"]["intrinsics"] = [[1.0, 2.0]]  # Not 3x3
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="intrinsics must be 3x3"):
            loader.load(p)

    def test_invalid_rotation_length_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        bad["cameras"]["front"]["sensor2ego_rotation"] = [1.0, 0.0, 0.0]  # Only 3 elements
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="rotation must have 4 elements"):
            loader.load(p)

    def test_invalid_bbox3d_length_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        bad["annotations"][0]["bbox_3d"] = [1.0, 2.0, 3.0]  # Wrong length
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="bbox_3d must have 10 values"):
            loader.load(p)

    def test_missing_annotation_key_raises(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        bad = copy.deepcopy(_VALID_FRAME)
        del bad["annotations"][0]["instance_id"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="instance_id"):
            loader.load(p)

    def test_empty_annotations_list_is_valid(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        frame = copy.deepcopy(_VALID_FRAME)
        frame["annotations"] = []
        p = tmp_path / "empty_ann.json"
        p.write_text(json.dumps(frame))
        data = loader.load(p)
        assert data["annotations"] == []

    def test_multiple_cameras_all_validated(self, tmp_path: Path):
        loader = JsonLoader(validate_schema=True)
        frame = copy.deepcopy(_VALID_FRAME)
        frame["cameras"]["back"] = copy.deepcopy(frame["cameras"]["front"])
        del frame["cameras"]["back"]["image_path"]  # Break back camera
        p = tmp_path / "multi_cam.json"
        p.write_text(json.dumps(frame))
        with pytest.raises(ValueError, match="image_path"):
            loader.load(p)


# ---------------------------------------------------------------------------
# ImageLoader tests
# ---------------------------------------------------------------------------


class TestImageLoaderInterface:
    """Verify the interface of ImageLoader."""

    def test_has_load_method(self):
        assert hasattr(ImageLoader, "load")

    def test_returns_dict_of_tensors(self, tmp_path: Path):
        img_path = _write_image(tmp_path, "cam.png")
        loader = ImageLoader(num_threads=1, normalize=False)
        result = loader.load({"front": str(img_path)})
        assert isinstance(result, dict)
        assert "front" in result
        assert isinstance(result["front"], torch.Tensor)


class TestImageLoaderOutput:
    """Test image loading output shape, type, and normalization."""

    def test_single_camera_shape(self, tmp_path: Path):
        img_path = _write_image(tmp_path, "cam.png", size=(64, 64))
        loader = ImageLoader(num_threads=1, normalize=False)
        result = loader.load({"front": str(img_path)})
        assert result["front"].shape == (3, 64, 64)

    def test_pixel_range_no_normalization(self, tmp_path: Path):
        img_path = _write_image(tmp_path, "cam.png", size=(32, 32))
        loader = ImageLoader(num_threads=1, normalize=False)
        result = loader.load({"front": str(img_path)})
        tensor = result["front"]
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_normalization_shifts_range(self, tmp_path: Path):
        """After ImageNet normalization, pixel values can go below 0 or above 1."""
        img_path = _write_image(tmp_path, "cam.png", size=(32, 32))
        loader_norm = ImageLoader(num_threads=1, normalize=True)
        loader_raw = ImageLoader(num_threads=1, normalize=False)
        normalized = loader_norm.load({"front": str(img_path)})["front"]
        raw = loader_raw.load({"front": str(img_path)})["front"]
        # Normalized and raw should differ
        assert not torch.allclose(normalized, raw)

    def test_target_size_resize(self, tmp_path: Path):
        img_path = _write_image(tmp_path, "cam.png", size=(64, 128))
        loader = ImageLoader(num_threads=1, target_size=(32, 64), normalize=False)
        result = loader.load({"front": str(img_path)})
        assert result["front"].shape == (3, 32, 64)

    def test_multiple_cameras_concurrent(self, tmp_path: Path):
        paths = {
            "front": str(_write_image(tmp_path / "front", "img.png")),
            "left": str(_write_image(tmp_path / "left", "img.png")),
            "right": str(_write_image(tmp_path / "right", "img.png")),
        }
        loader = ImageLoader(num_threads=3, normalize=False)
        result = loader.load(paths)
        assert set(result.keys()) == {"front", "left", "right"}
        for t in result.values():
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == 3  # C=3

    def test_output_is_float_tensor(self, tmp_path: Path):
        img_path = _write_image(tmp_path, "cam.png")
        loader = ImageLoader(num_threads=1, normalize=False)
        result = loader.load({"front": str(img_path)})
        assert result["front"].dtype == torch.float32
