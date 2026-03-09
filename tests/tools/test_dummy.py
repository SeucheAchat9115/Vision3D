"""Unit tests for DummyDatasetGenerator."""

from __future__ import annotations

import json
from pathlib import Path

from tools.generate_dummy_dataset import DummyDatasetGenerator

# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestDummyDatasetGeneratorInterface:
    """Verify DummyDatasetGenerator interface."""

    def test_has_generate_method(self):
        assert hasattr(DummyDatasetGenerator, "generate")

    def test_init_stores_params(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            output_root=str(tmp_path),
            num_frames=5,
            num_cameras=2,
            image_height=32,
            image_width=32,
            seed=42,
        )
        assert gen.num_frames == 5
        assert gen.num_cameras == 2
        assert gen.image_height == 32
        assert gen.image_width == 32

    def test_output_dir_created_on_init(self, tmp_path: Path):
        DummyDatasetGenerator(str(tmp_path), num_cameras=1, seed=0)
        assert (tmp_path / "train").exists()
        assert (tmp_path / "val").exists()


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


class TestDummyDatasetGeneratorGenerate:
    """Test that generate() produces valid output."""

    def test_generate_creates_json_files(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=3, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        json_files = list((tmp_path / "train").glob("*.json"))
        assert len(json_files) == 3

    def test_generate_creates_image_files(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=2, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        imgs = list((tmp_path / "images").rglob("*.png"))
        assert len(imgs) == 2  # 2 frames × 1 camera

    def test_generated_json_passes_schema(self, tmp_path: Path):
        from vision3d.data.loaders import JsonLoader

        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=2, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        loader = JsonLoader(validate_schema=True)
        for json_path in (tmp_path / "train").glob("*.json"):
            data = loader.load(json_path)
            assert "frame_id" in data
            assert "cameras" in data
            assert "annotations" in data

    def test_generated_json_has_correct_intrinsics_shape(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=1, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        json_files = list((tmp_path / "train").glob("*.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        for cam in data["cameras"].values():
            assert len(cam["intrinsics"]) == 3
            for row in cam["intrinsics"]:
                assert len(row) == 3

    def test_generated_json_has_valid_rotation(self, tmp_path: Path):
        import numpy as np

        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=2, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        for json_path in (tmp_path / "train").glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            for cam in data["cameras"].values():
                q = cam["sensor2ego_rotation"]
                assert len(q) == 4
                norm = np.linalg.norm(q)
                assert abs(norm - 1.0) < 1e-5, f"Quaternion not unit: {q}, norm={norm}"

    def test_generated_bbox3d_has_10_values(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path),
            num_frames=3,
            num_cameras=1,
            image_height=16,
            image_width=16,
            max_boxes_per_frame=5,
            seed=0,
        )
        gen.generate(split="train")
        for json_path in (tmp_path / "train").glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            for ann in data["annotations"]:
                assert len(ann["bbox_3d"]) == 10

    def test_generated_bbox3d_sin_cos_valid(self, tmp_path: Path):
        """sin^2(theta) + cos^2(theta) should equal 1."""
        gen = DummyDatasetGenerator(
            str(tmp_path),
            num_frames=3,
            num_cameras=1,
            image_height=16,
            image_width=16,
            max_boxes_per_frame=5,
            seed=0,
        )
        gen.generate(split="train")
        for json_path in (tmp_path / "train").glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            for ann in data["annotations"]:
                b = ann["bbox_3d"]
                sin_t, cos_t = b[6], b[7]
                assert abs(sin_t**2 + cos_t**2 - 1.0) < 1e-5

    def test_multiple_cameras(self, tmp_path: Path):
        num_cameras = 3
        gen = DummyDatasetGenerator(
            str(tmp_path),
            num_frames=1,
            num_cameras=num_cameras,
            image_height=16,
            image_width=16,
            seed=0,
        )
        gen.generate(split="train")
        json_files = list((tmp_path / "train").glob("*.json"))
        with open(json_files[0]) as f:
            data = json.load(f)
        assert len(data["cameras"]) == num_cameras

    def test_generate_val_split(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=2, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="val")
        json_files = list((tmp_path / "val").glob("*.json"))
        assert len(json_files) == 2

    def test_seeded_generation_reproducible(self, tmp_path: Path):
        """Two generators with the same seed should produce the same number of total boxes."""
        gen1 = DummyDatasetGenerator(
            str(tmp_path / "run1"),
            num_frames=3,
            num_cameras=1,
            image_height=16,
            image_width=16,
            seed=42,
        )
        gen1.generate(split="train")
        # Re-create a fresh generator with the same seed
        gen2 = DummyDatasetGenerator(
            str(tmp_path / "run2"),
            num_frames=3,
            num_cameras=1,
            image_height=16,
            image_width=16,
            seed=42,
        )
        gen2.generate(split="train")
        counts1 = sorted(
            [
                len(json.loads(p.read_text())["annotations"])
                for p in (tmp_path / "run1" / "train").glob("*.json")
            ]
        )
        counts2 = sorted(
            [
                len(json.loads(p.read_text())["annotations"])
                for p in (tmp_path / "run2" / "train").glob("*.json")
            ]
        )
        assert counts1 == counts2

    def test_metadata_has_expected_keys(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), num_frames=2, num_cameras=1, image_height=16, image_width=16, seed=0
        )
        gen.generate(split="train")
        for json_path in (tmp_path / "train").glob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            assert "weather" in data["metadata"]
            assert "time_of_day" in data["metadata"]

    def test_past_frame_ids_populated(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path),
            num_frames=5,
            num_cameras=1,
            image_height=16,
            image_width=16,
            num_past_frames=2,
            seed=0,
        )
        gen.generate(split="train")
        all_data = []
        for json_path in (tmp_path / "train").glob("*.json"):
            with open(json_path) as f:
                all_data.append(json.load(f))
        # At least some frames should have past_frame_ids
        all_past = [len(d["past_frame_ids"]) for d in all_data]
        assert max(all_past) > 0


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestDummyDatasetGeneratorHelpers:
    """Test internal helper methods."""

    def test_generate_random_intrinsics_shape(self, tmp_path: Path):
        gen = DummyDatasetGenerator(str(tmp_path), seed=0)
        intr = gen._generate_random_intrinsics()
        assert len(intr) == 3
        assert all(len(row) == 3 for row in intr)

    def test_generate_random_intrinsics_principal_point(self, tmp_path: Path):
        """Principal point should be near the image centre."""
        gen = DummyDatasetGenerator(str(tmp_path), image_height=100, image_width=200, seed=0)
        intr = gen._generate_random_intrinsics()
        cx, cy = intr[0][2], intr[1][2]
        assert 50.0 <= cx <= 150.0, f"cx={cx} not near centre"
        assert 0.0 <= cy <= 100.0, f"cy={cy} not near centre"

    def test_generate_random_extrinsics_shapes(self, tmp_path: Path):
        gen = DummyDatasetGenerator(str(tmp_path), seed=0)
        translation, rotation = gen._generate_random_extrinsics()
        assert len(translation) == 3
        assert len(rotation) == 4

    def test_generate_random_boxes_count(self, tmp_path: Path):
        gen = DummyDatasetGenerator(str(tmp_path), seed=0)
        for n in [0, 1, 5]:
            boxes = gen._generate_random_boxes(n)
            assert len(boxes) == n

    def test_generate_random_boxes_bbox3d_length(self, tmp_path: Path):
        gen = DummyDatasetGenerator(str(tmp_path), seed=0)
        boxes = gen._generate_random_boxes(3)
        for b in boxes:
            assert len(b["bbox_3d"]) == 10

    def test_generate_random_image_creates_file(self, tmp_path: Path):
        gen = DummyDatasetGenerator(
            str(tmp_path), image_height=16, image_width=16, num_cameras=1, seed=0
        )
        rel_path = gen._generate_random_image("front", "test_img")
        assert (tmp_path / rel_path).exists()
