"""Unit tests for BoxFilter and ImageFilter."""

from __future__ import annotations

import torch

from vision3d.config.schema import BoundingBox3DTarget
from vision3d.data.filters import BoxFilter, ImageFilter


def _make_boxes(
    xy: list[tuple[float, float]],
    dims: tuple[float, float, float] = (2.0, 1.5, 1.5),
) -> BoundingBox3DTarget:
    """Build a BoundingBox3DTarget with the given (x, y) centres."""
    N = len(xy)
    boxes = torch.zeros(N, 10)
    for i, (x, y) in enumerate(xy):
        boxes[i, 0] = x
        boxes[i, 1] = y
        boxes[i, 2] = 0.5  # z
        boxes[i, 3] = dims[0]
        boxes[i, 4] = dims[1]
        boxes[i, 5] = dims[2]
    return BoundingBox3DTarget(
        boxes=boxes,
        labels=torch.zeros(N, dtype=torch.long),
        instance_ids=[f"id_{i}" for i in range(N)],
    )


class TestBoxFilterInterface:
    """Verify BoxFilter interface."""

    def test_has_filter_method(self):
        assert hasattr(BoxFilter, "filter")

    def test_filter_returns_bounding_box_3d_target(self):
        bf = BoxFilter()
        result = bf.filter(_make_boxes([(0.0, 0.0)]))
        assert isinstance(result, BoundingBox3DTarget)


class TestBoxFilterDistanceFiltering:
    """Test spatial distance filtering in BoxFilter."""

    def test_box_within_range_is_kept(self):
        bf = BoxFilter(max_distance=50.0)
        tgt = _make_boxes([(5.0, 5.0)])
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 1

    def test_box_outside_range_is_removed(self):
        bf = BoxFilter(max_distance=10.0)
        tgt = _make_boxes([(100.0, 0.0)])
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 0

    def test_box_exactly_at_boundary_is_kept(self):
        bf = BoxFilter(max_distance=10.0)
        tgt = _make_boxes([(10.0, 0.0)])
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 1

    def test_mixed_distances(self):
        bf = BoxFilter(max_distance=20.0)
        tgt = _make_boxes([(5.0, 0.0), (50.0, 0.0), (10.0, 10.0)])
        result = bf.filter(tgt)
        # (5,0)=5m kept, (50,0)=50m removed, (10,10)=14.1m kept
        assert result.boxes.shape[0] == 2


class TestBoxFilterValidityFiltering:
    """Test filtering of physically invalid boxes."""

    def test_zero_width_removed(self):
        bf = BoxFilter()
        tgt = _make_boxes([(1.0, 1.0)], dims=(0.0, 1.5, 1.5))
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 0

    def test_zero_length_removed(self):
        bf = BoxFilter()
        tgt = _make_boxes([(1.0, 1.0)], dims=(2.0, 0.0, 1.5))
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 0

    def test_zero_height_removed(self):
        bf = BoxFilter()
        tgt = _make_boxes([(1.0, 1.0)], dims=(2.0, 1.5, 0.0))
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 0

    def test_valid_box_kept(self):
        bf = BoxFilter()
        tgt = _make_boxes([(1.0, 1.0)], dims=(2.0, 1.5, 1.5))
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 1


class TestBoxFilterPointCounts:
    """Test filtering by point counts."""

    def test_sufficient_points_keeps_box(self):
        bf = BoxFilter(min_points=5)
        tgt = _make_boxes([(1.0, 1.0)])
        metadata = {"point_counts": [10]}
        result = bf.filter(tgt, metadata)
        assert result.boxes.shape[0] == 1

    def test_insufficient_points_removes_box(self):
        bf = BoxFilter(min_points=5)
        tgt = _make_boxes([(1.0, 1.0)])
        metadata = {"point_counts": [2]}
        result = bf.filter(tgt, metadata)
        assert result.boxes.shape[0] == 0

    def test_missing_point_counts_keeps_box(self):
        """When point_counts not in metadata, box should not be filtered."""
        bf = BoxFilter(min_points=5)
        tgt = _make_boxes([(1.0, 1.0)])
        metadata: dict[str, object] = {}
        result = bf.filter(tgt, metadata)
        assert result.boxes.shape[0] == 1

    def test_none_metadata_keeps_box(self):
        bf = BoxFilter(min_points=5)
        tgt = _make_boxes([(1.0, 1.0)])
        result = bf.filter(tgt, None)
        assert result.boxes.shape[0] == 1


class TestBoxFilterEdgeCases:
    """Test BoxFilter edge cases."""

    def test_empty_input_returns_empty(self):
        bf = BoxFilter()
        tgt = BoundingBox3DTarget(
            boxes=torch.zeros(0, 10),
            labels=torch.zeros(0, dtype=torch.long),
            instance_ids=[],
        )
        result = bf.filter(tgt)
        assert result.boxes.shape[0] == 0

    def test_instance_ids_preserved(self):
        bf = BoxFilter(max_distance=50.0)
        tgt = _make_boxes([(5.0, 0.0), (100.0, 0.0)])
        result = bf.filter(tgt)
        assert len(result.instance_ids) == result.boxes.shape[0]

    def test_labels_preserved(self):
        bf = BoxFilter()
        boxes = torch.zeros(2, 10)
        boxes[:, 3:6] = 1.0  # valid dims
        boxes[0, 0] = 5.0
        boxes[1, 0] = 5.0
        tgt = BoundingBox3DTarget(
            boxes=boxes,
            labels=torch.tensor([0, 2], dtype=torch.long),
            instance_ids=["a", "b"],
        )
        result = bf.filter(tgt)
        assert result.labels.shape[0] == result.boxes.shape[0]


class TestImageFilterInterface:
    """Verify ImageFilter interface."""

    def test_has_should_keep_method(self):
        assert hasattr(ImageFilter, "should_keep")

    def test_default_keeps_frame(self):
        imf = ImageFilter()
        assert imf.should_keep({"weather": "clear"}, num_boxes_after_filter=5)


class TestImageFilterWeatherFiltering:
    """Test weather-based filtering in ImageFilter."""

    def test_allowed_weather_keeps_frame(self):
        imf = ImageFilter(rejected_weather=["fog"])
        assert imf.should_keep({"weather": "clear"}, 5)

    def test_rejected_weather_removes_frame(self):
        imf = ImageFilter(rejected_weather=["fog"])
        assert not imf.should_keep({"weather": "fog"}, 5)

    def test_multiple_rejected_weather(self):
        imf = ImageFilter(rejected_weather=["fog", "snow"])
        assert not imf.should_keep({"weather": "snow"}, 5)
        assert imf.should_keep({"weather": "clear"}, 5)

    def test_empty_weather_key_keeps_frame(self):
        imf = ImageFilter(rejected_weather=["fog"])
        assert imf.should_keep({}, 5)


class TestImageFilterAnnotationRequirement:
    """Test require_annotations filtering."""

    def test_no_annotations_removed_when_required(self):
        imf = ImageFilter(require_annotations=True)
        assert not imf.should_keep({"weather": "clear"}, 0)

    def test_with_annotations_kept_when_required(self):
        imf = ImageFilter(require_annotations=True)
        assert imf.should_keep({"weather": "clear"}, 3)

    def test_no_annotations_kept_when_not_required(self):
        imf = ImageFilter(require_annotations=False)
        assert imf.should_keep({"weather": "clear"}, 0)


class TestImageFilterMetadataFiltering:
    """Test arbitrary metadata key-value rejection."""

    def test_matching_metadata_removes_frame(self):
        imf = ImageFilter(rejected_metadata={"time_of_day": "night"})
        assert not imf.should_keep({"time_of_day": "night"}, 5)

    def test_non_matching_metadata_keeps_frame(self):
        imf = ImageFilter(rejected_metadata={"time_of_day": "night"})
        assert imf.should_keep({"time_of_day": "day"}, 5)

    def test_missing_metadata_key_keeps_frame(self):
        imf = ImageFilter(rejected_metadata={"time_of_day": "night"})
        assert imf.should_keep({}, 5)

    def test_multiple_rejected_metadata(self):
        imf = ImageFilter(rejected_metadata={"weather": "fog", "split": "test"})
        assert not imf.should_keep({"weather": "fog"}, 5)
        assert not imf.should_keep({"split": "test"}, 5)
        assert imf.should_keep({"weather": "clear", "split": "train"}, 5)
