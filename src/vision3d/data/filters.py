"""
Filters for Vision3D dataset loading.

Provides:
  - `BoxFilter`: Removes ground-truth boxes that are out of range, physically
    invalid, or have insufficient point coverage.
  - `ImageFilter`: Removes entire frames that fall outside the Operational
    Design Domain (ODD) defined by metadata criteria.
"""

from __future__ import annotations

from typing import Any

import torch

from vision3d.config.schema import BoundingBox3DTarget


class BoxFilter:
    """Filters ground-truth 3-D bounding boxes based on spatial and quality criteria."""

    def __init__(
        self,
        max_distance: float = 50.0,
        min_points: int = 1,
        allowed_classes: list[str] | None = None,
    ) -> None:
        self.max_distance = max_distance
        self.min_points = min_points
        self.allowed_classes = allowed_classes

    def filter(
        self,
        targets: BoundingBox3DTarget,
        metadata: dict[str, Any] | None = None,
    ) -> BoundingBox3DTarget:
        """Apply all configured filters and return the surviving boxes."""
        boxes = targets.boxes
        if boxes.shape[0] == 0:
            return targets
        dist = torch.norm(boxes[:, :2], dim=1)
        keep = dist <= self.max_distance
        keep = keep & (boxes[:, 3] > 0) & (boxes[:, 4] > 0) & (boxes[:, 5] > 0)
        if metadata is not None and self.min_points > 0:
            point_counts = metadata.get("point_counts", None)
            if point_counts is not None:
                pc_tensor = torch.tensor(point_counts, dtype=torch.float32)
                keep = keep & (pc_tensor >= self.min_points)
        mask = keep
        new_ids = [iid for iid, m in zip(targets.instance_ids, mask.tolist(), strict=False) if m]
        return BoundingBox3DTarget(
            boxes=boxes[mask],
            labels=targets.labels[mask],
            instance_ids=new_ids,
        )


class ImageFilter:
    """Filters entire frames based on Operational Design Domain (ODD) metadata."""

    def __init__(
        self,
        rejected_weather: list[str] | None = None,
        require_annotations: bool = False,
        rejected_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.rejected_weather = rejected_weather if rejected_weather is not None else []
        self.require_annotations = require_annotations
        self.rejected_metadata = rejected_metadata if rejected_metadata is not None else {}

    def should_keep(
        self,
        metadata: dict[str, Any],
        num_boxes_after_filter: int,
    ) -> bool:
        """Decide whether a frame should be kept or discarded."""
        weather = metadata.get("weather", "")
        if weather in self.rejected_weather:
            return False
        if self.require_annotations and num_boxes_after_filter == 0:
            return False
        for key, val in self.rejected_metadata.items():
            if metadata.get(key) == val:
                return False
        return True
