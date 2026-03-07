"""
Filters for Vision3D dataset loading.

Provides:
  - `BoxFilter`: Removes ground-truth boxes that are out of range, physically
    invalid, or have insufficient point coverage.
  - `ImageFilter`: Removes entire frames that fall outside the Operational
    Design Domain (ODD) defined by metadata criteria.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from vision3d.config.schema import BoundingBox3DTarget


class BoxFilter:
    """Filters ground-truth 3-D bounding boxes based on spatial and quality criteria.

    Applied after JSON loading but before augmentation. Removes boxes that
    would be harmful to train on (e.g. too far away, occluded, degenerate size).

    Filtering criteria (all configurable via constructor):
      - **Distance filter**: Drop boxes whose centre distance from ego exceeds
        `max_distance` metres.
      - **Point count filter**: Drop boxes with fewer than `min_points` LiDAR
        points (when point count is available in the metadata).
      - **Validity filter**: Drop boxes with non-positive width, length, or height.
      - **Class filter**: Optionally keep only boxes belonging to a set of
        allowed class names.

    Args:
        max_distance: Maximum radial distance (metres) to keep a box.
        min_points: Minimum number of LiDAR points inside a box. Set to 0 to
            disable this filter.
        allowed_classes: If provided, only boxes with a class name in this list
            are kept. Pass None to keep all classes.
    """

    def __init__(
        self,
        max_distance: float = 50.0,
        min_points: int = 1,
        allowed_classes: Optional[List[str]] = None,
    ) -> None:
        # TODO: store max_distance, min_points, allowed_classes as attributes
        raise NotImplementedError

    def filter(
        self,
        targets: BoundingBox3DTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BoundingBox3DTarget:
        """Apply all configured filters and return the surviving boxes.

        Args:
            targets: Full ground-truth target container for one frame.
            metadata: Optional per-annotation metadata dict (e.g. point counts).

        Returns:
            A new `BoundingBox3DTarget` containing only the boxes that passed
            all filters. May have zero boxes if everything is filtered out.
        """
        # TODO: compute per-box keep mask for distance criterion
        # TODO: compute per-box keep mask for point count criterion (if metadata available)
        # TODO: compute per-box keep mask for physical validity (w, l, h > 0)
        # TODO: compute per-box keep mask for class whitelist if set
        # TODO: combine masks with logical AND
        # TODO: index targets.boxes, targets.labels, targets.instance_ids by the mask
        # TODO: return a new BoundingBox3DTarget with the filtered data
        raise NotImplementedError


class ImageFilter:
    """Filters entire frames based on Operational Design Domain (ODD) metadata.

    Allows the dataset to skip frames that are outside the conditions the model
    is designed to operate in (e.g. night, rain, or frames with no annotations).

    Filtering criteria (all configurable via constructor):
      - **Weather filter**: Reject frames whose "weather" metadata field matches
        any entry in `rejected_weather` (e.g. ["rain", "fog"]).
      - **Daytime filter**: Reject frames outside the allowed time-of-day range.
      - **Annotation filter**: Optionally reject frames that have zero GT boxes
        after `BoxFilter` has been applied (useful for training).
      - **Custom metadata filter**: Reject frames where any key in
        `rejected_metadata` matches the provided value.

    Args:
        rejected_weather: Weather conditions to exclude. Empty list disables
            this filter.
        require_annotations: If True, frames with zero GT boxes are discarded.
        rejected_metadata: Dict of metadata key → rejected value pairs for
            arbitrary ODD filtering.
    """

    def __init__(
        self,
        rejected_weather: Optional[List[str]] = None,
        require_annotations: bool = False,
        rejected_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # TODO: store rejected_weather (defaulting to empty list if None)
        # TODO: store require_annotations flag
        # TODO: store rejected_metadata (defaulting to empty dict if None)
        raise NotImplementedError

    def should_keep(
        self,
        metadata: Dict[str, Any],
        num_boxes_after_filter: int,
    ) -> bool:
        """Decide whether a frame should be kept or discarded.

        Args:
            metadata: The "metadata" dict from the parsed frame JSON.
            num_boxes_after_filter: Number of GT boxes remaining after BoxFilter.

        Returns:
            True if the frame passes all ODD criteria and should be loaded;
            False if it should be skipped.
        """
        # TODO: check weather field against self.rejected_weather
        # TODO: check require_annotations against num_boxes_after_filter
        # TODO: iterate self.rejected_metadata and compare against metadata values
        # TODO: return True only when all checks pass
        raise NotImplementedError
