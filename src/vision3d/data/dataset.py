"""
Dataset module for Vision3D.

Provides the main PyTorch Dataset that orchestrates JSON loading, image loading,
box/image filtering, and data augmentation to produce standardised FrameData
batches consumed by the DataLoader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from vision3d.config.schema import BatchData, FrameData
from vision3d.data.augmentations import DataAugmenter
from vision3d.data.filters import BoxFilter, ImageFilter
from vision3d.data.loaders import ImageLoader, JsonLoader


class Vision3DDataset(Dataset):
    """PyTorch Dataset for the Vision3D generic frame format.

    Orchestrates the full data loading pipeline:
      1. Discover all frame JSON files in `data_root / split`.
      2. For each `__getitem__` call:
         a. Load the JSON via `JsonLoader`.
         b. Load all camera images via `ImageLoader` (multi-threaded).
         c. Load past-frame JSONs/images for temporal attention.
         d. Apply `BoxFilter` to discard invalid / out-of-range GT boxes.
         e. Apply `ImageFilter` to skip frames outside the ODD.
         f. Apply `DataAugmenter` to produce synchronised 2-D / 3-D augmentations.
         g. Assemble and return a `FrameData` instance.

    The companion `collate_fn` converts a list of `FrameData` objects to a
    `BatchData` object that is compatible with the Lightning training loop.

    Args:
        data_root: Root directory that contains the dataset split subdirectories.
        split: One of "train", "val", or "test".
        num_past_frames: How many past frames to include for temporal attention.
        box_filter: Optional `BoxFilter` instance; uses default settings if None.
        image_filter: Optional `ImageFilter` instance; uses default settings if None.
        augmenter: Optional `DataAugmenter`; augmentation is skipped if None.
        image_size: Target (height, width) tuple for image resizing.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_past_frames: int = 2,
        box_filter: Optional[BoxFilter] = None,
        image_filter: Optional[ImageFilter] = None,
        augmenter: Optional[DataAugmenter] = None,
        image_size: tuple[int, int] = (900, 1600),
    ) -> None:
        # TODO: store all constructor arguments as instance attributes
        # TODO: build a sorted list of all JSON frame paths under data_root/split
        # TODO: build a mapping from frame_id to JSON path for fast past-frame lookup
        # TODO: initialise JsonLoader, ImageLoader with appropriate parameters
        # TODO: store the provided (or default) BoxFilter, ImageFilter, DataAugmenter
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the total number of frames in this split."""
        # TODO: return the length of the discovered frame list
        raise NotImplementedError

    def __getitem__(self, index: int) -> FrameData:
        """Load, filter, and augment a single frame.

        Steps:
          1. Retrieve the JSON path for `index`.
          2. Parse the JSON with `JsonLoader` to get metadata, calibration, and annotations.
          3. Load all camera images with `ImageLoader`.
          4. Recursively load `num_past_frames` past frames (JSONs + images).
          5. Apply `BoxFilter` to the ground-truth annotation list.
          6. Apply `ImageFilter`; return an empty / flagged frame if filtered out.
          7. Apply `DataAugmenter`; augmenter must update camera parameters in-place.
          8. Assemble and return a `FrameData` dataclass.
        """
        # TODO: implement the full loading pipeline described above
        raise NotImplementedError

    @staticmethod
    def collate_fn(frames: List[FrameData]) -> BatchData:
        """Convert a list of FrameData objects into a single BatchData.

        Used as the `collate_fn` argument to `torch.utils.data.DataLoader`.
        Handles variable-length annotation lists by padding or stacking tensors
        appropriately.

        Args:
            frames: List of FrameData instances from `__getitem__`.

        Returns:
            A `BatchData` instance with `batch_size = len(frames)`.
        """
        # TODO: stack / pad all tensor fields across the batch dimension
        # TODO: return BatchData(batch_size=len(frames), frames=frames)
        raise NotImplementedError
