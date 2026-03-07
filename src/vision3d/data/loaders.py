"""
Data loaders for Vision3D.

Provides:
  - `JsonLoader`: Reads, validates, and parses the generic per-frame JSON schema.
  - `ImageLoader`: Efficiently loads multiple camera images using a thread pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class JsonLoader:
    """Reads, validates, and parses the Vision3D generic per-frame JSON format.

    The expected JSON schema is documented in the project README (Section 6).
    This class is the single point of truth for reading raw JSON data; it should
    not perform any tensor conversion or augmentation.

    Responsibilities:
      - Open and decode the JSON file at the given path.
      - Validate that all required top-level keys are present.
      - Validate the shape and types of calibration matrices, quaternions, etc.
      - Return a plain Python dictionary representing the parsed frame.

    Args:
        validate_schema: If True, perform strict key/type validation on load.
            Disable for speed-critical inference pipelines if schema is trusted.
    """

    def __init__(self, validate_schema: bool = True) -> None:
        # TODO: store validate_schema flag
        raise NotImplementedError

    def load(self, json_path: Path) -> Dict[str, Any]:
        """Load and parse a single frame JSON file.

        Args:
            json_path: Absolute or relative path to the `.json` file.

        Returns:
            A Python dict matching the schema described in README Section 6,
            containing keys: "frame_id", "timestamp", "past_frame_ids",
            "cameras", "annotations", "metadata".

        Raises:
            FileNotFoundError: If `json_path` does not exist.
            ValueError: If the JSON does not conform to the expected schema
                        (only when `validate_schema=True`).
        """
        # TODO: open and json.load the file
        # TODO: call self._validate(data) if self.validate_schema
        # TODO: return the parsed dict
        raise NotImplementedError

    def _validate(self, data: Dict[str, Any]) -> None:
        """Raise ValueError if `data` violates the expected schema.

        Checks:
          - Required top-level keys exist ("frame_id", "timestamp", "cameras", …).
          - Each camera entry contains "image_path", "intrinsics",
            "sensor2ego_translation", "sensor2ego_rotation".
          - Each annotation entry contains "instance_id", "class_name", "bbox_3d"
            with exactly 10 values.
          - "intrinsics" is a 3×3 list of floats.
          - "sensor2ego_rotation" is a list of exactly 4 floats.
        """
        # TODO: implement all validation checks listed above
        raise NotImplementedError


class ImageLoader:
    """Loads multiple camera images concurrently using a thread pool.

    Reads pre-undistorted PNG or JPEG images from disk and converts them to
    normalised float tensors of shape (C, H, W).

    Responsibilities:
      - Accept a list of image paths and load them in parallel.
      - Resize images to the requested target size if provided.
      - Normalise pixel values to [0, 1] (or ImageNet mean/std if configured).
      - Return a dict mapping camera name to image tensor.

    Args:
        num_threads: Number of I/O worker threads for parallel loading.
        target_size: Optional (height, width) tuple to resize images.
            No resizing is performed when None.
        normalize: If True, apply ImageNet channel mean/std normalisation.
    """

    def __init__(
        self,
        num_threads: int = 4,
        target_size: Optional[tuple[int, int]] = None,
        normalize: bool = True,
    ) -> None:
        # TODO: store num_threads, target_size, normalize
        # TODO: create a ThreadPoolExecutor with num_threads workers
        raise NotImplementedError

    def load(self, camera_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Load all camera images for a single frame concurrently.

        Args:
            camera_paths: Dict mapping camera name (e.g. "front") to the
                absolute image file path.

        Returns:
            Dict mapping camera name to a float32 image tensor of
            shape (3, H, W) with values in the expected normalisation range.

        Raises:
            FileNotFoundError: If any image path does not exist on disk.
        """
        # TODO: submit _load_single tasks to the thread pool for each path
        # TODO: collect futures and return results as a dict
        raise NotImplementedError

    def _load_single(self, name: str, path: str) -> tuple[str, torch.Tensor]:
        """Load, resize, and normalise a single image.

        Args:
            name: Camera name used as the key in the returned tuple.
            path: Filesystem path to the image file.

        Returns:
            Tuple of (name, tensor) where tensor has shape (3, H, W).
        """
        # TODO: open image with PIL or cv2
        # TODO: resize to self.target_size if provided
        # TODO: convert to float32 tensor and normalise
        # TODO: return (name, tensor)
        raise NotImplementedError

    def __del__(self) -> None:
        """Shut down the internal thread pool executor on garbage collection."""
        # TODO: call self._executor.shutdown(wait=False)
        pass
