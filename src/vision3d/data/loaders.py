"""
Data loaders for Vision3D.

Provides:
  - `JsonLoader`: Reads, validates, and parses the generic per-frame JSON schema.
  - `ImageLoader`: Efficiently loads multiple camera images using a thread pool.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class JsonLoader:
    """Reads, validates, and parses the Vision3D generic per-frame JSON format."""

    def __init__(self, validate_schema: bool = True) -> None:
        self.validate_schema = validate_schema

    def load(self, json_path: Path) -> dict[str, Any]:
        """Load and parse a single frame JSON file."""
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path) as f:
            data: dict[str, Any] = json.load(f)
        if self.validate_schema:
            self._validate(data)
        return data

    def _validate(self, data: dict[str, Any]) -> None:
        """Raise ValueError if `data` violates the expected schema."""
        required = ["frame_id", "timestamp", "cameras", "annotations", "metadata"]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        for cam_name, cam in data["cameras"].items():
            for cam_key in [
                "image_path",
                "intrinsics",
                "sensor2ego_translation",
                "sensor2ego_rotation",
            ]:
                if cam_key not in cam:
                    raise ValueError(f"Camera {cam_name} missing key: {cam_key}")
            intrinsics = cam["intrinsics"]
            if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
                raise ValueError(f"Camera {cam_name} intrinsics must be 3x3")
            if len(cam["sensor2ego_rotation"]) != 4:
                raise ValueError(f"Camera {cam_name} rotation must have 4 elements")
        for ann in data["annotations"]:
            for ann_key in ["instance_id", "class_name", "bbox_3d"]:
                if ann_key not in ann:
                    raise ValueError(f"Annotation missing key: {ann_key}")
            if len(ann["bbox_3d"]) != 10:
                raise ValueError("bbox_3d must have 10 values")


class ImageLoader:
    """Loads multiple camera images concurrently using a thread pool."""

    def __init__(
        self,
        num_threads: int = 4,
        target_size: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.num_threads = num_threads
        self.target_size = target_size
        self.normalize = normalize
        self._executor: ThreadPoolExecutor | None = None
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool executor (avoids pickling issues)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_threads)
        return self._executor

    def __getstate__(self) -> dict:
        """Exclude the unpicklable ThreadPoolExecutor from serialisation."""
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def load(self, camera_paths: dict[str, str]) -> dict[str, torch.Tensor]:
        """Load all camera images for a single frame concurrently."""
        executor = self._get_executor()
        futures = {
            name: executor.submit(self._load_single, name, path)
            for name, path in camera_paths.items()
        }
        return {name: fut.result()[1] for name, fut in futures.items()}

    def _load_single(self, name: str, path: str) -> tuple[str, torch.Tensor]:
        """Load, resize, and normalise a single image."""
        img = Image.open(path).convert("RGB")
        if self.target_size is not None:
            # PIL.resize expects (width, height); target_size is (height, width)
            img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(img)
        if self.normalize:
            tensor = TF.normalize(tensor, mean=self._mean, std=self._std)
        return name, tensor

    def __del__(self) -> None:
        """Shut down the internal thread pool executor on garbage collection."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
