"""
Dataset module for Vision3D.

Provides the main PyTorch Dataset that orchestrates JSON loading, image loading,
box/image filtering, and data augmentation to produce standardised FrameData
batches consumed by the DataLoader.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from vision3d.config.schema import (
    BatchData,
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
)
from vision3d.data.augmentations import DataAugmenter
from vision3d.data.filters import BoxFilter, ImageFilter
from vision3d.data.loaders import ImageLoader, JsonLoader


class Vision3DDataset(Dataset[FrameData]):
    """PyTorch Dataset for the Vision3D generic frame format."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_past_frames: int = 2,
        load_past_images: bool = False,
        box_filter: BoxFilter | None = None,
        image_filter: ImageFilter | None = None,
        augmenter: DataAugmenter | None = None,
        image_size: tuple[int, int] = (900, 1600),
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_past_frames = num_past_frames
        self.load_past_images = load_past_images
        self.box_filter = box_filter if box_filter is not None else BoxFilter()
        self.image_filter = image_filter if image_filter is not None else ImageFilter()
        self.augmenter = augmenter
        self.image_size = image_size
        split_dir = self.data_root / split
        self._frame_paths = sorted(split_dir.glob("*.json")) if split_dir.exists() else []
        self._frame_id_to_path: dict[str, Path] = {p.stem: p for p in self._frame_paths}
        self._json_loader = JsonLoader(validate_schema=True)
        self._image_loader = ImageLoader(num_threads=4, target_size=image_size, normalize=True)

    def _build_camera_views(
        self, camera_data: dict[str, dict[str, object]], images: dict[str, torch.Tensor]
    ) -> dict[str, CameraView]:
        """Build typed CameraView objects from JSON camera blocks and loaded tensors."""
        cameras: dict[str, CameraView] = {}
        for cam_name, cam in camera_data.items():
            intr = torch.tensor(cam["intrinsics"], dtype=torch.float32)
            trans = torch.tensor(cam["sensor2ego_translation"], dtype=torch.float32)
            rot = torch.tensor(cam["sensor2ego_rotation"], dtype=torch.float32)
            cameras[cam_name] = CameraView(
                image=images[cam_name],
                intrinsics=CameraIntrinsics(matrix=intr),
                extrinsics=CameraExtrinsics(translation=trans, rotation=rot),
                name=cam_name,
            )
        return cameras

    def _load_past_frame(self, past_id: str) -> FrameData | None:
        """Load a past frame, optionally including camera images."""
        if past_id not in self._frame_id_to_path:
            return None
        try:
            past_data = self._json_loader.load(self._frame_id_to_path[past_id])
            if self.load_past_images:
                past_cam_paths = {
                    name: str(self.data_root / cam["image_path"])
                    for name, cam in past_data["cameras"].items()
                }
                past_images = self._image_loader.load(past_cam_paths)
                past_cameras = self._build_camera_views(past_data["cameras"], past_images)
            else:
                # Keep temporal metadata while avoiding large unused image tensors.
                past_cameras = {}
            return FrameData(
                frame_id=past_id,
                timestamp=past_data.get("timestamp", 0.0),
                cameras=past_cameras,
            )
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, index: int) -> FrameData:
        json_path = self._frame_paths[index]
        data = self._json_loader.load(json_path)
        camera_paths = {
            name: str(self.data_root / cam["image_path"]) for name, cam in data["cameras"].items()
        }
        images = self._image_loader.load(camera_paths)
        cameras = self._build_camera_views(data["cameras"], images)
        annotations = data.get("annotations", [])
        if annotations:
            boxes = torch.tensor([ann["bbox_3d"] for ann in annotations], dtype=torch.float32)
            labels = torch.zeros(len(annotations), dtype=torch.long)
            instance_ids = [ann["instance_id"] for ann in annotations]
            targets: BoundingBox3DTarget = BoundingBox3DTarget(
                boxes=boxes, labels=labels, instance_ids=instance_ids
            )
        else:
            targets = BoundingBox3DTarget(
                boxes=torch.zeros((0, 10), dtype=torch.float32),
                labels=torch.zeros(0, dtype=torch.long),
                instance_ids=[],
            )
        metadata = data.get("metadata", {})
        targets = self.box_filter.filter(targets, metadata)
        if not self.image_filter.should_keep(metadata, targets.boxes.shape[0]):
            targets = BoundingBox3DTarget(
                boxes=torch.zeros((0, 10), dtype=torch.float32),
                labels=torch.zeros(0, dtype=torch.long),
                instance_ids=[],
            )
        past_frame_ids = data.get("past_frame_ids", [])[: self.num_past_frames]
        past_frames: list[FrameData] = []
        for past_id in past_frame_ids:
            past_frame = self._load_past_frame(past_id)
            if past_frame is not None:
                past_frames.append(past_frame)
        frame = FrameData(
            frame_id=data["frame_id"],
            timestamp=data.get("timestamp", 0.0),
            cameras=cameras,
            targets=targets,
            past_frames=past_frames,
        )
        if self.augmenter is not None:
            frame = self.augmenter(frame)
        return frame

    @staticmethod
    def collate_fn(frames: list[FrameData]) -> BatchData:
        """Convert a list of FrameData objects into a single BatchData."""
        return BatchData(batch_size=len(frames), frames=frames)
