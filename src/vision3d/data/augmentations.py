"""
Data augmentation for Vision3D.

Provides `DataAugmenter`, which applies synchronised 3-D and 2-D augmentations
to ensure that camera images and their associated calibration parameters remain
geometrically consistent after any spatial transformation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vision3d.config.schema import (
    CameraView,
    FrameData,
)


class DataAugmenter:
    """Applies synchronised 3-D and 2-D augmentations to a FrameData instance."""

    def __init__(
        self,
        global_rot_range: tuple[float, float] = (-0.3925, 0.3925),
        global_scale_range: tuple[float, float] = (0.95, 1.05),
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        crop_scale_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.global_rot_range = global_rot_range
        self.global_scale_range = global_scale_range
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.crop_scale_range = crop_scale_range
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def __call__(self, frame: FrameData) -> FrameData:
        """Apply the full augmentation pipeline to a single frame in-place."""
        if self.global_rot_range[0] != self.global_rot_range[1]:
            angle = float(
                torch.empty(1).uniform_(self.global_rot_range[0], self.global_rot_range[1])
            )
            self._rotate_3d(frame, angle)
        if self.global_scale_range[0] != self.global_scale_range[1]:
            scale = float(
                torch.empty(1).uniform_(self.global_scale_range[0], self.global_scale_range[1])
            )
            self._scale_3d(frame, scale)
        if torch.rand(1, generator=self._generator).item() < self.flip_prob:
            self._flip_horizontal(frame)
        if self.color_jitter_prob > 0:
            self._jitter_colors(frame.cameras, self.color_jitter_prob)
        if self.crop_scale_range is not None:
            scale = float(
                torch.empty(1).uniform_(self.crop_scale_range[0], self.crop_scale_range[1])
            )
            self._crop_resize(frame.cameras, scale)
        return frame

    def _rotate_3d(self, frame: FrameData, angle_rad: float) -> None:
        """Apply a global yaw rotation around the ego Z-axis."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        if frame.targets is not None and frame.targets.boxes.shape[0] > 0:
            boxes = frame.targets.boxes
            boxes[:, :2] = boxes[:, :2] @ R.T
            boxes[:, 6:8] = boxes[:, 6:8] @ R.T
            boxes[:, 8:10] = boxes[:, 8:10] @ R.T
        for cam in frame.cameras.values():
            t = cam.extrinsics.translation
            t[:2] = R @ t[:2]
            q = cam.extrinsics.rotation
            half = angle_rad / 2
            dq = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=q.dtype)
            w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
            w2, x2, y2, z2 = q[0], q[1], q[2], q[3]
            cam.extrinsics.rotation = torch.stack(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ]
            )

    def _scale_3d(self, frame: FrameData, scale: float) -> None:
        """Apply a uniform scale factor to all 3-D quantities."""
        if frame.targets is not None and frame.targets.boxes.shape[0] > 0:
            boxes = frame.targets.boxes
            boxes[:, :6] = boxes[:, :6] * scale
            boxes[:, 8:10] = boxes[:, 8:10] * scale
        for cam in frame.cameras.values():
            cam.extrinsics.translation = cam.extrinsics.translation * scale

    def _flip_horizontal(self, frame: FrameData) -> None:
        """Flip the scene left-to-right."""
        if frame.targets is not None and frame.targets.boxes.shape[0] > 0:
            boxes = frame.targets.boxes
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]
            boxes[:, 8] = -boxes[:, 8]
        for cam in frame.cameras.values():
            cam.image = torch.flip(cam.image, dims=[2])
            W = cam.image.shape[2]
            cam.intrinsics.matrix[0, 2] = W - cam.intrinsics.matrix[0, 2]
            cam.extrinsics.translation[1] = -cam.extrinsics.translation[1]
            cam.extrinsics.rotation[2] = -cam.extrinsics.rotation[2]

    def _jitter_colors(self, cameras: dict[str, CameraView], prob: float) -> None:
        """Apply independent random colour jitter to each camera image."""
        for cam in cameras.values():
            if torch.rand(1, generator=self._generator).item() < prob:
                brightness = float(torch.empty(1).uniform_(0.8, 1.2))
                contrast = float(torch.empty(1).uniform_(0.8, 1.2))
                img = cam.image.clamp(0, 1)
                img = img * brightness
                mean = img.mean(dim=[1, 2], keepdim=True)
                img = (img - mean) * contrast + mean
                cam.image = img.clamp(0, 1)

    def _crop_resize(self, cameras: dict[str, CameraView], scale: float) -> None:
        """Randomly crop and resize each camera image, updating intrinsics."""
        for cam in cameras.values():
            _C, H, W = cam.image.shape
            new_H = int(H * scale)
            new_W = int(W * scale)
            top = int(
                torch.randint(0, max(1, H - new_H + 1), (1,), generator=self._generator).item()
            )
            left = int(
                torch.randint(0, max(1, W - new_W + 1), (1,), generator=self._generator).item()
            )
            cropped = cam.image[:, top : top + new_H, left : left + new_W]
            cam.image = F.interpolate(
                cropped.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)
            fx = cam.intrinsics.matrix[0, 0].item()
            fy = cam.intrinsics.matrix[1, 1].item()
            cx = cam.intrinsics.matrix[0, 2].item()
            cy = cam.intrinsics.matrix[1, 2].item()
            scale_x = W / new_W
            scale_y = H / new_H
            cam.intrinsics.matrix[0, 0] = fx * scale_x
            cam.intrinsics.matrix[1, 1] = fy * scale_y
            cam.intrinsics.matrix[0, 2] = (cx - left) * scale_x
            cam.intrinsics.matrix[1, 2] = (cy - top) * scale_y
