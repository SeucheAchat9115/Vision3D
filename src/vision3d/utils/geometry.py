"""
Geometry utilities for Vision3D.

Provides `CameraProjector`, which projects 3-D reference points in the
ego-centric coordinate frame onto the 2-D image plane of each camera using
the camera intrinsics and sensor-to-ego extrinsic parameters.
"""

from __future__ import annotations

import torch


class CameraProjector:
    """Projects ego-centric 3-D points onto camera image planes."""

    def __init__(
        self,
        image_height: int = 900,
        image_width: int = 1600,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width

    def project(
        self,
        points_ego: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics_translation: torch.Tensor,
        extrinsics_rotation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3-D ego-frame points onto one or more camera image planes.

        Args:
            points_ego: Shape (N, 3).
            intrinsics: Shape (B, num_cameras, 3, 3).
            extrinsics_translation: Shape (B, num_cameras, 3).
            extrinsics_rotation: Shape (B, num_cameras, 4) — [w, x, y, z].

        Returns:
            uv_norm: (B, num_cameras, N, 2) normalised coords in [-1, 1].
            valid_mask: (B, num_cameras, N) bool mask.
        """
        R = self.quaternion_to_rotation_matrix(extrinsics_rotation)  # (B, C, 3, 3)
        R_e2c = R.transpose(-1, -2)  # (B, C, 3, 3)
        t = extrinsics_translation.unsqueeze(-1)  # (B, C, 3, 1)
        t_e2c = -(R_e2c @ t)  # (B, C, 3, 1)
        pts = points_ego.T.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, N)
        pts_cam = R_e2c @ pts + t_e2c  # (B, C, 3, N)
        pts_img = intrinsics @ pts_cam  # (B, C, 3, N)
        z = pts_img[:, :, 2:3, :]  # (B, C, 1, N)
        u = pts_img[:, :, 0, :] / (z.squeeze(2) + 1e-8)
        v = pts_img[:, :, 1, :] / (z.squeeze(2) + 1e-8)
        z_val = z.squeeze(2)
        valid = (z_val > 0) & (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u_norm = 2.0 * u / self.image_width - 1.0
        v_norm = 2.0 * v / self.image_height - 1.0
        uv_norm = torch.stack([u_norm, v_norm], dim=-1)  # (B, C, N, 2)
        return uv_norm, valid

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """Convert unit quaternions [w, x, y, z] to 3×3 rotation matrices.

        Args:
            quaternion: Shape (..., 4).

        Returns:
            Shape (..., 3, 3).
        """
        w = quaternion[..., 0]
        x = quaternion[..., 1]
        y = quaternion[..., 2]
        z = quaternion[..., 3]
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(quaternion.shape[:-1] + (3, 3))
        return R
