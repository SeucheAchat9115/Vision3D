"""
Geometry utilities for Vision3D.

Provides `CameraProjector`, which projects 3-D reference points in the
ego-centric coordinate frame onto the 2-D image plane of each camera using
the camera intrinsics and sensor-to-ego extrinsic parameters.

Used by `BEVEncoderLayer` during Spatial Cross-Attention to determine where
each BEV grid reference point falls on each camera image, enabling feature
sampling via `F.grid_sample`.
"""

from __future__ import annotations

from typing import Tuple

import torch


class CameraProjector:
    """Projects ego-centric 3-D points onto camera image planes.

    Handles the full projection pipeline:
      1. Transform 3-D points from ego frame to each camera's sensor frame
         using the inverse of the sensor-to-ego extrinsic (translation +
         quaternion rotation).
      2. Apply the intrinsic matrix to obtain homogeneous image coordinates.
      3. Divide by the depth (z component) to get pixel coordinates (u, v).
      4. Normalise pixel coordinates to the range [-1, 1] expected by
         `torch.nn.functional.grid_sample`.
      5. Mask out points that project behind the camera (z ≤ 0) or outside
         the image boundaries.

    The class is stateless (no learnable parameters) and all operations are
    differentiable to allow gradient flow when used inside the encoder.

    Args:
        image_height: Expected image height in pixels, used for normalisation.
        image_width: Expected image width in pixels, used for normalisation.
    """

    def __init__(
        self,
        image_height: int = 900,
        image_width: int = 1600,
    ) -> None:
        # TODO: store image_height and image_width
        raise NotImplementedError

    def project(
        self,
        points_ego: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics_translation: torch.Tensor,
        extrinsics_rotation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3-D ego-frame points onto one or more camera image planes.

        Args:
            points_ego: 3-D points in ego coordinates.
                Shape: (N, 3) where N is the number of points.
            intrinsics: Camera intrinsic matrices.
                Shape: (B, num_cameras, 3, 3).
            extrinsics_translation: Sensor-to-ego translation vectors.
                Shape: (B, num_cameras, 3).
            extrinsics_rotation: Sensor-to-ego rotation quaternions [w, x, y, z].
                Shape: (B, num_cameras, 4).

        Returns:
            Tuple of:
              - `uv_norm`: Normalised 2-D coordinates in [-1, 1] suitable for
                `F.grid_sample`. Shape: (B, num_cameras, N, 2).
              - `valid_mask`: Boolean mask indicating which projections are
                within the image boundaries and in front of the camera.
                Shape: (B, num_cameras, N).
        """
        # TODO: convert quaternion extrinsics to 3×3 rotation matrices
        # TODO: compute ego-to-sensor transform (inverse of sensor-to-ego):
        #         R_ego2cam = R_sensor2ego^T
        #         t_ego2cam = -R_ego2cam @ t_sensor2ego
        # TODO: transform points_ego to camera frame:
        #         points_cam = R_ego2cam @ points_ego.T + t_ego2cam
        # TODO: apply intrinsics: [u*z, v*z, z] = K @ points_cam
        # TODO: divide by depth z to get pixel coords (u, v)
        # TODO: compute valid_mask: z > 0 AND 0 <= u < W AND 0 <= v < H
        # TODO: normalise: u_norm = 2 * u / W - 1, v_norm = 2 * v / H - 1
        # TODO: return (uv_norm, valid_mask)
        raise NotImplementedError

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        """Convert unit quaternions to 3×3 rotation matrices.

        Args:
            quaternion: Unit quaternions in [w, x, y, z] format.
                Shape: (..., 4).

        Returns:
            Rotation matrices. Shape: (..., 3, 3).
        """
        # TODO: extract w, x, y, z from the last dimension
        # TODO: compute the 9 rotation matrix entries using the standard formula
        # TODO: stack into a (..., 3, 3) tensor and return
        raise NotImplementedError
