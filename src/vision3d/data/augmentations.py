"""
Data augmentation for Vision3D.

Provides `DataAugmenter`, which applies synchronised 3-D and 2-D augmentations
to ensure that camera images and their associated calibration parameters remain
geometrically consistent after any spatial transformation.

Key invariant: any operation that modifies the 3-D ego-centric space (e.g.
global rotation/scaling) **must** update `CameraExtrinsics` accordingly, and
any operation that modifies the 2-D image space (e.g. horizontal flip) **must**
update `CameraIntrinsics` (and, if applicable, `CameraExtrinsics`) accordingly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from vision3d.config.schema import (
    BoundingBox3DTarget,
    CameraExtrinsics,
    CameraIntrinsics,
    CameraView,
    FrameData,
)


class DataAugmenter:
    """Applies synchronised 3-D and 2-D augmentations to a FrameData instance.

    All augmentations are applied with configurable probabilities and are
    designed to be composable. The augmenter is applied **after** filtering and
    **before** tensor assembly into the DataLoader batch.

    Supported augmentations:
      - **Global rotation** around the Z-axis (ego-centric yaw): rotates all
        GT box centres, box headings, and velocity vectors, and updates each
        camera's sensor-to-ego extrinsic rotation accordingly.
      - **Global scaling**: uniformly scales box dimensions and centres, and
        updates extrinsic translations accordingly.
      - **Random horizontal flip** (left-right): mirrors the 2-D images, flips
        the sign of Y-axis box coordinates, sin(θ), vx, and updates intrinsics
        (cx ← W − cx) and extrinsics.
      - **Random image colour jitter**: applies brightness, contrast, saturation,
        and hue jitter independently per camera without affecting 3-D geometry.
      - **Random image crop / resize**: crops and resizes images and updates the
        intrinsic matrix (fx, fy, cx, cy) to match the new resolution.

    Args:
        global_rot_range: (min_rad, max_rad) range for random yaw rotation.
            Set to (0, 0) to disable.
        global_scale_range: (min_scale, max_scale) for uniform scaling.
            Set to (1, 1) to disable.
        flip_prob: Probability of applying horizontal flip. Set to 0 to disable.
        color_jitter_prob: Probability of applying colour jitter per camera.
        crop_scale_range: (min_scale, max_scale) for random crop resize.
            Disabled when None.
        seed: Optional fixed seed for reproducible augmentation during debugging.
    """

    def __init__(
        self,
        global_rot_range: Tuple[float, float] = (-0.3925, 0.3925),
        global_scale_range: Tuple[float, float] = (0.95, 1.05),
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        crop_scale_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        # TODO: store all configuration parameters as instance attributes
        # TODO: initialise a torch.Generator for reproducible sampling if seed is set
        raise NotImplementedError

    def __call__(self, frame: FrameData) -> FrameData:
        """Apply the full augmentation pipeline to a single frame in-place.

        Each augmentation is sampled independently; not all will be applied on
        every call. The returned FrameData object has updated images, box
        tensors, and camera calibration parameters.

        Args:
            frame: A `FrameData` instance populated by `Vision3DDataset`.

        Returns:
            The augmented `FrameData` (may be the same object mutated in-place
            or a new copy, depending on implementation preference).
        """
        # TODO: sample and apply global rotation (call _rotate_3d)
        # TODO: sample and apply global scaling (call _scale_3d)
        # TODO: sample and apply horizontal flip (call _flip_horizontal)
        # TODO: sample and apply colour jitter (call _jitter_colors)
        # TODO: sample and apply random crop/resize (call _crop_resize) if configured
        # TODO: return the augmented frame
        raise NotImplementedError

    def _rotate_3d(
        self,
        frame: FrameData,
        angle_rad: float,
    ) -> None:
        """Apply a global yaw rotation around the ego Z-axis.

        Modifies in-place:
          - `frame.targets.boxes[:, :2]` (x, y centre coordinates).
          - `frame.targets.boxes[:, 6:8]` (sin θ, cos θ heading encoding).
          - `frame.targets.boxes[:, 8:10]` (vx, vy velocity).
          - `camera.extrinsics.translation` for each camera view.
          - `camera.extrinsics.rotation` (quaternion) for each camera view.

        Args:
            frame: Frame to modify.
            angle_rad: Rotation angle in radians (positive = counter-clockwise).
        """
        # TODO: build a 2×2 rotation matrix from angle_rad
        # TODO: rotate box centres (x, y) using the rotation matrix
        # TODO: rotate (sin θ, cos θ) to update heading
        # TODO: rotate (vx, vy) velocity vectors
        # TODO: rotate each camera extrinsic translation (x, y components)
        # TODO: compose the yaw rotation into each camera extrinsic quaternion
        raise NotImplementedError

    def _scale_3d(
        self,
        frame: FrameData,
        scale: float,
    ) -> None:
        """Apply a uniform scale factor to all 3-D quantities.

        Modifies in-place:
          - `frame.targets.boxes[:, :6]` (x, y, z, w, l, h).
          - `frame.targets.boxes[:, 8:10]` (vx, vy — scaled proportionally).
          - `camera.extrinsics.translation` for each camera view.

        Args:
            frame: Frame to modify.
            scale: Positive scalar multiplier.
        """
        # TODO: multiply box centre and dimension parameters by scale
        # TODO: scale velocity components proportionally
        # TODO: scale each camera extrinsic translation by scale
        raise NotImplementedError

    def _flip_horizontal(self, frame: FrameData) -> None:
        """Flip the scene left-to-right.

        Modifies in-place:
          - All camera images (torch.flip along the W dimension).
          - `frame.targets.boxes[:, 1]` (y coordinate, negate).
          - `frame.targets.boxes[:, 6]` (sin θ, negate).
          - `frame.targets.boxes[:, 8]` (vx, negate).
          - `camera.intrinsics.matrix[0, 2]` (cx ← image_width − cx).
          - `camera.extrinsics.translation[1]` (y component, negate).
          - `camera.extrinsics.rotation` quaternion (conjugate y component).

        Args:
            frame: Frame to modify.
        """
        # TODO: flip each camera image horizontally
        # TODO: negate y coordinate, sin(θ), and vx in targets.boxes
        # TODO: update cx in each camera intrinsic matrix
        # TODO: negate y translation and update rotation quaternion for each camera
        raise NotImplementedError

    def _jitter_colors(
        self,
        cameras: Dict[str, CameraView],
        prob: float,
    ) -> None:
        """Apply independent random colour jitter to each camera image.

        Does not affect 3-D geometry or calibration parameters.

        Args:
            cameras: Dict of camera views to modify in-place.
            prob: Per-camera probability of applying jitter.
        """
        # TODO: for each camera, sample Bernoulli(prob)
        # TODO: if selected, apply random brightness, contrast, saturation, hue
        #       adjustments to the image tensor
        raise NotImplementedError

    def _crop_resize(
        self,
        cameras: Dict[str, CameraView],
        scale: float,
    ) -> None:
        """Randomly crop and resize each camera image, updating intrinsics.

        Args:
            cameras: Dict of camera views to modify in-place.
            scale: Fraction of the original image area to keep before resizing
                back to the original resolution.
        """
        # TODO: sample a random crop region for each camera
        # TODO: crop the image tensor
        # TODO: resize to original dimensions
        # TODO: update fx, fy, cx, cy in the intrinsic matrix to reflect the crop
        raise NotImplementedError
