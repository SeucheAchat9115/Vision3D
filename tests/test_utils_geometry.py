"""Unit tests for CameraProjector."""

from __future__ import annotations

import math

import torch

from vision3d.utils.geometry import CameraProjector


def _identity_quaternion(B: int = 1, C: int = 1) -> torch.Tensor:
    """Return identity quaternions [w=1, x=0, y=0, z=0] of shape (B, C, 4)."""
    q = torch.zeros(B, C, 4)
    q[..., 0] = 1.0  # w = 1
    return q


def _identity_intrinsics(
    B: int = 1,
    C: int = 1,
    fx: float = 800.0,
    fy: float = 800.0,
    cx: float = 400.0,
    cy: float = 300.0,
) -> torch.Tensor:
    K = torch.zeros(B, C, 3, 3)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0
    return K


class TestQuaternionToRotationMatrix:
    """Tests for CameraProjector.quaternion_to_rotation_matrix."""

    def test_identity_quaternion_gives_identity_matrix(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (1, 4)
        R = CameraProjector.quaternion_to_rotation_matrix(q)  # (1, 3, 3)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-6)

    def test_output_shape_1d_batch(self):
        q = torch.randn(4, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = CameraProjector.quaternion_to_rotation_matrix(q)
        assert R.shape == (4, 3, 3)

    def test_output_shape_2d_batch(self):
        q = torch.randn(2, 3, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = CameraProjector.quaternion_to_rotation_matrix(q)
        assert R.shape == (2, 3, 3, 3)

    def test_rotation_matrix_is_orthogonal(self):
        """R @ R.T should equal the identity matrix for a valid rotation."""
        q = torch.randn(8, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = CameraProjector.quaternion_to_rotation_matrix(q)
        product = R @ R.transpose(-1, -2)
        eye = torch.eye(3).unsqueeze(0).expand_as(product)
        assert torch.allclose(product, eye, atol=1e-5)

    def test_determinant_equals_one(self):
        """Rotation matrices should have determinant = +1."""
        q = torch.randn(8, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = CameraProjector.quaternion_to_rotation_matrix(q)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(8), atol=1e-5)

    def test_90deg_yaw_rotation(self):
        """A 90° CCW rotation around Z maps X-axis to Y-axis."""
        half = math.pi / 4
        q = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
        R = CameraProjector.quaternion_to_rotation_matrix(q)[0]
        x_rotated = R @ torch.tensor([1.0, 0.0, 0.0])
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(x_rotated, expected, atol=1e-5)

    def test_negative_w_quaternion(self):
        """Quaternion and its negation represent the same rotation."""
        q_pos = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q_neg = -q_pos
        R_pos = CameraProjector.quaternion_to_rotation_matrix(q_pos)
        R_neg = CameraProjector.quaternion_to_rotation_matrix(q_neg)
        assert torch.allclose(R_pos, R_neg, atol=1e-6)


class TestCameraProjectorProject:
    """Tests for CameraProjector.project."""

    def test_output_shapes(self):
        proj = CameraProjector(image_height=600, image_width=800)
        N, B, C = 10, 2, 3
        points = torch.randn(N, 3)
        intrinsics = _identity_intrinsics(B, C, fx=400.0, cx=400.0, cy=300.0)
        translation = torch.zeros(B, C, 3)
        rotation = _identity_quaternion(B, C)
        uv_norm, valid = proj.project(points, intrinsics, translation, rotation)
        assert uv_norm.shape == (B, C, N, 2)
        assert valid.shape == (B, C, N)

    def test_valid_mask_dtype(self):
        proj = CameraProjector(image_height=600, image_width=800)
        N, B, C = 5, 1, 1
        points = torch.randn(N, 3)
        intrinsics = _identity_intrinsics(B, C)
        translation = torch.zeros(B, C, 3)
        rotation = _identity_quaternion(B, C)
        _, valid = proj.project(points, intrinsics, translation, rotation)
        assert valid.dtype == torch.bool

    def test_point_behind_camera_is_invalid(self):
        """A point with negative Z in camera frame should be marked invalid."""
        proj = CameraProjector(image_height=600, image_width=800)
        # Place point at z = -5 in ego = camera frame (no extrinsic offset)
        points = torch.tensor([[-0.0, 0.0, -5.0]])  # N=1
        intrinsics = _identity_intrinsics(1, 1, fx=400.0, cx=400.0, cy=300.0)
        translation = torch.zeros(1, 1, 3)
        rotation = _identity_quaternion(1, 1)
        _, valid = proj.project(points, intrinsics, translation, rotation)
        assert not valid[0, 0, 0].item()

    def test_point_in_front_of_camera_is_valid(self):
        """A point directly in front of the camera at principal point should be valid."""
        H, W = 600, 800
        proj = CameraProjector(image_height=H, image_width=W)
        cx, cy = W / 2, H / 2
        # In ego=camera frame: project (0,0,10) with K[[cx,cy]] → (cx, cy) → in image
        points = torch.tensor([[0.0, 0.0, 10.0]])
        K = torch.zeros(1, 1, 3, 3)
        K[0, 0, 0, 0] = 400.0  # fx
        K[0, 0, 1, 1] = 400.0  # fy
        K[0, 0, 0, 2] = cx
        K[0, 0, 1, 2] = cy
        K[0, 0, 2, 2] = 1.0
        translation = torch.zeros(1, 1, 3)
        rotation = _identity_quaternion(1, 1)
        _, valid = proj.project(points, K, translation, rotation)
        assert valid[0, 0, 0].item()

    def test_normalized_coords_range_for_valid_points(self):
        """Valid projected points should have uv_norm in [-1, 1]."""
        H, W = 600, 800
        proj = CameraProjector(image_height=H, image_width=W)
        # Points along z-axis in front of camera, projected to principal point
        points = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 20.0]])
        K = torch.zeros(1, 1, 3, 3)
        K[0, 0, 0, 0] = 400.0
        K[0, 0, 1, 1] = 400.0
        K[0, 0, 0, 2] = W / 2
        K[0, 0, 1, 2] = H / 2
        K[0, 0, 2, 2] = 1.0
        translation = torch.zeros(1, 1, 3)
        rotation = _identity_quaternion(1, 1)
        uv_norm, valid = proj.project(points, K, translation, rotation)
        # Both points project to principal point → normalised coords = (0, 0)
        for i in range(2):
            if valid[0, 0, i]:
                assert abs(uv_norm[0, 0, i, 0].item()) <= 1.0
                assert abs(uv_norm[0, 0, i, 1].item()) <= 1.0

    def test_translation_shifts_projection(self):
        """Non-zero translation should shift the projected point."""
        H, W = 600, 800
        proj = CameraProjector(image_height=H, image_width=W)
        points = torch.tensor([[0.0, 0.0, 10.0]])
        K = torch.zeros(1, 1, 3, 3)
        K[0, 0, 0, 0] = 400.0
        K[0, 0, 1, 1] = 400.0
        K[0, 0, 0, 2] = W / 2
        K[0, 0, 1, 2] = H / 2
        K[0, 0, 2, 2] = 1.0
        # No translation
        t0 = torch.zeros(1, 1, 3)
        q = _identity_quaternion(1, 1)
        uv0, _ = proj.project(points, K, t0, q)
        # Translate sensor 1m to the right (ego y=1)
        t1 = torch.zeros(1, 1, 3)
        t1[0, 0, 1] = 1.0
        uv1, _ = proj.project(points, K, t1, q)
        # The two projected coordinates should differ
        assert not torch.allclose(uv0, uv1)

    def test_single_camera_single_batch(self):
        proj = CameraProjector()
        N, B, C = 3, 1, 1
        points = torch.randn(N, 3)
        points[:, 2] = 5.0  # Ensure positive z
        intrinsics = _identity_intrinsics(B, C)
        translation = torch.zeros(B, C, 3)
        rotation = _identity_quaternion(B, C)
        uv_norm, valid = proj.project(points, intrinsics, translation, rotation)
        assert uv_norm.shape == (1, 1, N, 2)
        assert valid.shape == (1, 1, N)
