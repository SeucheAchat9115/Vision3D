"""
NuScenes → Vision3D dataset converter.

Provides `NuScenesConverter`, a standalone utility that parses a NuScenes
dataset (via the nuscenes-devkit) and converts it to the Vision3D generic
format.

Usage:
    python tools/convert_nuscenes.py \\
        --nuscenes_root /data/nuscenes \\
        --output_root  /data/vision3d_nuscenes \\
        --version      v1.0-trainval \\
        --split        train
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    NuScenes = None


class NuScenesConverter:
    """Converts a NuScenes split to the Vision3D generic JSON + PNG format."""

    def __init__(
        self,
        nuscenes_root: str,
        output_root: str,
        version: str = "v1.0-trainval",
        classes: list[str] | None = None,
        num_past_frames: int = 2,
    ) -> None:
        self.nuscenes_root = nuscenes_root
        self.output_root = Path(output_root)
        self.version = version
        self.num_past_frames = num_past_frames
        if classes is None:
            classes = [
                "car",
                "truck",
                "bus",
                "pedestrian",
                "motorcycle",
                "bicycle",
                "trailer",
                "construction_vehicle",
                "traffic_cone",
                "barrier",
            ]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        if NuScenes is None:
            raise ImportError(
                "nuscenes-devkit is required. Install with: pip install nuscenes-devkit"
            )
        self.nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
        (self.output_root / "images").mkdir(parents=True, exist_ok=True)
        (self.output_root / "train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "val").mkdir(parents=True, exist_ok=True)

    def convert(self, split: str = "train") -> None:
        """Convert all samples in the given split to the Vision3D format."""
        split_file = Path(self.nuscenes_root) / self.version / f"v1.0-{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                split_scenes: set[str] = set(f.read().splitlines())
        else:
            split_scenes = {s["name"] for s in self.nusc.scene}
        count = 0
        for scene in self.nusc.scene:
            if scene["name"] not in split_scenes:
                continue
            sample_token = scene["first_sample_token"]
            while sample_token:
                self._convert_sample(sample_token, split)
                count += 1
                if count % 100 == 0:
                    print(f"Converted {count} samples...")
                sample = self.nusc.get("sample", sample_token)
                sample_token = sample["next"]

    def _convert_sample(self, sample_token: str, split: str) -> None:
        """Convert a single NuScenes sample to Vision3D format."""

        sample = self.nusc.get("sample", sample_token)
        first_sd_token = list(sample["data"].values())[0]
        ego_pose = self.nusc.get(
            "ego_pose", self.nusc.get("sample_data", first_sd_token)["ego_pose_token"]
        )
        cameras_data: dict[str, Any] = {}
        camera_channels = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        for ch in camera_channels:
            if ch not in sample["data"]:
                continue
            sd = self.nusc.get("sample_data", sample["data"][ch])
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            img_path = str(Path(self.nuscenes_root) / sd["filename"])
            cam_name = ch.lower()
            out_img_dir = self.output_root / "images" / cam_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            try:
                undistorted = self._undistort_image(
                    img_path,
                    {"camera_matrix": cs["camera_intrinsic"]},
                    cs.get("camera_distortion", [0, 0, 0, 0, 0]),
                )
                import cv2

                out_img_path = out_img_dir / f"{sample_token}.png"
                cv2.imwrite(str(out_img_path), undistorted)
                rel_path = f"images/{cam_name}/{sample_token}.png"
            except Exception:
                rel_path = sd["filename"]
            cameras_data[cam_name] = {
                "image_path": rel_path,
                "intrinsics": cs["camera_intrinsic"],
                "sensor2ego_translation": cs["translation"],
                "sensor2ego_rotation": cs["rotation"],
            }
        annotations = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"].split(".")[0]
            if cat not in self.class_to_idx:
                continue
            vel: list[float]
            if hasattr(self.nusc, "box_velocity"):
                vel = self.nusc.box_velocity(ann_token)[:2].tolist()
            else:
                vel = [0.0, 0.0]
            bbox = self._box_global_to_ego(
                ann["translation"],
                ann["size"],
                ann["rotation"],
                vel,
                ego_pose["translation"],
                ego_pose["rotation"],
            )
            annotations.append(
                {
                    "instance_id": ann["instance_token"],
                    "class_name": cat,
                    "bbox_3d": bbox,
                }
            )
        past_frame_ids: list[str] = []
        prev_token = sample["prev"]
        for _ in range(self.num_past_frames):
            if not prev_token:
                break
            past_frame_ids.append(prev_token)
            prev_token = self.nusc.get("sample", prev_token)["prev"]
        frame_data = {
            "frame_id": sample_token,
            "timestamp": sample["timestamp"] / 1e6,
            "past_frame_ids": past_frame_ids,
            "cameras": cameras_data,
            "annotations": annotations,
            "metadata": {"scene_token": sample["scene_token"]},
        }
        out_path = self.output_root / split / f"{sample_token}.json"
        with open(out_path, "w") as f:
            json.dump(frame_data, f)

    def _undistort_image(
        self,
        image_path: str,
        camera_intrinsics: dict[str, Any],
        distortion_coeffs: list[float],
    ) -> Any:
        """Load and undistort an image using Brown-Conrady distortion model."""
        import cv2
        import numpy as np

        img = cv2.imread(image_path)
        K = np.array(camera_intrinsics["camera_matrix"], dtype=np.float64)
        D = np.array(distortion_coeffs, dtype=np.float64)
        return cv2.undistort(img, K, D)

    def _box_global_to_ego(
        self,
        box_center_global: list[float],
        box_size: list[float],
        box_rotation_quat: list[float],
        box_velocity: list[float],
        ego_translation: list[float],
        ego_rotation_quat: list[float],
    ) -> list[float]:
        """Transform a single box from global to ego-centric coordinates."""
        import numpy as np

        w, x, y, z = ego_rotation_quat
        R_ego = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        c_global = np.array(box_center_global)
        t_ego = np.array(ego_translation)
        c_ego = R_ego.T @ (c_global - t_ego)
        bw, bx, by, bz = box_rotation_quat
        yaw_global = math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz))
        ew, ex, ey, ez = ego_rotation_quat
        yaw_ego_pose = math.atan2(2 * (ew * ez + ex * ey), 1 - 2 * (ey * ey + ez * ez))
        yaw = yaw_global - yaw_ego_pose
        vel_3d = np.array(list(box_velocity[:2]) + [0.0])
        vel_ego = R_ego.T @ vel_3d
        w_b, l_b, h_b = box_size
        return [
            float(c_ego[0]),
            float(c_ego[1]),
            float(c_ego[2]),
            float(w_b),
            float(l_b),
            float(h_b),
            float(math.sin(yaw)),
            float(math.cos(yaw)),
            float(vel_ego[0]),
            float(vel_ego[1]),
        ]


def main() -> None:
    """CLI entry point for the NuScenes conversion script."""
    parser = argparse.ArgumentParser(description="Convert NuScenes to Vision3D generic format.")
    parser.add_argument("--nuscenes_root", required=True, help="Path to raw NuScenes data")
    parser.add_argument("--output_root", required=True, help="Output directory for Vision3D data")
    parser.add_argument("--version", default="v1.0-trainval", help="NuScenes dataset version")
    parser.add_argument(
        "--split", default="train", choices=["train", "val"], help="Split to convert"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "car",
            "truck",
            "bus",
            "pedestrian",
            "motorcycle",
            "bicycle",
            "trailer",
            "construction_vehicle",
            "traffic_cone",
            "barrier",
        ],
        help="NuScenes category names to include",
    )
    args = parser.parse_args()

    converter = NuScenesConverter(
        nuscenes_root=args.nuscenes_root,
        output_root=args.output_root,
        version=args.version,
        classes=args.classes,
    )
    converter.convert(split=args.split)


if __name__ == "__main__":
    main()
