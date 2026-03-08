"""
Dummy dataset generator for Vision3D.

Provides `DummyDatasetGenerator`, which creates a fully self-contained synthetic
dataset in the Vision3D generic format (random PNGs + JSONs).

Usage:
    python tools/generate_dummy_dataset.py \\
        --output_root  /data/dummy \\
        --num_frames   50 \\
        --num_cameras  6 \\
        --image_height 900 \\
        --image_width  1600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage


class DummyDatasetGenerator:
    """Generates a synthetic Vision3D dataset with random images and annotations."""

    CAMERA_NAMES = ["front", "front_left", "front_right", "back", "back_left", "back_right"]

    def __init__(
        self,
        output_root: str,
        num_frames: int = 50,
        num_cameras: int = 6,
        image_height: int = 900,
        image_width: int = 1600,
        max_boxes_per_frame: int = 20,
        num_past_frames: int = 2,
        class_names: list[str] | None = None,
        seed: int | None = 42,
    ) -> None:
        self.output_root = Path(output_root)
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.max_boxes_per_frame = max_boxes_per_frame
        self.num_past_frames = num_past_frames
        self.class_names = (
            class_names
            if class_names is not None
            else [
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
        )
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.camera_names = self.CAMERA_NAMES[:num_cameras]
        for cam in self.camera_names:
            (self.output_root / "images" / cam).mkdir(parents=True, exist_ok=True)
        (self.output_root / "train").mkdir(parents=True, exist_ok=True)
        (self.output_root / "val").mkdir(parents=True, exist_ok=True)

    def generate(self, split: str = "train") -> None:
        """Generate the full dataset for the given split."""
        frame_ids = [str(uuid.uuid4()) for _ in range(self.num_frames)]
        for i, frame_id in enumerate(frame_ids):
            past_ids = frame_ids[max(0, i - self.num_past_frames) : i]
            self._generate_frame(frame_id, split, past_ids)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{self.num_frames} frames")

    def _generate_frame(
        self,
        frame_id: str,
        split: str,
        past_frame_ids: list[str],
    ) -> None:
        """Generate and write the image files and JSON for a single frame."""
        cameras: dict[str, dict[str, Any]] = {}
        for cam_name in self.camera_names:
            img_path = self._generate_random_image(cam_name, frame_id)
            intrinsics = self._generate_random_intrinsics()
            translation, rotation = self._generate_random_extrinsics()
            cameras[cam_name] = {
                "image_path": img_path,
                "intrinsics": intrinsics,
                "sensor2ego_translation": translation,
                "sensor2ego_rotation": rotation,
            }
        num_boxes = random.randint(0, self.max_boxes_per_frame)
        annotations = self._generate_random_boxes(num_boxes)
        frame_data = {
            "frame_id": frame_id,
            "timestamp": random.uniform(0, 1e9),
            "past_frame_ids": past_frame_ids,
            "cameras": cameras,
            "annotations": annotations,
            "metadata": {
                "weather": random.choice(["clear", "cloudy", "rain"]),
                "time_of_day": random.choice(["day", "night"]),
            },
        }
        out_path = self.output_root / split / f"{frame_id}.json"
        with open(out_path, "w") as f:
            json.dump(frame_data, f)

    def _generate_random_image(self, camera_name: str, frame_id: str) -> str:
        """Generate a random RGB image and save it as a PNG."""
        arr = np.random.randint(0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8)
        img = PILImage.fromarray(arr)
        rel_path = f"images/{camera_name}/{frame_id}.png"
        out_path = self.output_root / rel_path
        img.save(out_path)
        return rel_path

    def _generate_random_intrinsics(self) -> list[list[float]]:
        """Generate a plausible random camera intrinsic matrix."""
        fx = random.uniform(600, 1200)
        fy = random.uniform(600, 1200)
        cx = self.image_width / 2 + random.uniform(-50, 50)
        cy = self.image_height / 2 + random.uniform(-50, 50)
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    def _generate_random_extrinsics(self) -> tuple[list[float], list[float]]:
        """Generate a random sensor-to-ego translation and rotation quaternion."""
        translation = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(1, 2),
        ]
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        rotation = q.tolist()
        return translation, rotation

    def _generate_random_boxes(self, num_boxes: int) -> list[dict[str, Any]]:
        """Generate a list of random GT annotation dicts."""
        boxes = []
        for _ in range(num_boxes):
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            z = random.uniform(-1, 2)
            w = random.uniform(1.5, 5.0)
            box_length = random.uniform(1.5, 5.0)
            h = random.uniform(1.0, 4.0)
            theta = random.uniform(-math.pi, math.pi)
            vx = random.uniform(-10, 10)
            vy = random.uniform(-10, 10)
            boxes.append(
                {
                    "instance_id": str(uuid.uuid4()),
                    "class_name": random.choice(self.class_names),
                    "bbox_3d": [
                        x,
                        y,
                        z,
                        w,
                        box_length,
                        h,
                        math.sin(theta),
                        math.cos(theta),
                        vx,
                        vy,
                    ],
                }
            )
        return boxes


def main() -> None:
    """CLI entry point for dummy dataset generation."""
    parser = argparse.ArgumentParser(description="Generate a synthetic Vision3D dummy dataset.")
    parser.add_argument("--output_root", required=True, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=50, help="Total frames to generate")
    parser.add_argument("--num_cameras", type=int, default=6, help="Cameras per frame")
    parser.add_argument("--image_height", type=int, default=900, help="Image height in pixels")
    parser.add_argument("--image_width", type=int, default=1600, help="Image width in pixels")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = DummyDatasetGenerator(
        output_root=args.output_root,
        num_frames=args.num_frames,
        num_cameras=args.num_cameras,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
    )
    generator.generate(split=args.split)


if __name__ == "__main__":
    main()
