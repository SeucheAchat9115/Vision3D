"""
Dummy dataset generator for Vision3D.

Provides `DummyDatasetGenerator`, which creates a fully self-contained synthetic
dataset in the Vision3D generic format (random PNGs + JSONs). This is used for:

  - Testing that the dataloader and model work end-to-end without real data.
  - Continuous integration smoke tests that must run without an internet connection.
  - Rapid prototyping and debugging of new model components.

Usage:
    python scripts/generate_dummy_dataset.py \\
        --output_root  /data/dummy \\
        --num_frames   50 \\
        --num_cameras  6 \\
        --image_height 900 \\
        --image_width  1600
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class DummyDatasetGenerator:
    """Generates a synthetic Vision3D dataset with random images and annotations.

    Each generated frame consists of:
      - `num_cameras` random PNG images (uniform random pixel values).
      - A JSON file matching the Vision3D generic schema (Section 6 of README),
        with randomly generated bounding boxes, random calibration matrices,
        and random metadata fields.

    The generator also produces past-frame references so that temporal attention
    in the BEVEncoder can be exercised without requiring a real dataset.

    Args:
        output_root: Root directory where the dataset will be written.
        num_frames: Total number of frames to generate.
        num_cameras: Number of camera views per frame (default: 6).
        image_height: Height of generated images in pixels.
        image_width: Width of generated images in pixels.
        max_boxes_per_frame: Maximum number of GT boxes to generate per frame.
        num_past_frames: Number of past frame references to include per frame.
        class_names: List of class names for randomly generated annotations.
        seed: Optional random seed for reproducible generation.
    """

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
        class_names: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        # TODO: store all constructor arguments as instance attributes
        # TODO: set numpy / random seed if provided
        # TODO: create output directory structure (images/<camera>/, train/, val/)
        # TODO: default class_names to a standard list if None
        raise NotImplementedError

    def generate(self, split: str = "train") -> None:
        """Generate the full dataset for the given split.

        Args:
            split: Dataset split label ("train", "val", or "test"). Used as
                the subdirectory name for JSON files.
        """
        # TODO: generate a list of unique frame_ids (UUIDs or sequential strings)
        # TODO: for each frame_id call self._generate_frame(frame_id, split)
        # TODO: print a progress message every N frames
        raise NotImplementedError

    def _generate_frame(
        self,
        frame_id: str,
        split: str,
        past_frame_ids: List[str],
    ) -> None:
        """Generate and write the image files and JSON for a single frame.

        Args:
            frame_id: Unique identifier for this frame.
            split: Dataset split (used to determine JSON output path).
            past_frame_ids: List of preceding frame IDs to embed in the JSON.
        """
        # TODO: generate and save random PNG images for each camera
        #         use numpy random uint8 arrays converted to PNG via PIL
        # TODO: generate random camera calibration (intrinsics + extrinsics)
        # TODO: generate 0..max_boxes_per_frame random bounding boxes
        #         boxes in 10-DOF format [x,y,z,w,l,h,sin(t),cos(t),vx,vy]
        #         with physically plausible ranges
        # TODO: generate random metadata (weather, time_of_day, occlusion)
        # TODO: assemble the JSON dict matching the schema in README Section 6
        # TODO: write the JSON file to output_root/split/frame_id.json
        raise NotImplementedError

    def _generate_random_image(
        self,
        camera_name: str,
        frame_id: str,
    ) -> str:
        """Generate a random RGB image and save it as a PNG.

        Args:
            camera_name: Camera name used to determine the output subdirectory.
            frame_id: Frame identifier used as the PNG filename (without ext).

        Returns:
            Relative path to the saved PNG file (relative to output_root),
            which is the value stored in the JSON "image_path" field.
        """
        # TODO: generate a (image_height, image_width, 3) uint8 numpy array with
        #       uniform random values
        # TODO: save as PNG to output_root/images/<camera_name>/<frame_id>.png
        # TODO: return the relative path string
        raise NotImplementedError

    def _generate_random_intrinsics(self) -> List[List[float]]:
        """Generate a plausible random camera intrinsic matrix.

        Returns:
            3×3 intrinsic matrix as a nested list of floats with:
              - fx, fy ∈ [600, 1200] (focal lengths)
              - cx ≈ image_width / 2 ± 50 (principal point x)
              - cy ≈ image_height / 2 ± 50 (principal point y)
        """
        # TODO: sample fx, fy uniformly from [600, 1200]
        # TODO: sample cx, cy near the image centre
        # TODO: return [[fx,0,cx],[0,fy,cy],[0,0,1]]
        raise NotImplementedError

    def _generate_random_extrinsics(
        self,
    ) -> Tuple[List[float], List[float]]:
        """Generate a random sensor-to-ego translation and rotation quaternion.

        Returns:
            Tuple of (translation [x, y, z], rotation [w, x, y, z]).
        """
        # TODO: sample translation near the ego vehicle roof rack (x ∈ [-1,1], y ∈ [-1,1], z ∈ [1,2])
        # TODO: sample a random unit quaternion (normalise a 4-D uniform random vector)
        # TODO: return (translation, rotation)
        raise NotImplementedError

    def _generate_random_boxes(self, num_boxes: int) -> List[Dict]:
        """Generate a list of random GT annotation dicts.

        Args:
            num_boxes: Number of boxes to generate.

        Returns:
            List of annotation dicts matching the JSON schema:
            [{"instance_id": ..., "class_name": ..., "bbox_3d": [10 floats]}, ...]
        """
        # TODO: for each box:
        #         instance_id = random UUID string
        #         class_name = random.choice(self.class_names)
        #         x, y ∈ [-50, 50], z ∈ [-1, 2]
        #         w, l ∈ [1.5, 5.0], h ∈ [1.0, 4.0]
        #         theta = random angle → (sin(theta), cos(theta))
        #         vx, vy ∈ [-10, 10]
        # TODO: return the list of annotation dicts
        raise NotImplementedError


def main() -> None:
    """CLI entry point for dummy dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Vision3D dummy dataset."
    )
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
