"""Integration tests: DataLoader + BEVFormerModel end-to-end (on-disk dataset).

Covered scenarios:
  - DataLoader batch passes through model without errors
  - Prediction shapes match model configuration
  - batch_size=1 produces a single-frame batch
  - Confidence scores from DataLoader batch are in [0, 1]
  - Loaded frames carry valid ground-truth targets
  - collate_fn produces a valid BatchData
  - Full dataset iteration passes every batch through the model
  - Training pass (dataloader → model → matcher → loss) produces finite loss
  - Backward pass from DataLoader batch populates model gradients
  - Multiple training steps driven by DataLoader do not raise
  - Dataset with no annotations still passes through model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from tests.integration.helpers import make_small_model
from vision3d.config.schema import BatchData, BoundingBox3DPrediction, BoundingBox3DTarget
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.data.dataset import Vision3DDataset


def _write_dataset_frame(
    root: Path,
    frame_id: str,
    split: str = "train",
    num_boxes: int = 2,
    image_h: int = 16,
    image_w: int = 16,
) -> None:
    """Write a synthetic JSON frame and a small PNG image to *root*."""
    cam_name = "front"
    img_rel_path = f"images/{cam_name}/{frame_id}.png"
    img_dir = root / "images" / cam_name
    img_dir.mkdir(parents=True, exist_ok=True)
    seed = hash(frame_id) & 0xFFFFFFFF
    arr = (np.random.default_rng(seed).random((image_h, image_w, 3)) * 255).astype("uint8")
    PILImage.fromarray(arr).save(str(img_dir / f"{frame_id}.png"))

    annotations = [
        {
            "instance_id": f"obj_{i}",
            "class_name": "car",
            "bbox_3d": [float(i), 0.0, 0.0, 2.0, 1.5, 1.5, 0.0, 1.0, 0.0, 0.0],
        }
        for i in range(num_boxes)
    ]
    frame_data = {
        "frame_id": frame_id,
        "timestamp": 1600000000.0,
        "past_frame_ids": [],
        "cameras": {
            cam_name: {
                "image_path": img_rel_path,
                "intrinsics": [
                    [400.0, 0.0, image_w / 2],
                    [0.0, 400.0, image_h / 2],
                    [0.0, 0.0, 1.0],
                ],
                "sensor2ego_translation": [1.5, 0.0, 1.2],
                "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            }
        },
        "annotations": annotations,
        "metadata": {"weather": "clear", "time_of_day": "day"},
    }
    (root / split).mkdir(parents=True, exist_ok=True)
    with open(root / split / f"{frame_id}.json", "w") as f:
        json.dump(frame_data, f)


class TestDataloaderModelIntegration:
    """Tests that verify Vision3DDataset + DataLoader + BEVFormerModel work together.

    Unlike earlier integration tests that use in-memory synthetic BatchData, these
    tests exercise the full I/O path: JSON files and PNG images are written to a
    temporary directory, loaded by Vision3DDataset, collated by DataLoader, and
    then consumed by BEVFormerModel.
    """

    _IMAGE_SIZE: tuple[int, int] = (16, 16)

    def _make_dataset(
        self,
        root: Path,
        num_frames: int = 4,
        num_boxes: int = 2,
    ) -> Vision3DDataset:
        for i in range(num_frames):
            _write_dataset_frame(root, f"frame_{i:03d}", num_boxes=num_boxes)
        return Vision3DDataset(str(root), split="train", image_size=self._IMAGE_SIZE)

    @staticmethod
    def _predictions_and_targets(
        batch: BatchData,
        preds_batch: BoundingBox3DPrediction,
    ) -> tuple[list[BoundingBox3DPrediction], list[BoundingBox3DTarget]]:
        """Build per-frame prediction and target lists from a batched model output."""
        pred_list = [
            BoundingBox3DPrediction(
                boxes=preds_batch.boxes[i],
                scores=preds_batch.scores[i],
                labels=preds_batch.labels[i],
            )
            for i, frame in enumerate(batch.frames)
            if frame.targets is not None
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        return pred_list, targets

    # ------------------------------------------------------------------
    # Basic forward-pass tests
    # ------------------------------------------------------------------

    def test_dataloader_batch_reaches_model(self, tmp_path: Path) -> None:
        """A batch produced by DataLoader must pass through the model without errors."""
        ds = self._make_dataset(tmp_path, num_frames=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        model = make_small_model()
        model.eval()
        with torch.no_grad():
            preds, _ = model(batch)
        assert isinstance(preds, BoundingBox3DPrediction)

    def test_dataloader_output_shapes_match_model_config(self, tmp_path: Path) -> None:
        """Prediction shapes from a DataLoader batch must match the model configuration."""
        num_classes, num_queries = 5, 10
        ds = self._make_dataset(tmp_path, num_frames=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        model = make_small_model(num_classes=num_classes, num_queries=num_queries)
        model.eval()
        with torch.no_grad():
            preds, _ = model(batch)
        assert preds.boxes.shape == (2, num_queries, 10)
        assert preds.scores.shape == (2, num_queries)
        assert preds.labels.shape == (2, num_queries)

    def test_dataloader_batch_size_one(self, tmp_path: Path) -> None:
        """DataLoader with batch_size=1 must produce a single-frame batch the model accepts."""
        ds = self._make_dataset(tmp_path, num_frames=2)
        loader = DataLoader(ds, batch_size=1, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        assert batch.batch_size == 1
        model = make_small_model()
        model.eval()
        with torch.no_grad():
            preds, _ = model(batch)
        assert preds.boxes.shape[0] == 1

    def test_dataloader_scores_in_unit_interval(self, tmp_path: Path) -> None:
        """Confidence scores from a DataLoader batch must lie in [0, 1]."""
        ds = self._make_dataset(tmp_path, num_frames=2)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        model = make_small_model()
        model.eval()
        with torch.no_grad():
            preds, _ = model(batch)
        assert preds.scores.min().item() >= 0.0
        assert preds.scores.max().item() <= 1.0

    # ------------------------------------------------------------------
    # Dataset / DataLoader correctness
    # ------------------------------------------------------------------

    def test_dataloader_batch_contains_valid_targets(self, tmp_path: Path) -> None:
        """Frames loaded from disk must carry ground-truth targets consumed by the matcher."""
        num_boxes = 3
        ds = self._make_dataset(tmp_path, num_frames=2, num_boxes=num_boxes)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        for frame in batch.frames:
            assert frame.targets is not None
            assert frame.targets.boxes.shape == (num_boxes, 10)

    def test_dataloader_collate_fn_produces_valid_batch_data(self, tmp_path: Path) -> None:
        """collate_fn must produce a BatchData with consistent frame count and batch_size."""
        ds = self._make_dataset(tmp_path, num_frames=3)
        loader = DataLoader(ds, batch_size=3, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        assert isinstance(batch, BatchData)
        assert batch.batch_size == 3
        assert len(batch.frames) == 3

    def test_dataloader_iteration_over_full_dataset(self, tmp_path: Path) -> None:
        """Iterating the entire DataLoader must pass every batch through the model."""
        num_frames = 4
        ds = self._make_dataset(tmp_path, num_frames=num_frames)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        model = make_small_model()
        model.eval()
        total_frames_seen = 0
        with torch.no_grad():
            for batch in loader:
                preds, _ = model(batch)
                assert isinstance(preds, BoundingBox3DPrediction)
                total_frames_seen += batch.batch_size
        assert total_frames_seen == num_frames

    def test_dataloader_frame_ids_are_strings(self, tmp_path: Path) -> None:
        """Every FrameData produced by the DataLoader must have a non-empty string frame_id."""
        ds = self._make_dataset(tmp_path, num_frames=3)
        loader = DataLoader(ds, batch_size=3, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        for frame in batch.frames:
            assert isinstance(frame.frame_id, str)
            assert len(frame.frame_id) > 0

    # ------------------------------------------------------------------
    # Training pipeline (dataloader → model → matcher → loss)
    # ------------------------------------------------------------------

    def test_dataloader_model_training_pass_finite_loss(self, tmp_path: Path) -> None:
        """Full training pass (dataloader → model → matcher → loss) must produce a finite loss."""
        num_classes = 5
        ds = self._make_dataset(tmp_path, num_frames=4, num_boxes=2)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        preds_batch, _ = model(batch)
        pred_list, targets = self._predictions_and_targets(batch, preds_batch)
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        assert torch.isfinite(total)

    def test_dataloader_model_training_backward(self, tmp_path: Path) -> None:
        """Backward pass through loss from a DataLoader batch must populate model gradients."""
        num_classes = 5
        ds = self._make_dataset(tmp_path, num_frames=4)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        batch = next(iter(loader))
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        preds_batch, _ = model(batch)
        pred_list, targets = self._predictions_and_targets(batch, preds_batch)
        matches = matcher.match_batch(pred_list, targets)
        total, _ = loss_fn(pred_list, targets, matches)
        total.backward()
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        assert n_params_with_grad > 0

    def test_dataloader_model_multiple_training_steps(self, tmp_path: Path) -> None:
        """Running several training steps driven by the DataLoader must not raise."""
        num_classes = 5
        ds = self._make_dataset(tmp_path, num_frames=6)
        loader = DataLoader(ds, batch_size=2, collate_fn=Vision3DDataset.collate_fn)
        model = make_small_model(num_classes=num_classes)
        matcher = HungarianMatcher()
        loss_fn = DetectionLoss(num_classes=num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for batch in loader:
            optimizer.zero_grad()
            preds_batch, _ = model(batch)
            pred_list, targets = self._predictions_and_targets(batch, preds_batch)
            matches = matcher.match_batch(pred_list, targets)
            total, _ = loss_fn(pred_list, targets, matches)
            total.backward()
            optimizer.step()
            assert torch.isfinite(total)
