"""Regression test: DummyDatasetGenerator → Vision3DDataset → BEVFormerModel training.

Validates the full end-to-end pipeline on disk-generated data:
  1. A synthetic dataset is generated using the DummyDatasetGenerator tool helper.
  2. Vision3DDataset loads the generated frames and annotations.
  3. Vision3DLightningModule trains on that data via pl.Trainer.
  4. The classification loss decreases over training epochs, confirming the
     model actually learns from the generated data (regression performance gate).
  5. Validation metrics (mAP, NDS, …) are finite and in expected ranges after training.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tests.integration.helpers import make_small_model
from tools.generate_dummy_dataset import DummyDatasetGenerator
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.data.dataset import Vision3DDataset
from vision3d.engine.lit_module import Vision3DLightningModule

# ---------------------------------------------------------------------------
# Test-wide constants – kept small for fast CPU execution
# ---------------------------------------------------------------------------

_IMAGE_SIZE: tuple[int, int] = (16, 16)
_NUM_CLASSES: int = 5
_NUM_TRAIN_FRAMES: int = 4
_NUM_VAL_FRAMES: int = 4
_BATCH_SIZE: int = 2
_MAX_BOXES_PER_FRAME: int = 3
_SEED: int = 42

# Metric keys logged by Vision3DLightningModule.training_step via log_dict()
# (loss_dict keys from DetectionLoss, appended with '_epoch' by Lightning)
_METRIC_LOSS_CLS: str = "loss_cls_epoch"
_METRIC_LOSS_BBOX: str = "loss_bbox_epoch"
_METRIC_LOSS_GIOU: str = "loss_giou_epoch"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lit_module(num_classes: int = _NUM_CLASSES, max_epochs: int = 10) -> Vision3DLightningModule:
    """Return a small Vision3DLightningModule for fast regression tests.

    *max_epochs* is forwarded to :class:`Vision3DLightningModule` so that the
    cosine-annealing learning-rate scheduler in ``configure_optimizers`` uses the
    correct ``T_max``.  It must match the ``max_epochs`` passed to the Trainer.
    """
    model = make_small_model(num_classes=num_classes)
    return Vision3DLightningModule(
        model=model,
        matcher=HungarianMatcher(),
        loss=DetectionLoss(num_classes=num_classes),
        evaluator=Vision3DEvaluator(num_classes=num_classes),
        learning_rate=1e-3,
        max_epochs=max_epochs,
    )


def _generate_dataset(root: Path, seed: int = _SEED) -> None:
    """Write train and val splits to *root* using DummyDatasetGenerator."""
    gen = DummyDatasetGenerator(
        output_root=str(root),
        num_frames=_NUM_TRAIN_FRAMES,
        num_cameras=1,
        image_height=_IMAGE_SIZE[0],
        image_width=_IMAGE_SIZE[1],
        max_boxes_per_frame=_MAX_BOXES_PER_FRAME,
        seed=seed,
    )
    gen.generate(split="train")
    # Re-seed so that val frames differ from train frames
    gen2 = DummyDatasetGenerator(
        output_root=str(root),
        num_frames=_NUM_VAL_FRAMES,
        num_cameras=1,
        image_height=_IMAGE_SIZE[0],
        image_width=_IMAGE_SIZE[1],
        max_boxes_per_frame=_MAX_BOXES_PER_FRAME,
        seed=seed + 100,
    )
    gen2.generate(split="val")


def _make_loaders(root: Path) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) backed by on-disk generated data."""
    train_ds = Vision3DDataset(str(root), split="train", image_size=_IMAGE_SIZE)
    val_ds = Vision3DDataset(str(root), split="val", image_size=_IMAGE_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=Vision3DDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=Vision3DDataset.collate_fn,
    )
    return train_loader, val_loader


class _LossTracker(pl.Callback):
    """Callback that collects per-epoch classification loss from logged metrics.

    Reads ``_METRIC_LOSS_CLS`` (``"loss_cls_epoch"``) which is logged by
    :meth:`Vision3DLightningModule.training_step` via ``log_dict`` with
    ``on_epoch=True``.  If the key is absent (e.g., the metric name changes)
    ``cls_losses`` will remain empty and the caller's assertion will surface
    the problem clearly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cls_losses: list[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logs = trainer.logged_metrics
        if _METRIC_LOSS_CLS in logs:
            self.cls_losses.append(float(logs[_METRIC_LOSS_CLS]))


def _make_trainer(max_epochs: int, extra_callbacks: list[pl.Callback] | None = None) -> pl.Trainer:
    """Return a minimal CPU Trainer suitable for regression tests."""
    callbacks: list[pl.Callback] = extra_callbacks or []
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegressionEndToEndTraining:
    """Regression tests: generate dataset → train BEVFormerModel → verify performance."""

    def test_generated_dataset_loads_correctly(self, tmp_path: Path) -> None:
        """Vision3DDataset must load all frames generated by DummyDatasetGenerator."""
        _generate_dataset(tmp_path)
        train_ds = Vision3DDataset(str(tmp_path), split="train", image_size=_IMAGE_SIZE)
        val_ds = Vision3DDataset(str(tmp_path), split="val", image_size=_IMAGE_SIZE)

        assert len(train_ds) == _NUM_TRAIN_FRAMES
        assert len(val_ds) == _NUM_VAL_FRAMES

        frame = train_ds[0]
        assert frame.cameras, "Loaded frame must have at least one camera view"
        assert frame.targets is not None, "Loaded frame must carry annotation targets"
        assert frame.targets.boxes.ndim == 2
        assert frame.targets.boxes.shape[1] == 10

    def test_training_on_generated_dataset_completes_with_finite_losses(
        self, tmp_path: Path
    ) -> None:
        """Training via pl.Trainer on generated data must produce only finite losses."""
        _generate_dataset(tmp_path)
        train_loader, val_loader = _make_loaders(tmp_path)

        pl.seed_everything(_SEED, workers=True)
        lit = _make_lit_module(max_epochs=3)
        trainer = _make_trainer(max_epochs=3)
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

        for key in (_METRIC_LOSS_CLS, _METRIC_LOSS_BBOX, _METRIC_LOSS_GIOU):
            assert key in trainer.logged_metrics, f"Expected metric '{key}' in logged_metrics"
            value = float(trainer.logged_metrics[key])
            assert torch.isfinite(torch.tensor(value)), f"Non-finite loss: {key}={value}"

    def test_classification_loss_decreases_over_training(self, tmp_path: Path) -> None:
        """Classification loss must decrease over 10 training epochs on a fixed small dataset.

        This is the core regression performance gate: the model must demonstrably
        learn from the generated data rather than remaining at the initial random
        performance level.  A ≥ 5 % drop in loss_cls from epoch 1 to epoch 10
        is the threshold; with a fixed seed this is deterministic.
        """
        _generate_dataset(tmp_path)
        train_loader, val_loader = _make_loaders(tmp_path)

        pl.seed_everything(_SEED, workers=True)
        loss_tracker = _LossTracker()
        lit = _make_lit_module(max_epochs=10)
        trainer = _make_trainer(max_epochs=10, extra_callbacks=[loss_tracker])
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

        assert len(loss_tracker.cls_losses) == 10, (
            f"Expected 10 epoch loss values, got {len(loss_tracker.cls_losses)}"
        )

        initial_cls_loss = loss_tracker.cls_losses[0]
        final_cls_loss = loss_tracker.cls_losses[-1]

        assert final_cls_loss < initial_cls_loss * 0.95, (
            f"Classification loss did not decrease by ≥5 % over 10 epochs: "
            f"initial={initial_cls_loss:.4f}, final={final_cls_loss:.4f}"
        )

    def test_validation_metrics_are_finite_after_training(self, tmp_path: Path) -> None:
        """mAP and NDS computed after training must be finite scalars in [0, 1]."""
        _generate_dataset(tmp_path)
        train_loader, val_loader = _make_loaders(tmp_path)

        pl.seed_everything(_SEED, workers=True)
        lit = _make_lit_module(max_epochs=3)
        trainer = _make_trainer(max_epochs=3)
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

        for metric_key in ("val/mAP", "val/NDS"):
            assert metric_key in trainer.logged_metrics, (
                f"Expected validation metric '{metric_key}' in logged_metrics"
            )
            value = float(trainer.logged_metrics[metric_key])
            assert torch.isfinite(torch.tensor(value)), f"Non-finite metric: {metric_key}={value}"
            assert 0.0 <= value <= 1.0, f"Metric {metric_key}={value} out of [0, 1]"
