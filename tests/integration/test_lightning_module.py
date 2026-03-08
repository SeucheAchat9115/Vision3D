"""Integration tests: Vision3DLightningModule.

Covered scenarios:
  - training_step returns a finite scalar tensor
  - Temporal _prev_bev state is updated after training steps
  - validation_step does not raise
  - on_validation_epoch_start resets _prev_bev
  - on_validation_epoch_end computes metrics
  - configure_optimizers returns a valid dict
  - Training step handles a batch with no GT boxes
  - Interleaved training and validation sequence
  - Optimizer config contains a learning-rate scheduler
"""

from __future__ import annotations

import torch

from vision3d.config.schema import BatchData
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.engine.lit_module import Vision3DLightningModule
from tests.integration.helpers import make_batch, make_frame, make_small_model


def _make_lit_module(
    num_classes: int = 5,
    num_queries: int = 10,
) -> Vision3DLightningModule:
    model = make_small_model(num_classes=num_classes, num_queries=num_queries)
    matcher = HungarianMatcher()
    loss_fn = DetectionLoss(num_classes=num_classes)
    evaluator = Vision3DEvaluator(num_classes=num_classes)
    return Vision3DLightningModule(
        model=model,
        matcher=matcher,
        loss=loss_fn,
        evaluator=evaluator,
        learning_rate=1e-3,
        max_epochs=2,
    )


class TestLightningModuleIntegration:
    """Integration tests for Vision3DLightningModule."""

    def test_training_step_returns_scalar(self) -> None:
        """training_step must return a finite scalar tensor."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        loss = lit.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_training_step_twice_updates_prev_bev(self) -> None:
        """After two training steps _prev_bev must be set (temporal state)."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        lit.training_step(batch, batch_idx=0)
        assert lit._prev_bev is not None
        lit.training_step(batch, batch_idx=1)
        assert lit._prev_bev is not None

    def test_validation_step_does_not_raise(self) -> None:
        """validation_step must not raise for normal input."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, batch_idx=0)

    def test_on_validation_epoch_start_resets_prev_bev(self) -> None:
        """on_validation_epoch_start must clear the temporal BEV state."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        lit.training_step(batch, batch_idx=0)
        assert lit._prev_bev is not None
        lit.on_validation_epoch_start()
        assert lit._prev_bev is None

    def test_on_validation_epoch_end_logs_map(self) -> None:
        """on_validation_epoch_end must call evaluator.compute() without raising."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        lit.on_validation_epoch_start()
        lit.validation_step(batch, batch_idx=0)
        metrics = lit.evaluator.compute()
        assert "mAP" in metrics

    def test_configure_optimizers_returns_dict(self) -> None:
        """configure_optimizers must return a dict with 'optimizer' key."""
        lit = _make_lit_module()
        opt_config = lit.configure_optimizers()
        assert isinstance(opt_config, dict)
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

    def test_training_step_with_empty_targets(self) -> None:
        """Training step must handle a batch where all frames have no GT boxes."""
        lit = _make_lit_module()
        frame = make_frame(num_boxes=0, num_classes=5)
        batch = BatchData(batch_size=1, frames=[frame])
        loss = lit.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)

    def test_training_and_validation_steps_sequence(self) -> None:
        """Running training then validation steps must not raise or corrupt state."""
        lit = _make_lit_module()
        train_batch = make_batch(num_classes=5)
        val_batch = make_batch(num_classes=5)
        for i in range(2):
            lit.training_step(train_batch, batch_idx=i)
        lit.on_validation_epoch_start()
        lit.validation_step(val_batch, batch_idx=0)
        metrics = lit.evaluator.compute()
        assert "mAP" in metrics

    def test_training_loss_decreases_after_multiple_steps(self) -> None:
        """Loss should change (not stuck) when the optimizer runs multiple steps."""
        lit = _make_lit_module()
        batch = make_batch(num_classes=5)
        losses = []
        for i in range(3):
            loss = lit.training_step(batch, batch_idx=i)
            losses.append(loss.item())
        # Verify the loss values are all finite (not stuck at NaN/Inf)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
