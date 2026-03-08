"""
PyTorch Lightning training module for Vision3D.

Provides `Vision3DLightningModule`, which acts as a thin container around the
`BEVFormerModel`. All training logic (optimiser, lr schedule, loss, evaluation)
lives here, while the model itself remains a pure `nn.Module`.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from vision3d.config.schema import BatchData, BoundingBox3DPrediction
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.models.bevformer import BEVFormerModel


class Vision3DLightningModule(pl.LightningModule):
    """PyTorch Lightning container for the BEVFormer training pipeline."""

    def __init__(
        self,
        model: BEVFormerModel,
        matcher: HungarianMatcher,
        loss: DetectionLoss,
        evaluator: Vision3DEvaluator,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 24,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "matcher", "loss", "evaluator"])
        self.model = model
        self.matcher = matcher
        self.loss = loss
        self.evaluator = evaluator
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self._prev_bev: torch.Tensor | None = None

    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        batch = batch.to(self.device)
        predictions, new_bev = self.model(batch, prev_bev=self._prev_bev)
        self._prev_bev = new_bev.detach()
        targets = [f.targets for f in batch.frames if f.targets is not None]
        pred_list = [
            BoundingBox3DPrediction(
                boxes=predictions.boxes[i],
                scores=predictions.scores[i],
                labels=predictions.labels[i],
            )
            for i in range(batch.batch_size)
        ]
        matches = self.matcher.match_batch(pred_list, targets)
        total_loss, loss_dict = self.loss(pred_list, targets, matches)
        log_dict = {k: v.detach() for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        result: torch.Tensor = total_loss
        return result

    def on_validation_epoch_start(self) -> None:
        """Reset the evaluator and clear temporal BEV state at epoch start."""
        self.evaluator.reset()
        self._prev_bev = None

    def validation_step(self, batch: BatchData, batch_idx: int) -> None:
        """Run inference and accumulate predictions for epoch-level evaluation."""
        batch = batch.to(self.device)
        with torch.no_grad():
            predictions, new_bev = self.model(batch, prev_bev=self._prev_bev)
        self._prev_bev = new_bev.detach()
        pred_list = [
            BoundingBox3DPrediction(
                boxes=predictions.boxes[i],
                scores=predictions.scores[i],
                labels=predictions.labels[i],
            )
            for i in range(batch.batch_size)
        ]
        targets = [f.targets for f in batch.frames if f.targets is not None]
        self.evaluator.update(pred_list, targets)

    def on_validation_epoch_end(self) -> None:
        """Compute and log all evaluation metrics at the end of validation."""
        metrics = self.evaluator.compute()
        self.log_dict({"val/" + k: v for k, v in metrics.items()}, on_epoch=True)

    def configure_optimizers(self) -> Any:
        """Configure AdamW optimiser with a cosine annealing LR schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
