"""
PyTorch Lightning training module for Vision3D.

Provides `Vision3DLightningModule`, which acts as a thin container around the
`BEVFormerModel`. All training logic (optimiser, lr schedule, loss, evaluation)
lives here, while the model itself remains a pure `nn.Module`.

Responsibilities:
  - `training_step`: Forward pass → Hungarian matching → loss computation.
  - `validation_step`: Forward pass → accumulate predictions for the evaluator.
  - `on_validation_epoch_end`: Compute and log metrics via the evaluator.
  - `configure_optimizers`: Return AdamW + cosine LR scheduler.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import pytorch_lightning as pl

from vision3d.config.schema import BatchData
from vision3d.core.evaluators import Vision3DEvaluator
from vision3d.core.losses import DetectionLoss
from vision3d.core.matchers import HungarianMatcher
from vision3d.models.bevformer import BEVFormerModel
from vision3d.models.backbones.resnet import ResNetBackbone
from vision3d.models.encoders.bev_encoder import BEVEncoder
from vision3d.models.heads.detection_head import DetectionHead
from vision3d.models.necks.fpn import FPNNeck


class Vision3DLightningModule(pl.LightningModule):
    """PyTorch Lightning container for the BEVFormer training pipeline.

    Acts strictly as a wiring layer: all sub-components (backbone, neck,
    encoder, head, matcher, loss, evaluator) are injected via the constructor
    and composed into a `BEVFormerModel`. This design allows Hydra to
    recursively instantiate every component from config without any module
    knowing about Lightning.

    Args:
        backbone: Instantiated `ResNetBackbone`.
        neck: Instantiated `FPNNeck`.
        encoder: Instantiated `BEVEncoder`.
        head: Instantiated `DetectionHead`.
        matcher: Instantiated `HungarianMatcher`.
        loss: Instantiated `DetectionLoss`.
        evaluator: Instantiated `Vision3DEvaluator`.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 regularisation coefficient for AdamW.
        max_epochs: Total training epochs, used to configure the cosine LR schedule.
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        neck: FPNNeck,
        encoder: BEVEncoder,
        head: DetectionHead,
        matcher: HungarianMatcher,
        loss: DetectionLoss,
        evaluator: Vision3DEvaluator,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 24,
    ) -> None:
        super().__init__()
        # TODO: call self.save_hyperparameters() for logging and checkpointing
        # TODO: wrap backbone, neck, encoder, head into BEVFormerModel and assign
        #       to self.model
        # TODO: assign matcher, loss, evaluator as attributes
        # TODO: store learning_rate, weight_decay, max_epochs
        # TODO: initialise a buffer to hold the previous timestep's BEV features
        #       (self._prev_bev = None)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: BatchData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step.

        Steps:
          1. Move batch to the correct device.
          2. Run forward pass through self.model to get predictions and new BEV.
          3. Update self._prev_bev with the new BEV features.
          4. Run HungarianMatcher to compute optimal prediction → GT assignment.
          5. Populate `frame.matches` for each frame in the batch.
          6. Compute DetectionLoss from predictions, targets, and matches.
          7. Log individual loss components (loss_cls, loss_bbox, loss_giou).
          8. Return total loss scalar for Lightning to call `.backward()`.

        Args:
            batch: `BatchData` from the DataLoader.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            Scalar total loss tensor.
        """
        # TODO: batch = batch.to(self.device)
        # TODO: predictions, new_bev = self.model(batch, prev_bev=self._prev_bev)
        # TODO: self._prev_bev = new_bev.detach()
        # TODO: matches = self.matcher.match_batch(predictions, targets)
        # TODO: total_loss, loss_dict = self.loss(predictions, targets, matches)
        # TODO: self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        # TODO: return total_loss
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        """Reset the evaluator and clear temporal BEV state at epoch start."""
        # TODO: self.evaluator.reset()
        # TODO: self._prev_bev = None
        raise NotImplementedError

    def validation_step(
        self,
        batch: BatchData,
        batch_idx: int,
    ) -> None:
        """Run inference and accumulate predictions for epoch-level evaluation.

        Steps:
          1. Move batch to device.
          2. Run forward pass (no gradient computation).
          3. Update self._prev_bev.
          4. Accumulate predictions and targets via self.evaluator.update().

        Args:
            batch: `BatchData` from the validation DataLoader.
            batch_idx: Current batch index.
        """
        # TODO: batch = batch.to(self.device)
        # TODO: with torch.no_grad(): predictions, new_bev = self.model(...)
        # TODO: self._prev_bev = new_bev.detach()
        # TODO: extract per-frame predictions and targets from batch
        # TODO: self.evaluator.update(predictions_list, targets_list)
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        """Compute and log all evaluation metrics at the end of validation.

        Calls self.evaluator.compute() to obtain mAP, NDS, per-class APs, and
        TP metrics, then logs them all via self.log_dict for TensorBoard /
        WandB tracking.
        """
        # TODO: metrics = self.evaluator.compute()
        # TODO: self.log_dict({"val/" + k: v for k, v in metrics.items()}, ...)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimiser with a cosine annealing LR schedule.

        Returns:
            A Lightning-compatible dict with keys "optimizer" and "lr_scheduler".
        """
        # TODO: instantiate torch.optim.AdamW with self.learning_rate and self.weight_decay
        # TODO: instantiate torch.optim.lr_scheduler.CosineAnnealingLR with T_max=self.max_epochs
        # TODO: return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, ...}}
        raise NotImplementedError
