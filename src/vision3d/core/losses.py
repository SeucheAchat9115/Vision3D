"""
Detection losses for Vision3D.

Provides `DetectionLoss`, which computes the total training loss given
predictions, ground-truth targets, and Hungarian matching results.

Loss components:
  - **Classification loss**: Sigmoid Focal Loss applied to *all* predictions.
    Matched predictions are trained toward their assigned GT class; unmatched
    predictions are trained toward a background "no-object" label.
  - **Box regression loss (L1)**: L1 distance over all 10 box parameters
    applied only to matched predictions.
  - **Box regression loss (GIoU)**: A GIoU-like term on the matched boxes to
    stabilise regression of spatially overlapping boxes.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)


class DetectionLoss(nn.Module):
    """Computes the combined classification and bounding-box regression loss.

    Follows the DETR-style end-to-end training objective:
      1. Use `MatchingResult` to identify which predictions supervise which GT boxes.
      2. Compute Focal Loss over all predictions for classification.
      3. Compute L1 loss over matched boxes for box regression.
      4. Compute GIoU loss over matched boxes for geometric consistency.
      5. Return a weighted sum as the total scalar loss, plus a dict of
         individual loss components for logging.

    Args:
        num_classes: Number of object categories (excluding background).
        cls_weight: Scalar weight applied to the classification loss term.
        bbox_weight: Scalar weight applied to the L1 regression loss term.
        giou_weight: Scalar weight applied to the GIoU loss term.
        focal_alpha: Alpha parameter for Focal Loss class-imbalance correction.
        focal_gamma: Gamma parameter for Focal Loss easy-example down-weighting.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        giou_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        # TODO: store all hyperparameters as instance attributes
        raise NotImplementedError

    def forward(
        self,
        predictions: List[BoundingBox3DPrediction],
        targets: List[BoundingBox3DTarget],
        matches: List[MatchingResult],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total loss for a batch.

        Args:
            predictions: List of per-frame model predictions (length = batch_size).
            targets: List of per-frame ground-truth annotations (length = batch_size).
            matches: List of per-frame matching results from `HungarianMatcher`
                (length = batch_size).

        Returns:
            Tuple of:
              - `total_loss`: Scalar tensor used for `loss.backward()`.
              - `loss_dict`: Dict with keys "loss_cls", "loss_bbox", "loss_giou"
                and their unweighted scalar values for TensorBoard logging.
        """
        # TODO: call self._classification_loss(predictions, targets, matches)
        # TODO: call self._bbox_l1_loss(predictions, targets, matches)
        # TODO: call self._giou_loss(predictions, targets, matches)
        # TODO: combine: total = cls_weight*loss_cls + bbox_weight*loss_bbox + giou_weight*loss_giou
        # TODO: return (total_loss, {"loss_cls": ..., "loss_bbox": ..., "loss_giou": ...})
        raise NotImplementedError

    def _classification_loss(
        self,
        predictions: List[BoundingBox3DPrediction],
        targets: List[BoundingBox3DTarget],
        matches: List[MatchingResult],
    ) -> torch.Tensor:
        """Compute sigmoid Focal Loss across all queries in the batch.

        All `num_queries * batch_size` predictions are evaluated:
          - Matched predictions: one-hot target for the assigned GT class.
          - Unmatched predictions: all-zeros target (background).

        Args:
            predictions: Per-frame predictions.
            targets: Per-frame ground-truth.
            matches: Per-frame Hungarian matching results.

        Returns:
            Scalar focal loss averaged over all predictions and classes.
        """
        # TODO: build target class probability tensors for all predictions
        #       (batch_size, num_queries, num_classes) — zeros for background,
        #       one-hot for matched predictions
        # TODO: apply sigmoid focal loss formula:
        #       FL = -alpha * (1 - p)^gamma * log(p) for positives
        #            -(1-alpha) * p^gamma * log(1-p) for negatives
        # TODO: return mean over all elements
        raise NotImplementedError

    def _bbox_l1_loss(
        self,
        predictions: List[BoundingBox3DPrediction],
        targets: List[BoundingBox3DTarget],
        matches: List[MatchingResult],
    ) -> torch.Tensor:
        """Compute L1 loss on matched box parameter vectors.

        Only matched prediction–GT pairs contribute to this loss.

        Args:
            predictions: Per-frame predictions.
            targets: Per-frame ground-truth.
            matches: Per-frame matching results.

        Returns:
            Scalar L1 loss averaged over matched pairs and box parameters.
        """
        # TODO: for each frame, gather matched predicted boxes using match.pred_indices
        # TODO: gather corresponding GT boxes using match.gt_indices
        # TODO: compute F.l1_loss between the gathered tensors
        # TODO: return mean over all matched pairs
        raise NotImplementedError

    def _giou_loss(
        self,
        predictions: List[BoundingBox3DPrediction],
        targets: List[BoundingBox3DTarget],
        matches: List[MatchingResult],
    ) -> torch.Tensor:
        """Compute a GIoU-inspired geometric consistency loss on matched boxes.

        Operates on the (x, y, z, w, l, h) subset of the 10-DOF box parameters
        to penalise predictions that are spatially inconsistent with their
        assigned GT boxes (beyond what L1 alone would penalise).

        Args:
            predictions: Per-frame predictions.
            targets: Per-frame ground-truth.
            matches: Per-frame matching results.

        Returns:
            Scalar GIoU loss averaged over matched pairs.
        """
        # TODO: extract (x, y, z, w, l, h) from matched predicted and GT boxes
        # TODO: compute axis-aligned 3-D IoU between predicted and GT boxes
        # TODO: compute the enclosing box volume for the GIoU penalty term
        # TODO: GIoU = IoU - (enclosing_vol - union_vol) / enclosing_vol
        # TODO: loss = 1 - GIoU; return mean over matched pairs
        raise NotImplementedError
