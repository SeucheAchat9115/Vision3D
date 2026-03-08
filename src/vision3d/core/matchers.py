"""
Hungarian matcher for Vision3D.

Provides `HungarianMatcher`, which computes the optimal bipartite assignment
between a set of predicted boxes and a set of ground-truth boxes using the
Hungarian algorithm (scipy.optimize.linear_sum_assignment).
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)


class HungarianMatcher:
    """Computes optimal bipartite matching between predictions and ground truth."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 0.25,
    ) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def match(
        self,
        predictions: BoundingBox3DPrediction,
        targets: BoundingBox3DTarget,
    ) -> MatchingResult:
        """Compute the optimal bipartite matching for a single frame."""
        M = predictions.boxes.shape[0]
        N = targets.boxes.shape[0]
        device = predictions.boxes.device
        if N == 0 or M == 0:
            return MatchingResult(
                pred_indices=torch.zeros(0, dtype=torch.long, device=device),
                gt_indices=torch.zeros(0, dtype=torch.long, device=device),
            )
        if predictions.scores.dim() == 1:
            scores = predictions.scores.unsqueeze(-1).expand(M, 1)
        else:
            scores = predictions.scores
        num_classes = scores.shape[1]
        gt_labels = targets.labels.long()
        probs = scores.softmax(dim=-1)
        gt_labels_clamped = gt_labels.clamp(0, num_classes - 1)
        cls_cost = -probs[:, gt_labels_clamped]
        bbox_cost = torch.cdist(predictions.boxes, targets.boxes, p=1)
        cost = self.cost_class * cls_cost + self.cost_bbox * bbox_cost
        cost_np = cost.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        return MatchingResult(
            pred_indices=torch.tensor(row_ind, dtype=torch.long, device=device),
            gt_indices=torch.tensor(col_ind, dtype=torch.long, device=device),
        )

    def match_batch(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
    ) -> list[MatchingResult]:
        """Apply `match` independently to each frame in a batch."""
        assert len(predictions) == len(targets)
        return [self.match(p, t) for p, t in zip(predictions, targets, strict=False)]
