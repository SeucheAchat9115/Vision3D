"""
Hungarian matcher for Vision3D.

Provides `HungarianMatcher`, which computes the optimal bipartite assignment
between a set of predicted boxes and a set of ground-truth boxes using the
Hungarian algorithm (scipy.optimize.linear_sum_assignment).

The cost matrix combines a classification cost and a bounding-box distance
cost, weighted by configurable coefficients. The result is stored in a
`MatchingResult` dataclass.
"""

from __future__ import annotations

from typing import List

import torch

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)


class HungarianMatcher:
    """Computes optimal bipartite matching between predictions and ground truth.

    Uses the Hungarian algorithm to solve the linear assignment problem that
    minimises the total cost over all matched pairs. The cost of matching
    prediction *i* to ground-truth *j* is:

        cost(i, j) = cost_class * cls_cost(i, j)
                   + cost_bbox  * bbox_cost(i, j)

    where:
      - `cls_cost(i, j)` is the negative softmax probability of the
        ground-truth class for prediction *i* (focal-cost variant).
      - `bbox_cost(i, j)` is the L1 distance between the predicted and GT
        box parameter vectors (all 10 DOF).

    This class is not an `nn.Module` because it contains no learnable
    parameters. It is called once per training step to produce the
    `MatchingResult` used by `DetectionLoss`.

    Args:
        cost_class: Weight of the classification cost term.
        cost_bbox: Weight of the bounding-box L1 cost term.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 0.25,
    ) -> None:
        # TODO: store cost_class and cost_bbox as instance attributes
        raise NotImplementedError

    @torch.no_grad()
    def match(
        self,
        predictions: BoundingBox3DPrediction,
        targets: BoundingBox3DTarget,
    ) -> MatchingResult:
        """Compute the optimal bipartite matching for a single frame.

        This method is decorated with `@torch.no_grad()` because matching is
        purely used to select supervision pairs and should not participate in
        gradient computation.

        Args:
            predictions: Model predictions for one frame in the batch.
                `predictions.boxes` shape: (M, 10); `predictions.scores` shape:
                (M, num_classes).
            targets: Ground-truth for the same frame.
                `targets.boxes` shape: (N, 10); `targets.labels` shape: (N,).

        Returns:
            `MatchingResult` with:
              - `pred_indices`: 1-D int tensor of matched prediction indices.
              - `gt_indices`: 1-D int tensor of the corresponding GT indices.
        """
        # TODO: compute the classification cost matrix (M × N)
        # TODO: compute the bounding-box L1 cost matrix (M × N)
        # TODO: combine cost matrices: total_cost = cost_class * cls_cost + cost_bbox * bbox_cost
        # TODO: convert to numpy and call scipy.optimize.linear_sum_assignment
        # TODO: convert resulting index arrays back to torch tensors
        # TODO: return MatchingResult(pred_indices=..., gt_indices=...)
        raise NotImplementedError

    def match_batch(
        self,
        predictions: List[BoundingBox3DPrediction],
        targets: List[BoundingBox3DTarget],
    ) -> List[MatchingResult]:
        """Apply `match` independently to each frame in a batch.

        Args:
            predictions: List of per-frame predictions (length = batch_size).
            targets: List of per-frame ground-truth targets (length = batch_size).

        Returns:
            List of `MatchingResult` objects (length = batch_size).
        """
        # TODO: assert len(predictions) == len(targets)
        # TODO: call self.match(pred, tgt) for each (pred, tgt) pair
        # TODO: return the list of results
        raise NotImplementedError
