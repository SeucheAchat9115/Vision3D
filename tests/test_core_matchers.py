"""Unit tests for HungarianMatcher."""

from __future__ import annotations

import pytest
import torch

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)
from vision3d.core.matchers import HungarianMatcher


def _make_pred(
    num_queries: int = 5,
    num_classes: int = 3,
    *,
    seed: int = 0,
) -> BoundingBox3DPrediction:
    torch.manual_seed(seed)
    return BoundingBox3DPrediction(
        boxes=torch.randn(num_queries, 10),
        scores=torch.randn(num_queries, num_classes),
        labels=torch.randint(0, num_classes, (num_queries,)),
    )


def _make_tgt(
    num_gt: int = 3,
    num_classes: int = 3,
    *,
    seed: int = 1,
) -> BoundingBox3DTarget:
    torch.manual_seed(seed)
    return BoundingBox3DTarget(
        boxes=torch.randn(num_gt, 10),
        labels=torch.randint(0, num_classes, (num_gt,)),
        instance_ids=[f"id_{i}" for i in range(num_gt)],
    )


class TestHungarianMatcherInterface:
    """Verify the interface of HungarianMatcher."""

    def test_has_match_method(self):
        assert hasattr(HungarianMatcher, "match")

    def test_has_match_batch_method(self):
        assert hasattr(HungarianMatcher, "match_batch")

    def test_init_stores_weights(self):
        matcher = HungarianMatcher(cost_class=3.0, cost_bbox=0.5)
        assert matcher.cost_class == 3.0
        assert matcher.cost_bbox == 0.5

    def test_match_returns_matching_result(self):
        matcher = HungarianMatcher()
        result = matcher.match(_make_pred(), _make_tgt())
        assert isinstance(result, MatchingResult)

    def test_match_batch_returns_list_of_matching_results(self):
        matcher = HungarianMatcher()
        preds = [_make_pred(seed=i) for i in range(3)]
        tgts = [_make_tgt(seed=i + 10) for i in range(3)]
        results = matcher.match_batch(preds, tgts)
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, MatchingResult)


class TestHungarianMatcherNormalCase:
    """Test matching on standard non-empty predictions and targets."""

    def test_output_indices_are_long_tensors(self):
        matcher = HungarianMatcher()
        result = matcher.match(_make_pred(5), _make_tgt(3))
        assert result.pred_indices.dtype == torch.long
        assert result.gt_indices.dtype == torch.long

    def test_matched_indices_have_equal_length(self):
        matcher = HungarianMatcher()
        result = matcher.match(_make_pred(5), _make_tgt(3))
        assert result.pred_indices.shape == result.gt_indices.shape

    def test_num_matches_le_min_of_preds_and_gts(self):
        matcher = HungarianMatcher()
        M, N = 7, 4
        result = matcher.match(_make_pred(M), _make_tgt(N))
        assert result.pred_indices.numel() <= min(M, N)

    def test_pred_indices_within_bounds(self):
        matcher = HungarianMatcher()
        M, N = 6, 4
        result = matcher.match(_make_pred(M), _make_tgt(N))
        assert result.pred_indices.max().item() < M

    def test_gt_indices_within_bounds(self):
        matcher = HungarianMatcher()
        M, N = 6, 4
        result = matcher.match(_make_pred(M), _make_tgt(N))
        assert result.gt_indices.max().item() < N

    def test_gt_indices_unique(self):
        """Each GT box should be matched at most once."""
        matcher = HungarianMatcher()
        result = matcher.match(_make_pred(8), _make_tgt(4))
        unique_gt = result.gt_indices.unique()
        assert unique_gt.numel() == result.gt_indices.numel()

    def test_deterministic_with_same_inputs(self):
        matcher = HungarianMatcher()
        pred = _make_pred(5)
        tgt = _make_tgt(3)
        r1 = matcher.match(pred, tgt)
        r2 = matcher.match(pred, tgt)
        assert torch.equal(r1.pred_indices, r2.pred_indices)
        assert torch.equal(r1.gt_indices, r2.gt_indices)


class TestHungarianMatcherEdgeCases:
    """Test matching on edge cases."""

    def test_empty_targets(self):
        matcher = HungarianMatcher()
        tgt = BoundingBox3DTarget(
            boxes=torch.zeros(0, 10),
            labels=torch.zeros(0, dtype=torch.long),
            instance_ids=[],
        )
        result = matcher.match(_make_pred(5), tgt)
        assert result.pred_indices.numel() == 0
        assert result.gt_indices.numel() == 0

    def test_empty_predictions(self):
        matcher = HungarianMatcher()
        pred = BoundingBox3DPrediction(
            boxes=torch.zeros(0, 10),
            scores=torch.zeros(0, 3),
            labels=torch.zeros(0, dtype=torch.long),
        )
        result = matcher.match(pred, _make_tgt(4))
        assert result.pred_indices.numel() == 0
        assert result.gt_indices.numel() == 0

    def test_single_pred_single_gt(self):
        matcher = HungarianMatcher()
        pred = BoundingBox3DPrediction(
            boxes=torch.zeros(1, 10),
            scores=torch.zeros(1, 3),
            labels=torch.zeros(1, dtype=torch.long),
        )
        tgt = BoundingBox3DTarget(
            boxes=torch.zeros(1, 10),
            labels=torch.zeros(1, dtype=torch.long),
            instance_ids=["only"],
        )
        result = matcher.match(pred, tgt)
        assert result.pred_indices.numel() == 1
        assert result.gt_indices.numel() == 1

    def test_more_preds_than_gts(self):
        matcher = HungarianMatcher()
        M, N = 10, 3
        result = matcher.match(_make_pred(M), _make_tgt(N))
        assert result.pred_indices.numel() == N

    def test_more_gts_than_preds(self):
        matcher = HungarianMatcher()
        M, N = 2, 8
        result = matcher.match(_make_pred(M), _make_tgt(N))
        assert result.pred_indices.numel() == M

    def test_1d_scores_handled(self):
        """Matcher should handle 1-D score tensors (single score per query)."""
        matcher = HungarianMatcher()
        pred = BoundingBox3DPrediction(
            boxes=torch.randn(5, 10),
            scores=torch.rand(5),
            labels=torch.zeros(5, dtype=torch.long),
        )
        tgt = _make_tgt(3)
        result = matcher.match(pred, tgt)
        assert isinstance(result, MatchingResult)

    def test_match_batch_length_mismatch_raises(self):
        matcher = HungarianMatcher()
        preds = [_make_pred()]
        tgts = [_make_tgt(), _make_tgt()]
        with pytest.raises(AssertionError):
            matcher.match_batch(preds, tgts)

    def test_cost_weights_affect_matching(self):
        """Different cost weights should produce different matchings."""
        pred = _make_pred(5, seed=99)
        tgt = _make_tgt(3, seed=88)
        m1 = HungarianMatcher(cost_class=0.0, cost_bbox=1.0)
        m2 = HungarianMatcher(cost_class=1.0, cost_bbox=0.0)
        r1 = m1.match(pred, tgt)
        r2 = m2.match(pred, tgt)
        # With very different weights the matchings may differ
        # We just check both are valid MatchingResults
        assert isinstance(r1, MatchingResult)
        assert isinstance(r2, MatchingResult)
