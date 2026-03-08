"""
Detection losses for Vision3D.

Provides `DetectionLoss`, which computes the total training loss given
predictions, ground-truth targets, and Hungarian matching results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
    MatchingResult,
)


class DetectionLoss(nn.Module):
    """Computes the combined classification and bounding-box regression loss."""

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
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
        matches: list[MatchingResult],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss for a batch."""
        loss_cls = self._classification_loss(predictions, targets, matches)
        loss_bbox = self._bbox_l1_loss(predictions, targets, matches)
        loss_giou = self._giou_loss(predictions, targets, matches)
        total = (
            self.cls_weight * loss_cls + self.bbox_weight * loss_bbox + self.giou_weight * loss_giou
        )
        return total, {"loss_cls": loss_cls, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _classification_loss(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
        matches: list[MatchingResult],
    ) -> torch.Tensor:
        """Compute sigmoid Focal Loss across all queries in the batch."""
        all_logits = []
        all_targets_cls = []
        for pred, tgt, match in zip(predictions, targets, matches, strict=False):
            num_queries = pred.boxes.shape[0]
            target_cls = torch.zeros(num_queries, self.num_classes, device=pred.boxes.device)
            if match.pred_indices.numel() > 0:
                pi = match.pred_indices
                gi = match.gt_indices
                cls_idx = tgt.labels[gi].long()
                for j in range(pi.shape[0]):
                    target_cls[pi[j], cls_idx[j]] = 1.0
            if pred.scores.dim() == 1:
                logits = pred.scores.unsqueeze(-1).expand(-1, self.num_classes)
            else:
                logits = pred.scores
            all_logits.append(logits)
            all_targets_cls.append(target_cls)
        logits_cat = torch.cat(all_logits, dim=0)
        tgt_cat = torch.cat(all_targets_cls, dim=0)
        p = torch.sigmoid(logits_cat)
        ce = F.binary_cross_entropy_with_logits(logits_cat, tgt_cat, reduction="none")
        p_t = p * tgt_cat + (1 - p) * (1 - tgt_cat)
        alpha_t = self.focal_alpha * tgt_cat + (1 - self.focal_alpha) * (1 - tgt_cat)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * ce).mean()

    def _bbox_l1_loss(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
        matches: list[MatchingResult],
    ) -> torch.Tensor:
        """Compute L1 loss on matched box parameter vectors."""
        losses = []
        for pred, tgt, match in zip(predictions, targets, matches, strict=False):
            if match.pred_indices.numel() == 0:
                losses.append(pred.boxes.sum() * 0.0)
                continue
            pred_boxes = pred.boxes[match.pred_indices]
            gt_boxes = tgt.boxes[match.gt_indices]
            losses.append(F.l1_loss(pred_boxes, gt_boxes))
        return torch.stack(losses).mean()

    def _giou_loss(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
        matches: list[MatchingResult],
    ) -> torch.Tensor:
        """Compute a GIoU-inspired geometric consistency loss on matched boxes."""
        losses = []
        for pred, tgt, match in zip(predictions, targets, matches, strict=False):
            if match.pred_indices.numel() == 0:
                losses.append(pred.boxes.sum() * 0.0)
                continue
            pred_boxes = pred.boxes[match.pred_indices][:, :6]
            gt_boxes = tgt.boxes[match.gt_indices][:, :6]
            p_min = pred_boxes[:, :3] - pred_boxes[:, 3:] / 2
            p_max = pred_boxes[:, :3] + pred_boxes[:, 3:] / 2
            g_min = gt_boxes[:, :3] - gt_boxes[:, 3:] / 2
            g_max = gt_boxes[:, :3] + gt_boxes[:, 3:] / 2
            inter_min = torch.max(p_min, g_min)
            inter_max = torch.min(p_max, g_max)
            inter_dims = (inter_max - inter_min).clamp(min=0)
            inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]
            p_vol = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
            g_vol = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]
            union_vol = p_vol + g_vol - inter_vol + 1e-8
            iou = inter_vol / union_vol
            enc_min = torch.min(p_min, g_min)
            enc_max = torch.max(p_max, g_max)
            enc_dims = (enc_max - enc_min).clamp(min=0)
            enc_vol = enc_dims[:, 0] * enc_dims[:, 1] * enc_dims[:, 2] + 1e-8
            giou = iou - (enc_vol - union_vol) / enc_vol
            losses.append((1 - giou).mean())
        return torch.stack(losses).mean()
