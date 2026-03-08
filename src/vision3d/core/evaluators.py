"""
Evaluation orchestration for Vision3D.

Provides `Vision3DEvaluator`, which accumulates per-frame predictions and
ground-truth targets across an entire validation epoch, then computes
detection metrics (mAP and NDS-like score) at epoch end.

All metric computations are implemented in pure PyTorch / NumPy to avoid
heavy external dependencies like the nuScenes devkit.
"""

from __future__ import annotations

import numpy as np
import torch

from vision3d.config.schema import (
    BoundingBox3DPrediction,
    BoundingBox3DTarget,
)


class Vision3DEvaluator:
    """Orchestrates the end-of-epoch evaluation loop."""

    def __init__(
        self,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        eval_range: float = 50.0,
        distance_thresholds: list[float] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = (
            class_names if class_names is not None else [f"class_{i}" for i in range(num_classes)]
        )
        self.eval_range = eval_range
        self.distance_thresholds = (
            distance_thresholds if distance_thresholds is not None else [0.5, 1.0, 2.0, 4.0]
        )
        self._all_predictions: list[BoundingBox3DPrediction] = []
        self._all_targets: list[BoundingBox3DTarget] = []

    def reset(self) -> None:
        """Clear all accumulated predictions and targets."""
        self._all_predictions = []
        self._all_targets = []

    def update(
        self,
        predictions: list[BoundingBox3DPrediction],
        targets: list[BoundingBox3DTarget],
    ) -> None:
        """Accumulate predictions and targets from one validation step."""
        for pred, tgt in zip(predictions, targets, strict=False):
            centres = tgt.boxes[:, :2]
            dist = torch.norm(centres, dim=1)
            mask = dist <= self.eval_range
            filtered_tgt = BoundingBox3DTarget(
                boxes=tgt.boxes[mask],
                labels=tgt.labels[mask],
                instance_ids=[
                    iid for iid, m in zip(tgt.instance_ids, mask.tolist(), strict=False) if m
                ],
            )
            pred_centres = pred.boxes[:, :2]
            pred_dist = torch.norm(pred_centres, dim=1)
            pred_mask = pred_dist <= self.eval_range
            filtered_pred = BoundingBox3DPrediction(
                boxes=pred.boxes[pred_mask],
                scores=pred.scores[pred_mask],
                labels=pred.labels[pred_mask],
            )
            self._all_predictions.append(filtered_pred)
            self._all_targets.append(filtered_tgt)

    def compute(self) -> dict[str, float]:
        """Compute and return all evaluation metrics over the accumulated data."""
        ap_per_class = []
        for cls_idx in range(self.num_classes):
            ap_per_thr = [
                self._compute_ap_for_class(cls_idx, thr) for thr in self.distance_thresholds
            ]
            ap_per_class.append(float(np.mean(ap_per_thr)))
        mAP = float(np.mean(ap_per_class))
        tp_metrics = self._compute_tp_metrics()
        tp_values = [min(v, 1.0) for v in tp_metrics.values()]
        nds = 0.5 * (mAP + float(np.mean(tp_values))) if tp_values else mAP
        metrics: dict[str, float] = {"mAP": mAP, "NDS": nds}
        for i, name in enumerate(self.class_names):
            metrics[f"AP/{name}"] = ap_per_class[i]
        metrics.update(tp_metrics)
        return metrics

    def _compute_ap_for_class(
        self,
        class_idx: int,
        distance_threshold: float,
    ) -> float:
        """Compute average precision for a single class at one distance threshold."""
        all_scores: list[float] = []
        all_tp: list[int] = []
        num_gt = 0
        for pred, tgt in zip(self._all_predictions, self._all_targets, strict=False):
            gt_mask = tgt.labels == class_idx
            gt_boxes = tgt.boxes[gt_mask]
            num_gt += gt_boxes.shape[0]
            pred_mask = pred.labels == class_idx
            p_boxes = pred.boxes[pred_mask]
            p_scores = pred.scores[pred_mask]
            matched = [False] * gt_boxes.shape[0]
            for i in range(p_boxes.shape[0]):
                score = p_scores[i].item()
                all_scores.append(score)
                if gt_boxes.shape[0] == 0:
                    all_tp.append(0)
                    continue
                dists = torch.norm(p_boxes[i : i + 1, :2] - gt_boxes[:, :2], dim=1)
                best = int(torch.argmin(dists).item())
                if dists[best].item() <= distance_threshold and not matched[best]:
                    matched[best] = True
                    all_tp.append(1)
                else:
                    all_tp.append(0)
        if num_gt == 0 or len(all_scores) == 0:
            return 0.0
        order = np.argsort(-np.array(all_scores))
        tp_arr = np.array(all_tp)[order]
        cumtp = np.cumsum(tp_arr)
        cumfp = np.cumsum(1 - tp_arr)
        precision = cumtp / (cumtp + cumfp + 1e-8)
        recall = cumtp / (num_gt + 1e-8)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec = precision[recall >= t]
            ap += float(np.max(prec)) if len(prec) > 0 else 0.0
        return ap / 11.0

    def _compute_tp_metrics(self) -> dict[str, float]:
        """Compute mean TP metrics (ATE, ASE, AOE, AVE) over all matched TPs."""
        ate_list: list[float] = []
        ase_list: list[float] = []
        aoe_list: list[float] = []
        ave_list: list[float] = []
        for pred, tgt in zip(self._all_predictions, self._all_targets, strict=False):
            if tgt.boxes.shape[0] == 0 or pred.boxes.shape[0] == 0:
                continue
            dists = torch.cdist(pred.boxes[:, :2], tgt.boxes[:, :2])
            for _ in range(min(pred.boxes.shape[0], tgt.boxes.shape[0])):
                if dists.numel() == 0:
                    break
                min_val = dists.min()
                if min_val.item() > 2.0:
                    break
                idx = dists.argmin()
                pi = int(idx // dists.shape[1])
                gi = int(idx % dists.shape[1])
                p_box = pred.boxes[pi]
                g_box = tgt.boxes[gi]
                ate_list.append(float(torch.norm(p_box[:3] - g_box[:3]).item()))
                pw, pl, ph = p_box[3].item(), p_box[4].item(), p_box[5].item()
                gw, gl, gh = g_box[3].item(), g_box[4].item(), g_box[5].item()
                inter = min(pw, gw) * min(pl, gl) * min(ph, gh)
                union = pw * pl * ph + gw * gl * gh - inter + 1e-8
                ase_list.append(float(1.0 - inter / union))
                p_yaw = float(torch.atan2(p_box[6], p_box[7]).item())
                g_yaw = float(torch.atan2(g_box[6], g_box[7]).item())
                aoe_list.append(abs(p_yaw - g_yaw) % (2 * 3.14159))
                ave_list.append(float(torch.norm(p_box[8:10] - g_box[8:10]).item()))
                dists[pi, :] = 1e9
                dists[:, gi] = 1e9
        return {
            "ATE": float(np.mean(ate_list)) if ate_list else 0.0,
            "ASE": float(np.mean(ase_list)) if ase_list else 0.0,
            "AOE": float(np.mean(aoe_list)) if aoe_list else 0.0,
            "AVE": float(np.mean(ave_list)) if ave_list else 0.0,
        }
