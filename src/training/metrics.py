"""Confusion-matrix semantic segmentation metrics strictly excluding ignore_index=255."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


CLASS_NAMES: Dict[int, str] = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big rock",
}


class SegmentationMetrics:
    """Accumulates pixel confusion matrix across batches and computes IoU, Dice, Recall."""

    def __init__(self, num_classes: int = 4, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        if not valid.any():
            return
        t_valid = targets[valid].view(-1).cpu().numpy().astype(np.int64)
        p_valid = preds[valid].view(-1).cpu().numpy().astype(np.int64)
        indices = t_valid * self.num_classes + p_valid
        counts = np.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.conf_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        cm = self.conf_matrix.astype(np.float64)
        tp = np.diag(cm)
        gt_sum = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        union = gt_sum + pred_sum - tp

        per_class_iou: Dict[str, float] = {}
        per_class_dice: Dict[str, float] = {}
        per_class_recall: Dict[str, float] = {}
        per_class_precision: Dict[str, float] = {}

        valid_ious: List[float] = []
        valid_dices: List[float] = []

        for c in range(self.num_classes):
            name = CLASS_NAMES.get(c, str(c))
            iou = float(tp[c] / union[c]) if union[c] > 0 else 0.0
            dice = float(2.0 * tp[c] / (gt_sum[c] + pred_sum[c])) if (gt_sum[c] + pred_sum[c]) > 0 else 0.0
            recall = float(tp[c] / gt_sum[c]) if gt_sum[c] > 0 else 0.0
            prec = float(tp[c] / pred_sum[c]) if pred_sum[c] > 0 else 0.0

            per_class_iou[name] = round(iou, 6)
            per_class_dice[name] = round(dice, 6)
            per_class_recall[name] = round(recall, 6)
            per_class_precision[name] = round(prec, 6)

            if gt_sum[c] > 0:
                valid_ious.append(iou)
                valid_dices.append(dice)

        total_valid = float(cm.sum())
        pixel_acc = float(tp.sum() / total_valid) if total_valid > 0 else 0.0
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        mdice = float(np.mean(valid_dices)) if valid_dices else 0.0

        return {
            "mIoU": round(miou, 6),
            "mean_dice": round(mdice, 6),
            "pixel_accuracy": round(pixel_acc, 6),
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
            "per_class_recall": per_class_recall,
            "per_class_precision": per_class_precision,
            "sand_iou": per_class_iou["sand"],
            "sand_recall": per_class_recall["sand"],
            "big_rock_iou": per_class_iou["big rock"],
            "big_rock_recall": per_class_recall["big rock"],
            "valid_pixels_evaluated": int(total_valid),
        }
