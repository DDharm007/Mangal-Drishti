"""Segmentation loss functions consuming Step 2 verified class weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_step2_class_weights(
    class_weights_path: Path,
    method: str = "median_frequency_capped",
    num_classes: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Load verified train-only class weights from Step 2 metadata."""
    data = json.loads(class_weights_path.read_text(encoding="utf-8"))
    methods = data["weighting_methods"]
    if method not in methods:
        raise KeyError(f"Weighting method '{method}' not found. Available: {list(methods.keys())}")
    w_map = methods[method]
    weights = torch.tensor([float(w_map[str(i)]) for i in range(num_classes)], dtype=torch.float32)
    return weights, {str(i): float(w_map[str(i)]) for i in range(num_classes)}


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice loss ignoring `ignore_index` pixels."""

    def __init__(self, num_classes: int = 4, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0
        probs = F.softmax(logits, dim=1)
        safe_targets = targets.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid_mask.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        present = one_hot.sum(dim=dims) > 0
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice[present].mean()


class CombinedCEDiceLoss(nn.Module):
    """Weighted CrossEntropyLoss optionally combined with SoftDiceLoss."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        dice_weight: float = 0.0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index) if self.dice_weight > 0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, targets)
        if self.dice is not None:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        return loss
