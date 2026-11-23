"""Predictive Shannon entropy uncertainty quantification (H = -sum(p * log(p)))."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from src.intelligence.terrain_statistics import CLASS_ID_TO_KEY, get_valid_mask


def analyze_uncertainty(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    high_uncertainty_threshold: float = 0.55,
    ignore_index: int = 255,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Compute per-pixel Shannon entropy H = -sum(p * log(p)) and normalized entropy in [0, 1]."""
    num_classes = prob_map.shape[0]
    clipped = np.clip(prob_map.astype(np.float64), 1e-12, 1.0)
    entropy_map = (-np.sum(clipped * np.log(clipped), axis=0)).astype(np.float32)
    max_possible_entropy = math.log(float(num_classes))
    norm_entropy_map = np.clip(entropy_map / max(max_possible_entropy, 1e-8), 0.0, 1.0).astype(np.float32)

    valid_mask = get_valid_mask(pred_mask, ignore_index=ignore_index)
    valid_count = int(valid_mask.sum())
    per_class_ent: Dict[str, Any] = {}

    if valid_count == 0:
        for key in CLASS_ID_TO_KEY.values():
            per_class_ent[key] = None
        return {
            "entropy_formula": "H = -sum_c(p_c * ln(p_c))",
            "max_theoretical_entropy_nats": round(max_possible_entropy, 6),
            "mean_entropy": 0.0,
            "median_entropy": 0.0,
            "std_entropy": 0.0,
            "max_entropy": 0.0,
            "mean_normalized_entropy": 0.0,
            "high_uncertainty_pixel_percentage": 0.0,
            "high_uncertainty_normalized_threshold": float(high_uncertainty_threshold),
            "per_class_mean_entropy": per_class_ent,
        }, entropy_map, norm_entropy_map

    valid_ent = entropy_map[valid_mask]
    valid_norm = norm_entropy_map[valid_mask]
    valid_preds = pred_mask[valid_mask]

    for cid, key in CLASS_ID_TO_KEY.items():
        cls_mask = valid_preds == cid
        per_class_ent[key] = round(float(valid_ent[cls_mask].mean()), 6) if np.any(cls_mask) else None

    high_unc_pct = float((valid_norm >= high_uncertainty_threshold).mean() * 100.0)

    summary = {
        "entropy_formula": "H = -sum_c(p_c * ln(p_c))",
        "max_theoretical_entropy_nats": round(max_possible_entropy, 6),
        "mean_entropy": round(float(valid_ent.mean()), 6),
        "median_entropy": round(float(np.median(valid_ent)), 6),
        "std_entropy": round(float(valid_ent.std()), 6),
        "max_entropy": round(float(valid_ent.max()), 6),
        "mean_normalized_entropy": round(float(valid_norm.mean()), 6),
        "high_uncertainty_pixel_percentage": round(high_unc_pct, 4),
        "high_uncertainty_normalized_threshold": float(high_uncertainty_threshold),
        "per_class_mean_entropy": per_class_ent,
    }
    return summary, entropy_map, norm_entropy_map
