"""Softmax confidence distribution and per-class analysis (excluding ignore_index)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.intelligence.terrain_statistics import CLASS_ID_TO_KEY, get_valid_mask


def analyze_confidence(
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.55,
    ignore_index: int = 255,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute softmax max-probability map, summary statistics, and configurable confidence tiers."""
    max_prob_map = np.max(prob_map, axis=0).astype(np.float32)
    valid_mask = get_valid_mask(pred_mask, ignore_index=ignore_index)
    valid_count = int(valid_mask.sum())

    per_class_conf: Dict[str, Any] = {}
    if valid_count == 0:
        for key in CLASS_ID_TO_KEY.values():
            per_class_conf[key] = None
        empty_hist, edges = np.histogram(np.empty(0, dtype=np.float32), bins=10, range=(0.0, 1.0))
        return {
            "mean_confidence": 0.0,
            "median_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "std_confidence": 0.0,
            "low_confidence_pixel_percentage": 0.0,
            "per_class_mean_confidence": per_class_conf,
            "confidence_categories_pct": {"high": 0.0, "medium": 0.0, "low": 0.0},
            "thresholds": {"high_min": high_threshold, "medium_min": medium_threshold},
            "confidence_histogram": {
                "bin_edges": [round(float(x), 2) for x in edges.tolist()],
                "pixel_counts": [int(x) for x in empty_hist.tolist()],
            },
            "is_calibrated_probability": False,
            "calibration_note": (
                "Raw neural network softmax confidence is uncalibrated and does not guarantee "
                "prediction correctness or out-of-distribution reliability."
            ),
        }, max_prob_map

    valid_conf = max_prob_map[valid_mask]
    valid_preds = pred_mask[valid_mask]

    high_pct = float((valid_conf >= high_threshold).mean() * 100.0)
    med_pct = float(((valid_conf >= medium_threshold) & (valid_conf < high_threshold)).mean() * 100.0)
    low_pct = float((valid_conf < medium_threshold).mean() * 100.0)

    for cid, key in CLASS_ID_TO_KEY.items():
        cls_mask = valid_preds == cid
        per_class_conf[key] = round(float(valid_conf[cls_mask].mean()), 6) if np.any(cls_mask) else None

    hist_counts, bin_edges = np.histogram(valid_conf, bins=10, range=(0.0, 1.0))

    summary = {
        "mean_confidence": round(float(valid_conf.mean()), 6),
        "median_confidence": round(float(np.median(valid_conf)), 6),
        "min_confidence": round(float(valid_conf.min()), 6),
        "max_confidence": round(float(valid_conf.max()), 6),
        "std_confidence": round(float(valid_conf.std()), 6),
        "low_confidence_pixel_percentage": round(low_pct, 4),
        "per_class_mean_confidence": per_class_conf,
        "confidence_categories_pct": {
            "high": round(high_pct, 4),
            "medium": round(med_pct, 4),
            "low": round(low_pct, 4),
        },
        "thresholds": {
            "high_min": float(high_threshold),
            "medium_min": float(medium_threshold),
        },
        "confidence_histogram": {
            "bin_edges": [round(float(x), 2) for x in bin_edges.tolist()],
            "pixel_counts": [int(x) for x in hist_counts.tolist()],
        },
        "is_calibrated_probability": False,
        "calibration_note": (
            "Raw softmax probability reflects relative logit dominance under training priors "
            "and is not temperature-calibrated physical certainty."
        ),
    }
    return summary, max_prob_map
