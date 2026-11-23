"""2D Big Rock obstacle candidate identification using connected components and shape descriptors."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import cv2
import numpy as np

from src.intelligence.terrain_statistics import classify_relative_position, get_valid_mask

BIG_ROCK_CLASS_ID = 3


def detect_obstacle_candidates(
    pred_mask: np.ndarray,
    prob_map: np.ndarray,
    confidence_map: np.ndarray,
    min_area_px: int = 64,
    min_confidence: float = 0.35,
    ignore_index: int = 255,
) -> List[Dict[str, Any]]:
    """Identify 2D Big Rock regions as unverified obstacle candidates (never physical proof)."""
    h, w = pred_mask.shape
    total_pixels = float(h * w)
    valid_mask = get_valid_mask(pred_mask, ignore_index=ignore_index)
    big_rock_mask = ((pred_mask == BIG_ROCK_CLASS_ID) & valid_mask).astype(np.uint8)

    if not np.any(big_rock_mask):
        return []

    big_rock_prob = prob_map[BIG_ROCK_CLASS_ID]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(big_rock_mask, connectivity=8)
    candidates: List[Dict[str, Any]] = []

    cand_idx = 1
    for comp_idx in range(1, num_labels):
        area = int(stats[comp_idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        comp_mask = labels == comp_idx
        mean_br_prob = float(big_rock_prob[comp_mask].mean())
        mean_softmax_conf = float(confidence_map[comp_mask].mean())
        if mean_br_prob < min_confidence:
            continue

        x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
        y = int(stats[comp_idx, cv2.CC_STAT_TOP])
        bw = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[comp_idx, 0])
        cy = float(centroids[comp_idx, 1])

        ext_contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(ext_contours, key=cv2.contourArea) if ext_contours else np.empty((0, 1, 2), dtype=np.int32)
        perimeter = float(cv2.arcLength(contour, True)) if len(contour) > 0 else 0.0
        circularity = float((4.0 * math.pi * area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
        circularity = min(circularity, 1.0)

        hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if len(contour) >= 3 else float(area)
        solidity = float(area / hull_area) if hull_area > 0 else 1.0
        extent = float(area / max(bw * bh, 1))
        aspect_ratio = float(bw / max(bh, 1))

        candidates.append(
            {
                "candidate_id": f"OBS_CAND_{cand_idx:03d}",
                "source_class": "big_rock",
                "bbox": [x, y, bw, bh],
                "pixel_area": area,
                "image_coverage_pct": round((area / max(total_pixels, 1.0)) * 100.0, 4),
                "confidence": round(mean_br_prob, 6),
                "confidence_details": {
                    "mean_big_rock_probability": round(mean_br_prob, 6),
                    "max_big_rock_probability": round(float(big_rock_prob[comp_mask].max()), 6),
                    "mean_softmax_confidence": round(mean_softmax_conf, 6),
                },
                "centroid": [round(cx, 2), round(cy, 2)],
                "shape": {
                    "perimeter_px": round(perimeter, 2),
                    "aspect_ratio_w_over_h": round(aspect_ratio, 4),
                    "circularity": round(circularity, 4),
                    "solidity": round(solidity, 4),
                    "bbox_extent": round(extent, 4),
                },
                "relative_position": classify_relative_position(cx, cy, w, h),
                "is_confirmed_physical_obstacle": False,
                "scientific_note": (
                    "2D visual Big Rock candidate only; does not establish physical rock height, "
                    "3D volume, clearance, or collision risk without stereo range/elevation data."
                ),
            }
        )
        cand_idx += 1

    candidates.sort(key=lambda c: c["pixel_area"], reverse=True)
    return candidates
