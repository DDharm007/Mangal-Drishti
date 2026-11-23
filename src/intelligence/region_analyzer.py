"""Connected-component spatial terrain region analysis using OpenCV."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from src.intelligence.terrain_statistics import (
    CLASS_ID_TO_KEY,
    classify_relative_position,
    get_valid_mask,
)


def analyze_regions(
    pred_mask: np.ndarray,
    confidence_map: np.ndarray,
    min_region_area_px: int = 120,
    ignore_index: int = 255,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract connected components and contour summaries per class above min_region_area_px."""
    h, w = pred_mask.shape
    total_pixels = float(h * w)
    valid_mask = get_valid_mask(pred_mask, ignore_index=ignore_index)

    regions: List[Dict[str, Any]] = []
    filtered_small_count = 0
    region_label_map = np.zeros((h, w), dtype=np.int32)
    contours_by_region: List[Tuple[int, np.ndarray]] = []

    reg_counter = 1
    for cid, class_key in CLASS_ID_TO_KEY.items():
        binary = ((pred_mask == cid) & valid_mask).astype(np.uint8)
        if not np.any(binary):
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_idx in range(1, num_labels):
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < min_region_area_px:
                filtered_small_count += 1
                continue

            comp_mask = labels == comp_idx
            x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            y = int(stats[comp_idx, cv2.CC_STAT_TOP])
            bw = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[comp_idx, 0])
            cy = float(centroids[comp_idx, 1])

            mean_conf = float(confidence_map[comp_mask].mean())
            comp_u8 = comp_mask.astype(np.uint8)
            ext_contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(ext_contours, key=cv2.contourArea) if ext_contours else np.empty((0, 1, 2), dtype=np.int32)
            perimeter = float(cv2.arcLength(largest_contour, True)) if len(largest_contour) > 0 else 0.0
            approx = (
                cv2.approxPolyDP(largest_contour, epsilon=0.015 * perimeter, closed=True)
                if len(largest_contour) > 0
                else largest_contour
            )
            poly_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx[:32]]

            region_id = f"REG_{reg_counter:03d}"
            region_label_map[comp_mask] = reg_counter
            if len(largest_contour) > 0:
                contours_by_region.append((cid, largest_contour))

            regions.append(
                {
                    "region_id": region_id,
                    "class_id": cid,
                    "terrain_class": class_key,
                    "pixel_area": area,
                    "bbox": [x, y, bw, bh],
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "contour": {
                        "num_vertices": int(len(largest_contour)),
                        "simplified_vertices": poly_pts,
                        "perimeter_px": round(perimeter, 2),
                    },
                    "relative_position": classify_relative_position(cx, cy, w, h),
                    "mean_confidence": round(mean_conf, 6),
                    "image_coverage_pct": round((area / max(total_pixels, 1.0)) * 100.0, 4),
                }
            )
            reg_counter += 1

    regions.sort(key=lambda r: r["pixel_area"], reverse=True)
    meta = {
        "min_region_area_px": int(min_region_area_px),
        "retained_region_count": len(regions),
        "filtered_small_component_count": int(filtered_small_count),
        "region_label_map": region_label_map,
        "contours_by_class": contours_by_region,
    }
    return regions, meta
