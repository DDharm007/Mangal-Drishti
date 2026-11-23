"""Pixel-level terrain distribution, diversity, and boundary analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.intelligence.terrain_statistics import (
    CLASS_ID_TO_KEY,
    compute_boundary_mask,
    compute_shannon_diversity,
    get_valid_mask,
)


def analyze_terrain(pred_mask: np.ndarray, ignore_index: int = 255) -> Dict[str, Any]:
    """Compute valid-pixel terrain distribution, dominance, diversity, and boundary density."""
    valid_mask = get_valid_mask(pred_mask, ignore_index=ignore_index)
    total_pixels = int(pred_mask.size)
    valid_pixels = int(valid_mask.sum())
    ignored_pixels = int(total_pixels - valid_pixels)
    valid_pixel_pct = round((valid_pixels / max(total_pixels, 1)) * 100.0, 4)

    class_counts: Dict[str, int] = {}
    terrain_summary: Dict[str, float] = {}

    if valid_pixels > 0:
        valid_labels = pred_mask[valid_mask]
        counts = np.bincount(valid_labels.astype(np.int64), minlength=4)
    else:
        counts = np.zeros(4, dtype=np.int64)

    for cid, key in CLASS_ID_TO_KEY.items():
        c = int(counts[cid])
        class_counts[key] = c
        terrain_summary[key] = round((c / valid_pixels) * 100.0, 4) if valid_pixels > 0 else 0.0

    ranked = sorted(CLASS_ID_TO_KEY.values(), key=lambda k: class_counts[k], reverse=True)
    dominant: Optional[str] = ranked[0] if (valid_pixels > 0 and class_counts[ranked[0]] > 0) else None
    secondary: Optional[str] = ranked[1] if (valid_pixels > 0 and class_counts[ranked[1]] > 0) else None

    if valid_pixels > 0:
        props = counts.astype(np.float64) / float(valid_pixels)
        shannon_nats, shannon_norm = compute_shannon_diversity(props)
        boundary_mask = compute_boundary_mask(pred_mask, valid_mask)
        boundary_px = int(boundary_mask.sum())
        boundary_density_pct = round((boundary_px / float(valid_pixels)) * 100.0, 4)
    else:
        shannon_nats, shannon_norm = 0.0, 0.0
        boundary_px = 0
        boundary_density_pct = 0.0

    return {
        "terrain_summary": terrain_summary,
        "terrain_details": {
            "total_pixels": total_pixels,
            "total_valid_pixels": valid_pixels,
            "ignored_pixels": ignored_pixels,
            "valid_pixel_percentage": valid_pixel_pct,
            "class_pixel_counts": class_counts,
            "dominant_terrain": dominant,
            "secondary_terrain": secondary,
            "terrain_diversity_shannon_nats": round(shannon_nats, 6),
            "terrain_diversity_normalized": round(shannon_norm, 6),
            "boundary_pixel_count": boundary_px,
            "boundary_density_pct": boundary_density_pct,
        },
    }
