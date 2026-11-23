"""Canonical terrain class definitions and vectorized pixel/spatial utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

CLASS_ID_TO_KEY: Dict[int, str] = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big_rock",
}

CLASS_ID_TO_DISPLAY: Dict[int, str] = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
}

# Exact Step 2 class colors from datasets/metadata/class_mapping.json
TERRAIN_PALETTE_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (166, 97, 26),
    1: (55, 126, 184),
    2: (245, 194, 66),
    3: (228, 26, 28),
    255: (40, 44, 52),
}


def get_valid_mask(pred_mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Return boolean mask of valid terrain pixels strictly excluding ignore_index."""
    return (pred_mask != ignore_index) & (pred_mask >= 0) & (pred_mask < 4)


def compute_shannon_diversity(proportions: np.ndarray) -> Tuple[float, float]:
    """Compute Shannon diversity H (nats) and normalized diversity in [0, 1]."""
    nonzero = proportions[proportions > 0]
    if nonzero.size == 0:
        return 0.0, 0.0
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = float(np.log(4.0))
    norm_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return entropy, norm_entropy


def compute_boundary_mask(pred_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Identify valid pixels adjacent (4-connected) to a different valid terrain class."""
    h, w = pred_mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    if h > 1:
        diff_v = valid_mask[:-1, :] & valid_mask[1:, :] & (pred_mask[:-1, :] != pred_mask[1:, :])
        boundary[:-1, :] |= diff_v
        boundary[1:, :] |= diff_v
    if w > 1:
        diff_h = valid_mask[:, :-1] & valid_mask[:, 1:] & (pred_mask[:, :-1] != pred_mask[:, 1:])
        boundary[:, :-1] |= diff_h
        boundary[:, 1:] |= diff_h
    return boundary


def classify_relative_position(cx: float, cy: float, width: int, height: int) -> str:
    """Classify a centroid (cx, cy) into a 3x3 relative image grid position."""
    nx = cx / max(float(width), 1.0)
    ny = cy / max(float(height), 1.0)
    v_label = "top" if ny < (1.0 / 3.0) else ("middle" if ny < (2.0 / 3.0) else "bottom")
    h_label = "left" if nx < (1.0 / 3.0) else ("center" if nx < (2.0 / 3.0) else "right")
    return f"{v_label}_{h_label}"


def colorize_terrain_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Map class IDs {0, 1, 2, 3, 255} to consistent Step 2 RGB colors."""
    h, w = pred_mask.shape
    rgb = np.full((h, w, 3), TERRAIN_PALETTE_RGB[255], dtype=np.uint8)
    for cid, color in TERRAIN_PALETTE_RGB.items():
        if cid != 255:
            rgb[pred_mask == cid] = color
    return rgb
