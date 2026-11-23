"""Visualization suite (9 outputs) and JSON writer for MANGAL DRISHTI Terrain Intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.intelligence.terrain_statistics import (
    CLASS_ID_TO_DISPLAY,
    CLASS_ID_TO_KEY,
    TERRAIN_PALETTE_RGB,
    colorize_terrain_mask,
)


def _save_rgb_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _render_region_boundaries_image(
    overlay_rgb: np.ndarray,
    regions: List[Dict[str, Any]],
    contours_by_class: List[Tuple[int, np.ndarray]],
) -> np.ndarray:
    canvas = overlay_rgb.copy()
    for cid, contour in contours_by_class:
        color_rgb = TERRAIN_PALETTE_RGB.get(cid, (255, 255, 255))
        # Dark outline followed by colored boundary for high legibility
        cv2.drawContours(canvas, [contour], -1, (15, 15, 15), thickness=3)
        cv2.drawContours(canvas, [contour], -1, color_rgb, thickness=1)

    for reg in regions[:12]:
        cx, cy = int(round(reg["centroid"][0])), int(round(reg["centroid"][1]))
        label = f"{reg['region_id']} ({reg['terrain_class'][:3]})"
        cv2.putText(canvas, label, (max(4, cx - 25), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (max(4, cx - 25), max(14, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _render_big_rock_candidates_image(
    rgb: np.ndarray,
    pred_mask: np.ndarray,
    candidates: List[Dict[str, Any]],
) -> np.ndarray:
    canvas = rgb.copy()
    br_mask = pred_mask == 3
    if np.any(br_mask):
        tint = canvas.copy()
        tint[br_mask] = TERRAIN_PALETTE_RGB[3]
        canvas = np.clip(0.65 * canvas.astype(np.float32) + 0.35 * tint.astype(np.float32), 0, 255).astype(np.uint8)

    for cand in candidates:
        x, y, bw, bh = cand["bbox"]
        cx, cy = int(round(cand["centroid"][0])), int(round(cand["centroid"][1]))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 64, 64), 2)
        cv2.circle(canvas, (cx, cy), 4, (255, 235, 59), -1)
        tag = f"{cand['candidate_id']} ({cand['confidence']:.2f})"
        ty = y - 6 if y > 18 else min(canvas.shape[0] - 6, y + bh + 14)
        cv2.putText(canvas, tag, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(canvas, tag, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 160), 1, cv2.LINE_AA)
    return canvas


def _render_heatmap_rgb(values: np.ndarray, vmin: float, vmax: float, cmap_name: str, ignore_mask: np.ndarray) -> np.ndarray:
    norm = np.clip((values.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    rgb[ignore_mask] = TERRAIN_PALETTE_RGB[255]
    return rgb


def _save_terrain_distribution_chart(
    terrain_summary: Dict[str, float],
    confidence_summary: Dict[str, Any],
    uncertainty_summary: Dict[str, Any],
    traversability: Dict[str, Any],
    out_path: Path,
) -> np.ndarray:
    keys = ["soil", "bedrock", "sand", "big_rock"]
    labels = [CLASS_ID_TO_DISPLAY[i] for i in range(4)]
    vals = [float(terrain_summary[k]) for k in keys]
    colors = [np.array(TERRAIN_PALETTE_RGB[i]) / 255.0 for i in range(4)]

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=130)
    y_pos = np.arange(len(keys))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#222222", height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 108.0)
    ax.set_xlabel("Valid Terrain Coverage (%)", fontsize=9.5, fontweight="bold")
    ax.set_title("Predicted Terrain Distribution & KPIs", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, v in zip(bars, vals):
        ax.text(min(v + 1.5, 92.0), bar.get_y() + bar.get_height() / 2.0, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    kpi_text = (
        f"Mean Conf: {confidence_summary['mean_confidence']:.3f} (Uncalibrated)  |  "
        f"Mean Entropy: {uncertainty_summary['mean_entropy']:.3f} nats\n"
        f"Visual Assessment (Experimental): {traversability['assessment']} "
        f"(idx={traversability['experimental_visual_concern_index']:.3f})\n"
        f"Physical Safety Verified: {traversability['physical_safety_verified']}"
    )
    ax.text(0.01, -0.27, kpi_text, transform=ax.transAxes, fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f4f6f8", edgecolor="#cccccc"))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")

    fig.canvas.draw()
    chart_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return chart_rgb


def generate_intelligence_visualizations(
    output_dir: Path,
    rgb: np.ndarray,
    pred_mask: np.ndarray,
    conf_map: np.ndarray,
    norm_entropy_map: np.ndarray,
    region_meta: Dict[str, Any],
    report: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate all 9 intelligence visualizations and return saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = report["analysis"]
    ignore_bool = pred_mask == 255

    seg_rgb = colorize_terrain_mask(pred_mask)
    overlay_rgb = np.clip(0.52 * rgb.astype(np.float32) + 0.48 * seg_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    region_rgb = _render_region_boundaries_image(
        overlay_rgb, analysis["regions"], region_meta.get("contours_by_class", [])
    )
    obstacle_rgb = _render_big_rock_candidates_image(rgb, pred_mask, analysis["obstacle_candidates"])
    conf_rgb = _render_heatmap_rgb(conf_map, 0.25, 1.0, "viridis", ignore_bool)
    unc_rgb = _render_heatmap_rgb(norm_entropy_map, 0.0, 1.0, "inferno", ignore_bool)

    paths: Dict[str, Path] = {
        "original_image": output_dir / "01_original_image.png",
        "segmentation_mask": output_dir / "02_segmentation_mask.png",
        "terrain_overlay": output_dir / "03_terrain_overlay.png",
        "region_boundaries": output_dir / "04_region_boundaries.png",
        "big_rock_candidates": output_dir / "05_big_rock_candidates.png",
        "confidence_heatmap": output_dir / "06_confidence_heatmap.png",
        "uncertainty_heatmap": output_dir / "07_uncertainty_heatmap.png",
        "terrain_distribution_chart": output_dir / "08_terrain_distribution.png",
        "intelligence_summary_panel": output_dir / "09_intelligence_summary_panel.png",
    }

    _save_rgb_image(rgb, paths["original_image"])
    _save_rgb_image(seg_rgb, paths["segmentation_mask"])
    _save_rgb_image(overlay_rgb, paths["terrain_overlay"])
    _save_rgb_image(region_rgb, paths["region_boundaries"])
    _save_rgb_image(obstacle_rgb, paths["big_rock_candidates"])
    _save_rgb_image(conf_rgb, paths["confidence_heatmap"])
    _save_rgb_image(unc_rgb, paths["uncertainty_heatmap"])

    chart_rgb = _save_terrain_distribution_chart(
        analysis["terrain_summary"],
        analysis["confidence"],
        analysis["uncertainty"],
        analysis["traversability"],
        paths["terrain_distribution_chart"],
    )

    # Render combined 3x3 Final Intelligence Summary Panel
    fig, axes = plt.subplots(3, 3, figsize=(17, 15.5), dpi=135)
    panels = [
        (rgb, "1. Original Rover Image"),
        (seg_rgb, f"2. Segmentation Mask (Dom: {analysis['terrain_details']['dominant_terrain']})"),
        (overlay_rgb, "3. Colored Terrain Overlay"),
        (region_rgb, f"4. Region Boundaries (n={len(analysis['regions'])})"),
        (obstacle_rgb, f"5. Big Rock Candidates (n={len(analysis['obstacle_candidates'])})"),
        (conf_rgb, f"6. Softmax Confidence (Mean: {analysis['confidence']['mean_confidence']:.3f})"),
        (unc_rgb, f"7. Normalized Entropy (Mean: {analysis['uncertainty']['mean_normalized_entropy']:.3f})"),
        (chart_rgb, "8. Terrain Distribution & Visual Concern"),
    ]
    for idx, (img_data, title) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img_data)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Panel 9: Structured Diagnostic Summary Card
    ax_card = axes[2, 2]
    ax_card.axis("off")
    ts = analysis["terrain_summary"]
    trav = analysis["traversability"]
    card_lines = [
        f"MODEL: {report['model']} ({report['device']})",
        f"VALID PIXELS: {analysis['terrain_details']['valid_pixel_percentage']:.1f}%",
        f"TERRAIN: Soil {ts['soil']:.1f}% | Bed {ts['bedrock']:.1f}%",
        f"         Sand {ts['sand']:.1f}% | Rock {ts['big_rock']:.1f}%",
        f"REGIONS (>={region_meta['min_region_area_px']}px): {len(analysis['regions'])}",
        f"OBSTACLE CANDIDATES (2D): {len(analysis['obstacle_candidates'])}",
        f"CONFIDENCE: {analysis['confidence']['mean_confidence']:.3f} (Uncalibrated)",
        f"ENTROPY: {analysis['uncertainty']['mean_entropy']:.3f} nats",
        f"EXPERIMENTAL CONCERN: {trav['assessment']}",
        f"  Index: {trav['experimental_visual_concern_index']:.3f}",
        f"PHYSICAL SAFETY VERIFIED: {trav['physical_safety_verified']}",
        "MISSING: 3D depth, slope, wheel slip",
    ]
    ax_card.set_title("9. Intelligence Summary Card", fontsize=10, fontweight="bold")
    ax_card.text(
        0.04, 0.92, "\n".join(card_lines),
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#444444", linewidth=1.2),
    )

    handles = [
        mpatches.Patch(color=np.array(TERRAIN_PALETTE_RGB[i]) / 255.0, label=f"{CLASS_ID_TO_DISPLAY[i]} ({i})")
        for i in range(4)
    ]
    handles.append(mpatches.Patch(color=np.array(TERRAIN_PALETTE_RGB[255]) / 255.0, label="Ignored (255)"))

    stem = report["image_metadata"].get("image_id", "rover_image")
    fig.suptitle(f"MANGAL DRISHTI — Terrain Intelligence Summary ({stem})", fontsize=13, fontweight="bold", y=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.96])
    fig.savefig(paths["intelligence_summary_panel"], bbox_inches="tight")
    plt.close(fig)

    out_strs: Dict[str, str] = {}
    for k, p in paths.items():
        out_strs[k] = str(p.relative_to(project_root)).replace("\\", "/") if (project_root and p.is_relative_to(project_root)) else str(p)
    return out_strs


def save_intelligence_report(report: Dict[str, Any], path: Path) -> None:
    """Persist structured JSON report with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
