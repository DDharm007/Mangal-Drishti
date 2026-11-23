"""Visualization suite for MANGAL DRISHTI Step 01.

Generates verified exploratory visualizations directly from acquired NASA/JPL
AI4MARS imagery, ground-truth masks, auxiliary masks (MXY/RNG-30M), and
validated statistics in `datasets/inspection/dataset_summary.json`.

Outputs saved under `datasets/inspection/`:
  1. 01_original_rover_image.png
  2. 02_ground_truth_segmentation.png
  3. 03_segmentation_overlay.png
  4. dataset_samples.png
  5. terrain_distribution.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("mangal_drishti.viz")

# Distinct, high-contrast scientific color palette for AI4MARS NAV classes
# 0: Soil, 1: Bedrock, 2: Sand, 3: Big Rock, 255: NULL/Unlabeled
CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (166, 97, 26),     # Soil - warm earthy brown
    1: (55, 126, 184),    # Bedrock - slate steel blue
    2: (245, 194, 66),    # Sand - bright ochre gold
    3: (228, 26, 28),     # Big Rock - high-alert crimson red
    255: (40, 44, 52),    # NULL / Unlabeled - dark charcoal
}

CLASS_NAMES: Dict[int, str] = {
    0: "Soil (0)",
    1: "Bedrock (1)",
    2: "Sand (2)",
    3: "Big Rock (3)",
    255: "NULL / Unlabeled (255)",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_grayscale_or_rgb(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB array for consistent plotting."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, :3]


def load_mask(path: Path) -> np.ndarray:
    """Load a single-channel uint8 segmentation mask without interpolation."""
    with Image.open(path) as msk:
        arr = np.asarray(msk)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map AI4MARS class IDs {0,1,2,3,255} to an RGB visualization image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = CLASS_COLORS_RGB[255]
    for cid, color in CLASS_COLORS_RGB.items():
        if cid == 255:
            continue
        rgb[mask == cid] = color
    return rgb


def overlay_mask_on_image(
    image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.50
) -> np.ndarray:
    """Alpha-blend labeled pixels onto the rover image while leaving 255 pixels raw."""
    out = image_rgb.astype(np.float32).copy()
    colored = colorize_mask(mask).astype(np.float32)
    valid = mask != 255
    out[valid] = (1.0 - alpha) * out[valid] + alpha * colored[valid]
    return np.clip(out, 0, 255).astype(np.uint8)


def _legend_handles(include_null: bool = True) -> List[mpatches.Patch]:
    keys = [0, 1, 2, 3, 255] if include_null else [0, 1, 2, 3]
    return [
        mpatches.Patch(
            color=np.array(CLASS_COLORS_RGB[k]) / 255.0, label=CLASS_NAMES[k]
        )
        for k in keys
    ]


def _format_mask_stats(mask: np.ndarray) -> str:
    hist = np.bincount(mask.ravel(), minlength=256)
    total = mask.size
    labeled = total - int(hist[255])
    if labeled == 0:
        return "100% Unlabeled (255)"
    parts = []
    for cid, short in ((0, "Soil"), (1, "Bedrock"), (2, "Sand"), (3, "BigRock")):
        if hist[cid] > 0:
            parts.append(f"{short}: {hist[cid] / labeled * 100:.1f}%")
    unlbl_pct = hist[255] / total * 100.0
    return f"Labeled ({100 - unlbl_pct:.1f}%): " + ", ".join(parts)


def generate_single_sample_visualizations(
    ai4mars_root: Path,
    inspection_dir: Path,
    hero_sample: Dict[str, object],
) -> List[Path]:
    """Generate Visualizations 1, 2, and 3 for a hero multi-class sample."""
    img_path = ai4mars_root / str(hero_sample["image_relpath"])
    mask_path = ai4mars_root / str(hero_sample["mask_relpath"])
    img_rgb = load_grayscale_or_rgb(img_path)
    mask = load_mask(mask_path)
    colored_mask = colorize_mask(mask)
    overlay = overlay_mask_on_image(img_rgb, mask, alpha=0.52)

    meta_subtitle = (
        f"ID: {hero_sample['id']} | Mission: {hero_sample['mission']} | "
        f"Camera: {hero_sample['camera']} | SCLK: {hero_sample['sclk']} | "
        f"Split: {hero_sample['split']} | Size: {img_rgb.shape[1]}x{img_rgb.shape[0]}"
    )

    # Visualization 1 — Original Rover Image
    vis1_path = inspection_dir / "01_original_rover_image.png"
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)
    ax.imshow(img_rgb)
    ax.set_title(
        f"Visualization 1 — Original Mars Rover NavCam Image\n{meta_subtitle}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(vis1_path, bbox_inches="tight")
    plt.close(fig)

    # Visualization 2 — Ground-Truth Segmentation (Raw vs Color-Mapped)
    vis2_path = inspection_dir / "02_ground_truth_segmentation.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=160)
    axes[0].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        "Raw Grayscale Mask (uint8: 0, 1, 2, 3, 255)\n(Classes 0–3 appear near-black in standard viewers)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title(
        f"Color-Mapped Ground-Truth Mask\n{_format_mask_stats(mask)}",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].axis("off")

    fig.suptitle(
        f"Visualization 2 — Ground-Truth Semantic Segmentation Mask\n{meta_subtitle}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.legend(
        handles=_legend_handles(include_null=True),
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(vis2_path, bbox_inches="tight")
    plt.close(fig)

    # Visualization 3 — Segmentation Overlay + Auxiliary Masks (if available)
    vis3_path = inspection_dir / "03_segmentation_overlay.png"
    mxy_rel = str(hero_sample.get("mxy_relpath", ""))
    rng_rel = str(hero_sample.get("rng_relpath", ""))
    has_aux = bool(mxy_rel and rng_rel and (ai4mars_root / mxy_rel).exists() and (ai4mars_root / rng_rel).exists())

    if has_aux:
        mxy_arr = load_mask(ai4mars_root / mxy_rel)
        rng_arr = load_mask(ai4mars_root / rng_rel)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=160)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Rover Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Semantic Segmentation Overlay (alpha=0.52)\n{_format_mask_stats(mask)}",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[1].axis("off")

        # Combine MXY (rover mask) and RNG-30M (>30m distance mask) onto grayscale image
        aux_vis = img_rgb.astype(np.float32).copy()
        rng_mask = rng_arr > 0
        mxy_mask = mxy_arr > 0
        aux_vis[rng_mask] = 0.45 * aux_vis[rng_mask] + 0.55 * np.array([138, 43, 226])  # Purple: >30m
        aux_vis[mxy_mask] = 0.35 * aux_vis[mxy_mask] + 0.65 * np.array([0, 200, 200])   # Cyan: Rover body
        axes[2].imshow(np.clip(aux_vis, 0, 255).astype(np.uint8))
        axes[2].set_title(
            "Official Exclusion Masks (MXY + RNG-30M)\nCyan: Rover Hardware | Purple: >30m Horizon",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), dpi=160)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Rover Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Semantic Segmentation Overlay\n{_format_mask_stats(mask)}",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[1].axis("off")

    fig.suptitle(
        f"Visualization 3 — Ground-Truth Segmentation Overlay & Mask Context\n{meta_subtitle}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.legend(
        handles=_legend_handles(include_null=True),
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(vis3_path, bbox_inches="tight")
    plt.close(fig)

    return [vis1_path, vis2_path, vis3_path]


def generate_sample_gallery(
    ai4mars_root: Path,
    inspection_dir: Path,
    representative_samples: Dict[str, Dict[str, object]],
) -> Path:
    """Generate Visualization 4 — Multi-row Dataset Sample Gallery."""
    ordered_keys = [
        ("multi_class_hero_msl", "Multi-Class Terrain (MSL Curiosity)"),
        ("big_rock_hazard_msl", "Big Rock Hazard (MSL Curiosity)"),
        ("sand_dune_msl", "Sand Dunes & Ripples (MSL Curiosity)"),
        ("bedrock_outcrop_msl", "Bedrock Outcrop (MSL Curiosity)"),
        ("mer_opportunity_test", "Gold Test min3 (MER-1 Opportunity)"),
        ("mer_spirit_test", "Gold Test min3 (MER-2 Spirit)"),
    ]
    rows = [(label, representative_samples[k]) for k, label in ordered_keys if k in representative_samples]
    n_rows = len(rows)
    if n_rows == 0:
        raise RuntimeError("No representative samples available to generate gallery.")

    fig, axes = plt.subplots(n_rows, 3, figsize=(14.5, 4.3 * n_rows), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original Rover Image", "Ground-Truth Mask", "Segmentation Overlay"]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for r_idx, (row_label, sample) in enumerate(rows):
        img_rgb = load_grayscale_or_rgb(ai4mars_root / str(sample["image_relpath"]))
        mask = load_mask(ai4mars_root / str(sample["mask_relpath"]))
        cmask = colorize_mask(mask)
        overlay = overlay_mask_on_image(img_rgb, mask, alpha=0.50)

        axes[r_idx, 0].imshow(img_rgb)
        axes[r_idx, 0].set_ylabel(
            f"{row_label}\n{sample['id']}",
            fontsize=9.5,
            fontweight="bold",
            rotation=90,
            labelpad=10,
        )
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])

        axes[r_idx, 1].imshow(cmask)
        axes[r_idx, 1].set_xlabel(_format_mask_stats(mask), fontsize=8.5)
        axes[r_idx, 1].set_xticks([])
        axes[r_idx, 1].set_yticks([])

        axes[r_idx, 2].imshow(overlay)
        axes[r_idx, 2].set_xlabel(
            f"Split: {sample['split']} | Camera: {sample['camera']}", fontsize=8.5
        )
        axes[r_idx, 2].set_xticks([])
        axes[r_idx, 2].set_yticks([])

    fig.suptitle(
        "Visualization 4 — AI4MARS Dataset Sample Gallery (Real Mars Rover Imagery & Ground-Truth Masks)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=_legend_handles(include_null=True),
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.985])
    out_path = inspection_dir / "dataset_samples.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_terrain_distribution_chart(
    inspection_dir: Path, summary: Dict[str, object]
) -> Path:
    """Generate Visualization 5 — Terrain Class Distribution & Split Analytics."""
    overall = summary["overall_terrain_distribution"]
    splits = summary["split_terrain_distributions"]
    expert_comp = summary["expert_test_consensus_comparison"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), dpi=160)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel A: Labeled Pixel Distribution (excluding 255)
    ax_a = axes[0, 0]
    cids = ["0", "1", "2", "3"]
    cnames = ["Soil (0)", "Bedrock (1)", "Sand (2)", "Big Rock (3)"]
    colors = [np.array(CLASS_COLORS_RGB[int(c)]) / 255.0 for c in cids]
    pcts = [overall["classes"][c]["pct_of_labeled_pixels"] for c in cids]
    px_millions = [overall["classes"][c]["pixel_count"] / 1e6 for c in cids]

    bars = ax_a.bar(cnames, pcts, color=colors, edgecolor="black", linewidth=0.8)
    ax_a.set_title(
        f"A. Valid Labeled Pixel Distribution ({overall['num_images']} Images)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_a.set_ylabel("Share of Labeled Pixels (%)", fontsize=10)
    ax_a.set_ylim(0, max(pcts) * 1.22)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, pct, px_m in zip(bars, pcts, px_millions):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{pct:.2f}%\n({px_m:.1f}M px)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    imbalance = overall["max_to_min_class_imbalance_ratio"]
    ax_a.text(
        0.97,
        0.88,
        f"Max/Min Imbalance Ratio:\n{imbalance}:1 (Soil vs Big Rock)",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff4e5", edgecolor="#d97706"),
    )

    # Panel B: Stacked Pixel Budget Across Splits (including 255 NULL)
    ax_b = axes[0, 1]
    split_keys = ["train", "val", "test_msl", "test_mer"]
    split_labels = [
        f"Train\n(n={splits[k]['num_images']})" if k == "train" else
        f"Val\n(n={splits[k]['num_images']})" if k == "val" else
        f"MSL Test\n(n={splits[k]['num_images']})" if k == "test_msl" else
        f"MER Test\n(n={splits[k]['num_images']})"
        for k in split_keys
    ]
    all_cids = ["0", "1", "2", "3", "255"]
    bottoms = np.zeros(len(split_keys))
    for cid in all_cids:
        vals = np.array([splits[s]["classes"][cid]["pct_of_all_pixels"] for s in split_keys])
        col = np.array(CLASS_COLORS_RGB[int(cid)]) / 255.0
        lbl = CLASS_NAMES[int(cid)]
        bars_b = ax_b.bar(
            split_labels,
            vals,
            bottom=bottoms,
            color=col,
            edgecolor="white",
            linewidth=0.6,
            label=lbl,
        )
        for idx, v in enumerate(vals):
            if v >= 5.5:
                ax_b.text(
                    idx,
                    bottoms[idx] + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if int(cid) in (0, 1, 3, 255) else "black",
                    fontweight="bold",
                )
        bottoms += vals
    ax_b.set_title(
        "B. Total Pixel Breakdown by Split (Including Unlabeled/NULL 255)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_b.set_ylabel("Percentage of All Image Pixels (%)", fontsize=10)
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Panel C: Image-Level Class Presence (%) across splits
    ax_c = axes[1, 0]
    x = np.arange(len(cids))
    width = 0.2
    split_colors = ["#2b5c8f", "#4a90d9", "#e07a5f", "#3d9970"]
    for s_idx, (sk, slbl, sc) in enumerate(
        zip(split_keys, ["Train", "Val", "MSL Gold Test (min3)", "MER Gold Test (min3)"], split_colors)
    ):
        pres = [splits[sk]["classes"][c]["pct_images_with_class"] for c in cids]
        ax_c.bar(x + (s_idx - 1.5) * width, pres, width, label=slbl, color=sc, edgecolor="black", linewidth=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cnames)
    ax_c.set_ylim(0, 112)
    ax_c.set_ylabel("Images Containing Class (%)", fontsize=10)
    ax_c.set_title("C. Image-Level Terrain Class Occurrence by Split", fontsize=11.5, fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.legend(fontsize=8.5)

    # Panel D: Expert Test Set Consensus Threshold Impact (min1 vs min2 vs min3)
    ax_d = axes[1, 1]
    comp_keys = [
        "test_msl_min1_100agree",
        "test_msl_min2_100agree",
        "test_msl_min3_100agree",
        "test_mer_min1_100agree",
        "test_mer_min2_100agree",
        "test_mer_min3_100agree",
    ]
    comp_labels = [
        "MSL\nmin1",
        "MSL\nmin2",
        "MSL\nmin3",
        "MER\nmin1",
        "MER\nmin2",
        "MER\nmin3",
    ]
    labeled_cov = [expert_comp[k]["labeled_pct"] for k in comp_keys if k in expert_comp]
    valid_labels = [lbl for k, lbl in zip(comp_keys, comp_labels) if k in expert_comp]
    bars_d = ax_d.bar(
        valid_labels,
        labeled_cov,
        color=["#4c78a8", "#3b628c", "#264363", "#e45756", "#c43c3b", "#962423"][: len(labeled_cov)],
        edgecolor="black",
        linewidth=0.7,
    )
    ax_d.set_title(
        "D. Expert Gold Test Coverage vs. Agreement Threshold (100% Agree)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_d.set_ylabel("Valid Labeled Pixel Coverage (%)", fontsize=10)
    ax_d.set_ylim(0, max(labeled_cov) * 1.25 if labeled_cov else 100)
    ax_d.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars_d, labeled_cov):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        "Visualization 5 — AI4MARS Verified Terrain Class Distribution & Consensus Analysis (MANGAL DRISHTI)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    out_path = inspection_dir / "terrain_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_diverse_preprocessing_samples(
    split_manifest: Dict[str, object], num_samples: int = 12
) -> List[Dict[str, object]]:
    """Select diverse representative samples across splits, missions, and terrain classes."""
    train_recs = list(split_manifest["splits"]["train"])
    val_recs = list(split_manifest["splits"]["val"])
    test_recs = list(split_manifest["splits"]["test"])

    def score_class(rec: Dict[str, object], cid: str) -> float:
        pc = rec.get("pixel_counts", {})
        tot = max(sum(pc.get(str(k), 0) for k in range(4)), 1)
        return pc.get(cid, 0) / tot

    def diversity_score(rec: Dict[str, object]) -> Tuple[int, float]:
        pc = rec.get("pixel_counts", {})
        n_cls = sum(1 for k in ("0", "1", "2", "3") if pc.get(k, 0) > 400)
        return (n_cls, float(rec.get("labeled_pixels_raw", 0)))

    chosen: List[Dict[str, object]] = []
    seen_ids: set[str] = set()

    def add_pick(candidates: List[Dict[str, object]], key_fn) -> None:
        filtered = [c for c in candidates if c["id"] not in seen_ids]
        if not filtered:
            return
        best = max(filtered, key=key_fn)
        seen_ids.add(best["id"])
        chosen.append(best)

    # Curate rich diverse examples across train, val, and official expert test sets
    add_pick(train_recs, diversity_score)
    add_pick(train_recs, lambda r: (score_class(r, "3"), r.get("labeled_pixels_raw", 0)))
    add_pick(train_recs, lambda r: (score_class(r, "2"), r.get("labeled_pixels_raw", 0)))
    add_pick(train_recs, lambda r: (score_class(r, "1"), r.get("labeled_pixels_raw", 0)))
    add_pick(train_recs, lambda r: (score_class(r, "0"), r.get("labeled_pixels_raw", 0)))
    add_pick(train_recs, lambda r: (score_class(r, "3") * 3 + score_class(r, "2"), r.get("labeled_pixels_raw", 0)))

    add_pick(val_recs, diversity_score)
    add_pick(val_recs, lambda r: (score_class(r, "3"), r.get("labeled_pixels_raw", 0)))
    add_pick(val_recs, lambda r: (score_class(r, "2"), r.get("labeled_pixels_raw", 0)))

    msl_test = [r for r in test_recs if r.get("official_split") == "test_msl"]
    mer_opp = [r for r in test_recs if "Opportunity" in str(r.get("mission", ""))]
    mer_spr = [r for r in test_recs if "Spirit" in str(r.get("mission", ""))]
    add_pick(msl_test, diversity_score)
    add_pick(mer_opp, diversity_score)
    add_pick(mer_spr, diversity_score)

    # Top up if fewer than num_samples
    for rec in train_recs + val_recs + test_recs:
        if len(chosen) >= num_samples:
            break
        if rec["id"] not in seen_ids:
            seen_ids.add(rec["id"])
            chosen.append(rec)

    return chosen[:num_samples]


def run_preprocessing_visualizations(
    config_path: Path | str = Path("configs/data_config.yaml"),
    project_root: Optional[Path] = None,
) -> List[Path]:
    """Generate Step 2 visual validation comparisons and class distribution charts."""
    from src.data.config_utils import get_project_root, load_config, resolve_path
    from src.data.transforms import SegmentationTransformPipeline

    root = project_root or get_project_root()
    cfg = load_config(config_path)

    raw_root = resolve_path(cfg["paths"]["raw_ai4mars_dir"], root)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"], root)
    out_dir = resolve_path(cfg["paths"]["preprocessing_samples_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = json.loads(
        resolve_path(cfg["paths"]["split_manifest_file"], root).read_text(encoding="utf-8")
    )
    prep_cfg = json.loads(
        resolve_path(cfg["paths"]["preprocessing_config_file"], root).read_text(encoding="utf-8")
    )
    class_weights_data = json.loads(
        resolve_path(cfg["paths"]["class_weights_file"], root).read_text(encoding="utf-8")
    )

    num_samples = int(cfg.get("visualization", {}).get("num_samples", 12))
    alpha = float(cfg.get("visualization", {}).get("alpha_overlay", 0.50))
    samples = _select_diverse_preprocessing_samples(split_manifest, num_samples=num_samples)

    # Instantiate training-mode pipeline strictly for demonstrating Step 2 augmentation
    aug_pipeline = SegmentationTransformPipeline(
        cfg=cfg,
        split="train",
        mean=prep_cfg["normalization"]["active_mean"],
        std=prep_cfg["normalization"]["active_std"],
        seed=int(cfg["project"]["random_seed"]) + 101,
    )

    generated: List[Path] = []
    gallery_rows = []

    for idx, rec in enumerate(samples, start=1):
        raw_img = load_grayscale_or_rgb(raw_root / str(rec["image_relpath"]))
        raw_msk = load_mask(raw_root / str(rec["mask_relpath"]))
        proc_img = load_grayscale_or_rgb(processed_dir / str(rec["processed_image_relpath"]))
        proc_msk = load_mask(processed_dir / str(rec["processed_mask_relpath"]))

        raw_msk_color = colorize_mask(raw_msk)
        proc_msk_color = colorize_mask(proc_msk)
        proc_overlay = overlay_mask_on_image(proc_img, proc_msk, alpha=alpha)

        # Apply deterministic sample-indexed RNG so augmented visual is reproducible
        sample_rng = np.random.default_rng(int(cfg["project"]["random_seed"]) + idx * 17)
        aug_img, aug_msk = aug_pipeline.apply_numpy(proc_img, proc_msk, rng=sample_rng)
        aug_msk_color = colorize_mask(aug_msk)

        if idx <= 6:
            gallery_rows.append(
                (rec, raw_img, proc_img, raw_msk_color, proc_msk_color, proc_overlay, aug_img, aug_msk_color)
            )

        fig, axes = plt.subplots(1, 7, figsize=(24.5, 4.3), dpi=140)
        panels = [
            (raw_img, f"1. Original Image\n({raw_img.shape[1]}x{raw_img.shape[0]})"),
            (proc_img, f"2. Resized Image\n({proc_img.shape[1]}x{proc_img.shape[0]})"),
            (raw_msk_color, f"3. Original Mask\n({raw_msk.shape[1]}x{raw_msk.shape[0]})"),
            (proc_msk_color, f"4. Encoded Mask\nIDs: {sorted(int(v) for v in np.unique(proc_msk))}"),
            (proc_overlay, f"5. Resized Overlay\n({_format_mask_stats(proc_msk)})"),
            (aug_img, "6. Augmented Image\n(Geom + Photometric)"),
            (aug_msk_color, f"7. Augmented Mask\nIDs: {sorted(int(v) for v in np.unique(aug_msk))}"),
        ]
        for ax, (arr, title) in zip(axes, panels):
            ax.imshow(arr)
            ax.set_title(title, fontsize=8.8, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            f"Preprocessing Sample {idx:02d} — {rec['id']} | Split: {rec['split']} | "
            f"Mission: {rec['mission']} | Consensus: {rec.get('consensus_tier_used', '')}",
            fontsize=11.5,
            fontweight="bold",
            y=1.02,
        )
        fig.legend(
            handles=_legend_handles(include_null=True),
            loc="lower center",
            ncol=5,
            frameon=True,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.05),
        )
        fig.tight_layout()
        out_file = out_dir / f"sample_{idx:02d}_{rec['id']}.png"
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_file)

    # Combined 6-row summary gallery for rapid visual inspection
    fig, axes = plt.subplots(len(gallery_rows), 7, figsize=(23.5, 3.6 * len(gallery_rows)), dpi=140)
    col_headers = [
        "1. Original Image",
        "2. Resized Image",
        "3. Original Mask",
        "4. Encoded Mask",
        "5. Resized Overlay",
        "6. Augmented Image",
        "7. Augmented Mask",
    ]
    for c_i, ch in enumerate(col_headers):
        axes[0, c_i].set_title(ch, fontsize=10.5, fontweight="bold", pad=8)

    for r_i, (rec, r_img, p_img, r_mc, p_mc, p_ov, a_img, a_mc) in enumerate(gallery_rows):
        row_imgs = [r_img, p_img, r_mc, p_mc, p_ov, a_img, a_mc]
        for c_i, arr in enumerate(row_imgs):
            axes[r_i, c_i].imshow(arr)
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])
        axes[r_i, 0].set_ylabel(
            f"{rec['split'].upper()}\n{rec['id'][:18]}",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.suptitle(
        "MANGAL DRISHTI Step 2 — Preprocessing, Mask Encoding & Augmentation Visual Validation Gallery",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=_legend_handles(include_null=True),
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.tight_layout(rect=[0.015, 0.025, 1.0, 0.985])
    gallery_path = out_dir / "preprocessing_gallery.png"
    fig.savefig(gallery_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(gallery_path)

    # Preprocessed Split Class Distribution & Class Weights Chart
    split_stats = class_weights_data["split_pixel_distributions"]
    weights_map = class_weights_data["weighting_methods"]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=160)

    cids = ["0", "1", "2", "3"]
    cnames = ["Soil (0)", "Bedrock (1)", "Sand (2)", "Big Rock (3)"]
    x = np.arange(len(cids))
    w = 0.25
    for s_i, (sk, slbl, col) in enumerate(
        zip(["train", "val", "test"], ["Train", "Val", "Test (Expert)"], ["#2b5c8f", "#3d9970", "#e07a5f"])
    ):
        vals = [split_stats[sk]["classes"][c]["pct_of_valid_pixels"] for c in cids]
        bars = axes[0].bar(x + (s_i - 1) * w, vals, w, label=f"{slbl} (n={split_stats[sk]['num_images']})", color=col, edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2.0, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cnames)
    axes[0].set_ylim(0, 62)
    axes[0].set_ylabel("Share of Valid Pixels (%)")
    axes[0].set_title("A. Valid Class Balance Across SCLK Splits", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8.5)

    # Panel B: Valid vs Ignored (255) pixels per processed split
    s_keys = ["train", "val", "test"]
    s_lbls = [f"{k.title()}\n(n={split_stats[k]['num_images']})" for k in s_keys]
    valid_pcts = [split_stats[k]["valid_pct"] for k in s_keys]
    ign_pcts = [split_stats[k]["ignored_pct"] for k in s_keys]
    axes[1].bar(s_lbls, valid_pcts, label="Valid Terrain Labels (0–3)", color="#3b82f6", edgecolor="black", linewidth=0.6)
    axes[1].bar(s_lbls, ign_pcts, bottom=valid_pcts, label="Ignored / NULL (255)", color="#374151", edgecolor="black", linewidth=0.6)
    for idx_s, (vp, ip) in enumerate(zip(valid_pcts, ign_pcts)):
        axes[1].text(idx_s, vp / 2.0, f"{vp:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        axes[1].text(idx_s, vp + ip / 2.0, f"{ip:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Share of All 512x512 Pixels (%)")
    axes[1].set_title("B. Valid vs. Ignored (255) Pixel Budget", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.5)

    # Panel C: Computed Train-Only Class Weights Comparison
    methods = [
        ("median_frequency_capped", "Median Freq (Capped)", "#d97706"),
        ("inverse_frequency_capped", "Inverse Freq (Capped)", "#dc2626"),
        ("effective_num_samples", "Effective Num Samples", "#059669"),
        ("swan2021_one_minus_composition", "Swan2021 (1 - comp)", "#6366f1"),
    ]
    w_bar = 0.2
    for m_i, (mk, mlbl, mc) in enumerate(methods):
        w_vals = [weights_map[mk][c] for c in cids]
        axes[2].bar(x + (m_i - 1.5) * w_bar, w_vals, w_bar, label=mlbl, color=mc, edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cnames)
    axes[2].set_ylabel("Loss Weight Multiplier")
    axes[2].set_title("C. Train-Only Loss Weighting Schemes", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "MANGAL DRISHTI Step 2 — Processed Split Class Distribution & Training Loss Weights",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    dist_path = out_dir / "preprocessed_class_distribution.png"
    fig.savefig(dist_path, bbox_inches="tight")
    plt.close(fig)
    generated.append(dist_path)

    for p in generated:
        LOGGER.info("Generated Step 2 visualization: %s", p)
    return generated


def run_visualizations(project_root: Path) -> List[Path]:
    """Generate all 5 required visualizations from actual acquired data."""
    ai4mars_root = project_root / "datasets" / "raw" / "ai4mars"
    inspection_dir = project_root / "datasets" / "inspection"
    summary_path = inspection_dir / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run inspect_dataset.py before visualization.py."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rep_samples = summary["representative_samples"]
    hero = rep_samples.get("multi_class_hero_msl") or next(iter(rep_samples.values()))

    paths: List[Path] = []
    paths.extend(generate_single_sample_visualizations(ai4mars_root, inspection_dir, hero))
    paths.append(generate_sample_gallery(ai4mars_root, inspection_dir, rep_samples))
    paths.append(generate_terrain_distribution_chart(inspection_dir, summary))

    for p in paths:
        LOGGER.info("Generated visualization: %s", p)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual dataset exploration plots for MANGAL DRISHTI."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--mode",
        choices=["step1", "preprocessing"],
        default="step1",
        help="Visualization suite to generate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="YAML config path used when --mode preprocessing.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    if args.mode == "preprocessing":
        run_preprocessing_visualizations(args.config, project_root=args.project_root)
    else:
        run_visualizations(args.project_root)


if __name__ == "__main__":
    main()
