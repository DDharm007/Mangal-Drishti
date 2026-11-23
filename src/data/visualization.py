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
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    run_visualizations(args.project_root)


if __name__ == "__main__":
    main()
