"""Class imbalance analysis and loss weight computation for MANGAL DRISHTI.

Calculates per-class pixel frequencies strictly on processed datasets and computes
balanced training loss weights using ONLY the `train` split (never `val` or `test`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.config_utils import get_project_root, load_config, resolve_path

LOGGER = logging.getLogger("mangal_drishti.class_weights")


def _analyze_split_pixels(
    processed_dir: Path,
    split_records: list[Dict[str, Any]],
    class_names: Dict[int, str],
    ignore_index: int,
) -> Dict[str, Any]:
    """Compute exact pixel distributions and image presence over processed masks."""
    counts = np.zeros(256, dtype=np.int64)
    presence = {cid: 0 for cid in class_names}

    for rec in tqdm(split_records, desc=f"Analyzing {split_records[0]['split']} masks", unit="mask"):
        mask_path = processed_dir / rec["processed_mask_relpath"]
        arr = np.asarray(Image.open(mask_path))
        hist = np.bincount(arr.ravel(), minlength=256)
        counts += hist
        for cid in class_names:
            if hist[cid] > 0:
                presence[cid] += 1

    total_pixels = int(counts.sum())
    ignored_pixels = int(counts[ignore_index])
    valid_pixels = total_pixels - ignored_pixels
    num_images = len(split_records)

    per_class: Dict[str, Any] = {}
    for cid, cname in sorted(class_names.items()):
        c_px = int(counts[cid])
        per_class[str(cid)] = {
            "class_id": cid,
            "class_name": cname,
            "pixel_count": c_px,
            "pct_of_all_pixels": round(c_px / total_pixels * 100.0, 4) if total_pixels else 0.0,
            "pct_of_valid_pixels": round(c_px / valid_pixels * 100.0, 4) if valid_pixels else 0.0,
            "images_with_class": presence[cid],
            "pct_images_with_class": round(presence[cid] / num_images * 100.0, 2) if num_images else 0.0,
        }

    return {
        "num_images": num_images,
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,
        "valid_pct": round(valid_pixels / total_pixels * 100.0, 4) if total_pixels else 0.0,
        "ignored_pct": round(ignored_pixels / total_pixels * 100.0, 4) if total_pixels else 0.0,
        "classes": per_class,
    }


def compute_and_save_class_weights(
    cfg: Dict[str, Any], project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Compute pixel distributions across splits and training-only class weights."""
    root = project_root or get_project_root()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"], root)
    split_manifest = json.loads(
        resolve_path(cfg["paths"]["split_manifest_file"], root).read_text(encoding="utf-8")
    )
    class_mapping = json.loads(
        resolve_path(cfg["paths"]["class_mapping_file"], root).read_text(encoding="utf-8")
    )

    ignore_index = int(cfg["mask"]["ignore_index"])
    class_names = {
        int(c["training_id"]): c["class_name"]
        for c in class_mapping["classes"]
        if c["trainable"]
    }
    ordered_ids = sorted(class_names.keys())

    split_stats: Dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split_stats[split_name] = _analyze_split_pixels(
            processed_dir,
            split_manifest["splits"][split_name],
            class_names,
            ignore_index,
        )

    # STRICT RULE: Compute training loss weights using ONLY `train` split counts
    train_stats = split_stats["train"]
    train_counts = np.array(
        [train_stats["classes"][str(cid)]["pixel_count"] for cid in ordered_ids],
        dtype=np.float64,
    )
    train_freqs = train_counts / np.sum(train_counts)

    # 1. Capped & normalized inverse frequency
    eps = float(cfg["class_weights"]["smoothing_epsilon"])
    max_cap = float(cfg["class_weights"]["inverse_freq_max_cap"])
    raw_inv = 1.0 / (train_freqs + eps)
    inv_norm = raw_inv / np.mean(raw_inv)
    inv_capped = np.clip(inv_norm, a_min=None, a_max=max_cap)
    inv_capped = inv_capped / np.mean(inv_capped)

    # 2. Median frequency balancing (Eigen & Fergus, 2015)
    median_freq = np.median(train_freqs)
    med_bal = median_freq / (train_freqs + eps)
    med_bal_capped = np.clip(med_bal, a_min=None, a_max=max_cap)

    # 3. Effective number of samples (Cui et al., CVPR 2019)
    # Using effective number at superpixel/patch scale (n_c / 1024) or raw pixel count
    beta = float(cfg["class_weights"]["effective_num_beta"])
    effective_n = 1.0 - np.power(beta, np.maximum(train_counts / 1024.0, 1.0))
    eff_weights = (1.0 - beta) / np.maximum(effective_n, 1e-12)
    eff_weights = eff_weights / np.mean(eff_weights)
    eff_weights = np.clip(eff_weights, a_min=None, a_max=max_cap)
    eff_weights = eff_weights / np.mean(eff_weights)

    # 4. Swan et al. (2021) AI4MARS paper baseline: 1.0 - class_composition
    swan_weights = 1.0 - train_freqs
    swan_weights = swan_weights / np.mean(swan_weights)

    def _to_dict(arr: np.ndarray) -> Dict[str, float]:
        return {str(cid): round(float(arr[i]), 6) for i, cid in enumerate(ordered_ids)}

    max_px = float(train_counts.max())
    min_px = float(train_counts.min())

    output = {
        "dataset": "NASA/JPL AI4MARS v0.6 (Preprocessed)",
        "weight_calculation_split": "train_only",
        "ignore_index": ignore_index,
        "class_names": {str(cid): class_names[cid] for cid in ordered_ids},
        "train_imbalance_ratio_max_to_min": round(max_px / max(min_px, 1.0), 2),
        "recommended_default_method": "median_frequency_capped",
        "weighting_methods": {
            "median_frequency_capped": _to_dict(med_bal_capped),
            "inverse_frequency_capped": _to_dict(inv_capped),
            "effective_num_samples": _to_dict(eff_weights),
            "swan2021_one_minus_composition": _to_dict(swan_weights),
        },
        "split_pixel_distributions": split_stats,
    }

    out_path = resolve_path(cfg["paths"]["class_weights_file"], root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOGGER.info("Saved class imbalance & weight metadata to %s", out_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute class imbalance analysis and weights.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    cfg = load_config(args.config)
    compute_and_save_class_weights(cfg)


if __name__ == "__main__":
    main()
