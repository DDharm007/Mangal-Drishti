"""Leakage-free dataset splitting and manifest generation for MANGAL DRISHTI.

Preserves official NASA/JPL AI4MARS expert test splits (`test_msl` + `test_mer`)
while partitioning the valid MSL crowdsourced pool into `train` and `val` by
Spacecraft Clock (SCLK) temporal drive clusters with rare-class stratification.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.mask_encoder import MaskEncoder

LOGGER = logging.getLogger("mangal_drishti.split")


def _cluster_by_sclk(
    records: List[Dict[str, Any]], gap_seconds: int
) -> List[List[Dict[str, Any]]]:
    """Group sorted records into SCLK drive clusters so nearby frames stay together."""
    ordered = sorted(records, key=lambda r: (int(r.get("sclk") or 0), r["id"]))
    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for rec in ordered:
        sclk = int(rec.get("sclk") or 0)
        if not current or (sclk - int(current[-1].get("sclk") or 0)) <= gap_seconds:
            current.append(rec)
        else:
            clusters.append(current)
            current = [rec]
    if current:
        clusters.append(current)
    return clusters


def _cluster_stratum(cluster: List[Dict[str, Any]]) -> str:
    """Assign a stratification label to a drive cluster prioritizing rare classes."""
    tot = np.zeros(4, dtype=np.int64)
    for item in cluster:
        counts = item["pixel_counts"]
        for c in range(4):
            tot[c] += counts[str(c)]
    if tot[3] > 500:
        return "has_big_rock"
    if tot[2] > 5000:
        return "has_sand"
    return "bedrock_dom" if tot[1] >= tot[0] else "soil_dom"


def _assign_clusters_stratified(
    clusters: List[List[Dict[str, Any]]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically assign SCLK drive clusters to train vs val by stratum."""
    rng = random.Random(seed)
    by_stratum: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = collections.defaultdict(list)
    for idx, cluster in enumerate(clusters):
        group_id = f"msl_sclk_cluster_{idx:03d}"
        for rec in cluster:
            rec["group_id"] = group_id
        by_stratum[_cluster_stratum(cluster)].append((idx, cluster))

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for stratum in sorted(by_stratum.keys()):
        group_list = list(by_stratum[stratum])
        rng.shuffle(group_list)
        total_in_stratum = sum(len(c) for _, c in group_list)
        target_val = total_in_stratum * val_ratio

        # Sort only slightly? Keep shuffled order and greedily fill up to target_val
        # while ensuring at least 1 cluster remains in train
        val_acc = 0
        for g_pos, (_, cluster) in enumerate(group_list):
            rem_clusters = len(group_list) - g_pos - 1
            if (
                val_acc < target_val
                and (val_acc + len(cluster) <= target_val * 1.25 or val_acc == 0)
                and (len(train_records) > 0 or rem_clusters > 0)
            ):
                val_records.extend(cluster)
                val_acc += len(cluster)
            else:
                train_records.extend(cluster)

    train_records.sort(key=lambda r: (int(r.get("sclk") or 0), r["id"]))
    val_records.sort(key=lambda r: (int(r.get("sclk") or 0), r["id"]))
    return train_records, val_records


def _inspect_raw_mask_counts(
    encoder: MaskEncoder,
    mask_path: Path,
    mxy_path: Optional[Path] = None,
    rng_path: Optional[Path] = None,
) -> Tuple[Dict[str, int], int]:
    """Load raw mask (with optional MXY/RNG-30M) and return class pixel histogram."""
    arr = encoder.load_raw_mask(mask_path)
    arr = encoder.apply_auxiliary_exclusions(arr, mxy_path=mxy_path, rng_path=rng_path)
    hist = np.bincount(arr.ravel(), minlength=256)
    counts = {
        "0": int(hist[0]),
        "1": int(hist[1]),
        "2": int(hist[2]),
        "3": int(hist[3]),
        "255": int(hist[255]),
    }
    labeled_pixels = int(arr.size - hist[255])
    return counts, labeled_pixels


def generate_splits(
    cfg: Dict[str, Any], project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Build leakage-free split_manifest.json and dataset_manifest.csv."""
    root = project_root or get_project_root()
    raw_root = resolve_path(cfg["paths"]["raw_ai4mars_dir"], root)
    raw_manifest_path = resolve_path(cfg["paths"]["raw_split_manifest"], root)
    label_keys_path = resolve_path(cfg["paths"]["raw_label_keys"], root)

    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"Step 1 split manifest missing: {raw_manifest_path}")

    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    encoder = MaskEncoder(
        label_keys_path=label_keys_path,
        ignore_index=int(cfg["mask"]["ignore_index"]),
    )

    apply_mxy = bool(cfg["mask"].get("apply_mxy_rover_mask", True))
    apply_rng = bool(cfg["mask"].get("apply_rng30m_range_mask", True))
    filter_empty = bool(cfg["mask"].get("filter_empty_masks_train_val", True))
    gap_seconds = int(cfg["splitting"]["sclk_cluster_gap_seconds"])
    val_ratio = float(cfg["splitting"]["val_ratio_of_train_pool"])
    seed = int(cfg["project"]["random_seed"])

    # 1. Inspect and filter crowdsourced MSL train/val pool
    raw_tv_pool = raw_manifest["splits"]["train"] + raw_manifest["splits"]["val"]
    valid_tv_pool: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    for rec in raw_tv_pool:
        img_path = raw_root / rec["image_relpath"]
        mask_path = raw_root / rec["mask_relpath"]
        if not img_path.exists() or not mask_path.exists():
            rejected_records.append(
                {
                    "id": rec["id"],
                    "source_split": "msl_crowdsourced",
                    "image_relpath": rec["image_relpath"],
                    "mask_relpath": rec["mask_relpath"],
                    "reason": "missing_image_or_mask",
                }
            )
            continue

        mxy_p = raw_root / rec["mxy_relpath"] if (apply_mxy and rec.get("mxy_relpath")) else None
        rng_p = raw_root / rec["rng_relpath"] if (apply_rng and rec.get("rng_relpath")) else None
        counts, labeled_px = _inspect_raw_mask_counts(encoder, mask_path, mxy_p, rng_p)

        if filter_empty and labeled_px == 0:
            rejected_records.append(
                {
                    "id": rec["id"],
                    "source_split": "msl_crowdsourced",
                    "image_relpath": rec["image_relpath"],
                    "mask_relpath": rec["mask_relpath"],
                    "reason": "empty_all_255_mask",
                }
            )
            continue

        valid_tv_pool.append(
            {
                **rec,
                "official_split": "train",
                "consensus_tier_used": "crowdsourced_merged_2of3",
                "labeled_pixels_raw": labeled_px,
                "pixel_counts": counts,
            }
        )

    clusters = _cluster_by_sclk(valid_tv_pool, gap_seconds=gap_seconds)
    train_records, val_records = _assign_clusters_stratified(
        clusters, val_ratio=val_ratio, seed=seed
    )
    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"

    # 2. Preserve official expert gold-standard test sets (MSL + MER)
    test_records: List[Dict[str, Any]] = []
    for test_key in ("test_msl", "test_mer"):
        for rec in raw_manifest["splits"][test_key]:
            img_path = raw_root / rec["image_relpath"]
            if not img_path.exists():
                rejected_records.append(
                    {
                        "id": rec["id"],
                        "source_split": test_key,
                        "image_relpath": rec["image_relpath"],
                        "mask_relpath": rec["mask_relpath"],
                        "reason": "missing_test_image",
                    }
                )
                continue

            mxy_p = raw_root / rec["mxy_relpath"] if (apply_mxy and rec.get("mxy_relpath")) else None
            rng_p = raw_root / rec["rng_relpath"] if (apply_rng and rec.get("rng_relpath")) else None

            # Evaluate min3 first; fallback to min2 -> min1 only if min3 is 100% 255
            chosen_rel = rec["mask_relpath"]
            chosen_tier = "min3_100agree"
            chosen_counts: Dict[str, int] = {}
            chosen_labeled = 0

            candidates = [
                ("min3_100agree", rec.get("mask_relpath")),
                ("min2_100agree", rec.get("mask_min2_relpath")),
                ("min1_100agree", rec.get("mask_min1_relpath")),
            ]
            for tier_name, rel_path in candidates:
                if not rel_path:
                    continue
                cand_p = raw_root / rel_path
                if not cand_p.exists():
                    continue
                c_counts, c_labeled = _inspect_raw_mask_counts(encoder, cand_p, mxy_p, rng_p)
                if c_labeled > 0:
                    chosen_rel = rel_path
                    chosen_tier = tier_name
                    chosen_counts = c_counts
                    chosen_labeled = c_labeled
                    break

            if chosen_labeled == 0:
                rejected_records.append(
                    {
                        "id": rec["id"],
                        "source_split": test_key,
                        "image_relpath": rec["image_relpath"],
                        "mask_relpath": rec["mask_relpath"],
                        "reason": "empty_across_all_expert_tiers",
                    }
                )
                continue

            test_records.append(
                {
                    **rec,
                    "split": "test",
                    "official_split": test_key,
                    "group_id": f"{test_key}_{int(rec.get('sclk') or 0) // gap_seconds}",
                    "mask_relpath": chosen_rel,
                    "min3_mask_relpath": rec["mask_relpath"],
                    "consensus_tier_used": chosen_tier,
                    "labeled_pixels_raw": chosen_labeled,
                    "pixel_counts": chosen_counts,
                }
            )

    # 3. Leakage assertions
    train_ids = {r["id"] for r in train_records}
    val_ids = {r["id"] for r in val_records}
    test_ids = {r["id"] for r in test_records}
    assert not (train_ids & val_ids), "Data leakage detected: ID overlap between train and val!"
    assert not (train_ids & test_ids), "Data leakage detected: ID overlap between train and test!"
    assert not (val_ids & test_ids), "Data leakage detected: ID overlap between val and test!"

    train_groups = {r["group_id"] for r in train_records}
    val_groups = {r["group_id"] for r in val_records}
    assert not (train_groups & val_groups), "Group leakage: SCLK cluster shared between train and val!"

    # Attach standardized processed filenames
    for rec in train_records + val_records + test_records:
        sp = rec["split"]
        rec["processed_image_relpath"] = f"{sp}/images/{rec['id']}.png"
        rec["processed_mask_relpath"] = f"{sp}/masks/{rec['id']}.png"

    total_valid = len(train_records) + len(val_records) + len(test_records)
    tier_breakdown = collections.Counter(r["consensus_tier_used"] for r in test_records)

    split_manifest = {
        "dataset": "NASA/JPL AI4MARS v0.6",
        "random_seed": seed,
        "split_config": {
            "sclk_cluster_gap_seconds": gap_seconds,
            "val_ratio_of_train_pool": val_ratio,
            "total_sclk_clusters_in_train_val": len(clusters),
            "train_clusters": len(train_groups),
            "val_clusters": len(val_groups),
            "preserve_official_test_sets": True,
            "test_consensus_strategy": cfg["mask"]["test_consensus_strategy"],
        },
        "counts": {
            "raw_total_pairs": len(raw_tv_pool)
            + len(raw_manifest["splits"]["test_msl"])
            + len(raw_manifest["splits"]["test_mer"]),
            "valid_total_pairs": total_valid,
            "rejected_pairs": len(rejected_records),
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
            "train_pct": round(len(train_records) / total_valid * 100.0, 2),
            "val_pct": round(len(val_records) / total_valid * 100.0, 2),
            "test_pct": round(len(test_records) / total_valid * 100.0, 2),
            "test_consensus_tier_counts": dict(tier_breakdown),
        },
        "leakage_checks": {
            "train_val_id_overlap": 0,
            "train_test_id_overlap": 0,
            "val_test_id_overlap": 0,
            "train_val_sclk_cluster_overlap": 0,
        },
        "rejected_records": rejected_records,
        "splits": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
    }

    manifest_out = resolve_path(cfg["paths"]["split_manifest_file"], root)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # Write flat CSV manifest for easy inspection & Pandas loading
    csv_out = resolve_path(cfg["paths"]["dataset_manifest_csv"], root)
    fieldnames = [
        "id",
        "split",
        "official_split",
        "mission",
        "sclk",
        "group_id",
        "consensus_tier_used",
        "image_relpath",
        "mask_relpath",
        "processed_image_relpath",
        "processed_mask_relpath",
        "labeled_pixels_raw",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in train_records + val_records + test_records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    LOGGER.info(
        "Generated splits -> train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%), rejected=%d",
        len(train_records),
        split_manifest["counts"]["train_pct"],
        len(val_records),
        split_manifest["counts"]["val_pct"],
        len(test_records),
        split_manifest["counts"]["test_pct"],
        len(rejected_records),
    )
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-free AI4MARS dataset splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    cfg = load_config(args.config)
    generate_splits(cfg)


if __name__ == "__main__":
    main()
