"""Dataset inspection and validation engine for MANGAL DRISHTI.

Performs:
  A. Dataset inventory (counts, formats, sizes, duplicates, pairing)
  B. Image validation (readability, corruption, dimensions, channels, PDS metadata)
  C. Mask validation (readability, matching dimensions, official label key verification)
  D. Terrain-class analysis (pixel frequencies, valid vs unlabeled shares, class imbalance)

Outputs:
  - datasets/inspection/dataset_summary.json
  - datasets/inspection/validation_report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_inventory import (
    inspect_storage_and_duplicates,
    load_split_manifest,
)

LOGGER = logging.getLogger("mangal_drishti.inspect")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_official_nav_label_keys(ai4mars_root: Path) -> Dict[int, str]:
    """Load and verify the official NAV pixel mapping from label_keys.json."""
    key_path = ai4mars_root / "label_keys.json"
    if not key_path.exists():
        raise FileNotFoundError(
            f"Official label_keys.json not found at {key_path}. Cannot verify class mapping."
        )
    raw = json.loads(key_path.read_text(encoding="utf-8"))
    if "NAV" not in raw:
        raise KeyError("Expected 'NAV' key inside official label_keys.json")
    mapping: Dict[int, str] = {}
    for k, v in raw["NAV"].items():
        mapping[int(k)] = "unlabeled_null" if v is None else str(v)
    return mapping


def parse_pds_metadata(stem: str, mission: str) -> Dict[str, object]:
    """Extract camera, rover, and SCLK metadata encoded in official PDS filenames."""
    meta: Dict[str, object] = {"mission": mission, "camera_code": "unknown", "sclk": None}
    # MSL example: NLA_397586934EDR_F0010008AUT_04096M1
    m_msl = re.match(r"^([A-Z0-9]{3})_(\d{9,10})([A-Z]{3})_", stem)
    if m_msl:
        cam_prefix = m_msl.group(1)
        cam_desc = {
            "NLA": "NavCam Left-A",
            "NLB": "NavCam Left-B",
            "NRA": "NavCam Right-A",
            "NRB": "NavCam Right-B",
        }.get(cam_prefix, cam_prefix)
        meta["camera_code"] = cam_prefix
        meta["camera_name"] = cam_desc
        meta["sclk"] = int(m_msl.group(2))
        meta["product_type"] = m_msl.group(3)
        return meta

    # MER example: 1n129697839eff0338p1931l0m1 or 2n130887021eff10agp1863l0m1
    m_mer = re.match(r"^([12])([a-z])(\d{9})([a-z]{3})", stem)
    if m_mer:
        rover = "MER-1 (Opportunity)" if m_mer.group(1) == "1" else "MER-2 (Spirit)"
        cam_char = m_mer.group(2)
        meta["mission"] = rover
        meta["camera_code"] = f"MER-{cam_char.upper()}"
        meta["camera_name"] = "NavCam" if cam_char == "n" else cam_char
        meta["sclk"] = int(m_mer.group(3))
        meta["product_type"] = m_mer.group(4).upper()
        return meta

    return meta


def _compute_class_metrics(
    counts: Dict[int, int],
    image_presence: Dict[int, int],
    num_images: int,
    label_map: Dict[int, str],
) -> Dict[str, object]:
    """Summarize class frequencies, valid vs unlabeled shares, and imbalance."""
    total_pixels = sum(counts.values())
    unlabeled_pixels = counts.get(255, 0)
    labeled_pixels = total_pixels - unlabeled_pixels

    per_class: Dict[str, object] = {}
    valid_counts: Dict[str, int] = {}

    for cid, cname in sorted(label_map.items()):
        c_pixels = int(counts.get(cid, 0))
        pct_total = (c_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        if cid == 255:
            pct_labeled = None
        else:
            pct_labeled = (c_pixels / labeled_pixels * 100.0) if labeled_pixels > 0 else 0.0
            valid_counts[cname] = c_pixels

        presence_cnt = int(image_presence.get(cid, 0))
        per_class[str(cid)] = {
            "class_id": cid,
            "class_name": cname,
            "pixel_count": c_pixels,
            "pct_of_all_pixels": round(pct_total, 4),
            "pct_of_labeled_pixels": round(pct_labeled, 4) if pct_labeled is not None else None,
            "images_with_class": presence_cnt,
            "pct_images_with_class": round(
                (presence_cnt / num_images * 100.0) if num_images > 0 else 0.0, 2
            ),
        }

    nonzero_valid = {k: v for k, v in valid_counts.items() if v > 0}
    imbalance_ratio = None
    majority_class = None
    minority_class = None
    if nonzero_valid:
        majority_class = max(nonzero_valid, key=nonzero_valid.get)
        minority_class = min(nonzero_valid, key=nonzero_valid.get)
        imbalance_ratio = round(
            nonzero_valid[majority_class] / nonzero_valid[minority_class], 2
        )

    return {
        "num_images": num_images,
        "total_pixels": int(total_pixels),
        "labeled_pixels": int(labeled_pixels),
        "unlabeled_pixels": int(unlabeled_pixels),
        "labeled_pct": round(
            (labeled_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0, 4
        ),
        "unlabeled_pct": round(
            (unlabeled_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0, 4
        ),
        "majority_labeled_class": majority_class,
        "minority_labeled_class": minority_class,
        "max_to_min_class_imbalance_ratio": imbalance_ratio,
        "classes": per_class,
    }


def validate_and_analyze_splits(
    ai4mars_root: Path,
    manifest: Dict[str, object],
    label_map: Dict[int, str],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Validate every image and mask in the manifest and compute exact pixel statistics."""
    allowed_ids = set(label_map.keys())

    corrupted_images: List[Dict[str, str]] = []
    corrupted_masks: List[Dict[str, str]] = []
    dimension_mismatches: List[Dict[str, object]] = []
    unexpected_channels: List[Dict[str, object]] = []
    invalid_mask_values: List[Dict[str, object]] = []
    empty_masks: List[Dict[str, str]] = []

    image_dims: collections.Counter[str] = collections.Counter()
    image_modes: collections.Counter[str] = collections.Counter()
    image_formats: collections.Counter[str] = collections.Counter()
    mask_dims: collections.Counter[str] = collections.Counter()
    mask_modes: collections.Counter[str] = collections.Counter()
    camera_counts: collections.Counter[str] = collections.Counter()
    mission_counts: collections.Counter[str] = collections.Counter()

    # Auxiliary mask inspection (MXY rover mask, RNG-30M range mask)
    aux_stats = {
        "mxy_checked": 0,
        "mxy_corrupted": 0,
        "mxy_dim_mismatch": 0,
        "mxy_unique_values": set(),
        "rng_checked": 0,
        "rng_corrupted": 0,
        "rng_dim_mismatch": 0,
        "rng_unique_values": set(),
    }

    intensity_means: List[float] = []
    intensity_stds: List[float] = []
    global_min_val = 255
    global_max_val = 0

    overall_pixel_counts: Dict[int, int] = collections.defaultdict(int)
    overall_presence: Dict[int, int] = collections.defaultdict(int)
    overall_unique_mask_vals: set[int] = set()
    total_valid_pairs = 0

    split_class_summaries: Dict[str, object] = {}
    sclk_ranges: Dict[str, Dict[str, object]] = {}

    # Track rich candidate samples for visualization
    sample_candidates: List[Dict[str, object]] = []

    for split_name, records in manifest["splits"].items():
        split_counts: Dict[int, int] = collections.defaultdict(int)
        split_presence: Dict[int, int] = collections.defaultdict(int)
        split_sclks: List[int] = []
        split_missions: collections.Counter[str] = collections.Counter()

        for rec in tqdm(records, desc=f"Validating {split_name}", unit="pair"):
            img_path = ai4mars_root / rec["image_relpath"]
            mask_path = ai4mars_root / rec["mask_relpath"]

            pds = parse_pds_metadata(rec["id"], rec["mission"])
            camera_counts[str(pds.get("camera_name", "unknown"))] += 1
            mission_counts[str(pds["mission"])] += 1
            split_missions[str(pds["mission"])] += 1
            if pds.get("sclk") is not None:
                split_sclks.append(int(pds["sclk"]))

            # 1. Image validation (Pillow + OpenCV integrity check)
            img_arr = None
            img_w = img_h = 0
            try:
                with Image.open(img_path) as im:
                    im.load()
                    image_formats[str(im.format)] += 1
                    image_modes[str(im.mode)] += 1
                    img_w, img_h = im.size
                    image_dims[f"{img_w}x{img_h}"] += 1
                    img_arr = np.asarray(im)
                    if im.mode not in ("L", "RGB"):
                        unexpected_channels.append(
                            {
                                "split": split_name,
                                "path": rec["image_relpath"],
                                "mode": im.mode,
                                "shape": list(img_arr.shape),
                            }
                        )
                # Double-check decoding with OpenCV
                cv_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if cv_img is None:
                    raise ValueError("cv2.imread returned None")
            except Exception as exc:
                corrupted_images.append(
                    {"split": split_name, "path": rec["image_relpath"], "error": str(exc)}
                )
                continue

            if img_arr is not None:
                global_min_val = min(global_min_val, int(img_arr.min()))
                global_max_val = max(global_max_val, int(img_arr.max()))
                intensity_means.append(float(img_arr.mean()))
                intensity_stds.append(float(img_arr.std()))

            # 2. Mask validation
            mask_arr = None
            msk_w = msk_h = 0
            try:
                with Image.open(mask_path) as msk:
                    msk.load()
                    mask_modes[str(msk.mode)] += 1
                    msk_w, msk_h = msk.size
                    mask_dims[f"{msk_w}x{msk_h}"] += 1
                    mask_arr = np.asarray(msk)
                if mask_arr.ndim != 2:
                    unexpected_channels.append(
                        {
                            "split": split_name,
                            "path": rec["mask_relpath"],
                            "mode": "mask_multichannel",
                            "shape": list(mask_arr.shape),
                        }
                    )
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[:, :, 0]
            except Exception as exc:
                corrupted_masks.append(
                    {"split": split_name, "path": rec["mask_relpath"], "error": str(exc)}
                )
                continue

            if (img_w, img_h) != (msk_w, msk_h):
                dimension_mismatches.append(
                    {
                        "split": split_name,
                        "id": rec["id"],
                        "image_size": [img_w, img_h],
                        "mask_size": [msk_w, msk_h],
                    }
                )
                continue

            # 3. Auxiliary MXY / RNG-30M validation if present
            for aux_key, prefix in (("mxy_relpath", "mxy"), ("rng_relpath", "rng")):
                rel_aux = rec.get(aux_key)
                if rel_aux:
                    aux_p = ai4mars_root / rel_aux
                    if aux_p.exists():
                        aux_stats[f"{prefix}_checked"] += 1
                        try:
                            with Image.open(aux_p) as am:
                                am.load()
                                if am.size != (img_w, img_h):
                                    aux_stats[f"{prefix}_dim_mismatch"] += 1
                                uvals = np.unique(np.asarray(am)).tolist()
                                aux_stats[f"{prefix}_unique_values"].update(int(x) for x in uvals)
                        except Exception:
                            aux_stats[f"{prefix}_corrupted"] += 1

            # 4. Pixel value verification & bincount
            total_valid_pairs += 1
            hist = np.bincount(mask_arr.ravel(), minlength=256)
            present_vals = np.flatnonzero(hist)
            bad_vals = [int(v) for v in present_vals if int(v) not in allowed_ids]
            if bad_vals:
                invalid_mask_values.append(
                    {
                        "split": split_name,
                        "path": rec["mask_relpath"],
                        "invalid_values": bad_vals,
                    }
                )

            for val in present_vals:
                ival = int(val)
                cnt = int(hist[ival])
                overall_unique_mask_vals.add(ival)
                overall_pixel_counts[ival] += cnt
                overall_presence[ival] += 1
                split_counts[ival] += cnt
                split_presence[ival] += 1

            labeled_px = int(mask_arr.size - hist[255])
            if labeled_px == 0:
                empty_masks.append(
                    {"split": split_name, "id": rec["id"], "path": rec["mask_relpath"]}
                )

            # Collect metadata for selecting diverse visualization samples
            if labeled_px > 100_000:
                c0 = int(hist[0]) / labeled_px
                c1 = int(hist[1]) / labeled_px
                c2 = int(hist[2]) / labeled_px
                c3 = int(hist[3]) / labeled_px
                n_classes_present = sum(1 for c in (hist[0], hist[1], hist[2], hist[3]) if c > 500)
                sample_candidates.append(
                    {
                        "id": rec["id"],
                        "split": split_name,
                        "mission": str(pds["mission"]),
                        "camera": str(pds.get("camera_name", "NavCam")),
                        "sclk": pds.get("sclk"),
                        "image_relpath": rec["image_relpath"],
                        "mask_relpath": rec["mask_relpath"],
                        "mxy_relpath": rec.get("mxy_relpath", ""),
                        "rng_relpath": rec.get("rng_relpath", ""),
                        "labeled_coverage_pct": round(labeled_px / mask_arr.size * 100.0, 2),
                        "classes_present": n_classes_present,
                        "soil_share": round(c0, 4),
                        "bedrock_share": round(c1, 4),
                        "sand_share": round(c2, 4),
                        "big_rock_share": round(c3, 4),
                    }
                )

        split_class_summaries[split_name] = _compute_class_metrics(
            split_counts, split_presence, len(records), label_map
        )
        split_class_summaries[split_name]["missions"] = dict(split_missions)
        if split_sclks:
            sclk_ranges[split_name] = {
                "min_sclk": min(split_sclks),
                "max_sclk": max(split_sclks),
            }

    # Also compute pixel distributions for expert test min1 / min2 variants for scientific comparison
    expert_threshold_comparison: Dict[str, object] = {}
    for split_name in ("test_msl", "test_mer"):
        for variant_key, variant_label in (
            ("mask_min1_relpath", "min1_100agree"),
            ("mask_min2_relpath", "min2_100agree"),
            ("mask_relpath", "min3_100agree"),
        ):
            v_counts: Dict[int, int] = collections.defaultdict(int)
            v_pres: Dict[int, int] = collections.defaultdict(int)
            n_imgs = 0
            for rec in manifest["splits"][split_name]:
                rel = rec.get(variant_key)
                if not rel:
                    continue
                p = ai4mars_root / rel
                if not p.exists():
                    continue
                arr = np.asarray(Image.open(p))
                hist = np.bincount(arr.ravel(), minlength=256)
                for val in np.flatnonzero(hist):
                    v_counts[int(val)] += int(hist[int(val)])
                    v_pres[int(val)] += 1
                n_imgs += 1
            expert_threshold_comparison[f"{split_name}_{variant_label}"] = _compute_class_metrics(
                v_counts, v_pres, n_imgs, label_map
            )

    # Pick diverse representative samples across classes & rovers
    representative_samples: Dict[str, object] = {}
    if sample_candidates:
        msl_cands = [s for s in sample_candidates if s["split"] in ("train", "val", "test_msl")]
        mer_cands = [s for s in sample_candidates if s["split"] == "test_mer"]
        representative_samples["multi_class_hero_msl"] = max(
            msl_cands,
            key=lambda s: (
                s["classes_present"],
                min(s["big_rock_share"], 0.15) * 4 + s["labeled_coverage_pct"] / 100.0,
            ),
        )
        representative_samples["big_rock_hazard_msl"] = max(
            msl_cands, key=lambda s: (s["big_rock_share"], s["labeled_coverage_pct"])
        )
        representative_samples["sand_dune_msl"] = max(
            msl_cands, key=lambda s: (s["sand_share"], s["labeled_coverage_pct"])
        )
        representative_samples["bedrock_outcrop_msl"] = max(
            msl_cands, key=lambda s: (s["bedrock_share"], s["labeled_coverage_pct"])
        )
        representative_samples["soil_regolith_msl"] = max(
            msl_cands, key=lambda s: (s["soil_share"], s["labeled_coverage_pct"])
        )
        if mer_cands:
            opp = [s for s in mer_cands if "Opportunity" in s["mission"]]
            spr = [s for s in mer_cands if "Spirit" in s["mission"]]
            if opp:
                representative_samples["mer_opportunity_test"] = max(
                    opp, key=lambda s: (s["classes_present"], s["labeled_coverage_pct"])
                )
            if spr:
                representative_samples["mer_spirit_test"] = max(
                    spr, key=lambda s: (s["classes_present"], s["labeled_coverage_pct"])
                )

    overall_terrain = _compute_class_metrics(
        overall_pixel_counts, overall_presence, total_valid_pairs, label_map
    )

    validation_report = {
        "status": "PASS"
        if not corrupted_images
        and not corrupted_masks
        and not dimension_mismatches
        and not invalid_mask_values
        else "WARNINGS_DETECTED",
        "validated_image_mask_pairs": total_valid_pairs,
        "image_validation": {
            "corrupted_image_count": len(corrupted_images),
            "corrupted_images": corrupted_images,
            "unexpected_channel_count": len(unexpected_channels),
            "unexpected_channels": unexpected_channels,
            "image_formats": dict(image_formats),
            "image_color_modes": dict(image_modes),
            "image_dimensions": dict(image_dims),
            "pixel_intensity_summary": {
                "global_min": int(global_min_val),
                "global_max": int(global_max_val),
                "mean_of_image_means": round(float(np.mean(intensity_means)), 3)
                if intensity_means
                else None,
                "mean_of_image_stds": round(float(np.mean(intensity_stds)), 3)
                if intensity_stds
                else None,
            },
        },
        "mask_validation": {
            "official_label_key_source": "datasets/raw/ai4mars/label_keys.json (NAV)",
            "verified_label_mapping": {str(k): v for k, v in sorted(label_map.items())},
            "observed_unique_pixel_values": sorted(int(v) for v in overall_unique_mask_vals),
            "corrupted_mask_count": len(corrupted_masks),
            "corrupted_masks": corrupted_masks,
            "dimension_mismatch_count": len(dimension_mismatches),
            "dimension_mismatches": dimension_mismatches,
            "invalid_pixel_value_mask_count": len(invalid_mask_values),
            "invalid_mask_values": invalid_mask_values,
            "empty_all_255_mask_count": len(empty_masks),
            "empty_masks": empty_masks,
            "mask_color_modes": dict(mask_modes),
            "mask_dimensions": dict(mask_dims),
        },
        "auxiliary_mask_validation": {
            "mxy_rover_masks_checked": aux_stats["mxy_checked"],
            "mxy_corrupted": aux_stats["mxy_corrupted"],
            "mxy_dim_mismatch": aux_stats["mxy_dim_mismatch"],
            "mxy_unique_pixel_values": sorted(aux_stats["mxy_unique_values"]),
            "rng30m_masks_checked": aux_stats["rng_checked"],
            "rng30m_corrupted": aux_stats["rng_corrupted"],
            "rng30m_dim_mismatch": aux_stats["rng_dim_mismatch"],
            "rng30m_unique_pixel_values": sorted(aux_stats["rng_unique_values"]),
        },
    }

    dataset_summary_partial = {
        "project": "MANGAL DRISHTI",
        "phase": "Step 01 - Dataset Collection, Research & Intelligence Foundation",
        "dataset": {
            "name": "NASA/JPL AI4MARS",
            "version": "0.6",
            "doi": "10.5281/zenodo.15995036",
            "license": "CC-BY-4.0",
            "official_sources": [
                "https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars",
                "https://zenodo.org/records/15995036",
                "https://github.com/AI4MARS/AI4MARS",
                "https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html",
            ],
        },
        "missions_and_cameras": {
            "missions": dict(mission_counts),
            "cameras": dict(camera_counts),
            "sclk_ranges_by_split": sclk_ranges,
        },
        "overall_terrain_distribution": overall_terrain,
        "split_terrain_distributions": split_class_summaries,
        "expert_test_consensus_comparison": expert_threshold_comparison,
        "representative_samples": representative_samples,
    }

    return dataset_summary_partial, validation_report


def run_inspection(project_root: Path) -> Tuple[Path, Path]:
    """Run full inventory, image/mask validation, and terrain-class analysis."""
    ai4mars_root = project_root / "datasets" / "raw" / "ai4mars"
    external_root = project_root / "datasets" / "samples" / "external_nasa"
    inspection_dir = project_root / "datasets" / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running dataset inventory and duplicate check...")
    inventory = inspect_storage_and_duplicates(ai4mars_root, external_root)

    LOGGER.info("Loading official label keys and split manifest...")
    label_map = load_official_nav_label_keys(ai4mars_root)
    manifest = load_split_manifest(ai4mars_root)

    LOGGER.info("Validating all images, masks, and computing terrain-class distributions...")
    summary_partial, validation_report = validate_and_analyze_splits(
        ai4mars_root, manifest, label_map
    )

    # Combine inventory + summary into dataset_summary.json
    dataset_summary = {
        **summary_partial,
        "inventory": inventory,
        "split_strategy": manifest["split_strategy"],
    }

    # Include inventory correspondence checks inside validation_report as well
    validation_report["correspondence_checks"] = inventory["correspondence_checks"]
    validation_report["duplicate_checks"] = inventory["duplicates"]

    summary_path = inspection_dir / "dataset_summary.json"
    validation_path = inspection_dir / "validation_report.json"

    summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", validation_path)
    return summary_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate the acquired AI4MARS dataset."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    run_inspection(args.project_root)


if __name__ == "__main__":
    main()
