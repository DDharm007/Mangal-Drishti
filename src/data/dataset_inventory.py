"""Dataset inventory utilities for MANGAL DRISHTI.

Scans acquired NASA/JPL AI4MARS directories and external qualitative samples
to count files, summarize formats and storage sizes, compute SHA-256 hashes for
duplicate detection, and verify 1-to-1 image-mask correspondence across splits.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

LOGGER = logging.getLogger("mangal_drishti.inventory")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_split_manifest(ai4mars_root: Path) -> Dict[str, object]:
    """Load the chronological and official split manifest if present."""
    manifest_path = ai4mars_root / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing split_manifest.json at {manifest_path}. Run acquire_dataset.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def scan_directory_files(directory: Path, exts: Optional[set[str]] = None) -> List[Path]:
    """Return sorted list of files in a directory optionally filtered by extension."""
    if not directory.exists():
        return []
    files = [p for p in directory.iterdir() if p.is_file()]
    if exts is not None:
        files = [p for p in files if p.suffix.lower() in exts]
    return sorted(files)


def inspect_storage_and_duplicates(
    ai4mars_root: Path, external_samples_root: Optional[Path] = None
) -> Dict[str, object]:
    """Compute full file inventory, format counts, dimensions sample, storage size, and SHA-256 duplicates."""
    categories = {
        "msl_ncam_edr_images": ai4mars_root / "msl" / "ncam" / "images" / "edr",
        "msl_ncam_mxy_rover_masks": ai4mars_root / "msl" / "ncam" / "images" / "mxy",
        "msl_ncam_rng30m_masks": ai4mars_root / "msl" / "ncam" / "images" / "rng-30m",
        "msl_ncam_train_labels": ai4mars_root / "msl" / "ncam" / "labels" / "train",
        "msl_ncam_test_min1_labels": ai4mars_root / "msl" / "ncam" / "labels" / "test" / "masked-gold-min1-100agree",
        "msl_ncam_test_min2_labels": ai4mars_root / "msl" / "ncam" / "labels" / "test" / "masked-gold-min2-100agree",
        "msl_ncam_test_min3_labels": ai4mars_root / "msl" / "ncam" / "labels" / "test" / "masked-gold-min3-100agree",
        "mer_test_images": ai4mars_root / "mer" / "images" / "test",
        "mer_test_min1_labels": ai4mars_root / "mer" / "labels" / "test" / "masked-gold-min1-100agree",
        "mer_test_min2_labels": ai4mars_root / "mer" / "labels" / "test" / "masked-gold-min2-100agree",
        "mer_test_min3_labels": ai4mars_root / "mer" / "labels" / "test" / "masked-gold-min3-100agree",
    }

    category_stats: Dict[str, object] = {}
    total_ai4mars_bytes = 0
    format_counts: collections.Counter[str] = collections.Counter()
    image_hash_map: Dict[str, List[str]] = collections.defaultdict(list)
    mask_hash_map: Dict[str, List[str]] = collections.defaultdict(list)

    for cat_name, cat_dir in categories.items():
        files = scan_directory_files(cat_dir, IMAGE_EXTENSIONS)
        cat_bytes = 0
        cat_exts: collections.Counter[str] = collections.Counter()
        cat_dims: collections.Counter[str] = collections.Counter()
        unreadable: List[str] = []

        for fpath in files:
            fsize = fpath.stat().st_size
            cat_bytes += fsize
            total_ai4mars_bytes += fsize
            cat_exts[fpath.suffix.lower()] += 1
            format_counts[fpath.suffix.lower()] += 1

            rel = str(fpath.relative_to(ai4mars_root)).replace("\\", "/")
            digest = compute_sha256(fpath)
            if "images/edr" in rel or "mer/images/test" in rel:
                image_hash_map[digest].append(rel)
            elif "labels/" in rel:
                mask_hash_map[digest].append(rel)

            try:
                with Image.open(fpath) as im:
                    cat_dims[f"{im.width}x{im.height}"] += 1
            except Exception:
                unreadable.append(rel)

        category_stats[cat_name] = {
            "path": str(cat_dir),
            "file_count": len(files),
            "size_bytes": cat_bytes,
            "size_mb": round(cat_bytes / (1024 * 1024), 4),
            "extensions": dict(cat_exts),
            "dimensions": dict(cat_dims),
            "header_unreadable_count": len(unreadable),
        }

    # Include documentation/metadata files in total disk footprint
    all_files = [p for p in ai4mars_root.rglob("*") if p.is_file()]
    full_disk_bytes = sum(p.stat().st_size for p in all_files)

    image_duplicates = {
        h: paths for h, paths in image_hash_map.items() if len(paths) > 1
    }
    mask_duplicates = {
        h: paths for h, paths in mask_hash_map.items() if len(paths) > 1
    }

    # Verify pair correspondence across raw directories and manifest
    manifest = load_split_manifest(ai4mars_root)
    missing_from_manifest: List[Dict[str, str]] = []
    verified_pair_counts: Dict[str, int] = {}

    for split_name, records in manifest["splits"].items():
        ok_cnt = 0
        for rec in records:
            img_p = ai4mars_root / rec["image_relpath"]
            msk_p = ai4mars_root / rec["mask_relpath"]
            if img_p.exists() and msk_p.exists():
                ok_cnt += 1
            else:
                missing_from_manifest.append(
                    {
                        "split": split_name,
                        "id": rec["id"],
                        "image_exists": str(img_p.exists()),
                        "mask_exists": str(msk_p.exists()),
                    }
                )
        verified_pair_counts[split_name] = ok_cnt

    # Check raw directory pairing (any orphan EDR without train or test mask, etc.)
    edr_stems = {
        p.stem for p in scan_directory_files(categories["msl_ncam_edr_images"], IMAGE_EXTENSIONS)
    }
    train_stems = {
        p.stem for p in scan_directory_files(categories["msl_ncam_train_labels"], IMAGE_EXTENSIONS)
    }
    msl_test_stems = {
        p.name.replace("_merged.png", "")
        for p in scan_directory_files(categories["msl_ncam_test_min3_labels"], {".png"})
    }
    unmatched_msl_images = sorted(edr_stems - (train_stems | msl_test_stems))
    unmatched_msl_train_masks = sorted(train_stems - edr_stems)
    unmatched_msl_test_masks = sorted(msl_test_stems - edr_stems)

    mer_img_stems = {
        p.stem for p in scan_directory_files(categories["mer_test_images"], IMAGE_EXTENSIONS)
    }
    mer_test_stems = {
        p.name.split("_")[0]
        for p in scan_directory_files(categories["mer_test_min3_labels"], {".png"})
    }
    unmatched_mer_images = sorted(mer_img_stems - mer_test_stems)
    unmatched_mer_masks = sorted(mer_test_stems - mer_img_stems)

    # Check train/test ID overlap (leakage check)
    train_test_overlap = sorted((train_stems & msl_test_stems))

    external_info: Dict[str, object] = {"count": 0, "files": [], "size_bytes": 0}
    if external_samples_root and external_samples_root.exists():
        ext_imgs = scan_directory_files(external_samples_root, IMAGE_EXTENSIONS)
        ext_bytes = sum(p.stat().st_size for p in ext_imgs)
        external_info = {
            "path": str(external_samples_root),
            "count": len(ext_imgs),
            "size_bytes": ext_bytes,
            "size_mb": round(ext_bytes / (1024 * 1024), 4),
            "files": [p.name for p in ext_imgs],
        }

    total_raw_images = (
        category_stats["msl_ncam_edr_images"]["file_count"]
        + category_stats["mer_test_images"]["file_count"]
    )
    total_primary_masks = (
        category_stats["msl_ncam_train_labels"]["file_count"]
        + category_stats["msl_ncam_test_min3_labels"]["file_count"]
        + category_stats["mer_test_min3_labels"]["file_count"]
    )
    total_all_label_masks = (
        category_stats["msl_ncam_train_labels"]["file_count"]
        + category_stats["msl_ncam_test_min1_labels"]["file_count"]
        + category_stats["msl_ncam_test_min2_labels"]["file_count"]
        + category_stats["msl_ncam_test_min3_labels"]["file_count"]
        + category_stats["mer_test_min1_labels"]["file_count"]
        + category_stats["mer_test_min2_labels"]["file_count"]
        + category_stats["mer_test_min3_labels"]["file_count"]
    )
    total_aux_masks = (
        category_stats["msl_ncam_mxy_rover_masks"]["file_count"]
        + category_stats["msl_ncam_rng30m_masks"]["file_count"]
    )

    return {
        "ai4mars_root": str(ai4mars_root),
        "total_files_on_disk": len(all_files),
        "total_size_bytes": full_disk_bytes,
        "total_size_mb": round(full_disk_bytes / (1024 * 1024), 2),
        "total_size_gb": round(full_disk_bytes / (1024 * 1024 * 1024), 4),
        "totals": {
            "raw_rover_images": total_raw_images,
            "primary_segmentation_masks": total_primary_masks,
            "all_segmentation_mask_variants": total_all_label_masks,
            "auxiliary_masks_mxy_and_rng30m": total_aux_masks,
            "verified_manifest_pairs": sum(verified_pair_counts.values()),
        },
        "verified_pairs_by_split": verified_pair_counts,
        "formats": dict(format_counts),
        "categories": category_stats,
        "duplicates": {
            "exact_duplicate_image_groups": len(image_duplicates),
            "exact_duplicate_image_details": image_duplicates,
            "exact_duplicate_mask_groups": len(mask_duplicates),
            "note": "Exact mask SHA-256 duplicates across min1/min2/min3 variants of the same image are expected when all pixels match across consensus thresholds; duplicate masks within the same split or empty masks are flagged separately.",
        },
        "correspondence_checks": {
            "missing_manifest_pairs": len(missing_from_manifest),
            "missing_manifest_details": missing_from_manifest,
            "unmatched_msl_images": len(unmatched_msl_images),
            "unmatched_msl_train_masks": len(unmatched_msl_train_masks),
            "unmatched_msl_test_masks": len(unmatched_msl_test_masks),
            "unmatched_mer_images": len(unmatched_mer_images),
            "unmatched_mer_masks": len(unmatched_mer_masks),
            "train_test_id_leakage_count": len(train_test_overlap),
        },
        "external_nasa_samples": external_info,
    }
