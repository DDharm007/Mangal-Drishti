"""Automated data quality validation engine for Step 2 processed datasets.

Verifies correspondence, dimensions, file integrity, categorical mask label IDs
(zero interpolation artifacts), zero split leakage, PyTorch Dataset/DataLoader
tensor schemas, CrossEntropyLoss compatibility, and loading throughput.

Usage:
    python -m src.data.validation --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataloader import build_dataloaders

LOGGER = logging.getLogger("mangal_drishti.validation")


def validate_processed_dataset(
    cfg: Dict[str, Any],
    config_path: Path | str = Path("configs/data_config.yaml"),
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run exhaustive integrity checks on processed files, splits, and DataLoaders."""
    root = project_root or get_project_root()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"], root)
    split_manifest_path = resolve_path(cfg["paths"]["split_manifest_file"], root)
    class_mapping_path = resolve_path(cfg["paths"]["class_mapping_file"], root)
    report_out = resolve_path(cfg["paths"]["validation_report_file"], root)

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))

    expected_h, expected_w = int(cfg["image"]["target_size"][0]), int(cfg["image"]["target_size"][1])
    ignore_index = int(cfg["mask"]["ignore_index"])
    allowed_ids = {
        int(c["training_id"]) for c in class_mapping["classes"]
    } | {ignore_index}

    missing_files: List[Dict[str, str]] = []
    corrupted_files: List[Dict[str, str]] = []
    dimension_mismatches: List[Dict[str, Any]] = []
    invalid_mask_values: List[Dict[str, Any]] = []
    empty_processed_masks: List[Dict[str, str]] = []
    orphan_files: List[str] = []

    observed_mask_ids: set[int] = set()
    split_ids: Dict[str, set[str]] = {}
    split_groups: Dict[str, set[str]] = {}
    duplicate_ids: List[str] = []
    seen_all_ids: set[str] = set()

    for split_name in ("train", "val", "test"):
        records = split_manifest["splits"][split_name]
        split_ids[split_name] = set()
        split_groups[split_name] = set()

        img_dir = processed_dir / split_name / "images"
        msk_dir = processed_dir / split_name / "masks"
        expected_stems = {r["id"] for r in records}
        actual_img_stems = {p.stem for p in img_dir.glob("*.png")} if img_dir.exists() else set()
        actual_msk_stems = {p.stem for p in msk_dir.glob("*.png")} if msk_dir.exists() else set()

        for extra in sorted(actual_img_stems - expected_stems):
            orphan_files.append(f"{split_name}/images/{extra}.png")
        for extra in sorted(actual_msk_stems - expected_stems):
            orphan_files.append(f"{split_name}/masks/{extra}.png")

        for rec in tqdm(records, desc=f"Validating processed {split_name}", unit="pair"):
            rid = rec["id"]
            if rid in seen_all_ids:
                duplicate_ids.append(rid)
            seen_all_ids.add(rid)
            split_ids[split_name].add(rid)
            if rec.get("group_id"):
                split_groups[split_name].add(str(rec["group_id"]))

            img_p = processed_dir / rec["processed_image_relpath"]
            msk_p = processed_dir / rec["processed_mask_relpath"]
            if not img_p.exists() or not msk_p.exists():
                missing_files.append(
                    {
                        "split": split_name,
                        "id": rid,
                        "img_exists": str(img_p.exists()),
                        "msk_exists": str(msk_p.exists()),
                    }
                )
                continue

            try:
                with Image.open(img_p) as im:
                    im.load()
                    img_arr = np.asarray(im)
                if img_arr.shape != (expected_h, expected_w, 3):
                    dimension_mismatches.append(
                        {
                            "split": split_name,
                            "file": rec["processed_image_relpath"],
                            "expected": [expected_h, expected_w, 3],
                            "actual": list(img_arr.shape),
                        }
                    )
            except Exception as exc:
                corrupted_files.append({"file": rec["processed_image_relpath"], "error": str(exc)})
                continue

            try:
                with Image.open(msk_p) as msk:
                    msk.load()
                    msk_arr = np.asarray(msk)
                if msk_arr.shape != (expected_h, expected_w):
                    dimension_mismatches.append(
                        {
                            "split": split_name,
                            "file": rec["processed_mask_relpath"],
                            "expected": [expected_h, expected_w],
                            "actual": list(msk_arr.shape),
                        }
                    )
                uvals = set(int(v) for v in np.unique(msk_arr))
                observed_mask_ids.update(uvals)
                bad = uvals - allowed_ids
                if bad:
                    invalid_mask_values.append(
                        {
                            "split": split_name,
                            "file": rec["processed_mask_relpath"],
                            "invalid_ids": sorted(bad),
                        }
                    )
                if not np.any(msk_arr != ignore_index):
                    empty_processed_masks.append(
                        {"split": split_name, "id": rid, "file": rec["processed_mask_relpath"]}
                    )
            except Exception as exc:
                corrupted_files.append({"file": rec["processed_mask_relpath"], "error": str(exc)})

    leakage = {
        "train_val_id_overlap": len(split_ids["train"] & split_ids["val"]),
        "train_test_id_overlap": len(split_ids["train"] & split_ids["test"]),
        "val_test_id_overlap": len(split_ids["val"] & split_ids["test"]),
        "train_val_group_overlap": len(split_groups["train"] & split_groups["val"]),
    }

    # Validate PyTorch Dataset & DataLoader tensor schema + CrossEntropyLoss compatibility
    loaders = build_dataloaders(config_path=config_path, project_root=root)
    pytorch_checks: Dict[str, Any] = {}
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    num_classes = int(cfg["mask"]["num_classes"])

    for split_name, loader in loaders.items():
        t0 = time.perf_counter()
        batches_tested = 0
        samples_tested = 0
        sample_img_shape = None
        sample_msk_shape = None
        sample_img_dtype = None
        sample_msk_dtype = None
        loss_val = None

        for imgs, masks in loader:
            batches_tested += 1
            samples_tested += imgs.shape[0]
            if batches_tested == 1:
                sample_img_shape = list(imgs.shape)
                sample_msk_shape = list(masks.shape)
                sample_img_dtype = str(imgs.dtype)
                sample_msk_dtype = str(masks.dtype)

                assert imgs.ndim == 4 and imgs.shape[1:] == (3, expected_h, expected_w), (
                    f"Unexpected image batch shape {imgs.shape}"
                )
                assert masks.ndim == 3 and masks.shape[1:] == (expected_h, expected_w), (
                    f"Unexpected mask batch shape {masks.shape}"
                )
                assert imgs.dtype == torch.float32, f"Expected float32, got {imgs.dtype}"
                assert masks.dtype == torch.int64, f"Expected int64, got {masks.dtype}"
                assert torch.isfinite(imgs).all(), "Non-finite values in image tensor"

                # Dummy logit tensor to validate CrossEntropyLoss(ignore_index=255) forward pass
                dummy_logits = torch.zeros(
                    (imgs.shape[0], num_classes, expected_h, expected_w),
                    dtype=torch.float32,
                    requires_grad=True,
                )
                loss = criterion(dummy_logits, masks)
                loss.backward()
                loss_val = round(float(loss.item()), 6)

            if batches_tested >= 5:
                break

        dt = max(time.perf_counter() - t0, 1e-6)
        pytorch_checks[split_name] = {
            "dataset_length": len(loader.dataset),
            "batches_benchmarked": batches_tested,
            "samples_benchmarked": samples_tested,
            "throughput_samples_per_sec": round(samples_tested / dt, 2),
            "batch_image_shape": sample_img_shape,
            "batch_mask_shape": sample_msk_shape,
            "image_dtype": sample_img_dtype,
            "mask_dtype": sample_msk_dtype,
            "cross_entropy_ignore255_loss_check": loss_val,
        }

    critical_failures = (
        len(missing_files)
        + len(corrupted_files)
        + len(dimension_mismatches)
        + len(invalid_mask_values)
        + len(empty_processed_masks)
        + len(duplicate_ids)
        + len(orphan_files)
        + sum(leakage.values())
    )

    report = {
        "status": "PASS" if critical_failures == 0 else "FAIL",
        "total_validated_pairs": len(seen_all_ids),
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "expected_target_size": [expected_h, expected_w],
        "ignore_index": ignore_index,
        "observed_mask_pixel_values": sorted(observed_mask_ids),
        "checks": {
            "missing_files_count": len(missing_files),
            "corrupted_files_count": len(corrupted_files),
            "dimension_mismatch_count": len(dimension_mismatches),
            "invalid_mask_value_count": len(invalid_mask_values),
            "empty_processed_mask_count": len(empty_processed_masks),
            "duplicate_id_count": len(duplicate_ids),
            "orphan_file_count": len(orphan_files),
            "leakage_checks": leakage,
        },
        "details": {
            "missing_files": missing_files[:20],
            "corrupted_files": corrupted_files[:20],
            "dimension_mismatches": dimension_mismatches[:20],
            "invalid_mask_values": invalid_mask_values[:20],
            "empty_processed_masks": empty_processed_masks[:20],
            "duplicate_ids": duplicate_ids[:20],
            "orphan_files": orphan_files[:20],
        },
        "pytorch_dataloader_validation": pytorch_checks,
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Saved preprocessing validation report to %s (status=%s)", report_out, report["status"])

    if critical_failures > 0:
        raise RuntimeError(
            f"Preprocessing validation FAILED with {critical_failures} critical issue(s). "
            f"Inspect {report_out} for details."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessed datasets and PyTorch DataLoaders.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    validate_processed_dataset(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
