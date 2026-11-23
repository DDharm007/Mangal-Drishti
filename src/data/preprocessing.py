"""End-to-end preprocessing pipeline for MANGAL DRISHTI.

Transforms raw NASA/JPL AI4MARS rover imagery and categorical masks into a
clean, reproducible, PyTorch-ready processed dataset under `datasets/processed/`.

Usage:
    python -m src.data.preprocessing --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.class_weights import compute_and_save_class_weights
from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.mask_encoder import MaskEncoder, save_class_mapping
from src.data.split_dataset import generate_splits

LOGGER = logging.getLogger("mangal_drishti.preprocessing")


def _compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute stable SHA-256 fingerprint over preprocessing-critical settings."""
    relevant = {
        "random_seed": cfg["project"]["random_seed"],
        "image": cfg["image"],
        "mask": cfg["mask"],
        "normalization": cfg["normalization"],
        "splitting": cfg["splitting"],
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_preprocess_image(
    img_path: Path,
    target_size: Tuple[int, int],
    convert_to_rgb: bool = True,
    interpolation: str = "bilinear",
    aspect_ratio_strategy: str = "direct_square",
    pad_value: int = 0,
) -> np.ndarray:
    """Load raw rover image, validate dimensions, convert to RGB, and resize safely."""
    with Image.open(img_path) as im:
        im.load()
        if convert_to_rgb and im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im)

    if arr.ndim == 2 and convert_to_rgb:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    target_h, target_w = int(target_size[0]), int(target_size[1])
    src_h, src_w = arr.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return arr

    cv_interp = (
        cv2.INTER_CUBIC if interpolation.lower() == "bicubic" else cv2.INTER_LINEAR
    )

    if aspect_ratio_strategy == "letterbox_pad" and src_h != src_w:
        scale = min(target_h / src_h, target_w / src_w)
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))
        resized_inner = cv2.resize(arr, (new_w, new_h), interpolation=cv_interp)
        channels = arr.shape[2] if arr.ndim == 3 else 1
        canvas_shape = (target_h, target_w, channels) if arr.ndim == 3 else (target_h, target_w)
        out = np.full(canvas_shape, pad_value, dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        out[top : top + new_h, left : left + new_w] = resized_inner
        return out

    return cv2.resize(arr, (target_w, target_h), interpolation=cv_interp)


def _existing_outputs_valid(
    processed_dir: Path,
    prep_cfg_path: Path,
    split_manifest_path: Path,
    expected_hash: str,
) -> bool:
    """Check whether cached processed outputs match current configuration."""
    if not prep_cfg_path.exists() or not split_manifest_path.exists():
        return False
    try:
        prev_cfg = json.loads(prep_cfg_path.read_text(encoding="utf-8"))
        if prev_cfg.get("config_hash") != expected_hash:
            return False
        manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        for split_name, records in manifest["splits"].items():
            for rec in records:
                if not (processed_dir / rec["processed_image_relpath"]).exists():
                    return False
                if not (processed_dir / rec["processed_mask_relpath"]).exists():
                    return False
        return True
    except Exception:
        return False


def run_preprocessing(
    cfg: Dict[str, Any],
    force: bool = False,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute full preprocessing pipeline and persist metadata & statistics."""
    root = project_root or get_project_root()
    raw_root = resolve_path(cfg["paths"]["raw_ai4mars_dir"], root)
    processed_dir = resolve_path(cfg["paths"]["processed_dir"], root)
    prep_cfg_path = resolve_path(cfg["paths"]["preprocessing_config_file"], root)
    split_manifest_path = resolve_path(cfg["paths"]["split_manifest_file"], root)

    cfg_hash = _compute_config_hash(cfg)
    if not force and _existing_outputs_valid(
        processed_dir, prep_cfg_path, split_manifest_path, cfg_hash
    ):
        LOGGER.info(
            "Valid processed dataset already exists for config_hash=%s. Skipping regeneration (use --force to rebuild).",
            cfg_hash[:12],
        )
        return json.loads(prep_cfg_path.read_text(encoding="utf-8"))

    # 1. Export verified official class mapping
    save_class_mapping(cfg, root)

    # 2. Generate leakage-free SCLK-clustered splits
    split_manifest = generate_splits(cfg, root)

    encoder = MaskEncoder(
        label_keys_path=resolve_path(cfg["paths"]["raw_label_keys"], root),
        ignore_index=int(cfg["mask"]["ignore_index"]),
    )
    target_h, target_w = int(cfg["image"]["target_size"][0]), int(cfg["image"]["target_size"][1])
    target_size = (target_h, target_w)
    aspect_strategy = str(cfg["image"]["aspect_ratio_strategy"])
    img_interp = str(cfg["image"]["interpolation"])
    convert_rgb = bool(cfg["image"]["convert_to_rgb"])
    pad_val = int(cfg["image"]["image_pad_value"])
    apply_mxy = bool(cfg["mask"].get("apply_mxy_rover_mask", True))
    apply_rng = bool(cfg["mask"].get("apply_rng30m_range_mask", True))

    failed_files: List[Dict[str, str]] = []
    processed_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    # Accumulators strictly on `train` split for [0, 1] scaled RGB channel statistics
    train_sum = np.zeros(3, dtype=np.float64)
    train_sq_sum = np.zeros(3, dtype=np.float64)
    train_pixel_total = 0

    for split_name in ("train", "val", "test"):
        img_out_dir = processed_dir / split_name / "images"
        msk_out_dir = processed_dir / split_name / "masks"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        msk_out_dir.mkdir(parents=True, exist_ok=True)

        records = split_manifest["splits"][split_name]
        for rec in tqdm(records, desc=f"Preprocessing {split_name}", unit="pair"):
            raw_img_p = raw_root / rec["image_relpath"]
            raw_msk_p = raw_root / rec["mask_relpath"]
            mxy_p = raw_root / rec["mxy_relpath"] if (apply_mxy and rec.get("mxy_relpath")) else None
            rng_p = raw_root / rec["rng_relpath"] if (apply_rng and rec.get("rng_relpath")) else None

            try:
                proc_img = load_and_preprocess_image(
                    raw_img_p,
                    target_size=target_size,
                    convert_to_rgb=convert_rgb,
                    interpolation=img_interp,
                    aspect_ratio_strategy=aspect_strategy,
                    pad_value=pad_val,
                )
                raw_mask = encoder.load_raw_mask(raw_msk_p)
                raw_mask = encoder.apply_auxiliary_exclusions(
                    raw_mask, mxy_path=mxy_p, rng_path=rng_p
                )
                proc_mask = encoder.encode_and_resize(
                    raw_mask,
                    target_size=target_size,
                    aspect_ratio_strategy=aspect_strategy,
                )

                out_img_p = processed_dir / rec["processed_image_relpath"]
                out_msk_p = processed_dir / rec["processed_mask_relpath"]
                Image.fromarray(proc_img).save(out_img_p, format="PNG")
                Image.fromarray(proc_mask, mode="L").save(out_msk_p, format="PNG")
                processed_counts[split_name] += 1

                if split_name == "train":
                    scaled = proc_img.astype(np.float64) / 255.0
                    train_sum += scaled.sum(axis=(0, 1))
                    train_sq_sum += np.square(scaled).sum(axis=(0, 1))
                    train_pixel_total += proc_img.shape[0] * proc_img.shape[1]

            except Exception as exc:
                LOGGER.error("Failed preprocessing pair %s (%s): %s", rec["id"], split_name, exc)
                failed_files.append(
                    {
                        "id": rec["id"],
                        "split": split_name,
                        "image_relpath": rec["image_relpath"],
                        "mask_relpath": rec["mask_relpath"],
                        "error": str(exc),
                    }
                )

    if failed_files:
        raise RuntimeError(
            f"Preprocessing encountered {len(failed_files)} failed file(s). First failure: {failed_files[0]}"
        )

    train_mean = (train_sum / max(train_pixel_total, 1)).tolist()
    train_var = (train_sq_sum / max(train_pixel_total, 1)) - np.square(train_mean)
    train_std = np.sqrt(np.maximum(train_var, 1e-12)).tolist()

    norm_mode = str(cfg["normalization"]["mode"])
    active_mean = (
        [round(float(x), 6) for x in train_mean]
        if norm_mode == "dataset_train"
        else [float(x) for x in cfg["normalization"]["imagenet_mean"]]
    )
    active_std = (
        [round(float(x), 6) for x in train_std]
        if norm_mode == "dataset_train"
        else [float(x) for x in cfg["normalization"]["imagenet_std"]]
    )

    prep_summary = {
        "project": cfg["project"]["name"],
        "phase": cfg["project"]["phase"],
        "config_hash": cfg_hash,
        "random_seed": int(cfg["project"]["random_seed"]),
        "image_preprocessing": {
            "target_size": [target_h, target_w],
            "aspect_ratio_strategy": aspect_strategy,
            "interpolation": img_interp,
            "convert_to_rgb": convert_rgb,
            "output_format": "png",
        },
        "mask_preprocessing": {
            "interpolation": "nearest",
            "ignore_index": int(cfg["mask"]["ignore_index"]),
            "num_classes": int(cfg["mask"]["num_classes"]),
            "apply_mxy_rover_mask": apply_mxy,
            "apply_rng30m_range_mask": apply_rng,
            "filter_empty_masks_train_val": bool(
                cfg["mask"].get("filter_empty_masks_train_val", True)
            ),
            "test_consensus_strategy": cfg["mask"]["test_consensus_strategy"],
        },
        "normalization": {
            "mode": norm_mode,
            "statistics_source": "processed_train_split_only",
            "scale_to_unit": bool(cfg["normalization"]["scale_to_unit"]),
            "computed_train_mean_rgb": [round(float(x), 6) for x in train_mean],
            "computed_train_std_rgb": [round(float(x), 6) for x in train_std],
            "active_mean": active_mean,
            "active_std": active_std,
        },
        "counts": {
            "processed_train": processed_counts["train"],
            "processed_val": processed_counts["val"],
            "processed_test": processed_counts["test"],
            "processed_total": sum(processed_counts.values()),
            "rejected_empty_masks": split_manifest["counts"]["rejected_pairs"],
            "failed_files_count": len(failed_files),
        },
        "failed_files": failed_files,
    }

    prep_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    prep_cfg_path.write_text(json.dumps(prep_summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved preprocessing configuration summary to %s", prep_cfg_path)

    # 3. Compute class imbalance & training-only class weights
    compute_and_save_class_weights(cfg, root)

    return prep_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MANGAL DRISHTI preprocessing pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of processed files even if config hash matches.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    run_preprocessing(cfg, force=args.force)


if __name__ == "__main__":
    main()
