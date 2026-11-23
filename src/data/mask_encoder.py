"""Mask encoding, official label-key verification, and Nearest-Neighbor resizing.

Enforces strict categorical mask integrity:
  - Verified from official NASA/JPL AI4MARS `label_keys.json` (`NAV` schema):
      0 -> 0 (soil)
      1 -> 1 (bedrock)
      2 -> 2 (sand)
      3 -> 3 (big rock)
      255 -> 255 (null / unlabeled / ignored)
  - Never uses bilinear/bicubic interpolation on masks (strictly NEAREST only).
  - Applies official MSL auxiliary MXY (rover hardware) and RNG-30M (>30m stereo
    range) binary exclusion masks when present and configured.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.data.config_utils import get_project_root, load_config, resolve_path

LOGGER = logging.getLogger("mangal_drishti.mask_encoder")

TRAINABLE_PALETTE_RGB: Dict[int, List[int]] = {
    0: [166, 97, 26],     # soil
    1: [55, 126, 184],    # bedrock
    2: [245, 194, 66],    # sand
    3: [228, 26, 28],     # big rock
    255: [40, 44, 52],    # null / ignored
}


class MaskEncoder:
    """Encodes and validates AI4MARS categorical segmentation masks."""

    def __init__(
        self,
        label_keys_path: Path,
        ignore_index: int = 255,
    ) -> None:
        if not label_keys_path.exists():
            raise FileNotFoundError(
                f"Official label_keys.json not found at {label_keys_path}"
            )
        self.raw_label_keys = json.loads(label_keys_path.read_text(encoding="utf-8"))
        if "NAV" not in self.raw_label_keys:
            raise KeyError("Expected 'NAV' schema in official label_keys.json")

        self.ignore_index = int(ignore_index)
        self.nav_schema: Dict[int, str] = {}
        self.orig_to_train: Dict[int, int] = {}
        self.train_id_to_name: Dict[int, str] = {}

        for raw_k, raw_name in self.raw_label_keys["NAV"].items():
            orig_id = int(raw_k)
            if raw_name is None:
                self.nav_schema[orig_id] = "null"
                self.orig_to_train[orig_id] = self.ignore_index
            else:
                cname = str(raw_name)
                self.nav_schema[orig_id] = cname
                self.orig_to_train[orig_id] = orig_id
                self.train_id_to_name[orig_id] = cname

        self.valid_train_ids = sorted(self.train_id_to_name.keys())
        self.allowed_output_ids = set(self.valid_train_ids) | {self.ignore_index}

        # Build fast uint8 lookup table (maps any unexpected value to 254 so validator catches it)
        self._lut = np.full(256, 254, dtype=np.uint8)
        for orig_id, train_id in self.orig_to_train.items():
            self._lut[orig_id] = np.uint8(train_id)

    def load_raw_mask(self, mask_path: Path) -> np.ndarray:
        """Load a grayscale categorical PNG mask without interpolation."""
        with Image.open(mask_path) as msk:
            msk.load()
            arr = np.asarray(msk)
        if arr.ndim == 3:
            if arr.shape[2] >= 1 and np.all(arr[:, :, 0] == arr[:, :, 1]):
                arr = arr[:, :, 0]
            else:
                raise ValueError(
                    f"Expected single-channel mask, got shape {arr.shape} at {mask_path}"
                )
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def apply_auxiliary_exclusions(
        self,
        mask: np.ndarray,
        mxy_path: Optional[Path] = None,
        rng_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Mask out rover body (MXY == 1) and >30m horizon (RNG-30M == 1) as ignore_index."""
        out = mask.copy()
        for aux_path in (mxy_path, rng_path):
            if aux_path is not None and aux_path.exists():
                with Image.open(aux_path) as am:
                    am.load()
                    aux_arr = np.asarray(am)
                if aux_arr.ndim == 3:
                    aux_arr = aux_arr[:, :, 0]
                if aux_arr.shape != out.shape:
                    raise ValueError(
                        f"Auxiliary mask shape {aux_arr.shape} != mask shape {out.shape}: {aux_path}"
                    )
                out[aux_arr > 0] = np.uint8(self.ignore_index)
        return out

    def encode_and_resize(
        self,
        raw_mask: np.ndarray,
        target_size: Tuple[int, int],
        aspect_ratio_strategy: str = "direct_square",
    ) -> np.ndarray:
        """Validate original values, map to contiguous training IDs, and resize via NEAREST."""
        raw_unique = set(int(v) for v in np.unique(raw_mask))
        unexpected_raw = raw_unique - set(self.orig_to_train.keys())
        if unexpected_raw:
            raise ValueError(
                f"Invalid raw pixel IDs {sorted(unexpected_raw)} outside official NAV keys {sorted(self.orig_to_train.keys())}"
            )

        encoded = self._lut[raw_mask]
        target_h, target_w = int(target_size[0]), int(target_size[1])

        if encoded.shape == (target_h, target_w):
            resized = encoded
        elif aspect_ratio_strategy == "letterbox_pad" and encoded.shape[0] != encoded.shape[1]:
            src_h, src_w = encoded.shape
            scale = min(target_h / src_h, target_w / src_w)
            new_h, new_w = max(1, int(round(src_h * scale))), max(1, int(round(src_w * scale)))
            interp_arr = cv2.resize(encoded, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized = np.full((target_h, target_w), self.ignore_index, dtype=np.uint8)
            top = (target_h - new_h) // 2
            left = (target_w - new_w) // 2
            resized[top : top + new_h, left : left + new_w] = interp_arr
        else:
            resized = cv2.resize(
                encoded, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )

        self.validate_encoded_mask(resized)
        return resized

    def validate_encoded_mask(self, mask: np.ndarray) -> None:
        """Assert that every pixel ID belongs strictly to allowed training/ignore IDs."""
        if mask.ndim != 2:
            raise ValueError(f"Encoded mask must be 2D [H, W], got {mask.shape}")
        unique_vals = set(int(v) for v in np.unique(mask))
        invalid = unique_vals - self.allowed_output_ids
        if invalid:
            raise ValueError(
                f"Corrupted mask values detected after encoding/resizing: {sorted(invalid)}. "
                f"Allowed values: {sorted(self.allowed_output_ids)}"
            )

    def has_valid_pixels(self, mask: np.ndarray) -> bool:
        """Return True if at least one pixel is not ignore_index."""
        return bool(np.any(mask != self.ignore_index))

    def build_class_mapping_metadata(self) -> Dict[str, object]:
        """Generate complete machine-readable class_mapping.json structure."""
        nav_entries = []
        for orig_id in sorted(self.nav_schema.keys()):
            cname = self.nav_schema[orig_id]
            is_ignored = orig_id == self.ignore_index or cname == "null"
            train_id = self.orig_to_train[orig_id]
            nav_entries.append(
                {
                    "original_id": orig_id,
                    "original_mask_value": orig_id,
                    "training_id": train_id,
                    "class_name": cname if not is_ignored else "unlabeled_null",
                    "trainable": not is_ignored,
                    "ignored": is_ignored,
                    "palette_rgb": TRAINABLE_PALETTE_RGB.get(orig_id, [128, 128, 128]),
                }
            )

        geo_beta = {
            str(k): str(v) if v is not None else "unlabeled_null"
            for k, v in self.raw_label_keys.get("M2020_GEO", {}).items()
        }

        return {
            "dataset": "NASA/JPL AI4MARS v0.6",
            "source_label_keys": "datasets/raw/ai4mars/label_keys.json",
            "active_taxonomy": "NAV",
            "num_trainable_classes": len(self.valid_train_ids),
            "ignore_index": self.ignore_index,
            "mask_interpolation_rule": "NEAREST_NEIGHBOR_ONLY",
            "classes": nav_entries,
            "official_auxiliary_taxonomy_reference": {
                "name": "M2020_GEO (Beta - Perseverance Geology Only, not used in Step 2 NAV training)",
                "total_entries": len(geo_beta),
                "mapping": geo_beta,
            },
        }


def save_class_mapping(cfg: Dict[str, object], project_root: Optional[Path] = None) -> Path:
    """Create and persist datasets/metadata/class_mapping.json."""
    root = project_root or get_project_root()
    label_keys_path = resolve_path(cfg["paths"]["raw_label_keys"], root)
    out_path = resolve_path(cfg["paths"]["class_mapping_file"], root)
    encoder = MaskEncoder(
        label_keys_path=label_keys_path,
        ignore_index=int(cfg["mask"]["ignore_index"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(encoder.build_class_mapping_metadata(), indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved class mapping metadata to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mask encoding and export class_mapping.json.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    cfg = load_config(args.config)
    save_class_mapping(cfg)


if __name__ == "__main__":
    main()
