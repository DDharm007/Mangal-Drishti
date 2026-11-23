"""Reusable PyTorch Dataset implementation for MANGAL DRISHTI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.transforms import SegmentationTransformPipeline


class MangalDrishtiDataset(Dataset):
    """PyTorch Dataset returning (image [3, H, W] float32, mask [H, W] int64) pairs."""

    def __init__(
        self,
        split: str,
        config_path: Path | str = Path("configs/data_config.yaml"),
        transform: Optional[SegmentationTransformPipeline] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split '{split}'. Expected train, val, or test.")

        self.root = project_root or get_project_root()
        self.cfg = load_config(config_path)
        self.split = split

        self.processed_dir = resolve_path(self.cfg["paths"]["processed_dir"], self.root)
        manifest_path = resolve_path(self.cfg["paths"]["split_manifest_file"], self.root)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest missing at {manifest_path}. Run python -m src.data.preprocessing first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records: List[Dict[str, Any]] = manifest["splits"][split]
        if not self.records:
            raise RuntimeError(f"Split '{split}' contains zero records in {manifest_path}")

        # Verify image-mask filename correspondence at initialization
        for rec in self.records:
            img_p = self.processed_dir / rec["processed_image_relpath"]
            msk_p = self.processed_dir / rec["processed_mask_relpath"]
            if not img_p.exists() or not msk_p.exists():
                raise FileNotFoundError(
                    f"Missing processed pair for {rec['id']} in split '{split}': "
                    f"img={img_p.exists()}, mask={msk_p.exists()}"
                )
            if img_p.stem != msk_p.stem:
                raise ValueError(
                    f"Filename mismatch: image stem '{img_p.stem}' != mask stem '{msk_p.stem}'"
                )

        # Load training-only mean & std from preprocessing_config.json
        prep_cfg_path = resolve_path(self.cfg["paths"]["preprocessing_config_file"], self.root)
        mean = std = None
        if prep_cfg_path.exists():
            prep_cfg = json.loads(prep_cfg_path.read_text(encoding="utf-8"))
            mean = prep_cfg.get("normalization", {}).get("active_mean")
            std = prep_cfg.get("normalization", {}).get("active_std")

        self.transform = transform or SegmentationTransformPipeline(
            cfg=self.cfg,
            split=split,
            mean=mean,
            std=std,
        )
        self.ignore_index = int(self.cfg["mask"]["ignore_index"])
        self.num_classes = int(self.cfg["mask"]["num_classes"])

    def __len__(self) -> int:
        return len(self.records)

    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        """Return manifest metadata dictionary for sample index."""
        return dict(self.records[index])

    def load_raw_processed_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed uint8 RGB image [H, W, 3] and uint8 mask [H, W]."""
        rec = self.records[index]
        img_p = self.processed_dir / rec["processed_image_relpath"]
        msk_p = self.processed_dir / rec["processed_mask_relpath"]

        with Image.open(img_p) as im:
            img_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        with Image.open(msk_p) as msk:
            msk_arr = np.asarray(msk, dtype=np.uint8)
        if msk_arr.ndim == 3:
            msk_arr = msk_arr[:, :, 0]
        return img_arr, msk_arr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr, msk_arr = self.load_raw_processed_pair(index)
        img_tensor, msk_tensor = self.transform(img_arr, msk_arr)
        return img_tensor, msk_tensor
