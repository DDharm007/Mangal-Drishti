"""Realistic, deterministic augmentation and normalization pipeline for Mars rover imagery.

Uses OpenCV + NumPy + PyTorch so geometric transformations apply identically
to image and mask (bilinear for RGB image, nearest-neighbor with ignore_index
border for mask) while photometric transformations apply strictly to the image.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


class SegmentationTransformPipeline:
    """Configurable preprocessing and augmentation pipeline for semantic segmentation."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split '{split}'. Expected train, val, or test.")
        self.cfg = cfg
        self.split = split
        self.target_h = int(cfg["image"]["target_size"][0])
        self.target_w = int(cfg["image"]["target_size"][1])
        self.ignore_index = int(cfg["mask"]["ignore_index"])
        self.scale_to_unit = bool(cfg["normalization"].get("scale_to_unit", True))
        self.mean = np.array(
            mean if mean is not None else cfg["normalization"]["imagenet_mean"],
            dtype=np.float32,
        )
        self.std = np.array(
            std if std is not None else cfg["normalization"]["imagenet_std"],
            dtype=np.float32,
        )
        self.aug_cfg = cfg.get("augmentation", {})
        self.enable_aug = (split == "train") and bool(
            self.aug_cfg.get("enabled_for_train", True)
        )
        base_seed = int(seed if seed is not None else cfg["project"]["random_seed"])
        self.rng = np.random.default_rng(base_seed)

    def _apply_geometric(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        # 1. Horizontal flip (valid symmetry across Mars rover panoramic camera views)
        hf = self.aug_cfg.get("horizontal_flip", {})
        if hf.get("enabled", True) and rng.random() < float(hf.get("p", 0.5)):
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # 2. Combined rotation, scale, and shift via single affine matrix (minimizes resampling)
        rot_cfg = self.aug_cfg.get("rotation", {})
        sc_cfg = self.aug_cfg.get("scale_crop", {})
        sh_cfg = self.aug_cfg.get("shift", {})

        angle = 0.0
        scale = 1.0
        dx = 0.0
        dy = 0.0
        apply_affine = False

        if rot_cfg.get("enabled", True) and rng.random() < float(rot_cfg.get("p", 0.4)):
            lim = float(rot_cfg.get("limit_degrees", 15.0))
            angle = float(rng.uniform(-lim, lim))
            apply_affine = True

        if sc_cfg.get("enabled", True) and rng.random() < float(sc_cfg.get("p", 0.4)):
            s_lo, s_hi = sc_cfg.get("scale_range", [0.85, 1.15])
            scale = float(rng.uniform(float(s_lo), float(s_hi)))
            apply_affine = True

        if sh_cfg.get("enabled", True) and rng.random() < float(sh_cfg.get("p", 0.3)):
            s_lim = float(sh_cfg.get("shift_limit", 0.05))
            dx = float(rng.uniform(-s_lim, s_lim)) * w
            dy = float(rng.uniform(-s_lim, s_lim)) * h
            apply_affine = True

        if apply_affine:
            center = (w / 2.0, h / 2.0)
            mat = cv2.getRotationMatrix2D(center, angle, scale)
            mat[0, 2] += dx
            mat[1, 2] += dy
            image = cv2.warpAffine(
                image,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                mat,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.ignore_index,
            )

        return image, mask

    def _apply_photometric(
        self, image: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply photometric perturbations STRICTLY to the image (never to the mask)."""
        img = image.astype(np.float32)

        # 1. CLAHE local contrast enhancement (helps adapt to dark NavCam imagery)
        clahe_cfg = self.aug_cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True) and rng.random() < float(clahe_cfg.get("p", 0.25)):
            clip = float(clahe_cfg.get("clip_limit", 2.0))
            grid = tuple(int(x) for x in clahe_cfg.get("tile_grid_size", [8, 8]))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            u8 = np.clip(img, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

        # 2. Brightness & contrast adjustment
        bc_cfg = self.aug_cfg.get("brightness_contrast", {})
        if bc_cfg.get("enabled", True) and rng.random() < float(bc_cfg.get("p", 0.5)):
            b_lim = float(bc_cfg.get("brightness_limit", 0.15))
            c_lim = float(bc_cfg.get("contrast_limit", 0.15))
            alpha = float(rng.uniform(1.0 - c_lim, 1.0 + c_lim))
            beta = float(rng.uniform(-b_lim, b_lim)) * 255.0
            img = np.clip(img * alpha + beta, 0.0, 255.0)

        # 3. Mild gamma adjustment
        gam_cfg = self.aug_cfg.get("gamma", {})
        if gam_cfg.get("enabled", True) and rng.random() < float(gam_cfg.get("p", 0.35)):
            g_lo, g_hi = gam_cfg.get("gamma_range", [0.85, 1.15])
            gamma = float(rng.uniform(float(g_lo), float(g_hi)))
            normed = np.clip(img / 255.0, 0.0, 1.0)
            img = np.power(normed, gamma) * 255.0

        # 4. Mild Gaussian blur
        blur_cfg = self.aug_cfg.get("gaussian_blur", {})
        if blur_cfg.get("enabled", True) and rng.random() < float(blur_cfg.get("p", 0.15)):
            k = int(blur_cfg.get("kernel_size", 3))
            if k % 2 == 0:
                k += 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 5. Mild Gaussian sensor noise
        noise_cfg = self.aug_cfg.get("gaussian_noise", {})
        if noise_cfg.get("enabled", True) and rng.random() < float(noise_cfg.get("p", 0.2)):
            s_lo, s_hi = noise_cfg.get("std_range", [0.01, 0.03])
            sigma = float(rng.uniform(float(s_lo), float(s_hi))) * 255.0
            noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 255.0)

        return np.clip(img, 0.0, 255.0).astype(np.uint8)

    def apply_numpy(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return uint8 (image, mask) after deterministic resize and optional train augmentation."""
        if image.shape[:2] != (self.target_h, self.target_w):
            image = cv2.resize(
                image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
            )
        if mask.shape[:2] != (self.target_h, self.target_w):
            mask = cv2.resize(
                mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST
            )

        if self.enable_aug:
            active_rng = rng if rng is not None else self.rng
            image, mask = self._apply_geometric(image, mask, active_rng)
            image = self._apply_photometric(image, active_rng)

        return image, mask

    def normalize_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Scale uint8 RGB [H, W, 3] to normalized float32 tensor [3, H, W]."""
        arr = image.astype(np.float32)
        if self.scale_to_unit:
            arr = arr / 255.0
        arr = (arr - self.mean) / self.std
        chw = np.transpose(arr, (2, 0, 1)).copy()
        return torch.from_numpy(chw).to(dtype=torch.float32)

    def denormalize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Invert normalization back to uint8 RGB [H, W, 3] for visualization."""
        chw = tensor.detach().cpu().numpy()
        hwc = np.transpose(chw, (1, 2, 0))
        arr = hwc * self.std + self.mean
        if self.scale_to_unit:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform uint8 (image, mask) into PyTorch tensors ([3, H, W], [H, W])."""
        aug_img, aug_mask = self.apply_numpy(image, mask, rng=rng)
        img_tensor = self.normalize_to_tensor(aug_img)
        mask_tensor = torch.from_numpy(aug_mask.astype(np.int64)).to(dtype=torch.int64)
        return img_tensor, mask_tensor
