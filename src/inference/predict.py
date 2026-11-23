"""Inference and prediction visualization engine for MANGAL DRISHTI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataset import MangalDrishtiDataset
from src.data.preprocessing import load_and_preprocess_image
from src.data.transforms import SegmentationTransformPipeline
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint

LOGGER = logging.getLogger("mangal_drishti.predict")

CLASS_PALETTE: Dict[int, tuple[int, int, int]] = {
    0: (166, 97, 26),     # Soil
    1: (55, 126, 184),    # Bedrock
    2: (245, 194, 66),    # Sand
    3: (228, 26, 28),     # Big Rock
}
CLASS_NAMES: Dict[int, str] = {
    0: "Soil (0)",
    1: "Bedrock (1)",
    2: "Sand (2)",
    3: "Big Rock (3)",
}


def colorize_prediction(pred_mask: np.ndarray) -> np.ndarray:
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in CLASS_PALETTE.items():
        rgb[pred_mask == cid] = col
    return rgb


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_path: Path,
    pipeline: SegmentationTransformPipeline,
    cfg: Dict[str, object],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    target_size = (int(cfg["image"]["target_size"][0]), int(cfg["image"]["target_size"][1]))
    rgb = load_and_preprocess_image(
        image_path,
        target_size=target_size,
        convert_to_rgb=bool(cfg["image"]["convert_to_rgb"]),
        interpolation=str(cfg["image"]["interpolation"]),
        aspect_ratio_strategy=str(cfg["image"]["aspect_ratio_strategy"]),
    )
    tensor = pipeline.normalize_to_tensor(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # [4, H, W]
    pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = np.max(probs, axis=0).astype(np.float32)

    color_mask = colorize_prediction(pred_mask)
    overlay = np.clip(0.52 * rgb.astype(np.float32) + 0.48 * color_mask.astype(np.float32), 0, 255).astype(np.uint8)
    return {
        "rgb": rgb,
        "pred_mask": pred_mask,
        "color_mask": color_mask,
        "confidence": confidence,
        "overlay": overlay,
    }


def save_prediction_bundle(
    stem: str,
    outputs: Dict[str, np.ndarray],
    output_dir: Path,
    model_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save standalone mask, confidence, and overlay
    Image.fromarray(outputs["color_mask"]).save(output_dir / f"{stem}_pred_mask.png")
    conf_u8 = np.clip(outputs["confidence"] * 255.0, 0, 255).astype(np.uint8)
    conf_heat = cv2.cvtColor(cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    Image.fromarray(conf_heat).save(output_dir / f"{stem}_confidence.png")
    Image.fromarray(outputs["overlay"]).save(output_dir / f"{stem}_overlay.png")

    # Save 4-panel visual inspection figure
    pred = outputs["pred_mask"]
    counts = np.bincount(pred.ravel(), minlength=4)
    total = pred.size
    dist_str = " | ".join(f"{CLASS_NAMES[i].split()[0]}: {counts[i]/total*100:.1f}%" for i in range(4))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=140)
    axes[0].imshow(outputs["rgb"])
    axes[0].set_title("Input Rover Image (512x512)", fontsize=10, fontweight="bold")
    axes[1].imshow(outputs["color_mask"])
    axes[1].set_title(f"Predicted Mask\n({dist_str})", fontsize=9, fontweight="bold")
    im_c = axes[2].imshow(outputs["confidence"], cmap="viridis", vmin=0.25, vmax=1.0)
    axes[2].set_title(
        f"Softmax Confidence\n(Mean: {outputs['confidence'].mean():.3f})", fontsize=10, fontweight="bold"
    )
    plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(outputs["overlay"])
    axes[3].set_title("Prediction Overlay", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.axis("off")

    handles = [
        mpatches.Patch(color=np.array(CLASS_PALETTE[k]) / 255.0, label=CLASS_NAMES[k])
        for k in range(4)
    ]
    fig.suptitle(f"MANGAL DRISHTI ({model_name.upper()}) Prediction — {stem}", fontsize=12, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    panel_path = output_dir / f"{stem}_panel.png"
    fig.savefig(panel_path, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def run_inference(
    model_name: str,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    input_path: Optional[Path] = None,
    split: str = "val",
    num_samples: int = 8,
) -> List[Path]:
    root = get_project_root()
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep_cfg = json.loads(
        resolve_path(cfg["paths"]["preprocessing_config_file"], root).read_text(encoding="utf-8")
    )
    pipeline = SegmentationTransformPipeline(
        cfg=cfg,
        split="test",
        mean=prep_cfg["normalization"]["active_mean"],
        std=prep_cfg["normalization"]["active_std"],
    )

    model = (
        UNetResNet34(num_classes=4, pretrained=False)
        if model_name == "unet"
        else DeepLabV3PlusResNet50(num_classes=4, pretrained=False)
    ).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    image_paths: List[Path] = []
    if input_path is not None:
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            image_paths = sorted(
                p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )[:num_samples]
    else:
        ds = MangalDrishtiDataset(split=split, config_path=config_path, project_root=root)
        # Pick evenly spaced indices from the split
        indices = np.linspace(0, len(ds) - 1, min(num_samples, len(ds)), dtype=int)
        proc_dir = resolve_path(cfg["paths"]["processed_dir"], root)
        image_paths = [proc_dir / ds.records[int(i)]["processed_image_relpath"] for i in indices]

    saved: List[Path] = []
    for p in image_paths:
        res = predict_single_image(model, p, pipeline, cfg, device)
        panel = save_prediction_bundle(p.stem, res, output_dir, model_name)
        saved.append(panel)
        LOGGER.info("Saved prediction panel: %s", panel)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with trained MANGAL DRISHTI model.")
    parser.add_argument("--model", choices=["unet", "deeplabv3plus"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--input", type=Path, default=None, help="Optional single image or directory.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    run_inference(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        input_path=args.input,
        split=args.split,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
