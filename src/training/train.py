"""Training engine for MANGAL DRISHTI semantic segmentation models.

Usage:
    python -m src.training.train --model unet --epochs 10 --lr 0.0001 --config configs/data_config.yaml
    python -m src.training.train --model deeplabv3plus --epochs 10 --lr 0.0001 --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataloader import build_dataloaders
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import CombinedCEDiceLoss, load_step2_class_weights
from src.training.metrics import SegmentationMetrics

LOGGER = logging.getLogger("mangal_drishti.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_hardware(model_name: str, batch_size_override: Optional[int] = None) -> Dict[str, Any]:
    """Detect GPU/VRAM/RAM and adapt batch size & gradient accumulation safely."""
    has_cuda = torch.cuda.is_available()
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if has_cuda
        else 0.0
    )

    if batch_size_override is not None:
        bs = int(batch_size_override)
        accum = max(1, 8 // bs)
    elif not has_cuda:
        bs, accum = 2, 4
    elif vram_gb <= 4.5:
        bs, accum = 4, 2
    elif vram_gb <= 8.5:
        bs, accum = 8, 1
    else:
        bs, accum = 12, 1

    return {
        "device": "cuda" if has_cuda else "cpu",
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "pytorch_version": str(torch.__version__),
        "amp_enabled": has_cuda,
        "batch_size": bs,
        "grad_accum_steps": accum,
        "effective_batch_size": bs * accum,
        "model_name": model_name,
    }


def build_model(model_name: str, num_classes: int = 4, pretrained: bool = True) -> torch.nn.Module:
    if model_name == "unet":
        return UNetResNet34(num_classes=num_classes, pretrained=pretrained)
    if model_name == "deeplabv3plus":
        return DeepLabV3PlusResNet50(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model '{model_name}'. Choose 'unet' or 'deeplabv3plus'.")


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
    ignore_index: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
    desc: str = "Train",
) -> Tuple[float, Dict[str, Any]]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)

    total_loss = 0.0
    total_samples = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.enable_grad()
    num_batches = len(loader)

    with torch.set_grad_enabled(is_train):
        for step_idx, (imgs, masks) in enumerate(tqdm(loader, desc=desc, unit="batch")):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                loss_scaled = loss / grad_accum_steps
                if scaler is not None and amp_enabled:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == num_batches:
                    if scaler is not None and amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

            bs = imgs.shape[0]
            total_loss += float(loss.item()) * bs
            total_samples += bs
            metrics.update(logits.detach(), masks)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics.compute()


def train_model(
    model_name: str,
    config_path: Path,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dice_weight: float = 0.0,
    patience: int = 4,
    batch_size: Optional[int] = None,
    pretrained: bool = True,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = get_project_root()
    cfg = load_config(config_path)
    seed = int(cfg["project"]["random_seed"])
    set_seed(seed)

    hw = detect_hardware(model_name, batch_size_override=batch_size)
    device = torch.device(hw["device"])
    LOGGER.info("Hardware detected: %s", json.dumps(hw))

    exp_dir = root / "experiments" / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "predictions").mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        config_path=config_path,
        batch_size=hw["batch_size"],
        num_workers=0,
        project_root=root,
    )

    num_classes = int(cfg["mask"]["num_classes"])
    ignore_index = int(cfg["mask"]["ignore_index"])
    weights_path = resolve_path(cfg["paths"]["class_weights_file"], root)
    class_weights, weight_dict = load_step2_class_weights(
        weights_path, method="median_frequency_capped", num_classes=num_classes
    )
    class_weights = class_weights.to(device)

    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = CombinedCEDiceLoss(
        class_weights=class_weights,
        ignore_index=ignore_index,
        dice_weight=dice_weight,
        num_classes=num_classes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=hw["amp_enabled"])

    start_epoch = 1
    best_miou = -1.0
    best_val_loss = float("inf")
    best_val_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, device, optimizer, scheduler, scaler)
        start_epoch = int(ckpt["epoch"]) + 1
        best_miou = float(ckpt.get("best_miou", -1.0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        LOGGER.info("Resumed from %s at epoch %d (best_miou=%.4f)", resume_path, start_epoch, best_miou)

    model_cfg = {
        "model_name": model_name,
        "pretrained_requested": pretrained,
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "epochs_max": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "dice_weight": dice_weight,
        "class_weights_method": "median_frequency_capped",
        "class_weights": weight_dict,
        "hardware": hw,
        "data_config_path": str(config_path),
    }
    (exp_dir / "model_config.yaml").write_text(yaml.safe_dump(model_cfg, sort_keys=False), encoding="utf-8")

    no_improve_epochs = 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.perf_counter()
        train_loss, train_m = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            hw["amp_enabled"],
            num_classes,
            ignore_index,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=hw["grad_accum_steps"],
            desc=f"[{model_name}] Epoch {epoch}/{epochs} Train",
        )
        val_loss, val_m = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            hw["amp_enabled"],
            num_classes,
            ignore_index,
            optimizer=None,
            desc=f"[{model_name}] Epoch {epoch}/{epochs} Val",
        )
        scheduler.step()
        ep_sec = round(time.perf_counter() - ep_t0, 2)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": round(cur_lr, 8),
            "train_loss": round(train_loss, 6),
            "train_mIoU": train_m["mIoU"],
            "val_loss": round(val_loss, 6),
            "val_mIoU": val_m["mIoU"],
            "val_mean_dice": val_m["mean_dice"],
            "val_pixel_acc": val_m["pixel_accuracy"],
            "val_soil_iou": val_m["per_class_iou"]["soil"],
            "val_bedrock_iou": val_m["per_class_iou"]["bedrock"],
            "val_sand_iou": val_m["sand_iou"],
            "val_big_rock_iou": val_m["big_rock_iou"],
            "val_sand_recall": val_m["sand_recall"],
            "val_big_rock_recall": val_m["big_rock_recall"],
            "epoch_seconds": ep_sec,
        }
        history.append(row)

        LOGGER.info(
            "Epoch %02d/%02d (%.1fs) | train_loss=%.4f val_loss=%.4f | val_mIoU=%.4f | sand_IoU=%.4f big_rock_IoU=%.4f (rec=%.4f)",
            epoch,
            epochs,
            ep_sec,
            train_loss,
            val_loss,
            val_m["mIoU"],
            val_m["sand_iou"],
            val_m["big_rock_iou"],
            val_m["big_rock_recall"],
        )

        improved = val_m["mIoU"] > best_miou
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            best_miou = float(val_m["mIoU"])
            best_val_metrics = {
                "best_epoch": epoch,
                "best_val_loss_at_best_miou": round(val_loss, 6),
                **val_m,
            }
            save_checkpoint(
                exp_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_miou,
                best_val_loss,
                model_name,
                extra_meta=model_cfg,
            )
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        save_checkpoint(
            exp_dir / "last_checkpoint.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_miou,
            best_val_loss,
            model_name,
            extra_meta=model_cfg,
        )

        # Persist history CSV after every epoch
        hist_path = exp_dir / "training_history.csv"
        with hist_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if no_improve_epochs >= patience:
            LOGGER.info("Early stopping triggered after %d epochs without val mIoU improvement.", patience)
            break

    total_sec = round(time.perf_counter() - t_start, 2)
    final_metrics = {
        "model_name": model_name,
        "status": "trained",
        "epochs_completed": len(history),
        "training_duration_seconds": total_sec,
        "hardware": hw,
        "best_val_loss": round(best_val_loss, 6),
        "best_validation_metrics": best_val_metrics,
        "checkpoint_path": str(exp_dir / "best_model.pth"),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    LOGGER.info("Training finished for %s in %.1fs | Best val mIoU=%.4f", model_name, total_sec, best_miou)
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MANGAL DRISHTI segmentation model.")
    parser.add_argument("--model", choices=["unet", "deeplabv3plus"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train_model(
        model_name=args.model,
        config_path=args.config,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=args.dice_weight,
        patience=args.patience,
        batch_size=args.batch_size,
        pretrained=not args.no_pretrained,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
