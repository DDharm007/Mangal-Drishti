"""Controlled train/val optimization experiments for MANGAL DRISHTI.

Strictly initializes from completed Step 3 baselines, optimizes and selects checkpoints
using ONLY the train (1,407) and validation (363) splits, saves artifacts under
`experiments/evaluation/optimization/`, and evaluates validation-selected best checkpoints
on the untouched expert test set (526) for final before/after reporting.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from src.data.config_utils import get_project_root, load_config
from src.data.dataloader import build_dataloaders
from src.data.dataset import MangalDrishtiDataset
from src.evaluation.evaluate_test import evaluate_model_on_test
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import CombinedCEDiceLoss
from src.training.train import detect_hardware, run_epoch, set_seed

LOGGER = logging.getLogger("mangal_drishti.optimize")

TEMPERED_WEIGHTS = {
    "0": 0.65,   # Soil (was 0.545153)
    "1": 0.65,   # Bedrock (was 0.560134)
    "2": 3.80,   # Sand (was 4.657246)
    "3": 7.00,   # Big Rock (was 12.000000 -> reduces edge/shadow FP)
}


def run_controlled_experiment(
    exp_name: str,
    model_name: str,
    init_ckpt: Path,
    root: Path,
    cfg: Dict[str, Any],
    cfg_path: Path,
    epochs: int = 4,
    lr: float = 3e-5,
    dice_weight: float = 0.5,
) -> Dict[str, Any]:
    set_seed(int(cfg["project"]["random_seed"]))
    hw = detect_hardware(model_name)
    device = torch.device(hw["device"])

    exp_dir = root / "experiments" / "evaluation" / "optimization" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        config_path=cfg_path,
        batch_size=hw["batch_size"],
        num_workers=0,
        project_root=root,
    )

    num_classes = int(cfg["mask"]["num_classes"])
    ignore_index = int(cfg["mask"]["ignore_index"])
    class_weights = torch.tensor(
        [TEMPERED_WEIGHTS[str(i)] for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )

    model = (
        UNetResNet34(num_classes=num_classes, pretrained=False)
        if model_name == "unet"
        else DeepLabV3PlusResNet50(num_classes=num_classes, pretrained=False)
    ).to(device)
    load_checkpoint(init_ckpt, model, device)

    criterion = CombinedCEDiceLoss(
        class_weights=class_weights,
        ignore_index=ignore_index,
        dice_weight=dice_weight,
        num_classes=num_classes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=hw["amp_enabled"])

    # Measure validation metrics of the initial baseline before optimization epochs
    _, init_val_m = run_epoch(
        model, loaders["val"], criterion, device, hw["amp_enabled"],
        num_classes, ignore_index, optimizer=None, desc=f"[{exp_name}] Init Val",
    )

    best_miou = -1.0
    best_val_loss = float("inf")
    best_val_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    exp_cfg = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "initialized_from": str(init_ckpt.relative_to(root)).replace("\\", "/"),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": 1e-4,
        "dice_weight": dice_weight,
        "loss": "CombinedCEDiceLoss(tempered_weights + 0.5 * SoftDiceLoss)",
        "class_weights": TEMPERED_WEIGHTS,
        "selection_split": "val_only",
        "selection_criterion": "val_mIoU",
        "hardware": hw,
    }
    (exp_dir / "experiment_config.yaml").write_text(yaml.safe_dump(exp_cfg, sort_keys=False), encoding="utf-8")

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        train_loss, train_m = run_epoch(
            model, loaders["train"], criterion, device, hw["amp_enabled"],
            num_classes, ignore_index, optimizer=optimizer, scaler=scaler,
            grad_accum_steps=hw["grad_accum_steps"], desc=f"[{exp_name}] Ep {epoch}/{epochs} Train",
        )
        val_loss, val_m = run_epoch(
            model, loaders["val"], criterion, device, hw["amp_enabled"],
            num_classes, ignore_index, optimizer=None, desc=f"[{exp_name}] Ep {epoch}/{epochs} Val",
        )
        scheduler.step()
        ep_sec = round(time.perf_counter() - ep_t0, 2)

        row = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
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
            "val_big_rock_precision": val_m["per_class_precision"]["big rock"],
            "val_big_rock_recall": val_m["big_rock_recall"],
            "epoch_seconds": ep_sec,
        }
        history.append(row)
        LOGGER.info(
            "[%s] Ep %d/%d (%.1fs) | val_mIoU=%.4f | BR_IoU=%.4f BR_P=%.4f BR_R=%.4f",
            exp_name, epoch, epochs, ep_sec,
            val_m["mIoU"], val_m["big_rock_iou"],
            val_m["per_class_precision"]["big rock"], val_m["big_rock_recall"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_m["mIoU"] > best_miou:
            best_miou = float(val_m["mIoU"])
            best_val_metrics = {"best_epoch": epoch, "best_val_loss_at_best_miou": round(val_loss, 6), **val_m}
            save_checkpoint(
                exp_dir / "best_model.pth", model, optimizer, scheduler, scaler,
                epoch, best_miou, best_val_loss, model_name, extra_meta=exp_cfg,
            )

        save_checkpoint(
            exp_dir / "last_checkpoint.pth", model, optimizer, scheduler, scaler,
            epoch, best_miou, best_val_loss, model_name, extra_meta=exp_cfg,
        )
        with (exp_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    duration_sec = round(time.perf_counter() - t0, 2)

    # Evaluate validation-selected best checkpoint once on the untouched expert test set
    load_checkpoint(exp_dir / "best_model.pth", model, device)
    test_ds = MangalDrishtiDataset(split="test", config_path=cfg_path, project_root=root)
    test_res, _ = evaluate_model_on_test(model, test_ds, device, batch_size=4, desc=f"[{exp_name}] Expert Test")

    payload = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "epochs_completed": epochs,
        "duration_seconds": duration_sec,
        "baseline_validation_metrics": init_val_m,
        "best_validation_metrics": best_val_metrics,
        "expert_test_metrics": test_res,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    root = get_project_root()
    baseline_check = root / "experiments" / "evaluation" / "baseline" / "combined_comparison.json"
    if not baseline_check.exists():
        raise FileNotFoundError(
            f"Untouched baseline test evaluation must exist at {baseline_check} before running optimization."
        )

    cfg_path = root / "configs" / "data_config.yaml"
    cfg = load_config(cfg_path)

    dl_opt = run_controlled_experiment(
        exp_name="deeplabv3plus_ce_dice_balanced",
        model_name="deeplabv3plus",
        init_ckpt=root / "experiments" / "deeplabv3plus" / "best_model.pth",
        root=root,
        cfg=cfg,
        cfg_path=cfg_path,
        epochs=4,
        lr=3e-5,
        dice_weight=0.5,
    )
    unet_opt = run_controlled_experiment(
        exp_name="unet_ce_dice_balanced",
        model_name="unet",
        init_ckpt=root / "experiments" / "unet" / "best_model.pth",
        root=root,
        cfg=cfg,
        cfg_path=cfg_path,
        epochs=4,
        lr=3e-5,
        dice_weight=0.5,
    )

    baseline_comb = json.loads(baseline_check.read_text(encoding="utf-8"))
    summary = {
        "protocol": "Controlled train/val fine-tuning using CombinedCEDiceLoss(dice_weight=0.5) + tempered class weights (Big Rock 12.0 -> 7.0); checkpoint selection strictly on val mIoU.",
        "experiments": {
            "deeplabv3plus_ce_dice_balanced": {
                "baseline_val": dl_opt["baseline_validation_metrics"],
                "optimized_val": dl_opt["best_validation_metrics"],
                "baseline_test_combined": baseline_comb["deeplabv3plus"],
                "optimized_test_combined": dl_opt["expert_test_metrics"]["combined"],
                "optimized_test_msl": dl_opt["expert_test_metrics"]["msl"],
                "optimized_test_mer": dl_opt["expert_test_metrics"]["mer"],
            },
            "unet_ce_dice_balanced": {
                "baseline_val": unet_opt["baseline_validation_metrics"],
                "optimized_val": unet_opt["best_validation_metrics"],
                "baseline_test_combined": baseline_comb["unet"],
                "optimized_test_combined": unet_opt["expert_test_metrics"]["combined"],
                "optimized_test_msl": unet_opt["expert_test_metrics"]["msl"],
                "optimized_test_mer": unet_opt["expert_test_metrics"]["mer"],
            },
        },
    }
    summary_path = root / "experiments" / "evaluation" / "optimization" / "optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved controlled optimization summary to %s", summary_path)


if __name__ == "__main__":
    main()
