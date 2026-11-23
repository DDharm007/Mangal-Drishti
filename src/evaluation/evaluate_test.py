"""Untouched expert test evaluation and 6-panel visual error analysis for MANGAL DRISHTI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataset import MangalDrishtiDataset
from src.inference.predict import CLASS_NAMES, CLASS_PALETTE, colorize_prediction
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint
from src.training.metrics import CLASS_NAMES as METRIC_CLASS_NAMES
from src.training.metrics import SegmentationMetrics

LOGGER = logging.getLogger("mangal_drishti.evaluate_test")

ALLOWED_MASK_VALUES = {0, 1, 2, 3, 255}

ERROR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "correct": (46, 160, 67),          # Green
    "big_rock_fp": (255, 0, 128),      # Magenta
    "big_rock_fn": (0, 210, 255),      # Cyan
    "sand_bedrock_conf": (255, 140, 0),# Orange
    "other_error": (240, 220, 60),     # Yellow
    "ignore": (45, 45, 45),            # Dark gray
}


def format_metrics_with_cm(metrics: SegmentationMetrics, num_images: int) -> Dict[str, Any]:
    """Compute summary metrics alongside raw and row-normalized 4x4 confusion matrices."""
    out = metrics.compute()
    cm = metrics.conf_matrix.astype(np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums > 0,
    )
    out["num_images"] = int(num_images)
    out["confusion_matrix_labels"] = [METRIC_CLASS_NAMES[i] for i in range(4)]
    out["confusion_matrix"] = cm.tolist()
    out["confusion_matrix_row_normalized"] = np.round(cm_norm, 6).tolist()
    return out


def compute_sample_diagnostics(
    rec: Dict[str, Any],
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    conf_map: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-image diagnostics for error-case mining."""
    valid = (gt_mask != 255) & (gt_mask >= 0) & (gt_mask < 4)
    valid_px = int(valid.sum())

    if valid_px == 0:
        img_miou = 0.0
        pixel_acc = 0.0
        mean_conf_valid = float(conf_map.mean())
        low_conf_ratio = float((conf_map < 0.55).mean())
    else:
        t = gt_mask[valid]
        p = pred_mask[valid]
        pixel_acc = float((t == p).mean())
        ious = []
        for c in range(4):
            inter = int(((t == c) & (p == c)).sum())
            union = int(((t == c) | (p == c)).sum())
            if union > 0:
                ious.append(inter / union)
        img_miou = float(np.mean(ious)) if ious else 0.0
        mean_conf_valid = float(conf_map[valid].mean())
        low_conf_ratio = float((conf_map[valid] < 0.55).mean())

    br_gt = int((valid & (gt_mask == 3)).sum())
    br_pred = int((valid & (pred_mask == 3)).sum())
    br_tp = int((valid & (gt_mask == 3) & (pred_mask == 3)).sum())
    br_fn = int((valid & (gt_mask == 3) & (pred_mask != 3)).sum())
    br_fp = int((valid & (gt_mask != 3) & (pred_mask == 3)).sum())
    sb_conf = int((valid & (((gt_mask == 2) & (pred_mask == 1)) | ((gt_mask == 1) & (pred_mask == 2)))).sum())

    return {
        "id": rec["id"],
        "mission": rec["mission"],
        "valid_pixels": valid_px,
        "valid_pct": round(valid_px / float(gt_mask.size) * 100.0, 2),
        "pixel_accuracy": round(pixel_acc, 6),
        "image_miou": round(img_miou, 6),
        "mean_confidence": round(float(conf_map.mean()), 6),
        "mean_confidence_valid": round(mean_conf_valid, 6),
        "low_conf_pixel_ratio": round(low_conf_ratio, 6),
        "big_rock_gt_pixels": br_gt,
        "big_rock_pred_pixels": br_pred,
        "big_rock_tp_pixels": br_tp,
        "big_rock_fn_pixels": br_fn,
        "big_rock_fp_pixels": br_fp,
        "sand_bedrock_confusion_pixels": sb_conf,
    }


@torch.no_grad()
def evaluate_model_on_test(
    model: torch.nn.Module,
    dataset: MangalDrishtiDataset,
    device: torch.device,
    batch_size: int = 4,
    desc: str = "Evaluating",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a model on the untouched expert test set (Combined, MSL, MER, MER-1, MER-2)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    m_combined = SegmentationMetrics(num_classes=4, ignore_index=255)
    m_msl = SegmentationMetrics(num_classes=4, ignore_index=255)
    m_mer = SegmentationMetrics(num_classes=4, ignore_index=255)
    m_mer1 = SegmentationMetrics(num_classes=4, ignore_index=255)
    m_mer2 = SegmentationMetrics(num_classes=4, ignore_index=255)

    counts = {"combined": 0, "msl": 0, "mer": 0, "mer1": 0, "mer2": 0}
    per_image_stats: List[Dict[str, Any]] = []
    amp_enabled = device.type == "cuda"

    sample_offset = 0
    for imgs, masks in tqdm(loader, desc=desc, unit="batch"):
        unique_vals = set(torch.unique(masks).tolist())
        unexpected = unique_vals - ALLOWED_MASK_VALUES
        if unexpected:
            raise ValueError(f"Unexpected mask label IDs in test batch: {unexpected}")

        imgs_dev = imgs.to(device, non_blocking=True)
        masks_dev = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(imgs_dev)

        probs = F.softmax(logits.float(), dim=1)
        conf_batch, pred_batch = torch.max(probs, dim=1)

        m_combined.update(logits, masks_dev)

        bs = imgs.shape[0]
        for b in range(bs):
            rec = dataset.records[sample_offset + b]
            mission = rec["mission"]
            counts["combined"] += 1

            l_i = logits[b : b + 1]
            t_i = masks_dev[b : b + 1]
            if mission.startswith("MSL"):
                m_msl.update(l_i, t_i)
                counts["msl"] += 1
            elif mission.startswith("MER"):
                m_mer.update(l_i, t_i)
                counts["mer"] += 1
                if "MER-1" in mission:
                    m_mer1.update(l_i, t_i)
                    counts["mer1"] += 1
                elif "MER-2" in mission:
                    m_mer2.update(l_i, t_i)
                    counts["mer2"] += 1

            gt_np = masks[b].numpy().astype(np.uint8)
            pred_np = pred_batch[b].cpu().numpy().astype(np.uint8)
            conf_np = conf_batch[b].cpu().numpy().astype(np.float32)
            per_image_stats.append(compute_sample_diagnostics(rec, gt_np, pred_np, conf_np))

        sample_offset += bs

    result = {
        "combined": format_metrics_with_cm(m_combined, counts["combined"]),
        "msl": format_metrics_with_cm(m_msl, counts["msl"]),
        "mer": format_metrics_with_cm(m_mer, counts["mer"]),
        "mer_submissions": {
            "MER-1 (Opportunity)": format_metrics_with_cm(m_mer1, counts["mer1"]),
            "MER-2 (Spirit)": format_metrics_with_cm(m_mer2, counts["mer2"]),
        },
    }
    return result, per_image_stats


def build_comparison_dict(
    split_label: str,
    unet_metrics: Dict[str, Any],
    dl_metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create side-by-side comparison JSON between U-Net and DeepLabV3+."""
    keys_scalar = ["mIoU", "mean_dice", "pixel_accuracy", "sand_iou", "sand_recall", "big_rock_iou", "big_rock_recall"]
    deltas = {
        f"{k}_delta": round(float(dl_metrics[k]) - float(unet_metrics[k]), 6)
        for k in keys_scalar
    }
    deltas["soil_iou_delta"] = round(dl_metrics["per_class_iou"]["soil"] - unet_metrics["per_class_iou"]["soil"], 6)
    deltas["bedrock_iou_delta"] = round(dl_metrics["per_class_iou"]["bedrock"] - unet_metrics["per_class_iou"]["bedrock"], 6)
    deltas["big_rock_precision_delta"] = round(
        dl_metrics["per_class_precision"]["big rock"] - unet_metrics["per_class_precision"]["big rock"], 6
    )

    out: Dict[str, Any] = {
        "split": split_label,
        "num_images": unet_metrics["num_images"],
        "valid_pixels_evaluated": unet_metrics["valid_pixels_evaluated"],
        "unet": unet_metrics,
        "deeplabv3plus": dl_metrics,
        "comparison_deeplabv3plus_minus_unet": deltas,
    }
    if extra:
        out.update(extra)
    return out


def render_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Create RGB error visualization highlighting Big Rock FP/FN and Sand/Bedrock confusion."""
    h, w = gt_mask.shape
    err = np.full((h, w, 3), ERROR_PALETTE["ignore"], dtype=np.uint8)
    valid = (gt_mask != 255) & (gt_mask >= 0) & (gt_mask < 4)

    correct = valid & (gt_mask == pred_mask)
    wrong = valid & (gt_mask != pred_mask)
    br_fp = wrong & (pred_mask == 3)
    br_fn = wrong & (gt_mask == 3)
    sb_conf = wrong & (((gt_mask == 2) & (pred_mask == 1)) | ((gt_mask == 1) & (pred_mask == 2)))
    other = wrong & (~br_fp) & (~br_fn) & (~sb_conf)

    err[correct] = ERROR_PALETTE["correct"]
    err[other] = ERROR_PALETTE["other_error"]
    err[sb_conf] = ERROR_PALETTE["sand_bedrock_conf"]
    err[br_fn] = ERROR_PALETTE["big_rock_fn"]
    err[br_fp] = ERROR_PALETTE["big_rock_fp"]
    return err


def save_6panel_error_figure(
    out_path: Path,
    model_label: str,
    category: str,
    rec: Dict[str, Any],
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    conf_map: np.ndarray,
    diag: Dict[str, Any],
) -> None:
    """Render 6-panel figure: Original, GT, Predicted, Confidence, Overlay, Error Map."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_color = colorize_prediction(gt_mask)
    pred_color = colorize_prediction(pred_mask)
    overlay = np.clip(0.52 * rgb.astype(np.float32) + 0.48 * pred_color.astype(np.float32), 0, 255).astype(np.uint8)
    err_map = render_error_map(gt_mask, pred_mask)

    pred_counts = np.bincount(pred_mask.ravel(), minlength=4)
    total_px = pred_mask.size
    dist_str = " | ".join(f"{CLASS_NAMES[i].split()[0]}: {pred_counts[i]/total_px*100:.1f}%" for i in range(4))

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.9), dpi=140)
    axes[0].imshow(rgb)
    axes[0].set_title(f"Original Image\n({rec['mission']})", fontsize=9.5, fontweight="bold")

    axes[1].imshow(gt_color)
    axes[1].set_title(f"Ground-Truth Mask\n(Labeled: {diag['valid_pct']:.1f}%)", fontsize=9.5, fontweight="bold")

    axes[2].imshow(pred_color)
    axes[2].set_title(f"Predicted Mask\n({dist_str})", fontsize=8.5, fontweight="bold")

    im_c = axes[3].imshow(conf_map, cmap="viridis", vmin=0.25, vmax=1.0)
    axes[3].set_title(f"Confidence Map\n(Mean: {diag['mean_confidence']:.3f})", fontsize=9.5, fontweight="bold")
    plt.colorbar(im_c, ax=axes[3], fraction=0.046, pad=0.03)

    axes[4].imshow(overlay)
    axes[4].set_title(f"Overlay\n(Img mIoU: {diag['image_miou']:.3f})", fontsize=9.5, fontweight="bold")

    axes[5].imshow(err_map)
    axes[5].set_title(
        f"Error Map\n(BR FP:{diag['big_rock_fp_pixels']} FN:{diag['big_rock_fn_pixels']} SB:{diag['sand_bedrock_confusion_pixels']})",
        fontsize=8.5,
        fontweight="bold",
    )

    for ax in axes:
        ax.axis("off")

    class_handles = [mpatches.Patch(color=np.array(CLASS_PALETTE[k]) / 255.0, label=CLASS_NAMES[k]) for k in range(4)]
    err_handles = [
        mpatches.Patch(color=np.array(ERROR_PALETTE["correct"]) / 255.0, label="Correct"),
        mpatches.Patch(color=np.array(ERROR_PALETTE["big_rock_fp"]) / 255.0, label="Big Rock FP"),
        mpatches.Patch(color=np.array(ERROR_PALETTE["big_rock_fn"]) / 255.0, label="Big Rock FN"),
        mpatches.Patch(color=np.array(ERROR_PALETTE["sand_bedrock_conf"]) / 255.0, label="Sand/Bedrock Err"),
        mpatches.Patch(color=np.array(ERROR_PALETTE["other_error"]) / 255.0, label="Other Err"),
        mpatches.Patch(color=np.array(ERROR_PALETTE["ignore"]) / 255.0, label="Ignore (255)"),
    ]
    fig.suptitle(
        f"MANGAL DRISHTI ({model_label}) | Case: {category} | {rec['id']}",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        handles=class_handles + err_handles,
        loc="lower center",
        ncol=10,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def select_representative_cases(
    dl_stats: List[Dict[str, Any]],
    unet_stats: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Select 8 unique representative test samples spanning all 5 qualitative categories."""
    chosen_indices: set[int] = set()
    selected: List[Tuple[str, int]] = []

    def pick_best(category: str, candidate_indices: List[int], score_fn) -> None:
        ranked = sorted(
            [i for i in candidate_indices if i not in chosen_indices],
            key=score_fn,
            reverse=True,
        )
        if ranked:
            idx = ranked[0]
            chosen_indices.add(idx)
            selected.append((category, idx))

    all_idx = list(range(len(dl_stats)))
    msl_idx = [i for i in all_idx if dl_stats[i]["mission"].startswith("MSL")]
    mer_idx = [i for i in all_idx if dl_stats[i]["mission"].startswith("MER")]

    # 1. Correct predictions (1 MSL, 1 MER with strong multi-class & valid pixel coverage)
    pick_best(
        "correct_prediction",
        [i for i in msl_idx if dl_stats[i]["valid_pct"] >= 35.0],
        lambda i: dl_stats[i]["image_miou"] + 0.5 * dl_stats[i]["pixel_accuracy"],
    )
    pick_best(
        "correct_prediction",
        [i for i in mer_idx if dl_stats[i]["valid_pct"] >= 35.0],
        lambda i: dl_stats[i]["image_miou"] + 0.5 * dl_stats[i]["pixel_accuracy"],
    )

    # 2. Big Rock false negatives (2 cases across models)
    pick_best(
        "big_rock_false_negative",
        msl_idx,
        lambda i: dl_stats[i]["big_rock_fn_pixels"] + unet_stats[i]["big_rock_fn_pixels"],
    )
    pick_best(
        "big_rock_false_negative",
        mer_idx,
        lambda i: dl_stats[i]["big_rock_fn_pixels"] + unet_stats[i]["big_rock_fn_pixels"],
    )

    # 3. Big Rock false positives (1 high-impact case)
    pick_best(
        "big_rock_false_positive",
        all_idx,
        lambda i: dl_stats[i]["big_rock_fp_pixels"] + unet_stats[i]["big_rock_fp_pixels"],
    )

    # 4. Sand/Bedrock confusion (2 cases: 1 MSL, 1 MER)
    pick_best(
        "sand_bedrock_confusion",
        msl_idx,
        lambda i: dl_stats[i]["sand_bedrock_confusion_pixels"] + unet_stats[i]["sand_bedrock_confusion_pixels"],
    )
    pick_best(
        "sand_bedrock_confusion",
        mer_idx,
        lambda i: dl_stats[i]["sand_bedrock_confusion_pixels"] + unet_stats[i]["sand_bedrock_confusion_pixels"],
    )

    # 5. Low-confidence prediction (1 case with highest low-confidence ratio / lowest mean confidence)
    pick_best(
        "low_confidence_prediction",
        [i for i in all_idx if dl_stats[i]["valid_pct"] >= 20.0],
        lambda i: -dl_stats[i]["mean_confidence_valid"] + dl_stats[i]["low_conf_pixel_ratio"],
    )

    return selected


def render_confusion_matrix_comparison(
    unet_res: Dict[str, Any],
    dl_res: Dict[str, Any],
    out_path: Path,
) -> None:
    """Plot row-normalized confusion matrices for U-Net and DeepLabV3+ across Combined, MSL, MER."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Soil", "Bedrock", "Sand", "Big Rock"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    rows = [("U-Net + ResNet34", unet_res), ("DeepLabV3+ + ResNet50", dl_res)]
    cols = [("Combined (526)", "combined"), ("MSL (322)", "msl"), ("MER (204)", "mer")]

    for r_idx, (m_title, m_res) in enumerate(rows):
        for c_idx, (s_title, s_key) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            cm_norm = np.array(m_res[s_key]["confusion_matrix_row_normalized"], dtype=np.float64)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_yticklabels(labels, fontsize=8.5)
            ax.set_title(f"{m_title}\n{s_title} (mIoU={m_res[s_key]['mIoU']:.4f})", fontsize=9.5, fontweight="bold")
            if r_idx == 1:
                ax.set_xlabel("Predicted Class", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel("True Class", fontsize=9)
            for i in range(4):
                for j in range(4):
                    val = cm_norm[i, j]
                    ax.text(
                        j, i, f"{val*100:.1f}%",
                        ha="center", va="center",
                        color="white" if val > 0.55 else "black",
                        fontsize=8,
                    )

    fig.suptitle("MANGAL DRISHTI — Expert Test Row-Normalized Confusion Matrices", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    cbar_ax = fig.add_axes([0.965, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Recall / Row Fraction")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_visual_error_analysis(
    root: Path,
    dataset: MangalDrishtiDataset,
    models: Dict[str, torch.nn.Module],
    per_image_stats: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
) -> Path:
    """Render 6-panel error figures for selected cases for both models."""
    selected = select_representative_cases(per_image_stats["deeplabv3plus"], per_image_stats["unet"])
    viz_root = root / "experiments" / "evaluation" / "visualizations"
    amp_enabled = device.type == "cuda"
    manifest_items = []

    for category, idx in selected:
        rec = dataset.records[idx]
        rgb, gt_mask = dataset.load_raw_processed_pair(idx)
        img_tensor, _ = dataset[idx]
        x = img_tensor.unsqueeze(0).to(device)

        case_entry: Dict[str, Any] = {
            "category": category,
            "index": idx,
            "id": rec["id"],
            "mission": rec["mission"],
            "outputs": {},
        }

        for model_key, model in models.items():
            model.eval()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
            probs = F.softmax(logits.float(), dim=1)[0]
            conf_map, pred_mask = torch.max(probs, dim=0)
            pred_np = pred_mask.cpu().numpy().astype(np.uint8)
            conf_np = conf_map.cpu().numpy().astype(np.float32)
            diag = per_image_stats[model_key][idx]

            out_png = viz_root / model_key / f"{category}_{rec['id']}.png"
            save_6panel_error_figure(
                out_png,
                "U-Net + ResNet34" if model_key == "unet" else "DeepLabV3+ + ResNet50",
                category,
                rec,
                rgb,
                gt_mask,
                pred_np,
                conf_np,
                diag,
            )
            case_entry["outputs"][model_key] = {
                "panel_path": str(out_png.relative_to(root)).replace("\\", "/"),
                "diagnostics": diag,
            }
        manifest_items.append(case_entry)

    manifest_path = viz_root / "selected_cases_manifest.json"
    manifest_path.write_text(json.dumps({"cases": manifest_items}, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    root = get_project_root()
    cfg_path = root / "configs" / "data_config.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = MangalDrishtiDataset(split="test", config_path=cfg_path, project_root=root)
    LOGGER.info("Loaded untouched expert test dataset: %d images", len(test_ds))

    unet = UNetResNet34(num_classes=4, pretrained=False).to(device)
    load_checkpoint(root / "experiments" / "unet" / "best_model.pth", unet, device)
    unet_res, unet_stats = evaluate_model_on_test(unet, test_ds, device, batch_size=4, desc="[Baseline Test] U-Net")

    deeplab = DeepLabV3PlusResNet50(num_classes=4, pretrained=False).to(device)
    load_checkpoint(root / "experiments" / "deeplabv3plus" / "best_model.pth", deeplab, device)
    dl_res, dl_stats = evaluate_model_on_test(deeplab, test_ds, device, batch_size=4, desc="[Baseline Test] DeepLabV3+")

    base_dir = root / "experiments" / "evaluation" / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)

    unet_payload = {
        "model_name": "unet",
        "architecture": "U-Net + ResNet34",
        "checkpoint": "experiments/unet/best_model.pth",
        "evaluation_protocol": "untouched_expert_test_ignore_255",
        **unet_res,
    }
    dl_payload = {
        "model_name": "deeplabv3plus",
        "architecture": "DeepLabV3+ + ResNet50",
        "checkpoint": "experiments/deeplabv3plus/best_model.pth",
        "evaluation_protocol": "untouched_expert_test_ignore_255",
        **dl_res,
    }
    (base_dir / "unet_test_metrics.json").write_text(json.dumps(unet_payload, indent=2), encoding="utf-8")
    (base_dir / "deeplabv3plus_test_metrics.json").write_text(json.dumps(dl_payload, indent=2), encoding="utf-8")

    (base_dir / "combined_comparison.json").write_text(
        json.dumps(build_comparison_dict("combined_expert_test", unet_res["combined"], dl_res["combined"]), indent=2),
        encoding="utf-8",
    )
    (base_dir / "msl_comparison.json").write_text(
        json.dumps(build_comparison_dict("msl_expert_test", unet_res["msl"], dl_res["msl"]), indent=2),
        encoding="utf-8",
    )
    (base_dir / "mer_comparison.json").write_text(
        json.dumps(
            build_comparison_dict(
                "mer_expert_test",
                unet_res["mer"],
                dl_res["mer"],
                extra={
                    "mer_submissions": {
                        "MER-1 (Opportunity)": build_comparison_dict(
                            "MER-1 (Opportunity)",
                            unet_res["mer_submissions"]["MER-1 (Opportunity)"],
                            dl_res["mer_submissions"]["MER-1 (Opportunity)"],
                        ),
                        "MER-2 (Spirit)": build_comparison_dict(
                            "MER-2 (Spirit)",
                            unet_res["mer_submissions"]["MER-2 (Spirit)"],
                            dl_res["mer_submissions"]["MER-2 (Spirit)"],
                        ),
                    }
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    render_confusion_matrix_comparison(
        unet_res, dl_res, root / "experiments" / "evaluation" / "visualizations" / "confusion_matrices_test.png"
    )
    manifest_p = generate_visual_error_analysis(
        root,
        test_ds,
        {"unet": unet, "deeplabv3plus": deeplab},
        {"unet": unet_stats, "deeplabv3plus": dl_stats},
        device,
    )
    LOGGER.info("Baseline expert test evaluation & visualizations saved. Manifest: %s", manifest_p)


if __name__ == "__main__":
    main()
