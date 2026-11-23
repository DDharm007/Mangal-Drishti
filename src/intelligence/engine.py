"""Unified TerrainIntelligenceEngine entry point and CLI for MANGAL DRISHTI."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.config_utils import get_project_root, load_config, resolve_path
from src.data.dataset import MangalDrishtiDataset
from src.data.preprocessing import load_and_preprocess_image
from src.data.transforms import SegmentationTransformPipeline
from src.intelligence.confidence_analyzer import analyze_confidence
from src.intelligence.obstacle_detector import detect_obstacle_candidates
from src.intelligence.region_analyzer import analyze_regions
from src.intelligence.report_generator import generate_intelligence_visualizations, save_intelligence_report
from src.intelligence.schemas import IntelligenceConfig, validate_intelligence_report
from src.intelligence.terrain_analyzer import analyze_terrain
from src.intelligence.traversability import assess_experimental_traversability
from src.intelligence.uncertainty import analyze_uncertainty
from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.unet import UNetResNet34
from src.training.checkpoint import load_checkpoint

LOGGER = logging.getLogger("mangal_drishti.intelligence.engine")

DEFAULT_CHECKPOINTS: Dict[str, str] = {
    "deeplabv3plus": "experiments/deeplabv3plus/best_model.pth",
    "unet": "experiments/unet/best_model.pth",
    "deeplabv3plus_optimized": "experiments/evaluation/optimization/deeplabv3plus_ce_dice_balanced/best_model.pth",
    "unet_optimized": "experiments/evaluation/optimization/unet_ce_dice_balanced/best_model.pth",
}


class TerrainIntelligenceEngine:
    """Reusable Mars rover terrain perception and intelligence engine."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "deeplabv3plus",
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path] = Path("configs/data_config.yaml"),
        device: Optional[Union[str, torch.device]] = None,
        intelligence_config: Optional[IntelligenceConfig] = None,
    ) -> None:
        if model_name not in DEFAULT_CHECKPOINTS:
            raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(DEFAULT_CHECKPOINTS.keys())}")

        self.root = get_project_root()
        self.model_name = model_name
        self.cfg_path = Path(config_path) if Path(config_path).is_absolute() else (self.root / Path(config_path))
        self.cfg = load_config(self.cfg_path)
        self.intel_cfg = intelligence_config or IntelligenceConfig()

        # Resolve GPU/CPU with automatic safe CPU fallback
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            req = torch.device(device)
            if req.type == "cuda" and not torch.cuda.is_available():
                LOGGER.warning("CUDA requested but unavailable; falling back cleanly to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = req

        ckpt_rel = DEFAULT_CHECKPOINTS[model_name] if checkpoint_path is None else checkpoint_path
        self.checkpoint_path = resolve_path(str(ckpt_rel), self.root)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        prep_cfg = json.loads(
            resolve_path(self.cfg["paths"]["preprocessing_config_file"], self.root).read_text(encoding="utf-8")
        )
        self.pipeline = SegmentationTransformPipeline(
            cfg=self.cfg,
            split="test",
            mean=prep_cfg["normalization"]["active_mean"],
            std=prep_cfg["normalization"]["active_std"],
        )

        # Build and cache model once
        is_unet = model_name.startswith("unet")
        self.model = (
            UNetResNet34(num_classes=4, pretrained=False)
            if is_unet
            else DeepLabV3PlusResNet50(num_classes=4, pretrained=False)
        ).to(self.device)
        load_checkpoint(self.checkpoint_path, self.model, self.device)
        self.model.eval()
        self.load_count = 1
        LOGGER.info("Loaded %s onto %s from %s", model_name, self.device, self.checkpoint_path.name)

    @torch.no_grad()
    def _run_inference(self, image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        target_size = (int(self.cfg["image"]["target_size"][0]), int(self.cfg["image"]["target_size"][1]))
        rgb = load_and_preprocess_image(
            image_path,
            target_size=target_size,
            convert_to_rgb=bool(self.cfg["image"]["convert_to_rgb"]),
            interpolation=str(self.cfg["image"]["interpolation"]),
            aspect_ratio_strategy=str(self.cfg["image"]["aspect_ratio_strategy"]),
        )
        tensor = self.pipeline.normalize_to_tensor(rgb).unsqueeze(0).to(self.device)
        amp_enabled = self.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = self.model(tensor)
        prob_map = F.softmax(logits.float(), dim=1)[0].cpu().numpy().astype(np.float32)
        pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return rgb, pred_mask, prob_map, latency_ms

    def analyze_from_predictions(
        self,
        rgb: np.ndarray,
        pred_mask: np.ndarray,
        prob_map: np.ndarray,
        image_id: str = "synthetic_or_array",
        source_path: str = "",
        output_dir: Optional[Union[str, Path]] = None,
        save_visualizations: bool = True,
        inference_latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run full Terrain Intelligence pipeline from precomputed RGB, mask, and softmax probs."""
        terrain_res = analyze_terrain(pred_mask, ignore_index=self.intel_cfg.ignore_index)
        conf_res, conf_map = analyze_confidence(
            prob_map,
            pred_mask,
            high_threshold=self.intel_cfg.high_confidence_threshold,
            medium_threshold=self.intel_cfg.medium_confidence_threshold,
            ignore_index=self.intel_cfg.ignore_index,
        )
        unc_res, _, norm_entropy_map = analyze_uncertainty(
            prob_map,
            pred_mask,
            high_uncertainty_threshold=self.intel_cfg.high_uncertainty_threshold,
            ignore_index=self.intel_cfg.ignore_index,
        )
        regions, region_meta = analyze_regions(
            pred_mask,
            conf_map,
            min_region_area_px=self.intel_cfg.min_region_area_px,
            ignore_index=self.intel_cfg.ignore_index,
        )
        candidates = detect_obstacle_candidates(
            pred_mask,
            prob_map,
            conf_map,
            min_area_px=self.intel_cfg.min_obstacle_area_px,
            min_confidence=self.intel_cfg.min_obstacle_confidence,
            ignore_index=self.intel_cfg.ignore_index,
        )
        traversability = assess_experimental_traversability(
            terrain_res, regions, candidates, conf_res, unc_res, self.intel_cfg
        )

        h, w = pred_mask.shape
        ckpt_rel = (
            str(self.checkpoint_path.relative_to(self.root)).replace("\\", "/")
            if self.checkpoint_path.is_relative_to(self.root)
            else str(self.checkpoint_path)
        )
        report: Dict[str, Any] = {
            "project": "MANGAL DRISHTI",
            "schema_version": self.SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "checkpoint": ckpt_rel,
            "device": str(self.device),
            "image_metadata": {
                "image_id": image_id,
                "source_path": source_path,
                "width": int(w),
                "height": int(h),
                "inference_latency_ms": round(float(inference_latency_ms), 3) if inference_latency_ms is not None else None,
            },
            "analysis": {
                "terrain_summary": terrain_res["terrain_summary"],
                "terrain_details": terrain_res["terrain_details"],
                "regions": regions,
                "region_filtering": {
                    "min_region_area_px": region_meta["min_region_area_px"],
                    "retained_region_count": region_meta["retained_region_count"],
                    "filtered_small_component_count": region_meta["filtered_small_component_count"],
                },
                "obstacle_candidates": candidates,
                "confidence": conf_res,
                "uncertainty": unc_res,
                "traversability": traversability,
            },
            "artifacts": {},
        }

        if save_visualizations:
            out_path = (
                Path(output_dir)
                if output_dir is not None
                else (self.root / "experiments" / "intelligence" / image_id)
            )
            viz_paths = generate_intelligence_visualizations(
                out_path, rgb, pred_mask, conf_map, norm_entropy_map, region_meta, report, project_root=self.root
            )
            report_json_path = out_path / "intelligence_report.json"
            viz_paths["intelligence_report_json"] = (
                str(report_json_path.relative_to(self.root)).replace("\\", "/")
                if report_json_path.is_relative_to(self.root)
                else str(report_json_path)
            )
            report["artifacts"] = viz_paths
            is_valid, errs = validate_intelligence_report(report)
            if not is_valid:
                raise ValueError(f"Generated intelligence report failed schema validation: {errs}")
            save_intelligence_report(report, report_json_path)
        else:
            is_valid, errs = validate_intelligence_report(report)
            if not is_valid:
                raise ValueError(f"Generated intelligence report failed schema validation: {errs}")

        return report

    def analyze(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        ignore_mask: Optional[Union[str, Path, np.ndarray]] = None,
        save_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Run end-to-end terrain intelligence analysis on a Mars rover image."""
        img_p = Path(image_path)
        if not img_p.is_absolute():
            img_p = self.root / img_p
        if not img_p.exists() or not img_p.is_file():
            raise FileNotFoundError(f"Input rover image not found: {img_p}")

        rgb, pred_mask, prob_map, latency_ms = self._run_inference(img_p)

        if ignore_mask is not None:
            if isinstance(ignore_mask, np.ndarray):
                ign_arr = ignore_mask
            else:
                ign_arr = np.asarray(Image.open(ignore_mask), dtype=np.uint8)
                if ign_arr.ndim == 3:
                    ign_arr = ign_arr[:, :, 0]
            ign_bool = ign_arr == self.intel_cfg.ignore_index
            pred_mask = pred_mask.copy()
            pred_mask[ign_bool] = self.intel_cfg.ignore_index

        rel_src = str(img_p.relative_to(self.root)).replace("\\", "/") if img_p.is_relative_to(self.root) else str(img_p)
        return self.analyze_from_predictions(
            rgb=rgb,
            pred_mask=pred_mask,
            prob_map=prob_map,
            image_id=img_p.stem,
            source_path=rel_src,
            output_dir=output_dir,
            save_visualizations=save_visualizations,
            inference_latency_ms=latency_ms,
        )


def run_batch_real_samples(
    engine: TerrainIntelligenceEngine,
    num_samples: int = 6,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Select diverse validation/test Mars rover images and generate full intelligence reports."""
    root = engine.root
    out_base = output_root or (root / "experiments" / "intelligence")
    ds_val = MangalDrishtiDataset(split="val", config_path=engine.cfg_path, project_root=root)
    ds_test = MangalDrishtiDataset(split="test", config_path=engine.cfg_path, project_root=root)

    # Pick diverse real scenes across MSL (val + test) and MER (test) including rover/range ignore masks
    chosen: List[tuple[str, Path, Path]] = []
    proc_dir = resolve_path(engine.cfg["paths"]["processed_dir"], root)
    for idx in [0, len(ds_val) // 2, len(ds_val) - 1]:
        rec = ds_val.records[idx]
        chosen.append((rec["mission"], proc_dir / rec["processed_image_relpath"], proc_dir / rec["processed_mask_relpath"]))

    msl_test_rec = next(r for r in ds_test.records if r["mission"].startswith("MSL") and r["pixel_counts"].get("3", 0) > 500)
    mer1_rec = next(r for r in ds_test.records if "MER-1" in r["mission"] and r["pixel_counts"].get("2", 0) > 10000)
    mer2_rec = next(r for r in ds_test.records if "MER-2" in r["mission"])
    for rec in [msl_test_rec, mer1_rec, mer2_rec]:
        chosen.append((rec["mission"], proc_dir / rec["processed_image_relpath"], proc_dir / rec["processed_mask_relpath"]))

    summaries = []
    for mission, img_p, msk_p in chosen[:num_samples]:
        # Apply official ignore mask boundary (255 regions from MXY/RNG) without using labels
        gt_raw = np.asarray(Image.open(msk_p), dtype=np.uint8)
        ignore_only = np.where(gt_raw == 255, 255, 0).astype(np.uint8)
        rep = engine.analyze(
            image_path=img_p,
            output_dir=out_base / img_p.stem,
            ignore_mask=ignore_only,
            save_visualizations=True,
        )
        summaries.append(
            {
                "image_id": img_p.stem,
                "mission": mission,
                "dominant_terrain": rep["analysis"]["terrain_details"]["dominant_terrain"],
                "terrain_summary": rep["analysis"]["terrain_summary"],
                "valid_pixel_percentage": rep["analysis"]["terrain_details"]["valid_pixel_percentage"],
                "retained_regions": rep["analysis"]["region_filtering"]["retained_region_count"],
                "obstacle_candidates": len(rep["analysis"]["obstacle_candidates"]),
                "mean_confidence": rep["analysis"]["confidence"]["mean_confidence"],
                "mean_entropy": rep["analysis"]["uncertainty"]["mean_entropy"],
                "traversability_assessment": rep["analysis"]["traversability"]["assessment"],
                "physical_safety_verified": rep["analysis"]["traversability"]["physical_safety_verified"],
                "report_path": rep["artifacts"]["intelligence_report_json"],
                "summary_panel_path": rep["artifacts"]["intelligence_summary_panel"],
            }
        )

    batch_summary = {"model": engine.model_name, "device": str(engine.device), "num_reports": len(summaries), "reports": summaries}
    (out_base / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    return batch_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MANGAL DRISHTI Terrain Intelligence Engine.")
    parser.add_argument("--image", type=Path, default=None, help="Path to a single rover image.")
    parser.add_argument("--batch-samples", type=int, default=6, help="Number of real samples if --image is omitted.")
    parser.add_argument(
        "--model",
        choices=list(DEFAULT_CHECKPOINTS.keys()),
        default="deeplabv3plus",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    engine = TerrainIntelligenceEngine(model_name=args.model, device=args.device)

    if args.image is not None:
        rep = engine.analyze(args.image, output_dir=args.output_dir, save_visualizations=True)
        print(json.dumps({"image_id": rep["image_metadata"]["image_id"], "traversability": rep["analysis"]["traversability"]["assessment"]}, indent=2))
    else:
        summary = run_batch_real_samples(engine, num_samples=args.batch_samples, output_root=args.output_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
