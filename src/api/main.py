"""FastAPI mission control service layer for MANGAL DRISHTI."""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.data.config_utils import get_project_root
from src.intelligence.schemas import validate_intelligence_report

try:
    import torch
    from src.intelligence.engine import DEFAULT_CHECKPOINTS, TerrainIntelligenceEngine
except ImportError:  # Serverless / lightweight runtime without PyTorch
    torch = None  # type: ignore[assignment]
    TerrainIntelligenceEngine = None  # type: ignore[assignment,misc]
    DEFAULT_CHECKPOINTS = {
        "deeplabv3plus": "experiments/deeplabv3plus/best_model.pth",
        "unet": "experiments/unet/best_model.pth",
        "deeplabv3plus_optimized": "experiments/evaluation/optimization/deeplabv3plus_ce_dice_balanced/best_model.pth",
        "unet_optimized": "experiments/evaluation/optimization/unet_ce_dice_balanced/best_model.pth",
    }

APP_VERSION = "1.0.0"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

ROOT = get_project_root()
INTEL_DIR = ROOT / "experiments" / "intelligence"
EVAL_DIR = ROOT / "experiments" / "evaluation"
META_DIR = ROOT / "datasets" / "metadata"
DOCS_DIR = ROOT / "docs"

INTEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MANGAL DRISHTI Mission Control API",
    version=APP_VERSION,
    description="Planetary terrain intelligence, evaluation, and diagnostics API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts/intelligence", StaticFiles(directory=str(INTEL_DIR)), name="intelligence_artifacts")
app.mount("/artifacts/evaluation", StaticFiles(directory=str(EVAL_DIR)), name="evaluation_artifacts")

_ENGINE_CACHE: Dict[str, TerrainIntelligenceEngine] = {}
_MISSION_LOOKUP: Optional[Dict[str, str]] = None

MODEL_DISPLAY_META: Dict[str, Dict[str, str]] = {
    "deeplabv3plus": {
        "label": "DeepLabV3+ (Baseline)",
        "backbone": "ResNet50 + ASPP",
        "status_note": "Default cross-mission baseline",
    },
    "unet": {
        "label": "U-Net (Baseline)",
        "backbone": "ResNet34 Decoder",
        "status_note": "Lightweight skip-connection baseline",
    },
    "deeplabv3plus_optimized": {
        "label": "DeepLabV3+ (Optimized)",
        "backbone": "ResNet50 + CE/Dice",
        "status_note": "Fine-tuned on MSL train/val",
    },
    "unet_optimized": {
        "label": "U-Net (Optimized)",
        "backbone": "ResNet34 + CE/Dice",
        "status_note": "Fine-tuned on MSL train/val",
    },
}


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_mission_lookup() -> Dict[str, str]:
    global _MISSION_LOOKUP
    if _MISSION_LOOKUP is not None:
        return _MISSION_LOOKUP
    lookup: Dict[str, str] = {}
    manifest = _load_json_if_exists(META_DIR / "split_manifest.json")
    if manifest and "splits" in manifest:
        for split_records in manifest["splits"].values():
            for rec in split_records:
                rec_id = str(rec.get("id", ""))
                mission = rec.get("mission")
                if rec_id and mission:
                    lookup[rec_id] = str(mission)
    _MISSION_LOOKUP = lookup
    return lookup


def _resolve_mission(report_id: str, report: Dict[str, Any]) -> Optional[str]:
    lookup = _get_mission_lookup()
    if report_id in lookup:
        return lookup[report_id]
    img_id = str(report.get("image_metadata", {}).get("image_id", ""))
    if img_id in lookup:
        return lookup[img_id]
    # Check stripped timestamp suffix for uploaded files named after AI4MARS IDs
    base = re.sub(r"_\d{8}_\d{6}$", "", report_id)
    if base in lookup:
        return lookup[base]
    return None


def _get_engine(model_name: str) -> Any:
    if TerrainIntelligenceEngine is None:
        raise HTTPException(
            status_code=503,
            detail="Live PyTorch inference is unavailable in this serverless deployment. Run locally with GPU/CPU checkpoints.",
        )
    if model_name not in DEFAULT_CHECKPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}'. Allowed: {list(DEFAULT_CHECKPOINTS.keys())}",
        )
    if model_name not in _ENGINE_CACHE:
        _ENGINE_CACHE[model_name] = TerrainIntelligenceEngine(model_name=model_name)
    return _ENGINE_CACHE[model_name]


def _build_artifact_urls(report_id: str, report_dir: Path, report: Dict[str, Any]) -> Dict[str, Optional[str]]:
    artifacts = report.get("artifacts", {})
    key_to_filename = {
        "original_image": "01_original_image.png",
        "segmentation_mask": "02_segmentation_mask.png",
        "terrain_overlay": "03_terrain_overlay.png",
        "region_boundaries": "04_region_boundaries.png",
        "big_rock_candidates": "05_big_rock_candidates.png",
        "confidence_heatmap": "06_confidence_heatmap.png",
        "uncertainty_heatmap": "07_uncertainty_heatmap.png",
        "terrain_distribution_chart": "08_terrain_distribution.png",
        "intelligence_summary_panel": "09_intelligence_summary_panel.png",
        "intelligence_report_json": "intelligence_report.json",
    }
    urls: Dict[str, Optional[str]] = {}
    for k, default_name in key_to_filename.items():
        rel_val = artifacts.get(k)
        filename = Path(rel_val).name if rel_val else default_name
        candidate_path = report_dir / filename
        urls[k] = f"/artifacts/intelligence/{report_id}/{filename}" if candidate_path.exists() else None
    return urls


def _validate_safe_report_id(report_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", report_id):
        raise HTTPException(status_code=400, detail="Invalid report identifier.")
    target = (INTEL_DIR / report_id).resolve()
    if target.parent != INTEL_DIR.resolve() or not target.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return target


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "online",
        "service": "MANGAL DRISHTI Mission Control API",
        "version": APP_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/system/status")
def get_system_status() -> Dict[str, Any]:
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    active_device = "cuda" if has_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if has_cuda and torch is not None else None
    gpu_vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if has_cuda and torch is not None
        else None
    )

    checkpoints = []
    for name, rel_path in DEFAULT_CHECKPOINTS.items():
        p = ROOT / rel_path
        exists = p.exists()
        checkpoints.append(
            {
                "model_name": name,
                "label": MODEL_DISPLAY_META[name]["label"],
                "checkpoint_path": rel_path,
                "available": exists,
                "size_mb": round(p.stat().st_size / (1024**2), 2) if exists else None,
                "loaded_in_memory": name in _ENGINE_CACHE,
            }
        )

    reports = _scan_reports()
    latest_ts = reports[0]["generated_at_utc"] if reports else None
    latest_id = reports[0]["report_id"] if reports else None
    latest_latency = reports[0].get("inference_latency_ms") if reports else None

    return {
        "status": "online",
        "version": APP_VERSION,
        "pytorch_version": str(torch.__version__) if torch is not None else "unavailable (serverless)",
        "cuda_available": has_cuda,
        "active_device": active_device,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "host_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "checkpoints": checkpoints,
        "loaded_models": list(_ENGINE_CACHE.keys()),
        "stored_reports_count": len(reports),
        "latest_analysis_id": latest_id,
        "latest_analysis_timestamp_utc": latest_ts,
        "latest_inference_latency_ms": latest_latency,
        "report_storage_dir": "experiments/intelligence",
    }


def _scan_reports() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    if not INTEL_DIR.exists():
        return summaries

    for entry in INTEL_DIR.iterdir():
        if not entry.is_dir():
            continue
        rep_file = entry / "intelligence_report.json"
        if not rep_file.exists():
            continue
        try:
            rep = json.loads(rep_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        report_id = entry.name
        analysis = rep.get("analysis", {})
        td = analysis.get("terrain_details", {})
        ts = analysis.get("terrain_summary", {})
        trav = analysis.get("traversability", {})
        conf = analysis.get("confidence", {})
        unc = analysis.get("uncertainty", {})
        artifact_urls = _build_artifact_urls(report_id, entry, rep)

        summaries.append(
            {
                "report_id": report_id,
                "image_id": rep.get("image_metadata", {}).get("image_id", report_id),
                "mission": _resolve_mission(report_id, rep),
                "model": rep.get("model", "unknown"),
                "device": rep.get("device", "unknown"),
                "inference_latency_ms": rep.get("image_metadata", {}).get("inference_latency_ms"),
                "generated_at_utc": rep.get("generated_at_utc")
                or datetime.fromtimestamp(rep_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                "dominant_terrain": td.get("dominant_terrain"),
                "secondary_terrain": td.get("secondary_terrain"),
                "valid_pixel_percentage": td.get("valid_pixel_percentage"),
                "terrain_summary": ts,
                "retained_regions": len(analysis.get("regions", [])),
                "obstacle_candidates_count": len(analysis.get("obstacle_candidates", [])),
                "mean_confidence": conf.get("mean_confidence"),
                "mean_entropy": unc.get("mean_entropy"),
                "traversability_assessment": trav.get("assessment"),
                "experimental_visual_concern_index": trav.get("experimental_visual_concern_index"),
                "physical_safety_verified": trav.get("physical_safety_verified", False),
                "artifact_urls": artifact_urls,
            }
        )

    summaries.sort(key=lambda r: str(r.get("generated_at_utc") or ""), reverse=True)
    return summaries


@app.get("/api/reports")
def list_reports() -> Dict[str, Any]:
    items = _scan_reports()
    return {"count": len(items), "reports": items}


@app.get("/api/reports/{report_id}")
def get_report_detail(report_id: str) -> Dict[str, Any]:
    target_dir = _validate_safe_report_id(report_id)
    rep_file = target_dir / "intelligence_report.json"
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail=f"intelligence_report.json missing for '{report_id}'.")
    rep = json.loads(rep_file.read_text(encoding="utf-8"))
    return {
        "report_id": report_id,
        "mission": _resolve_mission(report_id, rep),
        "artifact_urls": _build_artifact_urls(report_id, target_dir, rep),
        "report": rep,
    }


@app.get("/api/reports/{report_id}/download")
def download_report_json(report_id: str) -> FileResponse:
    target_dir = _validate_safe_report_id(report_id)
    rep_file = target_dir / "intelligence_report.json"
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail=f"intelligence_report.json missing for '{report_id}'.")
    return FileResponse(
        path=rep_file,
        media_type="application/json",
        filename=f"{report_id}_intelligence_report.json",
    )


@app.post("/api/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("deeplabv3plus"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    engine = _get_engine(model_name)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename).stem)[:48] or "rover_upload"
    ts_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_id = f"{safe_stem}_{ts_suffix}"
    out_dir = INTEL_DIR / report_id

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_img = Path(tmp_dir) / f"{safe_stem}{ext}"
            tmp_img.write_bytes(payload)
            report = engine.analyze(
                image_path=tmp_img,
                output_dir=out_dir,
                save_visualizations=True,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    ok, errs = validate_intelligence_report(report)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Schema validation error: {errs}")

    return {
        "report_id": report_id,
        "mission": _resolve_mission(Path(file.filename).stem, report),
        "artifact_urls": _build_artifact_urls(report_id, out_dir, report),
        "report": report,
    }


@app.get("/api/models")
def get_models_and_performance() -> Dict[str, Any]:
    catalog = []
    for key, meta in MODEL_DISPLAY_META.items():
        p = ROOT / DEFAULT_CHECKPOINTS[key]
        catalog.append(
            {
                "id": key,
                "label": meta["label"],
                "backbone": meta["backbone"],
                "status_note": meta["status_note"],
                "is_default": key == "deeplabv3plus",
                "checkpoint_available": p.exists(),
                "loaded_in_memory": key in _ENGINE_CACHE,
            }
        )

    val_comp = _load_json_if_exists(ROOT / "experiments" / "model_comparison.json")
    test_combined = _load_json_if_exists(EVAL_DIR / "baseline" / "combined_comparison.json")
    test_msl = _load_json_if_exists(EVAL_DIR / "baseline" / "msl_comparison.json")
    test_mer = _load_json_if_exists(EVAL_DIR / "baseline" / "mer_comparison.json")
    benchmarks = _load_json_if_exists(EVAL_DIR / "benchmarks" / "inference_benchmark.json")
    optimization = _load_json_if_exists(EVAL_DIR / "optimization" / "optimization_summary.json")

    cm_fig = EVAL_DIR / "visualizations" / "confusion_matrices_test.png"
    return {
        "catalog": catalog,
        "splits": {
            "validation": val_comp,
            "test_combined": test_combined,
            "test_msl": test_msl,
            "test_mer": test_mer,
        },
        "benchmarks": benchmarks,
        "optimization": optimization,
        "artifacts": {
            "confusion_matrices_url": "/artifacts/evaluation/visualizations/confusion_matrices_test.png"
            if cm_fig.exists()
            else None,
        },
    }


@app.get("/api/research")
def get_research_metadata() -> Dict[str, Any]:
    class_mapping = _load_json_if_exists(META_DIR / "class_mapping.json")
    class_weights = _load_json_if_exists(META_DIR / "class_weights.json")
    split_manifest = _load_json_if_exists(META_DIR / "split_manifest.json")

    split_summary = None
    if split_manifest:
        split_summary = {
            "dataset": split_manifest.get("dataset"),
            "random_seed": split_manifest.get("random_seed"),
            "split_config": split_manifest.get("split_config"),
            "counts": split_manifest.get("counts"),
            "leakage_checks": split_manifest.get("leakage_checks"),
        }

    docs = []
    for doc_name, title in [
        ("dataset_research.md", "Dataset Collection & AI4MARS Research"),
        ("preprocessing_pipeline.md", "Data Preprocessing & Engineering Pipeline"),
        ("terrain_intelligence_engine.md", "Terrain Intelligence Engine Specification"),
    ]:
        p = DOCS_DIR / doc_name
        if p.exists():
            docs.append(
                {
                    "filename": doc_name,
                    "title": title,
                    "path": f"docs/{doc_name}",
                    "content": p.read_text(encoding="utf-8"),
                }
            )

    return {
        "class_mapping": class_mapping,
        "class_weights": class_weights,
        "split_summary": split_summary,
        "documentation": docs,
    }
