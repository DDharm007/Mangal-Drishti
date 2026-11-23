"""Unit and integration tests for MANGAL DRISHTI Step 5 Terrain Intelligence Engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import MangalDrishtiDataset
from src.intelligence import IntelligenceConfig, TerrainIntelligenceEngine, validate_intelligence_report
from src.intelligence.confidence_analyzer import analyze_confidence
from src.intelligence.obstacle_detector import detect_obstacle_candidates
from src.intelligence.region_analyzer import analyze_regions
from src.intelligence.terrain_analyzer import analyze_terrain
from src.intelligence.traversability import assess_experimental_traversability
from src.intelligence.uncertainty import analyze_uncertainty


def _make_synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 128x128 synthetic mask & softmax probability map with known geometry."""
    h, w = 128, 128
    mask = np.zeros((h, w), dtype=np.uint8)  # Default: Soil (0)
    mask[:, 64:] = 1                         # Bedrock (1)
    mask[80:, :64] = 2                       # Sand (2)
    mask[10:30, 10:30] = 3                   # Big Rock block 1 (rows 16..29 = 14x20 = 280 valid px)
    mask[100:103, 100:103] = 3               # Tiny Big Rock noise 1 (3x3 = 9 px < min_area)
    mask[50:54, 100:104] = 2                 # Tiny Sand noise 2 isolated inside Bedrock (4x4 = 16 px < min_area)
    mask[:16, :] = 255                       # Top 16 rows ignored (255)

    probs = np.full((4, h, w), 0.05, dtype=np.float32)
    for cid in range(4):
        sel = mask == cid
        probs[cid, sel] = 0.85
    # Ignored region filled uniformly (must be ignored in calculations)
    probs[:, mask == 255] = 0.25
    probs = probs / probs.sum(axis=0, keepdims=True)
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    return rgb, mask, probs


def test_terrain_distribution_and_ignored_pixels() -> None:
    _, mask, _ = _make_synthetic_scene()
    res = analyze_terrain(mask, ignore_index=255)
    td = res["terrain_details"]
    ts = res["terrain_summary"]

    # Top 16 rows ignored -> 16 * 128 = 2048 ignored pixels out of 16384
    assert td["ignored_pixels"] == 16 * 128
    assert td["total_valid_pixels"] == 128 * 128 - 16 * 128
    assert abs(sum(ts.values()) - 100.0) < 1e-2
    assert td["dominant_terrain"] == "bedrock"
    assert td["secondary_terrain"] in {"soil", "sand"}
    assert td["boundary_pixel_count"] > 0


def test_connected_components_and_region_filtering_preserves_mask() -> None:
    _, mask, probs = _make_synthetic_scene()
    original_copy = mask.copy()
    conf_map = probs.max(axis=0)

    regions, meta = analyze_regions(mask, conf_map, min_region_area_px=120, ignore_index=255)
    # Must not mutate original prediction mask
    assert np.array_equal(mask, original_copy)
    # Both 3x3 (9 px) Big Rock speck and 4x4 (16 px) Sand speck must be filtered (< 120), while 280 px Big Rock stays
    assert meta["filtered_small_component_count"] == 2
    assert all(r["pixel_area"] >= 120 for r in regions)
    assert any(r["terrain_class"] == "big_rock" and r["pixel_area"] == 280 for r in regions)
    assert all(r["terrain_class"] in {"soil", "bedrock", "sand", "big_rock"} for r in regions)


def test_big_rock_candidate_extraction() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:45] = 3   # 20x25 = 500 px Big Rock candidate
    mask[80:84, 80:84] = 3   # 4x4 = 16 px noise (should be filtered out)

    probs = np.full((4, 100, 100), 0.05, dtype=np.float32)
    probs[0, mask == 0] = 0.85
    probs[3, mask == 3] = 0.85
    probs /= probs.sum(axis=0, keepdims=True)
    conf_map = probs.max(axis=0)

    cands = detect_obstacle_candidates(mask, probs, conf_map, min_area_px=64, min_confidence=0.35)
    assert len(cands) == 1
    c = cands[0]
    assert c["pixel_area"] == 500
    assert c["bbox"] == [20, 20, 25, 20]
    assert c["is_confirmed_physical_obstacle"] is False
    assert "circularity" in c["shape"]
    assert "solidity" in c["shape"]


def test_confidence_and_exact_shannon_entropy() -> None:
    h, w = 16, 16
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, 8:] = 1
    mask[:4, :] = 255  # Ignore top 4 rows

    probs = np.zeros((4, h, w), dtype=np.float32)
    # Left half: uniform 0.25 -> max_prob=0.25, H = ln(4)
    probs[:, :, :8] = 0.25
    # Right half: near-certain class 1 -> H ~ 0.0
    probs[1, :, 8:] = 1.0
    # Top ignored rows: set adversarial values to confirm exclusion
    probs[:, :4, :] = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)[:, None, None]

    conf_summary, conf_map = analyze_confidence(probs, mask, high_threshold=0.80, medium_threshold=0.55)
    unc_summary, ent_map, norm_ent_map = analyze_uncertainty(probs, mask, high_uncertainty_threshold=0.55)

    assert conf_summary["is_calibrated_probability"] is False
    # Half valid pixels at 0.25, half at 1.0 -> mean = 0.625
    assert abs(conf_summary["mean_confidence"] - 0.625) < 1e-4
    assert abs(conf_summary["low_confidence_pixel_percentage"] - 50.0) < 1e-3

    expected_mean_h = 0.5 * math.log(4.0)
    assert abs(unc_summary["mean_entropy"] - expected_mean_h) < 1e-3
    assert abs(unc_summary["mean_normalized_entropy"] - 0.5) < 1e-3
    assert norm_ent_map[8, 2] > 0.99
    assert ent_map[8, 12] < 1e-4


def test_traversability_insufficient_evidence_and_schema_validation() -> None:
    rgb, mask, probs = _make_synthetic_scene()
    mask[:] = 255  # 100% ignored -> insufficient_visual_evidence
    cfg = IntelligenceConfig()
    terrain_res = analyze_terrain(mask, ignore_index=255)
    conf_res, conf_map = analyze_confidence(probs, mask)
    unc_res, _, _ = analyze_uncertainty(probs, mask)
    regions, _ = analyze_regions(mask, conf_map)
    cands = detect_obstacle_candidates(mask, probs, conf_map)
    trav = assess_experimental_traversability(terrain_res, regions, cands, conf_res, unc_res, cfg)

    assert trav["assessment"] == "insufficient_visual_evidence"
    assert trav["physical_safety_verified"] is False
    assert len(trav["missing_measurements"]) >= 3

    # Build minimal valid report and test schema validator rejection on unsafe mutation
    fake_report = {
        "project": "MANGAL DRISHTI",
        "schema_version": "1.0.0",
        "model": "deeplabv3plus",
        "checkpoint": "experiments/deeplabv3plus/best_model.pth",
        "device": "cpu",
        "image_metadata": {"image_id": "test", "width": 128, "height": 128},
        "analysis": {
            "terrain_summary": terrain_res["terrain_summary"],
            "terrain_details": terrain_res["terrain_details"],
            "regions": regions,
            "obstacle_candidates": cands,
            "confidence": conf_res,
            "uncertainty": unc_res,
            "traversability": trav,
        },
        "artifacts": {},
    }
    ok, errs = validate_intelligence_report(fake_report)
    assert ok, errs

    fake_report["analysis"]["traversability"]["physical_safety_verified"] = True
    ok_bad, errs_bad = validate_intelligence_report(fake_report)
    assert not ok_bad
    assert any("physical_safety_verified" in e for e in errs_bad)


def test_missing_image_raises_file_not_found() -> None:
    engine = TerrainIntelligenceEngine(model_name="deeplabv3plus", device="cpu")
    with pytest.raises(FileNotFoundError):
        engine.analyze("datasets/processed/does_not_exist_12345.png", save_visualizations=False)


def test_real_rover_cpu_inference_and_schema(tmp_path: Path) -> None:
    ds_val = MangalDrishtiDataset(split="val")
    img_path = ds_val.processed_dir / ds_val.records[0]["processed_image_relpath"]

    engine = TerrainIntelligenceEngine(model_name="deeplabv3plus", device="cpu")
    report = engine.analyze(img_path, output_dir=tmp_path / "cpu_run", save_visualizations=False)
    ok, errs = validate_intelligence_report(report)
    assert ok, errs
    assert report["device"] == "cpu"
    assert report["analysis"]["terrain_details"]["total_valid_pixels"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available in this environment")
def test_real_rover_gpu_inference_reuse_and_visualizations(tmp_path: Path) -> None:
    ds_val = MangalDrishtiDataset(split="val")
    img_1 = ds_val.processed_dir / ds_val.records[0]["processed_image_relpath"]
    img_2 = ds_val.processed_dir / ds_val.records[1]["processed_image_relpath"]

    engine = TerrainIntelligenceEngine(model_name="deeplabv3plus", device="cuda")
    rep_1 = engine.analyze(img_1, output_dir=tmp_path / "gpu_run_1", save_visualizations=True)
    rep_2 = engine.analyze(img_2, output_dir=tmp_path / "gpu_run_2", save_visualizations=True)

    assert engine.load_count == 1
    assert rep_1["device"].startswith("cuda")
    assert rep_2["device"].startswith("cuda")

    out_files = {p.name for p in (tmp_path / "gpu_run_1").iterdir()}
    expected_9_plus_json = {
        "01_original_image.png",
        "02_segmentation_mask.png",
        "03_terrain_overlay.png",
        "04_region_boundaries.png",
        "05_big_rock_candidates.png",
        "06_confidence_heatmap.png",
        "07_uncertainty_heatmap.png",
        "08_terrain_distribution.png",
        "09_intelligence_summary_panel.png",
        "intelligence_report.json",
    }
    assert expected_9_plus_json.issubset(out_files)
    loaded = json.loads((tmp_path / "gpu_run_1" / "intelligence_report.json").read_text(encoding="utf-8"))
    ok, errs = validate_intelligence_report(loaded)
    assert ok, errs
