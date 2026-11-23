"""Integration tests for Step 6 FastAPI Mission Control API."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import INTEL_DIR, app
from src.data.dataset import MangalDrishtiDataset

client = TestClient(app)


def test_health_and_system_status() -> None:
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "online"

    r_sys = client.get("/api/system/status")
    assert r_sys.status_code == 200
    data = r_sys.json()
    assert data["stored_reports_count"] >= 6
    assert len(data["checkpoints"]) == 4
    assert all(isinstance(c["available"], bool) for c in data["checkpoints"])


def test_models_research_and_reports_endpoints() -> None:
    r_models = client.get("/api/models")
    assert r_models.status_code == 200
    mdata = r_models.json()
    assert len(mdata["catalog"]) == 4
    assert mdata["splits"]["test_combined"] is not None
    assert mdata["splits"]["test_msl"] is not None
    assert mdata["splits"]["test_mer"] is not None

    r_res = client.get("/api/research")
    assert r_res.status_code == 200
    rdata = r_res.json()
    assert rdata["class_mapping"]["num_trainable_classes"] == 4
    assert rdata["split_summary"]["counts"]["test"] == 526

    r_reps = client.get("/api/reports")
    assert r_reps.status_code == 200
    rep_list = r_reps.json()["reports"]
    assert len(rep_list) >= 6

    first_id = rep_list[0]["report_id"]
    r_det = client.get(f"/api/reports/{first_id}")
    assert r_det.status_code == 200
    assert r_det.json()["report"]["analysis"]["traversability"]["physical_safety_verified"] is False

    r_dl = client.get(f"/api/reports/{first_id}/download")
    assert r_dl.status_code == 200
    assert "application/json" in r_dl.headers["content-type"]

    # Negative cases: missing report 404 & unsafe identifier 400
    r_missing = client.get("/api/reports/does_not_exist_999")
    assert r_missing.status_code == 404
    r_bad_id = client.get("/api/reports/invalid$id")
    assert r_bad_id.status_code == 400
    assert len(mdata["catalog"]) == 4
    assert {m["id"] for m in mdata["catalog"]} == {
        "deeplabv3plus",
        "unet",
        "deeplabv3plus_optimized",
        "unet_optimized",
    }


def test_analyze_upload_validation_and_real_inference() -> None:
    # Reject invalid extension
    r_bad = client.post(
        "/api/analyze",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
        data={"model_name": "deeplabv3plus"},
    )
    assert r_bad.status_code == 400

    # Real Mars rover image inference via multipart upload
    ds_val = MangalDrishtiDataset(split="val")
    img_path = ds_val.processed_dir / ds_val.records[0]["processed_image_relpath"]
    created_dir: Path | None = None
    try:
        with img_path.open("rb") as fh:
            r_ok = client.post(
                "/api/analyze",
                files={"file": ("api_test_rover.png", fh, "image/png")},
                data={"model_name": "deeplabv3plus"},
            )
        assert r_ok.status_code == 200
        body = r_ok.json()
        created_dir = INTEL_DIR / body["report_id"]
        assert body["report"]["project"] == "MANGAL DRISHTI"
        assert body["report"]["analysis"]["traversability"]["physical_safety_verified"] is False
        assert body["artifact_urls"]["intelligence_summary_panel"] is not None
    finally:
        # Keep stored experiments/intelligence clean from synthetic API test upload artifacts
        if created_dir is not None and created_dir.exists():
            shutil.rmtree(created_dir)
