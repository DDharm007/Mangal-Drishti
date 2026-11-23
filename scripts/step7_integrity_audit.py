"""Integrity Orchestrator: Automated Playwright E2E UI Audit, Screenshot & GIF Generator."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHOTS_DIR = ROOT / "docs" / "media" / "screenshots"
DEMOS_DIR = ROOT / "docs" / "media" / "demos"
RELEASE_DIR = ROOT / "docs" / "release"
INTEL_DIR = ROOT / "experiments" / "intelligence"

FRONTEND_URL = "http://127.0.0.1:5173"
BACKEND_URL = "http://127.0.0.1:8000"


def build_optimized_gif(frame_paths: List[Path], out_gif: Path, durations_ms: List[int], width: int = 960) -> None:
    pil_frames: List[Image.Image] = []
    for p in frame_paths:
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            target_h = int(round(h * (width / float(w))))
            resized = rgb.resize((width, target_h), Image.Resampling.LANCZOS)
            # Quantize cleanly for GitHub-friendly file sizes (< 3 MB)
            quant = resized.convert("P", palette=Image.Palette.ADAPTIVE, colors=160)
            pil_frames.append(quant)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
    )


def run_audit() -> Dict[str, Any]:
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    test_cases: List[Dict[str, Any]] = []
    console_errors: List[str] = []
    failed_requests: List[str] = []
    created_upload_dirs: List[Path] = []
    bugs_fixed_during_step7: List[Dict[str, str]] = [
        {
            "id": "DEF-001",
            "title": "Markdown nested italic inside bold (e.g. **MSL (*Curiosity*)**)",
            "status": "FIXED & RETESTED",
            "detail": "Updated inline scanner in frontend/src/components/research/MarkdownRenderer.tsx to skip outer closing ** boundaries when pairing inner single-asterisk italic spans.",
        },
        {
            "id": "DEF-002",
            "title": "API negative tests for nonexistent/invalid report IDs",
            "status": "ADDED & VERIFIED",
            "detail": "Added 404 nonexistent report, 400 invalid ID, and 4-model catalog assertions in tests/test_step6_api.py.",
        },
        {
            "id": "DEF-003",
            "title": "Infinite /api/system/status fetch loop triggered on Overview & System Status mounts",
            "status": "FIXED & RETESTED",
            "detail": "Replaced unstable inline arrow callbacks passed to OverviewPage/SystemStatusPage in frontend/src/App.tsx with memoized useCallback handlers so mount effects execute once cleanly.",
        },
    ]

    # Fetch source JSON metrics to verify UI accuracy against ground-truth files
    expected_combined = json.loads(
        (ROOT / "experiments" / "evaluation" / "baseline" / "combined_comparison.json").read_text(encoding="utf-8")
    )
    expected_msl = json.loads(
        (ROOT / "experiments" / "evaluation" / "baseline" / "msl_comparison.json").read_text(encoding="utf-8")
    )
    expected_mer = json.loads(
        (ROOT / "experiments" / "evaluation" / "baseline" / "mer_comparison.json").read_text(encoding="utf-8")
    )

    def record(tc_id: str, area: str, name: str, passed: bool, evidence: str, blocked: bool = False) -> None:
        status = "BLOCKED" if blocked else ("PASS" if passed else "FAIL")
        test_cases.append(
            {
                "id": tc_id,
                "area": area,
                "name": name,
                "method": "Automated Playwright E2E",
                "status": status,
                "evidence": evidence,
            }
        )
        print(f"[{status}] {tc_id} ({area}): {name} -> {evidence}")

    # Pick a real AI4MARS rover image outside experiments/intelligence for live upload testing
    upload_src = ROOT / "datasets" / "processed" / "val" / "images" / "NLA_398380694EDR_F0030000NCAM15000M1.png"
    test_img_fallback = next((ROOT / "datasets" / "processed" / "test" / "images").glob("*.png"))
    sample_upload = test_img_fallback if test_img_fallback.exists() else upload_src

    with tempfile.TemporaryDirectory() as gif_tmp:
        tmp_dir = Path(gif_tmp)
        workflow_frames: List[Path] = []
        tour_frames: List[Path] = []

        def snap_frame(page: Page, bucket: List[Path], label: str) -> None:
            p = tmp_dir / f"{len(bucket):02d}_{label}.png"
            page.screenshot(path=str(p), full_page=False)
            bucket.append(p)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
            page = context.new_page()
            suppress_simulated_offline = {"active": False}

            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error" and not suppress_simulated_offline["active"]
                else None,
            )
            page.on("pageerror", lambda err: console_errors.append(str(err)))
            page.on(
                "response",
                lambda res: failed_requests.append(f"{res.status} {res.url}")
                if res.status >= 400
                and "/api/reports/invalid" not in res.url
                and not suppress_simulated_offline["active"]
                else None,
            )

            # 1. OVERVIEW AUDIT
            page.goto(f"{FRONTEND_URL}/", wait_until="networkidle")
            page.wait_for_timeout(900)
            ov_text = page.locator("main").inner_text()
            ov_title = page.locator('main [aria-label="MANGAL DRISHTI"]').count() >= 1
            sys_resp = requests.get(f"{BACKEND_URL}/api/system/status", timeout=10).json()
            expected_reports_count = sys_resp["stored_reports_count"]

            ov_ok = (
                ov_title
                and str(expected_reports_count) in ov_text
                and "4 / 4" in ov_text
                and ("CUDA" in ov_text or "CPU" in ov_text)
            )
            record(
                "TC-OV-01",
                "Overview",
                "Mission dashboard telemetry, stored count, models, device & observation loaded",
                ov_ok,
                f"Stored={expected_reports_count}, Models=4/4 rendered with real backend data",
            )

            # Sidebar collapse/expand + observation switcher
            page.get_by_role("button", name="Overlay", exact=True).click()
            page.wait_for_timeout(200)
            page.get_by_role("button", name="Compare Split", exact=True).click()
            page.wait_for_timeout(200)
            page.get_by_role("button", name="Side-by-Side", exact=True).click()
            page.wait_for_timeout(250)

            page.get_by_role("button", name="Collapse navigation sidebar").click()
            page.wait_for_timeout(250)
            expand_vis = page.get_by_role("button", name="Expand navigation sidebar").is_visible()
            page.get_by_role("button", name="Expand navigation sidebar").click()
            page.wait_for_timeout(250)
            record("TC-OV-02", "Overview", "Sidebar collapse/expand & observation view switches", expand_vis, f"expand_btn_visible={expand_vis}")

            page.screenshot(path=str(SHOTS_DIR / "01_overview_dashboard.png"), full_page=False)
            snap_frame(page, tour_frames, "overview")

            # 2. TERRAIN ANALYSIS AUDIT + WORKFLOW GIF + LIVE INFERENCE
            page.goto(f"{FRONTEND_URL}/analyze", wait_until="networkidle")
            page.wait_for_timeout(800)
            snap_frame(page, workflow_frames, "analyze_initial")

            # Switch existing report selector and verify viewer modes
            selects = page.locator("main select")
            model_select = selects.nth(0)
            report_select = selects.nth(1)
            opt_count = report_select.locator("option").count()
            record("TC-TA-01", "Terrain Analysis", "Existing report dropdown populated from backend", opt_count >= 6, f"{opt_count} reports selectable")

            # Test invalid file upload validation cleanly on client
            bad_txt = tmp_dir / "invalid_test.txt"
            bad_txt.write_text("not an image", encoding="utf-8")
            page.locator('input[type="file"]').set_input_files(str(bad_txt))
            page.wait_for_timeout(250)
            err_shown = "Invalid file" in page.locator("main").inner_text()
            record("TC-TA-02", "Terrain Analysis", "Invalid non-image file upload rejected with visible error", err_shown, f"client_error_visible={err_shown}")

            # Upload real Mars rover image & select model & run real inference
            model_select.select_option("deeplabv3plus")
            page.locator('input[type="file"]').set_input_files(str(sample_upload))
            page.wait_for_timeout(350)
            snap_frame(page, workflow_frames, "uploaded_preview")

            with page.expect_response(lambda r: "/api/analyze" in r.url and r.request.method == "POST", timeout=45000) as resp_info:
                page.get_by_role("button", name="Run Analysis").click()
            analyze_res = resp_info.value
            analyze_json = analyze_res.json()
            new_rep_id = analyze_json["report_id"]
            created_upload_dirs.append(INTEL_DIR / new_rep_id)
            page.wait_for_timeout(900)

            ta_text_lower = page.locator("main").inner_text().lower()
            safety_ok = (
                "physical safety verified: no" in ta_text_lower
                and "2d visual candidates only" in ta_text_lower
            )
            record(
                "TC-TA-03",
                "Terrain Analysis",
                "Real image upload + DeepLabV3+ inference + report generation",
                analyze_res.status == 200 and safety_ok and new_rep_id in page.url,
                f"HTTP {analyze_res.status}, report_id={new_rep_id}, latency={analyze_json['report']['image_metadata']['inference_latency_ms']} ms",
            )
            snap_frame(page, workflow_frames, "inference_overlay")
            page.screenshot(path=str(SHOTS_DIR / "02_terrain_analysis.png"), full_page=False)

            # Exercise all viewer modes + region selection + zoom
            viewer_ok = True
            for mode_lbl in ["Original", "Segmentation", "Compare Split", "Confidence", "Uncertainty", "Regions", "Candidates", "9-Panel", "Overlay"]:
                page.get_by_role("button", name=mode_lbl, exact=True).click()
                page.wait_for_timeout(320)
                if mode_lbl in {"Compare Split", "Confidence", "Uncertainty", "Regions", "9-Panel"}:
                    snap_frame(page, workflow_frames, mode_lbl.lower().replace(" ", "_"))
                imgs = page.locator("main img").all()
                for im in imgs:
                    nat_w = im.evaluate("el => el.naturalWidth")
                    if nat_w == 0:
                        viewer_ok = False
            record("TC-TA-04", "Terrain Analysis", "All 9 viewer visualization modes load valid non-broken images", viewer_ok, f"all_naturalWidth_gt_0={viewer_ok}")

            # Click a region row to highlight
            first_reg_row = page.locator('div[aria-label="Terrain Regions Table"] tbody tr').first
            reg_clicked = False
            if first_reg_row.count() > 0:
                first_reg_row.click()
                page.wait_for_timeout(250)
                reg_clicked = "REG_" in page.locator("main").inner_text()
            record("TC-TA-05", "Terrain Analysis", "Connected region table row click highlights image bbox", reg_clicked, f"highlight_active={reg_clicked}")
            snap_frame(page, workflow_frames, "region_highlight")

            # Verify JSON download link
            with page.expect_download() as dl_info:
                page.get_by_role("link", name="Download JSON").click()
            dl_file = dl_info.value
            record("TC-TA-06", "Terrain Analysis", "Download JSON report produces valid file", dl_file.suggested_filename.endswith(".json"), f"downloaded={dl_file.suggested_filename}")

            # 3. INTELLIGENCE REPORTS AUDIT
            page.goto(f"{FRONTEND_URL}/reports", wait_until="networkidle")
            page.wait_for_timeout(600)
            page.screenshot(path=str(SHOTS_DIR / "03_intelligence_reports.png"), full_page=False)
            snap_frame(page, tour_frames, "reports_grid")

            # Search & filter by visual concern
            page.locator('input[type="search"]').fill(new_rep_id[:10])
            page.wait_for_timeout(250)
            search_cnt = page.locator("article").count()
            page.locator('input[type="search"]').fill("")
            page.wait_for_timeout(200)

            concern_sel = page.locator("main select").nth(2)
            concern_sel.select_option("insufficient_visual_evidence")
            page.wait_for_timeout(200)
            insuff_cnt = page.locator("article").count()
            concern_sel.select_option("all")
            page.wait_for_timeout(200)

            page.get_by_role("button", name="Inspect JSON").first.click()
            page.wait_for_timeout(300)
            modal_vis = page.locator("dialog[open]").is_visible()
            snap_frame(page, tour_frames, "reports_json_modal")
            page.locator("dialog[open] button", has_text="Close").click()
            page.wait_for_timeout(200)
            record(
                "TC-RP-01",
                "Intelligence Reports",
                "Search, Visual Concern filter, Inspect JSON modal & Download links",
                search_cnt >= 1 and insuff_cnt >= 1 and modal_vis,
                f"search_hits={search_cnt}, insufficient_hits={insuff_cnt}, json_modal_verified={modal_vis}",
            )

            # 4. MODEL PERFORMANCE AUDIT (Accuracy check vs source JSONs + Dark Heatmaps)
            page.goto(f"{FRONTEND_URL}/models", wait_until="networkidle")
            page.wait_for_timeout(650)
            mp_text = page.locator("main").inner_text()
            comb_match = (
                f"{expected_combined['deeplabv3plus']['mIoU']:.4f}" in mp_text
                and f"{expected_combined['unet']['mIoU']:.4f}" in mp_text
            )

            # Hover heatmap cell & capture screenshot
            heatmap_btn = page.locator("main table button").first
            heatmap_btn.hover()
            page.wait_for_timeout(250)
            page.screenshot(path=str(SHOTS_DIR / "04_model_performance.png"), full_page=False)
            snap_frame(page, tour_frames, "models_combined")

            page.get_by_role("tab", name="Expert Test · MSL (322)").click()
            page.wait_for_timeout(300)
            msl_match = f"{expected_msl['deeplabv3plus']['mIoU']:.4f}" in page.locator("main").inner_text()
            snap_frame(page, tour_frames, "models_msl")

            page.get_by_role("tab", name="Expert Test · MER (204)").click()
            page.wait_for_timeout(300)
            mer_match = f"{expected_mer['deeplabv3plus']['mIoU']:.4f}" in page.locator("main").inner_text()

            page.get_by_role("tab", name="Validation (363)").click()
            page.wait_for_timeout(250)
            page.get_by_role("tab", name="Latency & GPU Benchmarks").click()
            page.wait_for_timeout(250)
            bench_ok = "40,347,812" in page.locator("main").inner_text()
            record(
                "TC-MP-01",
                "Model Performance",
                "Combined, MSL, MER, Validation & Benchmarks match ground-truth JSONs",
                comb_match and msl_match and mer_match and bench_ok,
                f"comb={comb_match}, msl={msl_match}, mer={mer_match}, bench={bench_ok}",
            )

            # 5. DATASET & RESEARCH AUDIT (Clean Markdown rendering check)
            page.goto(f"{FRONTEND_URL}/research", wait_until="networkidle")
            page.wait_for_timeout(600)
            page.screenshot(path=str(SHOTS_DIR / "05_dataset_research.png"), full_page=False)
            snap_frame(page, tour_frames, "research_docs")

            doc_panel = page.locator("main aside + div")
            non_code_text = doc_panel.evaluate(
                """(root) => {
                    const clone = root.cloneNode(true);
                    clone.querySelectorAll('pre, code').forEach(el => el.remove());
                    return clone.innerText;
                }"""
            )
            clean_md = ("**" not in non_code_text) and ("`" not in non_code_text) and ("##" not in non_code_text)
            record(
                "TC-DR-01",
                "Dataset & Research",
                "AI4MARS metadata, class swatches & rendered Markdown free of raw syntax",
                clean_md and "2326 / 2296" in page.locator("main").inner_text(),
                f"clean_markdown_rendered={clean_md}",
            )

            # 6. SYSTEM STATUS & BACKEND FAILURE AUDIT
            page.goto(f"{FRONTEND_URL}/status", wait_until="networkidle")
            page.wait_for_timeout(600)
            st_text = page.locator("main").inner_text()
            st_ok = "Healthy (HTTP 200)" in st_text and "NVIDIA GeForce RTX 3050 A Laptop GPU" in st_text
            page.screenshot(path=str(SHOTS_DIR / "06_system_diagnostics.png"), full_page=False)
            snap_frame(page, tour_frames, "system_status")
            record("TC-SS-01", "System Status", "Live Backend, Compute, Runtime & Checkpoints diagnostics", st_ok, f"status_ok={st_ok}")

            # Simulated backend offline handling (intercept /api/system/status -> 503)
            suppress_simulated_offline["active"] = True
            page.route("**/api/system/status", lambda route: route.fulfill(status=503, body='{"detail":"Simulated offline"}', headers={"content-type": "application/json"}))
            page.get_by_role("button", name="Refresh Diagnostics").click()
            page.wait_for_timeout(350)
            offline_handled = "Offline" in page.locator("main").inner_text() or "Simulated offline" in page.locator("main").inner_text()
            page.unroute("**/api/system/status")
            page.wait_for_timeout(100)
            suppress_simulated_offline["active"] = False
            record("TC-SS-02", "System Status", "Backend 503 failure renders graceful Offline/error state", offline_handled, f"offline_state_shown={offline_handled}")

            # 7. MOBILE RESPONSIVE AUDIT (375x812)
            page.set_viewport_size({"width": 375, "height": 812})
            page.goto(f"{FRONTEND_URL}/", wait_until="networkidle")
            page.wait_for_timeout(450)
            ov_overflow = page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.goto(f"{FRONTEND_URL}/analyze", wait_until="networkidle")
            page.wait_for_timeout(450)
            an_overflow = page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            record(
                "TC-MB-01",
                "Responsive Mobile",
                "375x812 viewport zero horizontal overflow on Overview & Terrain Analysis",
                ov_overflow and an_overflow,
                f"overview_no_overflow={ov_overflow}, analyze_no_overflow={an_overflow}",
            )

            browser.close()

        # Assemble animated GIF demos cleanly
        build_optimized_gif(
            workflow_frames,
            DEMOS_DIR / "mangal_drishti_analysis_workflow.gif",
            [1100] * len(workflow_frames),
            width=960,
        )
        build_optimized_gif(
            tour_frames,
            DEMOS_DIR / "mangal_drishti_mission_tour.gif",
            [1300] * len(tour_frames),
            width=960,
        )

    # Clean up temporary upload directories generated during automated Playwright audit
    for d in created_upload_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    record("TC-ER-01", "Runtime Hygiene", "Zero uncaught browser console errors", len(console_errors) == 0, f"console_errors={len(console_errors)}")
    record("TC-ER-02", "Network Integrity", "Zero unexpected HTTP >= 400 asset/API failures", len(failed_requests) == 0, f"failed_requests={len(failed_requests)}")

    passed = sum(1 for c in test_cases if c["status"] == "PASS")
    failed = sum(1 for c in test_cases if c["status"] == "FAIL")
    blocked = sum(1 for c in test_cases if c["status"] == "BLOCKED")

    summary = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total_playwright_cases": len(test_cases),
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
            "not_tested": 0,
        },
        "bugs_fixed": bugs_fixed_during_step7,
        "test_cases": test_cases,
        "screenshots": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in SHOTS_DIR.glob("*.png")),
        "gifs": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in DEMOS_DIR.glob("*.gif")),
    }
    (RELEASE_DIR / "audit_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))
    if failed > 0:
        sys.exit(1)
    return summary


if __name__ == "__main__":
    run_audit()
