"""Independent Validation on Previously Unseen Authoritative NASA Mars Imagery."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intelligence.engine import TerrainIntelligenceEngine
from src.intelligence.terrain_statistics import CLASS_ID_TO_DISPLAY, TERRAIN_PALETTE_RGB

OUT_ROOT = ROOT / "docs" / "validation" / "unseen_mars_imagery"
ORIG_DIR = OUT_ROOT / "originals"
PRED_DIR = OUT_ROOT / "predictions"
COMP_DIR = OUT_ROOT / "comparisons"
META_DIR = OUT_ROOT / "metadata"
NOTES_DIR = OUT_ROOT / "research_notes"

MODELS = [
    ("deeplabv3plus", "DeepLabV3+ Baseline"),
    ("unet", "U-Net Baseline"),
    ("deeplabv3plus_optimized", "DeepLabV3+ Optimized"),
    ("unet_optimized", "U-Net Optimized"),
]

M2020_SURFACE_CAMS = {"NAVCAM_LEFT", "NAVCAM_RIGHT", "MCZ_LEFT", "MCZ_RIGHT", "FRONT_HAZCAM_LEFT_A"}
MSL_SURFACE_CAMS = {"NAV_LEFT_B", "NAV_RIGHT_B", "NAV_LEFT_A", "NAV_RIGHT_A", "MAST_LEFT", "MAST_RIGHT"}


def load_known_project_ids() -> Set[str]:
    known: Set[str] = set()
    manifest_p = ROOT / "datasets" / "metadata" / "split_manifest.json"
    if manifest_p.exists():
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        for records in manifest.get("splits", {}).values():
            for r in records:
                if r.get("id"):
                    known.add(str(r["id"]))
        for r in manifest.get("rejected_records", []):
            if r.get("id"):
                known.add(str(r["id"]))
    intel_dir = ROOT / "experiments" / "intelligence"
    if intel_dir.exists():
        for d in intel_dir.iterdir():
            if d.is_dir():
                known.add(d.name)
    return known


def fetch_candidates(
    category: str,
    allowed_cams: Set[str],
    min_sol: int,
    max_items: int = 80,
    search_cam: str | None = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    cam_param = f"&search={search_cam}" if search_cam else ""
    for page in (0, 1, 12, 20):
        url = (
            f"https://mars.nasa.gov/rss/api/?feed=raw_images&category={category}"
            f"&feedtype=json&num=60&page={page}&order=sol+desc{cam_param}"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        imgs = resp.json().get("images", [])
        for item in imgs:
            cam = str(item.get("camera", {}).get("instrument", ""))
            sample_type = str(item.get("sample_type", "")).lower()
            sol = int(item.get("sol") or 0)
            img_files = item.get("image_files") or {}
            dl_url = img_files.get("large") or img_files.get("medium") or img_files.get("full_res")
            if cam in allowed_cams and sample_type == "full" and sol >= min_sol and dl_url:
                candidates.append(item)
        if len(candidates) >= max_items:
            break
    return candidates


def is_usable_surface_frame(img_path: Path) -> tuple[bool, tuple[int, int]]:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w < 512 or h < 512:
            return False, (w, h)
        arr = np.asarray(rgb, dtype=np.float32)
        # Reject blank/pitch-black nighttime frames or flat calibration images
        if float(arr.mean()) < 22.0 or float(arr.std()) < 16.0:
            return False, (w, h)
        return True, (w, h)


def render_side_by_side(
    orig_path: Path,
    overlay_path: Path,
    record: Dict[str, Any],
    rep_dl: Dict[str, Any],
    out_path: Path,
) -> None:
    orig_img = np.asarray(Image.open(orig_path).convert("RGB"))
    over_img = np.asarray(Image.open(overlay_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=140, facecolor="#0D0E0E")
    for ax in axes:
        ax.set_facecolor("#121414")
        ax.axis("off")

    axes[0].imshow(orig_img)
    axes[0].set_title(
        f"ORIGINAL NASA RAW IMAGE\n{record['mission']} · {record['camera']} · Sol {record['sol']}\n"
        f"ID: {record['nasa_image_id']} ({record['original_resolution']})",
        color="#EAE9E4",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )

    ts = rep_dl["analysis"]["terrain_summary"]
    conf = rep_dl["analysis"]["confidence"]["mean_confidence"]
    ent = rep_dl["analysis"]["uncertainty"]["mean_entropy"]
    concern = rep_dl["analysis"]["traversability"]["assessment"].replace("_", " ").upper()

    axes[1].imshow(over_img)
    axes[1].set_title(
        f"MANGAL DRISHTI DETECTION (DeepLabV3+ Baseline · 512×512)\n"
        f"Soil {ts['soil']:.1f}% | Bedrock {ts['bedrock']:.1f}% | Sand {ts['sand']:.1f}% | Big Rock {ts['big_rock']:.1f}%\n"
        f"Mean Conf: {conf:.3f} | Entropy: {ent:.3f} nats | {concern}",
        color="#EAE9E4",
        fontsize=9.0,
        fontweight="bold",
        pad=8,
    )

    handles = [
        mpatches.Patch(color=np.array(TERRAIN_PALETTE_RGB[i]) / 255.0, label=f"{CLASS_ID_TO_DISPLAY[i]} ({i})")
        for i in range(4)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#181A19",
        edgecolor="#303431",
        labelcolor="#EAE9E4",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.5,
        0.008,
        "QUALITATIVE INFERENCE ON UNSEEN NASA IMAGERY — NO EXPERT GROUND-TRUTH MASK — PHYSICAL SAFETY VERIFIED: NO",
        ha="center",
        color="#C87951",
        fontsize=8,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(out_path, facecolor="#0D0E0E", bbox_inches="tight")
    plt.close(fig)


def render_detailed_6panel(
    pred_dir: Path,
    record: Dict[str, Any],
    rep_dl: Dict[str, Any],
    out_path: Path,
) -> None:
    files = [
        ("01_original_image.png", "1. Original Input (512×512)"),
        ("02_segmentation_mask.png", "2. Predicted Mask"),
        ("03_terrain_overlay.png", "3. Segmentation Overlay"),
        ("06_confidence_heatmap.png", f"4. Softmax Confidence ({rep_dl['analysis']['confidence']['mean_confidence']:.3f})"),
        ("07_uncertainty_heatmap.png", f"5. Predictive Entropy ({rep_dl['analysis']['uncertainty']['mean_entropy']:.3f} nats)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), dpi=135, facecolor="#0D0E0E")
    for idx, (fname, title) in enumerate(files):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#121414")
        ax.imshow(np.asarray(Image.open(pred_dir / fname).convert("RGB")))
        ax.set_title(title, color="#EAE9E4", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Panel 6: Dark Terrain Composition Bar Chart
    ax_bar = axes[1, 2]
    ax_bar.set_facecolor("#181A19")
    keys = ["soil", "bedrock", "sand", "big_rock"]
    labels = [CLASS_ID_TO_DISPLAY[i] for i in range(4)]
    ts = rep_dl["analysis"]["terrain_summary"]
    vals = [float(ts[k]) for k in keys]
    colors = [np.array(TERRAIN_PALETTE_RGB[i]) / 255.0 for i in range(4)]
    y_pos = np.arange(4)
    bars = ax_bar.barh(y_pos, vals, color=colors, edgecolor="#303431", height=0.55)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="#EAE9E4", fontsize=9.5, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 108)
    ax_bar.tick_params(colors="#9B9E98", labelsize=8.5)
    for spine in ax_bar.spines.values():
        spine.set_color("#303431")
    ax_bar.set_title("6. Terrain Composition (%)", color="#EAE9E4", fontsize=10, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax_bar.text(min(v + 1.5, 92), bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", color="#EAE9E4", fontsize=8.5)

    fig.suptitle(
        f"Unseen NASA Validation — {record['mission']} | {record['nasa_image_id']} (Qualitative Only, No GT)",
        color="#EAE9E4",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, facecolor="#0D0E0E", bbox_inches="tight")
    plt.close(fig)


def render_all_models_strip(
    orig_path: Path,
    model_results: List[tuple[str, str, Path, Dict[str, Any]]],
    record: Dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.8), dpi=135, facecolor="#0D0E0E")
    axes[0].imshow(np.asarray(Image.open(orig_path).convert("RGB")))
    axes[0].set_title(f"Original NASA\n{record['camera']} Sol {record['sol']}", color="#EAE9E4", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, (m_key, m_label, pred_dir, rep) in enumerate(model_results, start=1):
        ts = rep["analysis"]["terrain_summary"]
        dom = rep["analysis"]["terrain_details"]["dominant_terrain"]
        axes[i].imshow(np.asarray(Image.open(pred_dir / "03_terrain_overlay.png").convert("RGB")))
        axes[i].set_title(
            f"{m_label}\nDom: {dom} | Rock: {ts['big_rock']:.1f}% | Sand: {ts['sand']:.1f}%",
            color="#EAE9E4",
            fontsize=8.5,
            fontweight="bold",
        )
        axes[i].axis("off")

    fig.suptitle(
        f"Cross-Model Qualitative Comparison — {record['mission']} · {record['nasa_image_id']}",
        color="#EAE9E4",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out_path, facecolor="#0D0E0E", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for d in (ORIG_DIR, PRED_DIR, COMP_DIR, META_DIR, NOTES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    known_ids = load_known_project_ids()
    print(f"Loaded {len(known_ids)} known AI4MARS / existing report IDs for leakage prevention.")

    m2020_cands = fetch_candidates("mars2020", M2020_SURFACE_CAMS, min_sol=1200, search_cam="MCZ_LEFT")
    m2020_cands += fetch_candidates("mars2020", M2020_SURFACE_CAMS, min_sol=1200, search_cam="NAVCAM_LEFT")
    m2020_cands += fetch_candidates("mars2020", M2020_SURFACE_CAMS, min_sol=1200)
    msl_cands = fetch_candidates("msl", MSL_SURFACE_CAMS, min_sol=3800)

    selected_items: List[tuple[str, Dict[str, Any], Path, tuple[int, int]]] = []

    def try_select(mission_name: str, pool: List[Dict[str, Any]], target_count: int) -> None:
        used_cams: Set[str] = set()
        used_sols: Set[int] = set()
        picked = 0
        for prefer_new_cam in (True, False):
            for item in pool:
                if picked >= target_count:
                    break
                img_id = str(item.get("imageid", ""))
                sol = int(item.get("sol") or 0)
                if not img_id or img_id in known_ids or sol in used_sols:
                    continue
                cam = str(item.get("camera", {}).get("instrument", "UNKNOWN"))
                if prefer_new_cam and cam in used_cams:
                    continue
                img_files = item.get("image_files") or {}
                src_url = str(img_files.get("large") or img_files.get("medium") or img_files.get("full_res"))
                ext = Path(src_url.split("?")[0]).suffix or ".jpg"
                orig_p = ORIG_DIR / f"{img_id}{ext}"

                try:
                    dl = requests.get(src_url, timeout=25)
                    dl.raise_for_status()
                    orig_p.write_bytes(dl.content)
                    ok, dims = is_usable_surface_frame(orig_p)
                    if not ok:
                        orig_p.unlink(missing_ok=True)
                        continue
                except Exception as exc:
                    print(f"Skipping {img_id}: {exc}")
                    orig_p.unlink(missing_ok=True)
                    continue

                known_ids.add(img_id)
                used_cams.add(cam)
                used_sols.add(sol)
                selected_items.append((mission_name, item, orig_p, dims))
                picked += 1
                print(f"Selected unseen {mission_name}: {img_id} ({cam}, Sol {sol}, {dims[0]}x{dims[1]})")
            if picked >= target_count:
                break

    try_select("Mars 2020 (Perseverance)", m2020_cands, 2)
    try_select("MSL (Curiosity)", msl_cands, 2)

    if len(selected_items) < 4:
        raise RuntimeError(f"Expected 4 distinct unseen NASA images, only acquired {len(selected_items)}")

    # Initialize all 4 models once
    engines = {m_key: TerrainIntelligenceEngine(model_name=m_key) for m_key, _ in MODELS}
    dl_timestamp = datetime.now(timezone.utc).isoformat()

    manifest_records: List[Dict[str, Any]] = []
    notes_md_lines = [
        "# Independent Validation Notes — Previously Unseen NASA Mars Imagery\n",
        "**Important Scientific Notice:** All results below represent **qualitative inference** on newly acquired NASA surface frames with **no expert ground-truth masks**. IoU, Dice, precision, and recall are not computed or claimed, and outputs do not verify physical rover traversability.\n",
    ]

    for mission_name, item, orig_p, (orig_w, orig_h) in selected_items:
        img_id = str(item["imageid"])
        img_files = item.get("image_files") or {}
        src_url = str(img_files.get("large") or img_files.get("medium") or img_files.get("full_res"))
        cam = str(item.get("camera", {}).get("instrument", "UNKNOWN"))
        sol = int(item.get("sol") or 0)
        acq_date = item.get("date_taken_utc") or item.get("date_received") or "Unknown"

        rec_meta: Dict[str, Any] = {
            "nasa_image_id": img_id,
            "mission": mission_name,
            "rover": "Perseverance" if "Perseverance" in mission_name else "Curiosity",
            "camera": cam,
            "sol": sol,
            "acquisition_date_utc": str(acq_date),
            "download_date_utc": dl_timestamp,
            "source_url": src_url,
            "original_local_path": str(orig_p.relative_to(ROOT)).replace("\\", "/"),
            "original_resolution": f"{orig_w}x{orig_h}",
            "leakage_verified": True,
            "ground_truth_available": False,
            "caption": str(item.get("caption") or item.get("title") or ""),
            "model_predictions": {},
            "comparison_artifacts": {},
        }

        model_runs: List[tuple[str, str, Path, Dict[str, Any]]] = []
        for m_key, m_label in MODELS:
            m_out = PRED_DIR / m_key / img_id
            rep = engines[m_key].analyze(orig_p, output_dir=m_out, save_visualizations=True)
            ts = rep["analysis"]["terrain_summary"]
            td = rep["analysis"]["terrain_details"]
            conf = rep["analysis"]["confidence"]
            unc = rep["analysis"]["uncertainty"]
            trav = rep["analysis"]["traversability"]

            rec_meta["model_predictions"][m_key] = {
                "model_label": m_label,
                "device": rep["device"],
                "inference_latency_ms": rep["image_metadata"]["inference_latency_ms"],
                "processed_dimensions": f"{rep['image_metadata']['width']}x{rep['image_metadata']['height']}",
                "dominant_terrain": td["dominant_terrain"],
                "secondary_terrain": td["secondary_terrain"],
                "terrain_summary": ts,
                "mean_confidence": conf["mean_confidence"],
                "low_confidence_pixel_percentage": conf["low_confidence_pixel_percentage"],
                "mean_entropy_nats": unc["mean_entropy"],
                "mean_normalized_entropy": unc["mean_normalized_entropy"],
                "retained_regions": len(rep["analysis"]["regions"]),
                "big_rock_candidates_2d": len(rep["analysis"]["obstacle_candidates"]),
                "visual_concern_assessment": trav["assessment"],
                "experimental_concern_index": trav["experimental_visual_concern_index"],
                "physical_safety_verified": trav["physical_safety_verified"],
                "report_path": str((m_out / "intelligence_report.json").relative_to(ROOT)).replace("\\", "/"),
            }
            model_runs.append((m_key, m_label, m_out, rep))

        dl_dir = PRED_DIR / "deeplabv3plus" / img_id
        dl_rep = model_runs[0][3]
        sbs_path = COMP_DIR / f"{img_id}_side_by_side.png"
        pan6_path = COMP_DIR / f"{img_id}_detailed_6panel.png"
        all4_path = COMP_DIR / f"{img_id}_all_4_models.png"

        render_side_by_side(orig_p, dl_dir / "03_terrain_overlay.png", rec_meta, dl_rep, sbs_path)
        render_detailed_6panel(dl_dir, rec_meta, dl_rep, pan6_path)
        render_all_models_strip(orig_p, model_runs, rec_meta, all4_path)

        rec_meta["comparison_artifacts"] = {
            "side_by_side": str(sbs_path.relative_to(ROOT)).replace("\\", "/"),
            "detailed_6panel": str(pan6_path.relative_to(ROOT)).replace("\\", "/"),
            "all_4_models": str(all4_path.relative_to(ROOT)).replace("\\", "/"),
        }

        # Qualitative scientific observation note grounded strictly in predicted stats
        dl_m = rec_meta["model_predictions"]["deeplabv3plus"]
        unet_m = rec_meta["model_predictions"]["unet"]
        opt_m = rec_meta["model_predictions"]["deeplabv3plus_optimized"]
        dl_ts = dl_m["terrain_summary"]
        rec_meta["research_interpretation"] = (
            f"DeepLabV3+ Baseline segmented this unseen {mission_name} {cam} (Sol {sol}) frame primarily as "
            f"{dl_m['dominant_terrain']} ({dl_ts[dl_m['dominant_terrain']]:.1f}%), with secondary terrain mix "
            f"(Soil {dl_ts['soil']:.1f}%, Bedrock {dl_ts['bedrock']:.1f}%, Sand {dl_ts['sand']:.1f}%, "
            f"Big Rock {dl_ts['big_rock']:.1f}%). Mean softmax confidence was {dl_m['mean_confidence']:.3f} "
            f"(mean entropy {dl_m['mean_entropy_nats']:.3f} nats, {dl_m['big_rock_candidates_2d']} 2D Big Rock candidates). "
            f"U-Net Baseline predicted {unet_m['dominant_terrain']} as dominant ({unet_m['big_rock_candidates_2d']} rock candidates), "
            f"and DeepLabV3+ Optimized predicted {opt_m['dominant_terrain']} (mean conf {opt_m['mean_confidence']:.3f}). "
            f"Cross-mission optic/scale differences (especially Perseverance NAVCAM_LEFT/MCZ_LEFT FOV vs MSL AI4MARS training priors) "
            f"are treated strictly qualitatively without ground-truth labels."
        )

        notes_md_lines.extend([
            f"\n## {img_id} ({mission_name} · {cam} · Sol {sol})\n",
            f"- **Source URL:** {src_url}",
            f"- **Acquisition UTC:** {acq_date} | **Original Resolution:** {orig_w}×{orig_h} -> Processed 512×512",
            f"- **Ground Truth Available:** False (No expert annotation mask)",
            f"- **DeepLabV3+ Baseline:** Soil `{dl_ts['soil']:.1f}%`, Bedrock `{dl_ts['bedrock']:.1f}%`, Sand `{dl_ts['sand']:.1f}%`, Big Rock `{dl_ts['big_rock']:.1f}%` | Mean Conf `{dl_m['mean_confidence']:.3f}` | Entropy `{dl_m['mean_entropy_nats']:.3f}` | Assessment `{dl_m['visual_concern_assessment']}` | `physical_safety_verified: False`",
            f"- **Research Note:** {rec_meta['research_interpretation']}\n",
        ])

        manifest_records.append(rec_meta)

    manifest_out = {
        "title": "Independent Validation on Previously Unseen NASA Mars Imagery",
        "generated_at_utc": dl_timestamp,
        "dataset_leakage_checked": True,
        "ground_truth_available": False,
        "scientific_disclaimer": (
            "Qualitative inference only. Ground-truth segmentation masks do not exist for newly acquired "
            "NASA raw images; accuracy, IoU, Dice, precision, recall, and physical rover safety cannot be computed."
        ),
        "num_images": len(manifest_records),
        "observations": manifest_records,
    }
    (META_DIR / "unseen_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    (NOTES_DIR / "unseen_validation_notes.md").write_text("\n".join(notes_md_lines), encoding="utf-8")
    print(f"Saved unseen validation manifest and {len(manifest_records)} comparisons under {OUT_ROOT}")


if __name__ == "__main__":
    main()
