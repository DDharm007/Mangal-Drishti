# MANGAL DRISHTI — Step 5: Terrain Intelligence Engine

## 1. Architecture Overview

The **Terrain Intelligence Engine** (`src/intelligence/`) transforms pixel-level Mars rover semantic segmentation logits (`UNetResNet34`, `DeepLabV3PlusResNet50`, and their Step 4 optimized variants) into structured geospatial perception reports, 2D obstacle candidates, confidence/uncertainty diagnostics, experimental visual concern indicators, and a 9-artifact visualization bundle.

```text
Rover Image (512x512 RGB)
   │
   ▼
Segmentation Backbone (cached once in TerrainIntelligenceEngine)
   ├── Predicted Class Mask [H, W] ∈ {0: Soil, 1: Bedrock, 2: Sand, 3: Big Rock, 255: Ignore}
   └── Softmax Probability Tensor [4, H, W]
         │
         ├──► terrain_analyzer.py       (valid coverage, dominance, diversity, boundary density)
         ├──► confidence_analyzer.py    (max softmax prob, uncalibrated confidence tiers, histogram)
         ├──► uncertainty.py            (predictive Shannon entropy H = -Σ p log p)
         ├──► region_analyzer.py        (OpenCV 8-connected regions + min_region_area_px filtering)
         ├──► obstacle_detector.py      (2D Big Rock candidates + shape descriptors, never physical proof)
         ├──► traversability.py         (experimental 2D visual concern index + safety disclaimer)
         └──► report_generator.py       (9 visualizations + schema-validated intelligence_report.json)
```

---

## 2. Module Responsibilities

| Module | File | Responsibility |
| :--- | :--- | :--- |
| Engine & CLI | [engine.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/engine.py) | Model caching (single load), GPU/CPU inference & fallback, optional ignore-mask application, end-to-end orchestration & CLI. |
| Schemas & Validator | [schemas.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/schemas.py) | `IntelligenceConfig` thresholds/weights and strict `validate_intelligence_report()` enforcing non-safety invariants. |
| Terrain Statistics | [terrain_statistics.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/terrain_statistics.py) | Canonical class IDs, exact Step 2 RGB palette, boundary detection, Shannon diversity, and 3×3 grid localization. |
| Terrain Analyzer | [terrain_analyzer.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/terrain_analyzer.py) | Valid pixel counts (excluding `255`), class percentages, dominant/secondary classes, diversity, boundary density. |
| Region Analyzer | [region_analyzer.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/region_analyzer.py) | OpenCV connected components & external contours per class with configurable `min_region_area_px` filtering. |
| Obstacle Detector | [obstacle_detector.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/obstacle_detector.py) | 2D Big Rock candidate extraction with bounding boxes, centroids, circularity, solidity, extent, and confidence. |
| Confidence Analyzer | [confidence_analyzer.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/confidence_analyzer.py) | Max softmax probability stats, per-class mean confidence, configurable `high/medium/low` breakdown, histogram. |
| Uncertainty Analyzer | [uncertainty.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/uncertainty.py) | Predictive Shannon entropy $H = -\sum_{c=0}^{3} p_c \ln(p_c)$ and normalized uncertainty $H / \ln(4)$. |
| Traversability Proxy | [traversability.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/traversability.py) | Experimental 2D visual concern index combining terrain mix, Big Rock density, sand, uncertainty, and fragmentation. |
| Report Generator | [report_generator.py](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/src/intelligence/report_generator.py) | Generates all 9 visualizations + `09_intelligence_summary_panel.png` and persists `intelligence_report.json`. |

---

## 3. Terrain, Confidence & Uncertainty Methodologies

### 3.1 Terrain Calculations & Ignored Pixels (`255`)
- All valid calculations mask out `pred_mask == 255` (rover hardware MXY masks and out-of-range >30m RNG masks where supplied).
- Class percentages divide class pixel counts by `total_valid_pixels` (`Soil + Bedrock + Sand + Big Rock = 100%` when `total_valid_pixels > 0`).
- **Terrain Diversity:** Shannon entropy over class proportions $H_{\text{terrain}} = -\sum_{c} f_c \ln(f_c)$ normalized by $\ln(4)$.
- **Boundary Density:** Percentage of valid pixels adjacent (4-connected) to at least one pixel of a different valid terrain class.

### 3.2 Softmax Confidence Methodology
- Computed per valid pixel as $C(i, j) = \max_{c \in \{0,1,2,3\}} p_c(i, j)$.
- Default configurable categories:
  - **High Confidence:** $C \ge 0.80$
  - **Medium Confidence:** $0.55 \le C < 0.80$
  - **Low Confidence:** $C < 0.55$
- **Calibration Caveat:** Every report explicitly marks `"is_calibrated_probability": false`. Raw softmax probabilities reflect logit dominance under MSL training priors and do **not** guarantee physical accuracy (especially out-of-distribution on MER imagery).

### 3.3 Predictive Entropy Methodology
- Exact Shannon entropy over the 4-class softmax distribution:
  $$H(i, j) = -\sum_{c=0}^{3} p_c(i, j) \ln\bigl(\max(p_c(i, j), 10^{-12})\bigr)$$
- Normalized entropy $H_{\text{norm}}(i, j) = H(i, j) / \ln(4) \in [0, 1]$, where $0$ corresponds to a one-hot prediction and $1$ corresponds to uniform uncertainty across all 4 terrain classes (`0.25` each). High-uncertainty threshold defaults to $H_{\text{norm}} \ge 0.55$.

---

## 4. Region Analysis & 2D Obstacle Candidate Logic

- **Region Filtering:** OpenCV `cv2.connectedComponentsWithStats(..., connectivity=8)` extracts connected patches per class while discarding components with `pixel_area < min_region_area_px` (default `120` px) so isolated pixel noise does not pollute region counts. The raw `pred_mask` itself is never altered.
- **Big Rock Candidates (`class_id = 3`):** Components with `pixel_area >= min_obstacle_area_px` (default `64` px) and `mean_big_rock_probability >= min_obstacle_confidence` (default `0.35`) are extracted with `bbox`, `centroid`, `aspect_ratio`, `circularity` ($4\pi A / P^2$), `solidity` ($A / A_{\text{hull}}$), and `bbox_extent` ($A / (w \cdot h)$).
- **Non-Physical Guarantee:** Every candidate enforces `"is_confirmed_physical_obstacle": false`. 2D image segmentation cannot distinguish a small nearby rock edge from a large hazardous boulder without stereo range.

---

## 5. Experimental Visual Concern Formula & Scientific Limitations

The preliminary 2D image-based indicator produces a dimensionless **Experimental Visual Concern Index** $\in [0, 1]$:

$$\text{ConcernIndex} = 0.45 \cdot S_{\text{terrain}} + 0.20 \cdot S_{\text{rock\_density}} + 0.15 \cdot S_{\text{sand}} + 0.15 \cdot S_{\text{uncertainty}} + 0.05 \cdot S_{\text{fragmentation}}$$

where class risk weights default to `soil=0.10`, `bedrock=0.25`, `sand=0.60`, `big_rock=1.00`.

Assessment mapping:
- **`insufficient_visual_evidence`**: Triggered automatically if `valid_pixel_percentage < 15.0%`, `mean_confidence < 0.50`, or `low_confidence_pixel_percentage > 45.0%`.
- **`lower_visual_concern`**: $\text{ConcernIndex} < 0.30$
- **`moderate_visual_concern`**: $0.30 \le \text{ConcernIndex} < 0.58$
- **`higher_visual_concern`**: $\text{ConcernIndex} \ge 0.58$

**Mandatory Scientific Limitations:**
- Every JSON report enforces `"physical_safety_verified": false` and lists missing physical quantities (`"3D depth / stereo range"`, `"terrain slope / elevation gradient"`, `"obstacle physical height & clearance"`, `"wheel-terrain slip & soil mechanics"`, `"rover kinematics & suspension geometry"`).
- **Never** treat any output label (`lower_visual_concern`, etc.) as certified rover safety or autonomous path authorization.

---

## 6. Structured Output Schema Summary

```json
{
  "project": "MANGAL DRISHTI",
  "schema_version": "1.0.0",
  "model": "deeplabv3plus",
  "checkpoint": "experiments/deeplabv3plus/best_model.pth",
  "device": "cuda",
  "image_metadata": { "image_id": "...", "width": 512, "height": 512, "inference_latency_ms": 28.4 },
  "analysis": {
    "terrain_summary": { "soil": 0.0, "bedrock": 0.0, "sand": 0.0, "big_rock": 0.0 },
    "terrain_details": { "...": "..." },
    "regions": [ { "region_id": "REG_001", "...": "..." } ],
    "obstacle_candidates": [ { "candidate_id": "OBS_CAND_001", "is_confirmed_physical_obstacle": false } ],
    "confidence": { "mean_confidence": 0.0, "is_calibrated_probability": false },
    "uncertainty": { "mean_entropy": 0.0, "mean_normalized_entropy": 0.0 },
    "traversability": {
      "assessment": "moderate_visual_concern",
      "is_experimental": true,
      "physical_safety_verified": false,
      "missing_measurements": [ "..." ]
    }
  },
  "artifacts": { "intelligence_summary_panel": "...", "intelligence_report_json": "..." }
}
```

---

## 7. Usage Instructions

### Python API
```python
from src.intelligence import TerrainIntelligenceEngine

engine = TerrainIntelligenceEngine(model_name="deeplabv3plus")  # Loads model once
report = engine.analyze(
    image_path="datasets/processed/val/images/NLA_398380694EDR_F0030000NCAM15000M1.png",
    output_dir="experiments/intelligence/sample_run",
)
print(report["analysis"]["terrain_summary"])
print(report["analysis"]["traversability"]["assessment"])
```

### CLI
```powershell
# Analyze a single rover image
.\.venv\Scripts\python.exe -m src.intelligence.engine --image path/to/rover_image.png --model deeplabv3plus

# Run diverse multi-mission batch (6 real rover images)
.\.venv\Scripts\python.exe -m src.intelligence.engine --batch-samples 6 --model deeplabv3plus
```
