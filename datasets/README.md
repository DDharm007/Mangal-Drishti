# MANGAL DRISHTI — Dataset Directory Guide

This directory houses the real Mars rover imagery, ground-truth semantic segmentation annotations, curated quick-reference samples, and automated inspection artifacts for **Step 01** of **MANGAL DRISHTI**.

## 1. Directory Organization

```text
datasets/
├── raw/
│   └── ai4mars/                     # Official NASA/JPL AI4MARS v0.6 structure (gitignored)
│       ├── changelog.md             # Official v0.1–v0.6 release notes
│       ├── info.md                  # Official dataset documentation
│       ├── label_keys.json          # Authoritative NAV & M2020_GEO class mappings
│       ├── split_manifest.json      # Chronological SCLK train/val split + official test splits
│       ├── official_v0.6_archive_manifest_summary.json
│       ├── msl/
│       │   └── ncam/
│       │       ├── images/
│       │       │   ├── edr/         # Original MSL (Curiosity) NavCam 1024x1024 grayscale JPGs
│       │       │   ├── mxy/         # Official rover hardware exclusion masks (PNG)
│       │       │   └── rng-30m/     # Official >30m stereo distance exclusion masks (PNG)
│       │       └── labels/
│       │           ├── train/       # Crowdsourced consensus masks (>=3 labelers, >=2/3 agreement)
│       │           └── test/
│       │               ├── masked-gold-min1-100agree/
│       │               ├── masked-gold-min2-100agree/
│       │               └── masked-gold-min3-100agree/  # Strictest JPL expert test masks
│       └── mer/
│           ├── images/
│           │   └── test/            # Official MER (Opportunity & Spirit) NavCam test JPGs
│           └── labels/
│               └── test/
│                   ├── masked-gold-min1-100agree/
│                   ├── masked-gold-min2-100agree/
│                   └── masked-gold-min3-100agree/
│
├── samples/                         # Curated quick-inspection sample subset (gitignored)
│   └── external_nasa/               # Unlabeled Perseverance & Curiosity raw images from NASA APIs
│
├── inspection/                      # Verified reports & visualizations (tracked in Git)
│   ├── dataset_summary.json
│   ├── validation_report.json
│   ├── 01_original_rover_image.png
│   ├── 02_ground_truth_segmentation.png
│   ├── 03_segmentation_overlay.png
│   ├── dataset_samples.png
│   └── terrain_distribution.png
│
└── README.md
```

## 2. Where Original Images and Masks Are Stored

- **Original MSL (Curiosity) Images:** `datasets/raw/ai4mars/msl/ncam/images/edr/*.JPG`
- **Original MER (Opportunity & Spirit) Test Images:** `datasets/raw/ai4mars/mer/images/test/*.JPG`
- **MSL Crowdsourced Merged Train/Val Masks:** `datasets/raw/ai4mars/msl/ncam/labels/train/*.png`
- **MSL Expert Gold-Standard Test Masks:** `datasets/raw/ai4mars/msl/ncam/labels/test/masked-gold-min{1,2,3}-100agree/*.png`
- **MER Expert Gold-Standard Test Masks:** `datasets/raw/ai4mars/mer/labels/test/masked-gold-min{1,2,3}-100agree/*.png`
- **Auxiliary Rover & Range Exclusion Masks (MSL):** `datasets/raw/ai4mars/msl/ncam/images/mxy/*.png` and `rng-30m/*.png`
- **External Qualitative Generalization Samples:** `datasets/samples/external_nasa/` (strictly separated from training/validation/test data and **never** mixed into AI4MARS training).

Original source images and ground-truth masks are preserved byte-for-byte with original filenames and CRC32 verification against the official Zenodo v0.6 archive.

## 3. Reproducible Acquisition

To reproduce dataset acquisition into `datasets/raw/ai4mars/` using the official NASA/JPL Zenodo v0.6 release (`DOI: 10.5281/zenodo.15995036`):

```powershell
.\.venv\Scripts\python.exe src/data/acquire_dataset.py
```

By default, this acquires:
- All **322** MSL Curiosity expert gold-standard test pairs (`min1`, `min2`, `min3`, plus `mxy` and `rng-30m`).
- All **204** MER Opportunity & Spirit expert gold-standard test pairs (`min1`, `min2`, `min3`).
- **1,800** MSL Curiosity NavCam train/validation pairs partitioned chronologically by Spacecraft Clock (SCLK) into **1,500 Train** and **300 Validation** pairs to prevent scene/temporal leakage.
- Official documentation files (`info.md`, `changelog.md`, `label_keys.json`, `NOTE.txt` files) and qualitative external NASA Raw Image API samples.

You can adjust `--train-val-limit` and `--train-count` if you wish to download a larger slice or the full 16.23 GB archive directly from `https://zenodo.org/records/15995036`.

## 4. Running Inspection and Visualization

1. **Run full dataset inventory, image/mask validation, and terrain-class distribution analysis:**
   ```powershell
   .\.venv\Scripts\python.exe src/data/inspect_dataset.py
   ```
   This produces `datasets/inspection/dataset_summary.json` and `datasets/inspection/validation_report.json`.

2. **Generate all visual exploration figures:**
   ```powershell
   .\.venv\Scripts\python.exe src/data/visualization.py
   ```
   This produces `01_original_rover_image.png`, `02_ground_truth_segmentation.png`, `03_segmentation_overlay.png`, `dataset_samples.png`, and `terrain_distribution.png` in `datasets/inspection/`.

## 5. Git Exclusion Policy

Large raw datasets, binary archives (`.zip`, `.tar.gz`, `.parquet`), and raw rover image/mask directories (`datasets/raw/`, `datasets/samples/`) are strictly excluded in `.gitignore`. Tracking gigabytes of immutable imagery in Git bloats repository history, slows clones, and degrades developer tooling. Instead, acquisition is reproducible via `src/data/acquire_dataset.py`, while lightweight verified inspection summaries (`*.json`) and exploratory charts (`datasets/inspection/*.png`) are kept trackable.
