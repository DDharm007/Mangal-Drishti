# MANGAL DRISHTI — Step 02 Data Preprocessing & Engineering Pipeline

## 1. Pipeline Architecture

Step 02 transforms the raw NASA/JPL AI4MARS v0.6 dataset (`datasets/raw/ai4mars/`) into a deterministic, validated, PyTorch-ready semantic segmentation dataset under `datasets/processed/` (`train`, `val`, `test`) with full metadata traceability under `datasets/metadata/`.

```text
Raw AI4MARS v0.6 (1024×1024 Grayscale JPG + PNG Masks + MXY/RNG-30M)
   │
   ├──► src/data/mask_encoder.py      ──► datasets/metadata/class_mapping.json
   ├──► src/data/split_dataset.py     ──► datasets/metadata/split_manifest.json & dataset_manifest.csv
   ├──► src/data/preprocessing.py     ──► datasets/processed/{train,val,test}/{images,masks}/*.png
   │                                  ──► datasets/metadata/preprocessing_config.json
   ├──► src/data/class_weights.py     ──► datasets/metadata/class_weights.json
   ├──► src/data/transforms.py        ──► Geometric (image+mask) & Photometric (image-only) augmentations
   ├──► src/data/dataset.py           ──► MangalDrishtiDataset ([3, 512, 512] float32, [512, 512] int64)
   ├──► src/data/dataloader.py        ──► Train/Val/Test PyTorch DataLoaders
   ├──► src/data/validation.py        ──► datasets/inspection/preprocessing_validation_report.json
   └──► src/data/visualization.py     ──► datasets/inspection/preprocessing_samples/*.png
```

All pipeline parameters are centralized in [configs/data_config.yaml](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/configs/data_config.yaml).

---

## 2. Raw Dataset Structure

From Step 1 (`datasets/raw/ai4mars/`):
- **2,326 raw pairs** (`2,122` MSL Curiosity NavCam `EDR` + `204` MER Opportunity & Spirit NavCam `EFF`), all `1024 × 1024` single-channel grayscale JPEGs.
- **2,326 primary masks** (`1,800` MSL crowdsourced merged train/val + `322` MSL Gold Test `min3` + `204` MER Gold Test `min3`), plus `min1-100agree` and `min2-100agree` expert test masks (`322` MSL + `204` MER each) and `4,244` auxiliary MSL exclusion masks (`2,122` `MXY` rover hardware masks + `2,122` `RNG-30M` `>30 m` stereo range masks).
- Original raw images and masks are never modified or overwritten.

---

## 3. Image Preprocessing Methodology

Implemented in `src/data/preprocessing.py` (`load_and_preprocess_image`):
1. **Integrity Loading:** Each raw `.JPG` is opened and fully decoded with Pillow (`im.load()`). Any unreadable or corrupted file is logged and raises an explicit failure rather than being silently ignored (`failed_files_count: 0`).
2. **RGB Representation:** Single-channel grayscale (`mode="L"`) images are converted to 3-channel RGB (`[H, W, 3]`, `uint8`) so downstream ImageNet-pretrained backbones (e.g., ResNet-34/50/101 for U-Net and DeepLabV3+) operate natively without channel-replication hacks in model code.
3. **Lossless Output Storage:** Processed images are saved as lossless `.png` files under `datasets/processed/{train,val,test}/images/` to prevent repeated JPEG compression artifacts.

---

## 4. Image Resizing Strategy

- **Target Resolution:** **`512 × 512`** pixels (configurable in `configs/data_config.yaml` to support `256`, `512`, or `768` in future experiments).
- **Interpolation:** **Bilinear (`cv2.INTER_LINEAR`)** for RGB rover images.
- **Aspect Ratio Strategy:** `direct_square` (all 2,326 raw AI4MARS NavCam frames are natively `1024 × 1024` 1:1 square, so `2×` downsampling preserves exact 1:1 aspect ratio without cropping or distortion). For future non-square external imagery (such as `1200 × 874` Perseverance Mastcam-Z frames), `letterbox_pad` mode is also implemented (`0` image padding, `255` mask padding).

---

## 5. Normalization Strategy

- **Mode:** `dataset_train` (scales `[0, 255]` `uint8` pixels to `[0.0, 1.0]` `float32`, then standardizes per channel `(x - mean) / std`). ImageNet normalization is also supported via `normalization.mode: "imagenet"`.
- **Leakage Prevention:** Mean and standard deviation are computed **strictly across the `1,407` processed `train` split images** (never touching `val` or `test`):
  - **`computed_train_mean_rgb`:** `[0.24631, 0.24631, 0.24631]`
  - **`computed_train_std_rgb`:** `[0.113346, 0.113346, 0.113346]`
- Saved in `datasets/metadata/preprocessing_config.json`.

---

## 6. Mask Encoding & Class Mapping

Implemented in `src/data/mask_encoder.py` (`MaskEncoder`) and verified directly against `datasets/raw/ai4mars/label_keys.json` (`NAV` taxonomy):

| Original Pixel ID | Training Class ID | Terrain Class Name | Trainable | Ignored | Palette RGB |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **`0`** | **`0`** | `soil` | `true` | `false` | `[166, 97, 26]` |
| **`1`** | **`1`** | `bedrock` | `true` | `false` | `[55, 126, 184]` |
| **`2`** | **`2`** | `sand` | `true` | `false` | `[245, 194, 66]` |
| **`3`** | **`3`** | `big rock` | `true` | `false` | `[228, 26, 28]` |
| **`255`** | **`255`** | `unlabeled_null` | `false` | `true` | `[40, 44, 52]` |

- **Critical Interpolation Rule:** Segmentation masks are resized **strictly with Nearest-Neighbor interpolation (`cv2.INTER_NEAREST`)**, guaranteeing zero intermediate interpolated label values (e.g., `1.5`). Post-resize validation confirms `observed_mask_pixel_values == [0, 1, 2, 3, 255]`.
- **Auxiliary Exclusion Enforcement:** For MSL pairs, `MXY` (`rover hardware == 1`) and `RNG-30M` (`distance > 30 m == 1`) are explicitly overlaid onto raw masks as `255` prior to encoding.
- Exported to `datasets/metadata/class_mapping.json`.

---

## 7. Ignore Index Handling

- **Ignore Index:** **`255`** (`mask.ignore_index = 255`).
- Pixels encoded as `255` represent sky/horizon `> 30 m`, rover hardware (`MXY`), insufficient annotator consensus, or affine rotation border fill (`borderValue=255`).
- `MangalDrishtiDataset` outputs `int64` masks preserving `255`, directly compatible with `torch.nn.CrossEntropyLoss(ignore_index=255)`.

---

## 8. Augmentation Methodology

Implemented in `src/data/transforms.py` (`SegmentationTransformPipeline`) using OpenCV + NumPy:
- **Training Split Only (`split="train"`):**
  - **Geometric (applied identically to image and mask; `INTER_LINEAR` + `BORDER_REFLECT_101` for image, `INTER_NEAREST` + `borderValue=255` for mask):**
    - Horizontal flip (`p=0.5`)
    - Small-angle rotation (`±15°`, `p=0.4`)
    - Scale & crop (`scale=[0.85, 1.15]`, `p=0.4`)
    - Translation / shift (`±5%`, `p=0.3`)
  - **Photometric (applied strictly to image; mask untouched):**
    - CLAHE local contrast enhancement (`clip_limit=2.0`, `grid=8×8`, `p=0.25`)
    - Brightness & contrast jitter (`±15%`, `p=0.5`)
    - Mild gamma adjustment (`[0.85, 1.15]`, `p=0.35`)
    - Mild Gaussian blur (`3×3`, `p=0.15`)
    - Mild Gaussian sensor noise (`std=[0.01, 0.03]`, `p=0.20`)
- **Validation & Test Splits:** Zero random augmentation—deterministic normalization only.

---

## 9. Dataset Splitting Methodology

Implemented in `src/data/split_dataset.py`:
- **Empty-Mask Filtering:** Of the `1,800` MSL crowdsourced pairs, **`30`** masks with zero valid labeled pixels (`100%` `255`) were explicitly filtered and logged under `rejected_records` in `split_manifest.json`.
- **SCLK Temporal Drive Clustering:** The remaining `1,770` valid MSL pairs were grouped into **`193` independent Spacecraft Clock (`SCLK`) drive clusters** (`sclk_cluster_gap_seconds = 43,200` / 12 hours), ensuring consecutive frames from the same Sol/drive stay together.
- **Stratified Group Assignment:** Drive clusters were stratified by rare-class presence (`has_big_rock`, `has_sand`, `bedrock_dom`, `soil_dom`) and assigned deterministically (`seed=42`):
  - **`train`:** **`1,407`** pairs (`61.28%`, `164` SCLK clusters)
  - **`val`:** **`363`** pairs (`15.81%`, `29` SCLK clusters)
- **Official Expert Test Preservation:** All **`526`** official JPL expert gold-standard test pairs (`322` MSL + `204` MER, `22.91%`) remain strictly in `test`. For the `100` test images whose `min3-100agree` mask is empty (`100%` `255`), the active mask falls back cleanly to `min2-100agree` (`68` pairs) or `min1-100agree` (`32` pairs), with `consensus_tier_used` recorded per sample (`426` `min3`, `68` `min2`, `32` `min1`).

---

## 10. Data Leakage Prevention

Automated assertions in both `split_dataset.py` and `validation.py` verify:
- `train_val_id_overlap == 0`
- `train_test_id_overlap == 0`
- `val_test_id_overlap == 0`
- `train_val_sclk_cluster_overlap == 0`
- Normalization statistics (`mean`, `std`) and training class weights are computed **exclusively** on the `train` split.

---

## 11. Class Imbalance Analysis

Computed in `src/data/class_weights.py` (`datasets/metadata/class_weights.json`) on the processed `512 × 512` masks:

| Split (`n`) | Valid Pixels (%) | Ignored `255` (%) | Soil (`0`) | Bedrock (`1`) | Sand (`2`) | Big Rock (`3`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`train` (`1,407`)** | `53.69%` | `46.31%` | `47.55%` | `46.27%` | `5.57%` | `0.62%` |
| **`val` (`363`)** | `51.03%` | `48.97%` | `45.62%` | `47.58%` | `5.65%` | `1.14%` |
| **`test` (`526`)** | `38.51%` | `61.49%` | `48.65%` | `23.01%` | `27.95%` | `0.39%` |

*(Notice how SCLK cluster stratification resolved Step 1's single-cutoff drift: `train` and `val` match closely across all four classes.)*

### Train-Only Loss Weighting Schemes (`train_imbalance_ratio_max_to_min = 77.15 : 1`)

| Weighting Method | Soil (`0`) | Bedrock (`1`) | Sand (`2`) | Big Rock (`3`) |
| :--- | ---: | ---: | ---: | ---: |
| **`median_frequency_capped`** *(recommended)* | `0.5452` | `0.5601` | `4.6572` | `12.0000` |
| **`inverse_frequency_capped`** | `0.0456` | `0.0469` | `0.3896` | `3.5179` |
| **`effective_num_samples`** | `0.0476` | `0.0489` | `0.3908` | `3.5127` |
| **`swan2021_one_minus_composition`** | `0.6994` | `0.7164` | `1.2591` | `1.3251` |

---

## 12. PyTorch Dataset Implementation

Implemented in `src/data/dataset.py` (`MangalDrishtiDataset`):
- Verifies filename stem matching (`img_p.stem == msk_p.stem`) and file existence for every record.
- Returns `(image, mask)` where `image` is `torch.float32` of shape `[3, 512, 512]` and `mask` is `torch.int64` of shape `[512, 512]` preserving `255` ignore index.

---

## 13. DataLoader Configuration

Implemented in `src/data/dataloader.py` (`build_dataloaders`):
- `batch_size: 8`, `num_workers: 0` (safe CPU/Windows default, configurable via YAML), `pin_memory` enabled automatically when CUDA is available, deterministic `torch.Generator` (`seed=42`) and `worker_init_fn`.
- `shuffle=True` for `train`; `shuffle=False` for `val` and `test`.
- Measured validation throughput: **`20.7` samples/sec** (`train` with live augmentation), **`52.4` samples/sec** (`val`), **`52.39` samples/sec** (`test`).

---

## 14. Validation Methodology

Implemented in `src/data/validation.py` (`datasets/inspection/preprocessing_validation_report.json`) and `tests/test_step2_preprocessing.py`:
- Verified all `2,296` processed pairs (`1,407` train + `363` val + `526` test): `0` missing, `0` corrupted, `0` dimension mismatches, `0` invalid mask values, `0` empty processed masks, `0` duplicates, `0` orphans, `0` ID/cluster leakage.
- Verified `torch.nn.CrossEntropyLoss(ignore_index=255)` forward + backward pass on real DataLoader batches (`loss = 1.386296` for zero logits).
- `pytest tests/test_step2_preprocessing.py -v`: **`7 passed in 2.83s`**.

---

## 15. Commands to Execute the Pipeline

```powershell
# 1. Run full preprocessing, splitting, normalization stats, and class weights
.\.venv\Scripts\python.exe -m src.data.preprocessing --config configs/data_config.yaml

# 2. Run automated validation suite
.\.venv\Scripts\python.exe -m src.data.validation --config configs/data_config.yaml

# 3. Run unit & integration tests
.\.venv\Scripts\python.exe -m pytest tests/test_step2_preprocessing.py -v

# 4. Generate Step 2 visual validation samples & class charts
.\.venv\Scripts\python.exe -m src.data.visualization --mode preprocessing --config configs/data_config.yaml
```

*(Other standalone entrypoints: `python -m src.data.mask_encoder`, `python -m src.data.split_dataset`, `python -m src.data.class_weights`, `python -m src.data.inspect_dataset`.)*

---

## 16. Known Limitations

1. **`Big Rock` Scarcity:** `Big Rock` represents `0.62%` of valid `train` pixels (`182` images) and `0.39%` of `test` pixels (`19` images). Even with capped median frequency weighting (`12.0`), hybrid Dice/Focal losses will be important in Step 3.
2. **Test Set Domain Difference:** The official JPL `test` set was stratified across orbital dune/ripple units, giving `sand = 27.95%` vs `5.57%` in `train`.
3. **Windows Smart App Control DLL Restriction:** `albumentations` / `torchvision` were not required because `src/data/transforms.py` uses verified OpenCV + NumPy + PyTorch directly, avoiding Windows Application Control unsigned `.pyd` blocks.

---

## 17. Recommendations for Step 3 (Model Training)

1. Use `median_frequency_capped` (or `effective_num_samples`) class weights inside a combined `CrossEntropyLoss(ignore_index=255) + Dice/FocalLoss` objective.
2. Train baseline **U-Net** and **DeepLabV3+** models with ResNet encoders initialized on ImageNet, keeping `configs/data_config.yaml` (`512 × 512`, `seed=42`) fixed for fair model comparison.
3. Track per-class IoU (`soil`, `bedrock`, `sand`, `big rock`) and mean IoU (`mIoU`) ignoring `255` on `val` and `test` (`test_msl` and `test_mer` breakdowns).
