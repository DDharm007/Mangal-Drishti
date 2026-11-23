# MANGAL DRISHTI — Step 01 Dataset Research & Intelligence Report

## 1. Project Overview

**MANGAL DRISHTI (मंगल दृष्टि)** — *Mars Autonomous Navigation, Geospatial Analysis & Learning — Deep Rover Intelligence, Segmentation, Terrain Recognition & Imaging* — is an end-to-end planetary computer vision engineering project aimed at pixel-level semantic understanding of Martian surface imagery captured by NASA Mars rovers.

Step 01 establishes the verified data, research, validation, and documentation foundation upon which later modeling and perception subsystems will be built. Specifically, the future system targets pixel-level segmentation of rover imagery into operationally meaningful surface terrain classes (`soil`, `bedrock`, `sand`, `big rock`), uncertainty/confidence inspection, class-coverage estimation, and researcher-facing analysis interfaces.

> **Important Scientific Limitation:** Visual semantic segmentation of monocular or stereo rover imagery **does not independently establish physical rover traversability or drive safety**. Certified rover navigation requires 3D stereo terrain geometry, local slope and roughness estimation, articulated suspension kinematics, wheel-terrain terramechanical interaction (slip, sinkage), and vehicle dynamics telemetry. In MANGAL DRISHTI, visual terrain classification provides semantic context rather than autonomous driving certification.

---

## 2. Dataset Overview

**AI4MARS** (*AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*) is a large-scale planetary semantic segmentation benchmark developed by NASA's Jet Propulsion Laboratory (JPL), California Institute of Technology.

Prior Mars autonomy systems (such as NASA's classical **AutoNav** on Spirit, Opportunity, Curiosity, and Perseverance) rely predominantly on stereo geometry for obstacle detection while lacking semantic awareness of surface composition—even though terrain type strongly governs wheel slip (e.g., Spirit's sand-trap immobilization), drive energy consumption, visual odometry drift, and aluminum wheel puncture risk (Curiosity driving over sharp rocks embedded in hard surfaces). Earlier ML efforts such as **SPOC** (*Soil Property and Object Classification*, Rothrock et al., 2016) were bottlenecked by small in-house expert-labeled datasets (only a few thousand labels), while ESA's **LabelMars** used ~20 fine-grained geological classes requiring specialized domain expertise.

AI4MARS solves this scaling bottleneck by defining a concise, intuitive **four-class navigation taxonomy** (`soil`, `bedrock`, `sand`, `big rock`) that non-expert citizen scientists can annotate accurately after a short web tutorial via the **Zooniverse** platform, paired with a smaller, high-confidence **"gold-standard" test set** annotated by NASA/JPL Mars rover drivers and project scientists.

---

## 3. Official Dataset Sources

| Resource | Official Location / Identifier | Verification Status |
| :--- | :--- | :--- |
| **NASA Open Data Catalog** | `https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars` (short link `https://data.nasa.gov/d/cykx-2qix` cited in Swan et al., 2021) | Catalog entry verified via web index; links to Zenodo official repository |
| **Official Data Archive (Zenodo)** | `https://zenodo.org/records/15995036` (DOI: `10.5281/zenodo.15995036`) | Primary authoritative download source (`ai4mars-dataset-merged-0.6.zip`, `16.23 GB`, `CC-BY-4.0`) |
| **Original CVPRW 2021 Paper** | Swan et al., *"AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars"*, IEEE/CVF CVPR Workshops (AI4Space), June 2021, pp. 1982–1991 (`https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html`) | Full 10-page PDF inspected directly |
| **Follow-up Multi-Mission Paper** | Atha et al., *"Multi-mission Terrain Classifier for Safe Rover Navigation and Automated Science"*, IEEE Aerospace Conference, March 2022 (DOI: `10.1109/AERO53065.2022.9843615`) | Linked via Zenodo metadata `related_identifiers` |
| **Zooniverse Citizen Science Project** | `https://www.zooniverse.org/projects/hiro-ono/ai4mars` | Referenced in official `info.md` and Swan et al. (2021) |
| **GitHub URL Check** | `https://github.com/AI4MARS/AI4MARS` | Returned HTTP `404 Not Found` at time of Step 01 verification (Nov 2026); official binaries and documentation are preserved in Zenodo Record `15995036` |

---

## 4. Dataset Release & Version History

The CVPRW 2021 paper describes the **initial November 2020 release** (~35K images and ~326K unmerged crowdsourced labels from MSL, Opportunity, and Spirit). The authoritative Zenodo record (`10.5281/zenodo.15995036`, published October 22, 2022) hosts **Version 0.6** (`ai4mars-dataset-merged-0.6.zip`, `16,232,481,989` compressed bytes / `16,457,250,829` uncompressed bytes across `146,140` file entries, plus `ai4mars-labels-unmerged.zip` at `1.57 GB`).

Per the official `changelog.md` packaged inside the v0.6 archive:

| Version | Key Additions & Known State |
| :--- | :--- |
| **v0.1** | Initial Curiosity (`MSL`) NavCam data: `EDR` images, `MXY` rover masks, `RNG-30M` range masks, and masked+merged+cleaned labels. |
| **v0.2** | Added `MER` (Spirit & Opportunity) image products, unmasked+merged labels, and raw unmerged labels. Range/rover masks not yet provided for MER. |
| **v0.3** | `MER` fixes: added missing images, removed training labels overlapping with MER test images, added cleaned unmasked merged training labels (empty merges removed). |
| **v0.4** | Added `MSL` color Mastcam (`ML`) images and training labels. No expert test set and no rover/range masks exist for Mastcam. |
| **v0.5** | Added `M2020` (Perseverance) NavCam (`ncam`, `HAFIQ/Vgnc`) and Mastcam-Z (`mcam`, `ZL0`) imagery plus both `NAV` and `M2020_GEO` beta labels; added `label_keys.json`. Official cleaning ranking documented as: **`MSL NavCam > MER > MSL MastCam > M2020 NAV > M2020 Geology`**. |
| **v0.6** | **Current Official Release:** Fixed merged MER training labels. |

### Complete Official v0.6 Archive Breakdown (Measured from Zip64 Central Directory)

| Archive Subdirectory | File Count | Uncompressed Size | Purpose |
| :--- | ---: | ---: | :--- |
| `msl/ncam/images/edr` | 18,127 | 6,010.57 MB | Curiosity NavCam grayscale images |
| `msl/ncam/images/mxy` | 18,097 | 59.41 MB | Curiosity rover hardware masks |
| `msl/ncam/images/rng-30m` | 18,096 | 23.80 MB | Curiosity >30m stereo distance masks |
| `msl/ncam/labels/train` | 16,064 | 102.70 MB | Curiosity NavCam merged train masks |
| `msl/ncam/labels/test/masked-gold-min{1,2,3}-100agree` | 322 × 3 | 7.29 MB | Curiosity expert gold-standard test masks |
| `msl/mcam/images` & `labels/train` | 9,099 + 9,099 | 1,804.12 MB | Curiosity color Mastcam images & labels |
| `mer/images/eff` & `labels/train/merged-unmasked` | 16,300 + 8,507 | 4,579.69 MB | Opportunity (`1*`) & Spirit (`2*`) NavCam train |
| `mer/images/test` & `labels/test/*` | 204 + 1,171 | 66.31 MB | MER gold-standard test images, `min1/2/3`, `raw_unmerged` |
| `m2020/images/{HAFIQ,mcam,ncam}` | 18,064 | 3,675.89 MB | Perseverance NavCam & Mastcam-Z images |
| `m2020/labels/{NAV,M2020_GEO/*}` | 12,335 | 127.46 MB | Perseverance NAV & 19-class GEO beta labels |
| Root & directory docs (`*.md`, `*.txt`, `*.json`, `*.csv`) | 11 | 0.02 MB | Official metadata and label keys |
| **Total (`ai4mars-dataset-merged-0.6.zip`)** | **146,140** | **16,457.25 MB** | Complete official v0.6 dataset |

---

## 5. Rover Missions Represented

Four NASA Mars surface rovers are represented across AI4MARS v0.6, while our **Step 01 primary curated dataset (`2,326` pairs)** spans the three missions that possess verified **official expert gold-standard test sets** and the highest official data-cleaning rank (`MSL NavCam` and `MER`):

1. **Mars Science Laboratory (MSL) — *Curiosity*** (landed 2012, Gale Crater): Represented by `2,122` NavCam pairs in our Step 01 dataset (`1,447` NavCam Left-A `NLA_*`, `675` NavCam Left-B `NLB_*`).
2. **Mars Exploration Rover-1 (MER-1) — *Opportunity*** (landed 2004, Meridiani Planum): Filename prefix `1*` (e.g., `1n129697839eff...`); `141` pairs in our Step 01 official MER gold test set.
3. **Mars Exploration Rover-2 (MER-2) — *Spirit*** (landed 2004, Gusev Crater): Filename prefix `2*` (e.g., `2n130887021eff...`); `63` pairs in our Step 01 official MER gold test set.
4. **Mars 2020 (M2020) — *Perseverance*** (landed 2021, Jezero Crater): Added in v0.5 (`NLF`, `Vgnc`, `ZL0`), excluded from our primary Step 01 train/test benchmark because v0.6 provides no expert test set and `changelog.md` ranks M2020 lowest in cleaning quality. Instead, live Perseverance imagery from NASA Raw APIs is isolated under `datasets/samples/external_nasa/` for future out-of-domain qualitative evaluation.

*(Note: Microscopic MAHLI, telescopic ChemCam RMI, and body-mounted fish-eye HazCam images were deliberately excluded from AI4MARS by JPL because higher-mounted mast NavCam/Mastcam views provide superior angular resolution for terrain traversability perception.)*

---

## 6. Image Characteristics

### AI4MARS Acquired Primary Dataset (`2,326` Images)
- **File Format:** JPEG (`.JPG`, uppercase extension preserved from PDS naming).
- **Resolution:** Strictly uniform **`1024 × 1024`** pixels across all `2,326` images (`2,122` MSL `EDR` + `204` MER `EFF`).
- **Color Space / Channels:** Single-channel 8-bit grayscale (`mode="L"`), `0–255` pixel intensities.
- **Measured Radiometric Intensity Statistics (across `2,326` images):** Global min `0`, global max `255`, mean of per-image means **`62.155`**, mean of per-image standard deviations **`22.634`**—confirming Martian NavCam frames are relatively dark and low-contrast on average.
- **Auxiliary Exclusion Products (MSL NavCam, `2,122` each):**
  - **Rover Masks (`MXY`, `.png`):** `1024 × 1024` binary masks with unique values `{0, 1}` masking the rover body, deck, mast, or robotic arm (`1` = masked rover hardware).
  - **Stereo Range Masks (`RNG-30M`, `.png`):** `1024 × 1024` binary masks with unique values `{0, 1}` derived from stereo range products masking pixels beyond `30 m` (`1` = distance > 30 m or sky). Note from Swan et al. (2021) that stereo range errors over `1–30 m` scale linearly (`0.0005–0.4 m` for MSL NavCam; `0.001–0.5 m` for MER NavCam).

### Additional NASA External Sources Researched (Qualitative Testing Only)
We verified official NASA public endpoints and acquired `8` unlabeled qualitative samples (`datasets/samples/external_nasa/`, `0.66 MB`):
- **Perseverance Raw Images API** (`https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020,ingenuity&feedtype=json`): Active (HTTP `200`). Sampled `4` Sol 2024 Mastcam-Z (`MCZ_LEFT` / `MCZ_RIGHT`) `large` browse JPEGs—measured at **`1200 × 874` RGB (3-channel color)**, highlighting non-square aspect ratio and Bayer color differences relative to `1024 × 1024` grayscale NavCam training inputs.
- **Curiosity Raw Images API** (`https://mars.nasa.gov/api/v1/raw_image_items/?order=sol+desc&condition_1=msl:mission`): Active (HTTP `200`). Sampled `4` Sol 5082 `NAV_LEFT_B` `TRAV` JPEGs—measured at **`256 × 256` grayscale (`L`)**, reflecting downsampled traverse browse frames compared to full `1024 × 1024` PDS `EDR` frames.
- **NASA Images API** (`https://images-api.nasa.gov/search`): Active (HTTP `200`), suitable for public mission browse assets.
- **Legacy `api.nasa.gov/mars-photos` endpoint:** Returned **HTTP `404`** at verification time; direct `mars.nasa.gov` mission APIs are the reliable live source.

---

## 7. Terrain Categories

Verified directly from `datasets/raw/ai4mars/label_keys.json` (`"NAV"` key) and Swan et al. (2021):

| Pixel ID | Official Class Name | Operational Definition & Traversability Context |
| :---: | :--- | :--- |
| **`0`** | `soil` | Fine-grained, smooth, packable regolith and dirt; generally safe/moderate traction surface. |
| **`1`** | `bedrock` | Solid, continuous exposed rock formations; provides stable driving paths and primary geological science targets (note: sharp fractured edges can still cause wheel wear). |
| **`2`** | `sand` | Loose, granular sand sheets, ripples, and dunes; major slip and rover entrapment hazard (e.g., Spirit embedded in sand trap; Curiosity nearly embedded at Hidden Valley). |
| **`3`** | `big rock` | Isolated rocks/boulders protruding above the surface **>30 cm in height and ≥50 cm in width**; high-clearance/puncture/entrapment obstacle hazards. |
| **`255`** | `null` (unlabeled/ignored) | Regions with insufficient annotator agreement, unannotated background, sky/distant terrain **`> 30 m`**, or rover hardware masked by `MXY`. |

*(Note: AI4MARS v0.6 also defines a beta 19-class `M2020_GEO` taxonomy covering subtypes of `bedrock`, `float rock(s)`, `sand`, `pebbles`, `vein`, and `hill/peak`; Step 01 targets the primary `NAV` 4-class navigation taxonomy.)*

---

## 8. Annotation Methodology

1. **Pre-labeling Preprocessing (Zooniverse):**
   - Regions farther than `30 m` (determined from stereo range products) were darkened out so citizen scientists focused only on near-field traversable terrain (`≤ 30 m`).
   - A white trapezoid scale bar was overlaid at image center representing **`50 cm` width on Curiosity** and **`20 cm` width on Spirit/Opportunity** to calibrate labelers on the `Big Rock` threshold.
2. **Crowdsourced Training Annotations:**
   - Volunteer citizen scientists completed an interactive tutorial explaining the four terrain classes and non-overlapping polygon drawing.
   - Each image was annotated independently by multiple volunteers (~10 labelers per image; `93%` of initial release images had ≥9 labels).
   - **Cleaning & Acceptance:** Post-processing masked out pixels overlapping the rover (`MXY`) and pixels beyond `30 m` (`RNG-30M`). Pairwise mean Jaccard Index (mIoU) agreement scores were computed across each image's labels; the bottom `20%` lowest-agreement labels underwent manual review (`1–5` scale), discarding rating-`1` labels.
   - **Label Merging:** Training pixels were accepted only if at least **`3` labelers** labeled the pixel and the modal class achieved **`> 65%` (`2/3`) pixel-wise agreement** among non-blank votes; otherwise the pixel became `255` (`NULL`).
3. **Expert Gold-Standard Test Annotations:**
   - Curated to ~1% of the initial dataset scale by sampling evenly across **7 orbital geological terrain units** mapped by MSL scientists (`Smooth` 38, `Ridged` 45, `Pitted` 57, `Highly Dissected` 57, `Fractured` 41, `Sandy Pits and Ripple Fields` 46, `Sand Dunes` 39) after pruning dusk/low-quality, duplicate, and rover-occluded images.
   - Annotated independently by **3 NASA/JPL rover drivers and scientists** per orbital class (`min1`, `min2`, and `min3` variants all enforce **`100%` agreement** among at least 1, 2, or all 3 experts respectively).

---

## 9. Mask Encoding

- **Format:** Single-channel 8-bit lossless grayscale `.png` (`mode="L"`, `dtype=uint8`), `1024 × 1024`.
- **Verified Unique Pixel Values (Observed across all `2,326` primary masks):** **`[0, 1, 2, 3, 255]`** — zero unexpected or corrupted pixel values.
- **Viewer Appearance Caveat:** Because `0`, `1`, `2`, and `3` sit at the bottom 1.5% of the `0–255` luminance range while `255` is pure white, raw masks appear nearly solid black-and-white in standard image viewers. Visualization and modeling pipelines must decode raw integers without interpolation (`PIL.Image.open` / `cv2.IMREAD_UNCHANGED`) and map `0–3` via an explicit lookup palette while masking `255` as ignored.

---

## 10. Official Dataset Splits & MANGAL DRISHTI Partitioning

Official AI4MARS v0.6 enforces strict separation between crowdsourced training splits and JPL expert test sets:
- **MSL Train (`msl/ncam/labels/train`):** `16,064` crowdsourced merged masks.
- **MSL Expert Gold Test (`msl/ncam/labels/test/masked-gold-min{1,2,3}-100agree`):** `322` expert-labeled images.
- **MER Train (`mer/labels/train/merged-unmasked`):** `8,507` unmasked crowdsourced masks.
- **MER Expert Gold Test (`mer/labels/test/masked-gold-min{1,2,3}-100agree`):** `204` expert-labeled images (`141` Opportunity + `63` Spirit).
- **Mastcam & M2020:** Training labels only; **no expert test sets exist** in v0.6.

### MANGAL DRISHTI Step 01 Split Strategy (`2,326` Verified Pairs)
To avoid temporal/scene leakage between consecutive rover frames captured along the same drive/sol, the `1,800` acquired MSL NavCam training pairs were sorted strictly by **Spacecraft Clock (`SCLK`)** timestamp and partitioned chronologically, while preserving **100% of the official MSL and MER gold-standard expert test sets** as held-out evaluation benchmarks:

| Split | Mission(s) | Pairs | SCLK Range | Label Source |
| :--- | :--- | ---: | :---: | :--- |
| **`train`** | MSL Curiosity (`NLA` / `NLB`) | **1,500** | `397,586,934 – 597,779,450` | Crowdsourced merged (`≥3` labelers, `≥2/3` agree) |
| **`val`** | MSL Curiosity (`NLA`) | **300** | `597,857,359 – 605,153,623` | Crowdsourced merged (chronologically later sols) |
| **`test_msl`** | MSL Curiosity (`NLA` / `NLB`) | **322** | `409,036,068 – 613,584,326` | Official JPL Expert Gold Test (`min3-100agree`, plus `min1/min2`) |
| **`test_mer`** | MER-1 Opportunity (`141`) + MER-2 Spirit (`63`) | **204** | `129,697,839 – 538,600,066` | Official JPL Expert Gold Test (`min3-100agree`, plus `min1/min2`) |
| **Total** | **3 Mars Rovers** | **2,326** | — | **Zero train/test ID overlap (`0` leaked IDs)** |

---

## 11. Dataset Acquisition Procedure

All AI4MARS data was acquired directly from the official Zenodo v0.6 archive (`https://zenodo.org/api/records/15995036/files/ai4mars-dataset-merged-0.6.zip/content`) using `src/data/acquire_dataset.py`:
1. Fetches and parses the archive's Zip64 Central Directory (`146,140` entries across `16.23 GB`).
2. Batches target entries (`docs`, `1,800` contiguous MSL train/val pairs, all `322` MSL expert test pairs, and all `204` MER expert test pairs) into HTTP `Range` requests with automatic Zenodo `X-RateLimit-Remaining` / `Retry-After` handling.
3. Decompresses each entry in memory and verifies exact **`CRC32` checksums** against the official Zip64 Central Directory before writing to `datasets/raw/ai4mars/` with original filenames and official directory hierarchy intact.
4. Generates `split_manifest.json`, copies quick-inspection samples to `datasets/samples/`, and downloads `8` unlabeled qualitative samples from `mars.nasa.gov` APIs to `datasets/samples/external_nasa/`.

---

## 12. Actual Downloaded Statistics & Validation Results

Measured directly by `src/data/inspect_dataset.py` (`datasets/inspection/dataset_summary.json` & `validation_report.json`):

| Metric | Verified Measured Value |
| :--- | :--- |
| **Total Files on Disk (`datasets/raw/ai4mars/`)** | `9,961` files |
| **Total Storage Size on Disk** | `779,498,488` bytes (**`743.39 MB`** / `0.726 GB`) |
| **Total Raw Rover Images** | **`2,326`** JPEG (`2,122` MSL `EDR` + `204` MER `test`) |
| **Total Primary Segmentation Masks** | **`2,326`** PNG (`1,800` MSL train/val + `322` MSL `min3` + `204` MER `min3`) |
| **Total Label Mask Files (incl. `min1` & `min2` test variants)** | **`3,378`** PNG |
| **Total Auxiliary Masks (`MXY` + `RNG-30M`)** | **`4,244`** PNG (`2,122` `MXY` + `2,122` `RNG-30M`) |
| **External Qualitative NASA Samples** | **`8`** JPEG (`0.6625 MB`, strictly isolated in `datasets/samples/external_nasa/`) |
| **Corrupted Images / Corrupted Masks** | **`0`** / **`0`** |
| **Image-Mask Dimension Mismatches** | **`0`** (all `1024 × 1024`) |
| **Invalid Pixel IDs Outside `{0, 1, 2, 3, 255}`** | **`0`** |
| **Unmatched Images or Missing Manifest Pairs** | **`0`** |
| **Exact Duplicate Raw Rover Images (SHA-256)** | **`0`** |
| **Train/Test ID Leakage** | **`0`** |
| **Empty All-`255` Masks** | **`130`** (`29` in `train`, `1` in `val`, `35` in `test_msl` min3, `65` in `test_mer` min3) |

---

## 13. Class Distribution (Measured Pixel & Image Statistics)

Across all **`2,326` primary pairs** (`2,438,987,776` total pixels, using `min3-100agree` for expert test splits):
- **Valid Labeled Pixels:** `1,158,986,532` (**`47.52%`** of total pixels)
- **Unlabeled / Ignored Pixels (`255`):** `1,280,001,244` (**`52.48%`** of total pixels)
- **Overall Max-to-Min Labeled Class Imbalance Ratio (`Soil` vs. `Big Rock`):** **`70.65 : 1`**

| Class ID & Name | Total Pixels | Share of All Pixels (%) | Share of Valid Labeled Pixels (%) | Images Containing Class (`n=2,326`) |
| :--- | ---: | ---: | ---: | ---: |
| **`0`: Soil** | `548,583,531` | `22.49%` | **`47.33%`** | `1,320` (`56.75%`) |
| **`1`: Bedrock** | `500,362,927` | `20.52%` | **`43.17%`** | `1,245` (`53.53%`) |
| **`2`: Sand** | `102,275,761` | `4.19%` | **`8.82%`** | `448` (`19.26%`) |
| **`3`: Big Rock** | `7,764,313` | `0.32%` | **`0.67%`** | `254` (`10.92%`) |
| **`255`: NULL / Ignored** | `1,280,001,244` | `52.48%` | *Excluded* | `2,326` (`100.00%`) |

### Split-by-Split Labeled Distribution (% of Valid Labeled Pixels)

| Split (`n`) | Labeled Coverage (%) | Soil (`0`) | Bedrock (`1`) | Sand (`2`) | Big Rock (`3`) | Imbalance (Max:Min) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`train` (`1,500`)** | `51.27%` | `40.52%` | `52.36%` | `6.26%` | `0.86%` | `60.70 : 1` |
| **`val` (`300`)** | `57.17%` | `76.96%` | `20.41%` | `2.56%` | `0.08%` | `958.29 : 1` |
| **`test_msl` min3 (`322`)** | `34.65%` | `51.35%` | `22.39%` | `26.19%` | `0.07%` | `773.11 : 1` |
| **`test_mer` min3 (`204`)** | `26.04%` | `41.84%` | `27.35%` | `29.75%` | `1.05%` | `39.78 : 1` |

*(Expert Consensus Threshold Effect: On `test_msl`, valid labeled coverage drops from **`57.54%`** (`min1`) to **`48.40%`** (`min2`) to **`34.65%`** (`min3`); on `test_mer`, it drops from **`60.20%`** (`min1`) to **`47.84%`** (`min2`) to **`26.04%`** (`min3`), with `Big Rock` dropping most sharply as agreement strictness rises.)*

---

## 14. Known Dataset Limitations

1. **Severe Class Imbalance (`Big Rock` Scarcity):** `Big Rock` accounts for only **`0.67%`** of valid labeled pixels overall (`0.07%` in `test_msl` min3 across just `5` images; `0.197%` across `53` images even in `min1`). Standard unweighted pixel accuracy will hide failures on `big rock` obstacles.
2. **Sparsity & Empty Merged Masks (`130` All-`255` Masks):** Merging by consensus (`≥2/3` in train, `100%` in expert test) intentionally leaves `52.48%` of all pixels unlabeled (`255`) and leaves `130` masks completely empty (`30` in train/val, `35` in `test_msl` min3, `65` in `test_mer` min3).
3. **Inter-Annotator Boundary & Scale Ambiguity:** Expert agreement analysis (Swan et al., 2021, Fig. 6) and our `min1` → `min3` coverage drop (`16.46%` → `1.55%` image presence for MSL `Big Rock`) confirm that `Big Rock` (`>30 cm` height) and `Bedrock` boundaries carry high monocular ambiguity without stereo height context.
4. **Chronological Terrain Distribution Shift:** Partitioning MSL strictly chronologically reveals clear geological drift between Curiosity's earlier sols (`train`: `52.36%` bedrock vs `40.52%` soil) and later sols (`val`: `76.96%` soil vs `20.41%` bedrock), while the JPL gold test set contains `26.19%` sand because JPL stratified across orbital dune/ripple units.
5. **Official Archive Caveats (`TODO.md` & `changelog.md`):** MER training labels remain unmasked (no `RNG-30M` or `MRL` rover masks packaged yet); M2020 has duplicate `.jpg`/`.jpeg` files, unmerged `HAFIQ` `-1..-4` naming artifacts, and no expert test sets; Mastcam lacks range/rover masks.
6. **Physical Traversability Boundary:** Visual segmentation alone cannot measure ground slope, rock height, soil cohesion, or wheel-terrain slip.

---

## 15. Licensing & Usage Considerations

- **License:** **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** per Zenodo Record `15995036`.
- **Required Academic Citations:**
  - Swan, R. M., Atha, D., Leopold, H. A., Gildner, M., Oij, S., Chiu, C., & Ono, M. (2021). *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*. IEEE/CVF CVPR Workshops (pp. 1982–1991).
  - Atha, D., Swan, R. M., Didier, A., Hasnain, Z., & Ono, M. (2022). *Multi-mission Terrain Classifier for Safe Rover Navigation and Automated Science*. IEEE Aerospace Conference (`10.1109/AERO53065.2022.9843615`).
- **Repository Governance:** Original datasets (`datasets/raw/`) and archives (`*.zip`) must remain excluded from Git via `.gitignore`, while preserving original filenames and untouched raw ground-truth labels.

---

## 16. Recommendations for Step 2 (Modeling & Evaluation Strategy)

1. **Loss Masking & Empty-Mask Filtering:** Always configure `ignore_index=255` in segmentation losses (`CrossEntropyLoss`, `FocalLoss`, `DiceLoss`) and evaluation mIoU metrics, and filter out the `30` empty all-`255` masks from training/validation dataloaders.
2. **Mitigate Severe `Big Rock` & `Sand` Imbalance:** Apply inverse/smoothed class weights (`1 - composition` baseline from Swan et al., or effective-number weighting) combined with **Dice / Tversky / Focal loss**, and consider targeted oversampling or copy-paste augmentation of `big rock` frames during training.
3. **Mitigate Chronological Terrain Drift:** When expanding beyond the initial `1,800` MSL train/val slice in Step 2, group frames by **Sol / Drive sequence (`SCLK` clusters)** and stratify across Sol clusters rather than using a single late-mission cutoff—so `val` matches the multi-terrain diversity of `train` and `test_msl` without leaking adjacent sequential frames.
4. **Multi-Threshold Expert Evaluation:** Evaluate Step 2 models on **all three** expert consensus levels (`min1`, `min2`, and `min3`) for both `test_msl` and `test_mer`—reporting per-class IoU, F1, and confusion matrices (especially `Big Rock` recall/IoU, where `min3` only has `5` MSL images).
5. **Contrast Normalization & Cross-Rover Generalization:** Given dark grayscale NavCam statistics (`mean=62.16`, `std=22.63`), test photometric standardization / CLAHE + gamma jittering to bridge domain shift across `MSL` (`NLA/NLB`), `MER` (`Opportunity/Spirit`), and external `Perseverance` RGB/NavCam imagery.
6. **Leverage Auxiliary Masks at Inference/Training:** Apply `MXY` rover-body masks and `RNG-30M` range masks (where stereo range is available) during evaluation and post-processing so rover deck hardware and `>30 m` horizon sky are properly excluded from terrain statistics.
