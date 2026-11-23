# MANGAL DRISHTI — Step 4: Expert Test Evaluation, Benchmarking & Controlled Optimization Report

## 1. Combined Expert-Test Comparison (526 Images, 53,104,279 Valid Pixels)

Both Step 3 baseline checkpoints were evaluated untouched on the full JPL expert consensus test split (`526` images; `ignore_index=255` strictly excluded; no test-set tuning or threshold adjustment):

| Metric | U-Net + ResNet34 | DeepLabV3+ + ResNet50 | Delta (DeepLabV3+ − U-Net) |
| :--- | :---: | :---: | :---: |
| **Pixel Accuracy** | **0.921333** | 0.919592 | -0.001741 |
| **Mean IoU (mIoU)** | 0.681947 | **0.690048** | **+0.008101** |
| **Mean Dice** | 0.764646 | **0.776488** | **+0.011842** |
| **Soil IoU** | **0.894815** | 0.877703 | -0.017112 |
| **Bedrock IoU** | 0.813525 | **0.833118** | **+0.019593** |
| **Sand IoU** | **0.840489** | 0.836328 | -0.004161 |
| **Big Rock IoU** | 0.178958 | **0.213042** | **+0.034084** |
| **Sand Recall** | 0.881449 | **0.944334** | **+0.062885** |
| **Big Rock Precision** | 0.247443 | **0.343057** | **+0.095614** |
| **Big Rock Recall** | **0.392687** | 0.359848 | -0.032839 |

- **Source artifacts:** [combined_comparison.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/baseline/combined_comparison.json), [unet_test_metrics.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/baseline/unet_test_metrics.json), [deeplabv3plus_test_metrics.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/baseline/deeplabv3plus_test_metrics.json).

---

## 2. MSL vs. MER Domain-Shift Results

Because training (`1,407`) and validation (`363`) originate from **MSL (Curiosity)** Navcam imagery, evaluating separately on **MSL (322 images; 32,239,572 valid px)** vs. out-of-distribution **MER (204 images; 20,864,707 valid px — 141 Opportunity + 63 Spirit)** isolates cross-mission generalization:

| Split / Mission | Model | Pixel Acc | mIoU | Mean Dice | Soil IoU | Bedrock IoU | Sand IoU | Big Rock IoU | Big Rock Prec / Rec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MSL (322)** | U-Net + ResNet34 | 0.954742 | 0.714356 | 0.773598 | 0.947651 | 0.863627 | **0.908095** | 0.138051 | 0.139442 / **0.932617** |
| **MSL (322)** | DeepLabV3+ + ResNet50 | **0.957532** | **0.731131** | **0.795163** | **0.948598** | **0.885169** | 0.899790 | **0.190968** | **0.196616** / 0.869239 |
| **MER Combined (204)** | U-Net + ResNet34 | **0.869710** | **0.627629** | 0.735578 | **0.808376** | 0.744133 | 0.758950 | 0.199056 | 0.336157 / **0.327987** |
| **MER Combined (204)** | DeepLabV3+ + ResNet50 | 0.860969 | 0.626482 | **0.739192** | 0.760381 | **0.758742** | **0.764819** | **0.221986** | **0.463362** / 0.298807 |
| MER-1 Opportunity (141) | U-Net + ResNet34 | 0.883485 | 0.651092 | 0.750343 | **0.789004** | **0.843399** | 0.772462 | 0.199501 | 0.364359 / **0.306002** |
| MER-1 Opportunity (141) | DeepLabV3+ + ResNet50 | **0.885756** | **0.657367** | **0.758127** | 0.786060 | 0.806896 | **0.817171** | **0.219343** | **0.517377** / 0.275768 |
| MER-2 Spirit (63) | U-Net + ResNet34 | **0.838248** | **0.483942** | 0.594902 | **0.835528** | 0.176031 | **0.729086** | 0.195121 | 0.197762 / 0.935948 |
| MER-2 Spirit (63) | DeepLabV3+ + ResNet50 | 0.804354 | 0.463428 | **0.598251** | 0.728451 | **0.219765** | 0.659340 | **0.246158** | **0.250376** / 0.935948 |

- **Key Domain-Shift Observations:**
  - DeepLabV3+ outperforms U-Net on **MSL** by **+1.68 mIoU** (`0.7311` vs `0.7144`), with higher Soil, Bedrock, and Big Rock IoU.
  - Moving from **MSL to MER** drops mIoU by **-10.46 pts** for DeepLabV3+ (`0.7311 -> 0.6265`) and **-8.67 pts** for U-Net (`0.7144 -> 0.6276`), driven primarily by **MER-2 (Spirit)** (`mIoU = 0.4634 / 0.4839`), where fractured basaltic rock, distinct camera optics/contrast, and sparse Big Rock (`6,401` px) vs Bedrock (`164,212` px) annotations reduce generalization.
- **Source artifacts:** [msl_comparison.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/baseline/msl_comparison.json), [mer_comparison.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/baseline/mer_comparison.json).

---

## 3. Per-Class Performance (Combined Expert Test)

- **Soil (0)** (`25,836,654` valid test px): High stability in both models (`0.8948` U-Net IoU, `0.8777` DeepLabV3+ IoU; `0.9550` / `0.9596` precision).
- **Bedrock (1)** (`12,217,949` px): DeepLabV3+ leads (`0.8331` vs `0.8135` IoU; `0.9017` vs `0.8488` precision), aided by ASPP multi-scale context over fractured slabs.
- **Sand (2)** (`14,844,281` px): Nearly tied IoU (`0.8405` U-Net vs `0.8363` DeepLabV3+), with DeepLabV3+ achieving notably higher Sand recall (`0.9443` vs `0.8814`) at a trade-off against Soil (`5.2%` of Soil misclassified as Sand vs `1.8%` for U-Net).
- **Big Rock (3)** (`205,395` px — only **0.387%** of valid test pixels): Remains the primary bottleneck (`0.2130` DeepLabV3+ IoU vs `0.1790` U-Net IoU).

---

## 4. Big Rock Precision and Recall Analysis

- On **MSL (322 images)** (`21,979` Big Rock px), both models achieve high Big Rock recall (**93.26%** U-Net, **86.92%** DeepLabV3+) because the training set shares Curiosity Navcam scale and `median_frequency_capped` weight `12.0`, but suffer low precision (**13.94%** U-Net, **19.66%** DeepLabV3+) due to Bedrock/fractured outcrop edges being flagged as Big Rock (`83,935` U-Net / `67,719` DeepLabV3+ Bedrock pixels predicted as Big Rock on MSL).
- On **MER (204 images)** (`183,416` Big Rock px — **89.3%** of all expert-test Big Rock pixels, dominated by MER-1 Opportunity crater walls and large stratified outcrops), recall drops to **32.80%** (U-Net) and **29.88%** (DeepLabV3+), while precision is higher (**46.34%** DeepLabV3+ vs **33.62%** U-Net). Overall combined precision favors DeepLabV3+ (**34.31%** vs **24.74%**).

---

## 5. Confusion Matrix Analysis

Row-normalized confusion matrices ([confusion_matrices_test.png](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/confusion_matrices_test.png)) show:
1. **True Big Rock (`205,395` combined px) errors:** DeepLabV3+ misclassifies **38.54%** (`79,165` px) of Big Rock as **Sand** and **21.35%** (`43,861` px) as **Bedrock** (concentrated in MER-1 crater rim/outcrop images where windblown dark-rock/sand surfaces differ from MSL boulders). U-Net shows the same pattern (**31.17%** to Sand, **26.47%** to Bedrock).
2. **True Bedrock to Big Rock false positives:** `108,107` Bedrock px (`0.88%`) in DeepLabV3+ and `145,071` px (`1.19%`) in U-Net leak into Big Rock — small as a fraction of Bedrock, yet larger than all true Big Rock TP pixels (`73,911` / `80,656`), explaining Big Rock's low precision.
3. **Soil vs. Sand boundary drift:** `1,350,490` Soil px (`5.23%`) are predicted as Sand by DeepLabV3+ (vs `476,660` px / `1.84%` by U-Net), reflecting visual similarity between rippled dark regolith and flat Soil.

---

## 6. Qualitative Error Analysis (6-Panel Visualizations)

Eight representative expert test images (spanning MSL and MER) were rendered with 6-panel diagnostics (**Original Image, Ground-Truth Mask, Predicted Mask, Confidence Map, Overlay, Error Map**) in [selected_cases_manifest.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/selected_cases_manifest.json):
- **Correct Predictions (`1.000` img mIoU):** MSL [correct_prediction_NLA_601414568EDR_F0731482NCAM00253M1.png](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/deeplabv3plus/correct_prediction_NLA_601414568EDR_F0731482NCAM00253M1.png) and MER-1 [correct_prediction_1n129697839eff0338p1931l0m1.png](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/deeplabv3plus/correct_prediction_1n129697839eff0338p1931l0m1.png).
- **Big Rock False Negatives:** On MER-1 crater cliff `1n271377483eff90dcp1935l0m1` ([DeepLabV3+ panel](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/deeplabv3plus/big_rock_false_negative_1n271377483eff90dcp1935l0m1.png)), `75,493` / `83,513` Big Rock pixels are missed as Bedrock/Sand due to OOD cliff geometry unseen in MSL training.
- **Big Rock False Positives:** On MER-1 bright fractured bedrock `1n288519342eff98j5p1994l0m1` ([DeepLabV3+ panel](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/deeplabv3plus/big_rock_false_positive_1n288519342eff98j5p1994l0m1.png)), U-Net triggers `18,187` Big Rock FP pixels vs `9,451` for DeepLabV3+, with low softmax confidence (`~0.4–0.5`) along the bedrock boundary.
- **Sand/Bedrock & Low-Confidence Cases:** Shadowed ripples in MSL `NLB_505900047EDR_F0521162NCAM00352M1` ([U-Net panel](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/unet/sand_bedrock_confusion_NLB_505900047EDR_F0521162NCAM00352M1.png)) and rover deck/shadow transitions in `1n136619282eff2000p1985l0m1` ([DeepLabV3+ panel](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/visualizations/deeplabv3plus/low_confidence_prediction_1n136619282eff2000p1985l0m1.png)) drop valid confidence to `0.642` (DeepLabV3+) and `0.589–0.674` (U-Net).

---

## 7. Hardware & Inference Benchmarking (`512x512` RGB)

Measured on **NVIDIA GeForce RTX 3050 A Laptop GPU (4.0 GB VRAM)** and 8-thread CPU (`PyTorch 2.11.0+cu128`; warm-up + 20–100 timed runs; [inference_benchmark.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/benchmarks/inference_benchmark.json)):

| Model | Params | Ckpt / Weights (MB) | GPU FP16 bs=1 Avg / Med (ms) | GPU FP16 bs=1 FPS | GPU FP16 bs=1 Peak VRAM (MB) | GPU FP16 bs=4 FPS / VRAM (MB) | CPU FP32 bs=1 Avg (ms) / FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net + ResNet34** | **24,521,796** | **280.91 / 93.62** | **22.882 / 22.878** | **43.703** | 327.97 | **58.782** / 881.59 | **447.258 / 2.236** |
| **DeepLabV3+ + ResNet50** | 40,347,812 | 462.24 / 154.13 | 23.968 / 24.208 | 41.722 | **323.47** | 53.011 / **579.64** | 672.681 / 1.487 |

- On GPU FP16 (AMP, `bs=1`), DeepLabV3+ is within **1.09 ms** of U-Net (**41.7 FPS vs 43.7 FPS**) and uses **lower peak batch-4 VRAM** (`579.64 MB` vs `881.59 MB`) due to dilated output stride 16 instead of full-resolution skip concatenations.

---

## 8. Controlled Optimization Experiments & Outcomes (Train/Val Only)

- **Hypothesis:** Baseline models trained with pure weighted CE and Big Rock weight `12.0` over-predicted Big Rock around edges/shadows. Fine-tuning (`4` epochs, `AdamW lr=3e-5`, AMP) from each best baseline checkpoint using **Combined CE + Soft Dice (`dice_weight=0.5`)** and tempered weights (`Soil: 0.65, Bedrock: 0.65, Sand: 3.80, Big Rock: 7.00`), selecting strictly on `val_mIoU`, should boost Big Rock IoU and precision ([optimization_summary.json](file:///d:/Personal%20Projects/MANGAL%20DRISHTI/experiments/evaluation/optimization/optimization_summary.json)).

| Experiment | Val mIoU (Base -> Opt) | Val Big Rock IoU / Prec | Test Combined mIoU (Base -> Opt) | Test Combined Big Rock IoU / Prec / Rec | Test MSL mIoU / Big Rock IoU | Test MER mIoU / Big Rock IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `deeplabv3plus_ce_dice_balanced` (Best Ep. 4) | 0.7659 -> **0.8051** (+3.92 pts) | 0.5123 -> **0.5656** / 0.5488 -> **0.7043** | 0.6900 -> 0.6897 (-0.03 pts) | 0.2130 -> 0.1406 / **0.4490** / 0.1700 | 0.7311 -> **0.7811** (+5.00) / **0.3352** (+14.43) | 0.6265 -> 0.6165 / 0.0913 |
| `unet_ce_dice_balanced` (Best Ep. 2) | 0.7466 -> **0.7627** (+1.61 pts) | 0.4566 -> **0.5128** / 0.5249 -> **0.6382** | 0.6819 -> **0.6932** (+1.13 pts) | 0.1790 -> **0.1999** (+2.10) / **0.4106** / 0.2804 | 0.7144 -> **0.7707** (+5.64) / **0.3150** (+17.69) | 0.6276 -> 0.6156 / 0.1676 |

- **Scientific Interpretation:** Optimizing `CombinedCEDiceLoss` + tempered Big Rock weight (`7.0`) yields **large in-domain MSL gains** (`+5.00 mIoU`, `+14.43 Big Rock IoU` on DeepLabV3+; `+5.64 mIoU`, `+17.69 Big Rock IoU` on U-Net, plus `0.9344` / `0.9268` combined pixel accuracy), and raises `unet_ce_dice_balanced` combined test mIoU to **0.6932**. However, tightening MSL rock precision reduces zero-shot MER Big Rock recall (`0.2988 -> 0.0986` on DeepLabV3+), lowering DeepLabV3+ combined Big Rock IoU (`0.2130 -> 0.1406`).

---

## 9. Recommended Final Model

- **For Cross-Mission Robustness & Hazard Recall (Default Baseline):** **DeepLabV3+ + ResNet50 (Step 3 Baseline, `experiments/deeplabv3plus/best_model.pth`)** — retains highest combined Big Rock IoU (`0.2130`) and highest baseline combined Mean Dice (`0.7765`) at **41.7 FPS** (`bs=1` GPU FP16).
- **For MSL (Curiosity) In-Domain Precision & General Terrain Coverage:** **Optimized DeepLabV3+ (`deeplabv3plus_ce_dice_balanced`)** (`0.8051` val mIoU, `0.7811` MSL test mIoU, `0.9344` combined pixel accuracy, `0.8780` combined Sand IoU), or **`unet_ce_dice_balanced`** when maximum combined test mIoU (`0.6932`) with lower CPU latency (`447.3 ms`) is prioritized.

---

## 10. Remaining Limitations

1. **MSL-to-MER Distribution Shift:** Models trained exclusively on MSL experience a `~8.7–10.5` pt mIoU drop on Opportunity and severe Bedrock degradation on Spirit (`0.2198` Bedrock IoU). Adding labeled MER imagery or domain-adaptation augmentations is necessary for cross-mission deployment.
2. **Extreme Big Rock Sparsity:** Big Rock covers only `0.387%` of valid test pixels and skews heavily toward MER-1 cliff/outcrop geometry, limiting Big Rock IoU despite high Soil/Bedrock/Sand accuracy.
3. **Not a Standalone Traversability Certification:** Semantic terrain classes (`Soil`, `Bedrock`, `Sand`, `Big Rock`) provide 2D visual surface context only. Autonomous rover path planning and safety certification require 3D stereo elevation/slope geometry, wheel-soil slip dynamics, vehicle kinematics, and onboard hazard avoidance sensors.
