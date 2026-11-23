# Independent Validation Notes — Previously Unseen NASA Mars Imagery

**Important Scientific Notice:** All results below represent **qualitative inference** on newly acquired NASA surface frames with **no expert ground-truth masks**. IoU, Dice, precision, and recall are not computed or claimed, and outputs do not verify physical rover traversability.


## ZL0_2024_0846622583_145EBY_N0930000ZCAM10058_1100LMJ (Mars 2020 (Perseverance) · MCZ_LEFT · Sol 2024)

- **Source URL:** https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/02024/ids/edr/browse/zcam/ZL0_2024_0846622583_145EBY_N0930000ZCAM10058_1100LMJ02_1200.jpg
- **Acquisition UTC:** 2026-10-30T09:21:53.576 | **Original Resolution:** 1200×874 -> Processed 512×512
- **Ground Truth Available:** False (No expert annotation mask)
- **DeepLabV3+ Baseline:** Soil `11.4%`, Bedrock `5.1%`, Sand `38.1%`, Big Rock `45.3%` | Mean Conf `0.598` | Entropy `0.943` | Assessment `higher_visual_concern` | `physical_safety_verified: False`
- **Research Note:** DeepLabV3+ Baseline segmented this unseen Mars 2020 (Perseverance) MCZ_LEFT (Sol 2024) frame primarily as big_rock (45.3%), with secondary terrain mix (Soil 11.4%, Bedrock 5.1%, Sand 38.1%, Big Rock 45.3%). Mean softmax confidence was 0.598 (mean entropy 0.943 nats, 12 2D Big Rock candidates). U-Net Baseline predicted sand as dominant (0 rock candidates), and DeepLabV3+ Optimized predicted sand (mean conf 0.658). Cross-mission optic/scale differences (especially Perseverance NAVCAM_LEFT/MCZ_LEFT FOV vs MSL AI4MARS training priors) are treated strictly qualitatively without ground-truth labels.


## NLF_2023_0846549056_005ECM_N0930000NCAM00709_09_095J (Mars 2020 (Perseverance) · NAVCAM_LEFT · Sol 2023)

- **Source URL:** https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/02023/ids/edr/browse/ncam/NLF_2023_0846549056_005ECM_N0930000NCAM00709_09_095J01_1200.jpg
- **Acquisition UTC:** 2026-10-29T12:56:14.159 | **Original Resolution:** 1200×904 -> Processed 512×512
- **Ground Truth Available:** False (No expert annotation mask)
- **DeepLabV3+ Baseline:** Soil `20.5%`, Bedrock `5.1%`, Sand `8.0%`, Big Rock `66.4%` | Mean Conf `0.674` | Entropy `0.814` | Assessment `higher_visual_concern` | `physical_safety_verified: False`
- **Research Note:** DeepLabV3+ Baseline segmented this unseen Mars 2020 (Perseverance) NAVCAM_LEFT (Sol 2023) frame primarily as big_rock (66.4%), with secondary terrain mix (Soil 20.5%, Bedrock 5.1%, Sand 8.0%, Big Rock 66.4%). Mean softmax confidence was 0.674 (mean entropy 0.814 nats, 4 2D Big Rock candidates). U-Net Baseline predicted sand as dominant (0 rock candidates), and DeepLabV3+ Optimized predicted big_rock (mean conf 0.715). Cross-mission optic/scale differences (especially Perseverance NAVCAM_LEFT/MCZ_LEFT FOV vs MSL AI4MARS training priors) are treated strictly qualitatively without ground-truth labels.


## NLB_848553598EDR_F1260000CCAM01081M_ (MSL (Curiosity) · NAV_LEFT_B · Sol 5081)

- **Source URL:** https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/05081/opgs/edr/ncam/NLB_848553598EDR_F1260000CCAM01081M_-br2.jpg
- **Acquisition UTC:** 2026-11-21T18:33:07.78549 | **Original Resolution:** 1024×1024 -> Processed 512×512
- **Ground Truth Available:** False (No expert annotation mask)
- **DeepLabV3+ Baseline:** Soil `3.1%`, Bedrock `23.6%`, Sand `41.4%`, Big Rock `31.9%` | Mean Conf `0.764` | Entropy `0.615` | Assessment `higher_visual_concern` | `physical_safety_verified: False`
- **Research Note:** DeepLabV3+ Baseline segmented this unseen MSL (Curiosity) NAV_LEFT_B (Sol 5081) frame primarily as sand (41.4%), with secondary terrain mix (Soil 3.1%, Bedrock 23.6%, Sand 41.4%, Big Rock 31.9%). Mean softmax confidence was 0.764 (mean entropy 0.615 nats, 11 2D Big Rock candidates). U-Net Baseline predicted bedrock as dominant (2 rock candidates), and DeepLabV3+ Optimized predicted bedrock (mean conf 0.818). Cross-mission optic/scale differences (especially Perseverance NAVCAM_LEFT/MCZ_LEFT FOV vs MSL AI4MARS training priors) are treated strictly qualitatively without ground-truth labels.


## NRB_848386706EDR_F1260000CCAM05077M_ (MSL (Curiosity) · NAV_RIGHT_B · Sol 5079)

- **Source URL:** https://mars.nasa.gov/msl-raw-images/proj/msl/redops/ods/surface/sol/05079/opgs/edr/ncam/NRB_848386706EDR_F1260000CCAM05077M_-br2.jpg
- **Acquisition UTC:** 2026-11-19T20:11:34.28162 | **Original Resolution:** 1024×1024 -> Processed 512×512
- **Ground Truth Available:** False (No expert annotation mask)
- **DeepLabV3+ Baseline:** Soil `0.5%`, Bedrock `43.5%`, Sand `40.0%`, Big Rock `16.0%` | Mean Conf `0.796` | Entropy `0.555` | Assessment `higher_visual_concern` | `physical_safety_verified: False`
- **Research Note:** DeepLabV3+ Baseline segmented this unseen MSL (Curiosity) NAV_RIGHT_B (Sol 5079) frame primarily as bedrock (43.5%), with secondary terrain mix (Soil 0.5%, Bedrock 43.5%, Sand 40.0%, Big Rock 16.0%). Mean softmax confidence was 0.796 (mean entropy 0.555 nats, 10 2D Big Rock candidates). U-Net Baseline predicted bedrock as dominant (6 rock candidates), and DeepLabV3+ Optimized predicted bedrock (mean conf 0.836). Cross-mission optic/scale differences (especially Perseverance NAVCAM_LEFT/MCZ_LEFT FOV vs MSL AI4MARS training priors) are treated strictly qualitatively without ground-truth labels.
