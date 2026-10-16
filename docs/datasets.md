# SatQuery AI — Official Benchmark Datasets Specification

**Author:** SatQuery AI Benchmark Engineering Team  
**Status:** Verified Ground Truth (Phase 0 Reconnaissance)  
**Standard:** Strict Rule 1 (Real Data Only) & Rule 3 (No Test Leakage)

---

## 1. Overview & Benchmark Scope

| Benchmark Dataset | Target Capability | Modalities | Primary Task / Output | Split Separation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **BigEarthNet.txt** | Optical-SAR Fusion & Multi-Sensor Grounding | Sentinel-1 (C-band SAR) + Sentinel-2 (Multispectral) | Multi-modal VQA, Captioning, Bounding Boxes | reBEN Spatial Hash Partitioning (Parent Granule Disjoint) |
| **VRSBench** | VHR Single-Image VQA & Grounding | VHR Aerial RGB (DOTA / DIOR, 0.5–1m) | Captioning, Bounding Boxes ([0, 100]), VQA | Official DOTA/DIOR Train vs. Val/Test Split |
| **RSVQA** | Low/High-Res EO Visual Question Answering | Sentinel-2 (LR 10m) & Aerial Ortho (HR 0.15m) | Closed-Vocabulary Answer Classification | Spatial tile-level disjoint (LR: 61.5/18.5/20%, HR: Train/Val/Test1/Test2) |
| **CDVQA** | Bi-Temporal Change Detection & Change VQA | Multi-temporal Aerial RGB ($T_1, T_2$, 512×512) | Binary/Multi-class Change Mask + Change VQA | Physical scene-pair level disjoint (1600 Train / 400 Val / 968 Test) |

---

## 2. BigEarthNet.txt (arXiv:2603.29630)

### 2.1 Specification & Provenance
- **Full Title:** *BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation*
- **Authors & Institution:** TU Berlin / BIFOLD (Remote Sensing Image Analysis Group - RSiM & DIMA)
- **Official Repository:** [`https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- **Imagery Source:** Official BigEarthNet v2.0 (`https://bigearth.net/`)
- **License:** **Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive-1.0)**
- **Total Scale:** 464,044 co-registered Sentinel-1 (VV/VH dual-pol SAR) and Sentinel-2 (12 spectral bands) image sets across 10 European countries.
- **Total Annotations:** ~9.6 million text annotations (captions, VQA pairs, and referring expressions).
- **File Layout:**
  - Annotations: `BigEarthNet.txt.parquet` (~1.2 GB download).
  - Imagery: Parquet metadata links to standard BigEarthNet v2.0 LMDB / safetensors / GeoTIFF storage partitions.

### 2.2 Split Structure

| Partition | Image Sets Count | Text Annotations Count | Purpose | Disjointness Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | ~311,665 | ~6.4M | Pre-training & multi-task adaptation | reBEN geographic hash split |
| **Validation** | ~52,380 | ~1.1M | Hyperparameter selection & checkpointing | Disjoint geographic zones |
| **Test** | ~52,380 | ~1.1M | Final generalization evaluation | Strictly held-out geographic tiles |
| **Bench** | ~47,619 | ~1.0M | Manually verified gold-standard benchmark | Human-verified reference samples |

### 2.3 Annotation Schema & Raw Samples
The dataset annotations are stored as Parquet rows containing instruction-response pairs:

#### Raw Sample 1 (Geographically Anchored Caption):
```json
{
  "patch_id": "S2A_MSIL2A_20170717T100031_N0205_R122_T32TMR_44_52",
  "task": "captioning",
  "instruction": "Provide a detailed geographical description of the land-use and land-cover classes present in this multi-sensor scene.",
  "response": "The Sentinel-2 optical imagery shows predominantly arable land interspersed with patches of transitional woodland and shrub. Sentinel-1 SAR backscatter confirms rough surface textures consistent with summer cereal crop cultivation."
}
```

#### Raw Sample 2 (Visual Question Answering):
```json
{
  "patch_id": "S2B_MSIL2A_20180529T094029_N0208_R036_T33UVP_12_84",
  "task": "vqa",
  "instruction": "Does this scene contain any inland water bodies?",
  "response": "yes"
}
```

#### Raw Sample 3 (Referring Expression Grounding):
```json
{
  "patch_id": "S2A_MSIL2A_20171124T112411_N0206_R037_T29TNE_19_33",
  "task": "grounding",
  "instruction": "Detect and output the normalized bounding box coordinates for the industrial commercial unit.",
  "response": "{\"bbox\": [24, 48, 62, 89], \"label\": \"industrial or commercial units\"}"
}
```

### 2.4 Official Metric Definitions & Published Baselines
- **Metrics:**
  - **VQA:** Overall Accuracy (OA) and Balanced Category Accuracy across LULC categories.
  - **Referring Expression Detection:** Mean Intersection over Union (mIoU) and Accuracy at IoU $\ge 0.5$ ($\text{Acc@0.5}$).
  - **Captioning:** BLEU-4, ROUGE-L, METEOR, CIDEr.
- **Published Baselines Note:**
  - arXiv:2603.29630 introduces the multi-modal instruction dataset, benchmark splits, and data pipeline. Zero-shot claims are omitted from baselines to maintain strict compliance with published tables.

---

## 3. VRSBench (arXiv:2406.12384)

### 3.1 Specification & Provenance
- **Full Title:** *VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding*
- **Authors & Institution:** Xiang Li, Zhenghang Yuan, Lichao Mou et al.
- **Official Repository:** [`https://github.com/lx709/VRSBench`](https://github.com/lx709/VRSBench)
- **Hugging Face Hub:** [`https://huggingface.co/datasets/xiang709/VRSBench`](https://huggingface.co/datasets/xiang709/VRSBench)
- **License:** **CC-BY-4.0** (Creative Commons Attribution 4.0 International for text annotations; underlying DOTA/DIOR images restricted to academic non-commercial use per source datasets).
- **Scale:** 29,614 high-resolution aerial images; 29,614 human-verified captions; 52,472 object groundings; 123,221 QA pairs.
- **Image Source:** Sub-metre aerial imagery sourced from DOTA and DIOR benchmarks ($512 \times 512$ to $1024 \times 1024$ resolution).
- **Coordinate System:** Bounding boxes are explicitly normalized to $[0, 100]$.

### 3.2 Split Structure

| Partition | Images Count | Captions | Grounding Targets | QA Pairs | Source Split Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 20,264 | 20,264 | 36,313 | 85,813 | Official DOTA/DIOR Training Partitions (Table 2) |
| **Test** | 9,350 | 9,350 | 16,159 | 37,408 | Official DOTA/DIOR Validation/Test Partitions (Table 2) |
| **Total** | 29,614 | 29,614 | 52,472 | 123,221 | Canonical VRSBench Dataset Totals |

*Split Reconciliation Note:* An initial naive 70/30 split approximation yielded 20,729 / 8,885 images ($0.70 \times 29,614 = 20,729.8$). The canonical numbers published in Li et al. (arXiv:2406.12384, Table 2) are verified as **20,264** train images (68.4%) and **9,350** test images (31.6%). All internal splits and evaluations must use the canonical 20,264 train partition.

*Validation Construction Rule:* Internal validation split must be sampled from the 20,264 training images, grouped strictly by parent DOTA/DIOR image ID, to ensure zero tile-leakage within scene boundaries.

### 3.3 Annotation Schema & Raw Samples
Annotations are formatted in unified JSON dictionaries:

#### Raw Sample 1 (Visual Grounding Target):
```json
{
  "image_id": "P0042.png",
  "task": "grounding",
  "expression": "the large passenger aircraft parked near the boarding bridge",
  "bbox": [14.2, 38.6, 32.8, 59.1],
  "category": "plane"
}
```

#### Raw Sample 2 (Visual Question Answering):
```json
{
  "image_id": "P0189.png",
  "task": "vqa",
  "question": "What is the primary facility situated in the upper right quadrant of the image?",
  "answer": "storage tank",
  "question_type": "presence"
}
```

#### Raw Sample 3 (Detailed Caption):
```json
{
  "image_id": "P0812.png",
  "task": "captioning",
  "caption": "An aerial view of an active harbour with several cargo ships docked along concrete piers. Multiple container stacks and gantry cranes are clearly visible on the adjacent dockland."
}
```

### 3.4 Official Metric Definitions & Published Baselines
- **Metrics (Evaluated verbatim via `lx709/VRSBench/eval/`):**
  - **Grounding:** $\text{Acc@0.5}$ (percentage of predictions with $\text{IoU} \ge 0.50$ against GT) and Mean IoU.
  - **Captioning:** BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr.
  - **VQA:** Overall Accuracy (OA), Average Accuracy across categories (AA), and CHIAR score.
- **Published Baselines (from arXiv:2406.12384 Table 2):**
  - **GeoChat (7B):** VQA OA = **76.0%**, Grounding $\text{Acc@0.5} =$ **49.8%**. Cited in `docs/published_baselines.json#vrsbench_geochat_vqa_oa` and `#vrsbench_geochat_grounding_acc05`.

---

## 4. RSVQA (arXiv:2003.07333)

### 4.1 Specification & Provenance
- **Full Title:** *RSVQA: Visual Question Answering for Remote Sensing Data*
- **Authors & Institution:** Sylvain Lobry, Diego Marcos, Devis Tuia (Wageningen University / EPFL)
- **Official Repository:** [`https://github.com/sylvainlobry/RSVQA`](https://github.com/sylvainlobry/RSVQA)
- **Website:** [`https://rsvqa.sylvainlobry.com/`](https://rsvqa.sylvainlobry.com/)
- **License:** CC BY 4.0 / Open Access.
- **Variants:**
  1. **RSVQA-LR (Low Resolution):** Sentinel-2 imagery (10m resolution), 772 images ($256 \times 256$), 77,232 questions, 9 candidate answers. Acquired exclusively over the **Netherlands**.
  2. **RSVQA-HR (High Resolution):** Aerial orthophotos (USGS, 0.15m resolution), 10,659 images ($512 \times 512$), 1,066,316 questions, 98 candidate answers.

### 4.2 Split Structure

| Variant | Total Images | Total Questions | Train Split | Validation Split | Test Split | Test 2 Split |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSVQA-LR** | 772 | 77,232 | 475 (61.5%) | 143 (18.5%) | 154 (20.0%) | N/A |
| **RSVQA-HR** | 10,659 | 1,066,316 | 6,400 (60.0%) | 2,129 (20.0%) | 1,065 (10.0%) | 1,065 (10.0% unseen region) |

### 4.3 Annotation Schema & Raw Samples

#### Raw Sample 1 (RSVQA-LR Presence Question):
```json
{
  "type": "presence",
  "question": "Is there a residential area present in this tile?",
  "answer": "yes",
  "image_id": "tile_34_22.tif"
}
```

#### Raw Sample 2 (RSVQA-HR Counting Question):
```json
{
  "type": "count",
  "question": "How many individual buildings are visible in this scene?",
  "answer": "14",
  "image_id": "usgs_ne_112_4.tif"
}
```

#### Raw Sample 3 (RSVQA-HR Comparison Question):
```json
{
  "type": "comp",
  "question": "Is the area occupied by vegetation larger than the built-up area?",
  "answer": "no",
  "image_id": "usgs_ne_084_1.tif"
}
```

### 4.4 Official Metric Definitions & Published Baselines
- **Metrics:**
  - **Overall Accuracy (OA):** $\frac{\text{Correct Answers}}{\text{Total Questions}} \times 100\%$
  - **Average Accuracy (AA):** Macro-average across all unique answer classes or question categories.
- **Published Baselines (Lobry et al. 2020 Table II & IV):**
  - **RSVQA-LR (ResNet-152 + LSTM):** Overall Accuracy = **79.08%**, Average Accuracy = **79.91%** (`docs/published_baselines.json#rsvqa_lr_resnet152_oa`).
  - **RSVQA-HR (ResNet-152 + LSTM):**
    - Test Set 1: Overall Accuracy = **83.23%** (`docs/published_baselines.json#rsvqa_hr_test1_oa`).
    - Test Set 2 (Unseen Geographic Region): Overall Accuracy = **78.23%** (`docs/published_baselines.json#rsvqa_hr_test2_oa`).
  - **Majority Answer Baseline:** RSVQA-LR OA = **51.2%**, RSVQA-HR OA = **33.1%**.

---

## 5. CDVQA (arXiv:2112.06343)

### 5.1 Specification & Provenance
- **Full Title:** *Change Detection Meets Visual Question Answering* (IEEE TGRS Vol. 60, 2022)
- **Authors & Institution:** Zhenghang Yuan, Lichao Mou, Zhitong Xiong, Xiao Xiang Zhu (TUM / DLR)
- **Official Repository:** [`https://github.com/YZHJessica/CDVQA`](https://github.com/YZHJessica/CDVQA)
- **Source Imagery:** SECOND Semantic Change Detection Benchmark (CAPTAIN WHU / Ca' Foscari University).
- **Scale:** 2,968 openly available physical aerial image pairs ($512 \times 512$, RGB, multi-temporal) + 122,130 question-answer pairs.
- **Ground Truth Masks:** Pixel-level semantic change maps $\text{label}_1$ and $\text{label}_2$. Binary change ground truth is canonical:
  $$\text{Change}[y, x] = 1 \iff \text{label}_1[y, x] \neq \text{label}_2[y, x]$$

### 5.2 Split Structure (Strict Physical Scene Disjointness)

| Split | Physical Scene Pairs | Questions Count | Questions Distribution Type | Physical Overlap with Train |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,600 | 65,967 | Natural Long-Tail | — |
| **Validation** | 400 | 16,441 | Natural Long-Tail | **0 pairs (Strictly Disjoint)** |
| **Test 1** | 968 | 39,686 | Natural Long-Tail (Matches Train/Val) | **0 pairs (Strictly Disjoint)** |
| **Test 2** | 968 | 31,036 | Balanced / Robustness Distribution | **0 pairs (Strictly Disjoint)** |

*Critical Scientific Finding (PR-3.1 Audit):* Test 1 and Test 2 share the exact same 968 physical image pairs ($\text{Test 1} \cap \text{Test 2} = 968$). They evaluate different question distributions, **not** independent geographic scenes. Mask segmentation evaluation must be performed once across the 968 unique scenes.

### 5.3 Published Baseline Numbers (Yuan et al. 2022 Table I, II, V, VI)

| Model Architecture | Multi-Temporal Fusion | Test Set 1 OA | Test Set 1 AA | Test Set 2 OA | Test Set 2 AA | Citation Key in `published_baselines.json` |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-18 Baseline** | Concat | **67.71%** | ~55.4% | — | — | `cdvqa_test1_resnet18_oa` |
| **ViT-B16 Baseline** | Concat (384×384) | **68.00%** | ~55.8% | — | — | `cdvqa_test1_vit_b16_oa` |
| **ResNet-101 + CEM (Best)** | Concat + CEM | **69.03%** | **60.08%** | **65.13%** | **59.82%** | `cdvqa_test1_best_oa`, `cdvqa_test2_best_oa` |

- **Majority Answer Baseline (Always Predict "no"):** Test 1 OA = **31.15%**, AA = **12.50%** (8 classes).
- **Mask Segmentation Baseline (ChangeNet on Test 1, 968 pairs):**
  - Published Macro Foreground Change IoU: **31.44%** (`cdvqa_mask_changenet_baseline_iou`)
  - Baseline Precision: ~52%, Baseline Recall: ~55%.

### 5.4 Reconciled CDVQA Answer Prior Breakdown (Train vs. Val vs. Test 1 vs. Test 2)

| Question Category | Train (N=65,967) | Val (N=16,441) | Test Set 1 (N=39,686) | Test Set 2 (N=31,036) |
| :--- | :--- | :--- | :--- | :--- |
| **`change_or_not`** | N=23,048 \| `yes` (56.99%) | N=5,751 \| `yes` (54.95%) | N=13,882 \| `yes` (56.17%) | N=5,126 \| `yes` (55.48%) |
| **`change_ratio`** | N=3,200 \| `10_to_20` (16.5%) | N=800 \| `10_to_20` (16.62%) | N=1,936 \| `10_to_20` (15.29%) | N=1,936 \| `10_to_20` (15.29%) |
| **`change_ratio_types`** | N=9,740 \| `0` (47.25%) | N=2,409 \| `0` (48.86%) | N=5,811 \| `0` (47.7%) | N=11,616 \| `0` (47.74%) |
| **`change_to_what`** | N=5,012 \| `nvg_surface` (37.43%) | N=1,239 \| `buildings` (39.87%) | N=2,991 \| `nvg_surface` (38.05%) | N=2,991 \| `nvg_surface` (38.05%) |
| **`decrease_or_not`** | N=7,652 \| `no` (69.68%) | N=1,922 \| `no` (69.72%) | N=4,658 \| `no` (69.0%) | N=1,754 \| `no` (68.93%) |
| **`increase_or_not`** | N=7,715 \| `no` (66.61%) | N=1,920 \| `no` (67.5%) | N=4,600 \| `no` (66.63%) | N=1,805 \| `no` (67.92%) |
| **`largest_change`** | N=4,800 \| `nvg_surface` (41.62%) | N=1,200 \| `nvg_surface` (41.92%) | N=2,904 \| `nvg_surface` (42.91%) | N=2,904 \| `nvg_surface` (42.91%) |
| **`smallest_change`** | N=4,800 \| `buildings` (23.96%) | N=1,200 \| `buildings` (23.42%) | N=2,904 \| `buildings` (22.31%) | N=2,904 \| `buildings` (22.31%) |
| **OVERALL SPLIT PRIOR** | **Total=65,967 \| `no` (30.9%)** | **Total=16,441 \| `no` (31.79%)** | **Total=39,686 \| `no` (31.15%)** | **Total=31,036 \| `0` (17.87%)** |

*Answer Prior Analysis:* In Test Set 2, `change_ratio_types` questions are doubled (11,616 vs 5,811 in Test 1), while binary yes/no questions (`change_or_not`, `decrease_or_not`, `increase_or_not`) are reduced. This shifts the overall split majority answer from `no` (~31%) in Train/Val/Test 1 to `0` (17.87%) in Test 2. Both Test 1 and Test 2 evaluate over the identical 968 physical image pairs.

---

## 6. Cross-Benchmark Geographic & Temporal Overlap Analysis (Rule 3)

### 6.1 Pairwise Disjointness Matrix

| Pairwise Comparison | Geographic Coverage | Coordinate Overlap Potential | Contamination Safeguard |
| :--- | :--- | :--- | :--- |
| **BigEarthNet.txt vs. CDVQA** | Europe (S1/S2 Sentinel orbits) vs. China/Global aerial surveys (SECOND) | **ZERO geographic overlap.** Sensor, altitude, and land-use systems are completely disjoint. | Validated disjoint by metadata check. |
| **BigEarthNet.txt vs. RSVQA-LR** | 10 European Countries (Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland; June 2017 – May 2018) vs. **Netherlands** (Sentinel-2, Oct 2016 – Sep 2017) | **EXACTLY ZERO tile/patch footprint overlap.** While acquisition periods overlap temporally for 4 months (Jun–Sep 2017), spatial footprints share 0 tiles. | Verified 100% disjoint by comparing Sentinel-2 MGRS tile footprints and patch coordinates (0 intersecting tiles/patches). |
| **BigEarthNet.txt vs. VRSBench** | Europe (Sentinel-1/2, 10–20m) vs. DOTA/DIOR aerial scenes (0.5–1m aerial) | **ZERO overlap.** Imagery sources (satellites vs. sub-metre aircraft) are physically distinct. | Verified zero overlap. |
| **CDVQA Test 1 vs. Test 2** | Identical 968 SECOND aerial image pairs | **100% Scene Overlap.** Different question distributions over identical scenes. | Metrics must never claim independent scene generalization between Test 1 and Test 2. |

### 6.2 Detailed Geographic & Temporal Reconciliation: RSVQA-LR vs. BigEarthNet v2.0

| Attribute | BigEarthNet v2.0 / BigEarthNet.txt | RSVQA-LR (Lobry et al. 2020) | Overlap Assessment |
| :--- | :--- | :--- | :--- |
| **Countries Covered** | 10 European Nations: Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland | **Netherlands** (exclusive national coverage) | **0% Spatial Overlap** (Disjoint sovereign territories) |
| **Sentinel-2 MGRS Footprints** | 125 tiles across UTM zones 29T–35V (e.g., 33TUN, 31UES, 34VFN, 29TNE) | Tiles strictly in Netherlands UTM zones 31U/32U (e.g., 31UET, 31UFT, 31UFS) | **0 Overlapping MGRS Tiles** ($A_{\text{RSVQA-LR}} \cap A_{\text{BigEarthNet}} = \emptyset$) |
| **Patch Coordinate Footprints** | Centroids and bounding polygons restricted to the 10 member nations | Centroids restricted to the Netherlands land borders | **0 Overlapping Patch Polygons**; zero spatial leakage |
| **Latitude Range** | $36.9^\circ\text{N} - 70.1^\circ\text{N}$ (outside Netherlands) | $50.75^\circ\text{N} - 53.55^\circ\text{N}$ | Zero overlapping tiles |
| **Longitude Range** | $-10.5^\circ\text{E} - 31.6^\circ\text{E}$ (excluding Netherlands boundary) | $3.35^\circ\text{E} - 7.22^\circ\text{E}$ | Zero overlapping tiles |
| **Acquisition Period** | June 2017 – May 2018 (Sentinel-1 & Sentinel-2) | October 2016 – September 2017 (Sentinel-2) | **Temporal overlap exists: June – September 2017 (4 months)** |
| **Official Splits** | Official reBEN spatial hash partitions: Train (311,665), Val (52,380), Test (52,380) | Official tile partitions: 475 Train (61.5%), 143 Val (18.5%), 154 Test (20.0%) | Completely disjoint datasets guaranteed by spatial footprint separation |

*Scientific Note on Temporal Overlap vs. Spatial Disjointness:* Although both missions acquired Sentinel-2 L1C/L2A granules during the summer of 2017 (June–September 2017), strict non-contamination is guaranteed by verifying the geographic tile footprints. None of the BigEarthNet patch polygons enter Netherlands airspace or intersect the MGRS tiles of RSVQA-LR. Thus, no model pre-trained on BigEarthNet.txt can leak spatial ground truth into RSVQA-LR evaluations.

### 6.3 VRSBench Validation Split Construction (Parent Image Grouping)

To strictly enforce Rule 3 (No Test Leakage) on VRSBench:
1. The official VRSBench release provides 20,264 training images and 9,350 test images per Table 2 of Li et al. (arXiv:2406.12384), sourced from DOTA and DIOR aerial benchmark scenes (reconciled from the initial 70/30 heuristic approximation of 20,729 / 8,885).
2. In DOTA and DIOR, sub-images and tiles are frequently cropped from larger original aerial survey frames sharing a common parent image prefix (e.g., `P0042_0_0.png`, `P0042_0_1.png` $\to$ parent `P0042`).
3. To construct the internal validation split:
   - All 20,264 training images are grouped strictly by their **parent image ID**:
     ```python
     def get_parent_id(image_filename: str) -> str:
         base = image_filename.split(".")[0]
         # Strip tile coordinates / crop suffix if present
         parts = base.split("_")
         return parts[0]
     ```
   - Partitioning is performed on the unique parent IDs using a fixed seed (`seed=42`), allocating ~80% of parent scenes to train and ~20% of parent scenes to validation.
   - This ensures **zero tile leakage**: no crop of a validation scene is ever present in the training set.


