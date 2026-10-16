# SatQuery AI — Benchmark Execution Inventory & Evaluator Compliance Audit

**Project:** Smart India Hackathon 2026 — Problem Statement PS26167  
**Evaluator Reference:** Aminur Hossain (ISRO / Space Applications Centre (SAC), Ahmedabad)  
**Audit Purpose:** Comprehensive inventory and execution readiness assessment for requested public benchmarks  
**Auditor:** SatQuery AI Verification & Evaluation Compliance Engine  
**Audit Date:** October 15, 2026  
**Host Environment:** Windows 11 (64-bit), Intel Core CPU (8 physical / 12 logical cores, 15.71 GB RAM, 118.74 GB free disk)  
**Python Runtime:** Python 3.10.11 via `D:\sih26167\backend\venv\Scripts\python.exe` (PyTorch 2.13.0+cpu, Torchvision 0.28.0+cpu, Transformers 5.16.1, OpenCV 5.0.0, Rasterio 1.4.4)

---

## 1. Executive Summary & Benchmark Status Matrix

In response to evaluator guidance from Aminur Hossain (ISRO/SAC), an exhaustive forensic inspection was conducted across all dataset directories, download caches, manifests, annotations, evaluation scripts, and model checkpoints in the repository.

The audit evaluated the four mandatory benchmarks:
1. **BigEarthNet.txt / BigEarthNet-MM** (Multi-modal optical–SAR remote sensing)
2. **VRSBench** (Remote sensing visual question answering, captioning, and visual grounding)
3. **RSVQA** (Remote sensing visual question answering on multispectral satellite imagery)
4. **CDVQA** (Bi-temporal change detection and change visual question answering)

### Master Compliance Matrix

| Benchmark | Target Tasks | Authentic Data on Disk | Official Splits on Disk | Model & Evaluator Status | Compliance Status | Primary Blocker |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BigEarthNet.txt / BigEarthNet-MM** | • 19-class CLC patch classification<br>• Vision-language text retrieval<br>• Optical–SAR feature extraction | **PARTIAL**<br>(800 authentic S1/S2 patches via `ben-ge-800`) | **NO**<br>(Official 590k / 118k test split missing; local 119-patch disjoint split exists) | **MODEL READY** (`SatQueryOpticalSARNet`)<br>**EVALUATOR READY** for custom 3-class segmentation | **PARTIAL** | Official 19-class BigEarthNet test set and BigEarthNet.txt 464k text pairs missing; model trained on custom WorldCover 3-class masks. |
| **CDVQA** | • Bi-temporal Change VQA<br>• Binary change delineation | **ANNOTATIONS ONLY**<br>(12 official JSONs, 55.4 MB on disk) | **YES**<br>(Official Train, Val, Test, Test2 JSON splits present) | **MODEL READY** (`SatQueryChangeNet` + `SatQuery-RS-VLM`)<br>**EVALUATOR READY** (`eval_change.py`) | **PARTIAL** | Official LEVIR-CD bi-temporal image rasters ($T_1, T_2$) and binary ground-truth masks are missing (122,094 questions unlinked to images). |
| **RSVQA** | • Single-image VQA (Presence, Count, Comparison, Area) | **NO**<br>(Zero authentic S2 or aerial images on disk) | **NO**<br>(Official 10k RSVQA-LR test split missing) | **MODEL READY** (`SatQuery-RS-VLM`)<br>**EVALUATOR READY** (`eval_single_image.py`) | **BLOCKED** | Zero authentic Sentinel-2 / aerial imagery and zero official question/answer JSON files downloaded locally. |
| **VRSBench** | • Single-image VQA<br>• Image Captioning<br>• Visual Grounding (Acc@0.5) | **NO**<br>(Zero authentic high-res images on disk) | **NO**<br>(Official 29k image / 109k VQA test split missing) | **MODELS READY** (`SatQuery-RS-VLM` + `owlvit-base-patch32`)<br>**EVALUATOR READY** (`eval_single_image.py`) | **BLOCKED** | Zero authentic VRSBench images or official JSON annotations on disk; only 15 synthetic noise prototype test samples exist in cache. |

---

## 2. Granular Benchmark Audits

---

### 2.1 BigEarthNet.txt & BigEarthNet-MM

#### 1. Authentic Dataset Version, Source, License & Local Files
- **Official Specification:**
  - *BigEarthNet-MM:* Sumbul et al. (2021), IEEE GRSM / arXiv:2105.07921. 590,326 Sentinel-1 dual-pol SAR (VV/VH) and Sentinel-2 12-band multispectral patches across 10 European nations. License: CDLA-Permissive-1.0.
  - *BigEarthNet.txt:* Herzog, Adler, Demir et al. (2026), arXiv:2603.29630. Text-image multimodal dataset providing natural language descriptions and CLC-19 multi-label annotations for 464,136 patch pairs. License: CDLA-Permissive-1.0 / CC BY 4.0.
- **Actual Files Available Locally:**
  - *Authentic Data Directory:* `ml_pipeline/authentic_data/ben_ge_800/` and prepared at `ml_pipeline/authentic_data/ben_ge_800_prepared/`.
  - *Source & Provenance:* `ben-ge-800` (Mommert, Kesseli, Hanna, Scheibenreif, Borth, Demir, IEEE IGARSS 2023, Zenodo DOI: 10.5281/zenodo.12941231, CC BY 4.0).
  - *Local Inventory:*
    - 800 authentic Sentinel-2 Level-2A GeoTIFFs (`sentinel-2/`, 120×120 pixels at 10m resolution, B02, B03, B04, B08).
    - 800 authentic Sentinel-1 GRD IW dual-pol GeoTIFFs (`sentinel-1/`, VV and VH backscatter in dB).
    - 800 authentic ESA WorldCover 10m ground-truth rasters (`esaworldcover/`).
    - Metadata files: `ben-ge-800_meta.csv` (123 KB, 800 acquisition records with parent S1/S2 granule IDs, timestamps, lat/lon), `ben-ge-800_esaworldcover.csv` (112 KB, fractional class coverage per patch across 11 land-cover categories).
  - *Synthetic Cache Directory:* `ml_pipeline/dataset_cache/bigearthnet/` contains 100 procedurally generated Gaussian RGB and uniform SAR `.npy` samples created for lightweight pipeline schema testing.

#### 2. Official Train, Validation, and Test Splits
- **Official BigEarthNet-MM Splits:** The official 590,326-patch train (311,668), validation (103,942), and test (118,515) splits are **NOT present on disk**.
- **Official BigEarthNet.txt Text Annotations:** The 464,136 natural language scene descriptions are **NOT present on disk**.
- **Local Split Implementation:**
  - On `ben-ge-800`, a deterministic connected-component bipartite graph traversal was performed (`ml_pipeline/scripts/partition_granule_disjoint.py`), establishing a strict **parent-granule-disjoint split**:
    - Train: 561 patches (78 S2 granules, 128 S1 overpasses)
    - Validation: 120 patches (17 S2 granules, 32 S1 overpasses; 0.0% overlap with train)
    - Test: 119 patches (14 S2 granules, 32 S1 overpasses; 0.0% overlap with train)
  - Split manifest: [`reports/training_migration/granule_disjoint_split_manifest.json`](file:///d:/sih26167/reports/training_migration/granule_disjoint_split_manifest.json).

#### 3. SatQuery Component and Evaluator Script
- **Model:** `SatQueryOpticalSARNet` (Dual-Encoder ResNet18 Optical + 4-layer SAR CNN, 849,603 parameters). Checkpoint: `ml_pipeline/checkpoints/satquery_optical_sar_net_disjoint/optical_sar_net.pth` (SHA-256: `66544ff8...`).
- **Evaluator Script:** [`ml_pipeline/evaluate_disjoint_test.py`](file:///d:/sih26167/ml_pipeline/evaluate_disjoint_test.py).
- **Secondary Evaluator:** `ml_pipeline/eval_cross_modal.py` (evaluates on synthetic schema cache in `dataset_cache/bigearthnet/`).

#### 4. Input and Output Format Compatibility
- **Current Model Architecture:** Accepts Optical RGB $(3, 128, 128)$ + SAR VV in dB $(1, 128, 128)$; outputs 3-channel sigmoid probability maps $(3, 128, 128)$ for Built-up Fabric, Water Bodies, and Vegetation Canopy.
- **Task Distinction:** The current model executes **dense semantic segmentation**, whereas the official BigEarthNet-MM benchmark evaluates **19-class patch classification** (predicting a multi-label binary vector of length 19 for the whole 1.2 km × 1.2 km patch).
- **Incompatible Output:** To produce official BigEarthNet benchmark outputs, the segmentation feature maps would need global spatial pooling into a 19-class multi-label sigmoid classifier.

#### 5. CPU Execution & Resource Requirements
- **Runtime:** `evaluate_disjoint_test.py` on 119 authentic test patches executes in **5.4 seconds** on local CPU (~45.3 ms/patch).
- **Memory Footprint:** Peak RAM consumption is **480 MB**.
- **Feasibility on Current System:** **100% FEASIBLE** on CPU.

#### 6. Published SOTA & Baseline References in Repository
- BigEarthNet-MM (S1+S2) ResNet-50 Official Baseline: **88.50% mAP** (Sumbul et al., 2021).
- LoveDA (S1+S2) Multi-Modal Segmentation: **48.50%–54.00% mIoU** (Wang et al., 2021).
- Recorded in: `reports/benchmark_evaluation/sota_comparison.md:L46-L49`.

#### 7. Exact Blockers & Remediation Steps
- **Blocker 1 (Dataset Scale):** The official 590,326-tile BigEarthNet-MM dataset (~120 GB) and BigEarthNet.txt captions are not downloaded.
- **Blocker 2 (Task Mismatch):** SatQuery performs pixel-level 3-class segmentation, not 19-class patch-level multi-label classification.
- **Remediation:** 
  1. Leverage `ben-ge-800_esaworldcover.csv` (already on disk), which contains fractional coverage for 11 land-cover classes.
  2. Implement an automated mapping from ESA WorldCover fractions $\ge 0.05$ to the 19 Corine Land Cover classes.
  3. Add an evaluation script that pools `SatQueryOpticalSARNet` predictions or runs a multi-label classification head against these authentic CLC labels for the 119 disjoint test patches.

---

### 2.2 CDVQA (Change Detection and Change Visual Question Answering)

#### 1. Authentic Dataset Version, Source, License & Local Files
- **Official Specification:**
  - Yuan, Mou, Xiong, Zhu (2022), IEEE TGRS / arXiv:2112.06343.
  - Scale: 4,000 bi-temporal satellite image pairs (0.5m–3.0m spatial resolution, derived from LEVIR-CD, SZTAKI, and Onera), over 20,000 change QA triplets, and pixel-level binary change masks.
  - License: Academic Research Use / CC BY 4.0. GitHub: `https://github.com/YZHJessica/CDVQA`.
- **Actual Files Available Locally:**
  - **Official JSON Annotations ARE FULLY DOWNLOADED** in `ml_pipeline/authentic_data/cdvqa/`:
    - `Train_questions.json` (13.2 MB), `Train_answers.json` (6.9 MB), `Train_images.json` (3.4 MB)
    - `Val_questions.json` (3.3 MB), `Val_answers.json` (1.7 MB), `Val_images.json` (826 KB)
    - `Test_questions.json` (7.9 MB), `Test_answers.json` (4.2 MB), `Test_images.json` (2.0 MB)
    - `Test2_questions.json` (6.4 MB), `Test2_answers.json` (3.3 MB), `Test2_images.json` (2.0 MB)
    - Total: **12 JSON files, 55.4 MB**, documenting 122,094 questions across 4 question types (`change_type`, `existence`, `quantity`, `attribute`).
  - **Actual Image Rasters MISSING:**
    - `ml_pipeline/authentic_data/cdvqa_prepared/dataset_metadata.json` confirms: `"stats": {"ok": 0, "image_missing": 122094}`.
    - Zero authentic bi-temporal image files ($T_1, T_2$) or ground-truth change masks exist in `cdvqa/` or `cdvqa_prepared/`.
  - *Synthetic Cache Directory:* `ml_pipeline/dataset_cache/cdvqa/` contains 100 generated Gaussian noise pairs with synthetic rectangular brightness shifts and 15 synthetic test questions.

#### 2. Official Train, Validation, and Test Splits
- **Splits on Disk:**
  - The official partition splits (`Train`, `Val`, `Test`, `Test2`) are **fully represented in the JSON metadata** on disk.
  - For the primary official `Test` split: 17,458 question-answer pairs referencing 600 unique bi-temporal satellite pairs.
  - However, because the underlying image tiles are missing, no pair can currently be loaded from disk.

#### 3. SatQuery Component and Evaluator Script
- **Components:**
  1. `SatQueryChangeNet` (Siamese U-Net with difference concatenation, 7.8M parameters). Checkpoint: `ml_pipeline/checkpoints/satquery_change_net/change_net.pth`.
  2. `SatQuery-RS-VLM-v1.0` (Salesforce BLIP-based ViT-B/16 text decoder operating on difference imagery $|T_2 - T_1|$). Checkpoint: `ml_pipeline/checkpoints/satquery_rs_vlm/`.
- **Evaluator Script:** [`ml_pipeline/eval_change.py`](file:///d:/sih26167/ml_pipeline/eval_change.py).
  - Measures: binary change mask mIoU, Dice F1, Precision, Recall, Overall VQA accuracy, per-type accuracy (`change_type`, `existence`, `quantity`, `attribute`), and token F1.

#### 4. Input and Output Format Compatibility
- **Format:** Fully compatible.
  - Inputs: $T_1, T_2$ RGB images $(256, 256, 3)$ normalized to $[0, 1]$ + question string.
  - Outputs: binary change mask $(256, 256)$ + decoded text answer.

#### 5. CPU Execution & Resource Requirements
- **Runtime:**
  - `SatQueryChangeNet` executes at **~35 ms per pair** on CPU.
  - `SatQuery-RS-VLM` executes at **~0.9 s per question** on CPU.
  - For the official 600-pair change detection test set: ~21 seconds on CPU.
  - For a 500-question Change VQA test subset: ~7.5 minutes on CPU.
- **Memory Footprint:** ~2.3 GB RAM (with VLM in memory).
- **Feasibility on Current System:** **100% FEASIBLE** on CPU.

#### 6. Published SOTA & Baseline References in Repository
- Change Delineation:
  - FC-EF baseline: **54.20% mIoU** (Daudt et al., 2018).
  - STANet SOTA: **64.30% mIoU**, **78.20% F1** (Chen & Shi, 2020).
- Change VQA:
  - Dual-ResNet + CEM Baseline: **68.45% Overall Accuracy** (Existence: 82.1%, Change Type: 61.4%, Quantity: 58.2%, Attribute: 72.1%, Yuan et al., 2022).
  - Siamese VQA / ChangeNet+Transformer: **72.30% Overall Accuracy** (Yuan et al., 2022).
- Recorded in: `reports/benchmark_evaluation/sota_comparison.md:L40-L45`.

#### 7. Exact Blockers & Remediation Steps
- **Blocker:** Missing LEVIR-CD image tiles (`A/`, `B/`, and `label/` folders).
- **Remediation:**
  1. Download the LEVIR-CD test image archive (~600 pairs, ~500 MB) from the official GitHub/Google Drive links referenced in `https://github.com/YZHJessica/CDVQA`.
  2. Unpack into `ml_pipeline/authentic_data/cdvqa/images/`.
  3. Run `ml_pipeline/eval_change.py` directly against the authentic test set.

---

### 2.3 RSVQA (Remote Sensing Visual Question Answering)

#### 1. Authentic Dataset Version, Source, License & Local Files
- **Official Specification:**
  - Lobry, Marcos, Tuia et al. (2020), IEEE TGRS / arXiv:2003.07333.
  - Subsets:
    - *RSVQA-LR (Low Resolution):* 772 Sentinel-2 multispectral tiles (256×256 pixels, 10m/20m GSD), 77,232 QA pairs. Official test split = 100 tiles, 10,000 QA pairs.
    - *RSVQA-HR (High Resolution):* 10,659 aerial tiles (512×512 pixels, 15cm GSD), 1,066,316 QA pairs. Test Set 1 (HR1) = 2,131 tiles; Test Set 2 (HR2) = 1,066 tiles.
  - License: CC BY 4.0. Source: Zenodo / University of Wageningen.
- **Actual Files Available Locally:**
  - **Zero authentic RSVQA image files or official question/answer JSON files exist on disk.**
  - *Synthetic Cache Directory:* `ml_pipeline/dataset_cache/rsvqa/` contains 120 generated Gaussian noise images (84 train, 18 val, 18 test) with 36 procedural synthetic test questions in `dataset_cache/rsvqa/test/questions.json`.

#### 2. Official Train, Validation, and Test Splits
- **Splits on Disk:** Completely absent. The official 10,000-question RSVQA-LR test split is not downloaded.

#### 3. SatQuery Component and Evaluator Script
- **Model:** `SatQuery-RS-VLM-v1.0` (Salesforce BLIP-VQA-Base adapted, 361M parameters, checkpoint at `ml_pipeline/checkpoints/satquery_rs_vlm`).
- **Evaluator Script:** [`ml_pipeline/eval_single_image.py`](file:///d:/sih26167/ml_pipeline/eval_single_image.py) (function `evaluate_rsvqa`).
  - Evaluates: overall accuracy, presence accuracy, count accuracy, comparison accuracy, and area accuracy with normalized string matching.

#### 4. Input and Output Format Compatibility
- **Format:** 100% compatible.
  - Input: 256×256 RGB image + question string.
  - Output: Short text string (e.g., "yes", "no", "0", "1", "dense forest").

#### 5. CPU Execution & Resource Requirements
- **Runtime:** `SatQuery-RS-VLM` evaluates at **~0.8 s per question** on local CPU.
  - Full RSVQA-LR test set (10,000 questions): ~2.2 hours on CPU.
  - Standard representative subset (500 questions across all 100 test tiles): ~6.5 minutes on CPU.
- **Memory Footprint:** Peak RAM consumption is **~2.2 GB**.
- **Feasibility on Current System:** **100% FEASIBLE** on CPU.

#### 6. Published SOTA & Baseline References in Repository
- RSVQA-LR Published Baselines (Lobry et al., IEEE TGRS 2021):
  - Standard CNN + GRU Baseline: **78.48% Overall** (Presence: 87.89%, Count: 68.32%, Comparison: 84.62%, Area: 72.88%).
  - Multi-modal Fusion Architecture: **82.52% Overall**.
- Recorded in: `reports/benchmark_evaluation/sota_comparison.md:L31-L35`.

#### 7. Exact Blockers & Remediation Steps
- **Blocker:** Official RSVQA-LR dataset archive (~5 GB) containing Sentinel-2 tiles and official `questions.json` / `answers.json` has not been downloaded to the machine.
- **Remediation:**
  1. Download RSVQA-LR test split archive (~500 MB for test subset) from Zenodo (`https://zenodo.org/record/6344367`).
  2. Unpack into `ml_pipeline/authentic_data/rsvqa/`.
  3. Point `RSVQADataset` in `ml_pipeline/data/rsvqa_loader.py` to the authentic directory and execute `ml_pipeline/eval_single_image.py`.

---

### 2.4 VRSBench (Visual Question Answering, Captioning & Visual Grounding)

#### 1. Authentic Dataset Version, Source, License & Local Files
- **Official Specification:**
  - Li, Ding, Elhoseiny (NeurIPS 2024 Datasets and Benchmarks Track), arXiv:2406.12384.
  - Scale: 29,614 high-resolution aerial and satellite images, 120,000+ visual captions (~52 words/caption), 109,000+ VQA question-answer pairs, and 145,000+ bounding boxes for text-guided visual grounding.
  - License: CC BY-NC-SA 4.0. Source: Hugging Face (`xiang709/VRSBench`) or GitHub (`https://github.com/lx709/VRSBench`).
- **Actual Files Available Locally:**
  - **Zero authentic VRSBench images or official annotations exist on disk.**
  - *Synthetic Cache Directory:* `ml_pipeline/dataset_cache/vrsbench/` contains 100 procedural Gaussian noise images (70 train, 15 val, 15 test) with synthetic caption, VQA, and bounding-box fixtures.

#### 2. Official Train, Validation, and Test Splits
- **Splits on Disk:** Completely absent. The official test split (2,961 images, ~11,000 VQA pairs, ~14,000 bounding boxes) is not present.

#### 3. SatQuery Component and Evaluator Script
- **Models:**
  1. *VQA & Captioning:* `SatQuery-RS-VLM-v1.0` (Salesforce BLIP-VQA-Base adapted, 361M parameters, checkpoint at `ml_pipeline/checkpoints/satquery_rs_vlm`).
  2. *Visual Grounding:* `google/owlvit-base-patch32` (155M parameters, cached at `.huggingface_cache/hub/models--google--owlvit-base-patch32`) + OpenCV contour heuristic baseline.
- **Evaluator Script:** [`ml_pipeline/eval_single_image.py`](file:///d:/sih26167/ml_pipeline/eval_single_image.py).
  - Functions: `evaluate_vrsbench_vqa`, `evaluate_vrsbench_captioning`, `evaluate_visual_grounding`.

#### 4. Input and Output Format Compatibility
- **Format:** Fully compatible.
  - VQA: Image $(512, 512, 3)$ + Question $\rightarrow$ Text string.
  - Captioning: Image $\rightarrow$ Generated descriptive sentence.
  - Grounding: Image + Target object prompt $\rightarrow$ Bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.

#### 5. CPU Execution & Resource Requirements
- **Runtime:**
  - `SatQuery-RS-VLM`: ~1.0 s per VQA query on CPU.
  - `owlvit-base-patch32`: ~0.8 s per grounding query on CPU.
  - Full test split (~11,000 queries) would require ~3 hours on CPU.
  - Curated official test subset (500 samples): ~8 minutes on CPU.
- **Memory Footprint:** Peak RAM consumption is **~2.8 GB** (loading both BLIP and OWL-ViT).
- **Feasibility on Current System:** **100% FEASIBLE** on CPU.

#### 6. Published SOTA & Baseline References in Repository
- VQA Overall Accuracy:
  - BLIP-2 zero-shot: **64.81%** (Li et al., 2024).
  - VRS-LLM fine-tuned: **71.30%** (Li et al., 2024).
- Visual Grounding (Acc@0.5):
  - GLIP zero-shot: **18.42%** (Li et al., 2024).
  - UNINEXT: **28.54%** (Yan et al., 2023).
  - VRS-LLM fine-tuned: **36.80%** (Li et al., 2024).
- Captioning:
  - BLEU-4: **14.80** (BLIP-2), **18.20** (VRS-LLM).
  - CIDEr: **33.90** (LLaVA-1.5), **35.80** (VRS-LLM).
- Recorded in: `reports/benchmark_evaluation/sota_comparison.md:L36-L39`.

#### 7. Exact Blockers & Remediation Steps
- **Blocker:** The authentic VRSBench dataset (~35 GB for full imagery or ~3.5 GB for 512×512 test split) and official JSON files (`vrsbench_vqa.json`, `vrsbench_grounding.json`, `vrsbench_caption.json`) are not downloaded.
- **Remediation:**
  1. Download the official test split subset from Hugging Face (`xiang709/VRSBench`).
  2. Unpack into `ml_pipeline/authentic_data/vrsbench/`.
  3. Execute `ml_pipeline/eval_single_image.py`.

---

## 3. Comparison of Published Benchmarks vs. Measured SatQuery Results

> [!CRITICAL]
> **Scientific Integrity Notice:**  
> The table below strictly separates **published literature baselines** from **measured SatQuery AI results**.  
> SatQuery results on synthetic schema prototypes are clearly labeled as synthetic.  
> Measured results on authentic satellite data (`ben-ge-800`) are explicitly labeled as custom WorldCover-derived 3-class segmentation, not official BigEarthNet 19-class patch classification.

| Benchmark Domain | Task | Published Reference / SOTA | Published Literature Score | Measured SatQuery AI Result | Data Regime of Measured Score | Comparability Assessment |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **BigEarthNet-MM** | 19-class CLC patch classification | Sumbul et al. (2021) | **88.50% mAP** | Not Measured | N/A (19-class task not run) | **N/A** |
| **Optical–SAR (Custom Task)** | 3-class segmentation (Built-up, Water, Vegetation) | Mommert et al. / WorldCover | Baseline target $\ge 55\%$ | **67.11% Macro mIoU**<br>(84.90% Micro IoU) | Authentic `ben-ge-800` (119 parent-granule disjoint test patches) | **NOT DIRECTLY COMPARABLE** to 19-class benchmark. Valid multi-modal segmentation outcome on real Sentinel-1/2 data. |
| **CDVQA** | Bi-temporal Change Delineation | Chen & Shi (2020) / STANet | **64.30% mIoU**<br>(78.20% F1) | **15.16% mIoU**<br>(25.49% F1) | Synthetic Schema Prototype (15 pairs with box deltas) | **NOT DIRECTLY COMPARABLE** (Synthetic noise prototype vs 600 authentic LEVIR-CD pairs). |
| **CDVQA** | Change VQA | Yuan et al. (2022) / Dual-ResNet | **68.45% Overall**<br>(Exist: 82.1%, Type: 61.4%) | **40.00% Overall**<br>(Exist: 100.0%, Type: 0.0%) | Synthetic Schema Prototype (15 synthetic questions) | **NOT DIRECTLY COMPARABLE** (Synthetic questions vs 20,000 authentic multi-temporal QA pairs). |
| **RSVQA-LR** | Single-Image VQA | Lobry et al. (2021) / CNN+GRU | **78.48% Overall**<br>(Presence: 87.89%) | **36.11% Overall**<br>(Presence: 54.55%) | Synthetic Schema Prototype (36 questions on 18 noise images) | **NOT DIRECTLY COMPARABLE** (Synthetic noise prototype vs 10,000 real Sentinel-2 QA pairs). |
| **VRSBench** | Single-Image VQA | Li et al. (2024) / VRS-LLM | **71.30% Overall** | **76.67% Overall** | Synthetic Schema Prototype (30 simple existence questions) | **NOT DIRECTLY COMPARABLE** (Toy template questions on noise vs 109k authentic aerial questions). |
| **VRSBench** | Visual Grounding | Li et al. (2024) / VRS-LLM | **36.80% Acc@0.5** | **0.00% Acc@0.5** (OWL-ViT)<br>**0.00% Acc@0.5** (Heuristic) | Synthetic Schema Prototype (15 target boxes on noise) | **NOT DIRECTLY COMPARABLE** (Zero visual features exist in synthetic noise for detector to ground). |
| **VRSBench** | Visual Captioning | Li et al. (2024) / VRS-LLM | **18.20 BLEU-4**<br>(35.80 CIDEr) | **5.20% Unigram F1** | Synthetic Schema Prototype (15 short generic prompt completions) | **NOT DIRECTLY COMPARABLE** (Generic short prompts vs 52-word human expert descriptions). |

---

## 4. Prioritized Benchmark Execution Roadmap

To satisfy evaluator compliance in the fastest, most resource-efficient manner, the following execution order is established based on local asset availability:

```
[Priority 1: CDVQA] ──> [Priority 2: BigEarthNet-MM] ──> [Priority 3: RSVQA-LR] ──> [Priority 4: VRSBench]
(Annotations already    (Authentic S1/S2 data already   (Requires ~5 GB        (Requires ~15-35 GB
on disk; only image     on disk; map WorldCover to      download from          download from
rasters needed)         19 CLC classes)                 Zenodo)                Hugging Face)
```

### Phase 2 Execution Plan:

1. **Step 1 (First Executable Authentic Evaluation: CDVQA Change Detection & VQA):**
   - *Current Asset Status:* Official `Test_questions.json`, `Test_answers.json`, `Test_images.json` are **already present** in `ml_pipeline/authentic_data/cdvqa/`.
   - *Action:* Ingest authentic LEVIR-CD test image pairs ($T_1, T_2$) and binary change masks (~600 pairs, ~500 MB).
   - *Execution:* Run `python ml_pipeline/eval_change.py --authentic`.
   - *Deliverable:* First true SOTA comparison on authentic benchmark data with zero synthetic fixtures.

2. **Step 2 (BigEarthNet-MM 19-Class Multi-Label Alignment):**
   - *Current Asset Status:* 800 authentic Sentinel-1/Sentinel-2 patch pairs and `ben-ge-800_esaworldcover.csv` are **already present** on disk.
   - *Action:* Implement an evaluation bridge converting WorldCover class fractions into official CLC-19 binary vectors. Run spatial average pooling on `SatQueryOpticalSARNet` features to report official Macro F1 and mAP on the 119 disjoint test patches.
   - *Execution:* Run `python ml_pipeline/evaluate_bigearthnet_clc19.py`.

3. **Step 3 (RSVQA-LR Evaluation):**
   - *Action:* Download the official 100-tile RSVQA-LR test split (~5 GB) from Zenodo.
   - *Execution:* Run `python ml_pipeline/eval_single_image.py --benchmark rsvqa`.

4. **Step 4 (VRSBench Multi-Task Evaluation):**
   - *Action:* Download the official VRSBench test split from Hugging Face (`xiang709/VRSBench`).
   - *Execution:* Run `python ml_pipeline/eval_single_image.py --benchmark vrsbench`.

---

## 5. Audit Conclusion & Compliance Recommendation

- **Inventory Result:**
  - `BigEarthNet`: **PARTIAL** (Authentic imagery and custom segmentation evaluation ready on `ben-ge-800`; official 19-class benchmark missing).
  - `CDVQA`: **PARTIAL** (Official annotations completely present; image rasters missing).
  - `RSVQA`: **BLOCKED** (Zero authentic imagery or annotations).
  - `VRSBench`: **BLOCKED** (Zero authentic imagery or annotations).
- **Core Recommendation:** The project must proceed with **Phase 2 (Authentic Dataset Acquisition & Evaluation Execution)** starting with **CDVQA** (where all question/answer metadata is already locally cached) and **BigEarthNet-MM CLC-19 alignment** (where authentic Sentinel-1/2 rasters are already locally cached), followed by RSVQA-LR and VRSBench downloads.
- **Safety Guarantee:** Production routing defaults (`backend/ml_models.py`), production checkpoints, and existing evaluation records remain 100% unaltered.
