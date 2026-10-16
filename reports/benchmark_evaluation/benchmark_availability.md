# SatQuery AI — Official Benchmark Availability & Forensic Audit Report

**Project:** Smart India Hackathon 2026 — Problem Statement PS26167  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC), Ahmedabad  
**Evaluation Scope:** Benchmark dataset availability, provenance verification, split integrity, and pipeline task support  
**Audit Date:** October 15, 2026  
**Auditor:** SatQuery AI Forensic Verification & Benchmark Assessment Team  

---

## Executive Summary

This report establishes the forensic audit of benchmark dataset availability for SatQuery AI, addressing the evaluator's request for formal benchmark verification. 

A strict provenance inspection was performed across all dataset loaders, local disk caches, procedural scripts, model checkpoints, and evaluation harnesses. The findings confirm that while full end-to-end data pipelines, loaders, and evaluation interfaces exist for all required tasks, **the local cached data files consist of deterministic synthetic schema prototypes** created for schema validation, pipeline testing, and lightweight integration. **Authentic benchmark datasets (consisting of gigabytes of real satellite tiles) have not been downloaded to the local workspace**, and **no official ground-truth reference masks or verified QA pairs were supplied by ISRO/SAC for the Cartosat-2S / RISAT-1 demo pairs**.

Treating generated fixtures or cached demonstration examples as official benchmark samples is strictly avoided. Below is the complete forensic audit of each required benchmark.

---

## 1. Comprehensive Benchmark Audit Matrix

| Benchmark | Dataset Version & Reference | Local Path | Available Splits | Sample Counts (Images / Pairs / Annotations) | Authenticity Status | Supported Model & Evaluator | Prescribed Split Status |
|---|---|---|---|---|:---:|---|:---:|
| **VRSBench** | NeurIPS 2024 / arXiv:2406.12384 (Li, Ding, Elhoseiny et al.) | `ml_pipeline/dataset_cache/vrsbench/` | • `train`<br>• `val`<br>• `test` | **Train:** 70 images, 70 captions, 70 bboxes, 140 VQA<br>**Val:** 15 images, 15 captions, 15 bboxes, 30 VQA<br>**Test:** 15 images, 15 captions, 15 bboxes, 30 VQA | **SYNTHETIC SCHEMA PROTOTYPE** (Procedural Gaussian noise arrays with synthetic arithmetic annotations) | **VQA:** `SatQuery-RS-VLM-v1.0`<br>**Captioning:** `SatQuery-RS-VLM-v1.0`<br>**Grounding:** Heuristic baseline & `google/owlvit-base-patch32` | **PARTIAL** (Synthetic schema subset only; official 29k-image test split not on disk) |
| **RSVQA** | IEEE TGRS 2020 / arXiv:2003.07333 (Lobry, Marcos, Tuia et al.) | `ml_pipeline/dataset_cache/rsvqa/` | • `train`<br>• `val`<br>• `test` | **Train:** 84 images, 168 questions<br>**Val:** 18 images, 36 questions<br>**Test:** 18 images, 36 questions (Presence: 11, Count: 9, Comparison: 8, Area: 8) | **SYNTHETIC SCHEMA PROTOTYPE** (Procedural Gaussian noise arrays with template-derived QA records) | **VQA:** `SatQuery-RS-VLM-v1.0`<br>Evaluates presence, count, comparison, and area accuracy | **PARTIAL** (Synthetic schema subset only; official RSVQA-LR / RSVQA-HR test split not on disk) |
| **CDVQA** | IGARSS 2022 / IEEE TGRS 2022 / arXiv:2112.06343 (Yuan, Xia et al.) | `ml_pipeline/dataset_cache/cdvqa/` | • `train`<br>• `val`<br>• `test` | **Train:** 70 pairs, 70 masks, 70 QA<br>**Val:** 15 pairs, 15 masks, 15 QA<br>**Test:** 15 pairs, 15 masks, 15 QA (Existence: 6, Change Type: 3, Quantity: 3, Attribute: 3) | **SYNTHETIC SCHEMA PROTOTYPE** (T1 Gaussian noise, T2 synthetic rectangular brightness deltas, binary box masks) | **Mask:** `SatQueryChangeNet-v1.0` (Siamese U-Net)<br>**QA:** `SatQuery-RS-VLM-v1.0` on bitemporal difference imagery | **PARTIAL** (Synthetic schema subset only; official 600-pair CDVQA test split not on disk) |
| **BigEarthNet.txt (Optical–SAR Surrogate)** | arXiv:2603.29630 (Herzog, Adler, Demir et al.) | `ml_pipeline/dataset_cache/bigearthnet/` | • `train`<br>• `val`<br>• `test` | **Train:** 70 pairs, 70 annotations<br>**Val:** 15 pairs, 15 annotations<br>**Test:** 15 pairs, 15 annotations (S1 SAR .npy + S2 RGB .png) | **SYNTHETIC SCHEMA PROTOTYPE** (Synthetic Gaussian RGB + uniform SAR float arrays, heuristic pseudo-masks) | **Joint Segmentation:** `SatQueryOpticalSARNet-v1.0` (Dual-Encoder Fusion Network) | **PARTIAL** (Synthetic schema subset only; official BigEarthNet-MM test split not on disk) |
| **ISRO / SAC Cartosat-2S & RISAT-1** | SIH 2026 PS26167 (ISRO / SAC Ahmedabad) | `frontend/public/demo/`<br>`ml_pipeline/isro_evaluation_set.json` | None (Demo images only) | **Images:** 2 demo rasters (`mumbai_cartosat2s_optical.png`, `risat1_cband_radar_sar.png`)<br>**Reference Masks:** 0<br>**Verified QA:** 0 | **SYNTHETIC / DEMO STUB** (Procedural demo images created for UI demonstration without ground-truth labels) | `SatQueryOpticalSARNet-v1.0` & spatial co-registration guardrails | **BLOCKED** (Zero ground-truth reference masks or verified QA pairs supplied or available) |

---

## 2. Granular Benchmark Audits

### 2.1 VRSBench (NeurIPS 2024 / arXiv:2406.12384)
- **Official Specification:**  
  *Authors:* Xiang Li, Jian Ding, Mohamed Elhoseiny (NeurIPS 2024 Datasets and Benchmarks Track).  
  *Official Scale:* 29,614 high-resolution aerial and satellite images, 120,000+ visual captions (average word length ~52 tokens), 109,000+ VQA question-answer pairs, and 145,000+ horizontal bounding boxes for text-guided visual grounding.
- **Local Implementation & File Path:**  
  `ml_pipeline/dataset_cache/vrsbench/`
- **Splits on Disk:**
  - `train/`: 70 images (`train/images/vrs_train_0000.png` ... `vrs_train_0069.png`), 70 caption entries, 70 grounding entries, 140 VQA entries.
  - `val/`: 15 images, 15 caption entries, 15 grounding entries, 30 VQA entries.
  - `test/`: 15 images, 15 caption entries, 15 grounding entries, 30 VQA entries.
- **Authenticity Audit:**  
  Inspection of `ml_pipeline/data/vrsbench_loader.py` (lines 60–65) demonstrates that the images are generated on-the-fly using NumPy random state noise:
  ```python
  rng = np.random.RandomState(2000 + idx)
  base_color = rng.randint(40, 190, size=(3,), dtype=np.uint8)
  img_arr = np.clip(rng.normal(base_color, 30, (256, 256, 3)), 0, 255).astype(np.uint8)
  Image.fromarray(img_arr).save(img_path)
  ```
  The captions and bounding boxes are procedural synthetic templates (e.g., box `[20, 30, 80, 80]` for `category="airplane"`).
- **Task & Model Support:**
  - *VQA:* Fully supported by `SatQuery-RS-VLM-v1.0` (BLIP-based ViT-B/16 + cross-attention decoder).
  - *Captioning:* Supported by `SatQuery-RS-VLM-v1.0` with prompt-conditioned generation and unigram lexical scoring.
  - *Visual Grounding:* Evaluated via both a blind OpenCV spectral/contour heuristic baseline and neural zero-shot open-vocabulary detection using `google/owlvit-base-patch32` (cached in `.huggingface_cache`).
- **Conclusion:** Authentic VRSBench data is **NOT PRESENT**. The local files represent a functional schema prototype for integration testing.

---

### 2.2 RSVQA (IEEE TGRS 2020 / arXiv:2003.07333)
- **Official Specification:**  
  *Authors:* Sylvain Lobry, Diego Marcos, Devis Tuia (IEEE Transactions on Geoscience and Remote Sensing, 2021).  
  *Official Scale:*  
  - *RSVQA-LR:* 772 Sentinel-2 multispectral tiles (256×256 pixels, 10m/20m GSD), 77,232 QA pairs. Official test split consists of 100 tiles and 10,000 questions.  
  - *RSVQA-HR:* 10,659 high-resolution aerial tiles (512×512 pixels, 15cm GSD), 1,066,316 QA pairs. Test Set 1 (HR1): 2,131 tiles; Test Set 2 (HR2): 1,066 tiles.
- **Local Implementation & File Path:**  
  `ml_pipeline/dataset_cache/rsvqa/`
- **Splits on Disk:**
  - `train/`: 84 images, 168 questions, 168 answers.
  - `val/`: 18 images, 36 questions, 36 answers.
  - `test/`: 18 images, 36 questions, 36 answers.
- **Question Type Distribution (Test Split):**
  - *Presence:* 11 questions (e.g., "Is there a water body visible in this image?")
  - *Count:* 9 questions (e.g., "How many large industrial buildings are there?")
  - *Comparison:* 8 questions (e.g., "Is the vegetation area larger than the urban area?")
  - *Area:* 8 questions (e.g., "What is the primary land cover occupying this scene?")
- **Authenticity Audit:**  
  Inspection of `ml_pipeline/data/rsvqa_loader.py` (lines 70–75) reveals that images are procedural normal noise textures (`np.random.RandomState(1000 + img_idx)`). Authentic Sentinel-2 or aerial imagery is not downloaded.
- **Task & Model Support:**  
  Supported by `SatQuery-RS-VLM-v1.0` with full normalized string matching and question-type breakdown.
- **Conclusion:** Authentic RSVQA data is **NOT PRESENT**. Local data is a synthetic schema prototype.

---

### 2.3 CDVQA (IGARSS 2022 / IEEE TGRS 2022 / arXiv:2112.06343)
- **Official Specification:**  
  *Authors:* Zhenghang Yuan, Lichao Mou, Zhitong Xiong, Xiao Xiang Zhu.  
  *Official Scale:* 4,000 bi-temporal image pairs (0.5m–3.0m spatial resolution) derived from real multi-temporal datasets (SZTAKI, Onera, SECOND, LEVIR-CD), featuring over 20,000 change QA triplets and ground-truth pixel-level binary change masks. Official test split consists of 600 pairs.
- **Local Implementation & File Path:**  
  `ml_pipeline/dataset_cache/cdvqa/`
- **Splits on Disk:**
  - `train/`: 70 pairs (T1, T2, binary change mask, QA annotation).
  - `val/`: 15 pairs.
  - `test/`: 15 pairs (T1, T2, mask, QA annotation across 4 question categories: change_type: 3, existence: 6, quantity: 3, attribute: 3).
- **Authenticity Audit:**  
  `ml_pipeline/data/cdvqa_loader.py` (lines 78–97) generates T1 as Gaussian noise and T2 as a modified copy of T1 where an artificial rectangular delta `t2_arr[by1:by2, bx1:bx2] = np.clip(...) + 60` is injected, saving a rectangular binary mask. Authentic bi-temporal satellite pairs (e.g., from LEVIR-CD or SECOND) are not present.
- **Task & Model Support:**  
  Supported by:
  1. `SatQueryChangeNet-v1.0` (Siamese U-Net) for pixel-wise change delineation (evaluated via mIoU, Dice F1, Precision, Recall).
  2. `SatQuery-RS-VLM-v1.0` for Change VQA using difference imagery.
- **Conclusion:** Authentic CDVQA pairs are **NOT PRESENT**. Local data is a synthetic schema prototype.

---

### 2.4 BigEarthNet.txt (Optical–SAR Surrogate / arXiv:2603.29630)
- **Official Specification:**  
  *Authors:* Herzog, Adler, Demir et al. (BigEarthNet-MM / BigEarthNet.txt).  
  *Official Scale:* 590,326 pairs of Sentinel-1 dual-pol SAR (VV/VH) and Sentinel-2 12-band multispectral patches across 10 European countries, with 19-class Corine Land Cover (CLC) annotations.
- **Local Implementation & File Path:**  
  `ml_pipeline/dataset_cache/bigearthnet/`
- **Splits on Disk:**
  - `train/`: 70 S1 SAR (.npy) + S2 RGB (.png) pairs.
  - `val/`: 15 pairs.
  - `test/`: 15 pairs.
- **Authenticity Audit:**  
  `ml_pipeline/data/bigearthnet_loader.py` (lines 96–109) generates synthetic normal noise S2 RGB patches and uniform random float arrays in range `[-25.0, 0.0]` dB for S1 SAR. Semantic masks are generated procedurally by `generate_spatial_targets()` in `train_cross_modal.py` using color/variance thresholding as heuristic pseudo-targets.
- **Task & Model Support:**  
  Supported by `SatQueryOpticalSARNet-v1.0` (dual-encoder fusion network, 849,603 parameters) predicting built-up, water, and vegetation masks.
- **Conclusion:** Authentic BigEarthNet Sentinel-1/Sentinel-2 raw granules are **NOT PRESENT**. Local data is a synthetic schema prototype.

---

### 2.5 ISRO / SAC Cartosat-2S & RISAT-1 Data (SIH PS26167)
- **Official Specification:**  
  SIH 2026 Problem Statement PS26167 requires multimodal integration of optical (Cartosat-2S, 0.65m PAN / 2.0m MX) and SAR (RISAT-1 / RISAT-2B C-band radar) data.
- **Local Implementation & File Path:**  
  `frontend/public/demo/` contains demo files:
  - `mumbai_cartosat2s_optical.png` (512×512 RGB)
  - `risat1_cband_radar_sar.png` (512×512 single-channel radar mockup)
  - `bengaluru_sentinel2_optical.png` (512×512 RGB)
  `ml_pipeline/isro_evaluation_set.json` contains a 5-item metadata stub referencing nonexistent paths (`./dummy_dataset/val/images/image_000.tif`).
- **Authenticity Audit:**  
  As forensically verified in `reports/isro_evaluation.json`:
  1. No authentic Cartosat-2S GeoTIFF granules or calibrated RISAT-1 radar datasets were supplied to or stored within the repository.
  2. **Zero ground-truth reference masks exist** for built-up infrastructure, hydrological bodies, or vegetation on the demo imagery.
  3. **Zero verified ground-truth QA annotations exist** for the demo scenes.
- **Task & Model Support:**  
  `SatQueryOpticalSARNet-v1.0` can execute forward inference on the demo images, and spatial co-registration guardrails operate properly (verifying the Mumbai true pair and rejecting the Bengaluru/Mumbai disparate pair). However, quantitative accuracy (IoU, F1, QA accuracy) cannot be measured without ground-truth reference masks.
- **Conclusion:** Official ISRO benchmark evaluation is **BLOCKED** due to complete absence of reference ground-truth annotations.

---

## 3. Forensic Summary of Local Datasets

| Dataset Entity | Disk Location | Total Samples (All Splits) | True Authentic Satellite Data? | Official Ground-Truth Annotations? | Can Claim Official Benchmark Score? |
|---|---|:---:|:---:|:---:|:---:|
| **VRSBench Cache** | `ml_pipeline/dataset_cache/vrsbench/` | 100 images, 200 QA, 100 caps, 100 bboxes | NO (Gaussian Noise) | NO (Procedural Fixture) | **NO** (Prototype only) |
| **RSVQA Cache** | `ml_pipeline/dataset_cache/rsvqa/` | 120 images, 240 QA pairs | NO (Gaussian Noise) | NO (Procedural Fixture) | **NO** (Prototype only) |
| **CDVQA Cache** | `ml_pipeline/dataset_cache/cdvqa/` | 100 pairs, 100 masks, 100 QA pairs | NO (Gaussian Noise + Box Deltas) | NO (Procedural Fixture) | **NO** (Prototype only) |
| **BigEarthNet Cache**| `ml_pipeline/dataset_cache/bigearthnet/` | 100 S1/S2 pairs, 100 text captions | NO (Gaussian RGB + Uniform SAR) | NO (Heuristic Pseudo-Masks) | **NO** (Prototype only) |
| **ISRO Demo Images** | `frontend/public/demo/` | 3 demo rasters | NO (Procedural UI Mockups) | NO (0 reference masks/QA) | **NO** (Evaluation Blocked) |
| **LoveDA Archive** | `ml_pipeline/_tmp_downloads/train.zip` | 2.39 GB ZIP archive | YES (Unextracted LoveDA data) | Unextracted / Not hooked to loaders | **NO** (Not configured) |

---

## 4. Next Steps Required to Close Availability Gaps

To transition SatQuery AI from prototype schema evaluation to full, authentic benchmark validation:

1. **VRSBench:**
   - Download the official Hugging Face dataset (`xiang709/VRSBench`) or official release from `github.com/lx709/VRSBench` (requiring ~35 GB for full resolution tiles).
   - Ingest official test split JSON annotations (`vrsbench_test.json`).
2. **RSVQA:**
   - Download official RSVQA-LR (Sentinel-2, ~5 GB) or RSVQA-HR (aerial, ~30 GB) datasets from the official Zenodo / Univ. of Wageningen repositories.
   - Point `RSVQADataset` in `ml_pipeline/data/rsvqa_loader.py` to the authentic uncompressed imagery and official `questions.json` / `answers.json`.
3. **CDVQA:**
   - Download official CDVQA dataset release (4,000 bi-temporal pairs, ~15 GB) from the authors' official distribution repository.
   - Place true bi-temporal pairs and ground-truth binary change masks into `ml_pipeline/dataset_cache/cdvqa/`.
4. **ISRO / SAC Cartosat-2S & RISAT-1:**
   - Obtain officially distributed, sensor-calibrated Cartosat-2S (0.65m) and RISAT-1 (C-band) GeoTIFF tiles from ISRO/SAC coordinators.
   - Obtain or annotate expert-verified reference ground-truth masks for urban built-up, hydrological bodies, and vegetation canopy.
   - Ingest official domain-expert query-answer pairs for regional remote-sensing assessment.
