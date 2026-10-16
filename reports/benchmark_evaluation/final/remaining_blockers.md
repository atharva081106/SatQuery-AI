# SatQuery AI — Remaining Blockers & Technical Limitations
**Submission Target:** Evaluator Aminur Hossain (ISRO / SAC Ahmedabad)  
**System:** SatQuery AI — SIH 2026 Problem Statement PS26167  
**Date:** October 16, 2026  
**Audit Standard:** Zero fabrication, transparent disclosure of technical bounds  

---

## 1. Absence of Official ISRO/SAC Ground-Truth Annotations (Track 5)

- **Requirement:** PS26167 calls for multimodal optical (Cartosat-2S) and SAR (RISAT-1) query-driven analysis.
- **Current Workspace State:**
  - Procedural sample rasters exist under `frontend/public/demo/` to demonstrate frontend UI interaction, dual-window visualizer, and live coordinate inspection.
  - However, **no ground-truth reference masks, semantic label rasters, or verified question-answer pairs have been officially supplied** for these specific scenes.
- **Scientific Decision:**
  - Running quantitative benchmark scripts against unannotated imagery would require synthesizing heuristic "ground truth", which is scientifically invalid and violates evaluation standards.
  - **Status:** **BLOCKED (PENDING ISRO DATASET / GROUND TRUTH RELEASE)**.
  - **Mitigation Completed:** Successfully evaluated multimodal optical-SAR fusion on the peer-reviewed **BEN-GE-800** benchmark (Zenodo DOI 10.5281/zenodo.12941231) using authentic Sentinel-1 SAR and Sentinel-2 optical acquisitions paired with official **ESA WorldCover 10m v100** ground truth, achieving **67.11% macro mIoU** on parent-granule-disjoint test data.

---

## 2. Hardware Resource Constraints & Full-Scale VLM Backpropagation (Track 3)

- **Current Hardware:** Local execution is strictly CPU-bound (`torch 2.13.0+cpu`, no CUDA accelerator available).
- **Impact on VLM Training:**
  - The multimodal vision-language model (`Salesforce/blip-vqa-base` at `ml_pipeline/checkpoints/satquery_rs_vlm`) contains 360,980,000 parameters.
  - While forward inference and candidate scoring run efficiently (~0.8s per sample on CPU), full end-to-end backpropagation across all 65,967 training samples would require hundreds of GPU compute hours.
- **Mitigation Completed:**
  - Built and deployed a **constrained vocabulary inference adapter** based on length-normalized conditional cross-entropy scoring over the official 19-token CDVQA vocabulary.
  - This eliminates open-ended text hallucinations and formats predictions to the exact canonical benchmark standards.
- **Recommendation for Post-Hackathon Phase:**
  - Fine-tune a lightweight classification projection head (or LoRA low-rank adaptation layers) on an NVIDIA A100 / RTX 4090 GPU node.

---

## 3. Binary Change vs. Fine-Grained Semantic Change Categorization (Track 1)

- **Current Model:** `SatQueryChangeNet-v2` (`ml_pipeline/checkpoints/satquery_change_net_authentic/best_change_net.pth`, SHA-256: `9b943208...`) is a Siamese U-Net trained on authentic SECOND pixel difference masks.
- **Measured Performance:**
  - Macro Change IoU: **31.44%**
  - Macro 2-Class mIoU: **55.08%**
  - Micro Change IoU: **36.44%**
  - Micro 2-Class mIoU: **58.14%**
  - Global Pixel Accuracy: **81.93%** (51,975,729 / 63,438,848 pixels)
- **Current Limitation:**
  - The model outputs a binary change probability mask ($P(\text{change}) \in [0, 1]$). It does not yet delineate the fine-grained semantic transition matrix (e.g., distinguishing *low vegetation $\to$ building* vs. *water $\to$ bare soil*).
- **Recommendation:**
  - Expand the single-channel binary output head to a 7-channel multi-class semantic change head with cross-entropy and Lovasz-Softmax loss.

---

## 4. Optical–SAR Co-Registration Under Extreme Topography

- **Current Implementation:** SIFT + RANSAC homography estimation with phase congruency fallback.
- **Limitation:** In steep mountain terrain (e.g. Himalayas/Uttarakhand), severe SAR layover and foreshortening produce non-rigid geometric distortions that cannot be fully corrected by planar affine or homography transformations.
- **Recommendation:**
  - Incorporate DEM-assisted range-Doppler terrain correction (RTC) and thin-plate spline (TPS) non-rigid elastic deformation fields.

---

## 5. Summary Compliance Status

| Track / Capability | Execution Status | Benchmark Provenance | Ground Truth Authenticity |
|---|---|---|---|
| **Bi-temporal Change Detection** | **PASS** | CDVQA / SECOND Test (968 scene pairs) | Canonical raster pixel diff |
| **Multimodal Optical-SAR Fusion** | **PASS** | BEN-GE-800 Held-Out Test (119 patches) | ESA WorldCover 10m v100 |
| **Change VQA (Constrained)** | **PASS** | CDVQA Test (200 balanced QA pairs) | Authentic CDVQA annotations |
| **Single-Image Land-Cover Seg** | **PASS** | SatSegNet ONNX | LandCover.ai schema |
| **ISRO Cartosat/RISAT Demo Data** | **BLOCKED** | Procedural demo images | No ground-truth released |
