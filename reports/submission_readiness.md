# SatQuery AI — Phase 7: SIH Evidence & Submission-Readiness Audit Report

**Problem Statement:** ISRO / Space Applications Centre (SAC) — Multimodal Remote Sensing Image Analysis (PS 26167)  
**Evaluation Scope:** Read-Only Verification of Repository Implementation, Executed Benchmarks, Metric Integrity, and Hackathon-Day Evaluation Readiness  
**Date of Audit:** October 2026  
**Overall Readiness Verdict:** **PARTIAL (Pipeline Architecture & Agentic Engine Complete; Empirical Satellite Validation Pending Authentic Data)**

---

## 1. Executive Summary

This forensic audit evaluates the SatQuery AI repository against the official SIH 2026 PS26167 guidelines, which stipulate a blind evaluation on undisclosed test subsets containing pre-georeferenced, co-registered Cartosat-2S optical and RISAT C-band SAR pairs with task-specific reference masks and question-answer pairs.

### Summary of System Status
1. **Architectural & Pipeline Readiness:** **HIGH (87/87 Unit & Integration Tests Passing)**. The multi-modal agentic query router ([backend/agent_controller.py](file:///d:/sih26167/backend/agent_controller.py)), sliding window tiler ([backend/tiling.py](file:///d:/sih26167/backend/tiling.py)), 16-bit GeoTIFF decoder ([backend/model_interfaces.py](file:///d:/sih26167/backend/model_interfaces.py)), SIFT/RANSAC and NMI spatial guardrails, and typed `ToolSpec` registry ([backend/tool_registry.py](file:///d:/sih26167/backend/tool_registry.py)) are fully implemented and functional.
2. **Empirical Validation on Authentic ISRO Imagery:** **PENDING (ZERO AUTHENTIC SATELLITE SAMPLES EVALUATED)**. The repository contains zero authentic georeferenced Cartosat-2S or RISAT-1 GeoTIFF rasters with authentic ground-truth labels. The evaluated metrics in `reports/*.json` originate exclusively from small synthetic demo images or academic dataset loaders operating on simulated in-memory sample subsets.
3. **Honesty & Integrity Reconciliation:** All previously unbacked marketing numbers (`77.82% mIoU`, `71.3% mIoU`, `75.8% mIoU`, and `"BigEarthNet trained"`) have been audited, purged, and replaced with real tested prototype metrics and explicit disclosures.

---

## 2. Requirement-by-Requirement Evidence Table

| Requirement & SIH Track | Exact Source Implementation | Executed Test / Benchmark Evidence | Dataset & Sample Count | Ground-Truth Labels Used? | Authenticity Classification | Status | Remaining Blocker & Next Minimum Action |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **REQ-01: Deep-Bit Raster Ingestion** *(Track 6)* | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py)<br>[`backend/tiling.py`](file:///d:/sih26167/backend/tiling.py) | [`backend/test_input_robustness.py`](file:///d:/sih26167/backend/test_input_robustness.py)<br>[`reports/phase_7.json`](file:///d:/sih26167/reports/phase_7.json) | Synthetic in-memory GeoTIFFs (16-bit uint16, float32, 4-band) | Yes (programmatic nodata & min-max bounds) | `SYNTHETIC (In-Memory GeoTIFF)` | **PASS** | **Blocker:** Tested only on programmatic rasters.<br>**Next Action:** Test against authentic multi-gigabyte L1/L2 Cartosat-2S GeoTIFF with real RPC/GCP metadata. |
| **REQ-02: Sliding Window Tiler** *(Track 6)* | [`backend/tiling.py`](file:///d:/sih26167/backend/tiling.py) (`SlidingWindowTiler`) | [`backend/test_input_robustness.py`](file:///d:/sih26167/backend/test_input_robustness.py)<br>[`reports/latency.json`](file:///d:/sih26167/reports/latency.json) | 2048×2048 synthetic canvas, 512×512 tiles, 128px overlap | Yes (spatial reconstruction error < 1e-4) | `SYNTHETIC (Synthetic Canvas)` | **PASS** | **Blocker:** None for algorithmic tiling.<br>**Next Action:** Profile memory usage on 10,000×10,000 full-swath scenes. |
| **REQ-03: Spatial Compatibility Verifier** *(Track 6, Reliability)* | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py) (`check_spatial_compatibility`) | [`backend/test_cross_modal.py`](file:///d:/sih26167/backend/test_cross_modal.py)<br>[`backend/test_change_pipeline.py`](file:///d:/sih26167/backend/test_change_pipeline.py) | 5 disparate demo pairs, 1 identical pair, 2 true demo pairs | Yes (expected binary pass/reject decision) | `DEMO IMAGERY (frontend/public/demo)` | **PASS** | **Blocker:** SIFT/NMI thresholds calibrated on demo PNGs, not unreferenced raw GeoTIFFs.<br>**Next Action:** Validate NMI threshold (0.010) on authentic co-registered Cartosat-RISAT pairs. |
| **REQ-04: Single-Image VQA & Grounding** *(Track 1 & Track 2)* | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py)<br>[`ml_pipeline/sat_seg_model.py`](file:///d:/sih26167/ml_pipeline/sat_seg_model.py)<br>[`ml_pipeline/eval_single_image.py`](file:///d:/sih26167/ml_pipeline/eval_single_image.py) | [`reports/single_image.json`](file:///d:/sih26167/reports/single_image.json) | RSVQA (36 samples), VRSBench VQA (30 samples), VRSBench Grounding (15 samples) | Yes (reference QA text answers and bounding boxes) | `ACADEMIC BENCHMARK SUBSETS` | **PARTIAL** | **Blocker:** Grounding accuracy is 0.0% (mean IoU 0.054) on VRSBench; no Cartosat-2S single-tile VQA/grounding pairs evaluated.<br>**Next Action:** Adapt bounding box proposal head on remote sensing bounding box datasets. |
| **REQ-05: Bi-Temporal Change Detection** *(Track 3, Change)* | [`backend/tool_change_detection.py`](file:///d:/sih26167/backend/tool_change_detection.py)<br>[`ml_pipeline/change_model.py`](file:///d:/sih26167/ml_pipeline/change_model.py)<br>[`ml_pipeline/eval_change.py`](file:///d:/sih26167/ml_pipeline/eval_change.py) | [`reports/change.json`](file:///d:/sih26167/reports/change.json)<br>[`backend/test_change_pipeline.py`](file:///d:/sih26167/backend/test_change_pipeline.py) | CDVQA (15 test pairs); Uttarakhand pre/post demo PNGs | Yes (binary change masks and CDVQA question answers) | `ACADEMIC BENCHMARK SUBSET (CDVQA)` | **PARTIAL** | **Blocker:** Change mask mIoU is only 15.16% (recall 80%); no authentic Cartosat-2S T1/T2 disaster/urban pairs evaluated.<br>**Next Action:** Fine-tune ChangeNet on real satellite change pairs (e.g., LEVIR-CD or SpaceNet). |
| **REQ-06: Optical-SAR Cross-Modal Fusion** *(Track 4, Weighted 25%)* | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py)<br>[`ml_pipeline/cross_modal_model.py`](file:///d:/sih26167/ml_pipeline/cross_modal_model.py)<br>[`ml_pipeline/eval_cross_modal.py`](file:///d:/sih26167/ml_pipeline/eval_cross_modal.py)<br>[`ml_pipeline/eval_isro.py`](file:///d:/sih26167/ml_pipeline/eval_isro.py) | [`reports/cross_modal.json`](file:///d:/sih26167/reports/cross_modal.json)<br>[`reports/isro_evaluation.json`](file:///d:/sih26167/reports/isro_evaluation.json)<br>[`backend/test_cross_modal.py`](file:///d:/sih26167/backend/test_cross_modal.py) | BigEarthNet-schema synthetic loader (15 pairs); Mumbai + RISAT demo PNGs | Pseudo-masks (`generate_spatial_targets`); NO ground-truth in ISRO demo stub | `SYNTHETIC SCHEMA PROTOTYPE & DEMO STUB` | **PARTIAL** | **Blocker:** mIoU (24.93%) measures agreement with synthetic heuristic rules, NOT real ground-truth segmentation. Real ISRO evaluation has 0 reference masks.<br>**Next Action:** Run `eval_isro.py` with authentic jury-supplied Cartosat-2S and RISAT GeoTIFFs. |
| **REQ-07: Agentic Controller & Tool Registry** *(Track 6, Core Brain)* | [`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py)<br>[`backend/tool_registry.py`](file:///d:/sih26167/backend/tool_registry.py) | [`reports/routing.json`](file:///d:/sih26167/reports/routing.json)<br>[`backend/test_query_routing.py`](file:///d:/sih26167/backend/test_query_routing.py)<br>[`backend/test_tool_registry.py`](file:///d:/sih26167/backend/test_tool_registry.py) | 167 labelled natural language queries in [`tests/routing_cases.jsonl`](file:///d:/sih26167/tests/routing_cases.jsonl) | Yes (expected task classification, tool sequence, and parameters) | `EXPLICIT DOMAIN GROUND TRUTH (167 cases)` | **PASS** | **Blocker:** None for routing.<br>**Next Action:** Add edge cases for multi-turn conversational disambiguation. |
| **REQ-08: Latency & Scaling Telemetry** *(Track 6, Hardware)* | [`backend/benchmark_latency.py`](file:///d:/sih26167/backend/benchmark_latency.py)<br>[`ml_pipeline/profile_latency.py`](file:///d:/sih26167/ml_pipeline/profile_latency.py) | [`reports/latency.json`](file:///d:/sih26167/reports/latency.json) | 15 iterations across all 8 specialist tools on 12-core host CPU | Yes (millisecond wall-clock execution timer) | `MEASURED HOST CPU TELEMETRY` | **PASS** | **Blocker:** ONNX INT8 benchmarks require ONNX Runtime export script for cross-modal pipeline.<br>**Next Action:** Export `SatQueryOpticalSARNet` to ONNX INT8 format. |
| **REQ-09: Mission GUI, Telemetry & Export** | [`frontend/src/app/query/page.tsx`](file:///d:/sih26167/frontend/src/app/query/page.tsx)<br>[`frontend/src/app/dashboard/page.tsx`](file:///d:/sih26167/frontend/src/app/dashboard/page.tsx) | Live interactive Next.js application, PDF export generator | Multi-tab UI with 6 live telemetry cards and dynamic report inspector | Yes (JSON report schema consistency) | `INTERACTIVE PRODUCTION RUNTIME` | **PASS** | **Blocker:** Build bypasses TypeScript errors via `ignoreBuildErrors: true`.<br>**Next Action:** Resolve Next.js frontend ESLint/TypeScript warnings. |
| **REQ-10: Automated CI & Claims Integrity** | [`.github/workflows/ci.yml`](file:///.github/workflows/ci.yml)<br>[`scripts/audit_claims.py`](file:///d:/sih26167/scripts/audit_claims.py) | Pytest test runner (18/18 passed in `test_cross_modal` / `test_tool_registry`, 55/55 in `test_query_routing`) | AST/regex check across `backend/`, `frontend/`, `reports/`, and `README.md` | Yes (zero prohibited claims, zero hardcoded confidences) | `AUTOMATED AUDIT RUNTIME` | **PASS** | **Blocker:** GitHub Actions CI sets `ENABLE_ML_MODELS: "false"`.<br>**Next Action:** Configure CPU-only PyTorch model loading in CI workflow. |

---

## 3. Rubric-Weighted Gap Summary (Official SIH 2026 PS26167 Tracks)

The table below maps the official evaluation tracks from [docs/SIH_2026_PRESENTATION_MASTER_DECK.txt](file:///d:/sih26167/docs/SIH_2026_PRESENTATION_MASTER_DECK.txt#L144-L154) to actual repo readiness and estimated rubric weights:

| Track / Capability | Official Rubric Weight | Key Implemented Features | Verified Real Benchmark Score | Status on Blind ISRO Evaluation |
| :--- | :---: | :--- | :--- | :---: |
| **Track 1: Single-Image VQA & Feature Recon** | 15% | BLIP-VQA adapter, OpenCV HSV spectral discriminator, Grad-CAM overlays | VRSBench VQA: **76.67% Acc**<br>RSVQA: **36.11% Acc** | **PARTIAL** (VQA prompt logic solid; high-resolution optical feature extraction untested on authentic Cartosat-2S). |
| **Track 2: Target Grounding & Localization** | 15% | Connected-component contour extractor, RFC 7946 GeoJSON polygon generator | VRSBench Grounding: **0.0% Acc@0.5** (mIoU **0.0540**) | **FAIL** (Grounding head fails on academic benchmarks; bounding boxes for runways/tanks are coarse contour estimates). |
| **Track 3: Bi-Temporal Change Detection** | 20% | Siamese ChangeNet (1.7M params), SSIM + Otsu difference engine, hectare quantification | CDVQA: **15.16% mIoU**, **80.0% Recall**, **100% Existence** | **PARTIAL** (Existence detection works perfectly; exact pixel change boundary delineation is weak at 15.16% mIoU). |
| **Track 4: Optical-SAR Cross-Modal Fusion** | **25%** | `SatQueryOpticalSARNet` (Dual-Encoder), Radiometric NMI & gradient coherence guard, IHS fusion | Prototype mIoU: **24.93%** (on synthetic schema against heuristic targets) | **PARTIAL** (Guardrail works with 100% reject rate; neural fusion model has never seen real Cartosat-RISAT pairs). |
| **Track 5: Public Benchmark Generalization** | 15% | Dataset loaders for BigEarthNet, CDVQA, RSVQA, VRSBench with sample limiters | VQA Acc: **76.67%** (VRSBench)<br>Existence: **100%** (CDVQA) | **PARTIAL** (Full multi-gigabyte archives were not downloaded; tests run on small cached subsets). |
| **Track 6: System Reliability & Guardrails** | 10% | Spatial disparity rejection gate, typed `ToolSpec` registry, dynamic confidence derivation | Routing Acc: **91.02%**<br>Reject FPR: **0.00**<br>P50 Latency: **7.07 ms** | **PASS** (Zero hallucination on disparate inputs; deterministic query parsing and sub-second execution). |

---

## 4. Deep-Dive Forensic Audits

### 4.1 Single-Image VQA, Captioning, and Grounding
- **Inference Path:** When a single image is submitted with a query (e.g., *"Highlight water bodies and identify dense vegetation"*), [backend/agent_controller.py](file:///d:/sih26167/backend/agent_controller.py#L350-L420) dispatches to `AnalysisPlanner`. If `ENABLE_ML_MODELS` is true and fine-tuned weights exist at `ml_pipeline/satquery_finetuned_model/`, `BlipForQuestionAnswering` executes. Otherwise, `fallback_heuristic` executes using OpenCV HSV color thresholding (`backend/model_interfaces.py:_analyze_land_cover`).
- **Grounding Discrepancy:** While single-image classification and general VQA perform reasonably on VRSBench (76.67% exact/lexical match), grounding localization is severely uncalibrated: [reports/single_image.json](file:///d:/sih26167/reports/single_image.json#L58-L61) records `acc_at_0_5_pct: 0.0%` and `mean_iou: 0.054` across 15 VRSBench grounding questions.
- **Cause:** SatQuery does not use an explicit object detection anchor network (like YOLO or Faster R-CNN); it attempts spatial grounding by thresholding BLIP cross-attention or OpenCV contours.

### 4.2 Bi-Temporal Change Detection and Mask Metrics
- **Inference Path:** Multi-image inputs route through [backend/model_interfaces.py:check_spatial_compatibility](file:///d:/sih26167/backend/model_interfaces.py#L220). If identical images are provided, it cleanly rejects with `REJECTED_IDENTICAL_IMAGES` (`confidence: 0.04`). If disparate images are provided, SIFT keypoints with RANSAC homography yield an inlier count < 15 and inlier ratio < 0.30, cleanly blocking execution with `FAILED` (`confidence <= 0.35`).
- **Neural Model:** For true pairs (e.g. `uttarakhand_pre_event_t1.png` vs `uttarakhand_post_event_t2.png`), `SatQueryChangeNet` executes a forward pass on concatenated feature maps.
- **Metric Verification:** In [reports/change.json](file:///d:/sih26167/reports/change.json), the evaluated mask mIoU is **15.16%** with an F1 Dice score of **0.2549**. While the network achieved a high recall of **80.0%**, its precision was only **15.16%**, meaning it significantly over-segments change regions on complex terrain.

### 4.3 Co-Registered Optical–SAR Fusion (SIH 25% Criterion)
- **Modality Identification:** Optical vs SAR modality is determined by [backend/model_interfaces.py:detect_image_modality](file:///d:/sih26167/backend/model_interfaces.py#L110). If GeoTIFF metadata contains `POLARIZATION` or `BAND_NAMES` with HH/HV/VV/VH, or if filename contains `risat` or `sar`, it is classified as `SAR`.
- **Cross-Modal Co-Registration:** SIFT keypoints fail on Optical-SAR due to radiometric inversion. SatQuery uses Normalized Mutual Information (NMI >= 0.010) and Sobel Gradient Directional Correlation (>= 0.000). The Mumbai true pair scores **0.88** (passed); the disparate pair scores **0.04** (rejected).
- **Evaluation Status:** The reported 24.93% mIoU in [reports/cross_modal.json](file:///d:/sih26167/reports/cross_modal.json) was computed against synthetic pseudo-masks generated by `generate_spatial_targets()`. In [reports/isro_evaluation.json](file:///d:/sih26167/reports/isro_evaluation.json), default execution is explicitly documented as a `SYNTHETIC_DEMO_STUB` with `reference_masks_evaluated: false` and `qa_answers_evaluated: false`. **Empirical accuracy on authentic ISRO observations remains unverified.**

### 4.4 Agentic Routing, Input Handling, and Infrastructure
- **Query Understanding & Routing:** Tested by [backend/test_query_routing.py](file:///d:/sih26167/backend/test_query_routing.py) (55/55 passed). Evaluated on 167 labelled queries: **91.02% task classification accuracy**, **99.31% valid sequence rate**, and **90.91% invalid call rejection rate**.
- **GeoTIFF Ingestion:** Tested by [backend/test_input_robustness.py](file:///d:/sih26167/backend/test_input_robustness.py) (10/10 passed). Correctly scales 16-bit uint16 reflectance (using 2nd-98th percentiles) and handles float32 linear SAR backscatter.
- **Confidence Calibration:** Confidences are computed dynamically from softmax probabilities, spatial coherence scores, and land-cover area fractions. `scripts/audit_claims.py` proves **0 hardcoded confidence literals** exist in `backend/`.
- **Latency Profiling:** P50 median latencies on 12-core host CPU:
  - Optical-SAR Fusion: **7.07 ms** (139.75 QPS)
  - Land Cover Segmentation: **13.76 ms**
  - Bi-Temporal Change Detection: **51.76 ms**
  - Large Raster Tiling (2048×2048): **229.29 ms**

---

## 5. Metric Denominators & Reconciliation Audit

| Documented Claim (Pre-Audit) | Reconciled Ground-Truth Metric | Tested Denominator & Sample Count | Producing Artifact | Source of Discrepancy & Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **"77.82% mIoU (SatSegNet)"** | **76.67% VQA Acc**<br>(mIoU 0.054 grounding) | 30 VRSBench VQA questions, 15 grounding bounding boxes | [`reports/single_image.json`](file:///d:/sih26167/reports/single_image.json) | The 77.82% number had no backing report file on disk. The actual tested model achieved 76.67% VQA accuracy on VRSBench. Documentation was updated. |
| **"71.30% mIoU (ChangeNet)"** | **15.16% mask mIoU**<br>(80% recall, 40% exact QA) | 15 CDVQA bi-temporal test pairs | [`reports/change.json`](file:///d:/sih26167/reports/change.json) | 71.30% was an unverified claim. Actual CDVQA test evaluation outputs 15.16% mask mIoU and 80.0% recall. Claims were purged. |
| **"75.80% mIoU (OpticalSARNet)"** | **24.93% prototype mIoU** | 15 synthetic schema test pairs against heuristic targets | [`reports/cross_modal.json`](file:///d:/sih26167/reports/cross_modal.json) | 75.80% had no backing report file. Real script execution on synthetic schema test set yielded 24.93% agreement with heuristic pseudo-labels. |
| **"9.68 ms Latency"** | **7.07 ms P50 Latency** | 15 iterations on host CPU across 8 specialist pipeline tools | [`reports/latency.json`](file:///d:/sih26167/reports/latency.json) | 9.68 ms was an uncalibrated constant. Actual CPU profiling revealed 7.07 ms P50 for optical-SAR fusion and 13.76 ms for land-cover segmentation. |

---

## 6. Critical Blockers for Hackathon Jury Evaluation

1. **Jury Evaluation Uses Authentic Cartosat-2S & RISAT GeoTIFFs (Blind Dataset):**
   - *Risk:* If the judging panel uploads authentic multi-spectral Cartosat-2S GeoTIFFs (with sub-meter GSD) alongside RISAT-1 C-band SLC/GRD GeoTIFFs, the system must parse multi-band GeoTIFF geotransforms, verify spatial intersection, and resample without dying.
   - *Current Status:* Handled gracefully by [`ml_pipeline/eval_isro.py`](file:///d:/sih26167/ml_pipeline/eval_isro.py) via `--optical` and `--sar` CLI flags, but segmentation quality on real satellite backscatter has never been visually validated.
2. **Target Grounding Localization (Track 2) is Underperforming:**
   - *Risk:* Queries asking to *"Locate all oil storage tanks in the northern quadrant"* will emit GeoJSON bounding boxes, but bounding box accuracy against tight ground-truth annotations will score low (VRSBench grounding mIoU is 0.054).
3. **CI Pipeline Does Not Test PyTorch Weights:**
   - *Risk:* In GitHub Actions CI, `ENABLE_ML_MODELS` is set to `"false"` to save memory, meaning automated tests only exercise `fallback_heuristic` classical OpenCV code paths.
4. **Change Detection Precision Over-Segmentation:**
   - *Risk:* `SatQueryChangeNet` exhibits high recall (80%) but low precision (15.16%), producing false change positives on seasonal vegetation or sun glint.

---

## 7. Prioritized Next-Action List (Pre-Submission)

1. **High Priority — Dry-Run `eval_isro.py` on Real GeoTIFF Rasters:**
   - Obtain or prepare at least one real pair of georeferenced Cartosat-2S (or Sentinel-2) and RISAT-1 (or Sentinel-1 SAR) GeoTIFFs with real EPSG coordinates.
   - Execute:
     ```powershell
     python ml_pipeline/eval_isro.py --optical path/to/real_optical.tif --sar path/to/real_sar.tif --output reports/isro_evaluation.json
     ```
   - Verify that geographic coordinate boundary intersection, resampling to 2.5m, and IHS fusion complete without memory overflow or format errors.
2. **Medium Priority — Calibrate SIFT/NMI Thresholds for High-Resolution Data:**
   - Ensure the NMI threshold (`0.010`) and gradient correlation threshold (`0.000`) in `check_spatial_compatibility()` do not inadvertently reject real Cartosat/RISAT scenes having high speckle noise.
3. **Medium Priority — Enable Neural Smoke Test in CI:**
   - Add a lightweight CPU-only smoke test in `.github/workflows/ci.yml` that sets `ENABLE_ML_MODELS="true"` and tests 1 forward pass through `SatQueryOpticalSARNet` and `SatQueryChangeNet`.
4. **Low Priority — Frontend Cleanliness:**
   - Resolve ESLint warnings in `frontend/src/` to allow removing `ignoreBuildErrors: true` from `frontend/next.config.ts`.
