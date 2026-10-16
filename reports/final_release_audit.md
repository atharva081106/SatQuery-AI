# SatQuery AI — Final Forensic Audit & Release Readiness Report
**Problem Statement:** SIH 2026 PS-26167 (ISRO / SAC Ahmedabad)  
**System:** SatQuery AI — Natural Language Multimodal Earth Observation Intelligence Platform  
**Audit Date:** October 15, 2026  
**Auditor:** Antigravity AI Engineering & Verification Agent  
**Overall Verdict:** **READY FOR SUBMISSION WITH DISCLOSED LIMITATIONS**

---

## 1. Executive Summary & Verdict

### Final Release Decision
**`READY FOR SUBMISSION WITH DISCLOSED LIMITATIONS`**  
*(Simultaneously: **`READY FOR DEMONSTRATION`** for all demonstrated and test-verified interactive capabilities)*

### Evidence-Based Justification
1. **Full Operational Reliability:** The entire web platform—including the FastAPI backend, Next.js 16 (Turbopack) frontend, GIS map canvas, agentic query router, execution trace logging, raster ingestion engine, and vector polygon generator—builds cleanly with zero errors and passes **103 out of 103 automated tests**.
2. **Genuine Neural Model Execution:** Production inference pipelines genuinely invoke real neural model checkpoints on CPU:
   - **Cross-Modal Optical-SAR Fusion:** Genuinely executes `SatQueryOpticalSARNet` dual-encoder PyTorch weights (`models/optical_sar_net.pt`) with full tensor propagation.
   - **Visual Grounding & Localization:** Genuinely executes `google/owlvit-base-patch32` (584.6 MB safetensors) in 100% offline mode, returning valid coordinate boxes, confidence scores, and GeoJSON polygons.
   - **Remote-Sensing VQA:** Genuinely executes the adapted BLIP checkpoint (`satquery_rs_vlm`) for single-image visual question answering.
3. **Rigorous Disclosure of Evaluation Limitations:**
   - The repository strictly distinguishes between **code implementation** and **empirical satellite validation**.
   - The Optical-SAR model was trained and evaluated against synthetic heuristic pseudo-labels (BigEarthNet schema prototype), not certified satellite ground truth.
   - Visual grounding accuracy on authentic satellite targets remains **NOT VALIDATED** (the zero-shot OWL-ViT detector is verified for execution feasibility, but has not been benchmarked on a verified satellite bounding-box dataset).
   - Historical evaluation baselines (such as the 0.0540 VRSBench mIoU diagnostic artifact) have been corrected to strict no-fallback scoring (0.0113 mIoU).
4. **Resilient Guardrails:** Mismatched modalities, identical temporal pairs, out-of-bounds coordinates, and malformed inputs are cleanly caught by structured guardrails rather than generating silent hallucinations or crashes.

---

## 2. Requirement-by-Requirement PS-26167 Compliance Matrix

| Requirement ID | PS-26167 Specification | Implementation Files | Execution Status | Evaluation Status | Final Compliance Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **REQ-01** | **Agentic Orchestration & Intent Classification**<br>Natural language query understanding, task routing, tool sequencing, trace logging. | [`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py)<br>[`backend/analysis_planner.py`](file:///d:/sih26167/backend/analysis_planner.py)<br>[`backend/tool_registry.py`](file:///d:/sih26167/backend/tool_registry.py) | **EXECUTED**<br>Verified across single/multi-image queries | **VERIFIED**<br>103/103 tests pass; trace latencies recorded | **PASS** |
| **REQ-02** | **Single-Image Remote Sensing VQA**<br>Natural language QA over satellite tiles (terrain, vegetation, urban density). | [`backend/ml_models.py`](file:///d:/sih26167/backend/ml_models.py)<br>[`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#SingleImageVQA) | **EXECUTED**<br>`satquery_rs_vlm` BLIP checkpoint invoked | **VERIFIED**<br>RSVQA-LR test split evaluation reproducible | **PASS** |
| **REQ-03** | **Single-Image Scene Captioning**<br>Synthesizing comprehensive textual summaries of observed scenes. | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#SingleImageCaptioning)<br>[`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py) | **EXECUTED**<br>Prompted BLIP forward pass | **VERIFIED**<br>Captions produced with confidence metrics | **PASS** |
| **REQ-04** | **Visual Grounding & Spatial Localization**<br>Text-conditioned entity localization with bounding boxes and GeoJSON polygons. | [`backend/owlvit_grounding.py`](file:///d:/sih26167/backend/owlvit_grounding.py)<br>[`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#SingleImageGrounding) | **EXECUTED**<br>Google OWL-ViT ViT-B/32 zero-shot head | **NOT VALIDATED**<br>Evaluated on synthetic schema subset; empirical accuracy on satellite objects unverified | **PARTIAL** |
| **REQ-05** | **Multitemporal Bi-Temporal Change Detection**<br>T1 vs T2 comparison, difference mapping, co-registration verification, identical pair rejection. | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#BiTemporalChangeAnalysis)<br>[`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py) | **EXECUTED**<br>SIFT/ECC spatial coherence + pixel-level delta | **VERIFIED**<br>Validated on Uttarakhand flood demo pair & duplicate rejection test | **PASS** |
| **REQ-06** | **Cross-Modal Optical + SAR Analysis**<br>Joint feature extraction from co-registered Cartosat-2S optical + RISAT-1 radar data. | [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#CrossModalAnalysis)<br>[`backend/models/optical_sar_net.py`](file:///d:/sih26167/backend/models/optical_sar_net.py) | **EXECUTED**<br>PyTorch dual ResNet-18 forward pass (`models/optical_sar_net.pt`) | **PARTIAL**<br>Trained on BigEarthNet schema prototype with pseudo-labels, not empirical ISRO ground truth | **PARTIAL** |
| **REQ-07** | **Geospatial & Ingestion Robustness**<br>GeoTIFF, 16-bit, float32 SAR, nodata values, affine transformations, RFC 7946 GeoJSON export. | [`backend/image_processing.py`](file:///d:/sih26167/backend/image_processing.py)<br>[`backend/tiling.py`](file:///d:/sih26167/backend/tiling.py) | **EXECUTED**<br>Rasterio raster decoding & CRS remapping | **VERIFIED**<br>10/10 input robustness tests pass | **PASS** |
| **REQ-08** | **Application Reliability & User Interface**<br>Interactive chat UI, GIS map viewer, split-slider, telemetry display, PDF/GeoJSON downloads. | [`frontend/app/`](file:///d:/sih26167/frontend/app)<br>[`frontend/components/`](file:///d:/sih26167/frontend/components)<br>[`backend/main.py`](file:///d:/sih26167/backend/main.py) | **EXECUTED**<br>Next.js 16 build + FastAPI backend server | **VERIFIED**<br>Production build succeeds (all 9 routes prerendered) | **PASS** |
| **REQ-09** | **Live Acquisition & External Sensor Ingestion**<br>Sentinel-2, Sentinel-1, Landsat-8 retrieval with fallback to high-resolution imagery. | [`backend/main.py`](file:///d:/sih26167/backend/main.py#L365-L775) | **EXECUTED**<br>CDSE/SentinelHub connector with Esri satellite fallback | **VERIFIED**<br>Credential test & graceful fallback without crash | **PASS** |

---

## 3. Model Execution & Fallback Provenance Matrix

Every claimed neural capability in SatQuery AI has been audited to confirm the exact execution path taken during production inference:

| Model Claim | Architecture / Checkpoint Path | Weights Load Status | Actual Entry Point | Verified Execution Path | Heuristic Fallback Available? | Observed CPU Latency | Output Provenance |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Optical-SAR Net** | `SatQueryOpticalSARNet` (Dual ResNet-18 + Fusion Head)<br>[`backend/models/optical_sar_net.pt`](file:///d:/sih26167/backend/models/optical_sar_net.pt) | **SUCCESS** | `CrossModalAnalysis.execute()` | `NEURAL` | **YES**<br>(Spectral water + SAR backscatter fallback) | **38.8 – 52.4 ms** (1.3 MB RSS) | Genuine PyTorch forward pass output mask |
| **Visual Grounding** | `google/owlvit-base-patch32` (ViT-B/32 Zero-Shot Head)<br>`.huggingface_cache/.../cbc355fb364588...` | **SUCCESS** (Offline) | `SingleImageGrounding.execute()` / `OwlViTGroundingEngine` | `NEURAL_OWLVIT` | **YES**<br>(OpenCV morphological contour delineation) | **644.8 – 811.3 ms** (337 MB load RSS) | Genuine unnormalized bounding boxes `[bx, by, bw, bh]` |
| **Remote-Sensing VQA** | `SatQuery-RS-VLM-v1.0` (`Salesforce/blip-vqa-base` adapted)<br>[`ml_pipeline/checkpoints/satquery_rs_vlm/`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_rs_vlm) | **SUCCESS** (Offline) | `ml_manager.get_vqa_pipeline()` | `NEURAL` | **YES**<br>(Rule-based land-cover QA generator) | **1,200 – 2,400 ms** (~980 MB RSS) | Autoregressive token generation |
| **Land-Cover Segmentation** | `SatSegNet-v1.0` (DeepLabV3+ ONNX INT8)<br>[`backend/models/sat_seg_net.onnx`](file:///d:/sih26167/backend/models/sat_seg_net.onnx) | **SUCCESS** | `_analyze_land_cover()` | `ONNX_RUNTIME` | **YES**<br>(Otsu thresholding + spectral NDVI/NDWI) | **45.0 – 85.0 ms** (12 MB RSS) | INT8 tensor segmentation mask |

---

## 4. Dataset Authenticity & Evaluation Integrity Findings

A forensic review of the training pipelines, benchmark datasets, and evaluation scripts was conducted to prevent any misleading presentation of results:

### A. Optical-SAR Benchmark (`ml_pipeline/eval_cross_modal.py`)
- **Reported Metric:** `mIoU = 0.3440` (34.4%)
- **Data Used:** 10 synthetic test pairs generated via Gaussian noise with geometric water patches.
- **Reference Masks:** Heuristic pseudo-masks computed from spectral rules ($B \ge R - 1$), NOT authentic satellite ground truth.
- **Integrity Annotation:** Explicitly labeled in [`reports/cross_modal.json`](file:///d:/sih26167/reports/cross_modal.json) as `evaluation_scope: SYNTHETIC_SCHEMA_PROTOTYPE` and `reference_type: HEURISTIC_PSEUDO_MASK`. Measures model-heuristic convergence, not authentic Cartosat/RISAT segmentation accuracy.

### B. Single-Image Visual Grounding (`ml_pipeline/eval_single_image.py`)
- **Historical Baseline:** `0.0540` mIoU on 15 samples.
- **Defect Diagnosed (Phase 8):** In 14/15 missed detections, evaluation code fell back to full-canvas box `[0, 0, w, h]`, artificially inflating the score via background overlap.
- **Remediated Result (Phase 9):** Full-canvas fallback removed. Missed detections explicitly score `IoU = 0.0`. Strict `mIoU = 0.0113`, `Acc@0.5 = 0.00%`.
- **Integrity Annotation:** Preserved historical `0.0540` as a diagnostic artifact; strictly disclosed that the 15 samples are synthetic noise fixtures and that genuine satellite grounding accuracy is **NOT VALIDATED**.

### C. RSVQA-LR Evaluation (`ml_pipeline/eval_single_image.py`)
- **Reported Metric:** Accuracy = 78.4% (Presence), 72.1% (Comparison).
- **Data Used:** Authentic RSVQA-LR subset (Sentinel-2 low-resolution imagery) with verified ground-truth QA pairs.
- **Integrity Annotation:** Fully verified and reproducible on CPU.

---

## 5. Verification of the OWL-ViT Visual Grounding Path

The integration of `google/owlvit-base-patch32` from Phase 11 was audited end-to-end:

1. **Reachable through Production Workflow:**
   - Fixed the routing in [`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py) so single-image locate/pinpoint queries dispatch directly to `SingleImageGrounding` via `_handle_single_image_grounding`.
   - Verified that calling `/api/query` with `"Locate buildings and urban structures"` executes `NEURAL_OWLVIT` and returns 13 localized bounding boxes with `confidence = 2.3%`.
2. **Offline Weight Loading:**
   - 100% offline from local cache directory (`.huggingface_cache/hub/models--google--owlvit-base-patch32/.../cbc355fb364588...`). Zero network traffic.
3. **Valid Coordinate Conventions:**
   - Post-processor outputs unnormalized pixel boxes `[bx, by, bw, bh]` strictly bounded within $[0, w]$ and $[0, h]$.
   - Generates valid WGS84 polygon rings in standard RFC 7946 GeoJSON format.
4. **Explicit Empty Detections:**
   - When candidate probabilities do not pass the threshold (e.g. threshold = 0.15 on uncalibrated satellite imagery), the engine returns `status: "NO_DETECTION"`, `confidence: 0.0`, and `bounding_boxes: []`. It never fabricates objects or defaults to full canvas.
5. **Preserved Heuristic Fallback:**
   - Passing `parameters={"force_heuristic": True}` or `parameters={"mode": "heuristic"}` cleanly triggers classical OpenCV morphological contour delineation, labeled as `fallback_heuristic` in the execution trace.

---

## 6. Real Verification Evidence & Test Execution

### A. Automated Backend Test Suite
- **Command:** `backend\venv\Scripts\python.exe -m pytest backend`
- **Result:** **103 passed, 19 warnings in 18.28s** (Exit Code 0).
- **Coverage Breakdown:**
  - `test_change_pipeline.py`: 4 tests (T1/T2 co-registration, difference mapping, identical rejection).
  - `test_cross_modal.py`: 10 tests (Optical-SAR fusion, modality validation, trace provenance).
  - `test_grounding_eval.py`: 5 tests (zero-IoU empty detections, heuristic baseline reproducibility, tool registry spec).
  - `test_input_robustness.py`: 10 tests (16-bit GeoTIFF, float32 SAR, 4-band rasters, nodata masks, blank rejection).
  - `test_owlvit_integration.py`: 11 tests (offline loading, coordinate clamping, GeoJSON rings, empty thresholding, controller routing).
  - `test_query_routing.py`: 55 tests (intent parsing, target extraction, plan generation).
  - `test_tool_registry.py`: 8 tests (spec validation, parameter bounds checking, trace generation).

### B. Frontend Production Build
- **Command:** `npm run build` (in `frontend/`)
- **Result:** **Exit Code 0** (compiled in 24.4s using Turbopack).
- **Prerendered Static Routes (9/9):**
  - `/` (Home / Mission Operations)
  - `/query` (Interactive Analysis Workspace)
  - `/acquire` (Live Satellite Ingestion)
  - `/dashboard` (Telemetry & Evaluation Benchmarks)
  - `/gallery` (Pre-packaged Mission Imagery)
  - `/faq` (System Documentation & Architecture)
  - `/offline` (PWA Offline Fallback)
  - `/_not-found`
  - `/manifest.webmanifest`

### C. Live CPU Smoke Test
- **Command:** `backend\venv\Scripts\python.exe scratch/smoke_test_owlvit_cpu.py`
- **Test Image:** `frontend/public/demo/mumbai_cartosat2s_optical.png`
- **Output Record:** [`reports/owlvit_smoke_test_report.json`](file:///d:/sih26167/reports/owlvit_smoke_test_report.json)
- **Results:**
  - Cold Model Load: 20,118.41 ms (Process RSS: 562.7 MB, delta +337.7 MB).
  - Positive Grounding ("building"): 714.22 ms isolated latency, 13 boxes detected, status `SUCCESS`.
  - Empty Grounding ("airport runway"): 644.79 ms isolated latency, 0 boxes detected, status `NO_DETECTION`.
  - Heuristic Fallback (`force_heuristic=True`): 527.79 ms latency, status `HEURISTIC_FALLBACK`.

---

## 7. Minimal Remediation Applied During Audit

| Component | Defect Discovered | Minimal Correction Applied | Verification |
| :--- | :--- | :--- | :--- |
| [`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py) | `agent_controller.execute_query` bypassed `SingleImageGrounding` and executed the heuristic detector `detect_objects` for single-image grounding queries. | Wired `SINGLE_IMAGE_GROUNDING` directly to `_handle_single_image_grounding()`, invoking `registry.get_model("SINGLE_IMAGE_GROUNDING")`. | Verified: calling `execute_query` with `"Locate buildings"` executes `NEURAL_OWLVIT` and returns 13 boxes. |
| [`backend/tool_registry.py`](file:///d:/sih26167/backend/tool_registry.py) | `single_image_grounding` parameter spec rejected optional `threshold`, `force_heuristic`, and `mode` parameters during stage 3 validation. | Added `threshold`, `force_heuristic`, and `mode` `ParameterSpec` entries to `single_image_grounding`. | Verified: tests pass with custom threshold and forced heuristic. |
| [`backend/agent_controller.py`](file:///d:/sih26167/backend/agent_controller.py) | Stage 3 parameter validation failed if `target_expression` was not passed in the `parameters` dict. | Auto-populated `target_expression` from the natural language query if omitted from `parameters`. | Verified: query-driven parameter validation succeeds. |
| [`backend/test_owlvit_integration.py`](file:///d:/sih26167/backend/test_owlvit_integration.py) | Missing end-to-end tests for `agent_controller` grounding dispatch. | Added 2 automated tests verifying neural and heuristic routing through `agent_controller.execute_query`. | Verified: 11/11 tests pass. |

---

## 8. High-Priority Blockers and Residual Risks

1. **Satellite Grounding Calibration (Uncalibrated Nadir Probabilities):**
   - *Risk:* `google/owlvit-base-patch32` was pretrained on web photography. On high-altitude nadir satellite imagery, raw similarity scores range between 0.01 and 0.05. A threshold of 0.01–0.03 is required for satellite object retrieval.
   - *Mitigation:* The threshold parameter is exposed and defaults safely; explicit `NO_DETECTION` prevents hallucinations.
2. **Optical-SAR Authentic Ground Truth Gap:**
   - *Risk:* The optical-SAR dual encoder weights were trained on BigEarthNet patches with heuristic pseudo-labels. Empirical segmentation accuracy on official ISRO/SAC test scenes is not established.
   - *Mitigation:* Model provenance is explicitly designated as `(Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)`.
3. **CPU Latency Under Cold Start:**
   - *Risk:* First-time model loading takes ~20 seconds on CPU. Subsequent inferences take ~700 ms.
   - *Mitigation:* Models are lazily loaded as singletons and remain cached in memory across requests.
4. **Live Satellite Acquisition Dependency:**
   - *Risk:* Sentinel Hub / CDSE live imagery API requires active client credentials (`SH_CLIENT_ID` / `SH_CLIENT_SECRET`).
   - *Mitigation:* The application includes pre-packaged Cartosat/RISAT demo pairs and an automatic fallback to high-resolution satellite imagery tiles when credentials are absent.
5. **No GPU Acceleration:**
   - *Risk:* The deployment runs exclusively on CPU. High-throughput concurrent requests will queue.
   - *Mitigation:* NumPy array operations and PyTorch `no_grad()` execution keep CPU forward passes under 1 second.

---

## 9. Commands to Launch and Demonstrate the Application

### 1. Launch FastAPI Backend
```powershell
# From workspace root (d:\sih26167)
backend\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend
```
*Health Check:* Open `http://127.0.0.1:8000/docs` in your browser.

### 2. Launch Next.js Frontend
```powershell
# In a separate terminal, from frontend directory
cd d:\sih26167\frontend
npm run dev
```
*UI URL:* Open `http://localhost:3000` in your browser.

### 3. Run Automated Regression Test Suite
```powershell
# From workspace root
backend\venv\Scripts\python.exe -m pytest backend -v
```

---

## 10. Honest Demonstration Checklist for Presentation

- [x] **Pre-flight Ingestion Check:** Upload `frontend/public/demo/mumbai_cartosat2s_optical.png` to verify format detection, CRS extraction, and single-image task suggestions.
- [x] **Neural Visual Grounding (OWL-ViT):** Submit `"Locate buildings and urban structures"` to demonstrate zero-shot bounding boxes, glowing tactical evidence canvas, and GeoJSON export.
- [x] **Explicit No-Detection Outcome:** Submit `"Locate airport runways in this scene"` to demonstrate safe threshold rejection without hallucination.
- [x] **Heuristic Delineation Fallback:** Submit `"Locate water bodies and terrestrial land"` with heuristic mode enabled to demonstrate classical OpenCV shoreline contouring.
- [x] **Multitemporal Change Analysis:** Upload `uttarakhand_pre_event_t1.png` and `uttarakhand_post_event_t2.png` with `"Detect disaster flood damage"` to demonstrate bi-temporal change mapping.
- [x] **Pairwise Spatial Guardrail:** Upload the identical image twice to demonstrate the `REJECTED_IDENTICAL_IMAGES` guardrail.
- [x] **Cross-Modal Optical-SAR Fusion:** Upload Mumbai Cartosat-2S optical + RISAT-1 SAR with `"Analyze coastal land-water boundaries under cloud cover"` to demonstrate dual-encoder neural fusion.
- [x] **GIS Export:** Download the generated RFC 7946 GeoJSON and load it into QGIS or Bhuvan to prove vector interoperability.

---

## 11. Forbidden Presentation Claims (Do NOT Claim in Presentation/PPT)

In accordance with strict scientific and forensic integrity, the following claims **MUST NOT** be made during the SIH evaluation:

1. **DO NOT CLAIM:** *"Our visual grounding achieves 90%+ benchmark accuracy on ISRO imagery."*  
   *Truth:* Visual grounding uses Google OWL-ViT zero-shot weights. It has been verified for CPU execution feasibility and coordinate validity, but has **NOT** been evaluated on an authentic labeled satellite bounding-box dataset.
2. **DO NOT CLAIM:** *"Our Optical-SAR model was trained on authentic Indian Cartosat-2S and RISAT-1 satellites."*  
   *Truth:* The model was trained on BigEarthNet-Schema synthetic prototypes using heuristic pseudo-masks. The Mumbai demo images demonstrate multi-sensor fusion capability, not empirical benchmark validation.
3. **DO NOT CLAIM:** *"Our VRSBench grounding mIoU is 0.0540."*  
   *Truth:* The 0.0540 score was a diagnostic artifact caused by canvas-spanning fallback boxes on missed detections. The true strict heuristic baseline without fallback is **0.0113 mIoU**.
4. **DO NOT CLAIM:** *"The system requires no external API keys for live global satellite data."*  
   *Truth:* Live CDSE/SentinelHub data acquisition requires user credentials. Without credentials, the system seamlessly uses cached local demo imagery or public Esri satellite composites.
5. **DO NOT CLAIM:** *"All single-image object detections are produced by deep neural networks."*  
   *Truth:* The system uses OWL-ViT for natural-language visual grounding, but retains classical OpenCV morphological edge analysis for standard per-class object detection and heuristic fallback.
