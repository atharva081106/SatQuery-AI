# SatQuery AI — Multimodal Remote Sensing & Earth Observation Intelligence

**Smart India Hackathon (SIH) 2026**  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC)  
**Problem Statement:** PS 26167 — Multimodal Remote Sensing Image Analysis  

---

## 🛰️ Overview

SatQuery AI is an agentic, vision-language intelligence system engineered for automated analysis of remote sensing data, including optical/multispectral imagery, Synthetic Aperture Radar (SAR), and bi-temporal acquisitions. 

Unlike generic multimodal LLMs that hallucinate spatial boundaries and fail on multi-temporal rasters, SatQuery AI couples a deterministic orchestrator and remote-sensing-adapted vision models (`SatSegNet`, `ChangeNet`, `OpticalSARNet`) with strict spatial compatibility verification gates.

---

## 📋 Honest Problem Statement Mapping & Verified Metrics

Every capability required by ISRO/SAC Problem Statement 26167 is implemented, verified through automated test suites, and logged to persistent JSON reports in [`reports/`](file:///d:/sih26167/reports):

| Requirement | Implementation Files | Loaded Specialist Model vs Heuristic | Verified Measured Metric | Report Source |
|:---|:---|:---|:---|:---|
| **1. Multi-Format & Deep-Bit Raster Ingestion** | `backend/model_interfaces.py`<br>`backend/tiling.py` | Rasterio GDAL driver with nodata masking and adaptive percentile scaling | Ingestion of 16-bit uint16 Sentinel-2 and float32 SAR backscatter rasters | [`reports/phase_7.json#input_handling.geotiff_ingestion`](file:///d:/sih26167/reports/phase_7.json) |
| **2. Sliding Window Tiling & Blending** | `backend/tiling.py`<br>`backend/test_input_robustness.py` | `TilerStitcher` with 2D Gaussian window blending | 128px tile overlap, 0 edge boundary seam artifacts | [`reports/phase_7.json#input_handling.edge_case_guardrails`](file:///d:/sih26167/reports/phase_7.json) |
| **3. Spatial Compatibility Verification** | `backend/model_interfaces.py`<br>`backend/test_change_pipeline.py` | Geospatial CRS intersection + ORB/SIFT RANSAC homography matcher | 100% True Reject Rate on non-overlapping pairs (0.0 FPR) | [`reports/cross_modal.json#co_registration_guardrail_metrics.guardrail_verification_rate_pct`](file:///d:/sih26167/reports/cross_modal.json) |
| **4. Single-Image VQA & Semantic Grounding** | `backend/agent_controller.py`<br>`backend/satsegnet_infer.py`<br>`ml_pipeline/sat_seg_model.py` | **Primary:** SatSegNet (6-class custom U-Net) + Adapted RS-VLM<br>**Fallback:** Classical HSV + contour heuristic (`fallback_heuristic`) | VRSBench VQA: 76.67% Acc; RSVQA: 36.11% Acc; GeoJSON vector export (Authentic ISRO grounding pending) | [`reports/single_image.json#summary.vrsbench_vqa_accuracy`](file:///d:/sih26167/reports/single_image.json)<br>[`reports/single_image.json#datasets.rsvqa.overall_accuracy_pct`](file:///d:/sih26167/reports/single_image.json) |
| **5. Bi-Temporal Change Detection** | `backend/tool_change_detection.py`<br>`ml_pipeline/change_net.py` | **Primary:** SatQueryChangeNet (Siamese U-Net)<br>**Fallback:** SSIM + Otsu thresholding (`fallback_heuristic`) | CDVQA: 15.16% change mask mIoU, 80.0% recall, 100% existence; ha/sq km trends (Cartosat T1/T2 pending) | [`reports/phase_4.json#cdvqa_evaluation.mask_miou_pct`](file:///d:/sih26167/reports/phase_4.json)<br>[`reports/phase_4.json#cdvqa_evaluation.mask_recall`](file:///d:/sih26167/reports/phase_4.json)<br>[`reports/phase_4.json#cdvqa_evaluation.per_type_metrics.existence.exact_accuracy_pct`](file:///d:/sih26167/reports/phase_4.json) |
| **6. Optical-SAR Cross-Modal Fusion** | `backend/tool_cross_modal.py`<br>`ml_pipeline/optical_sar_net.py` | **Primary:** SatQueryOpticalSARNet-v1.0 (Dual-Encoder Prototype)<br>**Fallback:** HSV-SAR backscatter correlation (`fallback_heuristic`) | 24.93% synthetic schema prototype mIoU against heuristic targets; 100% guardrail on demo pairs (Authentic ISRO pair pending) | [`reports/cross_modal.json#land_cover_extraction_metrics.overall_mean_iou_pct`](file:///d:/sih26167/reports/cross_modal.json)<br>[`reports/cross_modal.json#co_registration_guardrail_metrics.guardrail_verification_rate_pct`](file:///d:/sih26167/reports/cross_modal.json) |
| **7. Agentic Controller & Tool Registry** | `backend/agent_controller.py`<br>`backend/tool_registry.py`<br>`tests/routing_cases.jsonl` | Deterministic Intent Router + Typed `ToolSpec` Registry | 91.02% routing accuracy on 167 ground-truth cases; 99.31% valid sequence rate | [`reports/routing.json#metrics.task_classification_accuracy_pct`](file:///d:/sih26167/reports/routing.json)<br>[`reports/routing.json#metrics.valid_sequence_rate_pct`](file:///d:/sih26167/reports/routing.json) |
| **8. Edge Acceleration & Profiling** | `backend/benchmark_latency.py`<br>`backend/tiling.py` | PyTorch CPU baseline vs Quantized ONNX INT8 runtime | 9.68 ms @ 128px INT8 (2.17x CPU speedup); 54.8 ms @ 256px to 726.8 ms @ 1024px full pipeline | [`reports/latency.json#benchmarks.onnx_int8`](file:///d:/sih26167/reports/latency.json) |
| **9. Interactive GUI & Audit Export** | `frontend/src/app/query/page.tsx`<br>`frontend/src/app/dashboard/page.tsx` | Next.js 16.3.4 (Turbopack) + jsPDF client-side export | Pre-flight telemetry modal, benchmark mode toggle, formula popover, PDF audit export | [`reports/phase_8.json#metrics.telemetry_sync`](file:///d:/sih26167/reports/phase_8.json) |
| **10. Automated CI & Claims Integrity** | `.github/workflows/ci.yml`<br>`scripts/audit_claims.py` | Pytest test suite + AST/regex Claims Auditor | 81/81 tests passing (100%); 0 claims audit violations; 14 verified reports | [`reports/phase_9.json#metrics.backend_test_suite.pass_rate_percent`](file:///d:/sih26167/reports/phase_9.json) |

For the detailed requirement-by-requirement scoring matrix, review [`reports/final_matrix.md`](file:///d:/sih26167/reports/final_matrix.md).

---

## 🔒 Rigorous Engineering Principles Enforced

1. **Zero Hardcoded Confidence Literals:** Every confidence value emitted by backend endpoints is dynamically computed from model softmax activations, keypoint spatial coherence, and validation agreement. (`scripts/audit_claims.py` verifies `backend/` contains 0 hardcoded confidence literals).
2. **Transparent Labeling of Classical Routines:** When deep neural networks are unavailable or offline, classical OpenCV routines are labeled explicitly as `fallback_heuristic` in logs, traces, and frontend status chips.
3. **Automated Continuous Integration:** A complete GitHub Actions workflow (`.github/workflows/ci.yml`) runs all backend tests, executes the claims auditor, validates report integrity, and tests the Next.js production build on every push and pull request.

---

## 🚀 Quickstart & Hackathon Execution

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) CUDA-enabled GPU for accelerated neural model inference

### 1. Backend Setup & Startup
```powershell
# Navigate to backend directory
cd backend

# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start backend server (FastAPI on port 8000)
python main.py
```

### 2. Frontend Setup & Startup
```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies and start development server
npm install
npm run dev
# Open http://localhost:3000 in your browser
```

### 3. Run Automated Tests & Claims Audit
```powershell
# Run the complete test suite (81 tests)
.\backend\venv\Scripts\pytest.exe backend/test_input_robustness.py backend/test_tool_registry.py backend/test_change_pipeline.py backend/test_cross_modal.py backend/test_query_routing.py -v

# Run Claims & Provenance Auditor (verifies 0 unbacked claims)
.\backend\venv\Scripts\python.exe scripts/audit_claims.py

# Run Next.js production build verification
cd frontend
npm run build
```

---

## 📁 Repository Architecture

```
sih26167/
├── backend/
│   ├── main.py                     # FastAPI server with /health, /api/query, /api/preflight
│   ├── agent_controller.py         # Deterministic orchestrator and query intent classifier
│   ├── tool_registry.py            # Typed tool registry (ToolSpec, ToolExecutionRecord)
│   ├── model_interfaces.py         # Spatial compatibility check, format and modality detection
│   ├── tiling.py                   # Sliding window tiler and 2D Gaussian stitcher
│   ├── tool_change_detection.py    # Siamese bi-temporal change detection tool
│   ├── tool_cross_modal.py         # Optical-SAR cross-modal fusion tool
│   ├── satsegnet_infer.py          # Land-cover segmentation inference engine
│   └── test_*.py                   # Complete pytest suite (81 tests)
├── ml_pipeline/
│   ├── sat_seg_model.py            # SatSegNet architecture
│   ├── change_net.py               # Siamese ChangeNet architecture
│   ├── optical_sar_net.py          # OpticalSARNet architecture
│   └── checkpoints/                # Model weights (SatSegNet, ChangeNet, OpticalSARNet)
├── frontend/
│   ├── src/app/query/page.tsx      # Main tactical console with pre-flight telemetry, trace, and PDF export
│   ├── src/app/dashboard/page.tsx  # Ground-truth telemetry dashboard displaying audited metrics
│   └── src/components/             # Visual components (SwipeSlider, GeoJSONMapOverlay, etc.)
├── reports/                        # Verified persistent JSON & MD benchmark reports (Phases 1-9)
├── scripts/
│   └── audit_claims.py             # AST/regex Claims & Provenance Auditor
├── tests/
│   └── routing_cases.jsonl         # 167 ground-truth query routing benchmark cases
└── .github/workflows/
    └── ci.yml                      # GitHub Actions CI workflow
```

---

## ⚡ Hackathon Day Evaluation: Cartosat-2S & RISAT SAR Harness

For the live hackathon jury evaluation, SatQuery AI provides a dedicated plug-and-play evaluation harness [`ml_pipeline/eval_isro.py`](file:///d:/sih26167/ml_pipeline/eval_isro.py) designed specifically for ISRO/SAC data:
- **Optical Ingestion:** Cartosat-2S Panchromatic/Multispectral sub-meter GeoTIFFs (~0.6m - 0.8m GSD)
- **SAR Ingestion:** RISAT-1 / RISAT-1A (EOS-04) C-Band circular/hybrid polarization GeoTIFFs (~3m - 10m GSD)
- **Resolution Resampling:** Automatically calculates geographic boundary intersections and resamples both rasters to a common 2.5m grid (Bilinear for optical spectral reflectance, Nearest-Neighbor for SAR backscatter amplitudes).

```powershell
# Run the ISRO evaluation harness with authentic observation rasters
python ml_pipeline/eval_isro.py --optical path/to/cartosat.tif --sar path/to/risat.tif --output reports/isro_evaluation.json
```

*(Note: Default evaluation without arguments runs on procedural synthetic demo pairs in `frontend/public/demo/` for regression testing. Authentic Cartosat-2S and RISAT GeoTIFFs should be supplied via `--optical` and `--sar` during jury evaluation.)*

---

## ☁️ Running on Free Tiers (Google Colab & Kaggle)

The training and evaluation pipelines support low-resource hardware constraints (free Google Colab T4 or Kaggle P100 GPUs):
- **Sample Limiter:** Pass `--max-samples <N>` (e.g. `--max-samples 100`) to test any pipeline without downloading multi-gigabyte archives.
- **CPU Fallback:** All specialist pipelines automatically detect CUDA availability and fall back to PyTorch CPU or quantized ONNX INT8 with zero code changes.

```bash
# Example low-resource evaluation on Colab/Kaggle
python ml_pipeline/eval_single_image.py --max-samples 50
python ml_pipeline/eval_change.py --max-samples 50
python ml_pipeline/eval_cross_modal.py --max-samples 50
```

---

## ⚠️ Known Limitations & Operational Boundaries

1. **Unreferenced Non-GeoTIFFs:** PNG/JPEG images lack embedded geospatial coordinate reference systems (CRS) and affine transforms; they are evaluated in local pixel space rather than WGS84 geographic coordinates.
2. **Optical-SAR Unassisted Co-Registration:** When georeferencing is missing from both optical and SAR rasters, mutual information / gradient structure similarity is used as a fallback with reduced confidence. Native GeoTIFFs with valid CRS are strongly recommended.
3. **CPU Inference Latencies:** Running full unquantized Vision Transformer models on host CPU requires ~2-5 seconds per query; enabling the quantized ONNX INT8 runtime reduces single-tile latency to 9.68 ms.

---

## 👨‍🏫 Mentorship & Acknowledgements

Developed for the Smart India Hackathon (SIH) 2026 under the problem statement defined by:  
**Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad**  
*Mentors:* Aminur Hossain, Sanjay K Singh, S Devakanth Naidu  
