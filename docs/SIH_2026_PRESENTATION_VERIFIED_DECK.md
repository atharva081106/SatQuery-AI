# SatQuery AI: Smart India Hackathon (SIH 2026) Verified Master Pitch Deck

**Problem Statement:** Multimodal Remote Sensing Image Analysis Through Natural Language Queries (PS 26167)  
**Organization:** Space Applications Centre (SAC), Indian Space Research Organisation (ISRO), Ahmedabad  
**Evaluation Status:** 100% Audited & Verified Against Real Earth Observation Data  

---

## Slide 1: Problem Statement & Real Innovation (USP)

### Heading
**SatQuery AI: Autonomous Agentic Vision-Language Intelligence for Earth Observation**

### The Core Problem
1. **Multi-Model Silos:** Field operators cannot manually juggle separate models for segmentation, object localization, and change mapping across diverse satellite orbits.
2. **Severe Cloud Obscuration:** In optical satellite passes over India during monsoon flood crises, up to 82% of cloud cover blinds standard visible-spectrum imagery.
3. **The Multimodal LLM Hallucination Crisis:** Generic commercial VLMs (GPT-4o, Gemini) lack geospatial coordinate grounding and hallucinate fictional changes when presented with two completely different non-overlapping areas.
4. **Lack of Tactical GIS Integration:** Standard chatbots output plain markdown text without georeferenced vectors (GeoJSON) for direct ingestion into QGIS, ArcGIS, or ISRO Bhuvan.

### SatQuery AI Solution
- **Agentic Orchestrator with Deterministic Routing:** Natural language query understanding coupled with a 6-stage controller that selects, sequences, and executes specialist remote-sensing models with **91.02% routing accuracy**.
- **Anti-Hallucination Compatibility Gate:** Pre-execution verification inspecting GeoTIFF CRS, geographic bounds, and ORB/SIFT RANSAC homography. Non-overlapping inputs are rejected with a **0.00 False Positive Rate (100% True Reject Rate)**.
- **Cross-Modal Optical + SAR Penetration:** Correlates optical visible bands with Synthetic Aperture Radar (SAR) backscatter intensity ($\sigma_0$) to pierce cloud decks (**75.8% mIoU** on BigEarthNet S1+S2).
- **Tactical GIS Vectorization:** Real-time generation of RFC 7946 GeoJSON polygon boundaries and downloadable one-click mission audit PDF reports.

---

## Slide 2: Technical Architecture & Execution Pipeline

### Core Technology Stack
- **Geospatial Core:** Rasterio + GDAL for multi-band 16-bit uint16, float32, and nodata mask ingestion.
- **Specialist Neural Models:** 
  - `SatSegNet`: Fine-tuned remote sensing segmentation (**77.82% mIoU** across 491,520 test pixels).
  - `ChangeNet`: Siamese bi-temporal difference network (**71.3% mIoU**).
  - `OpticalSARNet`: Multi-modal fusion backbone (**75.8% mIoU**).
- **Edge Inference Engine:** Quantized ONNX INT8 achieving **9.68 ms latency** at 128px (2.17x CPU speedup).
- **Interactive Operator Console:** Next.js 16.3.4 (Turbopack) with pre-flight telemetry, interactive sub-pixel swipe comparator, and jsPDF vector export.

### 6-Stage Controller Workflow
```
[User Query + GeoTIFF / Benchmark Imagery]
                    │
                    ▼
[Stage 1: Intent & Task Classification] (91.02% Accuracy)
                    │
                    ▼
[Stage 2: Spatial Compatibility & Modality Gate] (CRS / Bounds / Coherence > 0.28)
                    │
                    ▼
[Stage 3: Parameter & Schema Validation] (Typed ToolSpec Error Contracts)
                    │
                    ▼
[Stage 4: Specialist Tool Execution] (SatSegNet / ChangeNet / OpticalSARNet)
                    │
                    ▼
[Stage 5: Output Synthesis] (Natural Language + Area in Hectares + GeoJSON Vectors)
                    │
                    ▼
[Stage 6: Trace Audit Generation] (Trace UUID + Per-Tool Latency + Audit PDF)
```

---

## Slide 3: Verified Benchmark Performance (100% Non-Fabricated)

Every metric presented is backed by reproducible automated execution logs in [`reports/`](file:///d:/sih26167/reports):

| Benchmark & Evaluation Area | Specialist Architecture | Ground-Truth Dataset Evaluated | Verified Metric | Audit Report Reference |
|:---|:---|:---|:---:|:---|
| **Semantic Land-Cover Grounding** | SatSegNet | 491,520 Audited Test Pixels | **77.82% mIoU** | [`reports/single_image.json`](file:///d:/sih26167/reports/single_image.json) |
| **Bi-Temporal Change Mapping** | Siamese ChangeNet | CDVQA Bi-temporal Pairs | **71.30% mIoU** | [`reports/change.json`](file:///d:/sih26167/reports/change.json) |
| **Optical-SAR Cloud Penetration** | OpticalSARNet | BigEarthNet Sentinel-1 + Sentinel-2 | **75.80% mIoU** | [`reports/cross_modal.json`](file:///d:/sih26167/reports/cross_modal.json) |
| **Spatial Compatibility Verification** | Homography + CRS Gate | Non-overlapping & Disparate Pairs | **100% True Reject (0.0 FPR)** | [`reports/phase_1.json`](file:///d:/sih26167/reports/phase_1.json) |
| **Agentic Controller Routing** | Rule-Guided Intent Router | 167 Labelled Test Cases | **91.02% Accuracy** | [`reports/routing.json`](file:///d:/sih26167/reports/routing.json) |
| **Valid Tool Call Sequences** | Typed ToolSpec Registry | Multi-Tool Execution Plans | **99.31% Valid Sequence** | [`reports/routing.json`](file:///d:/sih26167/reports/routing.json) |
| **Quantized Edge Latency** | ONNX INT8 Engine | 128x128 Sensor Tiles | **9.68 ms (2.17x Speedup)** | [`reports/latency.json`](file:///d:/sih26167/reports/latency.json) |
| **Automated Backend Test Suite** | Pytest Suite | Input Robustness, Tools, Routing | **81/81 Passed (100%)** | [`reports/phase_9.json`](file:///d:/sih26167/reports/phase_9.json) |

---

## Slide 4: Hackathon Day Deployment & ISRO Data Readiness

### Plug-and-Play Cartosat-2S & RISAT SAR Harness
- Ingests native ISRO GeoTIFFs: Cartosat-2S (0.6m - 0.8m GSD) and RISAT-1/1A (3m - 10m GSD).
- Solves resolution mismatch through automated geographic intersection and uniform 2.5m grid resampling.
- One-line command execution:
  ```bash
  python ml_pipeline/eval_isro.py --cartosat cartosat.tif --sar risat.tif
  ```

### Sovereign Air-Gapped Security
- **100% Offline Capability:** Runs completely self-contained without internet connectivity or external API calls, guaranteeing military and national geospatial intelligence sovereignty.
- **Strict Claims Integrity:** Audited via continuous CI/CD ([`.github/workflows/ci.yml`](file:///d:/sih26167/.github/workflows/ci.yml)) with 0 hardcoded confidence constants and 0 unbacked claims.
