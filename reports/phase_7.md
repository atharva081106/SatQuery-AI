# Phase 7 Verification Report: Robust Input Handling & Performance Telemetry

**Generated:** 2026-10-14  
**Problem Statement Alignment:** SIH26167 (ISRO / Space Applications Centre)  
**Status:** PASSED  
**Latency Benchmark:** [reports/latency.json](file:///d:/sih26167/reports/latency.json)  

---

## 1. Executive Summary

Phase 7 hardens SatQuery AI for operational Earth observation ingestion:
1. **Scientific GeoTIFF Support**: Ingests multi-band, 16-bit (`uint16`), 32-bit floating point (`float32`), and multi-spectral satellite imagery (Sentinel-2, Landsat-8, Cartosat-2S, RISAT-1) via `rasterio`.
2. **Radiometric & Nodata Normalization**: Employs robust 2%-98% cumulative percentile contrast stretching excluding nodata/NaN values; transforms linear microwave radar backscatter ($\sigma_0$) to decibels (dB).
3. **Large-Image Sliding Window Tiling Engine**: Implements [backend/tiling.py](file:///d:/sih26167/backend/tiling.py) (`SlidingWindowTiler`) featuring 2D Hann / Cosine apodization window boundary blending, probability mask stitching, and cross-tile Non-Maximum Suppression (NMS) on bounding boxes.
4. **Benchmark Mode Format Policy**: Strictly rejects standard unreferenced PNG/JPEG rasters with `UNSUPPORTED_FORMAT` when operational georeferenced mode is active, while permitting them when the benchmark toggle is enabled.
5. **Real Latency Profiling**: Measured p50, p95, and throughput QPS for every specialist tool and tiling operation without mocked constants.

---

## 2. Ingestion & Preprocessing Verification

| Ingestion Capability | Test Scenario | Verified Behavior | Status |
| :--- | :--- | :--- | :--- |
| **16-bit GeoTIFF (`uint16`)** | Synthetic 3-band Cartosat/Sentinel-2 raster (DN 500..8000) | Extracted CRS (`EPSG:4326`), bounds, and 2%-98% stretched to uint8 without clipping | **PASSED** |
| **Float32 SAR Backscatter** | Linear radar backscatter $\sigma_0$ GeoTIFF | Log-transformed to dB and scaled to [0, 255] RGB display channels | **PASSED** |
| **Multi-band Selection** | 4-band / 12-band Multi-spectral GeoTIFF | Automatically selects Red, Green, Blue bands based on spectral band headers | **PASSED** |
| **Nodata Handling** | GeoTIFF with `nodata=0` in corner regions | Nodata values cleanly masked to background 0 without corrupting percentile stretch | **PASSED** |
| **Blank Raster Guardrail** | All-zero / uniform pixel stream | Cleanly caught and rejected with descriptive `ValueError` | **PASSED** |
| **Corrupt Stream Guardrail** | Truncated byte stream | Caught and rejected with `ValueError` | **PASSED** |
| **Benchmark Toggle** | Unreferenced PNG in non-benchmark mode | Rejected with `StructuredErrorCode.UNSUPPORTED_FORMAT` | **PASSED** |

---

## 3. Large Raster Tiling Engine ([backend/tiling.py](file:///d:/sih26167/backend/tiling.py))

For satellite scenes exceeding GPU/CPU memory boundaries (e.g., 2048x2048 to 10k x 10k+):
- **Window Grid Generation**: Slices image into 512x512 tiles with 64px overlap, guaranteeing 100% spatial coverage.
- **Hann Margin Blending**: Uses continuous cosine tapering across tile margins to eliminate abrupt boundary seam artifacts.
- **Cross-Tile Non-Maximum Suppression (NMS)**: Re-projects tile-relative bounding boxes into global coordinates and suppresses duplicates spanning overlap zones.
- **Telemetry on 2048x2048 Scene**: Completed 25-tile slicing and stitching in **229.29 ms** (p50) while keeping working set memory below 512 MB.

---

## 4. Real Latency Telemetry Benchmark ([reports/latency.json](file:///d:/sih26167/reports/latency.json))

Measured across repeated iterations on host hardware (Intel 12-core CPU, PyTorch 2.13):

| Remote Sensing Specialist Tool | Mean Latency (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | Throughput (QPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Image VQA** | 17.71 | **15.01** | 29.17 | 13.32 | 35.70 | **56.46** |
| **Single Image Captioning** | 14.13 | **13.72** | 16.35 | 13.30 | 19.10 | **70.79** |
| **Single Image Grounding** | 13.26 | **13.24** | 14.13 | 12.55 | 14.18 | **75.44** |
| **Land Cover Segmentation** | 14.97 | **13.76** | 21.31 | 12.25 | 25.42 | **66.80** |
| **Bi-Temporal Change Detection** | 50.55 | **51.76** | 58.05 | 40.26 | 58.35 | **19.78** |
| **Optical-SAR Cross-Modal Fusion** | 7.16 | **7.07** | 8.59 | 5.98 | 8.60 | **139.75** |
| **Spatial Compatibility Verifier** | 28.54 | **27.76** | 34.66 | 25.95 | 35.84 | **35.04** |
| **Large Raster Tiling (2048x2048)** | 236.86 | **229.29** | 274.93 | 215.93 | 291.60 | **4.22** |
| **Raster Ingestion & Decode** | 13.31 | **13.29** | 14.58 | 11.86 | 14.65 | **75.13** |

---

## 5. Verification & Regression Pass Rate

- **Phase 7 Test Suite:** [backend/test_input_robustness.py](file:///d:/sih26167/backend/test_input_robustness.py) (**10 / 10 passed**)
- **Full Backend Regression Suite:**
  - `backend/test_input_robustness.py`: 10/10 PASSED
  - `backend/test_tool_registry.py`: 7/7 PASSED
  - `backend/test_change_pipeline.py`: 4/4 PASSED
  - `backend/test_cross_modal.py`: 5/5 PASSED
  - `backend/test_query_routing.py`: 55/55 PASSED
  - **Total:** **81 / 81 passed (100.0%)**
- **Claims Auditor ([scripts/audit_claims.py](file:///d:/sih26167/scripts/audit_claims.py))**: **0 violations found** (no hardcoded confidence literals, 12 verified reports indexed).
