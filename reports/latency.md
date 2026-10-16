# SatQuery AI: Performance & Latency Profiling Report

**Generated:** 2026-10-14 18:00:23  
**Hardware Platform:** Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (CUDA: False, Cores: 12)  
**Operating System:** Windows 10  
**PyTorch Version:** 2.13.0+cpu  
**Status:** PASSED  

---

## 1. Executive Summary

In operational Earth observation and disaster response scenarios, inference latency directly governs mission responsiveness.
SatQuery AI provides sub-second multi-modal analysis, combining lightweight neural models (e.g., SatSegNet, SatQueryOpticalSARNet) and optimized sliding window tiling.

---

## 2. Quantitative Latency Benchmarks

| Remote Sensing Tool | Mean Latency (ms) | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) | Throughput (QPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Image Vqa** | **17.71** | **15.01** | **29.17** | 13.32 | 35.7 | **56.46** |
| **Single Image Captioning** | **14.13** | **13.72** | **16.35** | 13.3 | 19.1 | **70.79** |
| **Single Image Grounding** | **13.26** | **13.24** | **14.13** | 12.55 | 14.18 | **75.44** |
| **Land Cover Segmentation** | **14.97** | **13.76** | **21.31** | 12.25 | 25.42 | **66.8** |
| **Bi Temporal Change** | **50.55** | **51.76** | **58.05** | 40.26 | 58.35 | **19.78** |
| **Optical Sar Fusion** | **7.16** | **7.07** | **8.59** | 5.98 | 8.6 | **139.75** |
| **Spatial Compatibility Verifier** | **28.54** | **27.76** | **34.66** | 25.95 | 35.84 | **35.04** |
| **Large Raster Tiling Stitching** | **236.86** | **229.29** | **274.93** | 215.93 | 291.6 | **4.22** |
| **Raster Ingestion Decode** | **13.31** | **13.29** | **14.58** | 11.86 | 14.65 | **75.11** |

---

## 3. Large Raster Tiling Telemetry

- **Input Resolution Tested:** 2048 x 2048 pixels (12.5M pixel channels)
- **Tile Window Dimensions:** 512 x 512 with 64px overlap
- **Total Tiles per Image:** 25 tiles
- **End-to-End Tiling & Stitching p50 Latency:** 229.29 ms
- **Memory Bounding:** Kept peak working set below 512 MB via tile streaming.

---

## 4. Observations & SLA Compliance

- **Single-Image Queries (VQA/Grounding/Land Cover):** Sub-second response time across standard 1024x1024 satellite tiles.
- **Bi-Temporal Change Analysis:** Rapid feature difference and polygon extraction completed within SLA limits.
- **Optical-SAR Fusion:** Dual-encoder inference and NMI spatial verification executed in sub-second time on CPU.
