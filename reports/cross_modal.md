# Optical-SAR Joint Analysis & Feature Fusion Evaluation Report

**Benchmark:** BigEarthNet.txt (arXiv:2603.29630) Sentinel-1 + Sentinel-2 Pairs  
**Evaluation Scope:** `SYNTHETIC_SCHEMA_PROTOTYPE`  
**Reference Ground Truth Type:** `HEURISTIC_PSEUDO_MASK`  
**Integrity Notice:** Reported mIoU measures network convergence against synthetic heuristic targets (`generate_spatial_targets`), not empirical satellite segmentation accuracy on operational Earth observation data.  
**Execution Timestamp:** 2026-10-14 17:53:04 UTC  
**Model:** SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype) (849,603 parameters)  
**Hardware Device:** CPU  
**Evaluated Test Samples:** 15 co-registered multi-sensor tiles  
**Runtime:** 4.42 seconds  

---

## 1. Executive Summary & Verification

| Land Cover Target | Mean IoU (%) | Dice F1 Score | Precision | Recall | Primary Sensor Exploitation Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **🏘️ Urban / Built-Up** | **12.68%** | **0.1533** | **0.171** | **0.148** | High SAR double-bounce dihedral return + optical structural edge contrast |
| **💧 Hydrological Water** | **20.72%** | **0.2927** | **0.215** | **0.5806** | Specular microwave radar null (dark SAR) + optical blue-green absorption |
| **🌳 Forest / Vegetation** | **41.38%** | **0.4689** | **0.4307** | **0.5293** | Microwave volume scattering + optical NDVI photosynthetic green reflectance |
| **🌐 ALL-CLASS AVERAGE** | **24.93%** | **0.3049** | — | — | **Dual-Encoder Cross-Modal Feature Fusion** |

---

## 2. Cross-Modal Compatibility & Co-Registration Verification

| Test Scenario | Input Pair | Expected Outcome | Actual Outcome | Measured Coherence | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| True Co-Registered Pair (Mumbai Optical + RISAT-1 SAR) | Demo Tiles | PASSED | PASSED | 0.88 | ✅ PASSED |
| Disparate Regions (Bengaluru Optical + RISAT-1 SAR) | Demo Tiles | REJECTED | REJECTED | 0.04 | ✅ PASSED |

---

## 3. Co-Registration Methodology & Guardrails

1. **Georeferenced Pairs (GeoTIFF):** Compares CRS, geotransform, and bounding box overlap IoU via `rasterio`. Rejects pairs with zero coordinate intersection or IoU < 15%.
2. **Un-georeferenced Pairs (Visual / Benchmark Mode):** ORB and SIFT are strictly prohibited for cross-modal matching (due to distinct radar speckle and dielectric reflection physics). Alignment is measured via **Normalized Mutual Information (NMI)** and **Sobel Gradient Structural Correlation**.
   - Mismatched geographic scenes produce negative or negligible correlation (NMI < 0.010, Gradient Correlation <= 0) and are rejected with diagnostic comparison imagery.
   - Genuine co-registered pairs pass and are tagged as `"Unverified Co-Registration"` with calibrated confidence.
3. **Modality Auto-Detection:** Automatically inspects band count, dtype, and radiometric speckle distribution. Refuses queries demanding SAR cloud penetration when only Optical imagery is supplied.
