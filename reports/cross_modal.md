# Optical-SAR Joint Analysis & Feature Fusion Evaluation Report

**Benchmark:** BigEarthNet-MM (arXiv:2105.07921) — Real Sentinel-1 IW GRD + Sentinel-2 MSI L2A Pairs  
**Evaluation Scope:** `REAL_DATA`  
**Data Provenance:** Real ESA Copernicus tiles from BigEarthNet-MM (CDLA-Permissive-1.0)  
**Class Target Derivation:** `CLC_LABEL_DERIVED` — CORINE Land Cover 2018 patch annotations + radiometric thresholding (SAR double-bounce, NDVI proxy, spectral null)  
**Integrity Notice:** No synthetic imagery or procedurally-generated data. All test patches are authentic Sentinel-1 (VV+VH) + Sentinel-2 (12-band) acquisitions.  
**Execution Timestamp:** 2026-11-01 00:12:52 UTC  
**Model:** SatQueryOpticalSARNet-v4 (849,603 parameters)  
**Hardware Device:** CPU  
**Evaluated Test Samples:** 129 real co-registered multi-sensor tiles  
**Runtime:** 69.05 seconds  

---

## 1. Executive Summary & Verification

| Land Cover Target | Mean IoU (%) | Dice F1 Score | Precision | Recall | Primary Sensor Exploitation Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **🏘️ Urban / Built-Up** | **44.02%** | **0.4782** | **0.5046** | **0.5054** | High SAR double-bounce dihedral return + optical structural edge contrast |
| **💧 Hydrological Water** | **56.88%** | **0.6004** | **0.6688** | **0.5828** | Specular microwave radar null (dark SAR) + optical blue-green absorption |
| **🌳 Forest / Vegetation** | **78.28%** | **0.8236** | **0.8324** | **0.8589** | Microwave volume scattering + optical NDVI photosynthetic green reflectance |
| **🌐 ALL-CLASS AVERAGE** | **59.73%** | **0.6341** | — | — | **Dual-Encoder Cross-Modal Feature Fusion** |

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
