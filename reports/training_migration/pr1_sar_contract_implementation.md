# SatQuery AI — PR-1: SAR Radiometric Contract Remediation Report

**PR Identifier:** `PR-1` (from `reports/training_migration/final_production_gate.md`)  
**Implementation Date:** 2026-10-15  
**Review Type:** Code Remediation & Verification Audit  
**Target File:** [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py)  
**Test Suite:** [`backend/test_sar_contract.py`](file:///d:/sih26167/backend/test_sar_contract.py)  

---

## 1. Executive Summary

In accordance with Gate 3 of the pre-production readiness framework, **PR-1 (SAR Radiometric Contract Remediation)** has been implemented in [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py). 

The remediation eliminates the risk of uncalibrated or surrogate grayscale imagery silently masquerading as physical radar backscatter ($\sigma^0$). It establishes a verifiable radiometric contract, provides direct float32 pass-through for calibrated radar GeoTIFFs, and updates user-facing output labels and metadata to reflect the true physical provenance of the input data.

**Key Achievements:**
1. **Verifiable SAR Contract Taxonomy:** Implemented `resolve_sar_input_contract` identifying four explicit radiometric contracts: `CALIBRATED_SIGMA0_DB`, `SURROGATE_VISUAL_GRAYSCALE`, `UNCALIBRATED_DN`, and `UNKNOWN`.
2. **Strict Calibration Criteria:** `is_scientifically_calibrated` is set to `True` **only** when explicit calibration metadata tags/descriptions (or verified API declarations) confirm backscatter units in decibels ($\text{dB}$) and values fall within plausible physical bounds ($[-70.0, +30.0]\text{ dB}$). File extension or numeric dtype alone is explicitly rejected as proof of calibration.
3. **Lossless Direct Float32 Pass-Through:** Calibrated float32 GeoTIFFs are ingested directly into the neural tensor (`sar_t`) without intermediate 8-bit percentile quantization, preserving authentic decibel dynamics.
4. **Preserved Trained Normalization:** Non-calibrated visual grayscale and integer DN inputs continue to use the exact empirical surrogate mapping ($[0, 255] \to [-25.0, 0.0]\text{ dB}$) expected by the trained model.
5. **Truth in Labeling:** Output labels in `pair_comparison["after_label"]` dynamically reflect the verified contract. Arbitrary PNG/JPEG images are labeled as `SAR BROWSE / VISUAL GRAYSCALE (SURROGATE RADAR MAPPING, UNCALIBRATED)` and never as calibrated radar backscatter.
6. **Zero Pipeline Regression:** 28 of 28 automated tests passed across all backend test suites.

---

## 2. Radiometric Contract Taxonomy

| Contract ID | Data Type & Format | Calibration Established? | Description & Preprocessing Handling | User-Facing Output Label |
| :--- | :--- | :---: | :--- | :--- |
| **`CALIBRATED_SIGMA0_DB`** | Georeferenced `float32` / `float64` GeoTIFF | **YES** (`True`) | Georeferenced raster with explicit metadata tags (e.g. `unit: dB`, `Sigma0_VV_db`) and plausible values. Passed **directly as float32** into neural tensor without uint8 quantization. | `CALIBRATED C-BAND SAR (σ⁰ BACKSCATTER: Decibels (dB))` |
| **`SURROGATE_VISUAL_GRAYSCALE`** | 8-bit visual images (PNG, JPEG, WebP, BMP, uint8 TIFF) | **NO** (`False`) | Standard visual browse imagery lacking radar calibration metadata. Mapped via empirical linear surrogate $[0, 255] \to [-25.0, 0.0]\text{ dB}$. | `SAR BROWSE / VISUAL GRAYSCALE (SURROGATE RADAR MAPPING, UNCALIBRATED)` |
| **`UNCALIBRATED_DN`** | Georeferenced integer GeoTIFF (`uint16`, `int16`, etc.) | **NO** (`False`) | Raw radar digital numbers (DN) lacking calibration LUT XML. Mapped to surrogate decibels with explicit notice that LUT is missing. | `RAW SAR DIGITAL NUMBERS (UNCALIBRATED DN, SURROGATE MAPPING)` |
| **`UNKNOWN`** | Float rasters lacking calibration tags / unverified | **NO** (`False`) | Unverified float rasters. Numeric dtype or `.tif` extension alone does not establish calibration. Processed via fallback surrogate normalization. | `SYNTHETIC APERTURE RADAR (UNCALIBRATED / UNKNOWN RADIOMETRY)` |

---

## 3. Detailed Architectural Implementation

### 3.1 Contract Resolver (`resolve_sar_input_contract`)
Implemented at [`backend/model_interfaces.py:L262-L408`](file:///d:/sih26167/backend/model_interfaces.py#L262-L408):
- Inspects rasterio metadata (CRS, transform, band count, dtypes, descriptions, and tags).
- Searches tags and descriptions for decibel indicators (`unit=db`, `units=db`, `radiometry=sigma0`, `_db`, `(db)`).
- Validates that values in float rasters are physically plausible for microwave backscatter in decibels: $\min \ge -70.0\text{ dB}$, $\max \le 30.0\text{ dB}$, mean between $-40.0\text{ dB}$ and $+10.0\text{ dB}$.
- Falls back to `UNKNOWN` or `SURROGATE_VISUAL_GRAYSCALE` if provenance cannot be verified.

### 3.2 Direct Float32 Pass-Through in `CrossModalAnalysis`
Implemented at [`backend/model_interfaces.py:L2480-L2510`](file:///d:/sih26167/backend/model_interfaces.py#L2480-L2510):
```python
# ── SAR Input Contract & Radiometric Ingestion ──
if sar_contract["contract"] == "CALIBRATED_SIGMA0_DB":
    # Direct float32 pass-through: preserve true calibrated decibels without lossy uint8 conversion
    try:
        with rasterio.MemoryFile(t_sar_bytes) as memfile:
            with memfile.open() as ds:
                raw_sar = ds.read(1).astype(np.float32)
                if ds.nodata is not None:
                    raw_sar[raw_sar == ds.nodata] = -25.0
                raw_sar = np.nan_to_num(raw_sar, nan=-25.0, posinf=0.0, neginf=-35.0)
                raw_sar = np.clip(raw_sar, -50.0, 25.0)
                if raw_sar.shape[:2] != (h, w):
                    raw_sar = cv2.resize(raw_sar, (w, h), interpolation=cv2.INTER_LINEAR)
                sar_inp = raw_sar
    except Exception as sar_read_err:
        logger.warning(f"Direct float32 SAR read failed: {sar_read_err}; falling back to surrogate normalization")
        sar_inp = (sar_gray.astype(np.float32) / 255.0) * 25.0 - 25.0
else:
    # Surrogate 8-bit visual grayscale, uncalibrated DN, or unknown:
    # Apply exact trained surrogate normalization mapping [0, 255] -> [-25.0, 0.0] dB
    sar_inp = (sar_gray.astype(np.float32) / 255.0) * 25.0 - 25.0

inf_h, inf_w = min(h, 256), min(w, 256)
opt_res = cv2.resize(opt_rgb, (inf_w, inf_h))
sar_res = cv2.resize(sar_inp, (inf_w, inf_h))

opt_t = torch.from_numpy(opt_res).permute(2, 0, 1).unsqueeze(0).to(ml_manager.device)
sar_t = torch.from_numpy(sar_res).unsqueeze(0).unsqueeze(0).to(ml_manager.device)
```

### 3.3 Metadata and Pair Comparison Updates
1. **`inference_metadata`:**
   Includes the full `sar_input_contract` payload with dynamic range, units, calibration notice, and provenance details.
2. **`pair_comparison`:**
   Sets `after_label` dynamically based on the verified contract.
3. **Response Root:**
   Exposes `"sar_input_contract": sar_contract` for top-level programmatic inspection.

---

## 4. Verification & Test Results

### 4.1 Focused Test Suite: `backend/test_sar_contract.py`
Command executed:
```powershell
d:\sih26167\backend\venv\Scripts\pytest.exe backend/test_sar_contract.py
```

| Test Function | Description | Tested Contract | Status |
| :--- | :--- | :--- | :---: |
| `test_calibrated_float32_input_contract` | Verified float32 GeoTIFF with dB tags | `CALIBRATED_SIGMA0_DB` | **PASS** |
| `test_visual_grayscale_surrogate_contract` | 8-bit visual PNG browse image | `SURROGATE_VISUAL_GRAYSCALE` | **PASS** |
| `test_uncalibrated_dn_contract` | 16-bit integer GeoTIFF without LUT | `UNCALIBRATED_DN` | **PASS** |
| `test_unknown_calibration_float_raster` | Float32 raster lacking calibration metadata | `UNKNOWN` | **PASS** |
| `test_explicit_api_calibration_override` | `sar_calibration='sigma0_db'` override | `CALIBRATED_SIGMA0_DB` | **PASS** |
| `test_preprocessing_consistency_and_finite_outputs` | Output mask finiteness and bounds | Multi-contract consistency | **PASS** |

**Focused Suite Result: 6 / 6 PASSED (100%)**

---

### 4.2 Full Backend Regression Suite Results

| Test Suite | File Path | Total Tests | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SAR Radiometric Contract** | [`backend/test_sar_contract.py`](file:///d:/sih26167/backend/test_sar_contract.py) | 6 | 6 | 0 | **PASS** |
| **Cross-Modal Analysis Suite** | [`backend/test_cross_modal.py`](file:///d:/sih26167/backend/test_cross_modal.py) | 10 | 10 | 0 | **PASS** |
| **Change Detection Pipeline** | [`backend/test_change_pipeline.py`](file:///d:/sih26167/backend/test_change_pipeline.py) | 4 | 4 | 0 | **PASS** |
| **Tool Registry Specifications**| [`backend/test_tool_registry.py`](file:///d:/sih26167/backend/test_tool_registry.py) | 8 | 8 | 0 | **PASS** |
| **TOTAL BACKEND TESTS** | — | **28** | **28** | **0** | **100% PASS** |

---

## 5. Concise Code Diff Summary

```diff
--- a/backend/model_interfaces.py
+++ b/backend/model_interfaces.py
@@ -260,6 +260,155 @@ def get_image_format(image_bytes: bytes, filename: Optional[str] = None) -> str:
     return "PNG"
 
+def resolve_sar_input_contract(
+    image_bytes: bytes,
+    filename: Optional[str] = None,
+    explicit_calibration: Optional[str] = None
+) -> Dict[str, Any]:
+    """Evaluates and enforces the SAR Radiometric Input Contract."""
+    ...
+    # Classifies into: CALIBRATED_SIGMA0_DB | UNCALIBRATED_DN | SURROGATE_VISUAL_GRAYSCALE | UNKNOWN
+
 def detect_image_modality(image_bytes: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
...
+    sar_c = resolve_sar_input_contract(image_bytes, filename=filename)
     return {
         "modality": "SAR",
-        "sensor_family": "RISAT-1 / Sentinel-1 C-Band SAR",
+        "sensor_family": "RISAT-1 / Sentinel-1 C-Band SAR" if sar_c["is_scientifically_calibrated"] else "Single-Band SAR / Radar",
+        "sar_input_contract": sar_c
     }
@@ -2280,6 +2420,12 @@ class CrossModalAnalysis(SpecialistModel):
+        sar_contract = resolve_sar_input_contract(
+            t_sar_bytes,
+            filename=fn_sar,
+            explicit_calibration=kwargs.get("sar_calibration")
+        )
@@ -2341,11 +2487,27 @@ class CrossModalAnalysis(SpecialistModel):
-        sar_inp = (sar_gray.astype(np.float32) / 255.0) * 25.0 - 25.0
+        if sar_contract["contract"] == "CALIBRATED_SIGMA0_DB":
+            # Direct float32 pass-through: preserve true calibrated decibels without lossy uint8 conversion
+            try:
+                with rasterio.MemoryFile(t_sar_bytes) as memfile:
+                    with memfile.open() as ds:
+                        raw_sar = ds.read(1).astype(np.float32)
+                        ...
+                        sar_inp = raw_sar
+            except Exception:
+                sar_inp = (sar_gray.astype(np.float32) / 255.0) * 25.0 - 25.0
+        else:
+            sar_inp = (sar_gray.astype(np.float32) / 255.0) * 25.0 - 25.0
@@ -2449,3 +2610,12 @@ class CrossModalAnalysis(SpecialistModel):
-        "after_label": "RISAT-1 / SENTINEL-1 C-BAND SAR (RADAR BACKSCATTER)"
+        if sar_contract["contract"] == "CALIBRATED_SIGMA0_DB":
+            after_lbl = f"CALIBRATED C-BAND SAR (\u03c3\u2070 BACKSCATTER: {sar_contract['input_units']})"
+        elif sar_contract["contract"] == "SURROGATE_VISUAL_GRAYSCALE":
+            after_lbl = "SAR BROWSE / VISUAL GRAYSCALE (SURROGATE RADAR MAPPING, UNCALIBRATED)"
+        elif sar_contract["contract"] == "UNCALIBRATED_DN":
+            after_lbl = "RAW SAR DIGITAL NUMBERS (UNCALIBRATED DN, SURROGATE MAPPING)"
+        else:
+            after_lbl = "SYNTHETIC APERTURE RADAR (UNCALIBRATED / UNKNOWN RADIOMETRY)"
```

---

## 6. Constraints Adherence & Production Safety

1. **Checkpoints Preserved:**
   - Production Default: `ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth` (SHA-256: `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405`).
   - Staging Pilot: `ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth` (SHA-256: `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2`).
   - Neither checkpoint was modified, moved, or overwritten.
2. **Production Routing Untouched:**
   - Default environment variable `OPTICAL_SAR_STAGING_ENABLED` remains `"false"`. The legacy model remains the production default.
3. **Scope Boundary:**
   - PR-2 (Format guardrails) and PR-3 (Disjoint experiment) were **not** implemented in this phase, in strict compliance with instructions.
4. **Zero ISRO/SAC Claims:**
   - The contract explicitly prevents claiming calibrated RISAT-1 backscatter on uncalibrated images.

**PR-1 Status: COMPLETE & VERIFIED. Awaiting technical lead review.**
