# SatQuery AI — PR-1 Independent Verification Audit (Read-Only)

**Audit Version:** 1.0  
**Audit Date:** 2026-10-15  
**Review Type:** Independent Read-Only Code & Systems Audit  
**Target Pull Request:** PR-1 (SAR Radiometric Contract Remediation)  
**Primary Code Artifacts Inspected:**
- [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#L262-L485)
- [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#L2480-L2645)
- [`backend/test_sar_contract.py`](file:///d:/sih26167/backend/test_sar_contract.py)
- [`reports/training_migration/pr1_sar_contract_implementation.md`](file:///d:/sih26167/reports/training_migration/pr1_sar_contract_implementation.md)

---

## 1. Executive Summary & Verdict

This audit conducted an independent, read-only review of the **PR-1 SAR Radiometric Contract Remediation**. The review traced the mathematical mechanics of calibration provenance, compared training vs. inference tensor pipelines, evaluated output labeling integrity, and independently executed all 28 backend regression tests.

### **FINAL AUDIT VERDICT: ACCEPT WITH BLOCKERS**

PR-1 successfully resolves the primary staging defect identified in Gate 3: it eliminates the silent misrepresentation of 8-bit visual grayscale as calibrated radar backscatter, establishes an explicit 4-tier contract taxonomy, implements direct float32 GeoTIFF pass-through, and passes all 28 automated tests without pipeline regression.

However, the implementation contains **two material architectural blockers** that must be resolved prior to full production promotion:
1. **Provenance Bypass via API Override:** The `sar_calibration="sigma0_db"` parameter allows unverified float32 GeoTIFFs to bypass metadata tag inspection.
2. **Unverified Sensor Conflation:** The modality classifier groups Sentinel-1 and RISAT-1 into a single string (`"RISAT-1 / Sentinel-1 C-Band SAR"`) without verifying ISRO-specific calibration metadata or hybrid polarimetric headers.

---

## 2. Requirement-by-Requirement Audit Findings

| Requirement | Audit Focus | Evaluation Findings | Status |
| :--- | :--- | :--- | :---: |
| **1. Calibration Provenance** | Contract assignment & override security | Georeference + float + dB tags + physical range ($[-70, +30]\text{ dB}$) required. However, `sar_calibration="sigma0_db"` API parameter bypasses tag checks, and substring matching on `_db` can theoretically match non-radar metadata. | **PARTIAL** |
| **2. Training–Inference Compatibility** | Tensor shapes, units, normalization, NaNs | Pilot training used raw Sentinel-1 VV decibels from BEN-GE-800 ($\approx [-25.35, +3.78]\text{ dB}$). Inference direct float32 pass-through delivers identical decibel units and float32 dtype. Surrogate $[0, 255] \to [-25, 0]\text{ dB}$ mapping preserved for visual inputs. | **PASS** |
| **3. Contract & Labeling Integrity** | Visual browse labeling & sensor claims | 8-bit browse PNG/JPEGs and integer DN rasters are strictly prevented from claiming calibration. Contract metadata is consistent across all response structures. However, `sensor_family` conflates Sentinel-1 with RISAT-1 without sensor validation. | **PARTIAL** |
| **4. Regression Evidence** | Automated test execution & behavioral rigor | 28 / 28 tests passed across 4 test suites. Tests validate contract assignment, non-crashing execution, GeoJSON vector generation, and label agreement. However, tests do not inspect the intermediate PyTorch tensor directly via forward hooks. | **PASS** |

---

## 3. Deep Dive 1: Calibration Provenance & Security Analysis

### 3.1 Trace of `CALIBRATED_SIGMA0_DB` Assignment Path
In [`backend/model_interfaces.py:L262-L370`](file:///d:/sih26167/backend/model_interfaces.py#L262-L370), `resolve_sar_input_contract` inspects the input:
1. Opens `image_bytes` via `rasterio.MemoryFile`.
2. Inspects `ds.crs` and `ds.transform`. If CRS is `None` or transform is identity, georeferencing is deemed absent (`has_geo = False`).
3. Searches tags and descriptions for decibel units:
   - Line 308: `any(tags.get(k) in ("db", "decibel", "decibels") for k in ("unit", "units", "data_type", "radiometry"))`
   - Line 309: `"unit=db" in tags_str or "units=db" in tags_str or "(db)" in desc_str or "_db" in desc_str`
4. Checks floating-point type: `dtype_str.startswith("float")`.
5. Masks nodata and non-finite pixels (`np.isfinite(band1)`) and computes `p_min`, `p_max`, `p_mean`.
6. Assigns `CALIBRATED_SIGMA0_DB` **if and only if**:
   ```python
   if has_geo and provenance_established and (-70.0 <= p_min <= 10.0) and (-30.0 <= p_max <= 35.0):
   ```

### 3.2 Material Vulnerability: The `sar_calibration="sigma0_db"` Override
Lines 317–319 implement the parameter override:
```python
if explicit_cal in ("sigma0_db", "calibrated_db", "sigma0", "calibrated"):
    provenance_established = True
    prov_reason = f"Explicit API calibration declaration: '{explicit_calibration}'"
```
**Audit Finding:**  
If a caller supplies `sar_calibration="sigma0_db"` in `kwargs`:
- Embedded metadata tag verification is completely bypassed.
- Any georeferenced float32 GeoTIFF whose values fall between $-70.0\text{ dB}$ and $+35.0\text{ dB}$ (e.g. an elevation raster in kilometers or a normalized spectral difference index scaled by 10) will be accepted as `CALIBRATED_SIGMA0_DB`.
- **Mitigation Required for Production:** The override must only be honored if the file also passes radar-specific validation (e.g., verifying that the file contains SAR speckle statistics or explicit radar header metadata).

### 3.3 Handling of Non-Finite and Contradictory Pixels
- **Non-Finite Pixels (NaN/Inf):** Lines 332–334 correctly filter NaNs during range calculation. In `CrossModalAnalysis` line 2515, `np.nan_to_num(raw_sar, nan=-25.0, posinf=0.0, neginf=-35.0)` prevents NaN propagation into PyTorch.
- **Completely Blank / All-NaN Images:** If an image is 100% NaN, line 341 sets default range $[-25.0, 0.0]$. However, `decode_satellite_image` at line 97 raises `ValueError("GeoTIFF raster contains completely blank / uniform empty pixels.")`, preventing silent execution.

---

## 4. Deep Dive 2: Training vs. Inference Pipeline Comparison

Below is the side-by-side comparison of the exact preprocessing code:

```
┌───────────────────────────────────────────────────┬───────────────────────────────────────────────────┐
│        PILOT TRAINING PREPROCESSING               │         PR-1 INFERENCE PREPROCESSING              │
│     (ml_pipeline/train_optical_sar_pilot.py)      │            (backend/model_interfaces.py)          │
├───────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ # Optical (Sentinel-2 MSI)                        │ # Optical (Multispectral / RGB)                   │
│ s2_img = Image.open(s2_path).convert("RGB")       │ img1 = _bytes_to_cv2(t_opt_bytes)                 │
│ s2_img = s2_img.resize((128, 128), BILINEAR)      │ opt_rgb = cv2.cvtColor(img1, BGR2RGB) / 255.0     │
│ s2_arr = np.array(s2_img, dtype=float32) / 255.0  │ opt_res = cv2.resize(opt_rgb, (inf_w, inf_h))     │
│ opt_tensor = torch.from_numpy(s2_arr).permute(...)│ opt_t = torch.from_numpy(opt_res).permute(...)    │
│ Range: [0.0, 1.0]                                 │ Range: [0.0, 1.0]                                 │
├───────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ # SAR (Sentinel-1 VV Backscatter)                 │ # SAR: Case A (CALIBRATED_SIGMA0_DB)              │
│ s1_arr = np.load(s1_path) # [120, 120, 2]         │ with rasterio.MemoryFile(t_sar_bytes) as mem:     │
│ vv = s1_arr[..., 0] # Native Float32 dB           │     raw_sar = ds.read(1).astype(np.float32)       │
│ vv_img = Image.fromarray(vv).resize((128, 128))   │     raw_sar = np.nan_to_num(raw_sar, nan=-25.0)   │
│ vv_arr = np.array(vv_img, dtype=np.float32)       │     raw_sar = np.clip(raw_sar, -50.0, 25.0)       │
│ sar_tensor = torch.from_numpy(vv_arr).unsqueeze(0)│     sar_inp = raw_sar # Native Float32 dB         │
│                                                   │ sar_res = cv2.resize(sar_inp, (inf_w, inf_h))     │
│                                                   │ sar_t = torch.from_numpy(sar_res).unsqueeze(0)    │
│ Units: Decibels (dB)                              │ Units: Decibels (dB)                              │
│ Observed Mean: -12.03 dB                          │ Input Mean: -12.0 to -15.0 dB                     │
│ Observed Dynamic Range: [-25.35, +3.78] dB        │ Clipped Dynamic Range: [-50.0, +25.0] dB          │
├───────────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ N/A (Trained exclusively on authentic float32)    │ # SAR: Case B (SURROGATE_VISUAL_GRAYSCALE)        │
│                                                   │ sar_gray = cv2.cvtColor(img2, BGR2GRAY)           │
│                                                   │ sar_inp = (sar_gray / 255.0) * 25.0 - 25.0        │
│                                                   │ Units: Empirical Surrogate Decibels               │
│                                                   │ Scaled Dynamic Range: [-25.0, 0.0] dB             │
└───────────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

### Compatibility Findings:
1. **Tensor Shape & Rank:** Exact match. Both pipelines deliver optical tensor `[B, 3, H, W]` and SAR tensor `[B, 1, H, W]`.
2. **Decibel Domain Equivalence:** Authentic pilot training fed raw Sentinel-1 float32 decibels directly into `SpeckleFilter` and `sar_conv0`. Ingesting calibrated float32 GeoTIFFs without uint8 quantization directly restores the authentic training distribution.
3. **Surrogate Domain Alignment:** The visual surrogate linear map $[0, 255] \to [-25.0, 0.0]\text{ dB}$ aligns with the lower half of the training distribution (mean $\approx -12.5\text{ dB}$), allowing uncalibrated images to run through the neural weights without generating numerical blowups.

---

## 5. Deep Dive 3: Contract & Labeling Integrity

### 5.1 Verification of Anti-Spoofing Constraints
- **8-Bit Visual Inputs (PNG/JPEG):** Traced through `detect_image_modality` and `resolve_sar_input_contract`. Always assigned `SURROGATE_VISUAL_GRAYSCALE` with `is_scientifically_calibrated: False`. `after_label` is strictly set to:
  `"SAR BROWSE / VISUAL GRAYSCALE (SURROGATE RADAR MAPPING, UNCALIBRATED)"`.
  **Arbitrary visual images cannot be labeled as calibrated radar.**
- **Integer Digital Numbers (uint16 GeoTIFF):** Traced through line 371. Assigned `UNCALIBRATED_DN` with `is_scientifically_calibrated: False`. `after_label` is set to:
  `"RAW SAR DIGITAL NUMBERS (UNCALIBRATED DN, SURROGATE MAPPING)"`.

### 5.2 Sensor Conflation Blocker: Sentinel-1 vs. RISAT-1
At [`backend/model_interfaces.py:L438`](file:///d:/sih26167/backend/model_interfaces.py#L438) and [`L480`](file:///d:/sih26167/backend/model_interfaces.py#L480):
```python
"sensor_family": "RISAT-1 / Sentinel-1 C-Band SAR" if sar_c["is_scientifically_calibrated"] else "Single-Band SAR / Radar"
```
**Audit Finding:**  
When an input passes calibration checks, the system names the sensor family as `"RISAT-1 / Sentinel-1 C-Band SAR"` regardless of whether the imagery originated from ESA Sentinel-1 or ISRO RISAT-1.
- **The Issue:** `SatQueryOpticalSARNet` was trained on ESA Sentinel-1 (C-band linear VV/VH, 10m GRD). It has never been trained or validated on ISRO RISAT-1 (which features hybrid/circular polarimetry $RH/RV$).
- **Requirement Violation:** Labeling generic calibrated SAR as "RISAT-1" without verifying ISRO CEOS/GeoTIFF metadata violates the operational boundary established in Gate 7.
- **Remediation Required:** Disambiguate sensor metadata: only label "RISAT-1" if ISRO mission tags (`RISAT1`, `EOS-04`) are present, and label "Sentinel-1" if Copernicus tags are present.

---

## 6. Deep Dive 4: Regression Test Verification

All four backend test suites were executed independently during this audit session:

```powershell
# 1. Focused SAR Contract Tests
powershell -Command "d:\sih26167\backend\venv\Scripts\pytest.exe -v backend/test_sar_contract.py"
# Result: 6 passed, 10 warnings in 11.41s | Exit Code: 0

# 2. Cross-Modal Analysis Regression Suite
powershell -Command "d:\sih26167\backend\venv\Scripts\pytest.exe -v backend/test_cross_modal.py"
# Result: 10 passed, 7 warnings in 10.04s | Exit Code: 0

# 3. Change Detection Pipeline Regression Suite
powershell -Command "d:\sih26167\backend\venv\Scripts\pytest.exe -v backend/test_change_pipeline.py"
# Result: 4 passed, 3 warnings in 9.93s | Exit Code: 0

# 4. Tool Registry Specification Suite
powershell -Command "d:\sih26167\backend\venv\Scripts\pytest.exe -v backend/test_tool_registry.py"
# Result: 8 passed, 1 warning in 6.06s | Exit Code: 0
```

### Cumulative Test Metrics:
- **Total Tests Executed:** 28
- **Total Tests Passed:** 28 (100.0%)
- **Total Failures:** 0
- **Regression Status:** **ZERO REGRESSION**

### Test Rigor Assessment:
- The tests in `backend/test_sar_contract.py` validate actual end-to-end execution, non-crashing behavior, contour generation, and response serialization.
- **Limitation:** The tests do not intercept the PyTorch forward pass with a tensor spy to mathematically prove that the tensor received by the first convolutional layer equals `ds.read(1)`. This is acceptable for a staging integration test, but an explicit tensor equality assertion should be added in PR-2.

---

## 7. Remaining Blockers for Production Promotion

While PR-1 is safe for staging evaluation, the following blockers must be addressed before live production rollout:

```markdown
- [ ] BLOCKER 1 (Security/Provenance): Tighten `sar_calibration="sigma0_db"` override.
      Do not permit arbitrary float32 rasters to claim calibration based solely on an API argument. Require either embedded radar metadata or SAR speckle variance verification.

- [ ] BLOCKER 2 (Scientific Boundary): Disambiguate RISAT-1 from Sentinel-1 in `detect_image_modality`.
      Do not emit `"RISAT-1 / Sentinel-1 C-Band SAR"` jointly. Emit `"Sentinel-1 C-Band SAR"` for Copernicus products, and `"RISAT-1 / EOS-04 C-Band SAR"` only when ISRO payload tags are verified.

- [ ] BLOCKER 3 (Parent-Granule-Disjoint Validation): Complete Gate 1.4.
      Validate the candidate model on the parent-granule-disjoint test partition to confirm that the reported 63.26% test mIoU does not collapse when evaluated on unseen satellite granules.
```

---

## 8. Final Audit Sign-Off

- **Implementation Status:** Verified and functional in staging.
- **Production Default Routing:** Verified unchanged (`OPTICAL_SAR_STAGING_ENABLED="false"`).
- **Checkpoints:** Bit-for-bit verified (legacy production: `f8c61f87...`, candidate pilot: `93ec9d95...`).
- **Final Verdict:** **ACCEPT WITH BLOCKERS** (Safe for controlled staging evaluation; blockers must be remediated prior to live deployment).
