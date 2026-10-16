# SatQuery AI — PR-1.1 Blocker Remediation Report

**Document Version:** 1.0  
**Date:** 2026-10-15  
**Target Pull Request:** PR-1.1 (Close SAR Provenance and Sensor-Identity Blockers)  
**Audit Reference:** [`reports/training_migration/pr1_independent_verification.md`](file:///d:/sih26167/reports/training_migration/pr1_independent_verification.md)  
**Scope of Changes:**
- [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py)
- [`backend/test_sar_contract.py`](file:///d:/sih26167/backend/test_sar_contract.py)
- [`reports/training_migration/pr1_1_blocker_remediation.md`](file:///d:/sih26167/reports/training_migration/pr1_1_blocker_remediation.md)

---

## 1. Executive Summary & Verdict

Following the independent verification audit of PR-1 ([`pr1_independent_verification.md`](file:///d:/sih26167/reports/training_migration/pr1_independent_verification.md)) which returned **ACCEPT WITH BLOCKERS**, this engineering remediation closed both code-level blockers in strict adherence to narrow, evidence-driven requirements.

### **Remediation Verdict Table**

| Blocker ID | Description | Severity | Remediated Files | Verdict |
| :--- | :--- | :---: | :--- | :---: |
| **Blocker A** | Calibration Provenance Bypass via API Override & Substring Vulnerability | High | `backend/model_interfaces.py`, `backend/test_sar_contract.py` | **PASS** |
| **Blocker B** | Unverified Sensor Identity Conflation (`"RISAT-1 / Sentinel-1 C-Band SAR"`) | High | `backend/model_interfaces.py`, `backend/test_sar_contract.py` | **PASS** |

Both blockers are fully resolved. All 33 regression tests across 4 test suites pass with exit code 0. Both model checkpoint hashes remain strictly intact and the legacy production model remains the system default.

---

## 2. Detailed Technical Remediations

### 2.1 Blocker A: Calibration Provenance Hardening

#### Defects Identified in Audit
1. **API Override Bypass:** In PR-1, line 317 set `provenance_established = True` unconditionally if `explicit_cal in ("sigma0_db", "calibrated_db", ...)`, allowing callers to force `CALIBRATED_SIGMA0_DB` on uncalibrated or non-radar float rasters.
2. **Generic Substring Matching:** Matching `"_db"` in description matched substrings like `project_db` or `db_version`.
3. **No Contradiction Handling:** Conflicting metadata (such as `unit: meters` combined with a radar tag) was not flagged.

#### Remediation Implemented in [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#L300-L470)
- **Zero Override Bypass:** A client-supplied parameter (`sar_calibration="sigma0_db"`), numeric dtype (`float32`), plausible physical range, filename, or generic substring CANNOT independently establish scientific calibration.
- **Fail Closed to `UNKNOWN`:** If authentic metadata tags and units are missing, an explicit parameter override is rejected with a clear explanation:
  `"Client parameter 'sar_calibration=sigma0_db' rejected: cannot establish scientific calibration without verified product metadata tags and units"`.
- **Contradiction Detection:** Explicitly checks for non-decibel units (`m`, `meter`, `meters`, `deg`, `degree`, `k`, `kelvin`, `dn`, `count`, `counts`, `linear`, `w/m2`, `percent`) or non-SAR physical domains (`elevation`, `dem`, `dsm`, `temperature`, `radiance`, `reflectance`, `bathymetry`). Any contradiction immediately marks the raster as `UNKNOWN`.
- **Authentic Unit Verification:** Requires structured tags (`unit` or `units` in `("db", "decibel", "decibels")`) or regex token matching `\b(db|decibel|decibels)\b` in description.
- **Authentic Domain Verification:** Requires structured backscatter tags (`radiometry`, `calibration`, `product_type` in `("sigma0", "sigma0_db", "gamma0", "beta0", "backscatter")`) or regex token matching `\b(sigma0|sigma-0|gamma0|beta0|backscatter)\b`.
- **Speckle Statistics Protection:** Speckle statistics are preserved for visual monochrome detection in browse imagery, but are never treated as proof of scientific calibration.

---

### 2.2 Blocker B: Sensor Identity Disambiguation

#### Defects Identified in Audit
1. **Sensor Identity Conflation:** PR-1 emitted `"sensor_family": "RISAT-1 / Sentinel-1 C-Band SAR"` whenever any raster satisfied `is_scientifically_calibrated=True`, falsely claiming ISRO RISAT-1 compatibility for European Copernicus Sentinel-1 rasters.
2. **Missing Neutral Label:** There was no neutral classification for calibrated SAR data with unverified or third-party sensor origin.

#### Remediation Implemented in [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py#L262-L330)
- **`resolve_sar_sensor_identity` Engine:** Added an isolated disambiguation helper evaluating authentic product tags (`sensor`, `satellite`, `platform`, `mission`), band descriptions, and mission filename tokens:
  - **Sentinel-1:** Platform/mission matches `sentinel-1`, `s1a`, `s1b`, `copernicus`.
  - **RISAT-1 / EOS-04:** Platform/mission matches `risat-1`, `risat1`, `eos-04`, `isro`.
  - **Unverified:** Default when mission tags are absent or generic.
- **Neutral Label for Generic Calibrated SAR:** Calibrated SAR without specific mission provenance emits:
  `"CALIBRATED SAR — SENSOR UNVERIFIED"`.
- **Modality Detection Update:** All SAR classification paths in `detect_image_modality` dynamically retrieve `sar_c.get("sensor_family")`, completely eliminating blanket claims of `"RISAT-1 / Sentinel-1 C-Band SAR"`.
- **Pair Comparison Output Labeling:** Updated `CrossModalAnalysis.execute` ([`backend/model_interfaces.py:L2628-L2636`](file:///d:/sih26167/backend/model_interfaces.py#L2628-L2636)) to emit exact, verified labels:
  - Sentinel-1: `CALIBRATED SENTINEL-1 C-BAND SAR (σ⁰ BACKSCATTER: Decibels (dB))`
  - RISAT-1: `CALIBRATED RISAT-1 C-BAND SAR (σ⁰ BACKSCATTER: Decibels (dB))`
  - Generic Calibrated: `CALIBRATED SAR — SENSOR UNVERIFIED (σ⁰ BACKSCATTER: Decibels (dB))`
  - Visual Grayscale: `SAR BROWSE / VISUAL GRAYSCALE (SURROGATE RADAR MAPPING, UNCALIBRATED)`
  - Raw Integer DN: `RAW SAR DIGITAL NUMBERS (UNCALIBRATED DN, SURROGATE MAPPING)`
  - Unknown: `SYNTHETIC APERTURE RADAR (UNCALIBRATED / UNKNOWN RADIOMETRY)`

---

## 3. Regression Test Suite & Verification Results

All automated test suites were executed against the Python 3.10 virtual environment (`backend/venv`).

### 3.1 Test Execution Summary

| Test Suite File | Command Executed | Exit Code | Tests Passed | Tests Failed |
| :--- | :--- | :---: | :---: | :---: |
| `backend/test_sar_contract.py` | `python -m pytest -v backend/test_sar_contract.py` | **0** | **11** | **0** |
| `backend/test_cross_modal.py` | `python -m pytest -v backend/test_cross_modal.py` | **0** | **10** | **0** |
| `backend/test_change_pipeline.py` | `python -m pytest -v backend/test_change_pipeline.py` | **0** | **4** | **0** |
| `backend/test_tool_registry.py` | `python -m pytest -v backend/test_tool_registry.py` | **0** | **8** | **0** |
| **Total** | — | **0** | **33** | **0** |

### 3.2 Individual Test Cases in `backend/test_sar_contract.py`

1. `test_calibrated_float32_input_contract` — **PASSED**  
   *Verifies georeferenced float32 with authentic Sentinel-1 dB metadata propagates `CALIBRATED_SIGMA0_DB` through `CrossModalAnalysis`.*
2. `test_visual_grayscale_surrogate_contract` — **PASSED**  
   *Verifies 8-bit visual PNG maps to `SURROGATE_VISUAL_GRAYSCALE` and never claims calibrated radar backscatter.*
3. `test_uncalibrated_dn_contract` — **PASSED**  
   *Verifies integer GeoTIFFs (raw DN) map to `UNCALIBRATED_DN` and require radiometric LUT.*
4. `test_unknown_calibration_float_raster` — **PASSED**  
   *Verifies unverified float32 GeoTIFF without radar tags maps to `UNKNOWN`.*
5. `test_override_alone_rejected_without_metadata` (Blocker A) — **PASSED**  
   *Verifies that passing `sar_calibration="sigma0_db"` on an unverified float raster fails closed to `UNKNOWN`.*
6. `test_misleading_substring_metadata_rejected` (Blocker A) — **PASSED**  
   *Verifies that descriptions such as `project_db_version_2` do NOT trigger calibration.*
7. `test_contradictory_units_rejected` (Blocker A) — **PASSED**  
   *Verifies that contradictory metadata (`unit: meters`) fails closed to `UNKNOWN` even with explicit override.*
8. `test_generic_calibrated_sar_does_not_assert_risat` (Blocker B) — **PASSED**  
   *Verifies that generic calibrated SAR emits `CALIBRATED SAR — SENSOR UNVERIFIED` and does NOT assert RISAT-1.*
9. `test_verified_sentinel1_calibration` (Blocker B) — **PASSED**  
   *Verifies that authentic Sentinel-1 metadata assigns `Sentinel-1 C-Band SAR`.*
10. `test_verified_risat1_calibration` (Blocker B) — **PASSED**  
    *Verifies that authentic RISAT-1 metadata assigns `RISAT-1 C-Band SAR`.*
11. `test_preprocessing_consistency_and_finite_outputs` — **PASSED**  
    *Verifies non-empty GeoJSON, finite masks, and numerical stability across both calibrated and surrogate inputs.*

---

## 4. Preservation of Model Checkpoints and Numerical Pipeline

### 4.1 Checkpoint SHA-256 Integrity
Both model checkpoints were verified and remain identical to pre-audit states:

| Checkpoint Identifier | File Path | Expected SHA-256 | Verified SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Legacy Production** | `ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth` | `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405` | `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405` | **MATCH** |
| **Staging Pilot** | `ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth` | `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2` | `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2` | **MATCH** |

### 4.2 Production Routing Verification
In [`backend/ml_models.py:L172-L184`](file:///d:/sih26167/backend/ml_models.py#L172-L184):
- Default configuration (when `OPTICAL_SAR_STAGING_ENABLED` is unset or false) routes directly to the legacy production checkpoint (`satquery_optical_sar_net/optical_sar_net.pth`).
- Staging model is gated behind explicit environment configuration (`OPTICAL_SAR_STAGING_ENABLED="true"`).

### 4.3 Numerical Preprocessing Preservation
- In accordance with task constraints, no changes were made to the model's trained preprocessing, normalization, or input distribution.
- Existing finite value sanitation (`np.nan_to_num(raw_sar, nan=-25.0, posinf=0.0, neginf=-35.0)`) and boundary clipping (`np.clip(raw_sar, -35.0, 0.0)`) remain unmodified.

---

## 5. Scope Enforcement & Conclusion

- **Files Modified:** Only `backend/model_interfaces.py`, `backend/test_sar_contract.py`, and `reports/training_migration/pr1_1_blocker_remediation.md`.
- **Out of Scope Work:** No training was started, no checkpoint was altered, and PR-2 (Parent-Granule-Disjoint Re-Evaluation) and PR-3 (Promotion Gate Decision) were **NOT** initiated.
- **Final PR-1.1 Status:** **READY FOR FORMAL STAGING ACCEPTANCE**.
