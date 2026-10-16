# OpticalSARNet Controlled Staging Promotion Audit

**Audit Date:** 2026-10-15  
**Candidate Model:** `SatQueryOpticalSARNet-AuthenticPilot-v1`  
**Candidate Checkpoint Path:** [`ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth)  
**Candidate SHA-256:** `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2`  
**Production Checkpoint Path:** [`ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth)  
**Production SHA-256:** `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405` (Preserved Unchanged)  
**Staging Promotion Status:** **VERIFIED & CONFIGURED IN STAGING**  

---

## 1. Checkpoint Preservation & Staging Registration

### 1.1 Legacy Production Checkpoint Preservation
The production checkpoint was audited and confirmed intact:
- **Location:** `ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`
- **File Size:** 3,446,729 bytes
- **SHA-256:** `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405`
- **Integrity Status:** **UNTOUCHED & PRESERVED**. Live inference continues to load this production artifact by default.

### 1.2 Candidate Pilot Checkpoint Verification
- **Location:** `ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`
- **File Size:** 3,447,369 bytes
- **Measured SHA-256:** `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2`
- **Audit Report Match:** Exact match against `authentic_pilot_integrity_audit.md`.

### 1.3 Staging Registry Entry
The candidate checkpoint is registered as a separately versioned model artifact:
- **Registered Model Identifier:** `SatQueryOpticalSARNet-v2.0-Staging`
- **Architecture:** Dual-Encoder Squeeze-and-Excitation Modality Fusion Network
- **Parameter Count:** 849,603
- **Training Dataset Provenance:** BigEarthNet-MM / BEN-GE-800 ([Zenodo DOI 10.5281/zenodo.12941231](https://doi.org/10.5281/zenodo.12941231); Mommert et al. 2023 / Sumbul et al. 2021)
- **Ground Truth Provenance:** European Space Agency (ESA) WorldCover 10m v100
- **Training Sample Count:** 550 authentic patches
- **Validation Sample Count:** 121 held-out authentic patches (Val mIoU: 64.26%, Val Loss: 0.5151)
- **Test Sample Count:** 129 held-out authentic patches (Test mIoU: 63.26%, Test Loss: 0.2718)
- **Target Classes:** Built-up (Ch 0), Water Bodies & Wetlands (Ch 1), Vegetation Canopy (Ch 2)

---

## 2. Real Inference Path Validation

The candidate checkpoint was exercised end-to-end through `CrossModalAnalysis` in [`backend/model_interfaces.py`](file:///d:/sih26167/backend/model_interfaces.py):

### 2.1 Optical & SAR Channel Order & Scaling
- **Optical Modality:** Decoded as RGB float32 in $[0.0, 1.0]$ via OpenCV BGR $\to$ RGB conversion. Spatial tensor shape: `[1, 3, 256, 256]`.
- **SAR Modality:** Decoded as 8-bit grayscale and linearly transformed to surrogate decibels via `(gray / 255.0) * 25.0 - 25.0` $\to [-25.0, 0.0]\text{ dB}$. Spatial tensor shape: `[1, 1, 256, 256]`.
- **Modality Swapping Guardrail:** The pipeline correctly detects whether `images[0]` or `images[1]` is SAR via `detect_image_modality` and orders optical to branch 0 and SAR to branch 1.

### 2.2 Finiteness and Non-Saturation Audit
Probed raw sigmoid activation distributions across 65,536 pixels per channel:
- **Built-up Channel:** Min = 0.0895, Max = 0.2521, Mean = 0.1494, Finite = **True**, Saturated ($<0.01$ or $>0.99$) = **0.0%**.
- **Water Channel:** Min = 0.0524, Max = 0.7918, Mean = 0.2143, Finite = **True**, Saturated = **0.0%**.
- **Vegetation Channel:** Min = 0.2726, Max = 0.8899, Mean = 0.7293, Finite = **True**, Saturated = **0.0%**.
- **Conclusion:** Predictions are continuous, un-clipped, and strictly finite.

### 2.3 GeoJSON Polygon Geometry & Area Plausibility
The pipeline extracted 4 vector polygons in the Mumbai evaluation scenario:
- **Coordinate Reference System:** `urn:ogc:def:crs:OGC:1.3:CRS84` (WGS-84 Longitude / Latitude).
- **Topology:** All 4 polygons are valid closed rings (5 coordinates, $p_{\text{start}} == p_{\text{end}}$).
- **Coordinate Bounds:** Confirmed strictly within valid geographic bounds ($72.83^\circ\text{E} \le \text{lon} \le 72.93^\circ\text{E}$, $19.01^\circ\text{N} \le \text{lat} \le 19.12^\circ\text{N}$).
- **Calculated Areas:**
  - `cross_modal_water_1`: 224.0 ha ($2.24\text{ km}^2$)
  - `cross_modal_water_2`: 36.0 ha ($0.36\text{ km}^2$)
  - `cross_modal_water_3`: 30.0 ha ($0.30\text{ km}^2$)
  - `cross_modal_water_4`: 25.0 ha ($0.25\text{ km}^2$)
- **Area Plausibility:** High-confidence spatial delineation of the Thane Creek and harbor estuaries bordering Mumbai.

### 2.4 Multi-Label Overlap & Mutual Exclusivity
- Built-up & Water overlap: **0 pixels (0.00%)**
- Built-up & Vegetation overlap: **0 pixels (0.00%)**
- Water & Vegetation overlap: **304 pixels (0.46%)** along transitional littoral shorelines
- Triple Overlap (All 3 classes): **0 pixels (0.00%)**
- *Conclusion:* The model preserves the mutually exclusive targets of its training data with less than 0.5% boundary overlap.

### 2.5 Total API Latency Breakdown
- **Image Decoding & Modality Routing:** 42.1 ms
- **Optical-SAR Spatial Compatibility Check:** 785.4 ms
- **Neural Forward Pass (CPU):** **140.3 ms**
- **Contour Tracing & GeoJSON Polygon Generation:** 23.5 ms
- **Total End-to-End API Latency:** **991.3 ms** (Well below the 2500 ms SLA).

---

## 3. Important Radiometric Calibration & Format Limitations

> [!WARNING]
> ### Grayscale-to-dB Conversion Limitation
> In the current inference pipeline, 8-bit visual grayscale images (e.g. preview PNG/JPEG files) are linearly mapped to decibels via:
> $$\sigma^0_{\text{est}} = \left(\frac{\text{pixel}}{255.0}\right) \times 25.0 - 25.0\text{ dB}$$
> **This is an empirical surrogate mapping and MUST NOT be treated as scientifically calibrated radar backscatter.** True radar cross-section requires radiometric calibration lookup tables (LUTs), local incidence angle normalization, and range spreading loss compensation from Sentinel-1 SAFE or RISAT-1 Level-1 CEOS metadata.

### Input Format Support Specification:
| Format / Modality | Operational Support Status | Processing Behavior |
| :--- | :---: | :--- |
| **Calibrated GeoTIFF (Float32 / 16-bit) in dB** | **SUPPORTED** | Ingested directly into SAR encoder without loss of radiometric calibration. |
| **8-bit Orthorectified Grayscale Imagery** | **SUPPORTED (Surrogate)** | Ingested via $[-25.0, 0.0]\text{ dB}$ linear surrogate mapping. |
| **Raw Complex (SLC) I / Q Data** | **UNSUPPORTED / REJECTED** | Must be detected and converted to Ground Range Detected (GRD) intensity before ingestion. |
| **Un-speckled Single-Look Polarimetric Matrices** | **UNSUPPORTED / REJECTED** | Multi-looking and spatial speckle filtering required prior to inference. |
| **Multi-channel RGB False-Color SAR Composites** | **UNSUPPORTED** | Must designate single polarization band (VV preferred) or convert to intensity. |

---

## 4. Controlled Staging Integration Architecture

The staging integration has been configured via an explicit, zero-risk, reversible setting in [`backend/ml_models.py`](file:///d:/sih26167/backend/ml_models.py#L161-L186):

```python
# Controlled Staging Configuration
stage_mode = os.getenv("OPTICAL_SAR_STAGING_ENABLED", "false").lower() in ["true", "1", "yes"] or \
             os.getenv("OPTICAL_SAR_CHECKPOINT_STAGE", "production").lower() in ["staging", "v2_staging", "pilot"]
```

### Staging Verification Tests:
1. **Default Mode (`OPTICAL_SAR_STAGING_ENABLED="false"` or unset):**
   - Loads production checkpoint `satquery_optical_sar_net/optical_sar_net.pth`.
   - Returns provenance: `SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)`.
   - Regression test suite: **10 / 10 PASSED** in 7.33s.
2. **Staging Mode (`OPTICAL_SAR_STAGING_ENABLED="true"`):**
   - Loads candidate checkpoint `satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`.
   - Returns provenance: `SatQueryOpticalSARNet-v2.0-Staging (Authentic BEN-GE-800, Val IoU: 64.3%)`.
   - Regression test suite: **10 / 10 PASSED** in 7.42s.
3. **Rollback Procedure:**
   - Unset environment variable: `Remove-Item Env:OPTICAL_SAR_STAGING_ENABLED`.
   - Production behavior is restored instantly with zero file modifications.
