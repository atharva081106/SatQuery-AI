# OpticalSARNet Production Promotion Decision & Operational Readiness Review

**Document Version:** 1.0  
**Date:** 2026-10-15  
**Review Type:** Stage-Gate Decision Document  
**Model Name:** `SatQueryOpticalSARNet`  
**Candidate Artifact:** [`ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth)  
**Candidate Checkpoint SHA-256:** `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2`  
**Production Checkpoint Path:** [`ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth) (Active & Preserved)  
**Current Live Routing:** **LEGACY PRODUCTION CHECKPOINT REMAINS ACTIVE**  

---

## 1. Formal Stage-Gate Decision

### **FINAL DECISION: PROMOTE (Staging Verified — Pending User Approval Gate)**

The evaluation committee concludes that **`SatQueryOpticalSARNet-AuthenticPilot-v1`** is technically, empirically, and operationally superior to the legacy prototype checkpoint, and recommends promotion to production once authorized.

---

## 2. Summary of Evidence Supporting Promotion

| Criterion | Evaluation Requirement | Measured Result | Evidence Document | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Authentic Provenance** | No synthetic/heuristic labels | 550 authentic S1/S2 patches trained on official ESA WorldCover 10m raster ground truth. | [`authentic_pilot_integrity_audit.md`](file:///d:/sih26167/reports/training_migration/authentic_pilot_integrity_audit.md) | **PASS** |
| **Zero Data Leakage** | No train/val/test overlap | 0 shared patch IDs across splits; per-patch radiometric scaling; fixed 3-epoch budget. | [`authentic_pilot_integrity_audit.md`](file:///d:/sih26167/reports/training_migration/authentic_pilot_integrity_audit.md) | **PASS** |
| **Held-Out Test Accuracy** | Comparable benchmark metric | **63.26% Global mIoU**, **75.40% Mean F1** on 129 untouched authentic test patches (+28.84% gain). | [`authentic_pilot_test_results.md`](file:///d:/sih26167/reports/training_migration/authentic_pilot_test_results.md) | **PASS** |
| **Generalization Stability** | Minimal overfitting gap | Validation mIoU: 64.26% $\to$ Test mIoU: 63.26% ($\Delta = -1.00\%$). | [`authentic_pilot_test_results.md`](file:///d:/sih26167/reports/training_migration/authentic_pilot_test_results.md) | **PASS** |
| **Interface Compatibility** | 100% drop-in forward pass | Exact state dict match (849,603 parameters; 0 missing/unexpected keys). | [`staging_promotion_audit.md`](file:///d:/sih26167/reports/training_migration/staging_promotion_audit.md) | **PASS** |
| **Regression Testing** | Zero pipeline regression | **14 / 14 tests PASSED** in backend pytest suite. | [`staging_promotion_audit.md`](file:///d:/sih26167/reports/training_migration/staging_promotion_audit.md) | **PASS** |
| **False Alarm Reduction** | Improved tactical clarity | Spurious contour count reduced by 70-75% across fixed test cases. | [`old_vs_new_inference_comparison.md`](file:///d:/sih26167/reports/training_migration/old_vs_new_inference_comparison.md) | **PASS** |

---

## 3. Critical Geographic & Sensor Domain Boundary

> [!IMPORTANT]
> ### Crucial Boundary Statement: BigEarthNet/ESA WorldCover Validation Does NOT Establish ISRO/SAC Accuracy
> 
> While the candidate model demonstrates high empirical accuracy on the European Space Agency's BigEarthNet-MM / ESA WorldCover benchmark (63.26% test mIoU across European biomes), **this validation DOES NOT establish certified operational accuracy on Indian Space Research Organisation (ISRO) or Space Applications Centre (SAC) mission payloads (e.g. Cartosat-2S, Cartosat-3, RISAT-1, EOS-04).**
> 
> **Specific Domain Limitations:**
> 1. **Sensor Radiometry Differences:** Sentinel-1 (C-band, 10m GRD) and Sentinel-2 (MSI, 10m) have distinct radiometric calibrations, point spread functions, and orbital geometries compared to RISAT-1 Circular/Linear polarization and Cartosat sub-meter panchromatic/multispectral instruments.
> 2. **Geographical Generalization:** The training distribution covers European Corine Land Cover regimes (temperate forests, managed pastures, European urban topologies). Delineation over dense Indian urban informal settlements, monsoon-inundated agricultural deltas, and Western Ghats tropical canopies has not been validated against ground truth.
> 3. **Uncalibrated Grayscale Caveat:** When evaluating 8-bit visual grayscale images (e.g. `risat1_cband_radar_sar.png`), the model relies on a surrogate decibel mapping ($[-25.0, 0.0]\text{ dB}$) rather than calibrated radar cross-section ($\sigma^0$). True mission certification requires Level-1 calibrated products.

---

## 4. Current Deployment & Routing State

In accordance with strict change control instructions:
- **Live production routing has NOT been altered.**
- The production checkpoint [`ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth) (SHA-256: `f8c61f87...`) remains the active default in `backend/ml_models.py`.
- The candidate checkpoint is isolated in staging at [`ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth).
- Controlled staging activation is enabled exclusively via the environment variable `OPTICAL_SAR_STAGING_ENABLED="true"`.
- **Zero user-facing deployments have taken place.**

---

## 5. Rollback & Promotion Execution Protocol

When user approval is granted to proceed with live promotion, the transition can be executed safely through the following reversible procedure:

### Promotion Steps:
1. **Archive Production Baseline:**
   ```powershell
   Copy-Item ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net_legacy_v1.0.pth
   ```
2. **Promote Staging Checkpoint:**
   ```powershell
   Copy-Item ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth
   ```
3. **Update Default Production Metadata in [`backend/ml_models.py`](file:///d:/sih26167/backend/ml_models.py):**
   Set default validation IoU to `0.6426` and update provenance label to:
   `SatQueryOpticalSARNet-v2.0 (Dual-Encoder Optical-SAR Fusion, Authentic BEN-GE Val IoU: 64.3%)`.
4. **Instant Rollback (If needed):**
   ```powershell
   Copy-Item ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net_legacy_v1.0.pth ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth
   ```

---

## 6. Review Stop Notice

Staging verification is complete. The system is stopped and awaiting review before any live production deployment or track progression.
