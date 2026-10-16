# SatQuery AI — Final Pre-Production Promotion Gate & Readiness Audit

**Document Version:** 1.0  
**Audit Date:** 2026-10-15  
**Review Type:** Pre-Production Stage-Gate Operational Decision  
**Target Architecture:** `SatQueryOpticalSARNet`  
**Candidate Checkpoint:** [`ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net_authentic_pilot/optical_sar_net.pth)  
**Candidate SHA-256:** `93ec9d95d0a0d5217356626633d52491005dba7e61a02097e580d86c0d5f1dc2`  
**Production Checkpoint:** [`ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`](file:///d:/sih26167/ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth) (Active & Preserved)  
**Production SHA-256:** `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405`  
**Current Live Routing:** **LEGACY PRODUCTION CHECKPOINT ACTIVE (Default)**  

---

## 1. Formal Stage-Gate Decision & Recommendation

### **RECOMMENDATION: HOLD ON FULL PRODUCTION PROMOTION**
### **OPERATIONAL STATUS: READY-FOR-CONTROLLED-ROLLOUT (STAGING CANARY ONLY, CONDITIONED ON PRE-REQUISITES)**

Following comprehensive technical inspection of the SAR input pipeline, training split mechanics, and real inference paths, the review board issues a **HOLD** on unconditional direct production replacement, while certifying the candidate as **READY-FOR-CONTROLLED-ROLLOUT** in staging/canary mode subject to two mandatory technical pre-requisites:

1. **Pre-requisite 1 (SAR Calibration Contract):** The API must explicitly flag 8-bit surrogate visual inputs versus scientifically calibrated float32 $\sigma^0$ GeoTIFFs, eliminating the silent misrepresentation of uncalibrated grayscale as physical radar backscatter.
2. **Pre-requisite 2 (Parent-Granule-Disjoint Validation):** The model must be validated on a parent-granule-disjoint test partition to confirm that the reported 63.26% test mIoU does not suffer a catastrophic collapse due to spatial autocorrelation or scene leakage.

**Zero production changes have been executed.** Both checkpoints remain preserved bit-for-bit, and the legacy production model continues to serve all live traffic by default.

---

## 2. Measurable Pre-Production Promotion Gates

To ensure objective, reproducible, and verifiable operational readiness, seven measurable quality gates have been codified.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SATQUERY AI PROMOTION GATES                          │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ Gate                 │ Target Metric        │ Candidate Result              │
├──────────────────────┼──────────────────────┼───────────────────────────────┤
│ 1. Segmentation      │ Micro mIoU ≥ 60.0%   │ 63.26% (Test)                 │
│ 2. Per-Class         │ Built ≥ 25%, Veg ≥ 70│ Veg 76.5%, Wat 58.7%, Bld 29.8│
│ 3. SAR Calibration   │ Explicit flags / Floa│ CONDITIONAL (Audit findings)  │
│ 4. Latency           │ Pipeline ≤ 250 ms    │ 16.9 ms (CPU forward pass)    │
│ 5. Regression        │ 100% test pass rate  │ 14 / 14 tests PASSED (100%)   │
│ 6. Rollback          │ Zero downtime revert │ Configured & Verified         │
│ 7. ISRO/SAC Boundary │ Clear disclaimer     │ Documented & Enforced         │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

### Gate 1: Segmentation Quality & Generalization
- **Gate 1.1 — Global Micro mIoU:** $\ge 60.0\%$ on authentic held-out test data.  
  *Candidate Result:* **63.26%** ($\mathbf{+28.84\%}$ over legacy prototype baseline of 34.42%). **[PASS]**
- **Gate 1.2 — Validation-to-Test Generalization Gap:** $|\text{mIoU}_{\text{val}} - \text{mIoU}_{\text{test}}| \le 3.0\%$.  
  *Candidate Result:* **$1.00\%$** ($64.26\% \to 63.26\%$). **[PASS]**
- **Gate 1.3 — Sample-Level Macro mIoU:** $\ge 45.0\%$ across patches where classes are present.  
  *Candidate Result:* **46.72%** (Test). **[PASS]**
- **Gate 1.4 — Parent-Granule-Disjoint Generalization:** Disjoint test mIoU $\ge 55.0\%$.  
  *Candidate Result:* **PENDING** (Scene overlap audit revealed 93.7% parent granule sharing in current split; formal disjoint experiment planned in [`granule_disjoint_evaluation_plan.md`](file:///d:/sih26167/reports/training_migration/granule_disjoint_evaluation_plan.md)). **[HOLD / PENDING]**

### Gate 2: Per-Class Performance & Mask Integrity
- **Gate 2.1 — Vegetation Land-Cover:** IoU $\ge 70.0\%$, F1 $\ge 80.0\%$.  
  *Candidate Result:* **IoU = 76.54%, F1 = 86.71%**. **[PASS]**
- **Gate 2.2 — Water Bodies & Wetlands:** IoU $\ge 50.0\%$, F1 $\ge 65.0\%$.  
  *Candidate Result:* **IoU = 58.74%, F1 = 74.01%**. **[PASS]**
- **Gate 2.3 — Built-up Urban Fabric:** IoU $\ge 25.0\%$, F1 $\ge 40.0\%$.  
  *Candidate Result:* **IoU = 29.84%, F1 = 45.96%**. **[PASS]**
- **Gate 2.4 — Mask Non-Saturation & Finite Bounds:**  
  All predicted probabilities must be bounded $\in [0.0, 1.0]$, finite (zero `NaN` or `Inf`), and non-saturating ($<95\%$ total image coverage across typical mixed scenes).  
  *Candidate Result:* Verified across 129 test patches; maximum non-vegetation mask coverage $61.2\%$. **[PASS]**

### Gate 3: Radiometric & Input Calibration Contract
- **Gate 3.1 — Explicit Radiometric Status in API:**  
  The API response must distinguish whether the input was a calibrated float32 GeoTIFF or an 8-bit visual grayscale surrogate.  
  *Candidate Result:* **FAIL (Requires Remediation)**. Currently, `CrossModalAnalysis` outputs `"RISAT-1 / SENTINEL-1 C-BAND SAR (RADAR BACKSCATTER)"` for all inputs without indicating calibration status.
- **Gate 3.2 — Float32 Calibrated Pass-Through:**  
  Calibrated float32 GeoTIFFs must not be degraded by lossy uint8 intermediate quantization.  
  *Candidate Result:* **FAIL (Requires Remediation)**. Line 164 of `backend/model_interfaces.py` clips float32 to uint8 before passing to the model.
- **Gate 3.3 — Unsupported Format Rejection:**  
  Raw SLC (complex I/Q), full polarimetric matrices, and uncalibrated digital numbers must be explicitly rejected with informative error messages.  
  *Candidate Result:* **CONDITIONAL**. Decoded by rasterio without guardrails.

### Gate 4: End-to-End Latency & Pipeline Performance
- **Gate 4.1 — Model Forward Latency:** $\le 50.0\text{ ms}$ on standard CPU hardware for $256 \times 256$ inputs.  
  *Candidate Result:* **16.9 ms** (Intel Core i5/i7 workstation). **[PASS]**
- **Gate 4.2 — Total End-to-End Latency:** $\le 250.0\text{ ms}$ including GeoTIFF decoding, IHS fusion, contour extraction, and GeoJSON polygon serialization for $512 \times 512$ inputs.  
  *Candidate Result:* **142.3 ms** total pipeline time. **[PASS]**

### Gate 5: Regression & Fallback Integrity
- **Gate 5.1 — Automated Test Suite:** $100\%$ pass rate across backend test suites.  
  *Candidate Result:* **14 / 14 tests PASSED** (10/10 in `backend/test_cross_modal.py`, 4/4 in `backend/test_change_pipeline.py`). **[PASS]**
- **Gate 5.2 — Graceful Heuristic Fallback:**  
  If the checkpoint is corrupted, unreadable, or missing, the system must transparently fall back to the IHS Heuristic Fusion Engine without throwing unhandled HTTP 500 errors.  
  *Candidate Result:* Tested and verified; logs warning and sets `execution_path: "HEURISTIC_FALLBACK"`. **[PASS]**

### Gate 6: Reversibility & Rollback Protocol
- **Gate 6.1 — Instantaneous Reversible Control:**  
  Promotion and rollback must be controllable via an explicit environment variable without recompiling or redeploying code.  
  *Candidate Result:* Fully implemented via `OPTICAL_SAR_STAGING_ENABLED="false"` or `OPTICAL_SAR_CHECKPOINT_STAGE="production"`. **[PASS]**
- **Gate 6.2 — Bit-for-Bit Baseline Preservation:**  
  The legacy production checkpoint must remain intact and verified by cryptographic SHA-256 hash.  
  *Candidate Result:* Legacy hash verified as `f8c61f87da6be5c8de516651561804c256005394bb379d34319d68fa69d2e405`. **[PASS]**

### Gate 7: Scientific & Geographic Boundary Enforcement
- **Gate 7.1 — ISRO/SAC Mission Scope Boundary:**  
  System documentation and model cards must state explicitly that European BigEarthNet-MM / ESA WorldCover validation **does not establish certified operational accuracy on ISRO/SAC mission payloads (Cartosat-2S, Cartosat-3, RISAT-1, EOS-04)** or Indian subcontinent biomes.  
  *Candidate Result:* Formal disclaimer incorporated in model registry and technical audits. **[PASS]**

---

## 3. Pre-Production Audit Scorecard

| Gate | Category | Description | Status | Blocking Action |
| :---: | :--- | :--- | :---: | :--- |
| **1.1** | Quality | Global Micro mIoU $\ge 60\%$ (63.26% achieved) | **PASS** | None |
| **1.2** | Quality | Generalization Gap $\le 3\%$ (1.00% achieved) | **PASS** | None |
| **1.3** | Quality | Sample Macro mIoU $\ge 45\%$ (46.72% achieved) | **PASS** | None |
| **1.4** | Quality | Parent-Granule-Disjoint Generalization ($\ge 55\%$) | **HOLD** | Execute 30-min disjoint experiment |
| **2.1** | Per-Class | Vegetation IoU $\ge 70\%$ (76.54% achieved) | **PASS** | None |
| **2.2** | Per-Class | Water Bodies IoU $\ge 50\%$ (58.74% achieved) | **PASS** | None |
| **2.3** | Per-Class | Built-up Fabric IoU $\ge 25\%$ (29.84% achieved) | **PASS** | None |
| **2.4** | Integrity | Finite bounds & non-saturation verified | **PASS** | None |
| **3.1** | Radiometry | Explicit calibration flags in API metadata | **FAIL** | Add `sar_input_contract` metadata |
| **3.2** | Radiometry | Direct float32 pass-through for GeoTIFFs | **FAIL** | Bypass lossy uint8 clipping |
| **3.3** | Radiometry | Rejection of complex SLC / polarimetric data | **HOLD** | Add validation guards in `detect_image_modality` |
| **4.1** | Latency | Model forward latency $\le 50\text{ ms}$ (16.9 ms) | **PASS** | None |
| **4.2** | Latency | End-to-end API latency $\le 250\text{ ms}$ (142.3 ms) | **PASS** | None |
| **5.1** | Regression | Backend test suite pass rate $100\%$ (14/14) | **PASS** | None |
| **5.2** | Fallback | Graceful IHS heuristic fallback verified | **PASS** | None |
| **6.1** | Reversibility| Zero-downtime env switch configured | **PASS** | None |
| **6.2** | Preservation | Legacy checkpoint preserved bit-for-bit | **PASS** | None |
| **7.1** | Boundary | ISRO/SAC mission limitations documented | **PASS** | None |

---

## 4. Remediation Checklist Prior to Live Deployment

Before the transition from staging to live production routing can be authorized, the following three targeted pull-request items must be completed:

```markdown
- [ ] PR-1: Radiometric Contract Update (Gate 3.1 & 3.2)
      Modify backend/model_interfaces.py to add `sar_input_contract` to inference_metadata.
      Update after_label to distinguish calibrated backscatter from surrogate grayscale.
      Add direct float32 GeoTIFF tensor ingestion bypass.

- [ ] PR-2: Format Guardrails (Gate 3.3)
      Add guard in detect_image_modality to reject complex SLC and polarimetric matrices.

- [ ] PR-3: Parent-Granule-Disjoint Verification (Gate 1.4)
      Run partition_granule_disjoint.py and evaluate checkpoint on 120 disjoint test patches.
      Verify that disjoint test mIoU meets the ≥ 55.0% threshold.
```

---

## 5. Promotion Decision Summary & Next Steps

1. **Final Gate Verdict:**
   - **Full Production Replacement:** **HOLD** (Awaiting completion of Remediation PR-1, PR-2, and PR-3).
   - **Staging / Canary Access:** **READY-FOR-CONTROLLED-ROLLOUT** (Authorized for internal evaluation and staging validation via `OPTICAL_SAR_STAGING_ENABLED="true"`).
2. **Current System State:**
   - Legacy production checkpoint is actively serving all default traffic.
   - Candidate checkpoint is preserved in staging.
   - All tests pass (14/14).
   - System execution stopped. Awaiting user review and direction.
