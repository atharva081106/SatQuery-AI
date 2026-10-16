# CDVQA Bi-Temporal Change Detection & VQA Evaluation Report

**Benchmark:** CDVQA (arXiv:2112.06343 / IGARSS 2022)  
**Execution Timestamp:** 2026-10-15 18:32:13 UTC  
**Hardware Device:** CPU  
**Evaluated Test Samples:** 100 paired bi-temporal scenes  
**Runtime:** 316.50 seconds  

---

## 1. Executive Summary & Verification

| Task | Primary Metric | Measured Value | Benchmark Target / Reference |
| :--- | :--- | :--- | :--- |
| **Change Delineation Mask** | **Mean IoU (mIoU)** | **84.65%** | Siamese Difference U-Net baseline |
| **Change Delineation Mask** | **Dice F1 Score** | **0.9137** | Harmonic mean of precision & recall |
| **Change Delineation Mask** | **Precision** | **0.8549** | True changed pixels / predicted changed |
| **Change Delineation Mask** | **Recall** | **0.9893** | True changed pixels / ground truth changed |
| **Bi-Temporal CDVQA** | **Overall Normalized Acc** | **29.00%** | CDVQA Test Split |
| **Bi-Temporal CDVQA** | **Average Per-Type Acc** | **14.93%** | Mean across question categories |
| **Bi-Temporal CDVQA** | **Lexical F1** | **0.2700** | Token overlap similarity |

---

## 2. CDVQA Accuracy by Question Type

| Question Category | Samples Tested | Exact Match Acc (%) | Normalized Acc (%) | Lexical F1 |
| :--- | :--- | :--- | :--- | :--- |
| `change_or_not` | 41 | 51.22% | 51.22% | 0.5122 |
| `increase_or_not` | 12 | 16.67% | 16.67% | 0.1667 |
| `decrease_or_not` | 12 | 33.33% | 33.33% | 0.3333 |
| `smallest_change` | 7 | 0.0% | 0.0% | 0.0 |
| `largest_change` | 7 | 0.0% | 0.0% | 0.0 |
| `change_to_what` | 6 | 0.0% | 0.0% | 0.0 |
| `change_ratio` | 4 | 0.0% | 0.0% | 0.0 |
| `change_ratio_types` | 11 | 0.0% | 18.18% | 0.0 |

---

## 3. Verified Sample Predictions

| Pair ID | Question | Ground Truth | VLM Prediction | Mask IoU | Match Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `07308` | Did the areas of non-vegetated ground surface change? | **yes** | yes | 96.8% | ✅ Match |
| `07308` | Did the regions of trees change? | **no** | yes | 96.8% | ❌ Mismatch |
| `07308` | Have the areas of low vegetation changed? | **no** | yes | 96.8% | ❌ Mismatch |
| `07308` | Have the regions of water changed? | **no** | yes | 96.8% | ❌ Mismatch |
| `07308` | Have the regions of playgrounds changed? | **no** | yes | 96.8% | ❌ Mismatch |
| `07308` | Did the regions of non-vegetated ground surface change in the pre-event image? | **yes** | yes | 96.8% | ✅ Match |

---

## 4. System Co-Registration & Diagnostic Verification

In addition to test-split metrics, the bi-temporal change pipeline includes real-time spatial integrity checks:
1. **Identical Pair Guardrail:** SHA-256 and zero-pixel-difference detection rejects identical images with `REJECTED_IDENTICAL_IMAGES` and derived confidence 0.04 (`1/25`).
2. **Disparate Region Guardrail:** SIFT with RANSAC consensus (Lowe ratio: 0.75, RANSAC: 5.0) rejects mismatched image pairs.
3. **True Demonstration Pair (`uttarakhand_pre/post`):** Co-registration succeeds with verified inliers and spatial coherence score of 0.59.
4. **GeoJSON Vector Generation:** Detected change contours are converted to valid GeoJSON polygons with quadrant localization, pixel bounding boxes, and area in square kilometers and hectares.
