# CDVQA Bi-Temporal Change Detection & VQA Evaluation Report

**Benchmark:** CDVQA (arXiv:2112.06343 / IGARSS 2022)  
**Execution Timestamp:** 2026-10-31 23:06:08 UTC  
**Hardware Device:** CPU  
**Evaluated Test Samples:** 100 paired bi-temporal scenes  
**Runtime:** 233.11 seconds  

---

## 1. Executive Summary & Verification

| Task | Primary Metric | Measured Value | Benchmark Target / Reference |
| :--- | :--- | :--- | :--- |
| **Change Delineation Mask** | **Mean IoU (mIoU)** | **85.32%** | Siamese Difference U-Net baseline |
| **Change Delineation Mask** | **Dice F1 Score** | **0.9175** | Harmonic mean of precision & recall |
| **Change Delineation Mask** | **Precision** | **0.8556** | True changed pixels / predicted changed |
| **Change Delineation Mask** | **Recall** | **0.9971** | True changed pixels / ground truth changed |
| **Bi-Temporal CDVQA** | **Overall Normalized Acc** | **62.00%** | CDVQA Test Split |
| **Bi-Temporal CDVQA** | **Average Per-Type Acc** | **62.88%** | Mean across question categories |
| **Bi-Temporal CDVQA** | **Lexical F1** | **0.6100** | Token overlap similarity |

---

## 2. CDVQA Accuracy by Question Type

| Question Category | Samples Tested | Exact Match Acc (%) | Normalized Acc (%) | Lexical F1 |
| :--- | :--- | :--- | :--- | :--- |
| `change_or_not` | 41 | 56.1% | 56.1% | 0.561 |
| `increase_or_not` | 12 | 83.33% | 83.33% | 0.8333 |
| `decrease_or_not` | 12 | 66.67% | 66.67% | 0.6667 |
| `smallest_change` | 7 | 42.86% | 42.86% | 0.4286 |
| `largest_change` | 7 | 57.14% | 57.14% | 0.5714 |
| `change_to_what` | 6 | 83.33% | 83.33% | 0.8333 |
| `change_ratio` | 4 | 50.0% | 50.0% | 0.5 |
| `change_ratio_types` | 11 | 54.55% | 63.64% | 0.5455 |

---

## 3. Verified Sample Predictions

| Pair ID | Question | Ground Truth | VLM Prediction | Mask IoU | Match Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `07308` | Did the areas of non-vegetated ground surface change? | **yes** | yes | 97.8% | ✅ Match |
| `07308` | Did the regions of trees change? | **no** | no | 97.8% | ✅ Match |
| `07308` | Have the areas of low vegetation changed? | **no** | no | 97.8% | ✅ Match |
| `07308` | Have the regions of water changed? | **no** | no | 97.8% | ✅ Match |
| `07308` | Have the regions of playgrounds changed? | **no** | no | 97.8% | ✅ Match |
| `07308` | Did the regions of non-vegetated ground surface change in the pre-event image? | **yes** | no | 97.8% | ❌ Mismatch |

---

## 4. System Co-Registration & Diagnostic Verification

In addition to test-split metrics, the bi-temporal change pipeline includes real-time spatial integrity checks:
1. **Identical Pair Guardrail:** SHA-256 and zero-pixel-difference detection rejects identical images with `REJECTED_IDENTICAL_IMAGES` and derived confidence 0.04 (`1/25`).
2. **Disparate Region Guardrail:** SIFT with RANSAC consensus (Lowe ratio: 0.75, RANSAC: 5.0) rejects mismatched image pairs.
3. **True Demonstration Pair (`uttarakhand_pre/post`):** Co-registration succeeds with verified inliers and spatial coherence score of 0.59.
4. **GeoJSON Vector Generation:** Detected change contours are converted to valid GeoJSON polygons with quadrant localization, pixel bounding boxes, and area in square kilometers and hectares.
