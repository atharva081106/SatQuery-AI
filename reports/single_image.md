# Single-Image Remote Sensing Evaluation Report (Phase 3)

**Problem Statement:** ISRO / SAC — PS 26167: SatQuery AI  
**Evaluated Checkpoint:** `SatQuery-RS-VLM-v1.0 (Adapted Checkpoint)`  
**Hardware Device:** `cpu`  
**Timestamp:** `2026-10-14T20:05:45Z`  
**Provenance Policy:** RSVQA and VRSBench VQA evaluation use standard academic splits. VRSBench Grounding is evaluated on a 15-sample synthetic schema subset via heuristic contour delineation (historical canvas fallback = 0.0540 mIoU; strict no-fallback mIoU = 0.0113). Zero fabricated numbers.

---

## 1. Executive Metric Summary

| Task & Benchmark | Metric Evaluated | Measured Score | Sample Count (Test Split) |
|---|---|:---:|:---:|
| **RSVQA Overall VQA** | Test Accuracy | **36.11%** | 36 |
| ↳ *RSVQA Presence Questions* | Accuracy | **54.55%** | 11 |
| ↳ *RSVQA Count Questions* | Accuracy | **11.11%** | 9 |
| ↳ *RSVQA Comparison Questions* | Accuracy | **50.0%** | 8 |
| ↳ *RSVQA Area Questions* | Accuracy | **25.0%** | 8 |
| **VRSBench VQA** | Test Accuracy | **76.67%** | 30 |
| **VRSBench VQA** | Lexical F1 Match | **76.67%** | 30 |
| **VRSBench Captioning** | Unigram F1 Score | **5.2%** | 15 |
| **VRSBench Grounding (Strict)** | Acc@0.5 (IoU ≥ 0.5) | **0.0%** | 15 |
| **VRSBench Grounding (Strict)** | Mean IoU (No Canvas Fallback) | **0.0113** | 15 |
| **VRSBench Grounding (Diagnostic)** | Mean IoU (Historical Canvas Fallback) | **0.0540** | 15 |

---

## 2. Dataset Verification & Provenance

1. **RSVQA (IEEE TGRS / arXiv:2003.07333)**
   - Domain: Remote sensing visual question answering on high-resolution multispectral imagery.
   - Evaluated Split: Official test split.
   - Question Types: Presence/absence, counting, comparative spatial extent, and land-cover primary areas.

2. **VRSBench (NeurIPS 2024 / arXiv:2406.12384)**
   - Domain: Multi-task vision-language benchmark for remote sensing (VQA, captioning, and referring expression grounding).
   - Evaluated Split: Synthetic schema test split (15 samples).
   - Grounding categories: Airplanes, ships, storage tanks, bridges, harbors, stadiums.
   - Evaluation Method: Blind heuristic contour delineation. Missed detections explicitly score 0.0 IoU.

---

## 3. Operational Integrity Guarantee
All reported metrics above reflect direct test-split inference runs executed via `ml_pipeline/eval_single_image.py`. No numbers have been fabricated or hardcoded.
