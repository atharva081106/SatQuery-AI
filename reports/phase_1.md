# Phase 1 Verification Report: Truth Baseline and Repo Hygiene

**Date:** 2026-10-14  
**Evaluator:** SatQuery AI Engineering  
**Problem Statement:** ISRO / SAC — PS 26167  
**Status:** PASSED  

---

## 1. Summary of Changes

### 1.1 Root Directory Hygiene & Historical Provenance
- Created `reports/` and `docs/` directories.
- Moved 11 root clutter files into `docs/`:
  - `DEMO_SCRIPT_230_PRECISE.md`
  - `DEMO_SCRIPT_2MIN30SEC.md`
  - `DEPLOYMENT_GUIDE.md`
  - `FINAL_DEMO_QUERIES.md`
  - `FINAL_POSTER_CORRECTIONS.md`
  - `FINAL_PROJECT_AUDIT.md`
  - `MODEL_AUDIT_FACTS.json`
  - `MODEL_AUDIT_REPORT.md`
  - `PROJECT_CONTEXT.md`
  - `SIH_2026_PRESENTATION_MASTER_DECK.txt`
  - `VIDEO_PITCH_SCRIPT.md`
- Prepend prominent `> [!WARNING] SUPERSEDED HISTORICAL DOCUMENT` notices to historical audits, decks, and poster guides that contained unmeasured prototype claims, preserving history without misleading judges.

### 1.2 Automated Claims Auditor (`scripts/audit_claims.py`)
- Implemented `scripts/audit_claims.py` to automatically verify:
  1. No literal confidence constants in `backend/` (`grep -rn '"confidence": 0\.' backend`).
  2. `/health` does not declare unverified model backbones (e.g., Florence-2 or ResNet-18).
  3. Frontend copy does not display unbacked claims ("90.2% Accuracy", "Zero FPR", "millimeter precision", "<10ms").
  4. `README.md` is free of unmeasured claims.

### 1.3 Hardcoded Confidence Elimination
- Eliminated all occurrences of hardcoded confidence literals in `backend/` (`model_interfaces.py`, `object_detection.py`, `query_understanding.py`, `agent_controller.py`, `storage_manager.py`).
- All confidences are now derived from:
  - Detection contour solidity and foreground/background contrast (`object_detection.py`).
  - Land-cover feature margin and scene diversity (`model_interfaces.py`).
  - Spatial co-registration coherence scores (`check_spatial_compatibility`).
  - Query intent signal match strength (`query_understanding.py`).

### 1.4 Honest `/health` Endpoint & Provenance
- Refactored `backend/main.py` `/health` endpoint to perform live runtime inspection of loaded models:
  - Reports actual model class (`BlipForQuestionAnswering` or None/fallback).
  - Reports segmentation architecture honestly as `SatSegNet (From-scratch 6-class Multi-Scale U-Net, 482,822 parameters)` rather than the fake "ResNet-18 Backbone".
  - Reports runtime hardware (`cuda` vs `cpu`), offline flag (`TRANSFORMERS_OFFLINE=1`), and cache status.
- Added `/api/reports` endpoint to serve verified execution reports from `reports/*.json` directly to the frontend.

### 1.5 UI Telemetry Alignment
- Updated `frontend/src/app/page.tsx` to remove marketing claims ("90.2% Accuracy", "Zero FPR", "millimeter precision", "<10ms ONNX INT8") and replace them with links to verified telemetry and reports.
- Updated `frontend/src/app/dashboard/page.tsx` to fetch `/api/reports` dynamically and display `"not measured"` for metrics that have not yet undergone formal benchmark evaluation on real data splits.

---

## 2. Verification Evidence

| Acceptance Criterion | Command / Check | Result | Status |
| :--- | :--- | :--- | :--- |
| **No literal confidence constants** | `grep -rn '"confidence": 0\.' backend` | 0 occurrences | **PASSED** |
| **Automated claims auditor** | `python scripts/audit_claims.py` | Exit code 0, all checks passed | **PASSED** |
| **Unit regression testing** | `pytest backend/test_query_routing.py` | 51 passed in 0.45s | **PASSED** |
| **Honest `/health` response** | Code inspection of `backend/main.py` | Accurate model & checkpoint declaration | **PASSED** |
| **Reports API integration** | `GET /api/reports` | Returns `{ status: 'success', reports: {...} }` | **PASSED** |

---

## 3. What Could NOT Be Verified in Phase 1

1. **Model accuracy and IoU on real datasets**: Phase 1 establishes the truth baseline and eliminates fabricated numbers. Real VQA accuracy (RSVQA / VRSBench), change detection accuracy (CDVQA), and optical-SAR fusion metrics await training and evaluation on authentic datasets in Phases 2–5.
2. **GPU latency profiling**: Currently executed on host CPU environment; hardware-accelerated GPU inference profiling will be recorded in Phase 7.
