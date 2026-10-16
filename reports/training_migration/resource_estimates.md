# SatQuery AI — Resource Estimates for Training Migration
**Project:** SIH 2026 PS26167
**Report date:** 2026-10-15
**Disclaimer:** All estimates marked [ESTIMATE] are derived from model parameter counts,
standard throughput benchmarks for CPU PyTorch, and publicly available dataset size
documentation. They are NOT empirically measured and may vary by 2-3x.

---

## 1. Hardware Summary (Verified)

| Resource | Available |
|---|---|
| CPU | 12 logical cores |
| RAM | 15 GB total, ~2 GB idle |
| GPU/CUDA | None — CPU-only |
| Storage (D:) | ~119.5 GB free |
| Storage (C:) | ~11.2 GB free (not used for datasets) |

---

## 2. Download Size Estimates per Dataset

| Dataset | Full Release Size | Minimum Pilot Subset | Notes |
|---|---|---|---|
| BigEarthNet.txt (full) | ~65 GB [ESTIMATE] | ~500 MB (1,000 patches) | Pilot = stratified 1K sample |
| RSVQA-LR | ~1.5 GB | ~1.5 GB (no subset option) | Full LR is manageable |
| RSVQA-HR | ~12 GB | — | Skip for pilot; use LR only |
| VRSBench (images + anno) | ~8 GB [ESTIMATE] | ~2 GB (anno only + 1K images) | HF partial download possible |
| CDVQA | ~2 GB | ~2 GB | Full dataset; release is compact |
| **Total (pilot)** | **~7.5 GB** | | Well within 119.5 GB D: free |

---

## 3. Training Time Estimates (CPU-only, PyTorch 2.13)

### 3.1 SatQueryOpticalSARNet (849K params — lightweight)

| Setting | Value |
|---|---|
| Input shape | [B, 3, 256, 256] optical + [B, 1, 256, 256] SAR |
| Batch size | 2 (constrained by RAM) |
| Forward + backward time per sample | ~0.3 s [ESTIMATE] |
| Samples per epoch (1K pilot train split ~700) | 700 |
| Time per epoch | ~350 batches x 0.3 s x 2 items = ~3.5 min [ESTIMATE] |
| Recommended epochs | 5 |
| **Total training time (pilot)** | **~18 min [ESTIMATE]** |
| RAM peak | ~1.2 GB [ESTIMATE] |

### 3.2 SatQueryChangeNet (7.8M params — Siamese U-Net)

| Setting | Value |
|---|---|
| Input shape | [B, 3, 256, 256] x2 (T1 + T2) |
| Batch size | 4 |
| Forward + backward time per sample | ~0.5 s [ESTIMATE] |
| Samples per epoch (CDVQA pilot ~700) | 700 |
| Time per epoch | ~175 batches x 0.5 s x 4 = ~6 min [ESTIMATE] |
| Recommended epochs | 10 |
| **Total training time (pilot)** | **~60 min [ESTIMATE]** |
| RAM peak | ~2.5 GB [ESTIMATE] |

> [!WARNING]
> ChangeNet RAM peak of 2.5 GB exceeds available idle RAM (~2 GB). Reduce batch_size to 2
> or close background processes before training. Alternatively, add gradient checkpointing.

### 3.3 SatQuery-RS-VLM (BLIP, 361M params — VQA/Grounding)

#### Full fine-tuning (NOT recommended on CPU)

| Setting | Value |
|---|---|
| Input | [1, 3, 384, 384] image + tokenized text |
| Forward + backward per sample (CPU) | ~12-20 s [ESTIMATE] |
| 500 samples x 2 epochs | ~1000 steps |
| **Total time** | **~3.5-5.5 hr [ESTIMATE]** |
| RAM peak | ~6-8 GB [ESTIMATE] — likely exceeds available RAM |

#### PEFT / LoRA fine-tuning (RECOMMENDED)

| Setting | Value |
|---|---|
| Trainable parameters (LoRA r=16) | ~2.4M [ESTIMATE] |
| Forward + backward per sample (CPU, LoRA) | ~4-6 s [ESTIMATE] |
| 500 samples x 2 epochs | ~1000 steps |
| **Total time** | **~1.2-1.7 hr [ESTIMATE]** |
| RAM peak | ~3-4 GB [ESTIMATE] |

> [!IMPORTANT]
> PEFT (pip install peft) is NOT currently installed in the backend venv.
> train_rs_vlm.py does NOT currently implement LoRA — requires targeted code addition
> before VLM training can begin. This is the primary code change needed for migration.

---

## 4. Storage Requirements for Training Artifacts

| Artifact | Size [ESTIMATE] |
|---|---|
| BigEarthNet pilot (1K patches on disk) | ~500 MB |
| RSVQA-LR on disk | ~1.5 GB |
| VRSBench pilot (1K images + 3 JSON) | ~500 MB |
| CDVQA full (T1+T2+masks+JSON) | ~2 GB |
| New checkpoints (v2 per model) | ~1.4 GB (dominated by VLM .safetensors) |
| Training logs and reports | <50 MB |
| **Total new storage needed** | **~6 GB [ESTIMATE]** |

---

## 5. Pilot Acceptance Criteria Summary

| Model | Metric | Target | Baseline (synthetic) |
|---|---|---|---|
| SatQueryOpticalSARNet-v2 | Val mIoU (3-class) | > 40% | ~25% (heuristic) |
| SatQueryChangeNet-v2 | Val IoU (binary change) | > 50% | ~60% (overfitted rectangular masks) |
| SatQuery-RS-VLM-v2 (RSVQA) | Presence/count accuracy | > 65% / > 50% | Not measurable (synthetic) |
| SatQuery-RS-VLM-v2 (grounding) | Acc@0.5 IoU | > 25% | Not measurable (synthetic) |

---

## 6. Recommended Execution Order with Blockers

```
Step 1 [BLOCKER — user approval required]
  Confirm dataset downloads are approved.
  Confirm VRSBench CC BY-NC-SA license is acceptable for SIH use.

Step 2 [NEW — no code change]
  Download RSVQA-LR (~1.5 GB) and CDVQA (~2 GB).
  Download BigEarthNet.txt pilot subset (~500 MB).

Step 3 [NEW SCRIPT — prepare_bigearthnet_authentic.py]
  Convert BigEarthNet patches to loader-expected JSON + directory structure.
  Verify with BigEarthNetDataset loader.

Step 4 [NEW SCRIPT — prepare_cdvqa_authentic.py]
  Convert CDVQA official annotations to cdvqa_annotations.json schema.
  Verify with CDVQADataset loader.

Step 5 [TRAIN — Plan A, ~18 min]
  python ml_pipeline/train_cross_modal.py --epochs 5 --batch-size 2

Step 6 [TRAIN — Plan D ChangeNet, ~60 min]
  python ml_pipeline/train_change_detector.py --epochs 10 --batch-size 2

Step 7 [INSTALL + CODE CHANGE — PEFT]
  pip install peft  (in backend venv)
  Add LoRA wrapper to train_rs_vlm.py

Step 8 [DOWNLOAD — VRSBench pilot, ~500 MB]
  huggingface-cli download xiang709/VRSBench
  python scripts/rewrite_vrsbench_paths.py

Step 9 [TRAIN — Plan B VQA, ~1.5 hr]
  python ml_pipeline/train_rs_vlm.py --max_samples 500 --epochs 2

Step 10 [TRAIN — Plan C Grounding, runs after Step 9]
  (Included in Stage 3 of train_rs_vlm.py — no separate invocation needed)
```

---

## 7. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| RAM OOM during VLM training | High | Use LoRA; batch_size=1; close browser/other apps |
| BigEarthNet pilot not representative | Medium | Use stratified sampling across all 19 CLC classes |
| RSVQA-LR answer space too sparse | Low | Loader uses open-ended answers; no tokenizer padding issue |
| VRSBench download requires HF token | Medium | User creates HF account; use huggingface-cli login |
| CDVQA Google Drive link broken | Low | Mirror at github.com/YZHJessica/CDVQA; contact authors |
| CPU training time exceeds estimate | Medium | Accept; run overnight; use max_samples=200 for quick pilot |
