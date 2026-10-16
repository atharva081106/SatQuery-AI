# SatQuery AI — Authentic Dataset Training Plan
**Project:** SIH 2026 PS26167
**Report date:** 2026-10-15
**Status:** PLAN ONLY — awaiting user approval before any downloads or training

---

## Guiding Principles

1. One training pipeline per model — do not mix incompatible schemas or tasks.
2. Official dataset splits must be used as-is; no recombination that leaks test data.
3. All synthetic-prototype data stays in dataset_cache/ for regression reference.
4. Authentic data lands in a parallel directory: ml_pipeline/authentic_data/.
5. Checkpoints are versioned: existing checkpoints are kept; new ones use _v2 suffix.
6. PEFT (LoRA) is preferred for VLM fine-tuning given the CPU-only constraint.

---

## Plan A — Optical-SAR Land Cover (BigEarthNet.txt -> SatQueryOpticalSARNet)

### A1. Dataset preparation (one-time)

Step 1: Download BigEarthNet.txt from https://txt.bigearth.net/
  - The dataset is released as a BigEarthNet archive (~65 GB uncompressed for full set).
  - MINIMUM VIABLE PILOT: Request a 1,000-patch balanced subset. [ESTIMATE: ~500 MB]
  - Target directory: ml_pipeline/authentic_data/bigearthnet/

Step 2: Convert to loader-expected format
  - Write scripts/prepare_bigearthnet_authentic.py (NEW SCRIPT, not production code)
  - For each patch: copy s2_rgb patch as PNG to train/s2_rgb/, copy s1_sar as .npy
  - Generate train/train_annotations.json with fields: patch_id, s1_patch_id,
    s2_image_path, s1_sar_path, caption, clc_labels, split

Step 3: Verify with loader
  - python -c "from ml_pipeline.data.bigearthnet_loader import BigEarthNetDataset;
    ds = BigEarthNetDataset(data_dir='ml_pipeline/authentic_data/bigearthnet', split='train');
    print(len(ds), ds[0]['caption'])"

### A2. Training run

Command:
  cd ml_pipeline
  python train_cross_modal.py --epochs 5 --batch-size 2 --lr 1e-3

Key hyperparameter rationale:
  - batch_size=2: Accommodates 2 GB available RAM (S2 256x256x3 + S1 256x256 + targets)
  - epochs=5: [ESTIMATE] ~1-2 hr per epoch on 1,000 samples, CPU-only
  - lr=1e-3: Kept from existing training; AdamW with weight_decay=1e-4

Acceptance criteria:
  - Val mIoU > 40% on authentic BigEarthNet validation split (vs ~random baseline of <15%)
  - Loss decreasing monotonically across epochs
  - Built-up, Water, Vegetation masks produce visually distinct regions on real Sentinel imagery

Output checkpoint: ml_pipeline/checkpoints/satquery_optical_sar_net_v2/optical_sar_net.pth

### A3. Evaluation split protection

  - Use official BigEarthNet.txt test split as-is — do NOT mix into training
  - Record split provenance in checkpoint metadata

---

## Plan B — Single-Image VQA (RSVQA + VRSBench -> SatQuery-RS-VLM, Stage 2)

### B1. Dataset preparation

Step 1: Download RSVQA-LR from https://rsvqa.sylvainlobry.com/
  - LR dataset: ~1.5 GB (images + annotations)
  - Target directory: ml_pipeline/authentic_data/rsvqa/

Step 2: No schema conversion needed — rsvqa_loader.py already matches RSVQA-LR format
  - Verify: check that images.json, questions.json, answers.json are present

Step 3: (Optional) Download VRSBench VQA split from HuggingFace xiang709/VRSBench
  - Requires HF token + confirmation of CC BY-NC-SA licensing for SIH use
  - [ESTIMATE: ~8 GB for full image set; VQA annotations only = ~10 MB]

### B2. Training strategy — PEFT/LoRA (STRONGLY RECOMMENDED for CPU)

Why PEFT: Full fine-tuning of 361M parameter BLIP model on CPU would take
[ESTIMATE] 100-300 hr per epoch on 10,000+ samples. LoRA targeting attention layers
reduces trainable params to ~2-5M, making 1-2 hr/epoch feasible.

Required package: peft (pip install peft) — NOT currently installed.

LoRA configuration (proposed):
  r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
  lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"

Command (after PEFT is installed):
  cd ml_pipeline
  python train_rs_vlm.py --max_samples 500 --epochs 2 --lr 2e-5 --force_cpu

NOTE: train_rs_vlm.py does not currently implement LoRA. A targeted code addition
to wrap the model with get_peft_model() is required before this step.

Acceptance criteria:
  - RSVQA accuracy (yes/no questions) > 65% on authentic val split
  - RSVQA accuracy (count/comparison) > 50%
  - No test-split contamination (question IDs checked against train set)

Output checkpoint: ml_pipeline/checkpoints/satquery_rs_vlm_v2/

---

## Plan C — Visual Grounding / Captioning (VRSBench -> SatQuery-RS-VLM, Stage 3)

### C1. Dataset preparation

Same VRSBench download from Plan B. The grounding and captioning JSON files are
already present in the HuggingFace dataset.

Step 1: After VRSBench download, run path-rewriting script:
  python scripts/rewrite_vrsbench_paths.py --src <hf_download_dir> \
    --dst ml_pipeline/authentic_data/vrsbench/

### C2. Training run

Grounding is executed in Stage 3 of train_rs_vlm.py — no separate script needed.
Captioning uses the same BLIP model in generation mode (Stage 1 reuse).

Acceptance criteria (grounding):
  - Acc@0.5 IoU > 25% on VRSBench grounding val split [ESTIMATE: baseline from random ~5%]

Acceptance criteria (captioning):
  - CIDEr score > 0.5 on VRSBench captioning val split [ESTIMATE: BLIP zero-shot ~0.3-0.5]

---

## Plan D — Change Detection + Change VQA (CDVQA -> SatQueryChangeNet + VLM)

### D1. Dataset preparation

Step 1: Download CDVQA from https://github.com/YZHJessica/CDVQA
  - Dataset released as Google Drive link; ~2 GB
  - Target directory: ml_pipeline/authentic_data/cdvqa/

Step 2: Write conversion script scripts/prepare_cdvqa_authentic.py
  - Map official CDVQA annotation format to repo cdvqa_annotations.json schema
  - Verify T1/, T2/, masks/ directory structure is preserved

### D2. ChangeNet training

Command:
  cd ml_pipeline
  python train_change_detector.py --epochs 10 --batch-size 4 --max-samples 500

Acceptance criteria:
  - Val IoU > 50% on authentic CDVQA change masks
  - F1 > 0.55 for binary change detection

### D3. Change VQA using BLIP (after Plan B checkpoint)

CDVQA questions are fed through the VLM (train_rs_vlm.py Stage 2 can be extended).
No separate CDVQA-VLM training script currently exists — requires new integration code.

---

## Execution Order (Recommended)

| Priority | Plan | Reason |
|---|---|---|
| 1 | Plan A — Optical-SAR (BigEarthNet) | Smallest model, fastest iteration, no PEFT dependency |
| 2 | Plan D — ChangeNet (CDVQA) | Second smallest model, independent of VLM |
| 3 | Plan B — VQA (RSVQA + VRSBench) | Requires PEFT installation and most resources |
| 4 | Plan C — Grounding/Captioning | Depends on Plan B checkpoint + VRSBench download |

---

## Evaluation Integrity

- Test splits are NEVER used during training. Train/val/test splits come from official releases.
- All synthetic prototype data remains in dataset_cache/ and is NOT mixed with authentic splits.
- Checkpoint provenance (dataset name, split, date, sample count) is written to training_report.json.
- SHA-256 checksums of downloaded archives must be verified against official dataset documentation
  before any training begins.
