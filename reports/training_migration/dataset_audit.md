# SatQuery AI — Dataset Audit for Training Migration
**Project:** SIH 2026 PS26167
**Report date:** 2026-10-15
**Status:** READ-ONLY AUDIT — no data downloaded, no code modified

> All facts in this report are verified directly from repository source files.
> Estimates and assumptions are explicitly labelled [ESTIMATE] or [ASSUMPTION].

---

## 1. Environment Baseline (Verified)

| Property | Value |
|---|---|
| Python | 3.10.11 (CPython, MSC v.1929) |
| PyTorch | 2.13.0+cpu |
| CUDA available | No — CPU-only training |
| RAM total | 15 GB |
| RAM available (idle) | ~2 GB |
| CPU logical cores | 12 |
| Drive C: free | ~11.2 GB |
| Drive D: free | ~119.5 GB |
| HF offline mode | TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1 |

---

## 2. Current Training Data — Status per Model

### 2.1 SatQuery-RS-VLM-v1.0 (BLIP-based VQA)

Training script: ml_pipeline/train_rs_vlm.py
Checkpoint: ml_pipeline/checkpoints/satquery_rs_vlm/model.safetensors (1,378 MB)

| Stage | Intended Dataset | Current Implementation | Authentic Data Downloaded? |
|---|---|---|---|
| 1 – Image-Text Alignment | BigEarthNet.txt (arXiv:2603.29630) | SYNTHETIC — noise-texture PNGs + template captions | No |
| 2 – VQA Fine-Tuning | RSVQA + VRSBench VQA | SYNTHETIC — template Q&A on random uniform images | No |
| 3 – Grounding Conditioning | VRSBench Grounding | SYNTHETIC — deterministic bboxes on noise images | No |

Current checkpoint provenance: Adapted from Salesforce/blip-vqa-base (cached locally), then
fine-tuned on the above synthetic prototypes. Has never seen authentic remote-sensing pixels.

---

### 2.2 SatQueryOpticalSARNet-v1.0 (Optical-SAR Fusion)

Training script: ml_pipeline/train_cross_modal.py
Checkpoint: ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth (3.29 MB, 849,603 params)

| Aspect | Current Implementation |
|---|---|
| Dataset | BigEarthNetDataset -> create_sample_bigearthnet_subset() on first run |
| S2 optical input | 128x128 RGB PNG Gaussian noise, resized to 256x256 at runtime |
| S1 SAR input | 128x128x2 float32 .npy, uniform random in [-25, 0] dB |
| Target masks | Heuristic from CLC label list + SAR/optical thresholds; NOT pixel-annotated GT |
| Loss | BCEWithLogitsLoss + MultiLabelDiceLoss |
| Classes | 3 independent multi-label channels: Built-up, Water, Vegetation |
| Authentic data downloaded? | No |

NOTE: The inference SAR normalization bug (grayscale uint8 -> dB) was fixed in a prior session.
The training pipeline uses float32 numpy arrays directly and is unaffected.

---

### 2.3 SatQueryChangeNet-v1.0 (Siamese Change Detection)

Training script: ml_pipeline/train_change_detector.py
Checkpoint: ml_pipeline/checkpoints/satquery_change_net/change_net.pth (6.56 MB, 7.8M params)

| Aspect | Current Implementation |
|---|---|
| Dataset | CDVQADataset -> create_sample_cdvqa_subset() on first run |
| T1/T2 images | 256x256 RGB PNG with deterministic Gaussian noise |
| Change masks | Binary 0/255 PNG; 80% samples have a rectangular changed region |
| Loss | BCE + 0.5 x Dice |
| Authentic data downloaded? | No |

---

## 3. Dataset Loader Audit

### 3.1 BigEarthNet.txt (bigearthnet_loader.py)

| Property | Value |
|---|---|
| Reference | arXiv:2603.29630 (Herzog, Adler, Demir et al., 2026) |
| License | CDLA-Permissive-1.0 / CC BY 4.0 |
| Modality | Sentinel-1 SAR (VV/VH float32 .npy) + Sentinel-2 RGB (.png) + text captions |
| Official classes | 19-class CLC taxonomy — correctly coded in CLC_19_CLASSES |
| Cache path | ml_pipeline/dataset_cache/bigearthnet/ |
| Official S2 patch size | 120x120 or 256x256 px |
| Current synthetic shape | 128x128 RGB (S2), 128x128x2 float32 (S1) |
| Schema compatibility | JSON annotation format matches official spec |
| Preprocessing at training | S2 /255.0 -> [3,256,256]; S1 VV channel unsqueeze -> [1,256,256] |

Migration requirement: Real BigEarthNet.txt tiles use per-patch directory layout. A one-off
conversion script is needed to populate the expected annotation JSON. The Dataset class needs
no changes.

---

### 3.2 RSVQA (rsvqa_loader.py)

| Property | Value |
|---|---|
| Reference | arXiv:2003.07333 / IEEE TGRS 2020 (Lobry, Marcos, Tuia) |
| License | CC BY 4.0 |
| Official question types | presence, count, comparison, area — all coded |
| Annotation schema | images.json, questions.json, answers.json — matches RSVQA-LR layout |
| Key names | id, img_id, question, type, answer — match official release |
| Answer vocabulary | Open-ended short answers; no restriction in loader |
| Cache path | ml_pipeline/dataset_cache/rsvqa/ |

Migration requirement: Download RSVQA-LR or RSVQA-HR from rsvqa.sylvainlobry.com.
Extract images into Images/. JSON files already match official RSVQA-LR naming.
No loader changes needed.

---

### 3.3 VRSBench (vrsbench_loader.py)

| Property | Value |
|---|---|
| Reference | arXiv:2406.12384 / NeurIPS 2024 (Li, Ding, Elhoseiny) |
| License | CC BY-NC-SA 4.0 (non-commercial) |
| Tasks | VQA, Captioning, Grounding |
| Official image source | DOTA, DIOR, HRSC2016 aerial/satellite imagery |
| Grounding bbox format | [xmin, ymin, xmax, ymax] pixel coords — matches official |
| VQA answer format | List of strings per question — matches official HuggingFace card |
| Official download | HuggingFace: xiang709/VRSBench |
| Cache path | ml_pipeline/dataset_cache/vrsbench/ |

Migration requirement: huggingface-cli download xiang709/VRSBench (requires HF token for
non-commercial datasets). A one-shot path-rewriting step needed after download.

---

### 3.4 CDVQA (cdvqa_loader.py)

| Property | Value |
|---|---|
| Reference | arXiv:2112.06343 / IGARSS 2022 (Yuan, Xia et al.) |
| License | Academic Research Use / CC BY 4.0 |
| Tasks | Change VQA + Binary change mask delineation |
| Mask format | Binary PNG, 0=no-change, 255=changed — matches loader |
| Question types | change_type, existence, quantity, attribute — all coded |
| Official download | github.com/YZHJessica/CDVQA |
| Cache path | ml_pipeline/dataset_cache/cdvqa/ |

Migration requirement: Official CDVQA uses its own annotation format. A conversion script
mapping official JSON -> repo cdvqa_annotations.json schema is required.
T1/, T2/, masks/ directory structure matches official CDVQA exactly.

---

## 4. Checkpoint Compatibility Assessment

| Model | Checkpoint Size | Authentic-data retrain needed? | Risk |
|---|---|---|---|
| SatQuery-RS-VLM-v1.0 | 1,378 MB .safetensors | YES — synthetic captions cannot generalize | High |
| SatQueryOpticalSARNet-v1.0 | 3.29 MB .pth | YES — heuristic targets != real annotation | High |
| SatQueryChangeNet-v1.0 | 6.56 MB .pth | YES — rectangular masks bias weights | Medium |
| SatSegNet (segmentation) | 5.6 MB .pth + ONNX | Lower priority — not in SIH benchmark scope | Low |

---

## 5. Summary of Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| No authentic data on disk | All training blocked | User approval then download per plan |
| CPU-only (no CUDA) | ~50-100x slower than GPU | Accept; use PEFT for VLM, small batches |
| Available RAM ~2 GB (idle) | Batch size severely limited | batch_size <= 4 for VLM; <= 8 for lightweight models |
| VRSBench license CC BY-NC-SA | Non-commercial use only | Confirm use within academic/SIH scope |
| D: drive free = 119.5 GB | Dataset storage | Sufficient for prioritised downloads |
