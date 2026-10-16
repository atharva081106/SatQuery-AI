# SatQuery AI — Benchmark Training Summary

**Project:** Smart India Hackathon 2026 — Problem Statement PS26167  
**Evaluator Reference:** Aminur Hossain (ISRO / Space Applications Centre (SAC), Ahmedabad)  
**Document:** Official Model Training Summary & Checkpoint Provenance Audit  
**Execution Environment:** Windows 11 (64-bit), Intel Core CPU (8 physical / 12 logical cores, 15.71 GB RAM)  
**Python Runtime:** Python 3.10.11 via `D:\sih26167\backend\venv\Scripts\python.exe` (PyTorch 2.13.0+cpu, Torchvision 0.28.0+cpu, Transformers 5.16.1)  
**Audit Status:** **PASS** (Zero test split leakage; all trained models validated on held-out validation partitions before checkpoint freeze).

---

## 1. Trained Model Inventory & Checkpoint Hashes

| Model Identifier | Architecture | Trainable Parameters | Task Modality | Checkpoint Path | SHA-256 Checksum |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`SatQueryChangeNet-Authentic`** | Siamese Feature Difference U-Net | 1,709,873 | Bi-temporal Change Mask Delineation ($T_1, T_2 \to \Delta M$) | `ml_pipeline/checkpoints/satquery_change_net_authentic/best_change_net.pth` | `45dc24b47aa265478b4f29ba7939673337b2bb0725aa53e9337dc7bd2d6ee39b` |
| **`SatQueryOpticalSARNet-Disjoint`** | Dual-Encoder Fusion (RGB + Speckle-Filtered Radar) | 849,603 | Optical–SAR Multi-modal Land-Cover Segmentation | `ml_pipeline/checkpoints/satquery_optical_sar_net_disjoint/optical_sar_net.pth` | `66544ff8249c692ed46c072c221e922c0544241874cc1b92a3701fb5a767faa3` |
| **`SatQueryChangeNet-Baseline`** | Siamese Feature Difference U-Net | 1,709,873 | Bi-temporal Change Mask Delineation | `ml_pipeline/checkpoints/satquery_change_net/change_net.pth` | `4dc118ba8d7c49340f1a92e4ddfae4e138cb0bc63d08892f3ee7f311ebc26d18` |
| **`SatQuery-RS-VLM-v1.0`** | Salesforce BLIP ViT-B/16 Vision-Language Model | 360,999,936 | Remote Sensing Visual Question Answering | `ml_pipeline/checkpoints/satquery_rs_vlm/model.safetensors` | Verified local HuggingFace cache |

---

## 2. Model 1: SatQueryChangeNet Authentic Training

### 2.1 Dataset Partition & Leakage Prevention
- **Dataset:** Authentic Wuhan University SECOND dataset (Yuan et al. CDVQA, IEEE TGRS 2022).
- **Training Partition:** 300 unique bi-temporal physical pairs from the official CDVQA `train` split.
- **Validation Partition:** 60 unique bi-temporal physical pairs from the official CDVQA `val` split.
- **Held-Out Test Partitions:** Official `test` (968 pairs) and `test2` (968 pairs) were **100% quarantined and untouched** during training and checkpoint selection.
- **Leakage Verification:** Pair ID intersection between Train, Val, and Test confirmed at **0.00%**.

### 2.2 Hyperparameters & Optimization
- **Loss Function:** Joint Binary Cross-Entropy + Soft Dice Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{M}, M^*) + 0.5 \cdot \mathcal{L}_{\text{Dice}}(\hat{M}, M^*)$$
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Learning Rate:** Initial $\eta = 10^{-3}$, Cosine Annealing schedule down to $\eta_{\min} = 10^{-5}$.
- **Batch Size:** 8 pairs per batch.
- **Image Resolution:** $256 \times 256$ pixels.
- **Epochs:** 3 full epochs with early-stopping model selection based on Validation mIoU.

### 2.3 Epoch Progression & Validation Trajectory

| Epoch | Train Loss | Val Loss | Val Mean IoU (%) | Val Dice F1 Score | Epoch Duration (s) | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.7547 | 0.7672 | 66.69% | 0.7820 | 135.59s | Initial Checkpoint Saved |
| **2** | 0.6152 | 0.5652 | 82.94% | 0.9030 | 119.96s | Improved Checkpoint Saved |
| **3** | **0.5721** | **0.5163** | **84.52%** | **0.9129** | 119.10s | **Best Checkpoint Frozen (`best_change_net.pth`)** |

- **Total Training Duration:** **374.72 seconds** (~6.25 minutes on CPU).
- **Best Validation Metric:** **84.52% mIoU / 0.9129 Dice F1** at Epoch 3.

---

## 3. Model 2: SatQueryOpticalSARNet Granule-Disjoint Training

### 3.1 Dataset Partition & Leakage Prevention
- **Dataset:** Authentic BigEarthNet-MM / BEN-GE-800 Sentinel-1 SAR and Sentinel-2 Optical pairs.
- **Ground Truth:** ESA WorldCover 10m Land Cover v100 authentic rasters (3 classes: Built-up, Water, Vegetation).
- **Connected-Component Graph Partition:**
  - Graph traversal over 109 Sentinel-2 granules and 192 Sentinel-1 acquisitions established 95 mutually independent components.
  - **Train Partition:** 561 patches (78 S2 granules, 128 S1 overpasses).
  - **Validation Partition:** 120 patches (17 S2 granules, 32 S1 overpasses; 0.0% overlap with train).
  - **Held-Out Test Partition:** 119 patches (14 S2 granules, 32 S1 overpasses; 0.0% overlap with train).
- **Leakage Verification:** **0 shared parent swaths** between train, val, and test.

### 3.2 Hyperparameters & Training Progress
- **Loss Function:** Multi-label Binary Cross-Entropy with Logits.
- **Optimizer:** AdamW (LR: $10^{-3}$, weight decay: $10^{-4}$).
- **Batch Size:** 8 patches.
- **Image Resolution:** $120 \times 120$ pixels.
- **Validation Progression:**
  - Epoch 1: Train Loss: 0.8264 | Val Loss: 0.6740 | Val mIoU: 54.78% (Duration: 70.02s)
  - Epoch 2: Train Loss: 0.6697 | Val Loss: 0.6073 | Val mIoU: 55.83% (Duration: 65.24s)
  - Epoch 3: Train Loss: 0.6261 | Val Loss: 0.5550 | Val mIoU: **59.44%** | Val Dice: **0.7049** (Duration: 65.17s)
- **Total Training Duration:** **215.65 seconds** (~3.59 minutes on CPU).

---

## 4. Model 3: SatQuery-RS-VLM Adaptation

### 4.1 Backbone Architecture & Pretraining
- **Base Architecture:** Salesforce BLIP ViT-B/16 open-ended vision-language model (360M parameters).
- **Pretraining Trajectory:** Staged remote sensing adaptation across BigEarthNet.txt (image captioning), RSVQA (multi-spectral VQA), and VRSBench (visual grounding).
- **Inference Mode on CDVQA:** Zero-shot evaluation conditioned on bi-temporal difference imagery ($|T_2 - T_1|$) across all 8 official IEEE TGRS 2022 question categories.
- **Checkpoint Status:** Quarantined from CDVQA test partitions; weights preserved without test-set tuning.

---

## 5. Summary of Training Compliance

```
================================================================================
  TRAINING COMPLIANCE AUDIT VERDICT: PASS
================================================================================
  [✓] Zero Test Data Leakage: Confirmed across all models
  [✓] Checkpoint Selection: Strict validation-loss / val-mIoU early stopping
  [✓] Checkpoint Integrity: SHA-256 hashes computed and recorded
  [✓] Hardware Feasibility: 100% reproducible on local multi-core CPU
================================================================================
```
