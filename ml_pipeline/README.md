# SatQuery AI: Remote Sensing ML Pipeline & Data Engineering Guide

**Problem Statement:** Multimodal Remote Sensing Image Analysis (ISRO / SAC — PS 26167)  
**Strict Policy:** Real datasets only. Zero synthetic/random-noise data in training or evaluation.

---

## 1. Supported Benchmarks & Official Specifications

| Dataset | Paper Citation | Modality | Primary Tasks | License |
| :--- | :--- | :--- | :--- | :--- |
| **BigEarthNet.txt** | [arXiv:2603.29630](https://arxiv.org/abs/2603.29630) | Sentinel-1 SAR + Sentinel-2 Multispectral | Cross-modal text-image alignment, CLC-19 land cover retrieval | CDLA-Permissive-1.0 |
| **RSVQA** | [arXiv:2003.07333](https://arxiv.org/abs/2003.07333) | Optical (Sentinel-2 LR / Aerial HR) | Remote Sensing Visual Question Answering (presence, count, comp, area) | CC BY 4.0 |
| **VRSBench** | [arXiv:2406.12384](https://arxiv.org/abs/2406.12384) | High-Resolution Satellite | VQA, Visual Captioning, Text-Guided Visual Grounding (`[xmin, ymin, xmax, ymax]`) | CC BY-NC-SA 4.0 |
| **CDVQA** | [arXiv:2112.06343](https://arxiv.org/abs/2112.06343) | Bi-Temporal Image Pairs (T1/T2) | Bi-Temporal Change Question Answering, Binary Change Masks | CC BY 4.0 |

---

## 2. Resource Requirements (Disk & VRAM)

| Configuration | Disk Footprint | Peak RAM | Peak GPU VRAM | Recommended Execution Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Quick Validation (`--max-samples 100`)** | **~25 MB** | **< 2 GB** | **< 3 GB (CPU runnable)** | Local Laptop CPU / Free Colab |
| **Colab / Kaggle Free Tier (`--max-samples 2000`)** | **~1.5 GB** | **~6 GB** | **~6.5 GB (T4 / P100)** | Google Colab Free Tier (T4 GPU) |
| **Full Benchmark Training (All splits)** | **~45 GB** | **~16 GB** | **~14 GB (A100 / RTX 4090)** | Dedicated Cloud Compute |

---

## 3. Data Loaders & CLI Usage

All loaders reside in `ml_pipeline/data/` and support `--max-samples`, automatic disk caching, and PyTorch `DataLoader` compatibility.

### 3.1 BigEarthNet.txt Loader
```bash
python ml_pipeline/data/bigearthnet_loader.py --split train --max-samples 100
```
- Modalities: Sentinel-2 RGB (`s2_image`), Sentinel-1 dual-polarization SAR (`s1_sar`).
- Annotations: Corine Land Cover (CLC-19) multi-labels and multimodal natural language captions.

### 3.2 RSVQA Loader
```bash
python ml_pipeline/data/rsvqa_loader.py --split train --max-samples 120
```
- Questions parsed from `questions.json`, answers from `answers.json`, linked to `Images/`.
- Question types: `presence`, `count`, `comparison`, `area`.

### 3.3 VRSBench Loader
```bash
# VQA task
python ml_pipeline/data/vrsbench_loader.py --task vqa --split train --max-samples 100

# Visual Captioning task
python ml_pipeline/data/vrsbench_loader.py --task captioning --split train --max-samples 100

# Visual Grounding task
python ml_pipeline/data/vrsbench_loader.py --task grounding --split train --max-samples 100
```

### 3.4 CDVQA Loader
```bash
python ml_pipeline/data/cdvqa_loader.py --split train --max-samples 100
```
- Yields `t1_image`, `t2_image`, `diff_image`, `change_mask`, `question`, and `answer`.

---

## 4. Data Sanity Verification

To inspect all 4 datasets, verify class balances, and export 5 random samples per split:
```bash
python ml_pipeline/data_sanity.py
```
This generates the audited report at:
`reports/data_sanity.md`

---

## 5. Free Colab / Kaggle Execution Protocol

To train on a free GPU without local overhead:

1. Open a Google Colab notebook with GPU accelerator (`Runtime > Change runtime type > T4 GPU`).
2. Clone the repository and install dependencies:
   ```bash
   !git clone https://github.com/atharva081106/SatQuery-AI.git
   %cd SatQuery-AI
   !pip install -r backend/requirements.txt
   !pip install -r ml_pipeline/requirements.txt
   ```
3. Run the data sanity check:
   ```bash
   !python ml_pipeline/data_sanity.py
   ```
4. Train with sample capping to stay within free GPU limits:
   ```bash
   !python ml_pipeline/train_vqa.py --data_dir ml_pipeline/dataset_cache/rsvqa --output_dir ml_pipeline/checkpoints/vlm_rsvqa --batch_size 8 --epochs 5 --max_samples 1500
   ```
5. Download the resulting checkpoint directory (`ml_pipeline/checkpoints/vlm_rsvqa/`) and place it in the local workspace.
