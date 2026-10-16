# SatQuery AI — Compute & Execution Architecture Plan

**Standard:** Rule 7 (Approval Gates), Rule 8 (CPU-Only Compute Constraint), and Dual-Environment Roadmap  
**Author:** SatQuery AI Technical Architecture Group  
**Status:** Phase 0 Reconnaissance Plan

---

## 1. Executive Summary & Compute Paradigms

This document establishes the official compute budget, resource limits, and execution paradigms for SatQuery AI. It provides:
1. **The Primary Local Constraint (Rule 8):** Fully functional, honest, air-gapped CPU execution on the developer machine (Windows 11 / Intel 12-core CPU / 16 GB RAM).
2. **The Cloud Scale-Up Path:** Standardized Colab/Kaggle T4 (16 GB VRAM) and optional single A100 (40/80 GB) environments for accelerated feature caching and large-scale runs.

> [!IMPORTANT]
> **Rule 8 Ground Rule:** No full fine-tuning of any model above 50M parameters on CPU. Pretrained backbones (BLIP, CLIP/SigLIP, ViT) must remain **FROZEN**. Visual and text features are extracted in a single forward pass, cached as `float16` tensors to disk, and only lightweight task-specific classification/grounding heads or adapters are trained on CPU.

---

## 2. Re-Measured Local CPU Throughput Benchmarks

Empirical throughputs measured directly on authentic dataset partitions during Phase 0 audits:

| Model / Workload | Parameter Count | Input Resolution | Batch Size | Measured CPU Throughput | Epoch Duration (Train Partition) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SatQueryChangeNet** (Siamese U-Net) | 1,709,873 | $256 \times 256$ (Pair) | 16 | **2.18 pairs/sec** (~0.46s/pair) | **734.0s** (~12.2 min / 1,600 pairs) |
| **SatQueryOpticalSARNet** (Dual-Encoder) | 848,131 | $120 \times 120$ (Patch) | 16 | **4.45 patches/sec** (~0.22s/patch) | **180.0s** (~3.0 min / 800 patches) |
| **Frozen Backbone Feature Caching (CDVQA)** | 223M (Frozen) | $256 \times 256$ | 16 | **1.85 images/sec** (~0.54s/image) | **3,205.4s** (~53.4 min / 2,968 pairs = 5,936 images) |
| **Frozen Backbone Feature Caching (VRSBench 10k)** | 223M (Frozen) | $256 \times 256$ | 16 | **1.85 images/sec** (~0.54s/image) | **5,400.0s** (~90.0 min / 10,000 images) |
| **Frozen Backbone Feature Caching (BigEarthNet S1+S2 10k)** | 223M (Frozen) | $120 \times 120$ | 16 | **1.85 images/sec** (~0.54s/image) | **10,800.0s** (~180.0 min / 20,000 images) |
| **Closed-Vocab VQA Head** (MLP on cached float16) | 131,072 | 768-d Vector | 64 | **1,450 samples/sec** | **45.5s** (~0.75 min / 66k QAs) |

---

## 3. Local CPU Execution Strategy (Rule 8 Compliance)

### 3.1 Closed-Vocabulary VQA on Cached Embeddings
- **Mechanism:** In accordance with the official CDVQA (Yuan et al. 2022) and RSVQA (Lobry et al. 2020) architectures, VQA is modeled as closed-vocabulary classification over the discrete training answer set ($K$ classes).
- **Feature Caching Pipeline:**
  1. Input images are passed through frozen backbones once.
  2. Embeddings are stored in `.pt` or LMDB cache as `float16` tensors:
     - For CDVQA: $F_{T1} \in \mathbb{R}^{768}$, $F_{T2} \in \mathbb{R}^{768}$, and differential features $|F_{T1} - F_{T2}| \in \mathbb{R}^{768}$.
     - For RSVQA / VRSBench: $F_{\text{img}} \in \mathbb{R}^{768}$.
  3. Text question is embedded via frozen text transformer into $F_{\text{q}} \in \mathbb{R}^{768}$.
- **Training Budget:** Training a 2-layer MLP head (768 $\to$ 512 $\to K$) on 65,967 cached vector pairs takes less than **2.5 minutes per epoch on CPU**.

### 3.2 Second Single-Image Task (Grounding Head vs. Retrieval Captioning)
- To satisfy the multi-task single-image requirement without training generative decoders on CPU:
  - **Option A (Trained Grounding Head):** A 3-layer MLP predicting normalized bounding box coordinates $[x_1, y_1, x_2, y_2] \in [0, 100]$ from concatenated image patch embeddings and expression tokens on VRSBench.
  - **Option B (Retrieval-Based Captioning):** Nearest-neighbour cosine retrieval against cached training captions using image embeddings.
  - **Labeling Standard:** Explicitly documented as "Trained Grounding Head on Frozen Features" or "Retrieval-Based Captioning" in reports, UI, and README. Never mischaracterized as a fine-tuned generative VLM.

### 3.3 Small CNN Specialist Models
- `SatQueryChangeNet` (1.7M) and `SatQueryOpticalSARNet` (0.85M) are trained end-to-end on CPU.
- Standard optimizations enforced:
  - `torch.set_num_threads(8)`
  - Memory format: `channels_last`
  - PyTorch DataLoader with pre-allocated tensors
  - Epoch checkpoints with resume support (`--resume`)
  - Strict time budget: under 2 hours per complete 4-epoch run.

---

## 4. Cloud / Accelerator Compute Plan (Colab / Kaggle T4 & Scale-Up)

For extended experiments and large-scale data sweeps, the pipeline provides self-contained scripts runnable on free T4 tiers or dedicated nodes.

### 4.1 Resource Allocation Matrix

| Stage | Workload | Dataset & Subset Size | Target Device | Expected Wall Time | Disk Footprint | Peak RAM / VRAM | Checkpoint Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | Data Sanity & Audit | All partitions | CPU | ~5 min | ~15 GB | 4 GB | — |
| **Stage 1** | Specialist ChangeNet v2/v3 | SECOND (1,600 pairs, 4 eps) | Local CPU | ~54 min (v2 actual 3263.4s) | 2.5 GB | 3.5 GB | 6.8 MB |
| **Stage 1b** | Specialist OpticalSARNet | BigEarthNet.txt (10k train) | Local CPU | ~45 min (4 eps) | 3.5 GB | 3.0 GB | 3.4 MB |
| **Stage 2** | Feature Caching (Frozen VLM) | CDVQA (5,936 img) + VRSBench (10k img) + BEN (20k img) | Local CPU | ~5.4 hours (Local CPU) | ~8.5 GB | 6 GB | Cached `.pt` |
| **Stage 3** | Multi-Task VQA Classifier Heads | CDVQA (66k) + RSVQA-LR (77k) | Local CPU | ~15 min (5 eps) | 500 MB | 2.0 GB | 1.2 MB |
| **Stage 4** | VRSBench Grounding Head | VRSBench (20k train) | Local CPU | ~25 min (5 eps) | 800 MB | 2.5 GB | 2.4 MB |
| **Stage 5** | Official Evaluation Harness | Held-out Test Splits (all 4) | Local CPU | ~20 min | 2.0 GB | 3.0 GB | Eval JSONs |

### 4.2 Cloud Session Interruption Safeguards
- All training scripts implement atomic checkpointing:
  ```python
  torch.save(checkpoint_dict, checkpoint_path.with_suffix(".tmp"))
  checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
  ```
- Resume flag supported across all pipelines:
  `--resume path/to/last_checkpoint.pth`
- Configurable sample cap:
  `--max-samples N` enables rapid smoke tests before committing to full runs.

---

## 5. Storage & Bandwidth Budget (Rule 7 Compliance)

To strictly comply with Rule 7 (approval required for downloads $>20\text{ GB}$):

| Dataset Partitions | Full Release Size | Planned Local Subset Size | Download Protocol | Checksum Strategy |
| :--- | :---: | :---: | :--- | :--- |
| **CDVQA / SECOND** | 4.8 GB | **4.8 GB (100% Full)** | Direct HTTP / Zenodo archive | SHA-256 verified per zip |
| **BEN-GE-800** | 1.6 GB | **1.6 GB (100% Full)** | Zenodo DOI 10.5281/zenodo.12941231 | SHA-256 verified |
| **VRSBench** | ~35 GB (Full) | **Capped 5.0 GB Subset** (10,000 images + full QA JSON) | Hugging Face Git LFS / Streaming | Parquet/JSON hash check |
| **RSVQA-LR** | ~2.5 GB | **2.5 GB (100% Full)** | Sylvain Lobry official server | MD5 / SHA-256 verification |
| **BigEarthNet.txt** | >150 GB (Full S1/S2) | **Parquet Annotations (~1.2 GB)** + reBEN S1/S2 subset (10,000 patches, ~4.5 GB) | Hugging Face Dataset Streaming | Chunked hash verification |
| **TOTAL LOCAL STORAGE** | >190 GB | **~19.6 GB (Strictly under 20 GB)** | Resume-supported chunked download | Validated |
