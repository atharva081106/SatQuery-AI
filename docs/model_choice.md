# SatQuery AI: Vision-Language Model Architecture Selection & Justification

**Document:** `docs/model_choice.md`  
**Problem Statement:** Multimodal Remote Sensing Image Analysis (ISRO / SAC — PS 26167)  
**Target Capabilities:** Single-Image VQA, Earth Observation Scene Captioning, Text-Guided Visual Grounding, Out-of-Domain Guardrails  
**Deployment Constraints:** Air-Gapped / Offline Local Execution (`TRANSFORMERS_OFFLINE=1`), Free-Tier Training (Google Colab T4 / Kaggle P100), Sub-second CPU/Edge Inference  

---

## 1. Candidate Architecture Evaluation

We evaluated five candidate open vision-language architectures against the operational constraints of the ISRO SIH26167 problem statement:

| Architecture | Parameter Count | Pretrained License | Strengths | Limitations in Edge / Air-Gapped Context | Selection Decision |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **Salesforce/BLIP-VQA Base** | **385M** | **BSD-3-Clause** | Lightweight, modular cross-attention, sub-500ms CPU inference, proven remote sensing adaptation (Lobry et al.) | Requires multi-task head adaptation for grounding bounding boxes | **SELECTED** |
| **Microsoft/Florence-2-base** | 232M | MIT | Native bounding box coordinate tokens, compact size | Custom complex vision-token vocabulary; high CPU memory peak during autoregressive generation | Considered as head alternative |
| **Google/PaliGemma-3B** | 2.92B | Gemma License | Strong visual reasoning, multilingual | Requires >8 GB VRAM for inference; unusable on standard commodity edge CPUs | Rejected (Too heavy for CPU edge) |
| **GeoChat (Lien et al., 2024)** | 7.2B | LLaMA-2 License | Full RS domain pretraining | 7B parameters requires high-end server GPU (>=16 GB VRAM); fails air-gapped laptop requirement | Rejected (Excessive VRAM) |
| **RemoteCLIP (Liu et al., 2024)** | 300M | MIT | Excellent retrieval embeddings | Encoder-only contrastive architecture; cannot generate natural language answers without external decoder head | Rejected (No generative QA) |

---

## 2. Selected Architecture: SatQuery RS-Adapted BLIP-VQA Engine

### 2.1 Core Architectural Specifications
- **Backbone Vision Transformer:** ViT-B/16 (12 layers, 768 hidden dimension, 12 attention heads), patch size 16x16, normalized for multispectral/optical bands.
- **Multimodal Text-Image Decoder:** 12-layer BERT-based cross-attention language model with bidirectional image-text interaction and causal language modeling heads.
- **Total Parameters:** ~385M trainable/adapted parameters (~1.5 GB FP32 checkpoint / ~750 MB FP16).
- **Inference Footprint:** ~850 MB system RAM on CPU; capable of execution under `TRANSFORMERS_OFFLINE=1`.

### 2.2 Staged Training Strategy
To avoid catastrophic forgetting and establish genuine domain adaptation across heterogeneous remote sensing sensors:

1. **Stage 1 (Image-Text Land Cover Alignment):**
   - **Dataset:** BigEarthNet.txt (arXiv:2603.29630) Sentinel-2 optical RGB + Sentinel-1 dual-polarization SAR.
   - **Objective:** Cross-modal contrastive alignment and captioning against Corine Land Cover (CLC-19) descriptions.
   - **Loss:** Masked Language Modeling + Image-Text Matching (ITM).

2. **Stage 2 (Remote Sensing VQA Fine-Tuning):**
   - **Datasets:** RSVQA (arXiv:2003.07333) + VRSBench VQA (arXiv:2406.12384).
   - **Objective:** Visual reasoning over remote sensing phenomena: existence/presence, object counting, spatial comparison, and areal extent.
   - **Loss:** Cross-entropy on target answer tokens.

3. **Stage 3 (Visual Grounding & Coordinate Calibration):**
   - **Dataset:** VRSBench Grounding (arXiv:2406.12384).
   - **Objective:** Coordinate token conditioning (`[xmin, ymin, xmax, ymax]`) mapping referring expressions to subpixel spatial vectors and RFC 7946 GeoJSON polygons.

---

## 3. Out-of-Domain Guardrail Specification

A critical failure mode in baseline remote sensing systems is answering out-of-scope questions (e.g. "What is the capital of France?" or "Is there a nuclear submarine?") with generic scene summaries at high false confidence.

SatQuery AI enforces a **two-tier semantic guardrail**:
1. **Domain Intent Classification:** Queries without geospatial, terrain, infrastructural, or environmental context are flagged as `OUT_OF_DOMAIN`.
2. **Explicit Rejection Output:** Rather than hallucinating a land-cover breakdown, the engine explicitly returns:
   > *"Cannot answer from this remote sensing imagery: the query is outside the spatial resolution, spectral capability, or domain of this Earth observation sensor."*
   with a low calibrated uncertainty score ($\le 0.15$), documented in the execution trace.

---

## 4. Hardware Provenance & Reproducibility
- **Training Acceleration:** Supports LoRA / AdamW optimization with `--max-samples` subsetting, fitting within free Google Colab T4 (15 GB VRAM) or Kaggle P100 (16 GB VRAM) GPU allocations.
- **Edge Deployment:** Quantizable to INT8 via PyTorch / ONNX for deterministic offline execution on laptops and tactical edge stations without internet connectivity.
