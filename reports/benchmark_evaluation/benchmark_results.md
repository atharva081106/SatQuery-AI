# SatQuery AI — Official Benchmark Evaluation Results Report

**Project:** Smart India Hackathon 2026 — Problem Statement PS26167  
**Organization:** Indian Space Research Organisation (ISRO) / Space Applications Centre (SAC), Ahmedabad  
**Evaluation Scope:** Reproducible benchmark evaluations across RSVQA, VRSBench, CDVQA, and Optical–SAR Joint Analysis  
**Audit Date:** October 15, 2026  
**Auditor:** SatQuery AI Forensic Verification & Benchmark Assessment Team  

---

## Executive Summary

This report documents the official benchmark evaluation results for SatQuery AI. Evaluations were executed across all four required domains:
1. **RSVQA** (single-image remote-sensing visual question answering, arXiv:2003.07333)
2. **VRSBench** (VQA, captioning, and visual grounding, arXiv:2406.12384 / NeurIPS 2024)
3. **CDVQA** (bi-temporal change detection and change VQA, arXiv:2112.06343 / IGARSS 2022)
4. **Optical–SAR Joint Analysis** (BigEarthNet-MM schema, arXiv:2603.29630, and ISRO demo data audit)

In accordance with strict provenance standards:
- All evaluations were conducted using the models, weights, and configurations stored locally.
- **No data was trained or tuned on the test set**, and no post-test threshold manipulation was performed.
- Every individual prediction, ground-truth reference, token similarity score, bounding box, and mask metric has been preserved in `reports/benchmark_evaluation/predictions/`.
- Full execution commands and runtime traces are logged in `reports/benchmark_evaluation/logs/eval_execution.log`.
- Because local datasets represent **synthetic schema prototypes** (15–36 test samples) created for architecture validation, evaluations are assigned status **PARTIAL**, while the authentic ISRO evaluation is marked **BLOCKED** due to zero reference ground-truth annotations.

---

## 1. Master Benchmark Results Table

| Benchmark | Split | Sample Count | Metric | SatQuery AI Result | Published Baseline / SOTA | Comparability | Status | Evidence Source |
|---|---|:---:|---|:---:|:---:|---|:---:|---|
| **RSVQA** | Synthetic Schema Test Split | 36 QA pairs (18 images) | Overall Answer Accuracy | **36.11%** | 78.48% (CNN+GRU) / 82.52% (Fusion) | NOT DIRECTLY COMPARABLE (Synthetic schema test split vs 10k real Sentinel-2 QA pairs) | **PARTIAL** | [rsvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/rsvqa_predictions.json) |
| ↳ *Presence* | Synthetic Schema Test Split | 11 questions | Presence Accuracy | **54.55%** | 87.89% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [rsvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/rsvqa_predictions.json) |
| ↳ *Count* | Synthetic Schema Test Split | 9 questions | Count Accuracy | **11.11%** | 68.32% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [rsvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/rsvqa_predictions.json) |
| ↳ *Comparison* | Synthetic Schema Test Split | 8 questions | Comparison Accuracy | **50.00%** | 84.62% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [rsvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/rsvqa_predictions.json) |
| ↳ *Area* | Synthetic Schema Test Split | 8 questions | Area Accuracy | **25.00%** | 72.88% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [rsvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/rsvqa_predictions.json) |
| **VRSBench VQA** | Synthetic Schema Test Split | 30 questions (15 images) | Overall Answer Accuracy<br>Mean Lexical Token F1 | **76.67%**<br>**76.67%** | 64.81% (BLIP-2 zero-shot)<br>71.30% (VRS-LLM fine-tuned) | NOT DIRECTLY COMPARABLE (Synthetic schema test split vs 109k authentic VQA pairs) | **PARTIAL** | [vrsbench_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/vrsbench_predictions.json) |
| **VRSBench Captioning** | Synthetic Schema Test Split | 15 images | Unigram Lexical F1<br>Precision / Recall | **5.20%** F1<br>(3.90% P / 7.78% R) | 14.80 (BLEU-4, BLIP-2)<br>18.20 (BLEU-4, VRS-LLM)<br>33.90 (CIDEr, LLaVA-1.5) | NOT DIRECTLY COMPARABLE (Short synthetic prompt vs 52-word human references) | **PARTIAL** | [vrsbench_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/vrsbench_predictions.json) |
| **VRSBench Visual Grounding** | Synthetic Schema Test Split | 15 target bboxes | Acc@0.5 (IoU ≥ 0.50)<br>Mean IoU (Heuristic) | **0.00%** Acc@0.5<br>**0.0113** mIoU | 18.42% Acc@0.5 (GLIP zero-shot)<br>36.80% Acc@0.5 (VRS-LLM fine-tuned) | NOT DIRECTLY COMPARABLE (Blind contour baseline on Gaussian noise arrays) | **PARTIAL** | [vrsbench_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/vrsbench_predictions.json) |
| **VRSBench Visual Grounding (Neural)** | Synthetic Schema Test Split | 15 target bboxes | Acc@0.5 (IoU ≥ 0.50)<br>Mean IoU (OWL-ViT) | **0.00%** Acc@0.5<br>**0.0042** mIoU | 14.20%–19.80% (GeoChat zero-shot)<br>28.54% (UNINEXT) | NOT DIRECTLY COMPARABLE (Zero-shot ViT-B/32 evaluated on Gaussian noise arrays) | **PARTIAL** | [vrsbench_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/vrsbench_predictions.json) |
| **CDVQA Change Delineation** | Synthetic Schema Test Split | 15 bi-temporal pairs | Mean IoU %<br>Mean Dice F1<br>Precision / Recall | **15.16%** mIoU<br>**0.2549** Dice F1<br>(15.16% P / 80.00% R) | 54.20% mIoU (FC-EF baseline)<br>64.30% mIoU (STANet SOTA)<br>78.20% F1 | NOT DIRECTLY COMPARABLE (15 synthetic pairs with box deltas vs 600 authentic CDVQA pairs) | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| **CDVQA Change VQA** | Synthetic Schema Test Split | 15 questions | Overall Answer Accuracy<br>Average Per-Type Accuracy<br>Mean Lexical Token F1 | **40.00%** Overall<br>**25.00%** Avg Per-Type<br>**0.4333** Token F1 | 68.45% Overall (Dual-ResNet)<br>72.30% Overall (Siamese VQA) | NOT DIRECTLY COMPARABLE (15 synthetic questions vs 20,000 multi-temporal QA pairs) | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| ↳ *Existence* | Synthetic Schema Test Split | 6 questions | Existence Accuracy | **100.00%** | 82.10% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| ↳ *Change Type* | Synthetic Schema Test Split | 3 questions | Type Accuracy | **0.00%** | 61.40% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| ↳ *Quantity* | Synthetic Schema Test Split | 3 questions | Quantity Accuracy | **0.00%** | 58.20% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| ↳ *Attribute* | Synthetic Schema Test Split | 3 questions | Attribute Accuracy | **0.00%** | 72.10% | NOT DIRECTLY COMPARABLE | **PARTIAL** | [cdvqa_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/cdvqa_predictions.json) |
| **Optical–SAR Joint Segmentation** | BigEarthNet-Schema Test Split | 15 S1/S2 pairs | Overall Mean IoU %<br>Mean Dice F1 | **21.22%** mIoU<br>**0.2524** Dice F1 | 88.50% mAP (BigEarthNet-MM S1+S2)<br>48.50%–54.00% mIoU (LoveDA S1+S2) | NOT DIRECTLY COMPARABLE (Evaluated against synthetic pseudo-mask targets on 15 test pairs) | **PARTIAL** | [optical_sar_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/optical_sar_predictions.json) |
| ↳ *Built-up Extraction* | BigEarthNet-Schema Test Split | 15 pairs | Built-up IoU / F1 | **6.98%** IoU / **0.1034** F1 | 52.40% IoU (LoveDA Urban) | NOT DIRECTLY COMPARABLE | **PARTIAL** | [optical_sar_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/optical_sar_predictions.json) |
| ↳ *Water Body Extraction* | BigEarthNet-Schema Test Split | 15 pairs | Water IoU / F1 | **20.22%** IoU / **0.2534** F1 | 74.10% IoU (LoveDA Water) | NOT DIRECTLY COMPARABLE | **PARTIAL** | [optical_sar_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/optical_sar_predictions.json) |
| ↳ *Vegetation Extraction* | BigEarthNet-Schema Test Split | 15 pairs | Vegetation IoU / F1 | **36.48%** IoU / **0.4004** F1 | 61.80% IoU (LoveDA Vegetation) | NOT DIRECTLY COMPARABLE | **PARTIAL** | [optical_sar_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/optical_sar_predictions.json) |
| **Co-Registration Guardrail** | Demo Pairs | 2 pairs | Coherence / Verification Rate | **100.00%** Verification<br>(True: 0.88 / Disp: 0.04) | 100.00% Target | Direct Verification Protocol | **PASS** | [optical_sar_predictions.json](file:///d:/sih26167/reports/benchmark_evaluation/predictions/optical_sar_predictions.json) |
| **ISRO / SAC Optical–SAR Data** | None | 0 samples | Extraction IoU / QA Acc | **BLOCKED** | N/A | NOT COMPARABLE (Zero ground-truth reference masks or verified QA pairs exist) | **BLOCKED** | [reports/isro_evaluation.json](file:///d:/sih26167/reports/isro_evaluation.json) |

---

## 2. Detailed Technical & Execution Context

### 2.1 Hardware, Checkpoints & Inference Environment
- **Host Platform:** Windows 11 (64-bit)
- **Runtime Hardware:** Intel Core CPU (Host PyTorch CPU backend, CUDA unavailable)
- **PyTorch Version:** `2.13.0+cpu`
- **Transformers Version:** `5.16.1`
- **OpenCV Version:** `5.0.0`
- **Execution Mode:** 100% Offline (`TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HOME=d:\sih26167\.huggingface_cache`)
- **Evaluated Checkpoints:**
  1. `SatQuery-RS-VLM-v1.0` (BLIP ViT-B/16 + 12-layer text decoder, 361M parameters, checkpoint: `ml_pipeline/checkpoints/satquery_rs_vlm`)
  2. `SatQueryChangeNet-v1.0` (Siamese U-Net with difference concatenation, 7.8M parameters, checkpoint: `ml_pipeline/checkpoints/satquery_change_net/change_net.pth`)
  3. `SatQueryOpticalSARNet-v1.0` (Dual-Encoder Fusion Network, 849,603 parameters, checkpoint: `ml_pipeline/checkpoints/satquery_optical_sar_net/optical_sar_net.pth`)
  4. `google/owlvit-base-patch32` (Zero-shot open-vocabulary object detector, 155M parameters, snapshot: `.huggingface_cache/hub/models--google--owlvit-base-patch32`)

---

### 2.2 Preprocessing, Thresholding & Scoring Protocol
1. **RSVQA:**
   - Image input: RGB [0, 255] normalized to PyTorch tensor via `BlipProcessor`.
   - Text generation: Greedy autoregressive decoding (`max_new_tokens=24`).
   - Normalization: Punctuation removal, lowercasing, whitespace trimming.
   - Match criteria: Normalized exact match or substring containment (e.g., "water body" matches "water").
2. **VRSBench:**
   - VQA: Decoded with `max_new_tokens=32`, evaluated with exact match, substring match, and token-level F1 against reference list.
   - Captioning: Prompt `"A high-resolution satellite observation of"` with `max_new_tokens=48`. Token unigram precision, recall, and F1 computed against reference captions.
   - Grounding: Heuristic contour delineation filters contours between 30 px and 80% canvas area. OWL-ViT detects targets at confidence threshold 0.08. IoU computed as $\frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$; missed detections strictly score $\text{IoU} = 0.0$.
3. **CDVQA:**
   - ChangeNet: Inputs normalized to `[0.0, 1.0]`, bitemporal pair $(T_1, T_2)$ fed to Siamese encoder. Sigmoid activation thresholded at 0.45.
   - Change VQA: Difference image $|T_2 - T_1|$ fed with query to `SatQuery-RS-VLM-v1.0`.
4. **Optical–SAR Joint Segmentation:**
   - Optical RGB normalized to `[0.0, 1.0]`.
   - SAR VV channel in dB scaled from `[-25.0, 0.0]` dB linearly to `[-1.0, 1.0]`.
   - Sigmoid activation independent thresholding at 0.50 per channel without cross-class vetoes.

---

## 3. Analysis of Failure Modes & Discrepancies

1. **Counting & Quantitative Questions (RSVQA: 11.11%, CDVQA: 0.00%):**  
   Both RSVQA and CDVQA reveal that the vision-language decoder struggles with exact numeric counting queries (e.g., predicting "1" when reference is "0" or vice versa). Counting requires fine-grained spatial feature pooling that global ViT-B patch embeddings do not preserve without specialized counting heads.
2. **Visual Grounding on Synthetic Noise (Acc@0.5: 0.00%):**  
   Both heuristic OpenCV edge detection and neural OWL-ViT fail to achieve IoU $\ge 0.50$ against synthetic ground-truth bounding boxes. Because the test scenes consist of uniform Gaussian noise, no authentic object features (wings, fuselages, ship hulls) exist for OWL-ViT's text-image contrastive head to latch onto.
3. **Change Detection False Positives (ChangeNet Precision: 15.16%):**  
   While ChangeNet achieves 80.00% recall on synthetic rectangular change deltas, its low precision indicates false-positive activations across non-changed noisy regions, attributable to lightweight training (3 epochs) on synthetic data.
4. **Authentic ISRO Data Gap (Status: BLOCKED):**  
   No ground-truth reference masks exist for the demo imagery. Attempting to evaluate Cartosat-2S / RISAT-1 extraction accuracy without authoritative annotations would violate scientific standards.

---

## 4. Evaluator Verification Commands

To reproduce the benchmark evaluation runs exactly:

```powershell
# Set offline environment variables
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_DATASETS_OFFLINE="1"
$env:HF_HOME="d:\sih26167\.huggingface_cache"
$env:PYTHONIOENCODING="utf-8"

# Execute complete evaluation suite
backend\venv\Scripts\python.exe scratch\run_benchmark_eval_suite.py
```

All predictions will be written to `reports/benchmark_evaluation/predictions/` and execution traces to `reports/benchmark_evaluation/logs/eval_execution.log`.
