# SatQuery AI — CDVQA Final Held-Out Test Evaluation Report

**Project:** Smart India Hackathon 2026 — Problem Statement PS26167  
**Benchmark:** Change Detection Visual Question Answering (CDVQA / SECOND Wuhan University Aerial Dataset)  
**Evaluator Reference:** Aminur Hossain (ISRO / Space Applications Centre (SAC), Ahmedabad)  
**Execution Phase:** Phase 2 Final Controlled Held-Out Test Evaluation  
**Timestamp:** October 16, 2026 (19:53:52 IST / 14:23:52 UTC)  
**Quarantine Protocol:** First and only evaluation on the official CDVQA held-out test partition. Zero prior test exposures.

---

## 1. Executive Summary & Audit Verdict

| Evaluation Aspect | Verdict | Key Finding |
| :--- | :---: | :--- |
| **Phase 1: Pre-Evaluation Integrity** | **PASS** | ChangeNet v2 (`95FB4C...`) and RemoteCLIP (`60014E...`) SHA-256 hashes verified. 100% exact validation metric reproduction (Seed 42: Blind 67.78%, Feat 68.75%). |
| **Phase 2: Held-Out Test Evaluation** | **PASS** | 39,686 questions across 968 scene pairs evaluated exactly once. Test OA: Blind **67.41%** vs Feature Head **69.04%** ($\Delta\text{OA} = \mathbf{+1.64\%}$). Test AA: Blind **57.51%** vs Feature Head **60.08%** ($\Delta\text{AA} = \mathbf{+2.57\%}$). |
| **Phase 3: Clustered Uncertainty Analysis** | **PASS** | 1,000-resample scene-pair clustered bootstrap: $\Delta\text{OA}$ 95% CI is $[+1.21\%, +2.08\%]$ ($p < 0.001$ at $B=1,000$ resolution). `change_ratio` ($+11.26\%$), `decrease_or_not` ($+5.15\%$), and `change_or_not` ($+1.09\%$) survive strict Bonferroni family-wise error control. |
| **Disambiguation & Reporting** | **PASS** | Test results clearly partitioned from validation results. Oracle upper bounds (99.94% GT, 52.99% predicted) clearly distinguished from learned model accuracy. |

---

## 2. Experimental Setup & Quarantined Test Protocol

### 2.1 Partition & Dataset Scope
- **Training Split:** 1,600 scene pairs (65,967 questions).
- **Validation Split:** 400 scene pairs (16,441 questions).
- **Held-Out Test Split:** **968 scene pairs (39,686 questions)** located in `ml_pipeline/authentic_data/cdvqa_prepared/test/cdvqa_annotations.json`.
- **Geographic Coverage:** Wuhan University SECOND benchmark aerial scenes ($512 \times 512$ rasters at 0.5–3m GSD, Hangzhou, Chengdu, Shanghai). Disjoint spatial tiles.
- **Vocabulary:** 19 classes (identical to train/val: `0`, `0_to_10`, ..., `90_to_100`, `buildings`, `low_vegetation`, `no`, `nvg_surface`, `playgrounds`, `trees`, `water`, `yes`). Zero out-of-vocabulary answers (39,686 / 39,686 valid).

### 2.2 Model Architecture & Evaluated Checkpoints
1. **Visual Backbone (`SatQueryChangeNet v2`):**
   - Siamese encoder with multi-scale feature difference convolutions ($d_3, d_2$) and pooling, projecting to 512-d normalized difference vector.
   - Checkpoint: `ml_pipeline/checkpoints/satquery_change_net_finetune/v2/best_change_net_finetuned.pth`
   - **SHA-256:** `95FB4CEB93F9E40B3C778B2FAF384E12D3278C8961892FD8B4A287836207F6B9`
2. **Text Backbone (`RemoteCLIP ViT-B/32`):**
   - Domain-specific remote sensing text encoder projecting questions to 512-d normalized embedding.
   - Checkpoint: `ml_pipeline/authentic_data/_downloads/RemoteCLIP-ViT-B-32.pt`
   - **SHA-256:** `60014E395D930A3F2963D1D89C8522BF4AD56775571E4356E866864789AF85C4`
3. **Probe Heads (Trained on 1,600 train pairs for 15 epochs):**
   - `BlindVQAHead`: `Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256, 19)`
     - Checkpoint Seed 42: `ml_pipeline/checkpoints/cdvqa_heads/blind_head_seed42.pth` (SHA-256: `C0C35A03...`)
   - `FeatureVQAHead`: `Linear(1024, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256, 19)`
     - Checkpoint Seed 42: `ml_pipeline/checkpoints/cdvqa_heads/feat_head_seed42.pth` (SHA-256: `C2E55D40...`)

---

## 3. Primary Held-Out Test Results (Seed 42)

Evaluated on all **39,686 held-out test questions**:

| Metric | Blind Baseline | Feature Head | Paired Delta ($\Delta$) | 95% Clustered CI | Statistical Significance ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | **67.41%** | **69.04%** | **+1.64%** | $[+1.21\%, +2.08\%]$ | **$p < 0.001$** (Statistically Significant) |
| **Average Accuracy (AA)** | **57.51%** | **60.08%** | **+2.57%** | $[+1.98\%, +3.19\%]$ | **$p < 0.001$** (Statistically Significant) |

> [!NOTE]
> **Empirical Resolution Limit:** With $B = 1,000$ bootstrap resamples, the minimum discrete probability resolution is $1/B = 0.001$. When zero resamples fall non-positive ($k = 0$), the empirical upper bound is strictly $p < 0.001$. Claims of $p < 0.0001$ are unsupportable without at least $B = 10,000$ resamples.

### 3.1 Per-Category Breakdown (Seed 42)
Uncertainty estimated via **1,000-resample scene-pair clustered bootstrap** across the 968 test scene pairs:

| Question Category | Test Questions | Blind Acc | Feature Head | Delta ($\Delta$) | Uncorrected 95% CI | Bonferroni 99.375% CI | Multiple-Testing Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `change_ratio` | 1,936 | 29.13% | 40.39% | **+11.26%** | $[+8.21\%, +14.10\%]$ | $[+7.54\%, +14.86\%]$ | **Survives Bonferroni** ($p < 0.001$) |
| `decrease_or_not` | 4,658 | 69.21% | 74.37% | **+5.15%** | $[+3.99\%, +6.53\%]$ | $[+3.21\%, +6.87\%]$ | **Survives Bonferroni** ($p < 0.001$) |
| `largest_change` | 2,904 | 48.62% | 51.48% | **+2.86%** | $[+0.76\%, +4.82\%]$ | $[-0.03\%, +5.60\%]$ | Significant at nominal $\alpha=0.05$ ($p=0.014$); marginal under Bonferroni |
| `change_to_what` | 2,991 | 55.43% | 57.10% | **+1.67%** | $[+0.20\%, +3.21\%]$ | $[-0.43\%, +3.76\%]$ | Significant at nominal $\alpha=0.05$ ($p=0.024$); spans zero under Bonferroni |
| `smallest_change` | 2,904 | 29.79% | 31.16% | **+1.38%** | $[-0.14\%, +2.86\%]$ | $[-0.59\%, +3.44\%]$ | Not statistically significant ($p=0.072$, spans zero) |
| `change_or_not` | 13,882 | 82.77% | 83.86% | **+1.09%** | $[+0.65\%, +1.58\%]$ | $[+0.42\%, +1.83\%]$ | **Survives Bonferroni** ($p < 0.001$) |
| `change_ratio_types` | 5,811 | 71.38% | 71.19% | **-0.19%** | $[-1.04\%, +0.62\%]$ | $[-1.38\%, +0.88\%]$ | Not statistically significant ($p=0.626$, language prior dominates) |
| `increase_or_not` | 4,600 | 73.72% | 71.04% | **-2.67%** | $[-3.75\%, -1.61\%]$ | $[-4.15\%, -1.15\%]$ | Significant negative shift ($p < 0.001$, visual feature interference) |

---

## 4. Multi-Seed Stability on Test Partition ($N=3$)

To evaluate training and optimization variability across random weight initializations and data batching, models were evaluated across Seeds 42, 43, and 44:

| Seed | Blind Test OA | Feat Test OA | $\Delta\text{OA}$ | Blind Test AA | Feat Test AA | $\Delta\text{AA}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 67.41% | 69.04% | **+1.64%** | 57.51% | 60.08% | **+2.57%** |
| **43** | 67.82% | 68.69% | **+0.87%** | 58.14% | 59.58% | **+1.44%** |
| **44** | 67.68% | 68.76% | **+1.08%** | 57.85% | 59.73% | **+1.89%** |
| **Mean $\pm$ Std** | **67.64% $\pm$ 0.17%** | **68.83% $\pm$ 0.15%** | **+1.20% $\pm$ 0.32%** | **57.83% $\pm$ 0.26%** | **59.80% $\pm$ 0.21%** | **+1.97% $\pm$ 0.46%** |

### Two-Tailed Paired Student's t-Test Across Seeds ($N=3, df=2$):
- **For $\Delta\text{OA}$ on Test Set:**
  - $\bar{d} = +1.1967\%$, $s = 0.3995\%$, $SE = 0.2307\%$
  - $t = 5.188$, **$p = 0.0352 < 0.05$** (95% CI: $[+0.21\%, +2.18\%]$) $\implies$ **Statistically significant across training runs on the held-out test partition.**
- **For $\Delta\text{AA}$ on Test Set:**
  - $\bar{d} = +1.9667\%$, $s = 0.5693\%$, $SE = 0.3287\%$
  - $t = 5.983$, **$p = 0.0268 < 0.05$** (95% CI: $[+0.55\%, +3.38\%]$) $\implies$ **Statistically significant across training runs on the held-out test partition.**

### Methodological Distinction Between the Two Statistical Tests:
1. **Three-Seed Paired Student's t-Test ($N=3, df=2$):** Evaluates **training and optimization variability** (stochasticity in weight initialization and batch shuffling). With $N=3$, power is modest but reaches significance ($p=0.0352$ for OA, $p=0.0268$ for AA).
2. **Scene-Pair Clustered Bootstrap ($B=1,000$ clusters):** Evaluates **finite-sample geographic evaluation uncertainty** across the 968 test scene pairs for a fixed model checkpoint (Seed 42), accounting for the clustering of ~41 questions per scene pair ($p < 0.001$).
3. **Category Multiple-Testing Control:** Evaluated by comparing uncorrected nominal 95% percentile intervals ($[2.5\%, 97.5\%]$) against Bonferroni-adjusted 99.375% percentile intervals ($[0.3125\%, 99.6875\%]$) for family-wise error rate control across the 8 categories.

---

## 5. Explicit Disambiguation of Evaluation Tiers

To maintain strict scientific integrity, metrics from different evaluation regimes must never be conflated:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. GROUND-TRUTH SEMANTIC ORACLE: 99.94% OA / 99.89% AA                                 │
│    - Deterministic algorithmic formulas queried directly from ground-truth SECOND      │
│      masks. Represents the mathematical upper bound of the question generator.        │
│    - NOT a learned model. Forbidden from being cited as model accuracy.                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. PREDICTED-MAP ORACLE: 52.99% OA / 47.84% AA                                         │
│    - Deterministic rules applied to noisy predicted masks from a trained segmentation  │
│      network. Severely degraded by segmentation mask boundary errors.                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. VALIDATION SET AUDIT (400 Scene Pairs, 16,441 QAs):                                  │
│    - Blind Baseline: 67.97% OA / 58.39% AA                                             │
│    - Feature Head:   68.56% OA / 59.67% AA                                             │
│    - Delta:          +0.59% OA / +1.28% AA (Training variance across 3 seeds)          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. FINAL HELD-OUT TEST EVALUATION (968 Scene Pairs, 39,686 QAs):                       │
│    - Blind Baseline: 67.64% OA / 57.83% AA                                             │
│    - Feature Head:   68.83% OA / 59.80% AA                                             │
│    - Delta:          +1.20% OA / +1.97% AA (Statistically significant, p = 0.0352)     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Limitations & Generalization Scope

1. **Benchmark Specificity:** These results establish that the visual feature head provides a statistically validated $+1.20\%$ to $+1.64\%$ OA improvement over the language prior on unseen spatial tiles from the Wuhan University SECOND collection.
2. **Geographic Generalization:** Because test tiles were collected under the same sensor platforms and regional aerial campaigns as train/val, these results demonstrate **within-distribution unseen-scene generalization**, but do **NOT** claim out-of-distribution geographic generalization to unseen satellite sensors (e.g. Sentinel-2, Landsat-8) or distinct geographic continents.
3. **Category Asymmetry:** Visual features provide large, decisive gains on area quantization (`change_ratio`: $+11.26\%$) and directional reduction (`decrease_or_not`: $+5.15\%$), but show slight degradation on directional expansion (`increase_or_not`: $-2.67\%$).

---

## 7. Artifact & Reproducibility Inventory

- **Evaluation Script:** [`ml_pipeline/eval_final_held_out_test.py`](file:///d:/sih26167/ml_pipeline/eval_final_held_out_test.py)
- **Structured JSON Report:** [`reports/benchmark_evaluation/cdvqa_final_test_evaluation.json`](file:///d:/sih26167/reports/benchmark_evaluation/cdvqa_final_test_evaluation.json)
- **Raw Predictions Tensor:** [`reports/benchmark_evaluation/cdvqa_final_test_predictions.pt`](file:///d:/sih26167/reports/benchmark_evaluation/cdvqa_final_test_predictions.pt)
- **Saved Head Checkpoints:**
  - `ml_pipeline/checkpoints/cdvqa_heads/blind_head_seed42.pth` (SHA-256: `C0C35A03...`)
  - `ml_pipeline/checkpoints/cdvqa_heads/feat_head_seed42.pth` (SHA-256: `C2E55D40...`)
  - (Checkpoints for seeds 43 and 44 also permanently persisted).
