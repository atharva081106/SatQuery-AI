# SatQuery AI — Official Evaluation Harness Specification

**Document:** `docs/eval_harness_spec.md`  
**Standard:** Strict Rule 1 (Real Data Only), Rule 2 (Verified Metrics), Rule 3 (No Test Leakage)  
**Applies to:** CDVQA, RSVQA-LR/HR, VRSBench, BigEarthNet-MM  

---

## 1. Overview & Benchmark Principles

The SatQuery AI evaluation harness enforces standardized, scientifically defensible evaluation protocols across all remote sensing vision-language tasks. To prevent shortcut learning, spurious correlations, and data leakage, all model evaluations must adhere to the four pillars:

1. **Quarantined Test Sets:** Held-out test partitions (e.g. CDVQA 968 pairs, RSVQA-LR 154 tiles, VRSBench 9,350 test images) remain strictly untouched until final deployment gating.
2. **Cluster-Level Resampling:** All confidence intervals must use cluster-bootstrap over parent scene IDs to account for intra-scene question correlation.
3. **Mandatory Blind Language Baselines:** Every multimodal evaluation must report alongside a question-only blind baseline and state explicit paired gains ($\Delta\text{OA}$, $\Delta\text{AA}$).
4. **Image-Shuffle Ablation Gate:** Evaluates whether model accuracy collapses to the blind prior when visual-text correspondence is severed.

---

## 2. Statistical Testing & Confidence Intervals

### 2.1 Clustered Bootstrap Protocol
Questions evaluated on the same physical scene or tile share underlying visual land-cover distributions. Independent and identically distributed (i.i.d.) sample-level bootstrapping artificially shrinks confidence intervals.

- **Resampling Unit:** Physical scene pair ID (CDVQA) or Sentinel-2 tile ID (RSVQA).
- **Iterations:** Minimum $B = 1,000$ iterations with fixed random seed ($\text{seed}=42$).
- **Quantiles:** 95% Confidence Interval reported as $[q_{0.025}, q_{0.975}]$.

### 2.2 Paired Differences ($\Delta = \text{Model} - \text{Blind}$)
Separate confidence intervals do not test for statistical significance because errors are correlated across identical questions. The harness computes paired bootstrap differences per iteration:
$$\Delta^{(b)} = \text{Metric}_{\text{model}}^{(b)} - \text{Metric}_{\text{blind}}^{(b)}$$
A gain is declared statistically significant at $\alpha = 0.05$ if and only if:
$$0 \notin [q_{0.025}(\Delta), q_{0.975}(\Delta)]$$

---

## 3. Mandatory Blind Question-Only Baseline

Language priors in remote sensing VQA datasets allow question-only models to achieve high accuracy (e.g., 68.04% OA on CDVQA val, 50.73% per-category training prior).

- **Architecture:** Standard 2-layer MLP head directly over frozen question embeddings (no visual features).
- **Reporting Requirement:** All candidate models must report:
  - $\Delta\text{OA} = \text{OA}_{\text{candidate}} - \text{OA}_{\text{blind}}$
  - $\Delta\text{AA} = \text{AA}_{\text{candidate}} - \text{AA}_{\text{blind}}$
  - Paired 95% Confidence Interval for each category.

---

## 4. Image-Shuffle Ablation Protocol

### 4.1 Motivation & Provenance
As demonstrated by Lobry et al. (IEEE TGRS 2020, Section IV) and Goyal et al. (CVPR 2017), deep multimodal architectures can achieve high nominal scores while ignoring the visual modality. The **Image-Shuffle Ablation** tests whether the model's predictions are genuinely causally dependent on the input imagery.

### 4.2 Mathematical Definition
Given an evaluation dataset of $N$ question-image instances:
$$\mathcal{D} = \{(I_k, Q_k, A_k)\}_{k=1}^N$$
where $I_k$ denotes the image (or bi-temporal pair $(T_{1,k}, T_{2,k})$ for change detection).

1. Construct a random derangement permutation $\pi \in \mathcal{S}_N$ such that $\pi(k) \neq k$ and $\text{Scene}(I_{\pi(k)}) \neq \text{Scene}(I_k)$ for all $k$.
2. Form the shuffled dataset $\mathcal{D}_{\text{shuffled}} = \{(I_{\pi(k)}, Q_k, A_k)\}_{k=1}^N$.
3. Execute the frozen candidate model on $\mathcal{D}_{\text{shuffled}}$ without retraining.
4. Compute the Shuffled Accuracy $\text{OA}_{\text{shuffled}}$ and the **Visual Sensitivity Index (VSI)**:
   $$\text{VSI} = \text{OA}_{\text{standard}} - \text{OA}_{\text{shuffled}}$$

### 4.3 Evaluation Gate & Acceptance Criteria

| Condition | Diagnostic Finding | Verdict |
| :--- | :--- | :--- |
| $\text{OA}_{\text{shuffled}} \approx \text{OA}_{\text{blind}}$ and $\text{VSI} > 0$ | Model genuinely uses visual features; performance degrades to language prior under visual mismatch. | **PASS** |
| $\text{OA}_{\text{shuffled}} \approx \text{OA}_{\text{standard}}$ ($\text{VSI} \approx 0$) | **Modality Neglect Failure**: Model is completely blind to imagery and relying solely on language biases. | **FAIL (Gate Blocked)** |
| $\text{OA}_{\text{shuffled}} < \text{OA}_{\text{blind}}$ | Visual features from the wrong scene actively interfere with language prior, confirming strong visual conditioning. | **PASS (Strong Visual Conditioning)** |

---

## 5. Execution Interface & CLI Flag

All evaluation scripts in `ml_pipeline/` implement the `--shuffle-images` flag:

```bash
# Standard evaluation
python ml_pipeline/evaluate.py --model_path checkpoints/best_vlm --test_data val_split.json

# Image-shuffle ablation evaluation
python ml_pipeline/evaluate.py --model_path checkpoints/best_vlm --test_data val_split.json --shuffle-images --seed 42
```

Outputs are automatically logged to `reports/benchmark_evaluation/` with the `image_shuffle_ablation` key recording $\text{OA}_{\text{standard}}$, $\text{OA}_{\text{shuffled}}$, $\text{VSI}$, and comparison to the blind language prior.
