# SatQuery AI — Parent-Granule-Disjoint Evaluation Plan

**Document Version:** 1.0  
**Date:** 2026-10-15  
**Evaluation Scope:** OpticalSARNet Spatial Generalization & Leakage Audit  
**Target Architecture:** `SatQueryOpticalSARNet`  
**Related Scripts & Data:**
- [`ml_pipeline/scripts/prepare_ben_ge_800.py`](file:///d:/sih26167/ml_pipeline/scripts/prepare_ben_ge_800.py#L83-L86)
- [`ml_pipeline/authentic_data/ben_ge_800_prepared/`](file:///d:/sih26167/ml_pipeline/authentic_data/ben_ge_800_prepared/)
- [`ml_pipeline/train_optical_sar_pilot.py`](file:///d:/sih26167/ml_pipeline/train_optical_sar_pilot.py)
- [`ml_pipeline/evaluate_pilot_test.py`](file:///d:/sih26167/ml_pipeline/evaluate_pilot_test.py)

---

## 1. Executive Summary & Problem Formulation

In the initial authentic-data pilot for `SatQueryOpticalSARNet`, the dataset of 800 Sentinel-1/Sentinel-2/WorldCover patches was split using MD5 hashing on individual patch identifiers:
```python
def assign_split(patch_id):
    h = int(hashlib.md5(patch_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")
```
This produced 550 train, 121 validation, and 129 test patches with zero duplicate patch IDs.

However, an audit of the satellite granule hierarchy reveals that **individual patches are 1.2 km × 1.2 km sub-windows cropped from much larger Sentinel-2 and Sentinel-1 parent granules (100 km × 100 km swaths)**. 

Because the hash function acted on the sub-patch coordinate name (e.g., `S2A_MSIL2A_20170613T101031_46_82`), adjacent patches from the **exact same satellite overpass** were randomly scattered across train, validation, and test sets.

This causes **spatial autocorrelation and scene leakage**:
- Patches from the same parent granule share identical atmospheric conditions, identical acquisition timestamps, identical sun elevation, identical soil moisture, and identical sensor look angles.
- While the model did not see the exact pixel patch coordinates in training, it saw neighboring patches from the same satellite overpass.
- Consequently, the reported **63.26% test mIoU** reflects within-scene interpolation rather than true out-of-scene geographic and temporal generalization.

This plan details the empirical quantification of this overlap, proposes a **Parent-Granule-Disjoint** partitioning protocol, provides comprehensive resource and compute cost estimates, and establishes the criteria for a definitive out-of-distribution evaluation.

---

## 2. Empirical Quantification of Current Scene Overlap

An audit of the 800 patches prepared in `ml_pipeline/authentic_data/ben_ge_800_prepared/` was executed by extracting parent acquisition granules:
- **Sentinel-2 Parent Granule:** Acquisition granule ID extracted by stripping sub-patch row/col offsets (e.g., `S2A_MSIL2A_20170613T101031`).
- **Sentinel-1 Parent Acquisition:** S1 overpass ID extracted from the SAR GeoTIFF name (e.g., `S1B_IW_GRDH_1SDV_20170612T165809_33UUP`).

### 2.1 Granule Distribution Across Dataset

```
Total Patches in BEN-GE-800:        800 patches
Unique Sentinel-2 Parent Granules:  109 granules (1 to 32 patches/granule, median = 6)
Unique Sentinel-1 Parent Granules:  192 acquisitions (1 to 31 patches/granule, median = 3)
```

### 2.2 Split Overlap Breakdown

| Split | Patch Count | Unique S2 Parent Granules | S2 Granules Shared with Train | S2 Scene Overlap % | Unique S1 Parent Granules | S1 Granules Shared with Train | S1 Scene Overlap % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 550 | 102 | Baseline (102) | 100.0% | 174 | Baseline (174) | 100.0% |
| **Validation** | 121 | 70 | **66** | **94.3%** | 78 | **68** | **87.2%** |
| **Test** | 129 | 63 | **59** | **93.7%** | 84 | **74** | **88.1%** |

```
Train vs. Val S2 Shared Granules:   66 / 70 (94.3%)
Train vs. Test S2 Shared Granules:  59 / 63 (93.7%)
Val vs. Test S2 Shared Granules:    44 / 63 (69.8%)
```

### 2.3 Scientific Implication of Scene Overlap
1. **94.3% of validation granules** and **93.7% of test granules** had sibling patches in the training set.
2. In remote sensing literature (e.g., Rußwurm et al., *IEEE TGRS* 2020; Sumbul et al., *BigEarthNet Benchmark* 2019), random patch splitting without tile buffering overestimates true land-cover segmentation generalization by **3% to 8% mIoU**.
3. Therefore, before production deployment, `SatQueryOpticalSARNet` must undergo a **Parent-Granule-Disjoint** verification where no scene or satellite overpass is shared between splits.

---

## 3. Parent-Granule-Disjoint Split Architecture

### 3.1 Partitioning Principle
Let $\mathcal{G} = \{G_1, G_2, \dots, G_{109}\}$ be the set of all unique parent granules in the dataset, where each granule $G_i$ contains a set of patches $P(G_i)$ such that $\bigcup_{i} P(G_i) = \mathcal{D}$ and $P(G_i) \cap P(G_j) = \emptyset$ for $i \neq j$.

We partition $\mathcal{G}$ into three mutually disjoint subsets:
$$\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{val}} = \emptyset, \quad \mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{test}} = \emptyset, \quad \mathcal{G}_{\text{val}} \cap \mathcal{G}_{\text{test}} = \emptyset$$

All child patches belonging to any granule $G_k \in \mathcal{G}_{\text{split}}$ are assigned exclusively to that split:
$$\mathcal{D}_{\text{split}} = \bigcup_{G_k \in \mathcal{G}_{\text{split}}} P(G_k)$$

### 3.2 Stratified Partitioning Constraints
Because different satellite granules cover different geographic biomes, unconstrained random partitioning of granules could result in severe class imbalance (e.g., all water patches falling into training). The partition algorithm must satisfy:

1. **Split Ratio Target:**
   - Train: $\sim 70\%$ ($\approx 560 \pm 20$ patches; $\approx 76$ parent granules)
   - Val: $\sim 15\%$ ($\approx 120 \pm 15$ patches; $\approx 16$ parent granules)
   - Test: $\sim 15\%$ ($\approx 120 \pm 15$ patches; $\approx 17$ parent granules)
2. **Climate Zone Balance:**
   - Stratify across Köppen-Geiger climate zones (Column `climatezone` in `ben-ge-800_meta.csv`). Each major climate zone (Zone 8 Mediterranean, Zone 26 Marine West Coast, Zone 27 Humid Continental, Zone 15 Subtropical) must have representation in all three splits.
3. **Class Prevalence Balance:**
   - Each split must contain minimum pixel quotas:
     - Built-up (ESA WorldCover 50): $\ge 3.0\%$ of split pixels.
     - Water Bodies (ESA WorldCover 80/90): $\ge 8.0\%$ of split pixels.
     - Vegetation (ESA WorldCover 10/20/30/40): $\ge 60.0\%$ of split pixels.

### 3.3 Proposed Partitioning Algorithm (Pseudocode)
```python
def generate_granule_disjoint_splits(meta_df, patches_list):
    # 1. Group patches by parent granule
    granule_map = defaultdict(list)
    for p in patches_list:
        parent_id = extract_parent_granule_id(p["patch_id"])
        granule_map[parent_id].append(p)
    
    # 2. Extract granule-level statistics (climate zone, patch count, class distribution)
    granule_stats = []
    for gid, plist in granule_map.items():
        cz = meta_df.loc[plist[0]["patch_id"], "climatezone"]
        built_px, water_px, veg_px = aggregate_class_pixels(plist)
        granule_stats.append({
            "granule_id": gid,
            "patch_count": len(plist),
            "climate_zone": cz,
            "built_ratio": built_px / total_px,
            "water_ratio": water_px / total_px
        })
    
    # 3. Stratified Multi-Objective Greedy Knapsack Partition
    # Assign entire granules to train (70%), val (15%), test (15%)
    train_granules, val_granules, test_granules = stratified_granule_partition(
        granule_stats, target_ratios=(0.70, 0.15, 0.15)
    )
    
    # 4. Verify ZERO intersection
    assert set(train_granules).isdisjoint(set(val_granules))
    assert set(train_granules).isdisjoint(set(test_granules))
    assert set(val_granules).isdisjoint(set(test_granules))
    
    return train_granules, val_granules, test_granules
```

---

## 4. Resource & Cost Estimation for Additional Experiment

The resource, compute, and financial budget for executing this parent-granule-disjoint experiment is exceptionally lean because the 800 authentic patches are already downloaded, extracted, and verified on local disk.

### 4.1 Resource Breakdown Table

| Activity | Component / Tool | Data Volume | Compute Type | Memory (RAM) | Wall-Clock Time | Financial Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Split Generation** | `generate_disjoint_splits.py` | 800 patches metadata | CPU (1 thread) | $< 100\text{ MB}$ | $\approx 15\text{ seconds}$ | **$0.00** |
| **2. Annotation Indexing** | Create `train_disjoint.json`, etc. | 3 JSON files ($\sim 500\text{ KB}$) | Local disk write | $< 50\text{ MB}$ | $\approx 5\text{ seconds}$ | **$0.00** |
| **3. Training Run** | `SatQueryOpticalSARNet` (3 epochs) | 560 patches, batch size 4 | Local CPU (`torch cpu`) | $\approx 1.2\text{ GB}$ | $\approx 18 - 25\text{ minutes}$ | **$0.00** |
| **4. Val Evaluation** | Metric computation (epoch ends) | 120 patches | Local CPU | $\approx 800\text{ MB}$ | $\approx 2\text{ minutes}$ | **$0.00** |
| **5. Held-Out Test Eval** | `evaluate_disjoint_test.py` | 120 untouched patches | Local CPU | $\approx 800\text{ MB}$ | $\approx 45 - 60\text{ seconds}$ | **$0.00** |
| **6. Total Overhead** | **Full Disjoint Verification** | **800 authentic patches** | **Local CPU (Single Node)** | **$\le 1.5\text{ GB}$** | **$< 30\text{ minutes}$** | **$0.00** |

### 4.2 Resource Commentary:
- **Zero API / Cloud Spend:** The entire pipeline executes locally on the workstation (`d:\sih26167`) using Python 3.10 and PyTorch CPU. No external LLM, cloud VM, or API credits are consumed.
- **Zero Additional Disk Footprint:** The script generates lightweight annotation indices (`.json`) pointing to the existing `.npy` and `.png` files in `ml_pipeline/authentic_data/ben_ge_800_prepared/`. Zero raster duplication is required.
- **Minimal RAM Footprint:** Peak RAM of 1.2 GB is easily accommodated by the available 15.7 GB system RAM.

---

## 5. Experimental Hypotheses & Success Criteria

### 5.1 Hypotheses
1. **Hypothesis 1 (Generalization Penalty):** Due to the elimination of spatial autocorrelation and scene-specific atmospheric/sensor leakage, the parent-granule-disjoint test mIoU is expected to drop by **3.0% to 7.0%** compared to the patch-hashed test mIoU (63.26%).
2. **Hypothesis 2 (Prototype Outperformance):** Even with the expected generalization penalty, the authentic-data model will substantially outperform the legacy prototype baseline (34.42% mIoU) by $+20\%$ or more, proving genuine semantic feature learning.
3. **Hypothesis 3 (Class-Specific Sensitivity):** Vegetation and Water will remain robust ($>70\%$ and $>50\%$ IoU), while Built-up will be the most sensitive class due to regional differences in European urban morphology.

### 5.2 Quantitative Promotion Thresholds for Disjoint Evaluation

| Metric | Non-Disjoint Pilot Result | Disjoint Promotion Threshold | Rationale |
| :--- | :---: | :---: | :--- |
| **Global Micro mIoU** | 63.26% | **$\ge 55.0\%$** | Permits up to 8.26% generalization drop; still +20.58% above baseline (34.42%). |
| **Val-to-Test Gap ($|\Delta|$ )** | 1.00% | **$\le 5.0\%$** | Bound on overfitting to validation scenes. |
| **Sample-Level Macro mIoU** | 46.72% | **$\ge 40.0\%$** | Confirms per-image consistency across unseen geographic granules. |
| **Vegetation IoU** | 76.54% | **$\ge 70.0\%$** | Dominant land-cover class stability. |
| **Water Bodies IoU** | 58.74% | **$\ge 50.0\%$** | Critical for flood and water boundary delineation. |
| **Built-Up Fabric IoU** | 29.84% | **$\ge 22.0\%$** | Minority class robustness across distinct architectural styles. |

---

## 6. Execution Roadmap & Staging Integration

When approved by the technical lead, the experiment will follow this 4-step execution sequence:
1. **Script Implementation:** Author `ml_pipeline/scripts/partition_granule_disjoint.py` to create `train_disjoint.json`, `val_disjoint.json`, `test_disjoint.json`.
2. **Audit Verification:** Verify $0\%$ parent granule overlap and confirm class distributions.
3. **Model Training:** Run `python ml_pipeline/train_optical_sar_pilot.py --annotations train_disjoint.json --val_annotations val_disjoint.json` with checkpoint output directed to `ml_pipeline/checkpoints/satquery_optical_sar_net_disjoint/`.
4. **Held-Out Evaluation:** Run `python ml_pipeline/evaluate_pilot_test.py --test_annotations test_disjoint.json` and generate `reports/training_migration/granule_disjoint_test_results.md`.

**Current Status:** Ready for execution upon authorization. Zero code or routing changes made.
