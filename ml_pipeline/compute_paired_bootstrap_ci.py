#!/usr/bin/env python3
"""
ml_pipeline/compute_paired_bootstrap_ci.py

Computes Paired Cluster-Bootstrap 95% Confidence Intervals for:
Delta = Metric(ChangeNet) - Metric(Blind)
and for all other vision candidates vs Blind baseline:
- Delta OA (Overall Accuracy)
- Delta AA (Macro Average Accuracy across 8 categories)
- Delta Accuracy per category (for each of the 8 categories)

Uses 1,000 cluster-bootstrap iterations over physical scene pairs to account
for intra-scene question correlation.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
CACHE_DIR = ML_DIR / "dataset_cache" / "phase1_full_cdvqa"
OUT_REPORT = ROOT_DIR / "reports" / "benchmark_evaluation" / "cdvqa_paired_bootstrap_ci.json"
FULL_REPORT = ROOT_DIR / "reports" / "benchmark_evaluation" / "phase1_encoder_full_comparison.json"

CATEGORIES = [
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "change_to_what",
    "smallest_change",
    "largest_change",
    "change_ratio",
    "change_ratio_types"
]

VOCAB = [
    "0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
    "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100",
    "buildings", "low_vegetation", "no", "nvg_surface", "playgrounds",
    "trees", "water", "yes"
]
ANS2IDX = {a: i for i, a in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)
SEED = 42

class VQAProbeHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_annotations(split: str, max_pairs: int = None):
    ann_path = ML_DIR / "authentic_data" / "cdvqa_prepared" / split / "cdvqa_annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        raw_annots = json.load(f)

    seen_pairs = []
    pair_set = set()
    for a in raw_annots:
        pid = a["pair_id"]
        if pid not in pair_set:
            pair_set.add(pid)
            seen_pairs.append(pid)
            if max_pairs and len(seen_pairs) >= max_pairs:
                break

    allowed_pairs = set(seen_pairs)
    filtered = [
        a for a in raw_annots
        if a["pair_id"] in allowed_pairs and a.get("answer", "").lower().strip() in ANS2IDX
    ]
    return filtered, seen_pairs

def train_and_predict(x_tr: torch.Tensor, y_tr: torch.Tensor, x_v: torch.Tensor, seed: int = SEED, epochs: int = 15) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = x_tr.shape[1]
    head = VQAProbeHead(in_features=in_dim, hidden_dim=256, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(x_tr, y_tr)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    for ep in range(epochs):
        head.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(x_v)
        val_preds = torch.argmax(val_logits, dim=-1).cpu().numpy()
    return val_preds

def compute_paired_cluster_bootstrap(
    val_pairs_arr: np.ndarray,
    preds_model: np.ndarray,
    preds_blind: np.ndarray,
    targets: np.ndarray,
    categories: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    unique_pairs = np.unique(val_pairs_arr)
    n_pairs = len(unique_pairs)
    pair_to_indices = {p: np.where(val_pairs_arr == p)[0] for p in unique_pairs}

    # Point estimates on full validation set
    point_oa_model = float(np.mean(preds_model == targets) * 100.0)
    point_oa_blind = float(np.mean(preds_blind == targets) * 100.0)
    point_delta_oa = point_oa_model - point_oa_blind

    point_cat_accs_model = {}
    point_cat_accs_blind = {}
    point_cat_delta = {}
    for cat in CATEGORIES:
        mask = (categories == cat)
        acc_m = float(np.mean(preds_model[mask] == targets[mask]) * 100.0) if np.sum(mask) > 0 else 0.0
        acc_b = float(np.mean(preds_blind[mask] == targets[mask]) * 100.0) if np.sum(mask) > 0 else 0.0
        point_cat_accs_model[cat] = acc_m
        point_cat_accs_blind[cat] = acc_b
        point_cat_delta[cat] = acc_m - acc_b

    point_aa_model = float(np.mean(list(point_cat_accs_model.values())))
    point_aa_blind = float(np.mean(list(point_cat_accs_blind.values())))
    point_delta_aa = point_aa_model - point_aa_blind

    # Bootstrap distributions
    delta_oa_boots = np.zeros(n_bootstrap, dtype=np.float32)
    delta_aa_boots = np.zeros(n_bootstrap, dtype=np.float32)
    cat_delta_boots = {cat: np.zeros(n_bootstrap, dtype=np.float32) for cat in CATEGORIES}

    rng = np.random.RandomState(seed)
    for b in range(n_bootstrap):
        sampled_pairs = rng.choice(unique_pairs, size=n_pairs, replace=True)
        sampled_indices = np.concatenate([pair_to_indices[p] for p in sampled_pairs])

        s_m = preds_model[sampled_indices]
        s_b = preds_blind[sampled_indices]
        s_t = targets[sampled_indices]
        s_c = categories[sampled_indices]

        # Overall Accuracy difference
        oa_m = np.mean(s_m == s_t) * 100.0
        oa_b = np.mean(s_b == s_t) * 100.0
        delta_oa_boots[b] = oa_m - oa_b

        # Per-category and Average Accuracy difference
        c_accs_m = []
        c_accs_b = []
        for cat in CATEGORIES:
            c_mask = (s_c == cat)
            if np.sum(c_mask) > 0:
                acc_c_m = np.mean(s_m[c_mask] == s_t[c_mask]) * 100.0
                acc_c_b = np.mean(s_b[c_mask] == s_t[c_mask]) * 100.0
            else:
                acc_c_m = 0.0
                acc_c_b = 0.0
            cat_delta_boots[cat][b] = acc_c_m - acc_c_b
            c_accs_m.append(acc_c_m)
            c_accs_b.append(acc_c_b)

        delta_aa_boots[b] = np.mean(c_accs_m) - np.mean(c_accs_b)

    ci_delta_oa = (float(np.percentile(delta_oa_boots, 2.5)), float(np.percentile(delta_oa_boots, 97.5)))
    ci_delta_aa = (float(np.percentile(delta_aa_boots, 2.5)), float(np.percentile(delta_aa_boots, 97.5)))

    per_category_cis = {}
    for cat in CATEGORIES:
        boots = cat_delta_boots[cat]
        c_low = float(np.percentile(boots, 2.5))
        c_high = float(np.percentile(boots, 97.5))
        sig = (c_low > 0) or (c_high < 0)
        per_category_cis[cat] = {
            "model_acc": round(point_cat_accs_model[cat], 2),
            "blind_acc": round(point_cat_accs_blind[cat], 2),
            "delta": round(point_cat_delta[cat], 2),
            "ci_95": [round(c_low, 2), round(c_high, 2)],
            "statistically_significant": sig
        }

    return {
        "point_estimates": {
            "model_oa": round(point_oa_model, 2),
            "blind_oa": round(point_oa_blind, 2),
            "delta_oa": round(point_delta_oa, 2),
            "model_aa": round(point_aa_model, 2),
            "blind_aa": round(point_aa_blind, 2),
            "delta_aa": round(point_delta_aa, 2)
        },
        "paired_delta_oa_ci_95": [round(ci_delta_oa[0], 2), round(ci_delta_oa[1], 2)],
        "delta_oa_significant": (ci_delta_oa[0] > 0) or (ci_delta_oa[1] < 0),
        "paired_delta_aa_ci_95": [round(ci_delta_aa[0], 2), round(ci_delta_aa[1], 2)],
        "delta_aa_significant": (ci_delta_aa[0] > 0) or (ci_delta_aa[1] < 0),
        "per_category_delta": per_category_cis
    }

def main():
    print("=" * 70)
    print("Computing Paired Cluster-Bootstrap CIs for (ChangeNet - Blind)")
    print("=" * 70)

    train_annots, train_pairs = load_annotations("train", max_pairs=800)
    val_annots, val_pairs = load_annotations("val", max_pairs=None)

    print(f"Train annotations: {len(train_annots)} | Val annotations: {len(val_annots)}")

    q_cache = CACHE_DIR / "q_feat_dict.pt"
    cnet_cache = CACHE_DIR / "cnet_pair_feats.pt"

    if not q_cache.exists() or not cnet_cache.exists():
        print(f"Error: Missing cache files {q_cache} or {cnet_cache}")
        sys.exit(1)

    q_feat_dict = torch.load(q_cache, map_location="cpu")
    cnet_pair_feats = torch.load(cnet_cache, map_location="cpu")

    y_train = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in train_annots], dtype=torch.long)
    y_val = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in val_annots], dtype=torch.long)

    q_train_tensor = torch.stack([q_feat_dict[a["question"]] for a in train_annots])
    q_val_tensor = torch.stack([q_feat_dict[a["question"]] for a in val_annots])

    val_pairs_arr = np.array([a["pair_id"] for a in val_annots])
    val_targets_arr = y_val.numpy()
    val_cats_arr = np.array([a["question_type"] for a in val_annots])

    # 1. Blind Model Predictions
    print("\nTraining Blind Model (Q-only)...")
    blind_preds = train_and_predict(q_train_tensor, y_train, q_val_tensor, seed=SEED, epochs=15)

    # 2. ChangeNet Predictions
    print("Training ChangeNet Model (Diff + Q)...")
    cnet_train_x = torch.stack([
        torch.cat([cnet_pair_feats[a["pair_id"]], q_train_tensor[i]], dim=-1)
        for i, a in enumerate(train_annots)
    ])
    cnet_val_x = torch.stack([
        torch.cat([cnet_pair_feats[a["pair_id"]], q_val_tensor[i]], dim=-1)
        for i, a in enumerate(val_annots)
    ])
    cnet_preds = train_and_predict(cnet_train_x, y_train, cnet_val_x, seed=SEED, epochs=15)

    # 3. Compute Paired Cluster Bootstrap
    print("\nComputing 1,000-resample paired cluster bootstrap over 400 scene pairs...")
    paired_result = compute_paired_cluster_bootstrap(
        val_pairs_arr, cnet_preds, blind_preds, val_targets_arr, val_cats_arr, n_bootstrap=1000, seed=SEED
    )

    print("\n--- RESULTS: ChangeNet vs Blind Baseline ---")
    p = paired_result["point_estimates"]
    print(f"Overall Accuracy (OA): ChangeNet={p['model_oa']}% | Blind={p['blind_oa']}% | Delta={p['delta_oa']:+.2f}%")
    print(f"  Paired 95% CI: [{paired_result['paired_delta_oa_ci_95'][0]:+.2f}%, {paired_result['paired_delta_oa_ci_95'][1]:+.2f}%] (Significant: {paired_result['delta_oa_significant']})")
    print(f"Average Accuracy (AA): ChangeNet={p['model_aa']}% | Blind={p['blind_aa']}% | Delta={p['delta_aa']:+.2f}%")
    print(f"  Paired 95% CI: [{paired_result['paired_delta_aa_ci_95'][0]:+.2f}%, {paired_result['paired_delta_aa_ci_95'][1]:+.2f}%] (Significant: {paired_result['delta_aa_significant']})")

    print("\n--- Per-Category Breakdown ---")
    print(f"{'Category':<20} | {'ChangeNet':<10} | {'Blind':<10} | {'Delta':<8} | {'Paired 95% CI':<18} | {'Significant?'}")
    print("-" * 80)
    for cat, info in paired_result["per_category_delta"].items():
        ci_str = f"[{info['ci_95'][0]:+.2f}, {info['ci_95'][1]:+.2f}]"
        sig_str = "YES" if info["statistically_significant"] else "No (spans 0)"
        print(f"{cat:<20} | {info['model_acc']:>8.2f}% | {info['blind_acc']:>8.2f}% | {info['delta']:>+6.2f}% | {ci_str:<18} | {sig_str}")

    # Save output report
    report_data = {
        "benchmark": "Phase 1 CDVQA Paired Cluster-Bootstrap Analysis (ChangeNet vs Blind)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Paired cluster-bootstrap 95% confidence intervals evaluating whether ChangeNet visual features provide statistically significant improvement over language prior alone.",
        "n_bootstrap": 1000,
        "cluster_level": "scene_pair (400 validation scene pairs)",
        "n_validation_questions": len(val_annots),
        "results": paired_result
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved paired bootstrap report to: {OUT_REPORT}")

    # Update phase1_encoder_full_comparison.json with the paired CI
    if FULL_REPORT.exists():
        with open(FULL_REPORT, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        if "satquery_changenet" in full_data.get("candidates", {}):
            full_data["candidates"]["satquery_changenet"]["paired_vs_blind_ci_95_oa"] = paired_result["paired_delta_oa_ci_95"]
            full_data["candidates"]["satquery_changenet"]["paired_vs_blind_ci_95_aa"] = paired_result["paired_delta_aa_ci_95"]
            full_data["candidates"]["satquery_changenet"]["paired_vs_blind_per_category"] = paired_result["per_category_delta"]
        # Also clean up published_test1_oa in per_category_majority
        if "per_category_majority" in full_data.get("candidates", {}):
            pcm = full_data["candidates"]["per_category_majority"]
            if "published_test1_oa" in pcm:
                pcm["computed_test1_oa"] = pcm.pop("published_test1_oa")
            if "published_test1_aa" in pcm:
                pcm["computed_test1_aa"] = pcm.pop("published_test1_aa")
            pcm["provenance"] = "reports/computed_baselines.json#cdvqa_test1_per_category_majority_oa"
        with open(FULL_REPORT, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2)
        print(f"Updated full comparison report with paired CIs: {FULL_REPORT}")

if __name__ == "__main__":
    main()
