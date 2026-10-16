#!/usr/bin/env python3
"""
ml_pipeline/train_head_all_pairs_regularized.py

Phase 1 Item 3:
Reruns the CDVQA VQA Probe Head on ALL 1,600 authentic training pairs (65,967 questions)
with:
1. Visual-branch dropout (p = 0.4)
2. Random visual blanking (p = 0.2, zeroing visual feature vector during training to prevent shortcut learning)
3. Evaluates against Blind Question-Only baseline on all 400 validation pairs (16,441 questions)
4. Computes 1,000-resample Paired Cluster-Bootstrap 95% Confidence Intervals over physical scene pairs.

Outputs reports/benchmark_evaluation/cdvqa_head_1600pairs_regularized.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T
import open_clip
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
CACHE_DIR = ML_DIR / "dataset_cache" / "phase1_full_cdvqa"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = ROOT_DIR / "reports" / "benchmark_evaluation" / "cdvqa_head_1600pairs_regularized.json"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))
from change_model import SatQueryChangeNet

torch.set_num_threads(8)
SEED = 42

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

class RegularizedVQAHead(nn.Module):
    """
    VQA probe head with:
    - Visual-branch dropout (p_vis_drop = 0.4)
    - Random visual blanking (p_blank = 0.2)
    """
    def __init__(self, vis_dim: int, q_dim: int, hidden_dim: int = 256, num_classes: int = NUM_CLASSES,
                 p_vis_drop: float = 0.4, p_blank: float = 0.2):
        super().__init__()
        self.p_vis_drop = p_vis_drop
        self.p_blank = p_blank
        self.vis_dropout = nn.Dropout(p_vis_drop)
        self.net = nn.Sequential(
            nn.Linear(vis_dim + q_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, v_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Random visual blanking: zero out visual branch with probability p_blank
            blank_mask = (torch.rand(v_feat.size(0), 1, device=v_feat.device) > self.p_blank).float()
            v_reg = self.vis_dropout(v_feat) * blank_mask
        else:
            v_reg = v_feat
        fused = torch.cat([v_reg, q_feat], dim=-1)
        return self.net(fused)

class BlindVQAHead(nn.Module):
    def __init__(self, q_dim: int, hidden_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, q_feat: torch.Tensor) -> torch.Tensor:
        return self.net(q_feat)

def load_all_annotations(split: str = "train"):
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

    filtered = [
        a for a in raw_annots
        if a.get("answer", "").lower().strip() in ANS2IDX
    ]
    print(f"[{split.upper()}] Loaded {len(filtered):,} annotations across {len(seen_pairs)} scene pairs.")
    return filtered, seen_pairs

def ensure_changenet_features(all_annots: List[Dict], cnet_cache_path: Path) -> Dict[str, torch.Tensor]:
    if cnet_cache_path.exists():
        cnet_feats = torch.load(cnet_cache_path, map_location="cpu")
    else:
        cnet_feats = {}

    missing_pairs = {}
    for a in all_annots:
        pid = a["pair_id"]
        if pid not in cnet_feats and pid not in missing_pairs:
            missing_pairs[pid] = (ROOT_DIR / a["t1_path"], ROOT_DIR / a["t2_path"])

    if missing_pairs:
        print(f"Extracting ChangeNet features for {len(missing_pairs):,} newly required scene pairs...", flush=True)
        changenet = SatQueryChangeNet()
        ckpt_path = ML_DIR / "checkpoints" / "satquery_change_net_finetune" / "v2" / "best_change_net_finetuned.pth"
        raw_ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        changenet.load_state_dict(sd)
        changenet.eval()

        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

        for idx, (pid, (t1_p, t2_p)) in enumerate(missing_pairs.items(), 1):
            im1 = transform(Image.open(t1_p).convert("RGB")).unsqueeze(0)
            im2 = transform(Image.open(t2_p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                f0_1, f1_1, f2_1, f3_1 = changenet.extract_features(im1)
                f0_2, f1_2, f2_2, f3_2 = changenet.extract_features(im2)
                d3 = changenet.diff_conv3(torch.abs(f3_2 - f3_1))
                d2 = torch.abs(f2_2 - f2_1)

                p3_avg = F.adaptive_avg_pool2d(d3, 1).flatten(1)
                p3_max = F.adaptive_max_pool2d(d3, 1).flatten(1)
                p2_avg = F.adaptive_avg_pool2d(d2, 1).flatten(1)
                p2_max = F.adaptive_max_pool2d(d2, 1).flatten(1)
                feat = torch.cat([p3_avg, p3_max, p2_avg, p2_max], dim=-1).squeeze(0)
                cnet_feats[pid] = F.normalize(feat, dim=-1).cpu()

            if idx % 200 == 0 or idx == len(missing_pairs):
                print(f"  Extracted {idx}/{len(missing_pairs)} scene pairs", flush=True)

        torch.save(cnet_feats, cnet_cache_path)
        print(f"Updated cache with {len(cnet_feats):,} total scene pair features.")
    else:
        print(f"All {len(cnet_feats):,} required scene pair features present in cache.")

    return cnet_feats

def ensure_question_features(all_annots: List[Dict], q_cache_path: Path) -> Dict[str, torch.Tensor]:
    if q_cache_path.exists():
        q_feats = torch.load(q_cache_path, map_location="cpu")
    else:
        q_feats = {}

    unique_questions = list(set(a["question"] for a in all_annots if a["question"] not in q_feats))
    if unique_questions:
        print(f"Encoding {len(unique_questions):,} newly required questions with RemoteCLIP text encoder...", flush=True)
        rc_ckpt = ML_DIR / "authentic_data" / "_downloads" / "RemoteCLIP-ViT-B-32.pt"
        rc_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        rc_model.load_state_dict(torch.load(rc_ckpt, map_location="cpu"))
        rc_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')

        with torch.no_grad():
            for i in range(0, len(unique_questions), 64):
                batch_q = unique_questions[i:i + 64]
                tokens = tokenizer(batch_q)
                emb = rc_model.encode_text(tokens)
                emb = F.normalize(emb, dim=-1)
                for q_str, e in zip(batch_q, emb):
                    q_feats[q_str] = e.cpu()

        torch.save(q_feats, q_cache_path)
        print(f"Updated question cache with {len(q_feats):,} total unique questions.")
    else:
        print(f"All {len(q_feats):,} question embeddings present in cache.")

    return q_feats

def compute_paired_cluster_bootstrap(
    val_pairs_arr: np.ndarray,
    preds_model: np.ndarray,
    preds_blind: np.ndarray,
    targets: np.ndarray,
    categories: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED
) -> Dict[str, Any]:
    unique_pairs = np.unique(val_pairs_arr)
    n_pairs = len(unique_pairs)
    pair_to_indices = {p: np.where(val_pairs_arr == p)[0] for p in unique_pairs}

    point_oa_model = float(np.mean(preds_model == targets) * 100.0)
    point_oa_blind = float(np.mean(preds_blind == targets) * 100.0)
    point_delta_oa = point_oa_model - point_oa_blind

    point_cat_accs_m = {}
    point_cat_accs_b = {}
    for cat in CATEGORIES:
        mask = (categories == cat)
        point_cat_accs_m[cat] = float(np.mean(preds_model[mask] == targets[mask]) * 100.0) if np.sum(mask) > 0 else 0.0
        point_cat_accs_b[cat] = float(np.mean(preds_blind[mask] == targets[mask]) * 100.0) if np.sum(mask) > 0 else 0.0

    point_aa_model = float(np.mean(list(point_cat_accs_m.values())))
    point_aa_blind = float(np.mean(list(point_cat_accs_b.values())))
    point_delta_aa = point_aa_model - point_aa_blind

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

        delta_oa_boots[b] = np.mean(s_m == s_t) * 100.0 - np.mean(s_b == s_t) * 100.0

        c_m, c_b = [], []
        for cat in CATEGORIES:
            c_mask = (s_c == cat)
            if np.sum(c_mask) > 0:
                acc_m = np.mean(s_m[c_mask] == s_t[c_mask]) * 100.0
                acc_b = np.mean(s_b[c_mask] == s_t[c_mask]) * 100.0
            else:
                acc_m, acc_b = 0.0, 0.0
            cat_delta_boots[cat][b] = acc_m - acc_b
            c_m.append(acc_m)
            c_b.append(acc_b)
        delta_aa_boots[b] = np.mean(c_m) - np.mean(c_b)

    ci_delta_oa = (float(np.percentile(delta_oa_boots, 2.5)), float(np.percentile(delta_oa_boots, 97.5)))
    ci_delta_aa = (float(np.percentile(delta_aa_boots, 2.5)), float(np.percentile(delta_aa_boots, 97.5)))

    per_cat_cis = {}
    for cat in CATEGORIES:
        b_dist = cat_delta_boots[cat]
        low = float(np.percentile(b_dist, 2.5))
        high = float(np.percentile(b_dist, 97.5))
        per_cat_cis[cat] = {
            "model_acc": round(point_cat_accs_m[cat], 2),
            "blind_acc": round(point_cat_accs_b[cat], 2),
            "delta": round(point_cat_accs_m[cat] - point_cat_accs_b[cat], 2),
            "ci_95": [round(low, 2), round(high, 2)],
            "significant": (low > 0) or (high < 0)
        }

    return {
        "model_oa": round(point_oa_model, 2),
        "blind_oa": round(point_oa_blind, 2),
        "delta_oa": round(point_delta_oa, 2),
        "paired_delta_oa_ci_95": [round(ci_delta_oa[0], 2), round(ci_delta_oa[1], 2)],
        "delta_oa_significant": (ci_delta_oa[0] > 0) or (ci_delta_oa[1] < 0),
        "model_aa": round(point_aa_model, 2),
        "blind_aa": round(point_aa_blind, 2),
        "delta_aa": round(point_delta_aa, 2),
        "paired_delta_aa_ci_95": [round(ci_delta_aa[0], 2), round(ci_delta_aa[1], 2)],
        "delta_aa_significant": (ci_delta_aa[0] > 0) or (ci_delta_aa[1] < 0),
        "per_category": per_cat_cis
    }

def main():
    print("=" * 70)
    print("CDVQA Full 1,600-Train-Pair Head Retraining (Visual Dropout + Blanking)")
    print("=" * 70)

    train_annots, train_pairs = load_all_annotations("train")
    val_annots, val_pairs = load_all_annotations("val")

    print(f"Dataset Scope: 1,600 Train Pairs ({len(train_annots):,} QAs) | 400 Val Pairs ({len(val_annots):,} QAs)")

    cnet_cache = CACHE_DIR / "cnet_pair_feats.pt"
    q_cache = CACHE_DIR / "q_feat_dict.pt"

    cnet_feats = ensure_changenet_features(train_annots + val_annots, cnet_cache)
    q_feats = ensure_question_features(train_annots + val_annots, q_cache)

    y_train = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in train_annots], dtype=torch.long)
    y_val = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in val_annots], dtype=torch.long)

    v_train = torch.stack([cnet_feats[a["pair_id"]] for a in train_annots])
    v_val = torch.stack([cnet_feats[a["pair_id"]] for a in val_annots])

    q_train = torch.stack([q_feats[a["question"]] for a in train_annots])
    q_val = torch.stack([q_feats[a["question"]] for a in val_annots])

    val_pairs_arr = np.array([a["pair_id"] for a in val_annots])
    val_targets_arr = y_val.numpy()
    val_cats_arr = np.array([a["question_type"] for a in val_annots])

    # 1. Train Blind Baseline Head (Language only) on all 1,600 pairs
    print("\n--- [1/2] Training Blind Question-Only Baseline (All 1,600 Train Pairs) ---", flush=True)
    blind_head = BlindVQAHead(q_dim=q_train.shape[1], hidden_dim=256, num_classes=NUM_CLASSES)
    optimizer_b = torch.optim.AdamW(blind_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds_b = TensorDataset(q_train, y_train)
    loader_b = DataLoader(ds_b, batch_size=256, shuffle=True)

    for ep in range(1, 16):
        blind_head.train()
        for b_q, b_y in loader_b:
            optimizer_b.zero_grad()
            out = blind_head(b_q)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer_b.step()

    blind_head.eval()
    with torch.no_grad():
        blind_val_preds = torch.argmax(blind_head(q_val), dim=-1).cpu().numpy()

    blind_oa = float(np.mean(blind_val_preds == val_targets_arr)) * 100.0
    print(f"Blind Baseline on 400 Val Pairs: OA = {blind_oa:.2f}%", flush=True)

    # 2. Train Regularized ChangeNet Head (Visual Dropout 0.4 + Blanking 0.2)
    print("\n--- [2/2] Training Regularized ChangeNet Head (Visual Dropout 0.4 + Blanking 0.2) ---", flush=True)
    reg_head = RegularizedVQAHead(
        vis_dim=v_train.shape[1], q_dim=q_train.shape[1], hidden_dim=256, num_classes=NUM_CLASSES,
        p_vis_drop=0.4, p_blank=0.2
    )
    optimizer_r = torch.optim.AdamW(reg_head.parameters(), lr=1e-3, weight_decay=1e-4)

    ds_r = TensorDataset(v_train, q_train, y_train)
    loader_r = DataLoader(ds_r, batch_size=256, shuffle=True)

    for ep in range(1, 16):
        reg_head.train()
        for b_v, b_q, b_y in loader_r:
            optimizer_r.zero_grad()
            out = reg_head(b_v, b_q)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer_r.step()

    reg_head.eval()
    with torch.no_grad():
        cnet_val_preds = torch.argmax(reg_head(v_val, q_val), dim=-1).cpu().numpy()

    cnet_oa = float(np.mean(cnet_val_preds == val_targets_arr)) * 100.0
    print(f"Regularized ChangeNet on 400 Val Pairs: OA = {cnet_oa:.2f}%", flush=True)

    # 3. Compute Paired Cluster Bootstrap
    print("\nComputing 1,000-resample paired cluster bootstrap over 400 scene pairs...", flush=True)
    paired_result = compute_paired_cluster_bootstrap(
        val_pairs_arr, cnet_val_preds, blind_val_preds, val_targets_arr, val_cats_arr,
        n_bootstrap=1000, seed=SEED
    )

    print("\n" + "=" * 70)
    print("RESULTS: Regularized ChangeNet Head (All 1,600 Train Pairs) vs Blind Baseline")
    print("=" * 70)
    print(f"Overall Accuracy (OA): ChangeNet={paired_result['model_oa']}% | Blind={paired_result['blind_oa']}% | Delta={paired_result['delta_oa']:+.2f}%")
    print(f"  Paired 95% CI: [{paired_result['paired_delta_oa_ci_95'][0]:+.2f}%, {paired_result['paired_delta_oa_ci_95'][1]:+.2f}%] (Significant: {paired_result['delta_oa_significant']})")
    print(f"Average Accuracy (AA): ChangeNet={paired_result['model_aa']}% | Blind={paired_result['blind_aa']}% | Delta={paired_result['delta_aa']:+.2f}%")
    print(f"  Paired 95% CI: [{paired_result['paired_delta_aa_ci_95'][0]:+.2f}%, {paired_result['paired_delta_aa_ci_95'][1]:+.2f}%] (Significant: {paired_result['delta_aa_significant']})")

    print("\n--- Per-Category Breakdown ---")
    print(f"{'Category':<22} | {'ChangeNet':<10} | {'Blind':<10} | {'Delta':<8} | {'Paired 95% CI':<18} | {'Significant?'}")
    print("-" * 85)
    for cat, info in paired_result["per_category"].items():
        ci_str = f"[{info['ci_95'][0]:+.2f}, {info['ci_95'][1]:+.2f}]"
        sig_str = "YES" if info["significant"] else "No (spans 0)"
        print(f"{cat:<22} | {info['model_acc']:>8.2f}% | {info['blind_acc']:>8.2f}% | {info['delta']:>+6.2f}% | {ci_str:<18} | {sig_str}")

    report_payload = {
        "benchmark": "Phase 1 CDVQA Head Retrained on All 1,600 Pairs with Visual Dropout & Blanking",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regularization": {
            "visual_branch_dropout": 0.4,
            "random_visual_blanking_p": 0.2,
            "train_pairs_count": len(train_pairs),
            "train_questions_count": len(train_annots),
            "val_pairs_count": len(val_pairs),
            "val_questions_count": len(val_annots),
            "epochs": 15,
            "batch_size": 256
        },
        "results": paired_result
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\nWrote regularized retraining report to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
