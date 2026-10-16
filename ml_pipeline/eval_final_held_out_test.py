#!/usr/bin/env python3
"""
ml_pipeline/eval_final_held_out_test.py

Executes the final, controlled held-out test evaluation comparing:
1. Blind Question-Only Baseline
2. ChangeNet Visual + Question Feature Head

Protocol & Integrity Guarantees:
- Single execution on the primary official CDVQA held-out test set (39,686 questions, 968 scene pairs).
- Quarantined: Zero prior evaluations of this test partition for the feature head.
- Checkpoints audited: SatQueryChangeNet v2 (SHA-256 verified) + RemoteCLIP ViT-B/32 (SHA-256 verified).
- Trained heads: Deterministically trained on all 1,600 authentic train pairs (65,967 QAs) for seeds 42, 43, 44.
- Validation fidelity: Pre-evaluation check verifies identical validation metrics against cdvqa_head_vs_blind_3seeds.json.
- Checkpoint persistence: Exact model state dictionaries saved to checkpoints/cdvqa_heads/ with SHA-256 hashes recorded.
- Clustered Bootstrap: 1,000 resamples at the 968 scene-pair cluster level.

Output: reports/benchmark_evaluation/cdvqa_final_test_evaluation.json
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
CACHE_DIR = ML_DIR / "dataset_cache" / "phase1_full_cdvqa"
CHECKPOINTS_DIR = ML_DIR / "checkpoints" / "cdvqa_heads"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark_evaluation"
OUT_REPORT = REPORTS_DIR / "cdvqa_final_test_evaluation.json"
PRED_CACHE = REPORTS_DIR / "cdvqa_final_test_predictions.pt"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))
from change_model import SatQueryChangeNet

torch.set_num_threads(8)
SEEDS = [42, 43, 44]
PRIMARY_SEED = 42

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

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().upper()

class BlindVQAHead(nn.Module):
    def __init__(self, q_dim: int = 512, hidden_dim: int = 256, num_classes: int = NUM_CLASSES):
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

class FeatureVQAHead(nn.Module):
    def __init__(self, vis_dim: int = 512, q_dim: int = 512, hidden_dim: int = 256, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vis_dim + q_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, vis_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([vis_feat, q_feat], dim=-1)
        return self.net(x)

def load_split_annotations(split: str):
    ann_path = ML_DIR / "authentic_data" / "cdvqa_prepared" / split / "cdvqa_annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        annots = json.load(f)
    filtered = [a for a in annots if a["answer"].lower().strip() in ANS2IDX]
    return filtered

def extract_test_changenet_features(test_annots: List[Dict], cnet_test_cache: Path) -> Dict[str, torch.Tensor]:
    if cnet_test_cache.exists():
        print(f"Loading cached test ChangeNet features from: {cnet_test_cache}")
        return torch.load(cnet_test_cache, map_location="cpu")

    print(f"Extracting ChangeNet features for 968 test scene pairs from authentic disk rasters...")
    changenet_ckpt = ML_DIR / "checkpoints" / "satquery_change_net_finetune" / "v2" / "best_change_net_finetuned.pth"
    cnet_hash = compute_sha256(changenet_ckpt)
    print(f"  ChangeNet Checkpoint: {changenet_ckpt.name} (SHA-256: {cnet_hash})")

    changenet = SatQueryChangeNet()
    raw_ckpt = torch.load(changenet_ckpt, map_location="cpu")
    sd = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
    changenet.load_state_dict(sd)
    changenet.eval()

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    unique_pairs = sorted(list(set(a["pair_id"] for a in test_annots)))

    cnet_feats = {}
    cdvqa_img_dir = ML_DIR / "authentic_data" / "cdvqa" / "images"

    t0 = time.time()
    with torch.no_grad():
        for idx, pid in enumerate(unique_pairs, 1):
            im1_p = cdvqa_img_dir / "im1" / f"{pid}.png"
            im2_p = cdvqa_img_dir / "im2" / f"{pid}.png"

            im1 = transform(Image.open(im1_p).convert("RGB")).unsqueeze(0)
            im2 = transform(Image.open(im2_p).convert("RGB")).unsqueeze(0)

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

            if idx % 200 == 0 or idx == len(unique_pairs):
                print(f"  Processed {idx}/{len(unique_pairs)} test scene pairs ({idx / (time.time() - t0):.1f} pairs/sec)")

    torch.save(cnet_feats, cnet_test_cache)
    print(f"Saved {len(cnet_feats)} test pair features to {cnet_test_cache} in {time.time() - t0:.2f}s")
    return cnet_feats

def calc_metrics(preds: np.ndarray, targets: np.ndarray, cats: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    oa = float(np.mean(preds == targets) * 100.0)
    cat_accs = {}
    for c in CATEGORIES:
        mask = (cats == c)
        cat_accs[c] = float(np.mean(preds[mask] == targets[mask]) * 100.0) if np.sum(mask) > 0 else 0.0
    aa = float(np.mean(list(cat_accs.values())))
    return oa, aa, cat_accs

def run_clustered_bootstrap(
    test_pairs_arr: np.ndarray,
    feat_preds: np.ndarray,
    blind_preds: np.ndarray,
    targets: np.ndarray,
    cats: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    unique_pairs = np.unique(test_pairs_arr)
    n_pairs = len(unique_pairs)
    pair_to_indices = {p: np.where(test_pairs_arr == p)[0] for p in unique_pairs}

    rng = np.random.RandomState(seed)
    delta_oa_boots = np.zeros(n_bootstrap, dtype=np.float32)
    delta_aa_boots = np.zeros(n_bootstrap, dtype=np.float32)
    cat_delta_boots = {c: np.zeros(n_bootstrap, dtype=np.float32) for c in CATEGORIES}

    print(f"\nRunning {n_bootstrap}-resample scene-pair clustered bootstrap over {n_pairs} test scene pairs...")
    t0 = time.time()
    for b in range(n_bootstrap):
        sampled_pairs = rng.choice(unique_pairs, size=n_pairs, replace=True)
        sampled_indices = np.concatenate([pair_to_indices[p] for p in sampled_pairs])

        s_feat = feat_preds[sampled_indices]
        s_blind = blind_preds[sampled_indices]
        s_tgt = targets[sampled_indices]
        s_cat = cats[sampled_indices]

        # Overall Accuracy delta
        delta_oa_boots[b] = (np.mean(s_feat == s_tgt) - np.mean(s_blind == s_tgt)) * 100.0

        # Category and AA deltas
        c_deltas = []
        for c in CATEGORIES:
            c_mask = (s_cat == c)
            if np.sum(c_mask) > 0:
                acc_f = np.mean(s_feat[c_mask] == s_tgt[c_mask]) * 100.0
                acc_b = np.mean(s_blind[c_mask] == s_tgt[c_mask]) * 100.0
                d = acc_f - acc_b
            else:
                d = 0.0
            cat_delta_boots[c][b] = d
            c_deltas.append(d)

        delta_aa_boots[b] = np.mean(c_deltas)

    print(f"Bootstrap completed in {time.time() - t0:.2f}s.")

    # Calculate 95% empirical percentile CIs (nominal alpha = 0.05)
    ci_delta_oa = [round(float(np.percentile(delta_oa_boots, 2.5)), 2), round(float(np.percentile(delta_oa_boots, 97.5)), 2)]
    ci_delta_aa = [round(float(np.percentile(delta_aa_boots, 2.5)), 2), round(float(np.percentile(delta_aa_boots, 97.5)), 2)]

    # Empirical two-tailed p-value for OA and AA
    p_oa_empirical = float(2.0 * min(np.mean(delta_oa_boots <= 0), np.mean(delta_oa_boots >= 0)))
    p_aa_empirical = float(2.0 * min(np.mean(delta_aa_boots <= 0), np.mean(delta_aa_boots >= 0)))

    # Per-category intervals: both uncorrected 95% CIs and Bonferroni-corrected 99.375% CIs (alpha = 0.05 / 8 = 0.00625)
    bonf_alpha_tail = (0.05 / 8.0) / 2.0 * 100.0 # 0.3125%
    per_cat_analysis = {}
    for c in CATEGORIES:
        b_dist = cat_delta_boots[c]
        c_low_95 = round(float(np.percentile(b_dist, 2.5)), 2)
        c_high_95 = round(float(np.percentile(b_dist, 97.5)), 2)

        c_low_bonf = round(float(np.percentile(b_dist, bonf_alpha_tail)), 2)
        c_high_bonf = round(float(np.percentile(b_dist, 100.0 - bonf_alpha_tail)), 2)

        p_emp = float(2.0 * min(np.mean(b_dist <= 0), np.mean(b_dist >= 0)))
        survives_bonf = bool(c_low_bonf > 0.0 or c_high_bonf < 0.0)

        per_cat_analysis[c] = {
            "uncorrected_ci_95": [c_low_95, c_high_95],
            "bonferroni_ci_99_375": [c_low_bonf, c_high_bonf],
            "empirical_bootstrap_p_value": round(p_emp, 4),
            "statistically_significant_nominal": bool(c_low_95 > 0.0 or c_high_95 < 0.0),
            "survives_bonferroni": survives_bonf
        }

    return {
        "n_bootstrap": n_bootstrap,
        "cluster_level": f"scene_pair ({n_pairs} unique test scene pairs)",
        "delta_oa_ci_95": ci_delta_oa,
        "delta_oa_empirical_p": round(p_oa_empirical, 4),
        "delta_oa_statistically_significant": bool(ci_delta_oa[0] > 0.0),
        "delta_aa_ci_95": ci_delta_aa,
        "delta_aa_empirical_p": round(p_aa_empirical, 4),
        "delta_aa_statistically_significant": bool(ci_delta_aa[0] > 0.0),
        "per_category": per_cat_analysis
    }

def main():
    print("=" * 80)
    print("🛰️  SatQuery AI: FINAL HELD-OUT TEST EVALUATION (Phase 2)")
    print("=" * 80)

    # 1. Integrity Check & Load Checkpoints
    print("\n[Phase 1] Pre-Evaluation Integrity & Checkpoint Verification...")
    changenet_ckpt_path = ML_DIR / "checkpoints" / "satquery_change_net_finetune" / "v2" / "best_change_net_finetuned.pth"
    remoteclip_ckpt_path = ML_DIR / "authentic_data" / "_downloads" / "RemoteCLIP-ViT-B-32.pt"

    cnet_hash = compute_sha256(changenet_ckpt_path)
    rclip_hash = compute_sha256(remoteclip_ckpt_path)
    print(f"  ChangeNet v2 Checkpoint SHA-256: {cnet_hash}")
    print(f"  RemoteCLIP ViT-B/32 Checkpoint SHA-256: {rclip_hash}")

    assert cnet_hash == "95FB4CEB93F9E40B3C778B2FAF384E12D3278C8961892FD8B4A287836207F6B9", "ChangeNet hash mismatch!"
    assert rclip_hash == "60014E395D930A3F2963D1D89C8522BF4AD56775571E4356E866864789AF85C4", "RemoteCLIP hash mismatch!"

    # Load annotations
    train_annots = load_split_annotations("train")
    val_annots = load_split_annotations("val")
    test_annots = load_split_annotations("test")

    print(f"\nDataset Scope:")
    print(f"  Train: {len(train_annots):,} QAs across 1,600 scene pairs")
    print(f"  Val:   {len(val_annots):,} QAs across 400 scene pairs")
    print(f"  Test:  {len(test_annots):,} QAs across 968 scene pairs (HELD-OUT QUARANTINE)")

    # Load frozen representations
    cnet_trainval_feats = torch.load(CACHE_DIR / "cnet_pair_feats.pt", map_location="cpu")
    q_feats_dict = torch.load(CACHE_DIR / "q_feat_dict.pt", map_location="cpu")

    # Extract test ChangeNet features
    cnet_test_feats = extract_test_changenet_features(test_annots, CACHE_DIR / "cnet_test_pair_feats.pt")

    # Build evaluation tensors
    y_train = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in train_annots], dtype=torch.long)
    y_val = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in val_annots], dtype=torch.long)
    y_test = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in test_annots], dtype=torch.long)

    v_train = torch.stack([cnet_trainval_feats[a["pair_id"]] for a in train_annots])
    v_val = torch.stack([cnet_trainval_feats[a["pair_id"]] for a in val_annots])
    v_test = torch.stack([cnet_test_feats[a["pair_id"]] for a in test_annots])

    q_train = torch.stack([q_feats_dict[a["question"]] for a in train_annots])
    q_val = torch.stack([q_feats_dict[a["question"]] for a in val_annots])
    q_test = torch.stack([q_feats_dict[a["question"]] for a in test_annots])

    val_cats = np.array([a["question_type"] for a in val_annots])
    val_targets = y_val.numpy()

    test_pairs = np.array([a["pair_id"] for a in test_annots])
    test_cats = np.array([a["question_type"] for a in test_annots])
    test_targets = y_test.numpy()

    # Pre-evaluation check: Train seeds 42, 43, 44 and verify validation fidelity
    print("\n[Phase 1 Check] Verifying model training determinism and validation fidelity against cdvqa_head_vs_blind_3seeds.json...")
    audited_val_expected = {
        42: {"blind_oa": 67.78, "feat_oa": 68.75},
        43: {"blind_oa": 68.03, "feat_oa": 68.38},
        44: {"blind_oa": 68.09, "feat_oa": 68.55}
    }

    per_seed_results = {}
    saved_head_hashes = {}

    criterion = nn.CrossEntropyLoss()

    for s in SEEDS:
        print(f"\n--- Training Audited Models for Seed {s} (15 Epochs on 1,600 Train Pairs) ---")
        torch.manual_seed(s)
        np.random.seed(s)

        # 1. Blind Model
        blind_model = BlindVQAHead(q_dim=q_train.shape[1], hidden_dim=256, num_classes=NUM_CLASSES)
        opt_b = torch.optim.AdamW(blind_model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader_b = DataLoader(TensorDataset(q_train, y_train), batch_size=256, shuffle=True)

        for _ in range(15):
            blind_model.train()
            for bq, by in loader_b:
                opt_b.zero_grad()
                loss = criterion(blind_model(bq), by)
                loss.backward()
                opt_b.step()

        blind_model.eval()
        with torch.no_grad():
            b_val_p = torch.argmax(blind_model(q_val), dim=-1).cpu().numpy()
            b_test_p = torch.argmax(blind_model(q_test), dim=-1).cpu().numpy()

        b_val_oa = round(float(np.mean(b_val_p == val_targets) * 100.0), 2)

        # Save blind checkpoint
        blind_ckpt_file = CHECKPOINTS_DIR / f"blind_head_seed{s}.pth"
        torch.save(blind_model.state_dict(), blind_ckpt_file)
        b_hash = compute_sha256(blind_ckpt_file)
        saved_head_hashes[f"blind_head_seed{s}"] = {"path": str(blind_ckpt_file), "sha256": b_hash}

        # 2. Feature Head Model
        feat_model = FeatureVQAHead(vis_dim=v_train.shape[1], q_dim=q_train.shape[1], hidden_dim=256, num_classes=NUM_CLASSES)
        opt_f = torch.optim.AdamW(feat_model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader_f = DataLoader(TensorDataset(v_train, q_train, y_train), batch_size=256, shuffle=True)

        for _ in range(15):
            feat_model.train()
            for bv, bq, by in loader_f:
                opt_f.zero_grad()
                loss = criterion(feat_model(bv, bq), by)
                loss.backward()
                opt_f.step()

        feat_model.eval()
        with torch.no_grad():
            f_val_p = torch.argmax(feat_model(v_val, q_val), dim=-1).cpu().numpy()
            f_test_p = torch.argmax(feat_model(v_test, q_test), dim=-1).cpu().numpy()

        f_val_oa = round(float(np.mean(f_val_p == val_targets) * 100.0), 2)

        # Save feat checkpoint
        feat_ckpt_file = CHECKPOINTS_DIR / f"feat_head_seed{s}.pth"
        torch.save(feat_model.state_dict(), feat_ckpt_file)
        f_hash = compute_sha256(feat_ckpt_file)
        saved_head_hashes[f"feat_head_seed{s}"] = {"path": str(feat_ckpt_file), "sha256": f_hash}

        print(f"  Seed {s} Validation Check:")
        print(f"    Blind Val OA: {b_val_oa:.2f}% (Expected: {audited_val_expected[s]['blind_oa']}%)")
        print(f"    Feat  Val OA: {f_val_oa:.2f}% (Expected: {audited_val_expected[s]['feat_oa']}%)")
        assert abs(b_val_oa - audited_val_expected[s]['blind_oa']) <= 0.05, f"Validation Blind OA discrepancy for seed {s}"
        assert abs(f_val_oa - audited_val_expected[s]['feat_oa']) <= 0.05, f"Validation Feature OA discrepancy for seed {s}"

        # Evaluate on Final Held-Out Test Set
        b_test_oa, b_test_aa, b_test_cat = calc_metrics(b_test_p, test_targets, test_cats)
        f_test_oa, f_test_aa, f_test_cat = calc_metrics(f_test_p, test_targets, test_cats)

        d_test_oa = round(f_test_oa - b_test_oa, 2)
        d_test_aa = round(f_test_aa - b_test_aa, 2)
        d_test_cat = {c: round(f_test_cat[c] - b_test_cat[c], 2) for c in CATEGORIES}

        per_seed_results[s] = {
            "blind_test_preds": b_test_p,
            "feat_test_preds": f_test_p,
            "blind_test_oa": round(b_test_oa, 2),
            "blind_test_aa": round(b_test_aa, 2),
            "blind_test_categories": {k: round(v, 2) for k, v in b_test_cat.items()},
            "feat_test_oa": round(f_test_oa, 2),
            "feat_test_aa": round(f_test_aa, 2),
            "feat_test_categories": {k: round(v, 2) for k, v in f_test_cat.items()},
            "delta_oa": d_test_oa,
            "delta_aa": d_test_aa,
            "delta_categories": d_test_cat
        }
        print(f"  Seed {s} Final Test Results (N = 39,686 questions):")
        print(f"    Blind Test OA: {b_test_oa:.2f}% | Feat Test OA: {f_test_oa:.2f}% | Delta: {d_test_oa:+.2f}%")
        print(f"    Blind Test AA: {b_test_aa:.2f}% | Feat Test AA: {f_test_aa:.2f}% | Delta: {d_test_aa:+.2f}%")

    print("\n" + "=" * 80)
    print("PHASE 2: PRIMARY AUDIT EVALUATION (Seed 42)")
    print("=" * 80)

    # 3. Clustered Bootstrap on Primary Seed (Seed 42)
    s42 = per_seed_results[PRIMARY_SEED]
    bootstrap_results = run_clustered_bootstrap(
        test_pairs,
        s42["feat_test_preds"],
        s42["blind_test_preds"],
        test_targets,
        test_cats,
        n_bootstrap=1000,
        seed=PRIMARY_SEED
    )

    # 4. Aggregate Multi-Seed Test Analysis across seeds 42, 43, 44
    b_test_oas = [per_seed_results[s]["blind_test_oa"] for s in SEEDS]
    b_test_aas = [per_seed_results[s]["blind_test_aa"] for s in SEEDS]
    f_test_oas = [per_seed_results[s]["feat_test_oa"] for s in SEEDS]
    f_test_aas = [per_seed_results[s]["feat_test_aa"] for s in SEEDS]
    d_test_oas = [per_seed_results[s]["delta_oa"] for s in SEEDS]
    d_test_aas = [per_seed_results[s]["delta_aa"] for s in SEEDS]

    d_oa_mean = round(float(np.mean(d_test_oas)), 2)
    d_oa_std = round(float(np.std(d_test_oas)), 2)
    d_aa_mean = round(float(np.mean(d_test_aas)), 2)
    d_aa_std = round(float(np.std(d_test_aas)), 2)

    cat_stats_test = {}
    for c in CATEGORIES:
        b_c = [per_seed_results[s]["blind_test_categories"][c] for s in SEEDS]
        f_c = [per_seed_results[s]["feat_test_categories"][c] for s in SEEDS]
        d_c = [per_seed_results[s]["delta_categories"][c] for s in SEEDS]
        cat_stats_test[c] = {
            "blind_test_mean": round(float(np.mean(b_c)), 2),
            "blind_test_std": round(float(np.std(b_c)), 2),
            "feat_test_mean": round(float(np.mean(f_c)), 2),
            "feat_test_std": round(float(np.std(f_c)), 2),
            "delta_mean": round(float(np.mean(d_c)), 2),
            "delta_std": round(float(np.std(d_c)), 2),
            "individual_deltas": d_c
        }

    # Save raw prediction tensors for 100% reproducibility
    torch.save({
        "test_targets": test_targets,
        "test_cats": test_cats,
        "test_pairs": test_pairs,
        "blind_preds_by_seed": {s: per_seed_results[s]["blind_test_preds"] for s in SEEDS},
        "feat_preds_by_seed": {s: per_seed_results[s]["feat_test_preds"] for s in SEEDS}
    }, PRED_CACHE)
    print(f"\nRaw test prediction arrays saved to: {PRED_CACHE}")

    # Build Final JSON Report
    final_report = {
        "benchmark": "SatQuery AI CDVQA Final Held-Out Test Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_partition": "authentic_data/cdvqa_prepared/test/cdvqa_annotations.json",
        "dataset_scope": {
            "train_pairs": 1600,
            "train_questions": len(train_annots),
            "val_pairs": 400,
            "val_questions": len(val_annots),
            "test_pairs": len(np.unique(test_pairs)),
            "test_questions": len(test_annots),
            "geographic_coverage": "968 aerial scene pairs from SECOND Wuhan University benchmark; unseen spatial tiles within benchmark distribution."
        },
        "model_checkpoints": {
            "visual_backbone": {
                "name": "SatQueryChangeNet v2",
                "checkpoint": str(changenet_ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                "sha256": cnet_hash
            },
            "text_encoder": {
                "name": "RemoteCLIP ViT-B/32",
                "checkpoint": str(remoteclip_ckpt_path.relative_to(ROOT_DIR)).replace("\\", "/"),
                "sha256": rclip_hash
            },
            "evaluated_heads": saved_head_hashes
        },
        "primary_evaluation_seed_42": {
            "blind_test_oa": s42["blind_test_oa"],
            "blind_test_aa": s42["blind_test_aa"],
            "feat_test_oa": s42["feat_test_oa"],
            "feat_test_aa": s42["feat_test_aa"],
            "delta_oa": s42["delta_oa"],
            "delta_aa": s42["delta_aa"],
            "blind_categories": s42["blind_test_categories"],
            "feat_categories": s42["feat_test_categories"],
            "delta_categories": s42["delta_categories"],
            "scene_pair_clustered_bootstrap": bootstrap_results
        },
        "three_seed_test_aggregate": {
            "seeds": SEEDS,
            "blind_test_oa_values": b_test_oas,
            "blind_test_oa_mean": round(float(np.mean(b_test_oas)), 2),
            "blind_test_oa_std": round(float(np.std(b_test_oas)), 2),
            "blind_test_aa_values": b_test_aas,
            "blind_test_aa_mean": round(float(np.mean(b_test_aas)), 2),
            "blind_test_aa_std": round(float(np.std(b_test_aas)), 2),
            "feat_test_oa_values": f_test_oas,
            "feat_test_oa_mean": round(float(np.mean(f_test_oas)), 2),
            "feat_test_oa_std": round(float(np.std(f_test_oas)), 2),
            "feat_test_aa_values": f_test_aas,
            "feat_test_aa_mean": round(float(np.mean(f_test_aas)), 2),
            "feat_test_aa_std": round(float(np.std(f_test_aas)), 2),
            "delta_oa_values": d_test_oas,
            "delta_oa_mean": d_oa_mean,
            "delta_oa_std": d_oa_std,
            "delta_aa_values": d_test_aas,
            "delta_aa_mean": d_aa_mean,
            "delta_aa_std": d_aa_std,
            "per_category_analysis": cat_stats_test
        },
        "comparison_vs_validation_audit": {
            "notes": "Validation and test sets represent distinct non-overlapping geographic partitions. Comparing validation to test reflects generalization across disjoint tile sets within SECOND.",
            "validation_point_estimates": {
                "blind_oa_mean": 67.97,
                "feat_oa_mean": 68.56,
                "delta_oa_mean": 0.59,
                "blind_aa_mean": 58.39,
                "feat_aa_mean": 59.67,
                "delta_aa_mean": 1.28
            },
            "test_point_estimates": {
                "blind_oa_mean": round(float(np.mean(b_test_oas)), 2),
                "feat_oa_mean": round(float(np.mean(f_test_oas)), 2),
                "delta_oa_mean": d_oa_mean,
                "blind_aa_mean": round(float(np.mean(b_test_aas)), 2),
                "feat_test_aa_mean": round(float(np.mean(f_test_aas)), 2),
                "delta_aa_mean": d_aa_mean
            }
        }
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved final evaluation report to: {OUT_REPORT}")
    print("=" * 80)

if __name__ == "__main__":
    main()
