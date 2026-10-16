#!/usr/bin/env python3
"""
ml_pipeline/phase1_full_cdvqa_comparison.py

Phase 1 Redo: Full Task Multi-Modal Encoder Comparison Benchmark for CDVQA.
Evaluates across ALL 8 CDVQA Question Categories with the Question as Input:
1. change_or_not
2. increase_or_not
3. decrease_or_not
4. change_to_what
5. smallest_change
6. largest_change
7. change_ratio
8. change_ratio_types

Specifications:
- Training: >= 800 authentic scene pairs from CDVQA train split.
- Validation: ALL 400 authentic scene pairs (16,441 questions) from CDVQA val split.
- Held-out Test split: 100% UNTOUCHED (quarantined strictly for the final chosen model).
- Metrics:
  - Overall Accuracy (OA)
  - Average Accuracy (AA across the 8 categories)
  - Cluster Bootstrap 95% Confidence Intervals over scene pairs (1,000 resamples)
- Baselines Evaluated:
  1. Per-Category Majority Baseline (Training Prior & CDVQA Test 1 50.8 OA / 44.8 AA reference)
  2. Blind Question-Only Baseline (Question Q -> MLP, no images)
- Vision Candidates Evaluated (+ Question Q):
  3. ResNet-18 (ImageNet frozen, 512-d)
  4. ResNet-50 (ImageNet frozen, 2048-d)
  5. RemoteCLIP-ViT-B/32 (RS-domain frozen, 512-d)
  6. BLIP-ViT-B/16 (Salesforce BLIP vision frozen, 768-d)
  7. SatQueryChangeNet Encoder (Our trained ChangeNet v2 checkpoint, multi-scale difference features, 512-d)
- Reports explicit gain (Delta OA, Delta AA) over the blind baseline for each candidate.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

# Enforce UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image

import torchvision.models as models
import torchvision.transforms as T
import open_clip
from transformers import BlipForQuestionAnswering

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark_evaluation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ML_DIR / "dataset_cache" / "phase1_full_cdvqa"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = ML_DIR / "authentic_data" / "_downloads"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))
from change_model import SatQueryChangeNet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(8)

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

class VQAProbeHead(nn.Module):
    """Standardized 2-layer MLP classifier head over concatenated visual + question features."""
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

def load_cdvqa_annotations(split: str = "train", max_pairs: int = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    ann_path = ML_DIR / "authentic_data" / "cdvqa_prepared" / split / "cdvqa_annotations.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        raw_annots = json.load(f)

    # Filter to pairs if max_pairs is specified
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

    print(f"[{split.upper()}] Loaded {len(filtered):,} QA annotations across {len(allowed_pairs)} scene pairs.")
    return filtered, seen_pairs

def compute_cluster_bootstrap_ci(
    pair_ids: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    categories: np.ndarray,
    n_bootstrap: int = 1000
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence intervals for OA and AA clustered over physical scene pairs.
    Resamples unique pair IDs with replacement to account for intra-scene question correlation.
    """
    unique_pairs = np.unique(pair_ids)
    n_pairs = len(unique_pairs)
    
    # Pre-index question indices by pair
    pair_to_indices = {p: np.where(pair_ids == p)[0] for p in unique_pairs}
    
    oa_boots = np.zeros(n_bootstrap, dtype=np.float32)
    aa_boots = np.zeros(n_bootstrap, dtype=np.float32)

    rng = np.random.RandomState(SEED)
    for b in range(n_bootstrap):
        sampled_pairs = rng.choice(unique_pairs, size=n_pairs, replace=True)
        sampled_indices = np.concatenate([pair_to_indices[p] for p in sampled_pairs])

        s_preds = preds[sampled_indices]
        s_targets = targets[sampled_indices]
        s_cats = categories[sampled_indices]

        # Overall Accuracy
        oa = np.mean(s_preds == s_targets) * 100.0
        oa_boots[b] = oa

        # Average Accuracy (macro over categories present)
        cat_accs = []
        for cat in CATEGORIES:
            c_mask = (s_cats == cat)
            if np.sum(c_mask) > 0:
                cat_accs.append(np.mean(s_preds[c_mask] == s_targets[c_mask]) * 100.0)
        aa_boots[b] = np.mean(cat_accs) if cat_accs else 0.0

    oa_ci = (float(np.percentile(oa_boots, 2.5)), float(np.percentile(oa_boots, 97.5)))
    aa_ci = (float(np.percentile(aa_boots, 2.5)), float(np.percentile(aa_boots, 97.5)))
    return oa_ci, aa_ci

def extract_unique_image_features(
    unique_image_paths: List[Path],
    extractor_fn,
    batch_size: int = 32,
    desc: str = "images"
) -> Dict[str, torch.Tensor]:
    """Extracts features for a list of unique images using extractor_fn, returning path -> feature tensor."""
    features_dict = {}
    total = len(unique_image_paths)
    print(f"  Extracting features for {total:,} unique {desc}...", flush=True)

    for i in range(0, total, batch_size):
        batch_paths = unique_image_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        feats = extractor_fn(imgs)
        for p, f in zip(batch_paths, feats):
            features_dict[str(p)] = f.cpu()

    return features_dict

def main():
    print("=" * 70)
    print("SatQuery AI: Phase 1 Full CDVQA Multi-Modal Vision Encoder Benchmark")
    print("=" * 70, flush=True)

    # 1. Load Annotations: 800 Train Pairs, full 400 Val Pairs
    train_annots, train_pairs = load_cdvqa_annotations(split="train", max_pairs=800)
    val_annots, val_pairs = load_cdvqa_annotations(split="val", max_pairs=None)

    # Pre-extract unique image paths
    train_img_paths = sorted(list(set(
        [ROOT_DIR / a["t1_path"] for a in train_annots] + [ROOT_DIR / a["t2_path"] for a in train_annots]
    )))
    val_img_paths = sorted(list(set(
        [ROOT_DIR / a["t1_path"] for a in val_annots] + [ROOT_DIR / a["t2_path"] for a in val_annots]
    )))
    all_img_paths = sorted(list(set(train_img_paths + val_img_paths)))
    print(f"Total unique images to extract: {len(all_img_paths):,} (Train: {len(train_img_paths)}, Val: {len(val_img_paths)})", flush=True)

    # 2. Extract Question Features using RemoteCLIP Text Encoder
    print("\n--- [Step 1/7] Extracting Question Embeddings (RemoteCLIP Text Encoder) ---", flush=True)
    remoteclip_path = DOWNLOADS_DIR / "RemoteCLIP-ViT-B-32.pt"
    rc_model, _, rc_preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    rc_ckpt = torch.load(remoteclip_path, map_location="cpu")
    rc_model.load_state_dict(rc_ckpt)
    rc_model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    unique_questions = sorted(list(set([a["question"] for a in train_annots + val_annots])))
    print(f"Unique question templates: {len(unique_questions)}", flush=True)

    q_cache = CACHE_DIR / "q_feat_dict.pt"
    if q_cache.exists():
        print(f"  Loading cached question features from {q_cache}...", flush=True)
        q_feat_dict = torch.load(q_cache, map_location="cpu")
    else:
        q_feat_dict = {}
        with torch.no_grad():
            for i in range(0, len(unique_questions), 64):
                batch_q = unique_questions[i:i + 64]
                tokens = tokenizer(batch_q)
                q_emb = rc_model.encode_text(tokens)
                q_emb = F.normalize(q_emb, dim=-1)
                for q_str, emb in zip(batch_q, q_emb):
                    q_feat_dict[q_str] = emb.cpu()
        torch.save(q_feat_dict, q_cache)

    # Pre-build Question and Label tensors
    y_train = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in train_annots], dtype=torch.long)
    y_val = torch.tensor([ANS2IDX[a["answer"].lower().strip()] for a in val_annots], dtype=torch.long)

    val_pairs_arr = np.array([a["pair_id"] for a in val_annots])
    val_targets_arr = y_val.numpy()
    val_cats_arr = np.array([a["question_type"] for a in val_annots])

    q_train_tensor = torch.stack([q_feat_dict[a["question"]] for a in train_annots])
    q_val_tensor = torch.stack([q_feat_dict[a["question"]] for a in val_annots])
    print(f"Question embeddings ready: Train {q_train_tensor.shape}, Val {q_val_tensor.shape}", flush=True)

    # ---------------------------------------------------------
    # Baseline 1: Per-Category Majority Baseline
    # ---------------------------------------------------------
    print("\n--- [Step 2/7] Baseline 1: Per-Category Majority Baseline ---", flush=True)
    cat_train_answers = {cat: [] for cat in CATEGORIES}
    for a in train_annots:
        cat_train_answers[a["question_type"]].append(a["answer"].lower().strip())

    cat_majority_ans = {}
    for cat in CATEGORIES:
        counts = Counter(cat_train_answers[cat])
        maj_ans, _ = counts.most_common(1)[0]
        cat_majority_ans[cat] = maj_ans
    print("Learned Training Majority Answers per Category:", cat_majority_ans, flush=True)

    majority_preds_val = np.array([ANS2IDX[cat_majority_ans[a["question_type"]]] for a in val_annots])
    maj_oa = float(np.mean(majority_preds_val == val_targets_arr)) * 100.0
    maj_cat_accs = {
        cat: float(np.mean(majority_preds_val[val_cats_arr == cat] == val_targets_arr[val_cats_arr == cat])) * 100.0
        for cat in CATEGORIES
    }
    maj_aa = float(np.mean(list(maj_cat_accs.values())))
    maj_oa_ci, maj_aa_ci = compute_cluster_bootstrap_ci(val_pairs_arr, majority_preds_val, val_targets_arr, val_cats_arr)

    print(f"Per-Category Majority Baseline: OA = {maj_oa:.2f}% [{maj_oa_ci[0]:.2f}, {maj_oa_ci[1]:.2f}] | AA = {maj_aa:.2f}% [{maj_aa_ci[0]:.2f}, {maj_aa_ci[1]:.2f}]", flush=True)

    # ---------------------------------------------------------
    # Baseline 2: Blind Question-Only Baseline (No Images)
    # ---------------------------------------------------------
    print("\n--- [Step 3/7] Baseline 2: Blind Question-Only Baseline (Q -> MLP) ---", flush=True)
    def train_and_eval_head(x_tr: torch.Tensor, y_tr: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor, epochs: int = 15):
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
            val_preds = torch.argmax(val_logits, dim=-1).numpy()

        oa = float(np.mean(val_preds == y_v.numpy())) * 100.0
        cat_accs = {
            cat: float(np.mean(val_preds[val_cats_arr == cat] == y_v.numpy()[val_cats_arr == cat])) * 100.0
            for cat in CATEGORIES
        }
        aa = float(np.mean(list(cat_accs.values())))
        oa_ci, aa_ci = compute_cluster_bootstrap_ci(val_pairs_arr, val_preds, y_v.numpy(), val_cats_arr)
        return oa, aa, oa_ci, aa_ci, cat_accs

    blind_oa, blind_aa, blind_oa_ci, blind_aa_ci, blind_cat_accs = train_and_eval_head(
        q_train_tensor, y_train, q_val_tensor, y_val, epochs=15
    )
    print(f"Blind Question-Only Baseline: OA = {blind_oa:.2f}% [{blind_oa_ci[0]:.2f}, {blind_oa_ci[1]:.2f}] | AA = {blind_aa:.2f}% [{blind_aa_ci[0]:.2f}, {blind_aa_ci[1]:.2f}]", flush=True)

    results = {
        "per_category_majority": {
            "name": "Per-Category Majority Baseline",
            "published_test1_oa": 50.8,
            "published_test1_aa": 44.8,
            "val_oa": round(maj_oa, 2),
            "val_oa_ci_95": [round(maj_oa_ci[0], 2), round(maj_oa_ci[1], 2)],
            "val_aa": round(maj_aa, 2),
            "val_aa_ci_95": [round(maj_aa_ci[0], 2), round(maj_aa_ci[1], 2)],
            "category_accuracies": {k: round(v, 2) for k, v in maj_cat_accs.items()},
            "gain_over_blind_oa": round(maj_oa - blind_oa, 2),
            "gain_over_blind_aa": round(maj_aa - blind_aa, 2)
        },
        "blind_question_only": {
            "name": "Blind Question-Only Baseline",
            "val_oa": round(blind_oa, 2),
            "val_oa_ci_95": [round(blind_oa_ci[0], 2), round(blind_oa_ci[1], 2)],
            "val_aa": round(blind_aa, 2),
            "val_aa_ci_95": [round(blind_aa_ci[0], 2), round(blind_aa_ci[1], 2)],
            "category_accuracies": {k: round(v, 2) for k, v in blind_cat_accs.items()},
            "gain_over_blind_oa": 0.0,
            "gain_over_blind_aa": 0.0
        }
    }

    # Helper function to construct fused pair features
    def build_fused_pair_features(img_feats_dict: Dict[str, torch.Tensor], annots: List[Dict[str, Any]], q_tensor: torch.Tensor) -> torch.Tensor:
        fused_list = []
        for i, a in enumerate(annots):
            t1_p = str(ROOT_DIR / a["t1_path"])
            t2_p = str(ROOT_DIR / a["t2_path"])
            f1 = img_feats_dict[t1_p]
            f2 = img_feats_dict[t2_p]
            f_diff = torch.abs(f2 - f1)
            f_vis = torch.cat([f1, f2, f_diff], dim=-1)
            f_q = q_tensor[i]
            fused_list.append(torch.cat([f_vis, f_q], dim=-1))
        return torch.stack(fused_list)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Candidate 1: ResNet-18 (512-d)
    # ---------------------------------------------------------
    print("\n--- [Step 4/7] Candidate 1: ResNet-18 (Frozen ImageNet, 512-d) ---", flush=True)
    rn18_cache = CACHE_DIR / "rn18_feats.pt"
    if rn18_cache.exists():
        print(f"  Loading cached ResNet-18 features from {rn18_cache}...", flush=True)
        rn18_feats = torch.load(rn18_cache, map_location="cpu")
    else:
        rn18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        rn18_encoder = nn.Sequential(*list(rn18.children())[:-1]).eval()
        rn18_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def rn18_extractor_fn(pil_imgs):
            t = torch.stack([rn18_transform(img) for img in pil_imgs])
            with torch.no_grad():
                f = rn18_encoder(t).squeeze(-1).squeeze(-1)
                return F.normalize(f, dim=-1)

        rn18_feats = extract_unique_image_features(all_img_paths, rn18_extractor_fn, batch_size=64, desc="ResNet-18")
        torch.save(rn18_feats, rn18_cache)

    rn18_train_x = build_fused_pair_features(rn18_feats, train_annots, q_train_tensor)
    rn18_val_x = build_fused_pair_features(rn18_feats, val_annots, q_val_tensor)

    rn18_oa, rn18_aa, rn18_oa_ci, rn18_aa_ci, rn18_cats = train_and_eval_head(
        rn18_train_x, y_train, rn18_val_x, y_val, epochs=15
    )
    print(f"ResNet-18: OA = {rn18_oa:.2f}% [{rn18_oa_ci[0]:.2f}, {rn18_oa_ci[1]:.2f}] | AA = {rn18_aa:.2f}% [{rn18_aa_ci[0]:.2f}, {rn18_aa_ci[1]:.2f}] | Gain over Blind: +{rn18_oa - blind_oa:.2f}% OA, +{rn18_aa - blind_aa:.2f}% AA", flush=True)
    results["resnet18"] = {
        "name": "ResNet-18 (ImageNet Frozen)",
        "dim": 512,
        "val_oa": round(rn18_oa, 2),
        "val_oa_ci_95": [round(rn18_oa_ci[0], 2), round(rn18_oa_ci[1], 2)],
        "val_aa": round(rn18_aa, 2),
        "val_aa_ci_95": [round(rn18_aa_ci[0], 2), round(rn18_aa_ci[1], 2)],
        "category_accuracies": {k: round(v, 2) for k, v in rn18_cats.items()},
        "gain_over_blind_oa": round(rn18_oa - blind_oa, 2),
        "gain_over_blind_aa": round(rn18_aa - blind_aa, 2)
    }

    # ---------------------------------------------------------
    # Candidate 2: ResNet-50 (2048-d)
    # ---------------------------------------------------------
    print("\n--- [Step 5/7] Candidate 2: ResNet-50 (Frozen ImageNet, 2048-d) ---", flush=True)
    rn50_cache = CACHE_DIR / "rn50_feats.pt"
    if rn50_cache.exists():
        print(f"  Loading cached ResNet-50 features from {rn50_cache}...", flush=True)
        rn50_feats = torch.load(rn50_cache, map_location="cpu")
    else:
        rn50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        rn50_encoder = nn.Sequential(*list(rn50.children())[:-1]).eval()
        rn50_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def rn50_extractor_fn(pil_imgs):
            t = torch.stack([rn50_transform(img) for img in pil_imgs])
            with torch.no_grad():
                f = rn50_encoder(t).squeeze(-1).squeeze(-1)
                return F.normalize(f, dim=-1)

        rn50_feats = extract_unique_image_features(all_img_paths, rn50_extractor_fn, batch_size=32, desc="ResNet-50")
        torch.save(rn50_feats, rn50_cache)

    rn50_train_x = build_fused_pair_features(rn50_feats, train_annots, q_train_tensor)
    rn50_val_x = build_fused_pair_features(rn50_feats, val_annots, q_val_tensor)

    rn50_oa, rn50_aa, rn50_oa_ci, rn50_aa_ci, rn50_cats = train_and_eval_head(
        rn50_train_x, y_train, rn50_val_x, y_val, epochs=15
    )
    print(f"ResNet-50: OA = {rn50_oa:.2f}% [{rn50_oa_ci[0]:.2f}, {rn50_oa_ci[1]:.2f}] | AA = {rn50_aa:.2f}% [{rn50_aa_ci[0]:.2f}, {rn50_aa_ci[1]:.2f}] | Gain over Blind: +{rn50_oa - blind_oa:.2f}% OA, +{rn50_aa - blind_aa:.2f}% AA", flush=True)
    results["resnet50"] = {
        "name": "ResNet-50 (ImageNet Frozen)",
        "dim": 2048,
        "val_oa": round(rn50_oa, 2),
        "val_oa_ci_95": [round(rn50_oa_ci[0], 2), round(rn50_oa_ci[1], 2)],
        "val_aa": round(rn50_aa, 2),
        "val_aa_ci_95": [round(rn50_aa_ci[0], 2), round(rn50_aa_ci[1], 2)],
        "category_accuracies": {k: round(v, 2) for k, v in rn50_cats.items()},
        "gain_over_blind_oa": round(rn50_oa - blind_oa, 2),
        "gain_over_blind_aa": round(rn50_aa - blind_aa, 2)
    }

    # ---------------------------------------------------------
    # Candidate 3: RemoteCLIP-ViT-B/32 (512-d)
    # ---------------------------------------------------------
    print("\n--- [Step 6/7] Candidate 3: RemoteCLIP-ViT-B/32 (Frozen RS Domain, 512-d) ---", flush=True)
    rclip_cache = CACHE_DIR / "rclip_feats.pt"
    if rclip_cache.exists():
        print(f"  Loading cached RemoteCLIP features from {rclip_cache}...", flush=True)
        rclip_feats = torch.load(rclip_cache, map_location="cpu")
    else:
        def rclip_extractor_fn(pil_imgs):
            t = torch.stack([rc_preprocess(img) for img in pil_imgs])
            with torch.no_grad():
                f = rc_model.encode_image(t)
                return F.normalize(f, dim=-1)

        rclip_feats = extract_unique_image_features(all_img_paths, rclip_extractor_fn, batch_size=32, desc="RemoteCLIP")
        torch.save(rclip_feats, rclip_cache)

    rclip_train_x = build_fused_pair_features(rclip_feats, train_annots, q_train_tensor)
    rclip_val_x = build_fused_pair_features(rclip_feats, val_annots, q_val_tensor)

    rclip_oa, rclip_aa, rclip_oa_ci, rclip_aa_ci, rclip_cats = train_and_eval_head(
        rclip_train_x, y_train, rclip_val_x, y_val, epochs=15
    )
    print(f"RemoteCLIP: OA = {rclip_oa:.2f}% [{rclip_oa_ci[0]:.2f}, {rclip_oa_ci[1]:.2f}] | AA = {rclip_aa:.2f}% [{rclip_aa_ci[0]:.2f}, {rclip_aa_ci[1]:.2f}] | Gain over Blind: +{rclip_oa - blind_oa:.2f}% OA, +{rclip_aa - blind_aa:.2f}% AA", flush=True)
    results["remoteclip_vit_b32"] = {
        "name": "RemoteCLIP-ViT-B/32 (RS Domain Frozen)",
        "dim": 512,
        "val_oa": round(rclip_oa, 2),
        "val_oa_ci_95": [round(rclip_oa_ci[0], 2), round(rclip_oa_ci[1], 2)],
        "val_aa": round(rclip_aa, 2),
        "val_aa_ci_95": [round(rclip_aa_ci[0], 2), round(rclip_aa_ci[1], 2)],
        "category_accuracies": {k: round(v, 2) for k, v in rclip_cats.items()},
        "gain_over_blind_oa": round(rclip_oa - blind_oa, 2),
        "gain_over_blind_aa": round(rclip_aa - blind_aa, 2)
    }

    # ---------------------------------------------------------
    # Candidate 4: BLIP-ViT-B/16 (768-d)
    # ---------------------------------------------------------
    print("\n--- [Step 7/7] Candidate 4: BLIP-ViT-B/16 (Salesforce BLIP Vision, 768-d) ---", flush=True)
    blip_cache = CACHE_DIR / "blip_feats.pt"
    if blip_cache.exists():
        print(f"  Loading cached BLIP features from {blip_cache}...", flush=True)
        blip_feats = torch.load(blip_cache, map_location="cpu")
    else:
        blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        blip_vision = blip_model.vision_model.eval()
        blip_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        def blip_extractor_fn(pil_imgs):
            t = torch.stack([blip_transform(img) for img in pil_imgs])
            with torch.no_grad():
                out = blip_vision(t)
                f = out.pooler_output
                return F.normalize(f, dim=-1)

        blip_feats = extract_unique_image_features(all_img_paths, blip_extractor_fn, batch_size=32, desc="BLIP-ViT-B/16")
        torch.save(blip_feats, blip_cache)

    blip_train_x = build_fused_pair_features(blip_feats, train_annots, q_train_tensor)
    blip_val_x = build_fused_pair_features(blip_feats, val_annots, q_val_tensor)

    blip_oa, blip_aa, blip_oa_ci, blip_aa_ci, blip_cats = train_and_eval_head(
        blip_train_x, y_train, blip_val_x, y_val, epochs=15
    )
    print(f"BLIP-ViT-B/16: OA = {blip_oa:.2f}% [{blip_oa_ci[0]:.2f}, {blip_oa_ci[1]:.2f}] | AA = {blip_aa:.2f}% [{blip_aa_ci[0]:.2f}, {blip_aa_ci[1]:.2f}] | Gain over Blind: +{blip_oa - blind_oa:.2f}% OA, +{blip_aa - blind_aa:.2f}% AA", flush=True)
    results["blip_vit_b16"] = {
        "name": "BLIP-ViT-B/16 (Salesforce Frozen)",
        "dim": 768,
        "val_oa": round(blip_oa, 2),
        "val_oa_ci_95": [round(blip_oa_ci[0], 2), round(blip_oa_ci[1], 2)],
        "val_aa": round(blip_aa, 2),
        "val_aa_ci_95": [round(blip_aa_ci[0], 2), round(blip_aa_ci[1], 2)],
        "category_accuracies": {k: round(v, 2) for k, v in blip_cats.items()},
        "gain_over_blind_oa": round(blip_oa - blind_oa, 2),
        "gain_over_blind_aa": round(blip_aa - blind_aa, 2)
    }

    # ---------------------------------------------------------
    # Candidate 5: Trained SatQueryChangeNet Encoder (512-d)
    # ---------------------------------------------------------
    print("\n--- [Step 8/8] Candidate 5: SatQueryChangeNet v2 Encoder (512-d multi-scale difference) ---", flush=True)
    cnet_cache = CACHE_DIR / "cnet_pair_feats.pt"
    if cnet_cache.exists():
        print(f"  Loading cached ChangeNet features from {cnet_cache}...", flush=True)
        cnet_pair_feats = torch.load(cnet_cache, map_location="cpu")
    else:
        changenet = SatQueryChangeNet()
        changenet_ckpt = ML_DIR / "checkpoints" / "satquery_change_net_finetune" / "v2" / "best_change_net_finetuned.pth"
        raw_ckpt = torch.load(changenet_ckpt, map_location="cpu")
        state_dict = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        changenet.load_state_dict(state_dict)
        changenet.eval()

        cnet_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        # Extract pair difference features
        def extract_changenet_pair_feature(t1_path: Path, t2_path: Path) -> torch.Tensor:
            img1 = cnet_transform(Image.open(t1_path).convert("RGB")).unsqueeze(0)
            img2 = cnet_transform(Image.open(t2_path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                f0_1, f1_1, f2_1, f3_1 = changenet.extract_features(img1)
                f0_2, f1_2, f2_2, f3_2 = changenet.extract_features(img2)
                d3 = changenet.diff_conv3(torch.abs(f3_2 - f3_1))  # 128 ch
                d2 = torch.abs(f2_2 - f2_1)  # 128 ch

                p3_avg = F.adaptive_avg_pool2d(d3, 1).flatten(1)
                p3_max = F.adaptive_max_pool2d(d3, 1).flatten(1)
                p2_avg = F.adaptive_avg_pool2d(d2, 1).flatten(1)
                p2_max = F.adaptive_max_pool2d(d2, 1).flatten(1)
                feat = torch.cat([p3_avg, p3_max, p2_avg, p2_max], dim=-1).squeeze(0)  # 512-d
                return F.normalize(feat, dim=-1)

        # Pre-extract for all unique pairs in train and val
        all_pairs_info = {}
        for a in train_annots + val_annots:
            pid = a["pair_id"]
            if pid not in all_pairs_info:
                all_pairs_info[pid] = (ROOT_DIR / a["t1_path"], ROOT_DIR / a["t2_path"])

        cnet_pair_feats = {}
        print(f"Extracting ChangeNet difference features for {len(all_pairs_info):,} unique scene pairs...", flush=True)
        for p_idx, (pid, (t1_p, t2_p)) in enumerate(all_pairs_info.items(), 1):
            cnet_pair_feats[pid] = extract_changenet_pair_feature(t1_p, t2_p)
            if p_idx % 200 == 0 or p_idx == len(all_pairs_info):
                print(f"  Processed {p_idx}/{len(all_pairs_info)} scene pairs", flush=True)
        torch.save(cnet_pair_feats, cnet_cache)

    # Build fused ChangeNet + Q features
    cnet_train_x = torch.stack([
        torch.cat([cnet_pair_feats[a["pair_id"]], q_train_tensor[i]], dim=-1)
        for i, a in enumerate(train_annots)
    ])
    cnet_val_x = torch.stack([
        torch.cat([cnet_pair_feats[a["pair_id"]], q_val_tensor[i]], dim=-1)
        for i, a in enumerate(val_annots)
    ])

    cnet_oa, cnet_aa, cnet_oa_ci, cnet_aa_ci, cnet_cats = train_and_eval_head(
        cnet_train_x, y_train, cnet_val_x, y_val, epochs=15
    )
    print(f"SatQueryChangeNet: OA = {cnet_oa:.2f}% [{cnet_oa_ci[0]:.2f}, {cnet_oa_ci[1]:.2f}] | AA = {cnet_aa:.2f}% [{cnet_aa_ci[0]:.2f}, {cnet_aa_ci[1]:.2f}] | Gain over Blind: +{cnet_oa - blind_oa:.2f}% OA, +{cnet_aa - blind_aa:.2f}% AA", flush=True)
    results["satquery_changenet"] = {
        "name": "SatQueryChangeNet Encoder (Multi-Scale Difference)",
        "dim": 512,
        "val_oa": round(cnet_oa, 2),
        "val_oa_ci_95": [round(cnet_oa_ci[0], 2), round(cnet_oa_ci[1], 2)],
        "val_aa": round(cnet_aa, 2),
        "val_aa_ci_95": [round(cnet_aa_ci[0], 2), round(cnet_aa_ci[1], 2)],
        "category_accuracies": {k: round(v, 2) for k, v in cnet_cats.items()},
        "gain_over_blind_oa": round(cnet_oa - blind_oa, 2),
        "gain_over_blind_aa": round(cnet_aa - blind_aa, 2)
    }

    # Rank candidates by Validation OA and AA
    ranked_by_oa = sorted(
        [(k, v["name"], v["val_oa"], v["val_aa"], v["gain_over_blind_oa"], v["gain_over_blind_aa"]) for k, v in results.items()],
        key=lambda x: x[2],
        reverse=True
    )

    final_report = {
        "benchmark": "Phase 1 CDVQA Full-Task Multi-Modal Encoder Comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_scope": "All 8 CDVQA categories with bi-temporal images and question text as input",
        "categories_evaluated": CATEGORIES,
        "num_classes": NUM_CLASSES,
        "train_pairs_count": len(train_pairs),
        "train_questions_count": len(train_annots),
        "val_pairs_count": len(val_pairs),
        "val_questions_count": len(val_annots),
        "test_split_touched": False,
        "ci_method": "Cluster Bootstrap (1,000 resamples clustered over scene pairs)",
        "ranking_summary": [
            {
                "rank": idx + 1,
                "key": r[0],
                "model": r[1],
                "val_oa_pct": r[2],
                "val_aa_pct": r[3],
                "gain_over_blind_oa_pct": r[4],
                "gain_over_blind_aa_pct": r[5]
            }
            for idx, r in enumerate(ranked_by_oa)
        ],
        "candidates": results
    }

    out_file = REPORTS_DIR / "phase1_encoder_full_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n{'='*70}\n[COMPLETE] Phase 1 Full CDVQA Encoder Comparison Saved to {out_file.relative_to(ROOT_DIR)}\n{'='*70}", flush=True)

if __name__ == "__main__":
    main()
