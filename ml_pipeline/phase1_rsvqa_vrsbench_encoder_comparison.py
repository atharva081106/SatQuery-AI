#!/usr/bin/env python3
"""
ml_pipeline/phase1_rsvqa_vrsbench_encoder_comparison.py

Evaluates Multi-Modal Vision Encoders on RSVQA-LR and VRSBench Validation:
Candidates compared:
1. Blind Question-Only Baseline (Language-only prior)
2. ResNet-18 (ImageNet frozen, 512-d)
3. ResNet-50 (ImageNet frozen, 2048-d)
4. RemoteCLIP-ViT-B/32 (RS-domain frozen, 512-d)
5. BLIP-ViT-B/16 (Salesforce vision frozen, 768-d)

Reports Overall Accuracy (OA), Average Accuracy (AA), and Gain over Blind Baseline.
Outputs reports/benchmark_evaluation/rsvqa_vrsbench_encoder_comparison.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import open_clip
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark_evaluation"
OUT_REPORT = REPORTS_DIR / "rsvqa_vrsbench_encoder_comparison.json"

torch.set_num_threads(8)
SEED = 42

class VQAProbeHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256, num_classes: int = 10):
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

def train_and_eval_probe(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_v: torch.Tensor,
    y_v: torch.Tensor,
    categories_val: List[str],
    num_classes: int,
    epochs: int = 15,
    seed: int = SEED
) -> Tuple[float, float, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = VQAProbeHead(in_features=x_tr.shape[1], hidden_dim=256, num_classes=num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    ds = TensorDataset(x_tr, y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

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

    y_val_np = y_v.numpy()
    oa = float(np.mean(val_preds == y_val_np)) * 100.0

    cats_unique = sorted(list(set(categories_val)))
    cat_accs = {}
    for c in cats_unique:
        mask = np.array([cat == c for cat in categories_val])
        if np.sum(mask) > 0:
            cat_accs[c] = float(np.mean(val_preds[mask] == y_val_np[mask])) * 100.0
        else:
            cat_accs[c] = 0.0

    aa = float(np.mean(list(cat_accs.values()))) if cat_accs else 0.0
    return round(oa, 2), round(aa, 2), {k: round(v, 2) for k, v in cat_accs.items()}

def load_rsvqa_data() -> Tuple[List[Dict], List[Dict], List[str]]:
    rsvqa_dir = ML_DIR / "dataset_cache" / "rsvqa"
    train_q = json.load(open(rsvqa_dir / "train" / "questions.json"))
    train_a = json.load(open(rsvqa_dir / "train" / "answers.json"))
    train_imgs = {img["id"]: rsvqa_dir / "train" / "Images" / img["name"] for img in json.load(open(rsvqa_dir / "train" / "images.json"))}

    val_q = json.load(open(rsvqa_dir / "val" / "questions.json"))
    val_a = json.load(open(rsvqa_dir / "val" / "answers.json"))
    val_imgs = {img["id"]: rsvqa_dir / "val" / "Images" / img["name"] for img in json.load(open(rsvqa_dir / "val" / "images.json"))}

    train_data = []
    for q, a in zip(train_q, train_a):
        train_data.append({
            "question": q["question"],
            "answer": str(a["answer"]).strip().lower(),
            "type": q.get("type", "general"),
            "img_path": train_imgs[q["img_id"]]
        })

    val_data = []
    for q, a in zip(val_q, val_a):
        val_data.append({
            "question": q["question"],
            "answer": str(a["answer"]).strip().lower(),
            "type": q.get("type", "general"),
            "img_path": val_imgs[q["img_id"]]
        })

    all_answers = sorted(list(set([d["answer"] for d in train_data + val_data])))
    return train_data, val_data, all_answers

def load_vrsbench_data() -> Tuple[List[Dict], List[Dict], List[str]]:
    vrs_dir = ML_DIR / "dataset_cache" / "vrsbench"
    train_raw = json.load(open(vrs_dir / "train" / "vrsbench_vqa.json"))
    val_raw = json.load(open(vrs_dir / "val" / "vrsbench_vqa.json"))

    train_data = []
    for item in train_raw:
        ans = item.get("answers", ["no"])[0].strip().lower()
        train_data.append({
            "question": item["question"],
            "answer": ans,
            "type": item.get("question_type", "general"),
            "img_path": vrs_dir / item["image_path"].replace("\\", "/")
        })

    val_data = []
    for item in val_raw:
        ans = item.get("answers", ["no"])[0].strip().lower()
        val_data.append({
            "question": item["question"],
            "answer": ans,
            "type": item.get("question_type", "general"),
            "img_path": vrs_dir / item["image_path"].replace("\\", "/")
        })

    all_answers = sorted(list(set([d["answer"] for d in train_data + val_data])))
    return train_data, val_data, all_answers

def extract_features_for_dataset(data: List[Dict], tokenizer, rc_model, rn18, rn50, blip_vision, transform):
    # Unique questions
    unique_q = list(set(d["question"] for d in data))
    q_feats = {}
    with torch.no_grad():
        for i in range(0, len(unique_q), 32):
            batch_q = unique_q[i:i + 32]
            tokens = tokenizer(batch_q)
            emb = rc_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            for q_str, e in zip(batch_q, emb):
                q_feats[q_str] = e.cpu()

    # Unique images
    unique_imgs = list(set(d["img_path"] for d in data))
    rn18_feats = {}
    rn50_feats = {}
    rc_img_feats = {}
    blip_img_feats = {}

    with torch.no_grad():
        for p in unique_imgs:
            im = Image.open(p).convert("RGB")
            t_im = transform(im).unsqueeze(0)

            # ResNet-18
            f18 = rn18(t_im).squeeze()
            rn18_feats[str(p)] = F.normalize(f18, dim=-1).cpu()

            # ResNet-50
            f50 = rn50(t_im).squeeze()
            rn50_feats[str(p)] = F.normalize(f50, dim=-1).cpu()

            # RemoteCLIP
            frc = rc_model.encode_image(t_im).squeeze()
            rc_img_feats[str(p)] = F.normalize(frc, dim=-1).cpu()

            # BLIP Vision
            b_out = blip_vision(t_im)[0]
            f_blip = b_out[:, 0, :].squeeze()  # CLS token
            blip_img_feats[str(p)] = F.normalize(f_blip, dim=-1).cpu()

    return q_feats, rn18_feats, rn50_feats, rc_img_feats, blip_img_feats

def run_benchmark_on_dataset(
    dataset_name: str,
    train_data: List[Dict],
    val_data: List[Dict],
    vocab: List[str],
    tokenizer, rc_model, rn18, rn50, blip_vision, transform
) -> Dict[str, Any]:
    print(f"\n--- Running Benchmark on {dataset_name} ({len(train_data)} train, {len(val_data)} val) ---")
    ans2idx = {a: i for i, a in enumerate(vocab)}
    num_classes = len(vocab)

    all_data = train_data + val_data
    q_feats, rn18_f, rn50_f, rc_f, blip_f = extract_features_for_dataset(
        all_data, tokenizer, rc_model, rn18, rn50, blip_vision, transform
    )

    y_train = torch.tensor([ans2idx[d["answer"]] for d in train_data], dtype=torch.long)
    y_val = torch.tensor([ans2idx[d["answer"]] for d in val_data], dtype=torch.long)
    val_cats = [d["type"] for d in val_data]

    q_train = torch.stack([q_feats[d["question"]] for d in train_data])
    q_val = torch.stack([q_feats[d["question"]] for d in val_data])

    results = {}

    # 1. Blind Question-Only Baseline
    b_oa, b_aa, b_cats = train_and_eval_probe(q_train, y_train, q_val, y_val, val_cats, num_classes)
    results["blind_question_only"] = {
        "model": "Blind Question-Only Baseline",
        "oa_pct": b_oa,
        "aa_pct": b_aa,
        "gain_over_blind_oa_pct": 0.0,
        "gain_over_blind_aa_pct": 0.0,
        "per_category": b_cats
    }
    print(f"  Blind Question-Only: OA = {b_oa:.2f}%, AA = {b_aa:.2f}%")

    # 2. ResNet-18
    rn18_tr = torch.stack([torch.cat([rn18_f[str(d["img_path"])], q_train[i]]) for i, d in enumerate(train_data)])
    rn18_v = torch.stack([torch.cat([rn18_f[str(d["img_path"])], q_val[i]]) for i, d in enumerate(val_data)])
    r18_oa, r18_aa, r18_cats = train_and_eval_probe(rn18_tr, y_train, rn18_v, y_val, val_cats, num_classes)
    results["resnet18"] = {
        "model": "ResNet-18 (ImageNet frozen)",
        "oa_pct": r18_oa,
        "aa_pct": r18_aa,
        "gain_over_blind_oa_pct": round(r18_oa - b_oa, 2),
        "gain_over_blind_aa_pct": round(r18_aa - b_aa, 2),
        "per_category": r18_cats
    }
    print(f"  ResNet-18: OA = {r18_oa:.2f}% (Gain: {r18_oa - b_oa:+.2f}%), AA = {r18_aa:.2f}% (Gain: {r18_aa - b_aa:+.2f}%)")

    # 3. ResNet-50
    rn50_tr = torch.stack([torch.cat([rn50_f[str(d["img_path"])], q_train[i]]) for i, d in enumerate(train_data)])
    rn50_v = torch.stack([torch.cat([rn50_f[str(d["img_path"])], q_val[i]]) for i, d in enumerate(val_data)])
    r50_oa, r50_aa, r50_cats = train_and_eval_probe(rn50_tr, y_train, rn50_v, y_val, val_cats, num_classes)
    results["resnet50"] = {
        "model": "ResNet-50 (ImageNet frozen)",
        "oa_pct": r50_oa,
        "aa_pct": r50_aa,
        "gain_over_blind_oa_pct": round(r50_oa - b_oa, 2),
        "gain_over_blind_aa_pct": round(r50_aa - b_aa, 2),
        "per_category": r50_cats
    }
    print(f"  ResNet-50: OA = {r50_oa:.2f}% (Gain: {r50_oa - b_oa:+.2f}%), AA = {r50_aa:.2f}% (Gain: {r50_aa - b_aa:+.2f}%)")

    # 4. RemoteCLIP-ViT-B/32
    rc_tr = torch.stack([torch.cat([rc_f[str(d["img_path"])], q_train[i]]) for i, d in enumerate(train_data)])
    rc_v = torch.stack([torch.cat([rc_f[str(d["img_path"])], q_val[i]]) for i, d in enumerate(val_data)])
    rc_oa, rc_aa, rc_cats = train_and_eval_probe(rc_tr, y_train, rc_v, y_val, val_cats, num_classes)
    results["remoteclip_vit_b32"] = {
        "model": "RemoteCLIP-ViT-B/32 (RS Domain frozen)",
        "oa_pct": rc_oa,
        "aa_pct": rc_aa,
        "gain_over_blind_oa_pct": round(rc_oa - b_oa, 2),
        "gain_over_blind_aa_pct": round(rc_aa - b_aa, 2),
        "per_category": rc_cats
    }
    print(f"  RemoteCLIP-ViT-B/32: OA = {rc_oa:.2f}% (Gain: {rc_oa - b_oa:+.2f}%), AA = {rc_aa:.2f}% (Gain: {rc_aa - b_aa:+.2f}%)")

    # 5. BLIP-ViT-B/16
    blip_tr = torch.stack([torch.cat([blip_f[str(d["img_path"])], q_train[i]]) for i, d in enumerate(train_data)])
    blip_v = torch.stack([torch.cat([blip_f[str(d["img_path"])], q_val[i]]) for i, d in enumerate(val_data)])
    bl_oa, bl_aa, bl_cats = train_and_eval_probe(blip_tr, y_train, blip_v, y_val, val_cats, num_classes)
    results["blip_vit_b16"] = {
        "model": "BLIP-ViT-B/16 (Salesforce frozen)",
        "oa_pct": bl_oa,
        "aa_pct": bl_aa,
        "gain_over_blind_oa_pct": round(bl_oa - b_oa, 2),
        "gain_over_blind_aa_pct": round(bl_aa - b_aa, 2),
        "per_category": bl_cats
    }
    print(f"  BLIP-ViT-B/16: OA = {bl_oa:.2f}% (Gain: {bl_oa - b_oa:+.2f}%), AA = {bl_aa:.2f}% (Gain: {bl_aa - b_aa:+.2f}%)")

    return results

def main():
    print("=" * 70)
    print("Encoder Comparison on RSVQA-LR and VRSBench Validation with Blind Baselines")
    print("=" * 70)

    # 1. Load Encoders
    print("Loading models...")
    rc_ckpt = ML_DIR / "authentic_data" / "_downloads" / "RemoteCLIP-ViT-B-32.pt"
    rc_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
    rc_model.load_state_dict(torch.load(rc_ckpt, map_location="cpu"))
    rc_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    rn18_full = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    rn18 = nn.Sequential(*list(rn18_full.children())[:-1], nn.Flatten()).eval()

    rn50_full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    rn50 = nn.Sequential(*list(rn50_full.children())[:-1], nn.Flatten()).eval()

    from transformers import BlipForQuestionAnswering
    blip_full = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_vision = blip_full.vision_model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Benchmark RSVQA-LR
    rsvqa_tr, rsvqa_val, rsvqa_vocab = load_rsvqa_data()
    rsvqa_results = run_benchmark_on_dataset(
        "RSVQA-LR", rsvqa_tr, rsvqa_val, rsvqa_vocab,
        tokenizer, rc_model, rn18, rn50, blip_vision, transform
    )

    # 3. Benchmark VRSBench
    vrs_tr, vrs_val, vrs_vocab = load_vrsbench_data()
    vrs_results = run_benchmark_on_dataset(
        "VRSBench", vrs_tr, vrs_val, vrs_vocab,
        tokenizer, rc_model, rn18, rn50, blip_vision, transform
    )

    final_payload = {
        "benchmark": "Phase 1 Cross-Dataset Encoder Comparison (RSVQA-LR and VRSBench)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Evaluates candidate multi-modal vision encoders alongside blind question-only baselines on RSVQA-LR and VRSBench validation sets.",
        "datasets": {
            "rsvqa_lr": {
                "name": "RSVQA-LR (Sentinel-2 Low Resolution)",
                "train_samples": len(rsvqa_tr),
                "val_samples": len(rsvqa_val),
                "vocab_size": len(rsvqa_vocab),
                "candidates": rsvqa_results
            },
            "vrsbench": {
                "name": "VRSBench (VHR Aerial VQA)",
                "train_samples": len(vrs_tr),
                "val_samples": len(vrs_val),
                "vocab_size": len(vrs_vocab),
                "candidates": vrs_results
            }
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\nSaved cross-dataset encoder comparison report to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
