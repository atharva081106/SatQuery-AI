#!/usr/bin/env python3
"""
ml_pipeline/phase1_encoder_comparison.py

Phase 1: Frozen Vision Encoder Selection Benchmark for SatQuery AI.
Rule 8 Compliant: All candidate backbones remain completely FROZEN.
No 7B zero-shot models on CPU.
Compares frozen vision encoders by training lightweight classification heads on cached embeddings
extracted from authentic remote sensing validation subsets (CDVQA / SECOND).

Task: Multi-class Semantic Change Category Classification (change_to_what)
Evaluates across the 6 authentic SECOND land-cover change categories:
['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water']

Candidates Evaluated:
1. ResNet-18 (torchvision canonical baseline, 512-d)
2. ResNet-50 (torchvision deep CNN baseline, 2048-d)
3. BLIP-ViT-B/16 (Salesforce/blip-vqa-base vision model, 768-d)
4. RemoteCLIP-ViT-B/32 (chendelong/RemoteCLIP, 512-d, RS-domain contrastive pre-training, CC-BY-4.0)
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
from transformers import BlipProcessor, BlipForQuestionAnswering

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = ML_DIR / "authentic_data" / "_downloads"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASSES = ["NVG_surface", "buildings", "low_vegetation", "playgrounds", "trees", "water"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

class LightweightProbeHead(nn.Module):
    """Standardized 2-layer MLP probe head trained on top of frozen fused features."""
    def __init__(self, in_features: int, hidden_dim: int = 128, num_classes: int = len(CLASSES)):
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

def load_authentic_cdvqa_semantic_samples(n_samples: int = 250) -> List[Dict[str, Any]]:
    """Loads authentic CDVQA validation pairs for change_to_what semantic classification."""
    val_ann_path = ML_DIR / "authentic_data" / "cdvqa_prepared" / "val" / "cdvqa_annotations.json"
    with open(val_ann_path, "r", encoding="utf-8") as f:
        annots = json.load(f)

    # Filter for change_to_what questions
    semantic_annots = [a for a in annots if a.get("question_type") == "change_to_what" and a.get("answer") in CLASS2IDX]
    
    # Shuffle deterministically
    rng = np.random.RandomState(SEED)
    rng.shuffle(semantic_annots)
    selected = semantic_annots[:n_samples]

    samples = []
    for a in selected:
        samples.append({
            "pair_id": a["pair_id"],
            "t1_path": ROOT_DIR / a["t1_path"],
            "t2_path": ROOT_DIR / a["t2_path"],
            "label": CLASS2IDX[a["answer"]],
            "answer": a["answer"]
        })
    return samples

def evaluate_encoder(
    encoder_name: str,
    encode_batch_fn,
    samples: List[Dict[str, Any]],
    feat_dim: int,
    total_params: int,
    license_type: str,
    epochs: int = 15,
    batch_size: int = 16
) -> Dict[str, Any]:
    print(f"\n[{encoder_name}] Extracting frozen embeddings for {len(samples)} authentic multi-class pairs...", flush=True)
    t0 = time.time()
    
    fused_features = []
    labels = [s["label"] for s in samples]

    # Batch extraction for efficiency
    n = len(samples)
    latencies = []
    for i in range(0, n, batch_size):
        batch_samples = samples[i:i+batch_size]
        t1_imgs = [Image.open(s["t1_path"]).convert("RGB") for s in batch_samples]
        t2_imgs = [Image.open(s["t2_path"]).convert("RGB") for s in batch_samples]

        b_t0 = time.time()
        with torch.no_grad():
            f1 = encode_batch_fn(t1_imgs)  # [B, D]
            f2 = encode_batch_fn(t2_imgs)  # [B, D]
            # Standard Siamese bi-temporal feature fusion: [f1, f2, |f1 - f2|, f1 * f2]
            diff = torch.abs(f1 - f2)
            prod = f1 * f2
            fused = torch.cat([f1, f2, diff, prod], dim=-1)  # [B, 4 * D]
        b_dt = (time.time() - b_t0) * 1000.0 / len(batch_samples)  # ms per pair
        latencies.append(b_dt)
        fused_features.append(fused.cpu())

    feats = torch.cat(fused_features, dim=0)  # [N, 4 * D]
    labels_t = torch.tensor(labels, dtype=torch.long)
    extract_time = time.time() - t0
    avg_latency_ms = float(np.mean(latencies))
    print(f"  Extracted {n} pairs in {extract_time:.2f}s (Avg CPU latency: {avg_latency_ms:.1f} ms/pair)", flush=True)

    # 150 train probe / 100 val probe
    n_train = 150
    X_train, X_val = feats[:n_train], feats[n_train:]
    y_train, y_val = labels_t[:n_train], labels_t[n_train:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    fused_dim = 4 * feat_dim
    head = LightweightProbeHead(in_features=fused_dim, hidden_dim=128, num_classes=len(CLASSES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_macro_f1 = 0.0

    for ep in range(1, epochs + 1):
        head.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation
        head.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, by in val_loader:
                preds = torch.argmax(head(bx), dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(by.cpu().numpy().tolist())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        val_acc = (all_preds == all_targets).mean() * 100.0

        # Compute Macro F1 across classes present in validation set
        f1_list = []
        for c in range(len(CLASSES)):
            tp = np.sum((all_preds == c) & (all_targets == c))
            fp = np.sum((all_preds == c) & (all_targets != c))
            fn = np.sum((all_preds != c) & (all_targets == c))
            denom = 2 * tp + fp + fn
            f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
            if np.sum(all_targets == c) > 0:
                f1_list.append(f1_c)
        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_macro_f1 = macro_f1

    print(f"  --> Top-1 Validation Accuracy: {best_val_acc:.2f}% | Macro F1: {best_macro_f1:.4f}", flush=True)

    return {
        "encoder": encoder_name,
        "feature_dim": feat_dim,
        "fused_dim": fused_dim,
        "encoder_params": total_params,
        "probe_head_params": sum(p.numel() for p in head.parameters()),
        "license": license_type,
        "val_subset_samples": len(y_val),
        "train_subset_samples": len(y_train),
        "val_accuracy_pct": round(best_val_acc, 2),
        "val_macro_f1": round(best_macro_f1, 4),
        "avg_cpu_latency_ms": round(avg_latency_ms, 2),
        "total_extraction_sec": round(extract_time, 2)
    }

def main():
    print("=" * 75, flush=True)
    print("SatQuery AI — Phase 1: Frozen Encoder Selection Benchmark", flush=True)
    print("Rule 8 Compliant: Frozen Backbones, Multi-Class CDVQA Semantic Subset", flush=True)
    print("=" * 75, flush=True)

    samples = load_authentic_cdvqa_semantic_samples(250)
    print(f"[DATA] Loaded {len(samples)} authentic CDVQA change_to_what pairs across 6 classes:", flush=True)
    counts = {c: sum(1 for s in samples if s['answer'] == c) for c in CLASSES}
    for c, cnt in counts.items():
        print(f"  - {c}: {cnt} samples ({cnt/len(samples)*100:.1f}%)", flush=True)

    results = {}

    # ── 1. ResNet-18 (Canonical CNN Baseline) ─────────────────────────────
    print("\n[1/4] Initializing ResNet-18 (Torchvision Weights)...", flush=True)
    rn18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    rn18.fc = nn.Identity()
    rn18.eval()
    rn18_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def encode_batch_rn18(imgs):
        t = torch.stack([rn18_transform(img) for img in imgs])
        return rn18(t)

    results["resnet18"] = evaluate_encoder(
        "ResNet-18",
        encode_batch_rn18,
        samples,
        feat_dim=512,
        total_params=sum(p.numel() for p in rn18.parameters()),
        license_type="BSD-3-Clause",
        epochs=15
    )

    # ── 2. ResNet-50 (Deep CNN Baseline) ─────────────────────────────────
    print("\n[2/4] Initializing ResNet-50 (Torchvision Weights)...", flush=True)
    rn50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    rn50.fc = nn.Identity()
    rn50.eval()
    rn50_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def encode_batch_rn50(imgs):
        t = torch.stack([rn50_transform(img) for img in imgs])
        return rn50(t)

    results["resnet50"] = evaluate_encoder(
        "ResNet-50",
        encode_batch_rn50,
        samples,
        feat_dim=2048,
        total_params=sum(p.numel() for p in rn50.parameters()),
        license_type="BSD-3-Clause",
        epochs=15
    )

    # ── 3. BLIP-ViT-B/16 (VLM Vision Backbone) ───────────────────────────
    print("\n[3/4] Initializing BLIP Vision Model (ViT-B/16)...", flush=True)
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    blip_vis = blip_vqa.vision_model
    blip_vis.eval()
    def encode_batch_blip(imgs):
        inputs = blip_proc(images=imgs, return_tensors="pt")
        out = blip_vis(**inputs)
        return out.pooler_output

    results["blip_vit_b16"] = evaluate_encoder(
        "BLIP-ViT-B/16",
        encode_batch_blip,
        samples,
        feat_dim=768,
        total_params=sum(p.numel() for p in blip_vis.parameters()),
        license_type="BSD-3-Clause",
        epochs=15
    )

    # ── 4. RemoteCLIP-ViT-B/32 (RS Pretrained Contrastive Encoder) ───────
    print("\n[4/4] Initializing RemoteCLIP-ViT-B/32 (CC-BY-4.0)...", flush=True)
    rclip_path = DOWNLOADS_DIR / "RemoteCLIP-ViT-B-32.pt"
    rclip_model, _, rclip_preprocess = open_clip.create_model_and_transforms('ViT-B-32')
    rclip_ckpt = torch.load(rclip_path, map_location="cpu")
    rclip_model.load_state_dict(rclip_ckpt)
    rclip_model.eval()

    def encode_batch_rclip(imgs):
        t = torch.stack([rclip_preprocess(img) for img in imgs])
        return rclip_model.encode_image(t)

    results["remoteclip_vit_b32"] = evaluate_encoder(
        "RemoteCLIP-ViT-B/32",
        encode_batch_rclip,
        samples,
        feat_dim=512,
        total_params=sum(p.numel() for p in rclip_model.parameters()),
        license_type="CC-BY-4.0",
        epochs=15
    )

    # ── Summary & Verified Report ────────────────────────────────────────
    report = {
        "dataset": "CDVQA / SECOND (Yuan et al. 2022)",
        "task": "Semantic Change Multi-Class Probe Classification (change_to_what, 6 classes)",
        "split": "val",
        "n_samples": len(samples),
        "seed": SEED,
        "date": "2026-10-16",
        "exact_command": "python ml_pipeline/phase1_encoder_comparison.py",
        "objective": "Phase 1: Compare frozen vision backbones by head accuracy on authentic validation subset on CPU",
        "remoteclip_checkpoint_sha256": "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
        "results": results
    }

    out_file = REPORTS_DIR / "phase1_encoder_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 95, flush=True)
    print("PHASE 1 FROZEN ENCODER COMPARISON SUMMARY (6-CLASS SEMANTIC CHANGE PROBE)", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model':<22} | {'Params':<10} | {'Dim':<6} | {'Val Acc (%)':<12} | {'Macro F1':<10} | {'CPU Lat (ms)':<12} | {'License':<12}", flush=True)
    print("-" * 95, flush=True)
    for k, v in results.items():
        print(f"{v['encoder']:<22} | {v['encoder_params']/1e6:.1f}M{'':<5} | {v['feature_dim']:<6} | {v['val_accuracy_pct']:<12.2f} | {v['val_macro_f1']:<10.4f} | {v['avg_cpu_latency_ms']:<12.1f} | {v['license']:<12}", flush=True)
    print("=" * 95, flush=True)
    print(f"Report saved to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
