#!/usr/bin/env python3
"""
ml_pipeline/train_change_detection_sota.py

SatQuery AI SOTA Training & Benchmarking Pipeline for Bi-Temporal Aerial Change Detection.
Architecture: SatQueryChangeNetSOTA (ST-CADM + Feature Pyramid Decoder).
Benchmark: Authentic SECOND / CDVQA Dataset (arXiv:2112.06343 / IEEE TGRS 2021).
Ground Truth Protocol: Canonical semantic change (label1 != label2).
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
REPORTS_DIR = ROOT_DIR / "reports" / "benchmark_evaluation"
CHECKPOINT_DIR = ML_DIR / "checkpoints" / "sota" / "change_net"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from change_model_sota import SatQueryChangeNetSOTA, SatQueryChangeNetResNetSOTA, SOTAChangeLoss

def derive_label2_path(label1_path: Path) -> Path:
    return Path(str(label1_path).replace(
        str(label1_path.parent),
        str(label1_path.parent).replace("label1", "label2"),
        1
    ))

def compute_binary_change_mask_np(label1_rgb_path: Path, label2_rgb_path: Path, img_size=(256, 256)) -> np.ndarray:
    l1 = np.array(Image.open(label1_rgb_path).resize(img_size, Image.NEAREST))
    l2 = np.array(Image.open(label2_rgb_path).resize(img_size, Image.NEAREST))
    if l1.ndim == 3:
        change = np.any(l1 != l2, axis=-1).astype(np.float32)
    else:
        change = (l1 != l2).astype(np.float32)
    return change

class SOTACDVQAPairDataset(Dataset):
    def __init__(self, split: str = "train", img_size=(256, 256), augment: bool = True, max_pairs: int = None):
        self.split = split
        self.img_size = img_size
        self.augment = augment

        prep_dir = ML_DIR / "authentic_data" / "cdvqa_prepared" / split
        with open(prep_dir / "cdvqa_annotations.json", "r", encoding="utf-8") as f:
            annots = json.load(f)

        seen_pairs = set()
        self.pairs = []
        for a in annots:
            pid = a["pair_id"]
            if pid not in seen_pairs:
                seen_pairs.add(pid)
                label1_path = ROOT_DIR / a["mask_path"]
                label2_path = derive_label2_path(label1_path)
                self.pairs.append({
                    "pair_id": pid,
                    "t1_path": ROOT_DIR / a["t1_path"],
                    "t2_path": ROOT_DIR / a["t2_path"],
                    "label1_path": label1_path,
                    "label2_path": label2_path,
                })
        if max_pairs is not None:
            self.pairs = self.pairs[:max_pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        t1_img = Image.open(item["t1_path"]).convert("RGB").resize(self.img_size, Image.BILINEAR)
        t2_img = Image.open(item["t2_path"]).convert("RGB").resize(self.img_size, Image.BILINEAR)

        t1_arr = np.array(t1_img, dtype=np.float32) / 255.0
        t2_arr = np.array(t2_img, dtype=np.float32) / 255.0
        mask_arr = compute_binary_change_mask_np(item["label1_path"], item["label2_path"], self.img_size)

        if self.augment and self.split == "train":
            if random.random() > 0.5:
                t1_arr = np.fliplr(t1_arr)
                t2_arr = np.fliplr(t2_arr)
                mask_arr = np.fliplr(mask_arr)
            if random.random() > 0.5:
                t1_arr = np.flipud(t1_arr)
                t2_arr = np.flipud(t2_arr)
                mask_arr = np.flipud(mask_arr)
            k = random.randint(0, 3)
            if k > 0:
                t1_arr = np.rot90(t1_arr, k)
                t2_arr = np.rot90(t2_arr, k)
                mask_arr = np.rot90(mask_arr, k)

            t1_arr = np.ascontiguousarray(t1_arr)
            t2_arr = np.ascontiguousarray(t2_arr)
            mask_arr = np.ascontiguousarray(mask_arr)

        t1_t = torch.from_numpy(t1_arr).permute(2, 0, 1)
        t2_t = torch.from_numpy(t2_arr).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        return {"pair_id": item["pair_id"], "t1": t1_t, "t2": t2_t, "mask": mask_t}

def compute_mask_metrics(p: np.ndarray, m: np.ndarray) -> dict:
    tp = int(np.logical_and(p, m).sum())
    fp = int(np.logical_and(p, ~m).sum())
    fn = int(np.logical_and(~p, m).sum())
    tn = int(np.logical_and(~p, ~m).sum())

    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else (1.0 if (tp + fp + fn == 0) else 0.0)
    dice = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else (1.0 if union == 0 else 0.0)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {"iou": iou, "f1": dice, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def tune_change_threshold(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            masks = batch["mask"].numpy()

            logits = model(t1, t2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(masks)

    p = np.concatenate(all_probs).ravel()
    t = np.concatenate(all_targets).ravel()

    best_iou, best_thr = -1.0, 0.45
    for thr in np.arange(0.30, 0.65, 0.05):
        pred = (p > thr).astype(np.float32)
        inter = (pred * t).sum()
        union = (pred + t - pred * t).sum()
        iou = inter / (union + 1e-6)
        if iou > best_iou:
            best_iou, best_thr = iou, float(thr)

    print(f"[Tuning] Optimal Change Threshold: {best_thr:.2f} (Val IoU: {best_iou*100:.2f}%)")
    return best_thr

def evaluate_test_set_change_tta(model: nn.Module, test_loader: DataLoader, threshold: float, device: str) -> dict:
    model.eval()
    ious = []
    f1s = []
    precs = []
    recs = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for batch in test_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            masks = batch["mask"].numpy()

            # 4-Fold Dihedral Test-Time Augmentation
            p0 = torch.sigmoid(model(t1, t2))
            ph = torch.flip(torch.sigmoid(model(torch.flip(t1, [-1]), torch.flip(t2, [-1]))), [-1])
            pv = torch.flip(torch.sigmoid(model(torch.flip(t1, [-2]), torch.flip(t2, [-2]))), [-2])
            phv = torch.flip(torch.sigmoid(model(torch.flip(t1, [-2, -1]), torch.flip(t2, [-2, -1]))), [-2, -1])
            probs = ((p0 + ph + pv + phv) / 4.0).cpu().numpy()

            preds = (probs > threshold)

            for b in range(preds.shape[0]):
                p_bin = preds[b, 0]
                m_bin = masks[b, 0].astype(bool)
                met = compute_mask_metrics(p_bin, m_bin)
                ious.append(met["iou"])
                f1s.append(met["f1"])
                precs.append(met["precision"])
                recs.append(met["recall"])
                total_tp += met["tp"]
                total_fp += met["fp"]
                total_fn += met["fn"]
                total_tn += met["tn"]

    macro_iou = float(np.mean(ious)) * 100.0
    macro_f1 = float(np.mean(f1s))
    macro_prec = float(np.mean(precs)) * 100.0
    macro_rec = float(np.mean(recs)) * 100.0

    total_pixels = total_tp + total_fp + total_fn + total_tn
    pixel_acc = float((total_tp + total_tn) / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # Macro 2-class mIoU (change class + background no-change class)
    bg_union = total_fp + total_fn + total_tn
    bg_iou = float(total_tn / bg_union) if bg_union > 0 else 0.0
    macro_2class_miou = (macro_iou + bg_iou * 100.0) / 2.0

    return {
        "macro_change_iou_pct": round(macro_iou, 2),
        "macro_2class_miou_pct": round(macro_2class_miou, 2),
        "macro_dice_f1": round(macro_f1, 4),
        "macro_precision_pct": round(macro_prec, 2),
        "macro_recall_pct": round(macro_rec, 2),
        "pixel_accuracy_pct": round(pixel_acc, 2),
        "total_pairs_evaluated": len(ious),
    }

def run_training_and_eval(model_type: str = "resnet", epochs: int = 3, batch_size: int = 8, lr: float = 2e-4, max_train_pairs: int = 200, max_eval_pairs: int = 100):
    arch_name = "SatQueryChangeNetResNetSOTA (Pretrained ResNet-18 + ST-CADM + FPN)" if model_type == "resnet" else "SatQueryChangeNetSOTA (ST-CADM + FPN Decoder)"
    print("=" * 70)
    print("  SatQuery AI: SOTA Bi-Temporal Aerial Change Detection Training")
    print(f"  Architecture: {arch_name}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)

    train_ds = SOTACDVQAPairDataset(split="train", img_size=(256, 256), augment=True, max_pairs=max_train_pairs)
    val_ds = SOTACDVQAPairDataset(split="val", img_size=(256, 256), augment=False, max_pairs=max_eval_pairs)
    test_ds = SOTACDVQAPairDataset(split="test", img_size=(256, 256), augment=False, max_pairs=max_eval_pairs)
    print(f"[Dataset] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_type == "resnet":
        model = SatQueryChangeNetResNetSOTA(pretrained=True).to(device)
    else:
        model = SatQueryChangeNetSOTA(in_channels=3).to(device)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch_name} ({p_count:,} parameters)")

    criterion = SOTAChangeLoss(alpha=0.75, gamma=2.0, smooth=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_b = 0

        for batch in train_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = criterion(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            n_b += 1

        scheduler.step()
        train_loss /= max(1, n_b)

        model.eval()
        val_loss = 0.0
        v_b = 0
        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                mask = batch["mask"].to(device)
                loss = criterion(model(t1, t2), mask)
                val_loss += loss.item()
                v_b += 1
        val_loss /= max(1, v_b)

        sec = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ({sec:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print("\n[Calibration] Tuning optimal change detection threshold on validation split...")
    threshold = tune_change_threshold(model, val_loader, device)

    print("\n[Evaluation] Evaluating on authentic SECOND / CDVQA held-out test split with 4-fold dihedral TTA...")
    test_metrics = evaluate_test_set_change_tta(model, test_loader, threshold, device)

    print("\n" + "=" * 70)
    print(f"  SatQuery AI SOTA Aerial Change Detection Results ({arch_name}):")
    print(f"  Macro Change IoU:    {test_metrics['macro_change_iou_pct']}%")
    print(f"  Macro 2-Class mIoU:  {test_metrics['macro_2class_miou_pct']}%")
    print(f"  Macro Dice F1:       {test_metrics['macro_dice_f1']}")
    print(f"  Precision / Recall:  {test_metrics['macro_precision_pct']}% / {test_metrics['macro_recall_pct']}%")
    print(f"  Pixel Accuracy:      {test_metrics['pixel_accuracy_pct']}%")
    print("=" * 70)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_filename = "change_net_resnet_sota.pth" if model_type == "resnet" else "change_net_sota.pth"
    ckpt_path = CHECKPOINT_DIR / ckpt_filename
    torch.save({
        "model_name": arch_name,
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "parameters": p_count,
        "optimal_threshold": threshold,
        "test_metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ckpt_path)
    print(f"\n[Checkpoint] Saved SOTA checkpoint to: {ckpt_path}")

    # Also save as main SOTA checkpoint if it's the resnet or if better
    if model_type == "resnet":
        main_ckpt = CHECKPOINT_DIR / "change_net_sota.pth"
        torch.save({
            "model_name": arch_name,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "parameters": p_count,
            "optimal_threshold": threshold,
            "test_metrics": test_metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, main_ckpt)
        print(f"[Checkpoint] Also updated default SOTA checkpoint: {main_ckpt}")

    report_data = {
        "benchmark": "CDVQA / SECOND SOTA Evaluation",
        "model": arch_name,
        "architecture": "Spatial-Temporal Cross-Attention Difference Network with Pretrained Backbone",
        "parameters": p_count,
        "evaluation_protocol": "4-Fold Dihedral Test-Time Augmentation (TTA) + Calibrated Threshold",
        "threshold": threshold,
        "metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_filename = "change_net_resnet_sota_metrics.json" if model_type == "resnet" else "change_net_sota_metrics.json"
    out_json = REPORTS_DIR / report_filename
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[Report] Saved benchmark report to: {out_json}")

    # Update main sota metrics json
    if model_type == "resnet":
        with open(REPORTS_DIR / "change_net_sota_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "scratch"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_pairs", type=int, default=250)
    parser.add_argument("--eval_pairs", type=int, default=100)
    args = parser.parse_args()

    run_training_and_eval(model_type=args.model_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                          max_train_pairs=args.train_pairs, max_eval_pairs=args.eval_pairs)

