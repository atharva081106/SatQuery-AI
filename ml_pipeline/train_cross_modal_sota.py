#!/usr/bin/env python3
"""
ml_pipeline/train_cross_modal_sota.py

SatQuery AI SOTA Training Pipeline for Joint Optical-SAR Remote Sensing Analysis.
Architecture: SatQueryOpticalSARNetSOTA (Bidirectional Cross-Attention + ASPP + Lee Filter).
Benchmark: BigEarthNet-MM / BEN-GE-800 (Zenodo DOI 10.5281/zenodo.12941231).
Authentic Sentinel-1 SAR + Sentinel-2 Optical with ESA WorldCover 10m Ground Truth.
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
CHECKPOINT_DIR = ML_DIR / "checkpoints" / "sota" / "optical_sar"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from cross_modal_model_sota import SatQueryOpticalSARNetSOTA, SatQueryOpticalSARResNetSOTA, FocalTverskyLoss
from data.bigearthnet_loader import BigEarthNetDataset
from train_cross_modal import generate_spatial_targets

def build_target_hwc_to_chw(item: dict, s2_arr: np.ndarray, s1_arr: np.ndarray, img_size: int = 128) -> np.ndarray:
    if "target_mask" in item:
        raw = item["target_mask"].numpy() if hasattr(item["target_mask"], "numpy") else item["target_mask"]
        if raw.ndim == 3 and raw.shape[2] == 3:
            tgt_hwc = raw
        elif raw.ndim == 3 and raw.shape[0] == 3:
            return raw.astype(np.float32)
        else:
            tgt_hwc = raw
        channels = []
        for c in range(3):
            ch = Image.fromarray(tgt_hwc[..., c]).resize((img_size, img_size), Image.NEAREST)
            channels.append(np.array(ch, dtype=np.float32))
        return np.stack(channels, axis=0)
    else:
        tgt = generate_spatial_targets(s2_arr, s1_arr, item["clc_labels"])
        return tgt

def apply_augmentation(opt_t: torch.Tensor, sar_t: torch.Tensor, tgt_t: torch.Tensor, augment: bool) -> tuple:
    if not augment:
        return opt_t, sar_t, tgt_t

    if random.random() > 0.5:
        opt_t = torch.flip(opt_t, dims=[2])
        sar_t = torch.flip(sar_t, dims=[2])
        tgt_t = torch.flip(tgt_t, dims=[2])

    if random.random() > 0.5:
        opt_t = torch.flip(opt_t, dims=[1])
        sar_t = torch.flip(sar_t, dims=[1])
        tgt_t = torch.flip(tgt_t, dims=[1])

    k = random.randint(0, 3)
    if k > 0:
        opt_t = torch.rot90(opt_t, k, dims=[1, 2])
        sar_t = torch.rot90(sar_t, k, dims=[1, 2])
        tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])

    if random.random() > 0.5:
        speckle = 1.0 + 0.05 * torch.randn_like(sar_t)
        sar_t = sar_t * speckle

    return opt_t, sar_t, tgt_t

def collate_fn_sota(batch: list, img_size: int = 128, augment: bool = False) -> dict:
    opt_list, sar_list, tgt_list = [], [], []

    for item in batch:
        s2_img = item["s2_image"].resize((img_size, img_size), Image.Resampling.BILINEAR)
        s2_arr = np.array(s2_img, dtype=np.float32)

        s1_raw = item["s1_sar"].numpy() if hasattr(item["s1_sar"], "numpy") else item["s1_sar"]
        if s1_raw.ndim == 3:
            s1_img = Image.fromarray(s1_raw[..., 0]).resize((img_size, img_size), Image.Resampling.BILINEAR)
        else:
            s1_img = Image.fromarray(s1_raw).resize((img_size, img_size), Image.Resampling.BILINEAR)
        s1_arr = np.array(s1_img, dtype=np.float32)

        opt_t = torch.from_numpy(s2_arr / 255.0).permute(2, 0, 1)
        sar_t = torch.from_numpy(s1_arr).unsqueeze(0)
        tgt_arr = build_target_hwc_to_chw(item, s2_arr, s1_arr, img_size)
        tgt_t = torch.from_numpy(tgt_arr)

        opt_t, sar_t, tgt_t = apply_augmentation(opt_t, sar_t, tgt_t, augment=augment)

        opt_list.append(opt_t)
        sar_list.append(sar_t)
        tgt_list.append(tgt_t)

    return {
        "optical": torch.stack(opt_list),
        "sar": torch.stack(sar_list),
        "targets": torch.stack(tgt_list),
    }

class ClassWeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weights: torch.Tensor, smooth: float = 1.0, dice_weight: float = 0.5):
        super().__init__()
        self.register_buffer("pos_weights", pos_weights.view(1, -1, 1, 1))
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weights)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = (1.0 - dice).mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    tp = float(intersection)
    fp = float(np.logical_and(preds, np.logical_not(targets)).sum())
    fn = float(np.logical_and(np.logical_not(preds), targets).sum())

    if union == 0:
        return {"iou": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    iou = float(tp / union) if union > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}

def tune_thresholds_sota(model: nn.Module, val_loader: DataLoader, device: str) -> list:
    model.eval()
    all_probs = [[], [], []]
    all_targets = [[], [], []]

    with torch.no_grad():
        for batch in val_loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            tgt = batch["targets"].cpu().numpy()

            logits = model(opt, sar)
            probs = torch.sigmoid(logits).cpu().numpy()

            for c in range(3):
                all_probs[c].append(probs[:, c])
                all_targets[c].append(tgt[:, c])

    best_thresholds = []
    class_names = ["built_up", "water", "vegetation"]
    for c in range(3):
        p = np.concatenate(all_probs[c]).ravel()
        t = np.concatenate(all_targets[c]).ravel()
        best_iou, best_thr = -1.0, 0.5
        for thr in np.arange(0.2, 0.85, 0.05):
            pred = (p > thr).astype(np.float32)
            inter = (pred * t).sum()
            union = (pred + t - pred * t).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_thr = iou, float(thr)
        best_thresholds.append(best_thr)
        print(f"  [Tuning] {class_names[c]}: best_thr={best_thr:.2f}, val_iou={best_iou*100:.2f}%")

    return best_thresholds

def evaluate_test_set_tta(model: nn.Module, test_ds, thresholds: list, img_size: int, device: str) -> dict:
    model.eval()
    class_metrics = {"built_up": [], "water": [], "vegetation": []}

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        s2_img = item["s2_image"].resize((img_size, img_size), Image.Resampling.BILINEAR)
        s2_arr = np.array(s2_img, dtype=np.float32)

        s1_raw = item["s1_sar"].numpy() if isinstance(item["s1_sar"], torch.Tensor) else item["s1_sar"]
        if s1_raw.ndim == 3:
            s1_ch0 = Image.fromarray(s1_raw[..., 0]).resize((img_size, img_size), Image.Resampling.BILINEAR)
        else:
            s1_ch0 = Image.fromarray(s1_raw).resize((img_size, img_size), Image.Resampling.BILINEAR)
        s1_arr = np.array(s1_ch0, dtype=np.float32)

        opt_t = torch.from_numpy(s2_arr / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        sar_t = torch.from_numpy(s1_arr).unsqueeze(0).unsqueeze(0).to(device)
        tgt_arr = build_target_hwc_to_chw(item, s2_arr, s1_arr, img_size)

        with torch.no_grad():
            # 4-Fold Dihedral Test-Time Augmentation
            p0 = torch.sigmoid(model(opt_t, sar_t))
            ph = torch.flip(torch.sigmoid(model(torch.flip(opt_t, [-1]), torch.flip(sar_t, [-1]))), [-1])
            pv = torch.flip(torch.sigmoid(model(torch.flip(opt_t, [-2]), torch.flip(sar_t, [-2]))), [-2])
            phv = torch.flip(torch.sigmoid(model(torch.flip(opt_t, [-2, -1]), torch.flip(sar_t, [-2, -1]))), [-2, -1])
            probs = ((p0 + ph + pv + phv) / 4.0).squeeze().cpu().numpy()

        preds = np.stack([(probs[c] > thresholds[c]) for c in range(3)], axis=0)
        targets = (tgt_arr > 0.50)

        class_metrics["built_up"].append(compute_metrics(preds[0], targets[0]))
        class_metrics["water"].append(compute_metrics(preds[1], targets[1]))
        class_metrics["vegetation"].append(compute_metrics(preds[2], targets[2]))

    summary = {}
    ious = []
    for c_name in ["built_up", "water", "vegetation"]:
        c_ious = [m["iou"] for m in class_metrics[c_name]]
        c_f1s = [m["f1"] for m in class_metrics[c_name]]
        c_precs = [m["precision"] for m in class_metrics[c_name]]
        c_recs = [m["recall"] for m in class_metrics[c_name]]
        mean_iou = float(np.mean(c_ious))
        ious.append(mean_iou)
        summary[c_name] = {
            "mean_iou_pct": round(mean_iou * 100.0, 2),
            "dice_f1": round(float(np.mean(c_f1s)), 4),
            "precision": round(float(np.mean(c_precs)), 4),
            "recall": round(float(np.mean(c_recs)), 4),
        }

    summary["overall_mean_iou_pct"] = round(float(np.mean(ious)) * 100.0, 2)
    return summary

def run_training_and_eval(model_type: str = "resnet", epochs: int = 3, batch_size: int = 8, lr: float = 3e-4, img_size: int = 128):
    arch_name = "SatQueryOpticalSARResNetSOTA (Pretrained ResNet-18 + BCA + ASPP + Lee Filter)" if model_type == "resnet" else "SatQueryOpticalSARNetSOTA (BCA + ASPP + Lee Filter)"
    print("=" * 70)
    print("  SatQuery AI: SOTA Optical-SAR Cross-Modal Model Training & Benchmarking")
    print(f"  Architecture: {arch_name}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)

    train_ds = BigEarthNetDataset(split="train")
    val_ds = BigEarthNetDataset(split="val")
    test_ds = BigEarthNetDataset(split="test")
    print(f"[Dataset] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn_sota(b, img_size=img_size, augment=True)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn_sota(b, img_size=img_size, augment=False)
    )

    if model_type == "resnet":
        model = SatQueryOpticalSARResNetSOTA(sar_channels=1, num_classes=3, pretrained=True).to(device)
    else:
        model = SatQueryOpticalSARNetSOTA(sar_channels=1, num_classes=3).to(device)
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch_name} ({p_count:,} parameters)")

    pos_weights = torch.tensor([14.0, 6.0, 1.5], dtype=torch.float32)
    criterion = ClassWeightedBCEDiceLoss(pos_weights=pos_weights, dice_weight=0.5).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            opt = batch["optical"].to(device)
            sar = batch["sar"].to(device)
            tgt = batch["targets"].to(device)

            optimizer.zero_grad()
            logits = model(opt, sar)
            loss = criterion(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(1, n_batches)

        # Validation Loss
        model.eval()
        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["optical"].to(device)
                sar = batch["sar"].to(device)
                tgt = batch["targets"].to(device)
                loss = criterion(model(opt, sar), tgt)
                val_loss += loss.item()
                v_batches += 1
        val_loss /= max(1, v_batches)

        sec = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ({sec:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Threshold calibration
    print("\n[Calibration] Tuning optimal per-class probability thresholds on validation set...")
    thresholds = tune_thresholds_sota(model, val_loader, device)

    # Evaluation on Held-Out Test Set with 4-fold Dihedral TTA
    print("\n[Evaluation] Evaluating on authentic BigEarthNet-MM held-out test split with 4-fold dihedral TTA...")
    test_metrics = evaluate_test_set_tta(model, test_ds, thresholds, img_size, device)
    print("\n" + "=" * 70)
    print(f"  SatQuery AI SOTA Optical-SAR Benchmark Results ({arch_name}):")
    print(f"  Overall Macro mIoU: {test_metrics['overall_mean_iou_pct']}%")
    print(f"  Built-up IoU:       {test_metrics['built_up']['mean_iou_pct']}% (F1: {test_metrics['built_up']['dice_f1']})")
    print(f"  Water IoU:          {test_metrics['water']['mean_iou_pct']}% (F1: {test_metrics['water']['dice_f1']})")
    print(f"  Vegetation IoU:     {test_metrics['vegetation']['mean_iou_pct']}% (F1: {test_metrics['vegetation']['dice_f1']})")
    print("=" * 70)

    # Save Checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = "optical_sar_net_resnet_sota.pth" if model_type == "resnet" else "optical_sar_net_sota.pth"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save({
        "model_name": arch_name,
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "parameters": p_count,
        "optimal_thresholds": thresholds,
        "img_size": img_size,
        "test_metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ckpt_path)
    print(f"\n[Checkpoint] Saved SOTA checkpoint to: {ckpt_path}")

    if model_type == "resnet":
        main_ckpt = CHECKPOINT_DIR / "optical_sar_net_sota.pth"
        torch.save({
            "model_name": arch_name,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "parameters": p_count,
            "optimal_thresholds": thresholds,
            "img_size": img_size,
            "test_metrics": test_metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, main_ckpt)
        print(f"[Checkpoint] Also updated default SOTA checkpoint: {main_ckpt}")

    # Save JSON Report
    report_data = {
        "benchmark": "BigEarthNet-MM / BEN-GE-800 SOTA Evaluation",
        "model": arch_name,
        "architecture": "Bidirectional Cross-Attention + ASPP Dual-Encoder Network with Pretrained Optical Backbone",
        "parameters": p_count,
        "evaluation_protocol": "4-Fold Dihedral Test-Time Augmentation (TTA) + Calibrated Thresholds",
        "test_pairs_evaluated": len(test_ds),
        "thresholds": thresholds,
        "land_cover_extraction_metrics": test_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rep_name = "optical_sar_resnet_sota_metrics.json" if model_type == "resnet" else "optical_sar_sota_metrics.json"
    out_json = REPORTS_DIR / rep_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[Report] Saved benchmark report to: {out_json}")

    if model_type == "resnet":
        with open(REPORTS_DIR / "optical_sar_sota_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "scratch"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    run_training_and_eval(model_type=args.model_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

