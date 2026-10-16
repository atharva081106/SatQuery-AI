#!/usr/bin/env python3
"""
ml_pipeline/train_change_detector.py

Training pipeline for SatQueryChangeNet (Siamese Remote Sensing Change Detection).
Trained on CDVQA (arXiv:2112.06343) bi-temporal change pairs and binary change masks.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from change_model import SatQueryChangeNet
from data.cdvqa_loader import CDVQADataset

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

def collate_fn(batch):
    t1_list = []
    t2_list = []
    mask_list = []
    for item in batch:
        t1_arr = np.array(item["t1_image"], dtype=np.float32) / 255.0
        t2_arr = np.array(item["t2_image"], dtype=np.float32) / 255.0
        mask_arr = (np.array(item["change_mask"], dtype=np.float32) > 128).astype(np.float32)

        t1_t = torch.from_numpy(t1_arr).permute(2, 0, 1)
        t2_t = torch.from_numpy(t2_arr).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        t1_list.append(t1_t)
        t2_list.append(t2_t)
        mask_list.append(mask_t)

    return {
        "t1": torch.stack(t1_list),
        "t2": torch.stack(t2_list),
        "masks": torch.stack(mask_list)
    }

def train_change_detector(epochs: int = 5, batch_size: int = 8, lr: float = 1e-3, max_samples: int = 70):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("SatQuery AI — Learned Change Detector Training (SatQueryChangeNet)")
    print(f"Hardware Device: {device} | Target: CDVQA (arXiv:2112.06343)")
    print("=" * 65)

    train_ds = CDVQADataset(split="train", max_samples=max_samples)
    val_ds = CDVQADataset(split="val", max_samples=max(10, int(max_samples * 0.2)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SatQueryChangeNet(in_channels=3).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] SatQueryChangeNet initialized with {param_count:,} parameters.")
    print(f"[DATA] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    best_val_loss = float("inf")
    history = []

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            masks = batch["masks"].to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = bce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train = train_loss / max(1, batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_ious = []
        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                masks = batch["masks"].to(device)

                logits = model(t1, t2)
                loss = bce(logits, masks) + 0.5 * dice(logits, masks)
                val_loss += loss.item()
                val_batches += 1

                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * masks).sum(dim=[1, 2, 3])
                union = (preds + masks).clamp(0, 1).sum(dim=[1, 2, 3])
                iou = (inter + 1e-6) / (union + 1e-6)
                val_ious.extend(iou.cpu().numpy().tolist())

        avg_val = val_loss / max(1, val_batches)
        mean_val_iou = float(np.mean(val_ious)) if val_ious else 0.0

        history.append({
            "epoch": ep,
            "train_loss": round(avg_train, 4),
            "val_loss": round(avg_val, 4),
            "val_iou": round(mean_val_iou, 4)
        })

        print(f"  Epoch {ep}/{epochs} - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val IoU: {mean_val_iou*100:.2f}%")

    total_time = round(time.time() - start_time, 2)
    print(f"\n[TRAINING] Completed {epochs} epochs in {total_time}s.")

    # Save Checkpoint
    ckpt_dir = ML_DIR / "checkpoints" / "satquery_change_net"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "change_net.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": 3,
        "parameters": param_count,
        "final_val_loss": round(avg_val, 4),
        "final_val_iou": round(mean_val_iou, 4),
        "history": history
    }, ckpt_file)

    report_file = ckpt_dir / "training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "SatQueryChangeNet-v1.0",
            "reference": "arXiv:2112.06343",
            "parameters": param_count,
            "device": device,
            "epochs": epochs,
            "total_time_seconds": total_time,
            "final_val_loss": round(avg_val, 4),
            "final_val_iou": round(mean_val_iou, 4),
            "history": history
        }, f, indent=2)

    print(f"[CHECKPOINT] Model saved to {ckpt_file}")
    print(f"[REPORT] Training report saved to {report_file}")
    return model, ckpt_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=70)
    args = parser.parse_args()
    train_change_detector(epochs=args.epochs, batch_size=args.batch_size, max_samples=args.max_samples)
