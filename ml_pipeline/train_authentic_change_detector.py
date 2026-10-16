#!/usr/bin/env python3
"""
ml_pipeline/train_authentic_change_detector.py

Trains SatQueryChangeNet on authentic CDVQA / SECOND bi-temporal image pairs.
Follows strict partition protocol:
- Train: authentic CDVQA train split
- Val: authentic CDVQA val split (for checkpoint selection and early stopping)
- Test/Test2: NEVER touched during training or selection.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path

# Enforce UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))

from change_model import SatQueryChangeNet

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

def derive_label2_path(label1_path: Path) -> Path:
    """
    Derive the T2 semantic label path from the T1 semantic label path.
    SECOND dataset stores T1 labels in 'label1/' and T2 labels in 'label2/'.
    """
    return Path(str(label1_path).replace(
        str(label1_path.parent),
        str(label1_path.parent).replace("label1", "label2"),
        1
    ))


def compute_binary_change_mask(label1_rgb_path: Path, label2_rgb_path: Path, img_size) -> np.ndarray:
    """
    Derive the binary change mask for the CDVQA / SECOND dataset.

    The SECOND dataset stores 6-class semantic segmentation maps for T1 and T2
    as RGB PNG images in 'label1/' and 'label2/' respectively.  The binary
    change mask is defined as any pixel whose land-cover class changed between
    the two timestamps, i.e.:

        change_mask[y, x] = 1  iff  label1_rgb[y, x] != label2_rgb[y, x]

    Using label1 alone (thresholded to grayscale) is INCORRECT — it produces
    a near-100% positive rate and measures semantic foreground coverage, NOT
    actual between-epoch change.
    """
    l1 = np.array(Image.open(label1_rgb_path).resize(img_size, Image.NEAREST))
    l2 = np.array(Image.open(label2_rgb_path).resize(img_size, Image.NEAREST))
    # If loaded as RGB, compare across channels; if grayscale, compare directly
    if l1.ndim == 3:
        change = np.any(l1 != l2, axis=-1).astype(np.float32)
    else:
        change = (l1 != l2).astype(np.float32)
    return change


class AuthenticCDVQAPairDataset(Dataset):
    """Loads unique physical image pairs from authentic CDVQA/SECOND dataset.

    Ground-truth binary change masks are derived by comparing the T1 semantic
    label (label1/) with the T2 semantic label (label2/).  Any pixel whose
    land-cover class changed between timestamps is marked as changed (1),
    all other pixels are unchanged (0).
    """
    def __init__(self, split="train", max_pairs=None, img_size=(256, 256)):
        self.split = split
        self.img_size = img_size
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

        # Correct binary change GT: any pixel where semantic class changed T1->T2
        mask_arr = compute_binary_change_mask(
            item["label1_path"], item["label2_path"], self.img_size
        )

        t1_t = torch.from_numpy(t1_arr).permute(2, 0, 1)
        t2_t = torch.from_numpy(t2_arr).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        return {
            "pair_id": item["pair_id"],
            "t1": t1_t,
            "t2": t2_t,
            "mask": mask_t
        }

def train_authentic_model(
    train_pairs: int = 400,
    val_pairs: int = 100,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print("SatQuery AI — Authentic CDVQA ChangeNet Training Pipeline", flush=True)
    print(f"Hardware Device: {device.upper()} | PyTorch: {torch.__version__}", flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"[DATA] Loading authentic train split (target: {train_pairs} pairs)...", flush=True)
    train_ds = AuthenticCDVQAPairDataset(split="train", max_pairs=train_pairs)
    print(f"[DATA] Loading authentic val split (target: {val_pairs} pairs)...", flush=True)
    val_ds = AuthenticCDVQAPairDataset(split="val", max_pairs=val_pairs)

    print(f"[DATA] Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs (Zero overlap)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SatQueryChangeNet(in_channels=3).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] SatQueryChangeNet initialized ({param_count:,} trainable parameters)", flush=True)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_iou = -1.0
    best_epoch = 0
    history = []
    start_time = time.time()

    ckpt_dir = ML_DIR / "checkpoints" / "satquery_change_net_authentic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_change_net.pth"

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = bce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_batches)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_ious = []
        val_f1s = []

        with torch.no_grad():
            for batch in val_loader:
                t1 = batch["t1"].to(device)
                t2 = batch["t2"].to(device)
                masks = batch["mask"].to(device)

                logits = model(t1, t2)
                loss = bce(logits, masks) + 0.5 * dice(logits, masks)
                val_loss += loss.item()
                val_batches += 1

                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * masks).sum(dim=[1, 2, 3])
                union = (preds + masks).clamp(0, 1).sum(dim=[1, 2, 3])
                iou = (inter + 1e-6) / (union + 1e-6)
                dice_score = (2.0 * inter + 1e-6) / (preds.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3]) + 1e-6)

                val_ious.extend(iou.cpu().numpy().tolist())
                val_f1s.extend(dice_score.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(1, val_batches)
        mean_val_iou = float(np.mean(val_ious))
        mean_val_f1 = float(np.mean(val_f1s))
        ep_duration = round(time.time() - ep_start, 2)

        ep_record = {
            "epoch": ep,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_mean_iou": round(mean_val_iou, 4),
            "val_mean_f1": round(mean_val_f1, 4),
            "duration_seconds": ep_duration
        }
        history.append(ep_record)

        print(f"  Epoch {ep}/{epochs} [{ep_duration}s] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {mean_val_iou*100:.2f}% | Val Dice: {mean_val_f1:.4f}", flush=True)

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            best_epoch = ep
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_iou": mean_val_iou,
                "val_f1": mean_val_f1,
                "parameters": param_count,
                "architecture": "SatQueryChangeNet (Siamese Feature Diff U-Net)"
            }, best_ckpt_path)
            print(f"    [*] Checkpoint updated: val_iou improved to {mean_val_iou*100:.2f}%", flush=True)

    total_duration = round(time.time() - start_time, 2)
    print(f"\n[TRAINING COMPLETE] Best Epoch {best_epoch} with Val mIoU: {best_val_iou*100:.2f}% (Total Time: {total_duration}s)", flush=True)

    # Compute Checkpoint Checksum
    with open(best_ckpt_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    report = {
        "experiment": "Authentic CDVQA ChangeNet Training",
        "benchmark": "CDVQA / SECOND (arXiv:2112.06343)",
        "model_architecture": "SatQueryChangeNet (Siamese U-Net)",
        "parameters": param_count,
        "hardware": device,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_iou_pct": round(best_val_iou * 100.0, 2),
        "total_duration_seconds": total_duration,
        "checkpoint_path": str(best_ckpt_path.relative_to(ROOT_DIR)),
        "checkpoint_sha256": sha256_hash,
        "history": history
    }

    report_path = ckpt_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] Training report: {report_path}", flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=int, default=300)
    parser.add_argument("--val-pairs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_authentic_model(
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
