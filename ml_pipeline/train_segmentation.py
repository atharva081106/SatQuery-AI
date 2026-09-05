import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Optimize thread allocation across available cores
torch.set_num_threads(min(8, os.cpu_count() or 4))

sys.path.insert(0, os.path.dirname(__file__))
from sat_seg_model import SatSegNet, CombinedDiceLoss, SatelliteSegDataset

CLASS_NAMES = ["Background", "Water Bodies", "Vegetation", "Built-up", "Bare Soil", "Cloud"]

def compute_miou(preds, targets, num_classes=6):
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / float(union))
        
    return np.mean(ious) if ious else 1.0, ious

def train_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================", flush=True)
    print(f"🛰️  SatQuery AI: Rigorous Satellite Model Training Pipeline", flush=True)
    print(f"    Architecture: SatSegNet (Multi-Scale U-Net, 6 Classes)", flush=True)
    print(f"    Target Device: {device} | Threads: {torch.get_num_threads()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}", flush=True)
    print(f"============================================================", flush=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Datasets & Loaders
    data_dir = args.data_dir
    train_dataset = SatelliteSegDataset(data_dir, split="train", target_size=(128, 128))
    val_dataset = SatelliteSegDataset(data_dir, split="val", target_size=(128, 128))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset Loaded: {len(train_dataset)} Train Samples | {len(val_dataset)} Validation Samples", flush=True)
    
    # 2. Model & Loss & Optimizer
    model = SatSegNet(n_channels=3, n_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} trainable weights", flush=True)
    
    criterion = CombinedDiceLoss(num_classes=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_miou": [],
        "epochs": [],
        "best_epoch": 0,
        "best_miou": 0.0
    }
    
    best_miou = 0.0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= max(1, len(train_loader))
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                miou, _ = compute_miou(preds, masks)
                val_ious.append(miou)
                
        val_loss /= max(1, len(val_loader))
        mean_val_miou = float(np.mean(val_ious))
        
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_miou"].append(round(mean_val_miou, 4))
        history["epochs"].append(epoch)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {mean_val_miou * 100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
        
        # Save Best Checkpoint
        if mean_val_miou > best_miou or epoch == 1:
            best_miou = mean_val_miou
            history["best_epoch"] = epoch
            history["best_miou"] = round(best_miou, 4)
            ckpt_path = os.path.join(args.output_dir, "best_satsegnet.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": best_miou,
                "classes": CLASS_NAMES,
                "num_classes": 6
            }, ckpt_path)
            print(f"  ⭐ Saved new best model checkpoint to {ckpt_path} (mIoU: {best_miou * 100:.2f}%)", flush=True)
            
    total_time = time.time() - start_time
    history["total_training_time_seconds"] = round(total_time, 2)
    history["final_best_miou"] = round(best_miou, 4)
    
    # Save training metrics log
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n============================================================", flush=True)
    print(f"✅ Training Successfully Finished in {total_time:.1f} seconds ({total_time/60.0:.2f} minutes)!", flush=True)
    print(f"   Peak Validation mIoU: {best_miou * 100:.2f}%", flush=True)
    print(f"   Metrics Log: {metrics_path}", flush=True)
    print(f"============================================================", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatSegNet Multi-Class Remote Sensing Model")
    parser.add_argument("--data_dir", type=str, default="d:/sih26167/ml_pipeline/dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="d:/sih26167/ml_pipeline/checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate")
    args = parser.parse_args()
    train_segmentation(args)
