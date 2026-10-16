"""
SatQuery AI - Remote Sensing Vision-Language Model Training Pipeline
Stages:
  Stage 1: Image-Text Alignment & Captioning on BigEarthNet.txt (arXiv:2603.29630)
  Stage 2: VQA Fine-Tuning on RSVQA (arXiv:2003.07333) + VRSBench VQA (arXiv:2406.12384)
  Stage 3: Text-Guided Visual Grounding on VRSBench (arXiv:2406.12384)

Outputs:
  - Checkpoint files in ml_pipeline/checkpoints/satquery_rs_vlm/
  - Authentic training report in ml_pipeline/checkpoints/satquery_rs_vlm/training_report.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering

# Local imports
ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from data.bigearthnet_loader import BigEarthNetDataset
from data.rsvqa_loader import RSVQADataset
from data.vrsbench_loader import VRSBenchDataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set offline and local cache
if os.name == "nt" and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "d:\\sih26167\\.huggingface_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def train_staged_vlm(args):
    print("=" * 65, flush=True)
    print("SatQuery AI: Staged Remote Sensing VLM Training Pipeline", flush=True)
    print("=" * 65, flush=True)
    
    start_time = time.time()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Device: {device} | Random Seed: {args.seed} | Max Samples: {args.max_samples}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = "Salesforce/blip-vqa-base"
    print(f"Loading base VLM: {base_model_id} (Local Cache Only)...")
    processor = BlipProcessor.from_pretrained(base_model_id, local_files_only=True)
    model = BlipForQuestionAnswering.from_pretrained(base_model_id, local_files_only=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_history = []

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: BigEarthNet.txt Image-Text Alignment & Captioning
    # ─────────────────────────────────────────────────────────────
    print("\n[STAGE 1/3] BigEarthNet.txt Image-Text Captioning Alignment...")
    ds_stage1 = BigEarthNetDataset(split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(ds_stage1)} BigEarthNet.txt samples.")
    
    model.train()
    stage1_loss = 0.0
    steps = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in range(len(ds_stage1)):
            item = ds_stage1[i]
            img = item["s2_image"]
            prompt = "A high-resolution satellite observation showing:"
            target_caption = item["caption"]
            
            inputs = processor(img, prompt, return_tensors="pt").to(device)
            labels = processor.tokenizer(target_caption, return_tensors="pt", max_length=64, truncation=True).input_ids.to(device)
            
            # BLIP VQA training forward pass with labels
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_ep_loss = epoch_loss / max(1, len(ds_stage1))
        loss_history.append({"stage": 1, "epoch": epoch + 1, "dataset": "BigEarthNet.txt", "loss": round(avg_ep_loss, 4)})
        print(f"  Stage 1 Epoch {epoch+1}/{args.epochs} - Loss: {avg_ep_loss:.4f}")

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: RSVQA + VRSBench VQA Fine-Tuning
    # ─────────────────────────────────────────────────────────────
    print("\n[STAGE 2/3] RSVQA + VRSBench VQA Fine-Tuning...")
    ds_rsvqa = RSVQADataset(split="train", max_samples=args.max_samples)
    ds_vrs_vqa = VRSBenchDataset(task="vqa", split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(ds_rsvqa)} RSVQA pairs + {len(ds_vrs_vqa)} VRSBench VQA pairs.")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_items = min(len(ds_rsvqa), len(ds_vrs_vqa))
        for i in range(n_items):
            # Mix RSVQA and VRSBench questions
            r_item = ds_rsvqa[i]
            inputs = processor(r_item["image"], r_item["question"], return_tensors="pt").to(device)
            labels = processor.tokenizer(r_item["answer"], return_tensors="pt", max_length=16, truncation=True).input_ids.to(device)
            
            outputs = model(**inputs, labels=labels)
            loss1 = outputs.loss

            v_item = ds_vrs_vqa[i]
            inputs2 = processor(v_item["image"], v_item["question"], return_tensors="pt").to(device)
            labels2 = processor.tokenizer(v_item["answers"][0], return_tensors="pt", max_length=16, truncation=True).input_ids.to(device)
            
            outputs2 = model(**inputs2, labels=labels2)
            loss2 = outputs2.loss

            tot_loss = (loss1 + loss2) / 2.0
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            epoch_loss += tot_loss.item()

        avg_ep_loss = epoch_loss / max(1, n_items)
        loss_history.append({"stage": 2, "epoch": epoch + 1, "dataset": "RSVQA + VRSBench VQA", "loss": round(avg_ep_loss, 4)})
        print(f"  Stage 2 Epoch {epoch+1}/{args.epochs} - Loss: {avg_ep_loss:.4f}")

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: VRSBench Text-Guided Visual Grounding Conditioning
    # ─────────────────────────────────────────────────────────────
    print("\n[STAGE 3/3] VRSBench Visual Grounding Coordinate Conditioning...")
    ds_grd = VRSBenchDataset(task="grounding", split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(ds_grd)} VRSBench Grounding items.")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in range(len(ds_grd)):
            item = ds_grd[i]
            img = item["image"]
            query = f"Locate {item['expression']} and output coordinates:"
            bbox_str = f"[{item['bbox'][0]}, {item['bbox'][1]}, {item['bbox'][2]}, {item['bbox'][3]}]"
            
            inputs = processor(img, query, return_tensors="pt").to(device)
            labels = processor.tokenizer(bbox_str, return_tensors="pt", max_length=32, truncation=True).input_ids.to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_ep_loss = epoch_loss / max(1, len(ds_grd))
        loss_history.append({"stage": 3, "epoch": epoch + 1, "dataset": "VRSBench Grounding", "loss": round(avg_ep_loss, 4)})
        print(f"  Stage 3 Epoch {epoch+1}/{args.epochs} - Loss: {avg_ep_loss:.4f}")

    # ─────────────────────────────────────────────────────────────
    # Checkpoint Serialization & Training Provenance Logging
    # ─────────────────────────────────────────────────────────────
    wall_time = round(time.time() - start_time, 2)
    print(f"\n[SAVING] Serializing adapted weights to: {output_dir}")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    training_report = {
        "model_name": "SatQuery-RS-VLM-v1.0",
        "architecture": "BLIP-VQA RS-Adapted (ViT-B/16 + 12-layer Cross-Attention Decoder)",
        "base_model": base_model_id,
        "total_parameters": param_count,
        "trainable_parameters": trainable_count,
        "training_stages": [
            {"stage": 1, "name": "Image-Text Alignment & Captioning", "dataset": "BigEarthNet.txt (arXiv:2603.29630)", "samples": len(ds_stage1)},
            {"stage": 2, "name": "RS Visual Question Answering", "dataset": "RSVQA (arXiv:2003.07333) + VRSBench VQA", "samples": len(ds_rsvqa) + len(ds_vrs_vqa)},
            {"stage": 3, "name": "Text-Guided Grounding", "dataset": "VRSBench Grounding (arXiv:2406.12384)", "samples": len(ds_grd)}
        ],
        "random_seed": args.seed,
        "learning_rate": args.lr,
        "epochs_per_stage": args.epochs,
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not args.force_cpu else "CPU (Host)"
        },
        "wall_time_seconds": wall_time,
        "loss_trajectory": loss_history,
        "date_executed": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print(f"[SUCCESS] Checkpoint and training report saved to: {report_path}")
    print(f"Total Wall Time: {wall_time}s")
    return training_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatQuery Remote Sensing Adapted VLM")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/satquery_rs_vlm")
    parser.add_argument("--max_samples", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    
    train_staged_vlm(args)
