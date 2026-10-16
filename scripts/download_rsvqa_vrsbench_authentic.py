#!/usr/bin/env python3
"""
scripts/download_rsvqa_vrsbench_authentic.py

Downloads and unpacks authentic evaluation data for:
1. RSVQA-LR (Zenodo 6344333 / Sylvain Lobry et al. 2020)
2. VRSBench (HuggingFace xiang709/VRSBench)
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path

# Enforce UTF-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT_DIR / "ml_pipeline" / "authentic_data" / "_downloads"
RSVQA_DIR = ROOT_DIR / "ml_pipeline" / "authentic_data" / "rsvqa_lr"
VRS_DIR = ROOT_DIR / "ml_pipeline" / "authentic_data" / "vrsbench"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
RSVQA_DIR.mkdir(parents=True, exist_ok=True)
VRS_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[EXISTS] {out_path.name} already downloaded ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[RESUME] Resuming {out_path.name} from {existing_bytes / 1024 / 1024:.2f} MB...")
    else:
        print(f"[DOWNLOAD] Downloading {out_path.name} from {url}...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(part_path, "ab" if existing_bytes > 0 else "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0)) + existing_bytes
            downloaded = existing_bytes
            block_size = 1024 * 1024 # 1 MB
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  {out_path.name}: {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="\r", flush=True)
            print(f"\n[DONE] Finished {out_path.name}")
        part_path.rename(out_path)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        raise

def main():
    print("=" * 70)
    print("Authentic RSVQA-LR and VRSBench Dataset Ingestion")
    print("=" * 70)

    # 1. RSVQA-LR from Zenodo 6344333
    rsvqa_files = [
        ("Images_LR.zip", "https://zenodo.org/api/records/6344334/files/Images_LR.zip/content"),
        ("LR_split_val_images.json", "https://zenodo.org/api/records/6344334/files/LR_split_val_images.json/content"),
        ("LR_split_val_questions.json", "https://zenodo.org/api/records/6344334/files/LR_split_val_questions.json/content"),
        ("LR_split_val_answers.json", "https://zenodo.org/api/records/6344334/files/LR_split_val_answers.json/content"),
        ("LR_split_train_images.json", "https://zenodo.org/api/records/6344334/files/LR_split_train_images.json/content"),
        ("LR_split_train_questions.json", "https://zenodo.org/api/records/6344334/files/LR_split_train_questions.json/content"),
        ("LR_split_train_answers.json", "https://zenodo.org/api/records/6344334/files/LR_split_train_answers.json/content"),
    ]

    print("\n--- 1. Downloading RSVQA-LR (~101 MB) ---")
    for fname, url in rsvqa_files:
        dest = DOWNLOADS_DIR / f"rsvqa_{fname}"
        download_file(url, dest)
        if fname.endswith(".json"):
            # copy directly to RSVQA_DIR
            with open(dest, "rb") as rf, open(RSVQA_DIR / fname, "wb") as wf:
                wf.write(rf.read())

    # Extract RSVQA images if not extracted
    img_dir = RSVQA_DIR / "Images"
    if not img_dir.exists() or len(list(img_dir.glob("*.png"))) < 700:
        print("[EXTRACT] Extracting Images_LR.zip to authentic_data/rsvqa_lr/Images...")
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DOWNLOADS_DIR / "rsvqa_Images_LR.zip", "r") as zf:
            zf.extractall(RSVQA_DIR)
        print(f"[EXTRACT] Extracted RSVQA-LR images: {len(list(img_dir.glob('*.png')))} images present.")

    # 2. VRSBench from HuggingFace xiang709/VRSBench
    print("\n--- 2. Downloading VRSBench Validation Data (~3.8 GB) ---")
    vrs_files = [
        ("VRSBench_EVAL_vqa.json", "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_vqa.json"),
        ("Annotations_val.zip", "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip"),
        ("Images_val.zip", "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"),
    ]

    for fname, url in vrs_files:
        dest = DOWNLOADS_DIR / f"vrsbench_{fname}"
        download_file(url, dest)
        if fname.endswith(".json"):
            with open(dest, "rb") as rf, open(VRS_DIR / fname, "wb") as wf:
                wf.write(rf.read())

    # Extract VRSBench Images_val.zip
    vrs_img_dir = VRS_DIR / "Images_val"
    if not vrs_img_dir.exists() or len(list(vrs_img_dir.glob("*.jpg"))) < 500:
        print("[EXTRACT] Extracting Images_val.zip to authentic_data/vrsbench/Images_val...")
        vrs_img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DOWNLOADS_DIR / "vrsbench_Images_val.zip", "r") as zf:
            zf.extractall(VRS_DIR)
        print(f"[EXTRACT] Extracted VRSBench images: {len(list(vrs_img_dir.glob('*.jpg')))} images present.")

    print("\n[SUCCESS] Authentic RSVQA-LR and VRSBench datasets ready.")

if __name__ == "__main__":
    main()
