#!/usr/bin/env python3
"""
ml_pipeline/scripts/unpack_cdvqa_images.py

Unpacks authentic SECOND dataset imagery from second_dataset.zip into
ml_pipeline/authentic_data/cdvqa/images/ for authentic CDVQA evaluation.
"""

import sys
import os
import zipfile
import subprocess
from pathlib import Path

ZIP_PATH = Path("D:/sih26167/ml_pipeline/authentic_data/_downloads/second_dataset.zip")
TARGET_IMAGES_DIR = Path("D:/sih26167/ml_pipeline/authentic_data/cdvqa/images")
SCRATCH_DIR = Path("D:/sih26167/ml_pipeline/authentic_data/_downloads")

def unpack():
    if not ZIP_PATH.exists():
        print(f"[ERROR] {ZIP_PATH} does not exist.")
        return False

    print(f"[INFO] Inspecting {ZIP_PATH} ({ZIP_PATH.stat().st_size:,d} bytes)...")
    TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    rar_path = SCRATCH_DIR / "SECOND_train_set.rar"
    if not rar_path.exists():
        print(f"[INFO] Extracting SECOND_train_set.rar from {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extract("SECOND_train_set.rar", SCRATCH_DIR)
        print(f"[DONE] Extracted {rar_path} ({rar_path.stat().st_size:,d} bytes)")
    else:
        print(f"[SKIP] {rar_path} already extracted ({rar_path.stat().st_size:,d} bytes)")

    print(f"[INFO] Unpacking SECOND_train_set.rar into {TARGET_IMAGES_DIR} via tar.exe...")
    cmd = ["tar.exe", "-xf", str(rar_path), "-C", str(TARGET_IMAGES_DIR)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[WARN] tar.exe stderr: {res.stderr[:500]}")
    else:
        print(f"[DONE] Extraction completed successfully.")

    for d in ["im1", "im2", "label1", "label2"]:
        p = TARGET_IMAGES_DIR / d
        if p.exists():
            cnt = len(list(p.glob("*.png")))
            print(f"  Directory {d}/: {cnt:,d} authentic PNGs present")
        else:
            print(f"  [MISSING] Directory {d}/ not found")

    return True

if __name__ == "__main__":
    unpack()
