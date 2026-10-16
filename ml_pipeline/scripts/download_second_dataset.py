#!/usr/bin/env python3
"""
ml_pipeline/scripts/download_second_dataset.py

Authentic Dataset Downloader for SECOND (CAPTAIN WHU) / CDVQA Benchmark.
Downloads second_dataset.zip (3,785,055,247 bytes / ~3.52 GiB) from official Google Drive source.
Supports chunked streaming, HTTP Range resume, and robust unbuffered progress reporting.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

GOOGLE_DRIVE_URL = "https://drive.usercontent.google.com/download?id=1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u&confirm=t"
EXPECTED_SIZE = 3785055247
TARGET_DIR = Path("D:/sih26167/ml_pipeline/authentic_data/_downloads")
TARGET_FILE = TARGET_DIR / "second_dataset.zip"

def download(url: str = GOOGLE_DRIVE_URL, target: Path = TARGET_FILE, expected_size: int = EXPECTED_SIZE):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    if target.exists():
        downloaded = target.stat().st_size
        if downloaded == expected_size:
            print(f"[SKIP] Archive already fully downloaded: {target} ({downloaded:,d} bytes)", flush=True)
            return target
        elif downloaded > expected_size:
            print(f"[WARN] Local file larger than expected ({downloaded:,d} > {expected_size:,d}). Truncating/restarting...", flush=True)
            downloaded = 0
            target.unlink()
        else:
            print(f"[RESUME] Resuming from byte {downloaded:,d} / {expected_size:,d} ({100.0*downloaded/expected_size:.1f}%)", flush=True)

    headers = {'User-Agent': 'Mozilla/5.0'}
    if downloaded > 0:
        headers['Range'] = f"bytes={downloaded}-{expected_size-1}"

    req = urllib.request.Request(url, headers=headers)
    chunk_size = 2 * 1024 * 1024 # 2MB chunks
    t_start = time.time()
    t_last_log = t_start
    bytes_since_log = 0

    mode = 'ab' if downloaded > 0 else 'wb'
    with urllib.request.urlopen(req, timeout=30) as resp, open(target, mode) as f:
        print(f"[START] Connected to stream. Downloading {expected_size - downloaded:,d} bytes...", flush=True)
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            f.flush()
            downloaded += len(chunk)
            bytes_since_log += len(chunk)
            
            now = time.time()
            if now - t_last_log >= 5.0 or downloaded == expected_size:
                dt = now - t_last_log
                speed = (bytes_since_log / (1024 * 1024)) / dt if dt > 0 else 0
                pct = 100.0 * downloaded / expected_size
                print(f"[{pct:5.1f}%] {downloaded:,d} / {expected_size:,d} bytes | Speed: {speed:5.2f} MB/s | Elapsed: {now-t_start:.1f}s", flush=True)
                t_last_log = now
                bytes_since_log = 0

    print(f"[DONE] Download complete: {target} ({downloaded:,d} bytes)", flush=True)
    return target

if __name__ == "__main__":
    download()
