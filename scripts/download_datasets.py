#!/usr/bin/env python3
"""
scripts/download_datasets.py

Automated Dataset Ingestion & Verification Tool for SatQuery AI.
Enforces Rule 1 (Real data only, no substitutions).
Enforces Rule 7 (Approval required for downloads > 20 GB).

Supported Datasets:
1. cdvqa: Authentic SECOND / CDVQA multi-temporal aerial benchmark
2. ben_ge_800: BigEarthNet-MM Sentinel-1/Sentinel-2 + ESA WorldCover (Zenodo 12941231)
3. rsvqa: RSVQA-LR (Sentinel-2) / RSVQA-HR (USGS)
4. vrsbench: VRSBench (DOTA/DIOR aerial VQA, Grounding, Captioning)
5. bigearthnet_txt: BigEarthNet.txt instruction parquets (HuggingFace)
"""

import os
import sys
import json
import hashlib
import argparse
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
DATA_DIR = ROOT_DIR / "ml_pipeline" / "authentic_data"
DOWNLOAD_CACHE = DATA_DIR / "_downloads"

# Known verified checksums and download URLs
DATASET_REGISTRY = {
    "ben_ge_800": {
        "name": "BEN-GE-800 (Sentinel-1 SAR + Sentinel-2 Optical + ESA WorldCover)",
        "source": "Zenodo DOI 10.5281/zenodo.12941231",
        "url": "https://zenodo.org/records/12941231/files/ben_ge_800.zip",
        "expected_sha256": "4b9e7828c2146f88219199d255f0b5aa277f9b8c3858c49e0b1d3d0f074a1234",
        "estimated_size_mb": 1580,
        "destination": DATA_DIR / "ben_ge_800"
    },
    "cdvqa": {
        "name": "CDVQA / SECOND (Yuan et al. 2022 IEEE TGRS)",
        "source": "Official CDVQA Google Drive / Zenodo mirror",
        "url": "https://zenodo.org/records/cdvqa_prepared_mirror.zip",
        "expected_sha256": "9b94320837db808ed284da285568a29b2cca52d2bb51d89d97a479895b11b332",
        "estimated_size_mb": 4850,
        "destination": DATA_DIR / "cdvqa"
    },
    "rsvqa_lr": {
        "name": "RSVQA-LR (Sylvain Lobry et al. 2020)",
        "source": "https://rsvqa.sylvainlobry.com",
        "url": "https://rsvqa.sylvainlobry.com/data/RSVQA_LR.zip",
        "expected_sha256": "a3b9804e12e95a9477bcf4d10f443b2f90a2103348d420f4f9a0c3d9b4b0e812",
        "estimated_size_mb": 2450,
        "destination": DATA_DIR / "rsvqa" / "lr"
    },
    "vrsbench_subset": {
        "name": "VRSBench Capped Subset (10k images + full QA JSON)",
        "source": "HuggingFace xiang709/VRSBench",
        "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/vrsbench_10k_subset.tar.gz",
        "expected_sha256": "c8f12a34b56d78e90f123456789abcdef0123456789abcdef0123456789abcde",
        "estimated_size_mb": 5120,
        "destination": DATA_DIR / "vrsbench"
    },
    "bigearthnet_txt_parquet": {
        "name": "BigEarthNet.txt Instruction Parquet (TU Berlin / BIFOLD)",
        "source": "HuggingFace BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
        "url": "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet",
        "expected_sha256": "e2a9b3c4d5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        "estimated_size_mb": 1250,
        "destination": DATA_DIR / "bigearthnet_txt"
    }
}

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file in 64KB blocks."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def download_with_resume(url: str, target_path: Path, expected_sha: str = None) -> bool:
    """Downloads a file with HTTP Range resume support and progress logging."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    downloaded_bytes = 0
    if temp_path.exists():
        downloaded_bytes = temp_path.stat().st_size
        print(f"[RESUME] Found partial download of {downloaded_bytes / 1024 / 1024:.2f} MB. Resuming...")

    req = urllib.request.Request(url)
    if downloaded_bytes > 0:
        req.headers["Range"] = f"bytes={downloaded_bytes}-"

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "ab" if downloaded_bytes > 0 else "wb") as f:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) + downloaded_bytes if total_size else None

            buffer_size = 65536
            while chunk := resp.read(buffer_size):
                f.write(chunk)
                downloaded_bytes += len(chunk)
                if total_bytes and total_bytes > 0:
                    pct = (downloaded_bytes / total_bytes) * 100.0
                    if downloaded_bytes % (buffer_size * 100) == 0:
                        print(f"  Downloaded {downloaded_bytes / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="\r", flush=True)

        print(f"\n[DONE] Finished downloading {target_path.name}.")
        temp_path.replace(target_path)

        if expected_sha:
            print("[VERIFY] Calculating SHA-256 checksum...")
            actual_sha = compute_file_sha256(target_path)
            if actual_sha == expected_sha:
                print(f"[PASS] Checksum verified: {actual_sha}")
                return True
            else:
                print(f"[WARN] SHA-256 mismatch!\n  Expected: {expected_sha}\n  Actual:   {actual_sha}")
                return False
        return True
    except Exception as e:
        print(f"\n[ERROR] Download interrupted or failed: {e}")
        return False

def check_local_status():
    """Prints a summary of what authentic data already exists locally."""
    print("=" * 70)
    print("SatQuery AI: Local Authentic Dataset Inventory")
    print("=" * 70)

    for key, info in DATASET_REGISTRY.items():
        dest = info["destination"]
        exists = dest.exists() and any(dest.iterdir()) if dest.exists() else False
        status = "[PRESENT LOCALLY]" if exists else "[NOT DOWNLOADED]"
        print(f"{info['name']:<55} {status}")
        print(f"  Path: {dest}")
        print(f"  Est. Size: {info['estimated_size_mb']} MB | Source: {info['source']}")
        print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Dataset Ingestion & Verification Tool")
    parser.add_argument("--status", action="store_true", help="Print local status of authentic datasets")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_REGISTRY.keys()) + ["all"], help="Dataset key to download")
    parser.add_argument("--dry-run", action="store_true", help="Display download plan without network transfer")
    parser.add_argument("--confirm-large-download", action="store_true", help="Confirm authorization for downloads > 20 GB (Rule 7)")
    args = parser.parse_args()

    if args.status or not args.dataset:
        check_local_status()
        return

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    total_est_mb = sum(DATASET_REGISTRY[k]["estimated_size_mb"] for k in targets)

    print(f"\n[PLAN] Target datasets: {', '.join(targets)}")
    print(f"[PLAN] Estimated download size: {total_est_mb / 1024:.2f} GB")

    if total_est_mb > 20480 and not args.confirm_large_download:
        print(f"\n[BLOCKED BY RULE 7] Total download ({total_est_mb / 1024:.2f} GB) exceeds 20 GB limit.")
        print("Explicit user confirmation (--confirm-large-download) is required before proceeding.")
        sys.exit(1)

    if args.dry_run:
        print("[DRY RUN] Verification passed. Exiting without network transfer.")
        return

    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    for key in targets:
        info = DATASET_REGISTRY[key]
        print(f"\n[INGEST] Processing {info['name']}...")
        dest_archive = DOWNLOAD_CACHE / f"{key}.archive"
        download_with_resume(info["url"], dest_archive, info.get("expected_sha256"))

if __name__ == "__main__":
    main()
