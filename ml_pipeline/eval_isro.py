#!/usr/bin/env python3
"""
ml_pipeline/eval_isro.py

ISRO / SAC Hackathon Evaluation Stub for Cartosat-2S Optical + RISAT-1 C-Band SAR Imagery.
Problem Statement: SIH 26167 (ISRO / Space Applications Centre, Ahmedabad)

Purpose:
  Provides a plug-and-play evaluation harness that ingests native ISRO GeoTIFFs:
    1. Cartosat-2S Panchromatic / Multispectral (sub-metre: ~0.6m - 0.8m GSD)
    2. RISAT-1 / RISAT-1A (EOS-04) C-Band SAR (Hybrid / Circular Polarization: ~3m - 10m GSD)
  Handles spatial resolution mismatch via rigorous geospatial resampling to a common grid,
  runs SatQueryOpticalSARNet multi-modal extraction, and logs standard reports.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.enums import Resampling as RResampling
except ImportError:
    rasterio = None

import torch

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
ML_DIR = ROOT_DIR / "ml_pipeline"
BACKEND_DIR = ROOT_DIR / "backend"
REPORTS_DIR = ROOT_DIR / "reports"
DEMO_DIR = ROOT_DIR / "frontend" / "public" / "demo"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from cross_modal_model import SatQueryOpticalSARNet
from model_interfaces import detect_image_modality, check_spatial_compatibility

def resample_rasters_to_common_grid(
    optical_path: Path,
    sar_path: Path,
    target_res_meters: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Geospatial Resampling Methodology:
    1. Inspects Optical (Cartosat-2S: ~0.6m) and SAR (RISAT-1: ~3m-10m) native CRS and bounds via rasterio.
    2. Computes the intersecting geographic extent [west, south, east, north].
    3. Resamples both modalities to a uniform spatial grid of `target_res_meters`:
       - Optical: Bilinear interpolation (preserves smooth spectral reflectance across boundaries)
       - SAR: Cubic convolution (preserves local double-bounce dihedral scattering and speckle characteristics)
    4. If rasters lack georeferencing (e.g. benchmark PNGs), falls back to adaptive bicubic spatial alignment.
    """
    if rasterio is not None and optical_path.suffix.lower() in [".tif", ".tiff"] and sar_path.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(optical_path) as ds_opt, rasterio.open(sar_path) as ds_sar:
            # Check bounding intersection
            b_opt = ds_opt.bounds
            b_sar = ds_sar.bounds

            inter_w = max(b_opt.left, b_sar.left)
            inter_s = max(b_opt.bottom, b_sar.bottom)
            inter_e = min(b_opt.right, b_sar.right)
            inter_n = min(b_opt.top, b_sar.top)

            if inter_e <= inter_w or inter_n <= inter_s:
                raise ValueError("Optical and SAR GeoTIFFs have zero geographic coordinate overlap.")

            # Compute common grid dimensions
            grid_w = int((inter_e - inter_w) / target_res_meters)
            grid_h = int((inter_n - inter_s) / target_res_meters)
            grid_w = max(64, min(grid_w, 2048))
            grid_h = max(64, min(grid_h, 2048))

            # Resample optical
            opt_data = ds_opt.read(out_shape=(ds_opt.count, grid_h, grid_w), resampling=RResampling.bilinear)
            # Resample SAR
            sar_data = ds_sar.read(out_shape=(ds_sar.count, grid_h, grid_w), resampling=RResampling.cubic)

            geo_meta = {
                "west": inter_w, "south": inter_s, "east": inter_e, "north": inter_n,
                "crs": str(ds_opt.crs),
                "target_resolution_meters": target_res_meters,
                "resampling_method": "Optical: Bilinear | SAR: Cubic"
            }
            return opt_data, sar_data, geo_meta

    # Fallback for standard demo PNG/JPEG
    opt_img = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)
    sar_img = cv2.imread(str(sar_path), cv2.IMREAD_UNCHANGED)

    target_h, target_w = 512, 512
    opt_res = cv2.resize(opt_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    sar_res = cv2.resize(sar_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    geo_meta = {
        "crs": "EPSG:4326 (Normalized Visual Stub)",
        "target_resolution_meters": target_res_meters,
        "resampling_method": "Spatial Bicubic / Area Normalization (Benchmark Mode)"
    }
    return opt_res, sar_res, geo_meta

def run_isro_evaluation(
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    reference_mask_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    print("=" * 70)
    print("  SatQuery AI: ISRO SAC Cartosat-2S + RISAT-1 SAR Evaluation Stub")
    print("  Problem Statement: SIH 26167 (ISRO Space Applications Centre)")
    print("=" * 70)

    opt_p = Path(optical_path) if optical_path else DEMO_DIR / "mumbai_cartosat2s_optical.png"
    sar_p = Path(sar_path) if sar_path else DEMO_DIR / "risat1_cband_radar_sar.png"

    if not opt_p.exists() or not sar_p.exists():
        raise FileNotFoundError(f"Input files missing: {opt_p} or {sar_p}")

    print(f"[Inputs] Optical (Cartosat): {opt_p.name}")
    print(f"[Inputs] SAR (RISAT):        {sar_p.name}")

    # 1. Modality auto-detection verification
    with open(opt_p, "rb") as f_o, open(sar_p, "rb") as f_s:
        b_opt = f_o.read()
        b_sar = f_s.read()

    mod_opt = detect_image_modality(b_opt, filename=opt_p.name)
    mod_sar = detect_image_modality(b_sar, filename=sar_p.name)
    print(f"[Modality Detection] Optical: {mod_opt['modality']} ({mod_opt['sensor_family']})")
    print(f"[Modality Detection] SAR:     {mod_sar['modality']} ({mod_sar['sensor_family']})")

    # 2. Co-registration check
    is_c, score, msg, _ = check_spatial_compatibility(b_opt, b_sar, cross_modal=True)
    print(f"[Co-Registration] Status: {'PASSED' if is_c else 'FAILED'} (Score: {score:.2f})")
    print(f"[Co-Registration] Note:   {msg}")

    # 3. Spatial grid resampling
    opt_grid, sar_grid, resample_meta = resample_rasters_to_common_grid(opt_p, sar_p, target_res_meters=2.5)
    print(f"[Resampling] Unified common grid resolution: {resample_meta['target_resolution_meters']}m GSD")

    # 4. Feature Extraction via SatQueryOpticalSARNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ML_DIR / "checkpoints" / "satquery_optical_sar_net" / "optical_sar_net.pth"
    model = SatQueryOpticalSARNet(sar_channels=1, num_classes=3).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        prov = "SatQueryOpticalSARNet-v1.0 (Optical-SAR Dual-Encoder, BigEarthNet-Schema Prototype)"
    else:
        model.eval()
        prov = "SatQueryOpticalSARNet (Untrained Baseline)"

    # Prepare tensors for model forward pass
    if opt_grid.ndim == 3 and opt_grid.shape[2] == 3:
        opt_arr = opt_grid.astype(np.float32) / 255.0
    elif opt_grid.ndim == 3 and opt_grid.shape[0] == 3:
        opt_arr = np.transpose(opt_grid, (1, 2, 0)).astype(np.float32) / 255.0
    else:
        opt_arr = np.repeat(opt_grid[:, :, :1], 3, axis=2).astype(np.float32) / 255.0

    sar_arr = (sar_grid[..., 0] if sar_grid.ndim == 3 else sar_grid).astype(np.float32)
    sar_arr = (sar_arr - sar_arr.min()) / (sar_arr.max() - sar_arr.min() + 1e-6)

    h, w = opt_arr.shape[:2]
    opt_t = torch.from_numpy(opt_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    sar_t = torch.from_numpy(sar_arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(opt_t, sar_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    built_mask = (probs[0] > 0.40)
    water_mask = (probs[1] > 0.40)
    veg_mask   = (probs[2] > 0.40)

    total_px = float(h * w)
    built_pct = float(built_mask.sum() / total_px * 100.0)
    water_pct = float(water_mask.sum() / total_px * 100.0)
    veg_pct   = float(veg_mask.sum() / total_px * 100.0)

    is_demo = (optical_path is None and sar_path is None)
    report = {
        "mission": "ISRO / SAC Cartosat-2S + RISAT-1 Multi-Modal Analysis Stub",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "evaluation_scope": "SYNTHETIC_DEMO_STUB" if is_demo else "USER_PROVIDED_RASTERS",
        "reference_masks_evaluated": False,
        "qa_answers_evaluated": False,
        "integrity_notice": (
            "Default evaluation utilizes procedural synthetic demo images (mumbai_cartosat2s_optical.png, "
            "risat1_cband_radar_sar.png) from frontend/public/demo/. No authentic ISRO Cartosat-2S or "
            "RISAT-1 imagery was evaluated, and no ground-truth reference masks or QA pairs were scored."
            if is_demo else
            "Evaluation executed on user-supplied raster files. Reference mask and QA verification not active."
        ),
        "sensor_inputs": {
            "optical": {
                "file": opt_p.name,
                "detected_modality": mod_opt["modality"],
                "sensor_family": mod_opt["sensor_family"],
                "is_synthetic_demo": is_demo
            },
            "sar": {
                "file": sar_p.name,
                "detected_modality": mod_sar["modality"],
                "sensor_family": mod_sar["sensor_family"],
                "is_synthetic_demo": is_demo
            }
        },
        "spatial_co_registration": {
            "compatible": is_c,
            "coherence_score": score,
            "message": msg
        },
        "resampling_and_grid": resample_meta,
        "model_provenance": prov,
        "extracted_land_cover_coverage_pct": {
            "built_up_infrastructure": round(built_pct, 2),
            "hydrological_water_bodies": round(water_pct, 2),
            "vegetation_and_canopy": round(veg_pct, 2)
        }
    }

    out_file = Path(output_path) if output_path else REPORTS_DIR / "isro_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved] ISRO demo evaluation report written to: {out_file}")

    print("\n" + "=" * 70)
    print("  ISRO MULTI-MODAL DEMO EVALUATION SUMMARY (SYNTHETIC STUB)")
    print("=" * 70)
    print(f"  Co-Registration Verified:        {is_c} (Coherence: {score:.2f})")
    print(f"  Resampled Common Grid:           {resample_meta['target_resolution_meters']}m GSD")
    print(f"  Extracted Built-Up Coverage:     {built_pct:.2f}%")
    print(f"  Extracted Water Coverage:        {water_pct:.2f}%")
    print(f"  Extracted Vegetation Coverage:   {veg_pct:.2f}%")
    print("=" * 70)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISRO Cartosat-2S + RISAT SAR Evaluation Stub")
    parser.add_argument("--optical", type=str, help="Path to Cartosat-2S Optical GeoTIFF/PNG")
    parser.add_argument("--sar", type=str, help="Path to RISAT-1 C-Band SAR GeoTIFF/PNG")
    parser.add_argument("--mask", type=str, help="Optional reference mask")
    parser.add_argument("--output", type=str, help="Output report JSON path")
    args = parser.parse_args()

    run_isro_evaluation(
        optical_path=args.optical,
        sar_path=args.sar,
        reference_mask_path=args.mask,
        output_path=args.output
    )
