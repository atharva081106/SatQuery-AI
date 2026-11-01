"""
BigEarthNet-MM Dataset Loader
Source:  https://huggingface.co/datasets/torchgeo/bigearthnet  (torchgeo V2 mirror)
Paper:   Sumbul et al., BigEarthNet-MM, IEEE GRSM 2021  (arXiv:2105.07921)
Sensors: Sentinel-1 IW GRD (VV + VH backscatter) + Sentinel-2 MSI Level-2A (12 bands)
Labels:  CORINE Land Cover 2018 — 19-class scheme
License: CDLA-Permissive-1.0 / CC BY 4.0

This loader NEVER falls back to synthetic/procedurally-generated imagery.
If the real dataset is not present on disk, it raises FileNotFoundError with
download instructions rather than silently substituting noise images.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    HAS_TORCH = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Official CORINE Land Cover 19-class nomenclature used in BigEarthNet-MM
CLC_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

# Mapping CLC-19 labels to 3-class fusion target (Built-up, Water, Vegetation)
CLC_TO_FUSION_CLASS = {
    "Urban fabric": "built_up",
    "Industrial or commercial units": "built_up",
    "Inland waters": "water",
    "Marine waters": "water",
    "Coastal wetlands": "water",
    "Inland wetlands": "water",
    "Broad-leaved forest": "vegetation",
    "Coniferous forest": "vegetation",
    "Mixed forest": "vegetation",
    "Arable land": "vegetation",
    "Pastures": "vegetation",
    "Complex cultivation patterns": "vegetation",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "vegetation",
    "Natural grassland and sparsely vegetated areas": "vegetation",
    "Transitional woodland, shrub": "vegetation",
    "Agro-forestry areas": "vegetation",
    "Moors, heathland and sclerophyllous vegetation": "vegetation",
}

AUTHENTIC_DATA_DIR = Path(__file__).resolve().parent.parent / "authentic_data" / "ben_ge_800_prepared"
DEFAULT_CACHE_DIR = AUTHENTIC_DATA_DIR if AUTHENTIC_DATA_DIR.exists() else (Path(__file__).resolve().parent.parent / "dataset_cache" / "bigearthnet")

DOWNLOAD_INSTRUCTIONS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BigEarthNet-MM real dataset not found at: {data_dir}

 To download (requires ~65 GB free disk space for full dataset,
 or use --max-samples to cap at a subset):

   pip install torchgeo huggingface_hub
   python -c "
from torchgeo.datasets import BigEarthNet
BigEarthNet(root='{data_dir}', split='train', bands='all', num_classes=19, download=True)
BigEarthNet(root='{data_dir}', split='val',   bands='all', num_classes=19, download=True)
BigEarthNet(root='{data_dir}', split='test',  bands='all', num_classes=19, download=True)
"

 Or use the torchgeo-based bootstrap helper:
   python ml_pipeline/data/bigearthnet_loader.py --download --data-dir {data_dir} --max-samples 500

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def download_bigearthnet_torchgeo(data_dir: Path, max_samples: Optional[int] = None):
    """
    Download real BigEarthNet-MM via TorchGeo and convert to the flat cache format
    expected by BigEarthNetDataset.

    Sentinel-2 patches: 120x120 px, 12 spectral bands (10-60 m GSD) saved as uint16 GeoTIFFs.
    Sentinel-1 patches: 120x120 px, VV+VH dual-polarization SAR (IW GRD) saved as float32 .npy.
    """
    try:
        from torchgeo.datasets import BigEarthNet as TGBigEarthNet
    except ImportError:
        raise RuntimeError(
            "torchgeo is required to download BigEarthNet-MM.\n"
            "Install it with:  pip install torchgeo"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    for split in splits:
        print(f"[BigEarthNet-MM] Downloading '{split}' split (real Sentinel-1+2 patches)...")
        tg_ds = TGBigEarthNet(
            root=str(data_dir),
            split=split,
            bands="all",        # loads both S1 (VV,VH) and S2 (12-band) tensors
            num_classes=19,
            download=True,
        )

        split_dir = data_dir / split
        s2_dir = split_dir / "s2_rgb"
        s1_dir = split_dir / "s1_sar"
        s2_dir.mkdir(parents=True, exist_ok=True)
        s1_dir.mkdir(parents=True, exist_ok=True)

        n_total = len(tg_ds) if max_samples is None else min(max_samples, len(tg_ds))
        annotations = []

        for idx in range(n_total):
            sample = tg_ds[idx]
            # TorchGeo returns image tensor [C, H, W]:
            #   S1: channels 0-1 (VV, VH backscatter, float32 dB)
            #   S2: channels 2-13 (B01-B12, uint16 reflectance)
            img_t = sample["image"]  # [14, 120, 120] float32

            s1_arr = img_t[:2].numpy().transpose(1, 2, 0)   # [120,120,2] float32 VV+VH
            s2_all = img_t[2:].numpy()                       # [12,120,120] uint16 reflectance

            # Derive Sentinel-2 RGB (B4=Red, B3=Green, B2=Blue) normalized to uint8
            # Channels in BigEarthNet S2 order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
            # Indices:                          0    1    2    3    4    5    6    7   8   9   10   11
            s2_r = s2_all[3]  # B04 Red
            s2_g = s2_all[2]  # B03 Green
            s2_b = s2_all[1]  # B02 Blue
            # Percentile stretch to uint8
            def stretch(arr):
                lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                return np.clip((arr.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

            rgb = np.stack([stretch(s2_r), stretch(s2_g), stretch(s2_b)], axis=-1)  # [120,120,3]

            # Decode CLC-19 multi-hot label tensor → string list
            label_tensor = sample["label"]  # [19] multi-hot
            label_indices = label_tensor.nonzero(as_tuple=True)[0].tolist()
            clc_labels = [CLC_19_CLASSES[i] for i in label_indices]

            patch_id = f"S2A_MSIL2A_2018_{split}_{idx:05d}"
            s1_patch_id = f"S1A_IW_GRDH_2018_{split}_{idx:05d}"
            s2_path = s2_dir / f"{patch_id}.png"
            s1_path = s1_dir / f"{s1_patch_id}.npy"

            Image.fromarray(rgb).save(s2_path)
            np.save(str(s1_path), s1_arr)

            annotations.append({
                "patch_id": patch_id,
                "s1_patch_id": s1_patch_id,
                "s2_image_path": str(s2_path.relative_to(data_dir)),
                "s1_sar_path": str(s1_path.relative_to(data_dir)),
                "clc_labels": clc_labels,
                "split": split,
                "data_source": "BigEarthNet-MM (real Sentinel-1 IW GRD + Sentinel-2 MSI L2A)"
            })

            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{n_total} patches saved...")

        annot_path = split_dir / f"{split}_annotations.json"
        with open(annot_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)
        print(f"[BigEarthNet-MM] '{split}' split complete: {len(annotations)} real patches cached.")

    meta = {
        "dataset_name": "BigEarthNet-MM",
        "reference": "Sumbul et al., arXiv:2105.07921 (BigEarthNet-MM, IEEE GRSM 2021)",
        "hf_mirror": "torchgeo/bigearthnet (V2)",
        "license": "CDLA-Permissive-1.0",
        "modality": "Sentinel-1 SAR IW GRD (VV+VH) + Sentinel-2 MSI L2A (12-band → RGB)",
        "total_samples": max_samples,
        "clc_classes": CLC_19_CLASSES,
        "data_provenance": "REAL — no synthetic generation; original ESA Copernicus tiles"
    }
    with open(data_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[BigEarthNet-MM] Real dataset ready at: {data_dir}")


class BigEarthNetDataset(Dataset):
    """
    BigEarthNet-MM PyTorch Dataset Loader.

    Loads REAL Sentinel-1 SAR (VV+VH) and Sentinel-2 optical (RGB from B4/B3/B2)
    paired patches. Raises FileNotFoundError if the real dataset has not been downloaded;
    will NEVER substitute synthetic/procedurally-generated imagery.

    Parameters
    ----------
    data_dir : str
        Root directory of the cached BigEarthNet-MM data.
    split : str
        One of 'train', 'val', 'test'.
    modality : str
        's2' (optical only), 's1' (SAR only), or 'both' (default).
    max_samples : int, optional
        Cap number of samples. If None, loads entire split.
    """

    def __init__(
        self,
        data_dir: str = str(DEFAULT_CACHE_DIR),
        split: str = "train",
        modality: str = "both",
        max_samples: Optional[int] = None,
        processor: Any = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.processor = processor

        annot_path = self.data_dir / split / f"{split}_annotations.json"
        if not annot_path.exists():
            raise FileNotFoundError(
                DOWNLOAD_INSTRUCTIONS.format(data_dir=str(self.data_dir)) +
                f"\nAnnotation file not found: {annot_path}"
            )

        with open(annot_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        # Verify at least the first sample is real (not synthetic fallback)
        if self.samples:
            first = self.samples[0]
            if first.get("data_source", "").startswith("SYNTHETIC"):
                raise FileNotFoundError(
                    "Existing cache contains synthetic data. Delete it and re-download.\n" +
                    DOWNLOAD_INSTRUCTIONS.format(data_dir=str(self.data_dir))
                )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        output = {
            "patch_id": item["patch_id"],
            "clc_labels": item["clc_labels"],
            "split": self.split,
            "data_source": item.get("data_source", "BigEarthNet-MM (real)"),
        }

        # Sentinel-2 RGB (B4/B3/B2 percentile-stretched to uint8)
        if self.modality in ["s2", "both"]:
            s2_full_path = self.data_dir / item["s2_image_path"]
            if not s2_full_path.exists():
                raise FileNotFoundError(f"S2 patch missing: {s2_full_path}")
            output["s2_image"] = Image.open(s2_full_path).convert("RGB")

        # Sentinel-1 SAR VV+VH dual-polarization float32 backscatter [H,W,2]
        if self.modality in ["s1", "both"]:
            s1_full_path = self.data_dir / item["s1_sar_path"]
            if not s1_full_path.exists():
                raise FileNotFoundError(f"S1 patch missing: {s1_full_path}")
            s1_data = np.load(str(s1_full_path))
            output["s1_sar"] = torch.from_numpy(s1_data) if HAS_TORCH else s1_data

        # ESA WorldCover 10m Ground Truth Mask [H, W, 3] (if available in authentic dataset)
        if "target_mask_path" in item:
            tgt_full_path = self.data_dir / item["target_mask_path"]
            if tgt_full_path.exists():
                tgt_data = np.load(str(tgt_full_path))
                output["target_mask"] = torch.from_numpy(tgt_data) if HAS_TORCH else tgt_data

        # Optional VLM processor
        if self.processor is not None and "s2_image" in output:
            caption = item.get("caption", " ".join(item["clc_labels"]))
            enc = self.processor(output["s2_image"], caption, return_tensors="pt")
            for k, v in enc.items():
                output[k] = v.squeeze(0)

        return output

    def get_class_distribution(self) -> Dict[str, int]:
        counts = {cls_name: 0 for cls_name in CLC_19_CLASSES}
        for item in self.samples:
            for lbl in item["clc_labels"]:
                if lbl in counts:
                    counts[lbl] += 1
        return counts

    def get_fusion_label_distribution(self) -> Dict[str, int]:
        """Distribution of samples by coarse 3-class fusion target."""
        counts = {"built_up": 0, "water": 0, "vegetation": 0, "other": 0}
        for item in self.samples:
            mapped = set(CLC_TO_FUSION_CLASS.get(l, "other") for l in item["clc_labels"])
            for m in mapped:
                counts[m] += 1
        return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigEarthNet-MM Loader & Real-Data Downloader")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Limit samples per split (None = full ~590k patches)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--download", action="store_true",
                        help="Download real BigEarthNet-MM via TorchGeo before loading")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.download:
        download_bigearthnet_torchgeo(data_dir, max_samples=args.max_samples)

    ds = BigEarthNetDataset(data_dir=str(data_dir), split=args.split, max_samples=args.max_samples)
    print(f"Loaded {len(ds)} REAL BigEarthNet-MM samples for split '{args.split}'.")
    sample = ds[0]
    print(f"Sample 0 ID:     {sample['patch_id']}")
    print(f"Sample 0 Labels: {sample['clc_labels']}")
    print(f"Sample 0 Source: {sample['data_source']}")
    print(f"S2 image size:   {sample['s2_image'].size}")
    print(f"S1 SAR shape:    {sample['s1_sar'].shape}")
    dist = ds.get_fusion_label_distribution()
    print(f"Fusion label distribution: {dist}")
