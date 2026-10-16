#!/usr/bin/env python3
"""
ml_pipeline/scripts/prepare_ben_ge_800.py

Converts extracted ben-ge-800 into format expected by BigEarthNetDataset loader.
Reads authentic Sentinel-1 VV/VH GeoTIFFs, Sentinel-2 B04/B03/B02 bands,
and authentic ESA WorldCover 10m land cover GeoTIFF ground truth masks.
Maps Corine Land Cover (CLC) 43 raw classes to official BigEarthNet 19 classes.
"""
import os, sys, csv, json, hashlib, argparse, logging, datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

CLC_19_CLASSES = [
    "Urban fabric","Industrial or commercial units","Arable land","Permanent crops",
    "Pastures","Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas","Broad-leaved forest","Coniferous forest","Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation","Transitional woodland, shrub",
    "Beaches, dunes, sands","Inland wetlands","Coastal wetlands","Inland waters","Marine waters"
]

CLC_43_TO_19 = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Pastures": "Pastures",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grassland": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Bare rock": "Natural grassland and sparsely vegetated areas",
    "Burnt areas": "Natural grassland and sparsely vegetated areas",
    "Glaciers and perpetual snow": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub": "Transitional woodland, shrub",
    "Transitional woodland, shrub": "Transitional woodland, shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Inland marshes": "Inland wetlands",
    "Peatbogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Intertidal flats": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
    "Mineral extraction sites": "Industrial or commercial units",
    "Dump sites": "Industrial or commercial units",
    "Construction sites": "Urban fabric",
    "Green urban areas": "Urban fabric",
    "Sport and leisure facilities": "Urban fabric",
    "Road and rail networks and associated land": "Urban fabric",
    "Port areas": "Industrial or commercial units",
    "Airports": "Industrial or commercial units",
    "Annual crops associated with permanent crops": "Complex cultivation patterns",
}

def assign_split(patch_id):
    h = int(hashlib.md5(patch_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")

def load_s2_rgb(s2_patch_dir):
    """Load B04/B03/B02 GeoTIFFs, percentile-stretch to uint8 PNG [H,W,3]."""
    rgb = []
    for band in ["B04","B03","B02"]:
        candidates = list(s2_patch_dir.glob(f"*_{band}.tif"))
        if not candidates:
            raise FileNotFoundError(f"Missing {band} in {s2_patch_dir}")
        with rasterio.open(candidates[0]) as src:
            arr = src.read(1).astype(np.float32)
        valid = arr[arr > 0]
        p2, p98 = (np.percentile(valid,[2,98]) if valid.size > 0 else (0,10000))
        arr = np.clip((arr-p2)/(p98-p2+1e-6)*255.0,0,255).astype(np.uint8)
        rgb.append(arr)
    return np.stack(rgb,axis=2)

def load_s1_sar(s1_patch_dir):
    """Load VV+VH float32 dB GeoTIFFs -> shape [H,W,2]."""
    sar = []
    for pol in ["VV","VH"]:
        candidates = list(s1_patch_dir.glob(f"*_{pol}.tif"))
        if not candidates:
            raise FileNotFoundError(f"Missing {pol} in {s1_patch_dir}")
        with rasterio.open(candidates[0]) as src:
            sar.append(src.read(1).astype(np.float32))
    return np.stack(sar,axis=2)

def load_worldcover_targets(wc_path):
    """
    Load authentic ESA WorldCover 10m land cover GeoTIFF.
    Maps classes:
      - Channel 0: Built-up (class 50)
      - Channel 1: Water bodies & wetlands (classes 80, 90)
      - Channel 2: Vegetation (tree 10, shrubland 20, grassland 30, cropland 40)
    Returns float32 array [H, W, 3] with binary values in {0.0, 1.0}.
    """
    with rasterio.open(wc_path) as src:
        wc = src.read(1)
    h, w = wc.shape
    targets = np.zeros((h, w, 3), dtype=np.float32)
    targets[..., 0] = (wc == 50).astype(np.float32)
    targets[..., 1] = np.isin(wc, [80, 90]).astype(np.float32)
    targets[..., 2] = np.isin(wc, [10, 20, 30, 40]).astype(np.float32)
    return targets

def load_labels(labels_json):
    with open(labels_json,"r",encoding="utf-8") as f:
        meta = json.load(f)
    raw_labels = meta.get("labels", meta.get("class_labels", []))
    mapped = []
    for rl in raw_labels:
        if rl in CLC_19_CLASSES:
            mapped.append(rl)
        elif rl in CLC_43_TO_19:
            mapped.append(CLC_43_TO_19[rl])
    return list(dict.fromkeys(mapped))

def build_caption(labels):
    if not labels:
        return "A co-registered Sentinel-1/2 image patch over Europe."
    return (f"A co-registered Sentinel-1/2 image patch showing {', '.join(labels[:3])}. "
            "Acquired over Europe 2017-2018 (BigEarthNet-MM).")

def prepare(src_root, dst_root, max_patches=None):
    assert HAS_RASTERIO, "pip install rasterio required"
    s2_root = src_root/"sentinel-2"
    s1_root = src_root/"sentinel-1"
    wc_root = src_root/"esaworldcover"
    meta_csv = src_root/"ben-ge-800_meta.csv"
    patch_pairs = {}
    if meta_csv.exists():
        with open(meta_csv,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                patch_pairs[row["patch_id"]] = row["patch_id_s1"]
        log.info(f"Loaded {len(patch_pairs)} patch pairs from meta CSV")
    s2_patches = sorted([d for d in s2_root.iterdir() if d.is_dir()])
    log.info(f"Found {len(s2_patches)} S2 patch dirs")
    if max_patches:
        s2_patches = s2_patches[:max_patches]
    annotations = {"train":[],"val":[],"test":[]}
    stats = {"ok":0,"skipped":0,"missing_s1":0,"missing_wc":0}
    for s2_dir in s2_patches:
        pid = s2_dir.name
        split = assign_split(pid)
        s1_id = patch_pairs.get(pid)
        if s1_id is None:
            stats["missing_s1"]+=1; continue
        s1_dir = s1_root/s1_id
        if not s1_dir.exists():
            stats["missing_s1"]+=1; continue

        wc_cand = wc_root/f"{pid}_esaworldcover.tif"
        if not wc_cand.exists():
            stats["missing_wc"]+=1; continue

        s2o_dir = dst_root/split/"s2_rgb"; s2o_dir.mkdir(parents=True,exist_ok=True)
        s1o_dir = dst_root/split/"s1_sar"; s1o_dir.mkdir(parents=True,exist_ok=True)
        wco_dir = dst_root/split/"target_masks"; wco_dir.mkdir(parents=True,exist_ok=True)

        s2_out = s2o_dir/f"{pid}.png"
        s1_out = s1o_dir/f"{s1_id}.npy"
        wc_out = wco_dir/f"{pid}.npy"

        try:
            if not s2_out.exists():
                Image.fromarray(load_s2_rgb(s2_dir)).save(s2_out)
            if not s1_out.exists():
                np.save(s1_out, load_s1_sar(s1_dir))
            if not wc_out.exists():
                np.save(wc_out, load_worldcover_targets(wc_cand))
            lj = list(s2_dir.glob("*_labels_metadata.json"))
            labels = load_labels(lj[0]) if lj else []
            annotations[split].append({
                "patch_id":pid,"s1_patch_id":s1_id,
                "s2_image_path":str(s2_out.relative_to(dst_root)),
                "s1_sar_path":str(s1_out.relative_to(dst_root)),
                "target_mask_path":str(wc_out.relative_to(dst_root)),
                "caption":build_caption(labels),"clc_labels":labels,"split":split,
                "source":"ben-ge-800",
                "provenance":"Zenodo DOI 10.5281/zenodo.12941231; Mommert et al 2023; ESA WorldCover 10m v100"
            })
            stats["ok"]+=1
        except Exception as e:
            log.warning(f"  skip {pid}: {e}"); stats["skipped"]+=1
    for split,ann in annotations.items():
        sp_dir = dst_root/split; sp_dir.mkdir(parents=True,exist_ok=True)
        ann_path = sp_dir/f"{split}_annotations.json"
        with open(ann_path,"w",encoding="utf-8") as f:
            json.dump(ann,f,indent=2)
        log.info(f"  {split}: {len(ann)} samples -> {ann_path}")
    meta = {
        "dataset_name":"ben-ge-800 (BigEarthNet-MM subset with ESA WorldCover ground truth)",
        "source":"Mommert et al 2023 / Sumbul et al 2021 / European Space Agency",
        "zenodo_doi":"10.5281/zenodo.12941231","license":"BigEarthNet License / CC BY 4.0",
        "modality":"Sentinel-1 SAR VV/VH float32 dB npy + Sentinel-2 B04B03B02 uint8 png + ESA WorldCover 10m ground truth",
        "split_method":"MD5-hash deterministic 70/15/15",
        "clc_classes":CLC_19_CLASSES,"stats":stats,
        "s2_preprocessing":"B04/B03/B02 2nd-98th percentile stretch to uint8",
        "s1_preprocessing":"VV/VH float32 dB saved as-is [H,W,2]",
        "target_mask_definition":"Channel 0: Built-up (ESA WorldCover 50), Channel 1: Water/Wetland (ESA WorldCover 80, 90), Channel 2: Vegetation (ESA WorldCover 10, 20, 30, 40)",
        "date_prepared":datetime.datetime.utcnow().isoformat()
    }
    with open(dst_root/"dataset_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    log.info(f"Done: ok={stats['ok']} skipped={stats['skipped']} missing_s1={stats['missing_s1']} missing_wc={stats['missing_wc']}")
    return stats

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",required=True)
    p.add_argument("--dst",required=True)
    p.add_argument("--max-patches",type=int,default=None)
    a = p.parse_args()
    prepare(Path(a.src),Path(a.dst),a.max_patches)
