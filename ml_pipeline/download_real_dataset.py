"""
SatQuery AI - Real Satellite Dataset Downloader
Phase 3: Replace synthetic training data with real, labelled EO dataset.

Dataset: LoveDA (Land-cover cOVErage Dataset from Aerial images)
Source: https://zenodo.org/record/5706578
License: CC BY 4.0

Class mapping LoveDA 7-class -> SatSegNet 6-class:
  0: Background  -> 0: Background
  1: Building    -> 3: Built-up
  2: Road        -> 3: Built-up
  3: Water       -> 1: Water Bodies
  4: Barren      -> 4: Bare Soil
  5: Forest      -> 2: Vegetation
  6: Agriculture -> 2: Vegetation

Usage:
  python ml_pipeline/download_real_dataset.py
"""

import os, sys, urllib.request, zipfile, json, numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "real_dataset"

LOVEDA_URLS = {
    "train": "https://zenodo.org/record/5706578/files/Train.zip?download=1",
    "val":   "https://zenodo.org/record/5706578/files/Val.zip?download=1",
}

LOVEDA_TO_SATSEGNET = {0:0, 1:3, 2:3, 3:1, 4:4, 5:2, 6:2}
TARGET_SIZE = (128, 128)
MAX_TRAIN = 2000
MAX_VAL = 400

def download_file(url, dest, label):
    if dest.exists():
        print(f"  [CACHED] {label}", flush=True)
        return True
    print(f"  [DOWNLOAD] {label}...", flush=True)
    try:
        def prog(n, bs, ts):
            if ts > 0: print(f"\r    {min(100,int(n*bs*100/ts))}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=prog)
        print(f"\r    Done ({dest.stat().st_size//1024//1024} MB)", flush=True)
        return True
    except Exception as e:
        print(f"\n  [WARN] {e}", flush=True)
        return False

def remap_mask(arr):
    out = np.zeros_like(arr, dtype=np.uint8)
    for s,d in LOVEDA_TO_SATSEGNET.items():
        out[arr==s] = d
    return out

def process_split(img_dir, mask_dir, split, max_n):
    img_out = OUTPUT_DIR/split/"images"; img_out.mkdir(parents=True, exist_ok=True)
    msk_out = OUTPUT_DIR/split/"masks";  msk_out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.tif")))
    count = 0
    for ip in imgs:
        if count >= max_n: break
        mp = mask_dir/ip.name
        if not mp.exists(): mp = mask_dir/(ip.stem+".png")
        if not mp.exists(): continue
        try:
            img = Image.open(ip).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
            m   = np.array(Image.open(mp))
            if m.ndim==3: m=m[:,:,0]
            m   = Image.fromarray(remap_mask(m)).resize(TARGET_SIZE, Image.NEAREST)
            img.save(img_out/f"{split}_tile_{count:05d}.png")
            m.save(msk_out/f"{split}_mask_{count:05d}.png")
            count += 1
        except: continue
    return count

def find_dirs(base):
    for img_n in ["images_png","images","img","Images"]:
        for msk_n in ["masks_png","masks","ann","Masks","gt"]:
            ig = next(base.rglob(img_n), None)
            mg = next(base.rglob(msk_n), None)
            if ig and mg: return ig, mg
    cands=[d for d in base.rglob("*") if d.is_dir() and len(list(d.glob("*.png")))>50]
    if len(cands)>=2: return cands[0], cands[1]
    return None, None

def main():
    print("="*60, flush=True)
    print("SatQuery AI: LoveDA Real Dataset Download", flush=True)
    tmp = ROOT/"_tmp_downloads"; tmp.mkdir(exist_ok=True)
    totals = {"train":0,"val":0}
    for split, url in LOVEDA_URLS.items():
        zp = tmp/f"{split}.zip"
        if not download_file(url, zp, split): continue
        ed = tmp/split
        if not ed.exists():
            with zipfile.ZipFile(zp,"r") as z: z.extractall(ed)
        ig,mg = find_dirs(ed)
        if not ig or not mg:
            print(f"  [WARN] dirs not found in {split}", flush=True); continue
        max_n = MAX_TRAIN if split=="train" else MAX_VAL
        n = process_split(ig, mg, split, max_n)
        totals[split] = n
        print(f"  {n} {split} samples", flush=True)

    manifest = {"dataset":"LoveDA","source":"https://zenodo.org/record/5706578",
                "license":"CC BY 4.0","class_mapping":LOVEDA_TO_SATSEGNET,
                "satsegnet_classes":{"0":"Background","1":"Water Bodies","2":"Vegetation","3":"Built-up","4":"Bare Soil","5":"Cloud"},
                "train_samples":totals["train"],"val_samples":totals["val"],
                "target_size":list(TARGET_SIZE)}
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone! Train:{totals['train']} Val:{totals['val']}", flush=True)
    print("Next: python ml_pipeline/train_segmentation.py --data_dir ml_pipeline/real_dataset --epochs 20 --batch_size 8", flush=True)

if __name__=="__main__":
    main()
