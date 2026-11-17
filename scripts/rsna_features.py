#!/usr/bin/env python3
"""Second, independent dataset: RSNA Pneumonia Detection Challenge (adult chest radiographs, NIH).

    python scripts/rsna_features.py --zip rsna.zip --out cache/rsna_resnet50_224.npz

Binary task matching the Kermany benchmark: 'Normal' (0) versus 'Lung Opacity' (1, pneumonia);
images labelled 'No Lung Opacity / Not Normal' are excluded. The dataset has one image per patient,
so a stratified image split is a patient split. RSNA has no labelled public test set, so a
stratified 20% test set is held out with a fixed seed; prepare() then splits the rest into
optimisation and validation subsets exactly as for Kermany.

Features: frozen ImageNet ResNet-50 (IMAGENET1K_V2), identical preprocessing to
extract_cnn_features.py (grayscale -> 3 channels, 224x224 INTER_AREA, ImageNet normalisation).
DICOM pixel data are min-max scaled to 0-255 per image (MONOCHROME1 inverted).
"""
import argparse
import time
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True, help="competition archive from Kaggle")
ap.add_argument("--out", default="cache/rsna_resnet50_224.npz")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--test_fraction", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

z = zipfile.ZipFile(args.zip)
cls = pd.read_csv(z.open("stage_2_detailed_class_info.csv")).drop_duplicates("patientId")
cls = cls[cls["class"].isin(["Normal", "Lung Opacity"])].reset_index(drop=True)
ids = cls["patientId"].to_numpy()
y = (cls["class"] == "Lung Opacity").to_numpy().astype(np.int64)
tr_i, te_i = train_test_split(np.arange(len(ids)), test_size=args.test_fraction, stratify=y,
                              random_state=args.seed)
print(f"{len(ids)} images: {np.bincount(y)} (normal, opacity); train {len(tr_i)}, test {len(te_i)}")

weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
net = torchvision.models.resnet50(weights=weights)
net.fc = torch.nn.Identity()
net.eval()
torch.set_num_threads(max(1, torch.get_num_threads()))
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(pid):
    ds = pydicom.dcmread(BytesIO(z.read(f"stage_2_train_images/{pid}.dcm")))
    a = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
    return cv2.resize(a.astype(np.uint8), (args.size, args.size), interpolation=cv2.INTER_AREA)


def feats(idx):
    out = np.empty((len(idx), 2048), np.float32)
    for s in range(0, len(idx), args.batch):
        imgs = [load(ids[i]) for i in idx[s:s + args.batch]]
        x = torch.from_numpy(np.stack(imgs)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            out[s:s + len(imgs)] = net((x - mean) / std).numpy()
        print(f"  {s + len(imgs)}/{len(idx)}", end="\r", flush=True)
    print()
    return out


t0 = time.time()
F_tr, F_te = feats(tr_i), feats(te_i)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, F_train=F_tr, y_train=y[tr_i], F_test=F_te, y_test=y[te_i],
         train_files=np.array([f"rsna/{ids[i]}.dcm" for i in tr_i]),
         test_files=np.array([f"rsna/{ids[i]}.dcm" for i in te_i]),
         weights=np.array(str(weights)), torch=np.array(torch.__version__))
print(f"saved {args.out} in {time.time() - t0:.0f}s")
