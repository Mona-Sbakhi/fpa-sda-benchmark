#!/usr/bin/env python3
"""Extract frozen ImageNet ResNet-50 features (2048-d global average pool) for every image.

    python scripts/extract_cnn_features.py --root /home/user/data/chest_xray --out cache/resnet50_224.npz

Images: grayscale -> 3 channels, resized to 224x224 (cv2.INTER_AREA), ImageNet normalisation.
Weights: torchvision ResNet50_Weights.IMAGENET1K_V2. No fine-tuning; the network is frozen.
File order matches fpasda.data.load_dataset (train + val merged, then test).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpasda.data import _list_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="cache/resnet50_224.npz")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

root = Path(args.root)
tr_p, tr_y = _list_split(root, "train")
va_p, va_y = _list_split(root, "val")
te_p, te_y = _list_split(root, "test")
tr_p, tr_y = tr_p + va_p, np.concatenate([tr_y, va_y])

weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
net = torchvision.models.resnet50(weights=weights)
net.fc = torch.nn.Identity()
net.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
net.to(dev)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def feats(paths):
    out = np.empty((len(paths), 2048), np.float32)
    for s in range(0, len(paths), args.batch):
        imgs = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (args.size, args.size),
                           interpolation=cv2.INTER_AREA) for p in paths[s:s + args.batch]]
        x = torch.from_numpy(np.stack(imgs)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            out[s:s + len(imgs)] = net(((x - mean) / std).to(dev)).cpu().numpy()
        print(f"  {s + len(imgs)}/{len(paths)}", end="\r", flush=True)
    print()
    return out


t0 = time.time()
F_tr, F_te = feats(tr_p), feats(te_p)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, F_train=F_tr, y_train=tr_y, F_test=F_te, y_test=te_y,
         train_files=np.array([str(p.relative_to(root)) for p in tr_p]),
         test_files=np.array([str(p.relative_to(root)) for p in te_p]),
         weights=np.array(str(weights)), torch=np.array(torch.__version__))
print(f"saved {args.out} in {time.time() - t0:.0f}s on {dev}")
