#!/usr/bin/env python3
"""Grad-CAM error analysis of the fine-tuned ResNet-50 (seed 0) on the Kermany test set (exploratory).

For every test image it computes the Grad-CAM map of the pneumonia logit at layer4 and the share of the map
that falls outside the central box (the middle 70% of rows and columns, which contains the lung fields in
these centred frontal radiographs). A large outside share means that the decision rests on borders,
labels or the body outline rather than on the lungs. The share is compared between true positives,
true negatives, false positives and false negatives (threshold 0.5), and a figure shows the most confident
false positives next to typical correct cases.

    python scripts/gradcam_errors.py --data /home/user/ft/cxr256.npz --ft ft_dev/s0 --out results/gradcam
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--ft", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(2)
d = np.load(a.data); X, y = d["X_test"], d["y_test"].astype(int)
net = torchvision.models.resnet50(); net.fc = nn.Linear(2048, 2)
net.load_state_dict(torch.load(os.path.join(a.ft, "best.pt"), map_location="cpu")); net.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
acts = {}
net.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))


def prep(idx):
    x = torch.from_numpy(X[idx]).float().div(255).unsqueeze(1)
    x = nn.functional.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD, x[:, 0].numpy()


cams, probs, imgs = [], [], []
for s in range(0, len(y), 16):
    idx = np.arange(s, min(s + 16, len(y))); x, raw = prep(idx)
    logits = net(x); A = acts["a"]
    g = torch.autograd.grad(logits[:, 1].sum(), A)[0]
    w = g.mean((2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=224, mode="bilinear", align_corners=False)[:, 0].detach().numpy()
    cams.append(cam); probs.append(torch.softmax(logits, 1)[:, 1].detach().numpy()); imgs.append(raw)
cam = np.concatenate(cams); p = np.concatenate(probs); img = np.concatenate(imgs)
stored = np.array(json.load(open(os.path.join(a.ft, "metrics.json")))["test_proba"])
assert ((p >= 0.5) == (stored >= 0.5)).all(), "fine-tuned model does not reproduce its stored predictions"

lo, hi = int(0.15 * 224), int(0.85 * 224)
tot = cam.reshape(len(cam), -1).sum(1) + 1e-12
inside = cam[:, lo:hi, lo:hi].reshape(len(cam), -1).sum(1)
outside = 1 - inside / tot
pred = (p >= 0.5).astype(int)
grp = {"TP": (pred == 1) & (y == 1), "TN": (pred == 0) & (y == 0), "FP": (pred == 1) & (y == 0), "FN": (pred == 0) & (y == 1)}
uniform = 1 - (hi - lo) ** 2 / 224 ** 2
summary = {"n_test": int(len(y)), "uniform_outside_share": uniform, "box": [lo, hi],
           "groups": {k: {"n": int(m.sum()), "outside_share_mean": float(outside[m].mean()) if m.any() else None,
                          "outside_share_median": float(np.median(outside[m])) if m.any() else None,
                          "prob_mean": float(p[m].mean()) if m.any() else None} for k, m in grp.items()}}
from scipy.stats import mannwhitneyu
if grp["FP"].any():
    summary["FP_vs_TP_mannwhitney_p"] = float(mannwhitneyu(outside[grp["FP"]], outside[grp["TP"]]).pvalue)
    summary["FP_vs_TN_mannwhitney_p"] = float(mannwhitneyu(outside[grp["FP"]], outside[grp["TN"]]).pvalue)
json.dump(summary, open(os.path.join(a.out, "gradcam_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

# Figure: 4 most confident false positives (top row), 2 typical true positives and 2 true negatives (bottom row)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fp = np.flatnonzero(grp["FP"]); fp = fp[np.argsort(-p[fp])][:4]
tp = np.flatnonzero(grp["TP"]); tp = tp[np.argsort(np.abs(p[tp] - np.median(p[tp])))][:2]
tn = np.flatnonzero(grp["TN"]); tn = tn[np.argsort(np.abs(p[tn] - np.median(p[tn])))][:2]
sel = [("False positive", i) for i in fp] + [("True positive", i) for i in tp] + [("True negative", i) for i in tn]
fig, ax = plt.subplots(2, 4, figsize=(10, 5.6))
for k, (lab, i) in enumerate(sel):
    axk = ax.flat[k]; c = cam[i] / (cam[i].max() + 1e-12)
    axk.imshow(img[i], cmap="gray"); axk.imshow(c, cmap="jet", alpha=0.35)
    axk.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec="white", lw=0.8, ls="--"))
    axk.set_title(f"{lab}\np(pneumonia)={p[i]:.2f}, outside={outside[i]:.2f}", fontsize=8.5)
    axk.axis("off")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, f"gradcam_errors.{ext}"), dpi=200, bbox_inches="tight")
np.savez_compressed(os.path.join(a.out, "gradcam_outside_share.npz"), outside=outside, prob=p, y=y)
print("figure written")
