#!/usr/bin/env python3
"""Confirmatory external test on VinDr-PCXR (PediCXR), pre-registration amendment 2.

Runs entirely on the local computer. Reads the VinDr-PCXR DICOM files and label CSVs, applies the stored
models (trained on Kermany only, fixed before the test) and writes ONLY aggregate results
(counts, means and confidence intervals) to results_pedicxr.json. No image, label or identifier is written.

    python pedicxr_run.py selftest                       # checks the models (no VinDr-PCXR data needed)
    python pedicxr_run.py check --data /path/to/vindr-pcxr   # finds files and labels, prints counts only
    python pedicxr_run.py run   --data /path/to/vindr-pcxr   # the test itself (run once)
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M = HERE / "models"
EXPS = {"frozen": ["ext_final_frozen"], "finetuned": ["ext_final_ft0", "ext_final_ft1", "ext_final_ft2"]}
GF = ["APH", "APH-noO3", "APH-O3only", "L-SHADE"]
REF = ["LogReg", "MLP-Adam"]
B, SEOI = 5000, 0.05


# ------------------------------------------------------------------ models (pure numpy)
def head_proba(theta, X, sizes=(2048, 16, 2)):
    i, Ws, bs = 0, [], []
    for a, b in zip(sizes[:-1], sizes[1:]):
        Ws.append(theta[i:i + a * b].reshape(a, b)); i += a * b
    for b in sizes[1:]:
        bs.append(theta[i:i + b]); i += b
    A = X.astype(np.float32)
    for k, (W, b) in enumerate(zip(Ws, bs)):
        A = A @ W + b
        if k < len(Ws) - 1:
            A = np.maximum(A, 0)
    Z = A.astype(np.float64); Z -= Z.max(1, keepdims=True); P = np.exp(Z)
    return (P / P.sum(1, keepdims=True))[:, 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_all(exp, X_std):
    """Probabilities (runs x images) of every stored model of one experiment on standardised features."""
    h = np.load(M / exp / "heads.npz"); r = np.load(M / exp / "references.npz")
    out = {a: np.array([head_proba(t, X_std) for t in h[a.replace("-", "_")]]) for a in GF}
    out["LogReg"] = np.array([sigmoid(X_std @ w + b) for w, b in zip(r["LogReg_w"], r["LogReg_b"])])
    out["MLP-Adam"] = np.array([sigmoid(np.maximum(X_std @ W1 + b1, 0) @ w2 + b2)
                                for W1, b1, w2, b2 in zip(r["MLP_W1"], r["MLP_b1"], r["MLP_w2"], r["MLP_b2"])])
    return out


def selftest():
    for exps in EXPS.values():
        for e in exps:
            c = np.load(M / e / "kermany_check.npz"); P = predict_all(e, c["X_test"])
            for a in GF + REF:
                d = np.abs(P[a] - c[a.replace("-", "_")]).max()
                assert d < 0.01 and ((P[a] >= 0.5) == (c[a.replace("-", "_")] >= 0.5)).all(), (e, a, d)
    print("selftest passed: stored models reproduce their Kermany predictions")


# ------------------------------------------------------------------ data
def find_labels(root, label_csvs):
    files = label_csvs or [f for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
                           if "label" in os.path.basename(f).lower() and "image" in os.path.basename(f).lower()]
    if not files:
        sys.exit("No label CSV found (expected names like image_labels_train.csv). Pass --labels explicitly.")
    rows, cols = {}, set()
    for f in files:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            idcol = next((c for c in rd.fieldnames if c.lower() in ("image_id", "imageid", "image")), None)
            if idcol is None:
                sys.exit(f"{f}: no image_id column; columns are {rd.fieldnames}")
            for row in rd:
                labs = {k: v for k, v in row.items() if k != idcol and k.lower() not in ("rad_id", "rad_ids", "radiologist")}
                cols |= set(labs)
                rows.setdefault(row[idcol].strip(), []).append(labs)
    pneu = sorted(c for c in cols if "pneumonia" in c.lower())
    nof = next((c for c in cols if c.lower().replace("_", " ").strip() == "no finding"), None)
    if not pneu or nof is None:
        sys.exit(f"Label columns not recognised. Found: {sorted(cols)}")
    ids, y = [], []
    for img, lab_rows in rows.items():
        on = lambda r, c: str(r.get(c, "0")).strip() not in ("", "0", "0.0", "False", "false")
        if any(on(r, c) for r in lab_rows for c in pneu):
            ids.append(img); y.append(1)
        elif all(on(r, nof) and not any(on(r, c) for c in cols if c != nof) for r in lab_rows):
            ids.append(img); y.append(0)
    return files, pneu, nof, ids, np.array(y)


def find_dicoms(root):
    paths = {}
    for ext in ("dicom", "dcm", "DCM", "DICOM"):
        for p in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            paths[Path(p).stem] = p
    return paths


def pixels(path):
    import pydicom
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    a = ((a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0).astype(np.uint8)
    return a, str(getattr(ds, "PatientID", "") or "")


def prepare(root, label_csvs):
    files, pneu, nof, ids, y = find_labels(root, label_csvs)
    paths = find_dicoms(root)
    keep = [k for k, i in enumerate(ids) if i in paths]
    missing = len(ids) - len(keep)
    ids = [ids[k] for k in keep]; y = y[keep]
    return {"label_files": [os.path.basename(f) for f in files], "pneumonia_columns": pneu, "no_finding_column": nof,
            "ids": ids, "y": y, "paths": [paths[i] for i in ids], "missing_dicom": missing}


# ------------------------------------------------------------------ features
def features(paths):
    import cv2
    import torch
    import torchvision
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    nets = {"frozen": torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)}
    nets["frozen"].fc = torch.nn.Identity()
    for s in (0, 1, 2):
        n = torchvision.models.resnet50(weights=None); n.fc = torch.nn.Linear(2048, 2)
        n.load_state_dict(torch.load(M / f"resnet50_finetuned_s{s}.pt", map_location="cpu")); n.fc = torch.nn.Identity()
        nets[f"ft{s}"] = n
    for n in nets.values():
        n.eval()
    F = {k: [] for k in nets}; pids = []; t0 = time.time()
    for s in range(0, len(paths), 16):
        imgs, i224, i256 = [], [], []
        for p in paths[s:s + 16]:
            a, pid = pixels(p); pids.append(pid)
            i224.append(cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA))
            i256.append(cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA))
        x224 = (torch.from_numpy(np.stack(i224)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1) - mean) / std
        x256 = torch.from_numpy(np.stack(i256)).float().div(255).unsqueeze(1)
        x256 = torch.nn.functional.interpolate(x256, size=224, mode="bilinear", antialias=True, align_corners=False)
        x256 = (x256.repeat(1, 3, 1, 1) - mean) / std
        with torch.inference_mode():
            F["frozen"].append(nets["frozen"](x224).numpy())
            for k in ("ft0", "ft1", "ft2"):
                F[k].append(nets[k](x256).numpy())
        done = min(s + 16, len(paths))
        print(f"  features {done}/{len(paths)}  ({(time.time() - t0) / done:.2f} s/image)", end="\r", flush=True)
    print()
    return {k: np.concatenate(v).astype(np.float32) for k, v in F.items()}, np.array(pids)


# ------------------------------------------------------------------ analysis (as pre-registered)
def mcc_rows(P, yy):
    tp = (P & yy).sum(-1); tn = (~P & ~yy).sum(-1); fp = (P & ~yy).sum(-1); fn = (~P & yy).sum(-1)
    den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros(np.broadcast(tp, den).shape), where=den > 0)


def analyse(F, y, pids):
    from sklearn.metrics import roc_auc_score
    y = y.astype(bool)
    unit = "patient" if pids.size and all(pids) and len(set(pids)) < len(pids) else "image"
    groups_key = pids if unit == "patient" else np.arange(len(y)).astype(str)
    groups = [np.flatnonzero(groups_key == u) for u in np.unique(groups_key)]
    rep = {"frozen": [], "finetuned": []}
    for r, exps in EXPS.items():
        for e in exps:
            feat = F["frozen"] if r == "frozen" else F["ft" + e[-1]]
            sc = np.load(M / e / "scaler.npz")
            rep[r].append(predict_all(e, ((feat - sc["mean"]) / sc["scale"]).astype(np.float32)))

    def delta(blocks, idx=None):
        yy = y if idx is None else y[idx]; v = []
        for b in blocks:
            f_ = b["APH"] if idx is None else b["APH"][:, idx]; n_ = b["APH-noO3"] if idx is None else b["APH-noO3"][:, idx]
            v.append((mcc_rows(f_ >= 0.5, yy) - mcc_rows(n_ >= 0.5, yy)).mean())
        return float(np.mean(v))

    D = {r: delta(rep[r]) for r in rep}; I = D["finetuned"] - D["frozen"]
    rng = np.random.default_rng(20261123); bF, bT, bI = [], [], []
    for k in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        f_, t_ = delta(rep["frozen"], idx), delta(rep["finetuned"], idx); bF.append(f_); bT.append(t_); bI.append(t_ - f_)
        if k % 500 == 0:
            print(f"  bootstrap {k}/{B}", end="\r", flush=True)
    print()
    ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    lo, hi = ci(bI)
    if lo > 0:
        verdict = "H1 supported" + (", meaningful" if I >= SEOI else ", but I < 0.05")
    elif hi < 0:
        verdict = "reversed" + (", meaningful (I <= -0.05)" if I <= -SEOI else "")
    elif lo > -SEOI and hi < SEOI:
        verdict = "no meaningful interaction"
    else:
        verdict = "inconclusive"
    sec = {}
    for r, blocks in rep.items():
        for a in GF + REF:
            P = np.concatenate([b[a] for b in blocks]); pr = P >= 0.5
            sec[f"{r}/{a}"] = {"mcc": float(mcc_rows(pr, y).mean()), "auc": float(np.mean([roc_auc_score(y, p) for p in P])),
                               "sens": float((pr & y).sum(1).mean() / y.sum()), "spec": float((~pr & ~y).sum(1).mean() / (~y).sum())}
    return {"n_images": int(len(y)), "class_counts": np.bincount(y.astype(int)).tolist(), "resampling_unit": unit,
            "n_units": len(groups), "Delta_frozen": D["frozen"], "Delta_frozen_CI": ci(bF),
            "Delta_finetuned": D["finetuned"], "Delta_finetuned_CI": ci(bT), "I": I, "I_CI": [lo, hi],
            "verdict": verdict, "secondary": sec}


# ------------------------------------------------------------------ main
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["selftest", "check", "run"])
ap.add_argument("--data"); ap.add_argument("--labels", nargs="*")
ap.add_argument("--out", default=str(HERE / "results_pedicxr.json"))
a = ap.parse_args()
selftest()
if a.step == "selftest":
    sys.exit(0)
if not a.data:
    sys.exit("--data is required")
D = prepare(a.data, a.labels)
summary = {"label_files": D["label_files"], "pneumonia_columns": D["pneumonia_columns"],
           "no_finding_column": D["no_finding_column"], "included_images": len(D["ids"]),
           "class_counts": np.bincount(D["y"], minlength=2).tolist(), "labelled_but_no_dicom": D["missing_dicom"]}
print(json.dumps(summary, indent=1))
if a.step == "check":
    sys.exit(0)
if Path(a.out).exists():
    sys.exit(f"{a.out} already exists: the external test is run only once. Remove it only if the run failed before finishing.")
F, pids = features(D["paths"])
res = analyse(F, D["y"], pids); res["data_summary"] = summary
res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
Path(a.out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "secondary"}, indent=2))
print("Aggregate results written to", a.out, "- send this file only.")
