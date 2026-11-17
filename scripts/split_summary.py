#!/usr/bin/env python3
"""Split study: the same methods under three evaluation protocols on Kermany frozen ResNet-50 features.

A: official split (patient-grouped optimisation/validation, official test folder)  -- results/cnn_head, cnn_redesign
B: pooled images, random image-level 80/20 split (patients shared between training and test)
C: pooled images, patient-grouped 80/20 split
Reports mean test metrics per protocol and Wilcoxon tests (paired by run seed) for B-C and C-A."""
import glob
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(path):
    R = {}
    for f in glob.glob(f"{path}/runs/*/run_*.json"):
        r = json.load(open(f))
        R.setdefault(r["algorithm"], {})[r["run"]] = r
    return R


A = load(f"{BASE}/cnn_head")
A["L-SHADE"] = load(f"{BASE}/cnn_redesign")["L-SHADE"]
B = load("results/split_B_random_image")
C = load("results/split_C_random_patient")
methods = ["LogReg", "MLP-Adam", "L-SHADE", "GA", "SDA", "Hybrid"]
ids = list(range(20))


def v(R, a, m):
    return np.array([R[a][i]["test"][m] for i in ids])


print(f"{'method':10s} | {'acc A':>7s} {'acc C':>7s} {'acc B':>7s} | {'MCC A':>6s} {'MCC C':>6s} {'MCC B':>6s} | "
      f"{'B-C acc':>8s} {'p':>8s} | {'C-A acc':>8s} {'p':>8s}")
rows = {}
for a in methods:
    acc = {k: 100 * v(R, a, "accuracy") for k, R in (("A", A), ("B", B), ("C", C))}
    mcc = {k: v(R, a, "mcc") for k, R in (("A", A), ("B", B), ("C", C))}

    def p(x, y):
        return wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
    bc, ca = acc["B"] - acc["C"], acc["C"] - acc["A"]
    rows[a] = (acc, mcc)
    print(f"{a:10s} | {acc['A'].mean():7.2f} {acc['C'].mean():7.2f} {acc['B'].mean():7.2f} | "
          f"{mcc['A'].mean():6.3f} {mcc['C'].mean():6.3f} {mcc['B'].mean():6.3f} | "
          f"{bc.mean():+8.2f} {p(acc['B'], acc['C']):8.2g} | {ca.mean():+8.2f} {p(acc['C'], acc['A']):8.2g}")
for k in "ABC":
    order = sorted(methods, key=lambda a: -rows[a][1][k].mean())
    print(f"ranking by MCC, protocol {k}: " + " > ".join(order))
for name in ("split_B_random_image", "split_C_random_patient"):
    info = json.load(open(f"results/{name}/data_info.json"))
    print(name, {k: info[k] for k in ("n_opt", "n_val", "n_test", "test_images_with_patient_in_train")})
