#!/usr/bin/env python3
"""Cross-dataset generalisation: in-domain versus cross-dataset test performance of the same methods.

Kermany->Kermany: results/cnn_head (+ L-SHADE from cnn_redesign), runs 0-19
RSNA->RSNA:       results/rsna_head, runs 0-19
Kermany->RSNA:    results/cross_K2R      RSNA->Kermany: results/cross_R2K"""
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


KK = load(f"{BASE}/cnn_head"); KK["L-SHADE"] = load(f"{BASE}/cnn_redesign")["L-SHADE"]
RR = load("results/rsna_adult")
KR = load("results/cross_K2R_adult"); RK = load("results/cross_R2K_adult")
methods = ["LogReg", "MLP-Adam", "L-SHADE", "GA", "SDA", "Hybrid"]
ids = list(range(20))


def v(R, a, m):
    return np.array([R[a][i]["test"][m] for i in ids])


for title, ind, cross in (("tested on RSNA: trained on RSNA vs trained on Kermany", RR, KR),
                          ("tested on Kermany: trained on Kermany vs trained on RSNA", KK, RK)):
    print(title)
    print(f"  {'method':10s} {'acc in':>7s} {'acc x':>7s} {'MCC in':>7s} {'MCC x':>7s} {'AUC in':>7s} {'AUC x':>7s} "
          f"{'sens x':>7s} {'spec x':>7s} {'p(MCC)':>8s}")
    for a in methods:
        p = wilcoxon(v(ind, a, "mcc"), v(cross, a, "mcc")).pvalue if np.any(v(ind, a, "mcc") != v(cross, a, "mcc")) else 1.0
        print(f"  {a:10s} {100 * v(ind, a, 'accuracy').mean():7.2f} {100 * v(cross, a, 'accuracy').mean():7.2f} "
              f"{v(ind, a, 'mcc').mean():7.3f} {v(cross, a, 'mcc').mean():7.3f} {v(ind, a, 'auc').mean():7.3f} "
              f"{v(cross, a, 'auc').mean():7.3f} {100 * v(cross, a, 'recall').mean():7.1f} "
              f"{100 * v(cross, a, 'specificity').mean():7.1f} {p:8.2g}")
    order = sorted(methods, key=lambda a: -v(cross, a, "mcc").mean())
    print("  cross-dataset ranking by MCC: " + " > ".join(order))
