#!/usr/bin/env python3
"""Summarise results/<exp>: mean +- SD of test metrics, median training fitness, validation MCC,
Friedman test over gradient-free optimisers (test MCC), and Holm-adjusted Wilcoxon tests of
every method against the hybrid on the three endpoints."""
import glob
import json
import sys

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

exp = sys.argv[1] if len(sys.argv) > 1 else "rsna_head"
R = {}
for f in glob.glob(f"results/{exp}/runs/*/run_*.json"):
    r = json.load(open(f))
    R.setdefault(r["algorithm"], {})[r["run"]] = r
ids = sorted(set.intersection(*[set(v) for v in R.values()]))
GF = [a for a in R if "best_fitness" in next(iter(R[a].values()))]


def v(a, k, m="mcc"):
    if k == "fit":
        return np.array([R[a][i]["best_fitness"] for i in ids])
    if k == "time":
        return np.array([R[a][i]["training_time"] for i in ids])
    return np.array([R[a][i][k][m] for i in ids])


print(f"{exp}: {len(ids)} paired runs, {len(R)} methods")
print(f"{'method':16s} {'acc':>12s} {'sens':>12s} {'spec':>12s} {'MCC':>13s} {'AUC':>13s} {'valMCC':>7s} {'fit_med':>8s}")
order = sorted(R, key=lambda a: -v(a, "test").mean())
for a in order:
    cells = []
    for m, k in [("accuracy", 100), ("recall", 100), ("specificity", 100), ("mcc", 1), ("auc", 1)]:
        x = v(a, "test", m) * k
        cells.append(f"{x.mean():.2f}±{x.std(ddof=1):.2f}" if k == 100 else f"{x.mean():.3f}±{x.std(ddof=1):.3f}")
    fit = f"{np.median(v(a, 'fit')):.3f}" if a in GF else "-"
    print(f"{a:16s} " + " ".join(f"{c:>12s}" for c in cells) + f" {v(a, 'val').mean():7.3f} {fit:>8s}")

st = friedmanchisquare(*[v(a, "test") for a in GF])
print(f"\nFriedman over {len(GF)} gradient-free optimisers (test MCC): chi2={st.statistic:.1f}, p={st.pvalue:.2g}")
ranks = np.argsort(np.argsort(-np.stack([v(a, "test") for a in GF]), axis=0), axis=0) + 1
for a, r in sorted(zip(GF, ranks.mean(1)), key=lambda t: t[1]):
    print(f"  mean rank {r:5.2f}  {a}")

if "Hybrid" in R:
    print("\nAgainst the hybrid (Holm-adjusted within each endpoint; wins = runs better than the hybrid)")
    for k in ["fit", "val", "test"]:
        others = [a for a in (GF if k == "fit" else R) if a != "Hybrid"]
        ps = [wilcoxon(v(a, k), v("Hybrid", k)).pvalue for a in others]
        o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
        for j, i in enumerate(o):
            run = max(run, min(1.0, ps[i] * (len(ps) - j))); adj[i] = run
        print(f"  [{k}]")
        for a, p in zip(others, adj):
            w = (v(a, k) < v("Hybrid", k)).sum() if k == "fit" else (v(a, k) > v("Hybrid", k)).sum()
            print(f"    {a:16s} wins {w:2d}/{len(ids)}  p_Holm={p:.2g}")
