#!/usr/bin/env python3
"""Aggregate run JSONs into tables, statistics and figures. Every number comes from the saved runs.

    python scripts/analyze.py results/pca_mlp_revised [--reference Hybrid]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpasda.stats import bootstrap_ci, friedman, nemenyi, wilcoxon_vs  # noqa: E402

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "auc"]

ap = argparse.ArgumentParser()
ap.add_argument("result_dir")
ap.add_argument("--reference", default="Hybrid")
ap.add_argument("--stat-metrics", nargs="*", default=["mcc", "accuracy", "auc", "balanced_accuracy"])
args = ap.parse_args()
root = Path(args.result_dir)
cfg = json.loads((root / "meta.json").read_text())["config"]
fig_dir = root / "figures"; fig_dir.mkdir(exist_ok=True)

runs = {}
for p in sorted((root / "runs").glob("*/run_*.json")):
    r = json.loads(p.read_text())
    runs.setdefault(r["algorithm"], {})[r["run"]] = r
order = [n for n in list(cfg["algorithms"]) + list(cfg.get("baselines", {})) if n in runs]
meta_algos = [n for n in cfg["algorithms"] if n in runs]

# ---- summary table --------------------------------------------------------------------------
rows = []
for n in order:
    rs = [runs[n][k] for k in sorted(runs[n])]
    row = {"algorithm": n, "runs": len(rs)}
    for m in METRICS + ["training_time"]:
        vals = np.array([r["test"].get(m, np.nan) if m != "training_time" else r["training_time"] for r in rs], float)
        if np.all(np.isnan(vals)):
            continue
        lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (vals[0], vals[0])
        row[f"{m}_mean"], row[f"{m}_sd"] = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        row[f"{m}_median"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = np.median(vals), lo, hi
    if "n_evals" in rs[0]:
        row["n_evals"] = int(np.median([r["n_evals"] for r in rs]))
        row["best_fitness_mean"] = float(np.mean([r["best_fitness"] for r in rs]))
        row["fitness_ce_mean"] = float(np.mean([r["fitness_ce"] for r in rs]))
        row["fitness_l2_mean"] = float(np.mean([r["fitness_l2"] for r in rs]))
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(root / "summary.csv", index=False)

def fmt(r, m, pct=True):
    k = 100 if pct else 1
    if f"{m}_mean" not in r or pd.isna(r.get(f"{m}_mean")):
        return "–"
    return f"{k*r[f'{m}_mean']:.2f} ± {k*r[f'{m}_sd']:.2f} [{k*r[f'{m}_ci_lo']:.2f}, {k*r[f'{m}_ci_hi']:.2f}]"

lines = [f"# Results: {cfg['name']}", "", cfg.get("description", ""), "",
         "Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.", "",
         "| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in summary.iterrows():
    lines.append(f"| {r['algorithm']} | {r['runs']} | {fmt(r,'accuracy')} | {fmt(r,'balanced_accuracy')} | "
                 f"{fmt(r,'precision')} | {fmt(r,'recall')} | {fmt(r,'specificity')} | {fmt(r,'f1')} | "
                 f"{fmt(r,'mcc',False)} | {fmt(r,'auc',False)} | {r['training_time_mean']:.1f} | "
                 f"{'' if pd.isna(r.get('n_evals', np.nan)) else int(r['n_evals'])} |")
info = json.loads((root / "data_info.json").read_text())
lines += ["", f"Data: {json.dumps(info)}", ""]

# ---- statistics (paired by run) -------------------------------------------------------------
common = sorted(set.intersection(*[set(runs[n]) for n in meta_algos])) if meta_algos else []
if len(meta_algos) >= 3 and len(common) >= 5:
    lines += ["## Statistical tests (gradient-free optimisers, paired by run seed)", ""]
    for m in args.stat_metrics:
        blocks = {n: np.array([runs[n][k]["test"].get(m, np.nan) for k in common]) for n in meta_algos}
        if any(np.isnan(v).any() for v in blocks.values()):
            continue
        fr = friedman(blocks)
        lines += [f"### {m}", "",
                  f"Friedman χ² = {fr['statistic']:.3f}, p = {fr['p']:.3g}, blocks = {fr['n_blocks']}.", "",
                  "Mean rank (1 = best): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(fr['mean_rank'].items(), key=lambda kv: kv[1])), ""]
        nm = nemenyi(blocks); nm.to_csv(root / f"nemenyi_{m}.csv")
        if args.reference in blocks:
            wx = wilcoxon_vs(blocks[args.reference], {n: v for n, v in blocks.items() if n != args.reference})
            lines += [f"Wilcoxon signed-rank, {args.reference} vs each (Holm-adjusted); A = Vargha–Delaney "
                      f"(>0.5 favours {args.reference}).", "", "| vs | median diff | p | p (Holm) | A |", "|---|---|---|---|---|"]
            for n, s in wx.items():
                lines.append(f"| {n} | {s['median_diff']:+.4f} | {s['p']:.3g} | {s['p_holm']:.3g} | {s['A']:.3f} |")
            lines.append("")
(root / "results.md").write_text("\n".join(lines))

# ---- figures --------------------------------------------------------------------------------
if meta_algos:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in meta_algos:
        H = np.array([runs[n][k]["history"] for k in sorted(runs[n])])
        x = np.arange(H.shape[1])
        med, q1, q3 = np.median(H, 0), np.quantile(H, .25, 0), np.quantile(H, .75, 0)
        ax.plot(x, med, label=n); ax.fill_between(x, q1, q3, alpha=.2)
    taus = {s.get("transition") for s in cfg["algorithms"].values() if s.get("type") == "hybrid"}
    for t in taus:
        ax.axvline(t, ls="--", c="grey", lw=.8)
    ax.set(xlabel="Iteration", ylabel="Best fitness (weighted CE + L2)", title="Convergence: median and IQR over runs")
    ax.set_yscale("log"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_dir / "convergence.png", dpi=200); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, m in zip(axes.flat, ["accuracy", "balanced_accuracy", "recall", "specificity", "mcc", "auc"]):
    data = [[runs[n][k]["test"].get(m, np.nan) for k in sorted(runs[n])] for n in order]
    ax.boxplot(data, tick_labels=order, showmeans=True); ax.set_title(m); ax.tick_params(axis="x", rotation=60, labelsize=7)
fig.tight_layout(); fig.savefig(fig_dir / "boxplots.png", dpi=200); plt.close(fig)

# Confusion matrix of the median-MCC run of each algorithm; annotations computed from the cells.
fig, axes = plt.subplots(1, len(order), figsize=(2.6 * len(order), 3))
for ax, n in zip(np.atleast_1d(axes), order):
    rs = sorted(runs[n].values(), key=lambda r: r["test"]["mcc"])
    r = rs[len(rs) // 2]["test"]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.set_title(f"{n}\nacc {acc:.3f} rec {cm[1,1]/cm[1].sum():.3f} spec {cm[0,0]/cm[0].sum():.3f}", fontsize=8)
    ax.set_xticks([0, 1], ["Normal", "Pneum."]); ax.set_yticks([0, 1], ["Normal", "Pneum."])
fig.suptitle("Test confusion matrices (median-MCC run); rows = true, columns = predicted", fontsize=9)
fig.tight_layout(); fig.savefig(fig_dir / "confusion_median_run.png", dpi=200); plt.close(fig)
print((root / "results.md").read_text())
