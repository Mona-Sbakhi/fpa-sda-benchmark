#!/usr/bin/env python3
"""Tables for the second backbone (DenseNet-121) and the calibration/threshold analysis.

    python scripts/backbone_calib_tables.py OUTDIR
Writes OUTDIR/densenet.tex, OUTDIR/calibration.tex and results/backbone_summary.json.
Needs results/cnn_densenet, results/cnn_calib (and calibration_cnn_calib.json from calibration_threshold.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
ORDER = ["FPA", "SDA", "Hybrid", "PSO", "GA", "APH", "L-SHADE", "sep-CMA-ES", "LogReg", "MLP-weighted"]
LABEL = {"Hybrid": "Hybrid FPA--SDA", "APH": "APH (redesigned hybrid)", "LogReg": "Logistic regression",
         "MLP-weighted": "MLP, backprop (weighted)"}


def load(exp, a, n=20):
    rs = [json.load(open(ROOT / "results" / exp / "runs" / a / f"run_{i:02d}.json")) for i in range(n)]
    return rs


def col(rs, part, k):
    return np.array([r[part][k] for r in rs], float)


summ = {}
for a in ORDER:
    if not (ROOT / "results/cnn_densenet/runs" / a).exists():
        continue
    dn = load("cnn_densenet", a); rn = load("cnn_calib", a)
    summ[a] = {"densenet": {k: [float(col(dn, "test", k).mean()), float(col(dn, "test", k).std(ddof=1))]
                            for k in ("mcc", "accuracy", "auc", "balanced_accuracy")},
               "resnet": {k: [float(col(rn, "test", k).mean()), float(col(rn, "test", k).std(ddof=1))]
                          for k in ("mcc", "accuracy", "auc", "balanced_accuracy")},
               "dn_fitness": [float(r.get("best_fitness", np.nan)) for r in dn]}
# paired comparisons on DenseNet features
h, s = load("cnn_densenet", "Hybrid"), load("cnn_densenet", "SDA")
fh, fs = np.array([r["best_fitness"] for r in h]), np.array([r["best_fitness"] for r in s])
mh, ms = col(h, "test", "mcc"), col(s, "test", "mcc")
vh, vs = col(h, "val", "mcc"), col(s, "val", "mcc")
summ["hybrid_vs_sda"] = {"fitness_hybrid_worse": int((fh > fs).sum()), "fitness_p": float(wilcoxon(fh, fs).pvalue),
                         "val_mcc_hybrid_lower": int((vh < vs).sum()), "val_p": float(wilcoxon(vh, vs).pvalue),
                         "test_mcc_hybrid_lower": int((mh < ms).sum()), "test_p": float(wilcoxon(mh, ms).pvalue)}
gf = [a for a in ORDER if a in summ and a not in ("LogReg", "MLP-weighted")]
rank_dn = sorted(gf, key=lambda a: -summ[a]["densenet"]["mcc"][0])
rank_rn = sorted(gf, key=lambda a: -summ[a]["resnet"]["mcc"][0])
from scipy.stats import spearmanr
summ["rank_densenet"], summ["rank_resnet"] = rank_dn, rank_rn
summ["spearman_rank"] = float(spearmanr([summ[a]["densenet"]["mcc"][0] for a in gf],
                                        [summ[a]["resnet"]["mcc"][0] for a in gf]).correlation)
(ROOT / "results" / "backbone_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict) or "densenet" not in v}, indent=1))

f = lambda m: f"{m[0]:.3f}$\\pm${m[1]:.3f}"
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{Second backbone: test performance with frozen DenseNet-121 features (1024--16--2 head, $D=16{,}434$) "
     r"and, for the same 20 seeds, with frozen ResNet-50 features (2048--16--2 head, $D=32{,}818$); mean$\pm$SD over 20 "
     r"paired runs, 6,795 fitness evaluations for every gradient-free method.}",
     r"\label{tab:densenet}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{lcccc}", r"\toprule",
     r" & \multicolumn{2}{c}{DenseNet-121} & \multicolumn{2}{c}{ResNet-50} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     r"Method & MCC & AUC & MCC & AUC \\", r"\midrule"]
for a in ORDER:
    if a not in summ:
        continue
    if a == "LogReg":
        L.append(r"\midrule")
    d, r = summ[a]["densenet"], summ[a]["resnet"]
    L.append(f"{LABEL.get(a, a)} & {f(d['mcc'])} & {f(d['auc'])} & {f(r['mcc'])} & {f(r['auc'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "densenet.tex").write_text("\n".join(L) + "\n")

C = json.load(open(ROOT / "results" / "calibration_cnn_calib.json"))
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{Calibration and decision threshold on the Kermany test set (frozen ResNet-50 features, 20 runs, seeds of "
     r"runs 0--19). PR-AUC: average precision; ECE: expected calibration error (10 bins); $t_\mathrm{val}$: threshold "
     r"maximising MCC on the validation subset; last column: runs in which $t_\mathrm{val}$ improved test MCC.}",
     r"\label{tab:calibration}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
     r"\begin{tabular}{lccccccc}", r"\toprule",
     r"Method & PR-AUC & Brier & ECE & MCC$_{0.5}$ & $t_\mathrm{val}$ & MCC$_{t_\mathrm{val}}$ & Improved \\", r"\midrule"]
for a in ORDER:
    if a not in C:
        continue
    if a == "LogReg":
        L.append(r"\midrule")
    m = C[a]["mean"]
    L.append(f"{LABEL.get(a, a)} & {m['pr_auc']:.3f} & {m['brier']:.3f} & {m['ece']:.3f} & {m['mcc_05']:.3f} & "
             f"{m['t_val']:.2f} & {m['mcc_tval']:.3f} & {C[a]['gain_runs_improved']}/{C[a]['n_runs']} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "calibration.tex").write_text("\n".join(L) + "\n")
print("tables written to", OUT)
