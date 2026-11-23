#!/usr/bin/env python3
"""LaTeX tables for the split study, the cross-dataset evaluation and the budget study
(paper/tables/split.tex, cross.tex, budget.tex), generated from the per-run results."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
NEW = ROOT / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "tables"
LAB = {"LogReg": "Logistic regression", "MLP-Adam": "MLP, backprop (Adam)", "Hybrid": "Hybrid FPA--SDA",
       "sep-CMA-ES": "sep-CMA-ES"}
METHODS = ["LogReg", "MLP-Adam", "L-SHADE", "GA", "SDA", "Hybrid"]
IDS = list(range(20))


def load(path):
    R = {}
    for f in glob.glob(f"{path}/runs/*/run_*.json"):
        r = json.load(open(f))
        R.setdefault(r["algorithm"], {})[r["run"]] = r
    return R


def v(R, a, m, ids=IDS):
    return np.array([R[a][i]["best_fitness"] if m == "fit" else R[a][i]["test"][m] for i in ids])


def pfmt(p):
    if p is None:
        return "--"
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    return "$<$0.001"


def pw(x, y):
    d = x - y
    if np.allclose(d, d[0]):      # deterministic method (e.g. logistic regression): a test over runs is meaningless
        return None
    return wilcoxon(x, y).pvalue


def table(caption, label, cols, head, rows, spec):
    L = [r"\begin{table}[!t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + spec + "}", r"\toprule"]
    L += head + [r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def split_table():
    A = load(BASE / "cnn_head"); A["L-SHADE"] = load(BASE / "cnn_redesign")["L-SHADE"]
    B = load(NEW / "split_B_random_image"); C = load(NEW / "split_C_random_patient")
    nb = json.load(open(NEW / "split_B_random_image/data_info.json"))
    rows = []
    for a in METHODS:
        acc = {k: 100 * v(R, a, "accuracy") for k, R in (("A", A), ("B", B), ("C", C))}
        mcc = {k: v(R, a, "mcc") for k, R in (("A", A), ("B", B), ("C", C))}
        rows.append(" & ".join([LAB.get(a, a)] + [f"{acc[k].mean():.1f}" for k in "ACB"] +
                               [f"{mcc[k].mean():.3f}" for k in "ACB"] +
                               [f"{(acc['C'] - acc['A']).mean():+.1f}", pfmt(pw(acc['C'], acc['A'])),
                                f"{(acc['B'] - acc['C']).mean():+.1f}", pfmt(pw(acc['B'], acc['C']))]) + r" \\")
    cap = (r"Effect of the evaluation protocol on Kermany (frozen ResNet-50 features, 2048--16--2 head; mean over 20 runs). "
           r"A: official split (patient-grouped optimisation/validation subsets, official test folder of 624 images). "
           r"C: all 5,856 images pooled and split 80/20 by patient (1,171 test images). "
           r"B: the same pooled images split 80/20 at image level, as in much of the literature (1,172 test images, "
           + f"{nb['test_images_with_patient_in_train']}" +
           r" of them from patients that also occur in training). $p$: Wilcoxon signed-rank test over runs with the same seed; --: deterministic method.")
    head = [r" & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{3}{c}{MCC} & \multicolumn{2}{c}{C $-$ A (acc.)} & \multicolumn{2}{c}{B $-$ C (acc.)} \\",
            r"Method & A & C & B & A & C & B & $\Delta$ & $p$ & $\Delta$ & $p$ \\"]
    return table(cap, "tab:split", 11, head, rows, "lcccccccccc")


def cross_table():
    KK = load(BASE / "cnn_head"); KK["L-SHADE"] = load(BASE / "cnn_redesign")["L-SHADE"]
    RR = load(NEW / "rsna_adult"); KR = load(NEW / "cross_K2R_adult"); RK = load(NEW / "cross_R2K_adult")
    rows = []
    for a in METHODS:
        c = [LAB.get(a, a)]
        for ind, x in ((RR, KR), (KK, RK)):
            c += [f"{v(ind, a, 'mcc').mean():.3f}", f"{v(x, a, 'mcc').mean():.3f}", f"{v(x, a, 'auc').mean():.3f}",
                  f"{100 * v(x, a, 'recall').mean():.1f}", f"{100 * v(x, a, 'specificity').mean():.1f}"]
        rows.append(" & ".join(c) + r" \\")
    cap = (r"Cross-dataset generalisation (frozen ResNet-50 features, 2048--16--2 head; mean over 20 runs). "
           r"In: trained and tested on the same dataset (RSNA patient-grouped held-out test set of adults; Kermany official test folder). "
           r"Cross: trained on the other dataset (Kermany, paediatric $\rightarrow$ RSNA, adults; RSNA $\rightarrow$ Kermany), "
           r"with standardisation statistics of the training dataset. For every stochastic method the in-domain versus "
           r"cross-dataset MCC difference has $p<10^{-4}$ (Wilcoxon signed-rank test); logistic regression is deterministic.")
    head = [r" & \multicolumn{5}{c}{Tested on RSNA} & \multicolumn{5}{c}{Tested on Kermany} \\",
            r" & MCC & \multicolumn{4}{c}{Trained on Kermany} & MCC & \multicolumn{4}{c}{Trained on RSNA} \\",
            r"Method & in & MCC & AUC & Sens. & Spec. & in & MCC & AUC & Sens. & Spec. \\"]
    return table(cap, "tab:cross", 11, head, rows, "lcccccccccc")


def budget_table():
    K = load(BASE / "cnn_head"); Rd = load(BASE / "cnn_redesign"); C1 = load(NEW / "cnn_cma")
    one = {"Hybrid": K["Hybrid"], "SDA": K["SDA"], "GA": K["GA"], "L-SHADE": Rd["L-SHADE"], "sep-CMA-ES": C1["sep-CMA-ES"]}
    five = load(NEW / "cnn_budget5x")
    rows = []
    order = ["L-SHADE", "GA", "sep-CMA-ES", "SDA", "Hybrid"]
    for a in order:
        f1, f5 = v(one, a, "fit"), v(five, a, "fit"); m1, m5 = v(one, a, "mcc"), v(five, a, "mcc")
        rows.append(" & ".join([LAB.get(a, a), f"{np.median(f1):.3f}", f"{np.median(f5):.3f}", f"{m1.mean():.3f}",
                                f"{m5.mean():.3f}", f"{(m5 - m1).mean():+.3f}", pfmt(pw(m5, m1))]) + r" \\")
    r1 = sorted(order, key=lambda a: -v(one, a, "mcc").mean()); r5 = sorted(order, key=lambda a: -v(five, a, "mcc").mean())
    cap = (r"Budget sensitivity on the main Kermany task: 6,795 versus 33,795 fitness evaluations (45 $\times$ 750 iterations; "
           r"hybrid transition at iteration 375), 20 runs with the same seeds. Fitness: median final training fitness; "
           r"MCC: mean test MCC; $p$: Wilcoxon signed-rank test of the MCC change. Ranking by test MCC at 1$\times$: "
           + " $>$ ".join(LAB.get(a, a) for a in r1) + r"; at 5$\times$: " + " $>$ ".join(LAB.get(a, a) for a in r5) + ".")
    head = [r" & \multicolumn{2}{c}{Training fitness} & \multicolumn{2}{c}{Test MCC} & & \\",
            r"Method & 1$\times$ & 5$\times$ & 1$\times$ & 5$\times$ & $\Delta$MCC & $p$ \\"]
    return table(cap, "tab:budget", 7, head, rows, "lcccccc")


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "split.tex").write_text(split_table())
(OUT / "cross.tex").write_text(cross_table())
if (NEW / "cnn_budget5x").exists() and len(glob.glob(str(NEW / "cnn_budget5x/runs/*/run_*.json"))) >= 100:
    (OUT / "budget.tex").write_text(budget_table())
print("tables written to", OUT)
