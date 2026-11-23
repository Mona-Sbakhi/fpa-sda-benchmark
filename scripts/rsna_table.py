#!/usr/bin/env python3
"""Write the LaTeX table of the RSNA replication (paper/tables/rsna.tex) from results/<experiment> (default rsna_adult)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parents[1]
R = {}
EXP = sys.argv[1] if len(sys.argv) > 1 else "rsna_adult"
OUTF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper/tables/rsna.tex"
for f in glob.glob(str(ROOT / "results" / EXP / "runs/*/run_*.json")):
    r = json.load(open(f))
    R.setdefault(r["algorithm"], {})[r["run"]] = r
ids = sorted(R["GA"])
GF = [a for a in R if "best_fitness" in R[a][ids[0]]]


def v(a, k, m="mcc"):
    return np.array([R[a][i][k][m] for i in ids])


ranks = np.argsort(np.argsort(-np.stack([v(a, "test") for a in GF]), axis=0), axis=0) + 1
mr = dict(zip(GF, ranks.mean(1)))
LAB = {"Hybrid": "Hybrid FPA--SDA (transfer)", "Hybrid-reinit": "Hybrid FPA--SDA (re-init)", "Random": "Random search",
       "APH": "APH (redesigned hybrid)", "LogReg": "Logistic regression", "MLP-Adam": "MLP, backprop (Adam)",
       "MLP-SGD": "MLP, backprop (SGD)", "Majority": "Majority class"}


def row(a):
    c = [LAB.get(a, a)]
    for m, k in [("accuracy", 100), ("recall", 100), ("specificity", 100), ("mcc", 1), ("auc", 1)]:
        x = v(a, "test", m) * k
        c.append(f"{x.mean():.1f}$\\pm${x.std(ddof=1):.1f}" if k == 100 else f"{x.mean():.3f}$\\pm${x.std(ddof=1):.3f}")
    c.append(f"{mr[a]:.2f}" if a in mr else "--")
    return " & ".join(c) + r" \\"


st = friedmanchisquare(*[v(a, "test") for a in GF])
mant, ex = f"{st.pvalue:.1e}".split("e")
p = f"{mant}\\times 10^{{{int(ex)}}}"
L = [r"\begin{table}[!t]", r"\centering",
     r"\caption{External replication on the RSNA Pneumonia dataset (adults aged 18--100 years, normal vs lung opacity, split by patient; 2048--16--2 "
     r"head on frozen ResNet-50 features, $D=32{,}818$); mean$\pm$SD over 20 paired runs on the held-out test set "
     r"(" + f"{len(next(iter(R.values()))[ids[0]]['test_proba']):,}" + r" images). Rank: mean Friedman rank of test MCC among the " + str(len(GF)) +
     r" gradient-free optimisers (1 = best); Friedman $\chi^2_F=" + f"{st.statistic:.1f}" + r"$, $p=" + p + r"$.}",
     r"\label{tab:rsna}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Method & Acc. (\%) & Sens. (\%) & Spec. (\%) & MCC & AUC & Rank \\", r"\midrule",
     r"\multicolumn{7}{l}{\textit{Gradient-free optimisers (6,795 fitness evaluations each)}} \\"]
L += [row(a) for a in sorted(GF, key=lambda a: mr[a])]
L += [r"\multicolumn{7}{l}{\textit{Reference classifiers trained by gradient methods}} \\"]
L += [row(a) for a in ["LogReg", "MLP-Adam", "MLP-SGD", "Majority"]]
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
out = OUTF
out.write_text("\n".join(L) + "\n")
print("wrote", out)
