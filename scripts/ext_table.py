#!/usr/bin/env python3
"""LaTeX table of the pre-registered external paediatric test (paper/tables/external.tex), from
results/ext_paed_analysis.json and the internal Kermany results of the final runs."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/tables/external.tex"
R = json.load(open(ROOT / "results/ext_paed_analysis.json"))
REPS = {"frozen": ["ext_final_frozen"], "finetuned": [f"ext_final_ft{s}" for s in (0, 1, 2)]}
LAB = {"APH": "APH (O1--O4)", "APH-noO3": "APH without O3", "APH-O3only": "O3 (current-to-$p$best) only",
       "L-SHADE": "L-SHADE", "LogReg": "Logistic regression", "MLP-Adam": "MLP, backprop (Adam)"}


def kermany_mcc(exps, a):
    return np.mean([np.mean([json.load(open(ROOT / f"results/{e}/runs/{a}/run_{i:02d}.json"))["test"]["mcc"]
                             for i in range(20)]) for e in exps])


ci = lambda v: f"[{v[0]:+.3f}, {v[1]:+.3f}]"
cap = (r"Pre-registered external test on children (RSNA, NIH Clinical Center; 847 images of 448 patients aged 1--17 years, "
       r"normal vs lung opacity). All models were trained on Kermany development data only and fixed before the test. "
       r"Frozen: ImageNet ResNet-50 features, 20 runs; fine-tuned: ResNet-50 fine-tuned on the Kermany optimisation subset, "
       r"3 fine-tuning seeds $\times$ 20 runs. MCC (Kermany): official Kermany test folder. External columns: mean over runs. "
       r"$\Delta$: MCC of APH minus APH without O3 on the external set; $I=\Delta_{\mathrm{fine\text{-}tuned}}-\Delta_{\mathrm{frozen}}$; "
       r"95\% intervals from a bootstrap over external patients (5,000 resamples). "
       rf"$\Delta_{{\mathrm{{frozen}}}}={R['Delta_frozen']:+.3f}$ {ci(R['Delta_frozen_CI'])}, "
       rf"$\Delta_{{\mathrm{{fine\text{{-}}tuned}}}}={R['Delta_finetuned']:+.3f}$ {ci(R['Delta_finetuned_CI'])}, "
       rf"$I={R['I']:+.3f}$ {ci(R['I_CI'])}.")
L = [r"\begin{table}[!t]", r"\centering", r"\caption{" + cap + "}", r"\label{tab:external}", r"\footnotesize",
     r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccc}", r"\toprule",
     r" & & MCC & \multicolumn{4}{c}{External paediatric test} \\",
     r"Representation & Method & (Kermany) & MCC & AUC & Sens. (\%) & Spec. (\%) \\", r"\midrule"]
for rep, name in (("frozen", "Frozen"), ("finetuned", "Fine-tuned")):
    for k, a in enumerate(LAB):
        s = R["secondary_external"][f"{rep}/{a}"]
        L.append(" & ".join([name if k == 0 else "", LAB[a], f"{kermany_mcc(REPS[rep], a):.3f}", f"{s['mcc']:.3f}",
                             f"{s['auc']:.3f}", f"{100 * s['sens']:.1f}", f"{100 * s['spec']:.1f}"]) + r" \\")
    if rep == "frozen":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
OUT.write_text("\n".join(L) + "\n")
print("wrote", OUT)
