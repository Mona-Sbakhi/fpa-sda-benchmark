#!/usr/bin/env python3
"""Post-hoc sensitivity analyses requested in review (not pre-registered):
(a) external paediatric test restricted to images never seen in any earlier experiment (excluding the
    images that were part of the uncorrected RSNA test set);
(b) main Kermany benchmark without the pilot seeds 0-4 of the methods added after the first benchmark."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache"
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
out = {}

# (a) -------------------------------------------------------------------------------------------
E = np.load(CACHE / "ext_paed.npz")
old = np.load(CACHE / "rsna_resnet50_224.npz")
seen = {str(f).split("/")[-1][:-4] for f in old["test_files"]}
mask = np.array([i not in seen for i in E["ids"]])
import importlib.util  # reuse the frozen analysis functions without re-running its main block
src = open(ROOT / "scripts/ext_analysis.py").read().split("rep = {")[0]
ns = {"__file__": str(ROOT / "scripts/ext_analysis.py"), "__name__": "ext"}
exec(compile(src.replace('CACHE = ROOT / "cache"', f'CACHE = Path({str(CACHE)!r})'), "ext", "exec"), ns)
rep = {"frozen": [ns["heads"]("ext_final_frozen", E["F_frozen"])],
       "finetuned": [ns["heads"](f"ext_final_ft{s}", np.load(CACHE / f"ext_paed_ft_s{s}.npz")["F"]) for s in (0, 1, 2)]}
y = E["y"].astype(bool)[mask]; pat = E["patient"][mask]
groups = [np.flatnonzero(pat == u) for u in np.unique(pat)]
mr = ns["mcc_rows"]


def delta(blocks, idx):
    v = []
    for b in blocks:
        f_ = b["APH"][:, mask][:, idx]; n_ = b["APH-noO3"][:, mask][:, idx]
        v.append((mr(f_ >= 0.5, y[idx]) - mr(n_ >= 0.5, y[idx])).mean())
    return float(np.mean(v))


allidx = np.arange(mask.sum()); D = {r: delta(rep[r], allidx) for r in rep}
rng = np.random.default_rng(20261124); bI = []
for _ in range(5000):
    idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
    bI.append(delta(rep["finetuned"], idx) - delta(rep["frozen"], idx))
out["unseen_external"] = {"n_images": int(mask.sum()), "n_excluded": int((~mask).sum()), "n_patients": len(groups),
                          "Delta_frozen": D["frozen"], "Delta_finetuned": D["finetuned"],
                          "I": D["finetuned"] - D["frozen"], "I_CI": [float(np.quantile(bI, .025)), float(np.quantile(bI, .975))]}

# (b) -------------------------------------------------------------------------------------------
def load(exp):
    R = {}
    for f in glob.glob(str(RES / exp / "runs/*/run_*.json")):
        r = json.load(open(f)); R.setdefault(r["algorithm"], {})[r["run"]] = r
    return R


H = load("cnn_head"); Rd = load("cnn_redesign"); Ab = load("cnn_redesign_ablation")
M = {a: H[a] for a in ["FPA", "SDA", "Hybrid", "Hybrid-reinit", "PSO", "DE", "GA", "Random"]}
M.update({"APH": Rd["APH"], "L-SHADE": Rd["L-SHADE"]})
GF = list(M)
for label, ids in (("all_50", list(range(50))), ("without_pilot_seeds_0_4", list(range(5, 50)))):
    v = {a: np.array([M[a][i]["test"]["mcc"] for i in ids]) for a in GF}
    st = friedmanchisquare(*v.values())
    ranks = np.argsort(np.argsort(-np.stack(list(v.values())), axis=0), axis=0) + 1
    mrk = dict(zip(GF, ranks.mean(1).round(2).tolist()))
    pv = lambda a, b: float(wilcoxon(v[a], v[b]).pvalue)
    fit = lambda a: np.array([M[a][i]["best_fitness"] for i in ids])
    out[label] = {"n_runs": len(ids), "friedman_chi2": float(st.statistic), "friedman_p": float(st.pvalue),
                  "mean_rank": dict(sorted(mrk.items(), key=lambda t: t[1])),
                  "mean_mcc": {a: round(float(x.mean()), 3) for a, x in v.items()},
                  "APH_vs_LSHADE_mcc_p": pv("APH", "L-SHADE"), "APH_vs_GA_mcc_p": pv("APH", "GA"),
                  "APH_lower_fitness_than_LSHADE_runs": int((fit("APH") < fit("L-SHADE")).sum())}
print(json.dumps(out, indent=1))
(RES / "sensitivity_extra.json").write_text(json.dumps(out, indent=1))
