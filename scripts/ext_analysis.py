#!/usr/bin/env python3
"""Pre-registered primary analysis on the external paediatric test set (amendment 1).

Applies every stored head (trained on Kermany only) once to the external features, then estimates
    Delta_r = MCC(APH full) - MCC(APH without O3)    for r in {frozen, fine-tuned}
    I = Delta_finetuned - Delta_frozen
with a cluster bootstrap over external patients (5,000 resamples, run seeds fixed), and applies the
pre-registered decision rule (smallest effect of interest 0.05 MCC). Secondary: AUROC, sensitivity,
specificity, internal Kermany Delta. Gradient-trained references are retrained deterministically from their
stored seeds on the same development data (no external information is used).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpasda.model import FlatMLP  # noqa: E402

CACHE = ROOT / "cache"   # cache/ext_paed.npz and cache/ext_paed_ft_s*.npz from scripts/ext_paed_features.py
GF = ["APH", "APH-noO3", "APH-O3only", "L-SHADE"]
REF = ["LogReg", "MLP-Adam"]
B, SEOI = 5000, 0.05
E = np.load(CACHE / "ext_paed.npz")
y = E["y"].astype(bool); pat = E["patient"]
groups = [np.flatnonzero(pat == u) for u in np.unique(pat)]


def mcc_rows(P, yy):
    tp = (P & yy).sum(-1); tn = (~P & ~yy).sum(-1); fp = (P & ~yy).sum(-1); fn = (~P & yy).sum(-1)
    den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros(np.broadcast(tp, den).shape), where=den > 0)


def heads(exp, F_ext):
    """Probabilities of every stored head of experiment `exp` on the external features."""
    d = ROOT / "results" / exp
    sc = np.load(d / "scaler.npz"); X = ((F_ext - sc["mean"]) / sc["scale"]).astype(np.float32)
    net = FlatMLP(X.shape[1], [16]); cfg = json.load(open(ROOT / f"configs/{exp}.json"))
    out = {}
    for a in GF:
        out[a] = np.array([net.predict_proba(np.load(d / f"runs/{a}/run_{i:02d}.npy"), X)[:, 1] for i in range(20)])
    prep = np.load(d / "prepared.npz")
    for a in REF:
        spec = cfg["baselines"][a]; P = []
        for i in range(20):
            seed = int(cfg.get("seed0", 0)) * 1000 + i
            if spec["type"] == "logreg":
                m = LogisticRegression(C=spec.get("C", 1.0), class_weight="balanced", max_iter=5000)
            else:
                m = MLPClassifier(hidden_layer_sizes=tuple(spec.get("hidden", [16])), solver=spec.get("solver", "adam"),
                                  learning_rate_init=spec.get("lr", 1e-3), momentum=spec.get("momentum", 0.9),
                                  alpha=spec.get("alpha", 1e-3), batch_size=spec.get("batch_size", 64),
                                  max_iter=spec.get("epochs", 150), early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=15, random_state=seed)
            m.fit(prep["X_opt"], prep["y_opt"])
            stored = json.load(open(d / f"runs/{a}/run_{i:02d}.json"))["test_proba"]   # reproducibility check
            pk = m.predict_proba(prep["X_test"])[:, 1]; stored = np.array(stored)   # deviation 1: see docs
            assert ((pk >= 0.5) == (stored >= 0.5)).all() and np.abs(pk - stored).max() < 0.01, (exp, a, i)
            P.append(m.predict_proba(X)[:, 1])
        out[a] = np.array(P)
    return out


rep = {"frozen": [heads("ext_final_frozen", E["F_frozen"])],
       "finetuned": [heads(f"ext_final_ft{s}", np.load(CACHE / f"ext_paed_ft_s{s}.npz")["F"]) for s in (0, 1, 2)]}


def delta(blocks, idx=None):
    vals = []
    for b in blocks:
        yy = y if idx is None else y[idx]
        full = b["APH"] if idx is None else b["APH"][:, idx]; no = b["APH-noO3"] if idx is None else b["APH-noO3"][:, idx]
        vals.append((mcc_rows(full >= 0.5, yy) - mcc_rows(no >= 0.5, yy)).mean())
    return float(np.mean(vals))


D = {r: delta(rep[r]) for r in rep}; I = D["finetuned"] - D["frozen"]
rng = np.random.default_rng(20261122); bI, bF, bT = [], [], []
for _ in range(B):
    idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
    f_, t_ = delta(rep["frozen"], idx), delta(rep["finetuned"], idx); bF.append(f_); bT.append(t_); bI.append(t_ - f_)
ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
lo, hi = ci(bI)
if lo > 0 and I >= SEOI:
    verdict = "H1 supported, meaningful (lower bound > 0 and I >= 0.05)"
elif lo > 0:
    verdict = "H1 supported (lower bound > 0), but I < 0.05"
elif lo > -SEOI and hi < SEOI:
    verdict = "no meaningful interaction (interval within +-0.05)"
else:
    verdict = "inconclusive"
res = {"n_images": int(len(y)), "n_patients": len(groups), "Delta_frozen": D["frozen"], "Delta_frozen_CI": ci(bF),
       "Delta_finetuned": D["finetuned"], "Delta_finetuned_CI": ci(bT), "I": I, "I_CI": [lo, hi], "verdict": verdict}
sec = {}
for r, blocks in rep.items():
    for a in GF + REF:
        P = np.concatenate([b[a] for b in blocks]); pr = P >= 0.5
        sec[f"{r}/{a}"] = {"mcc": float(mcc_rows(pr, y).mean()), "auc": float(np.mean([roc_auc_score(y, p) for p in P])),
                           "sens": float((pr & y).sum(1).mean() / y.sum()), "spec": float((~pr & ~y).sum(1).mean() / (~y).sum())}
res["secondary_external"] = sec
internal = {}
for r, exps in (("frozen", ["ext_final_frozen"]), ("finetuned", [f"ext_final_ft{s}" for s in (0, 1, 2)])):
    dd = []
    for e in exps:
        g = lambda a: np.array([json.load(open(ROOT / f"results/{e}/runs/{a}/run_{i:02d}.json"))["test"]["mcc"] for i in range(20)])
        dd.append((g("APH") - g("APH-noO3")).mean())
    internal[r] = float(np.mean(dd))
res["internal_kermany_Delta"] = internal
(ROOT / "results/ext_paed_analysis.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "secondary_external"}, indent=2))
for k, v in sec.items():
    print(f"  {k:22s} MCC {v['mcc']:.3f}  AUC {v['auc']:.3f}  sens {100*v['sens']:.1f}  spec {100*v['spec']:.1f}")
