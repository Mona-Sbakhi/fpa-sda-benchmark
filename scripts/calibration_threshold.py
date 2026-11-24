#!/usr/bin/env python3
"""Calibration and decision-threshold analysis (exploratory, not pre-registered).

For every method of an experiment that stored validation probabilities (cnn_calib: ResNet-50 features;
cnn_densenet: DenseNet-121 features) it reports, per run and then as mean (SD) over runs:
  - discrimination: AUROC and PR-AUC (average precision) on the test set;
  - calibration: Brier score and expected calibration error (ECE, 10 equal-width bins) on the test set;
  - the effect of the decision threshold: test MCC at the fixed threshold 0.5 and at the threshold that
    maximises MCC on the validation subset (grid 0.01..0.99; the test set is not used to choose it).
Usage: python scripts/calibration_threshold.py cnn_calib [cnn_densenet ...]
Writes results/calibration_<exp>.json and prints a summary table.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpasda.model import FlatMLP  # noqa: E402
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def mcc(y, p):
    y = y.astype(bool); p = p.astype(bool)
    tp = float((p & y).sum()); tn = float((~p & ~y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def ece(y, q, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(q[m].mean() - y[m].mean())
    return float(e)


def analyse(exp):
    d = ROOT / "results" / exp
    prep = np.load(d / "prepared.npz"); yv = prep["y_val"].astype(int); yt = prep["y_test"].astype(int)
    cfg = json.load(open(ROOT / "configs" / f"{exp}.json"))
    net = FlatMLP(prep["X_val"].shape[1], cfg["model"]["hidden"])
    out = {}
    for md in sorted((d / "runs").iterdir()):
        rows = []
        for f in sorted(md.glob("run_*.json")):
            r = json.load(open(f))
            if "val_proba" not in r:
                continue
            w = f.with_suffix(".npy")
            if w.exists():   # full precision: the stored probabilities are rounded to 6 decimals, which creates
                th = np.load(w)   # ties among saturated outputs and biases AUROC/PR-AUC downwards
                qv = net.predict_proba(th, prep["X_val"])[:, 1]; qt = net.predict_proba(th, prep["X_test"])[:, 1]
                assert ((qt >= 0.5) == (np.array(r["test_proba"]) >= 0.5)).mean() > 0.998
            else:
                qv = np.array(r["val_proba"]); qt = np.array(r["test_proba"])
            vm = [mcc(yv, qv >= t) for t in GRID]
            t_star = float(GRID[int(np.argmax(vm))])
            sens = lambda t: float(((qt >= t) & (yt == 1)).sum() / (yt == 1).sum())
            spec = lambda t: float(((qt < t) & (yt == 0)).sum() / (yt == 0).sum())
            rows.append({"run": r["run"], "auroc": roc_auc_score(yt, qt), "pr_auc": average_precision_score(yt, qt),
                         "brier": float(np.mean((qt - yt) ** 2)), "ece": ece(yt, qt),
                         "mcc_05": mcc(yt, qt >= 0.5), "t_val": t_star, "mcc_tval": mcc(yt, qt >= t_star),
                         "sens_05": sens(0.5), "spec_05": spec(0.5), "sens_tval": sens(t_star), "spec_tval": spec(t_star)})
        if rows:
            keys = [k for k in rows[0] if k != "run"]
            out[md.name] = {"n_runs": len(rows),
                            "mean": {k: float(np.mean([x[k] for x in rows])) for k in keys},
                            "sd": {k: float(np.std([x[k] for x in rows], ddof=1)) for k in keys},
                            "gain_runs_improved": int(sum(x["mcc_tval"] > x["mcc_05"] for x in rows)),
                            "runs": rows}
    (ROOT / "results" / f"calibration_{exp}.json").write_text(json.dumps(out, indent=1))
    print(f"\n== {exp} (test set; threshold chosen on validation) ==")
    print(f"{'method':14s} {'n':>3s} {'AUROC':>6s} {'PR-AUC':>6s} {'Brier':>6s} {'ECE':>6s} {'MCC@.5':>7s} {'t_val':>5s} "
          f"{'MCC@t':>6s} {'impr':>5s}")
    for a, v in sorted(out.items(), key=lambda kv: -kv[1]["mean"]["mcc_05"]):
        m = v["mean"]
        print(f"{a:14s} {v['n_runs']:3d} {m['auroc']:6.3f} {m['pr_auc']:6.3f} {m['brier']:6.3f} {m['ece']:6.3f} "
              f"{m['mcc_05']:7.3f} {m['t_val']:5.2f} {m['mcc_tval']:6.3f} {v['gain_runs_improved']:3d}/{v['n_runs']}")
    return out


if __name__ == "__main__":
    for e in sys.argv[1:] or ["cnn_calib"]:
        analyse(e)
