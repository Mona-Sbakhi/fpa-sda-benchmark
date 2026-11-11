"""Prepare features/splits once, then run every (algorithm, run) pair and save one JSON per pair."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import __version__
from .baselines import run_baseline
from .data import load_dataset, patient_id
from .features import PCAFeatures, load_precomputed
from .metrics import classification_metrics
from .model import FlatMLP, Fitness, ScaledObjective, balanced_class_weights, glorot_scale
from .optimizers import build

_DATA: dict | None = None
_CFG: dict | None = None


def prepare(cfg: dict, out_dir: Path) -> Path:
    """Split train into optimisation / validation subsets by patient, build features, cache them."""
    prepared = out_dir / "prepared.npz"
    if prepared.exists():
        return prepared
    fc = cfg["features"]
    if fc["type"] == "pca":
        d = load_dataset(cfg["data"]["root"], cfg["data"].get("size", 150), cfg["data"].get("cache_dir", "cache"))
        X_tr, y_tr, X_te, y_te, files = d["X_train"], d["y_train"], d["X_test"], d["y_test"], d["train_files"]
    else:
        d = load_precomputed(fc["path"])
        X_tr, y_tr, X_te, y_te = d["F_train"], d["y_train"], d["F_test"], d["y_test"]
        files = list(d["train_files"])

    groups = np.array([patient_id(f) for f in files])
    k = int(round(1 / cfg["split"].get("val_fraction", 0.2)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["split"].get("seed", 42))
    opt_idx, val_idx = next(sgkf.split(X_tr, y_tr, groups))
    assert not set(groups[opt_idx]) & set(groups[val_idx]), "patient leakage between opt/val"

    info = {"n_opt": int(len(opt_idx)), "n_val": int(len(val_idx)), "n_test": int(len(y_te)),
            "opt_class_counts": np.bincount(y_tr[opt_idx]).tolist(),
            "val_class_counts": np.bincount(y_tr[val_idx]).tolist(),
            "test_class_counts": np.bincount(y_te).tolist()}
    if fc["type"] == "pca":
        feat = PCAFeatures(fc.get("n_components", 1000), fc.get("standardize", True), cfg["split"].get("seed", 42))
        feat.fit(X_tr[opt_idx])
        F_opt, F_val, F_te = feat.transform(X_tr[opt_idx]), feat.transform(X_tr[val_idx]), feat.transform(X_te)
        info["pca_explained_variance"] = feat.explained_variance
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_tr[opt_idx]) if fc.get("standardize", True) else None
        tf = (lambda Z: sc.transform(Z)) if sc else (lambda Z: Z)
        F_opt, F_val, F_te = tf(X_tr[opt_idx]), tf(X_tr[val_idx]), tf(X_te)

    np.savez(prepared, X_opt=F_opt.astype(np.float32), y_opt=y_tr[opt_idx],
             X_val=F_val.astype(np.float32), y_val=y_tr[val_idx],
             X_test=F_te.astype(np.float32), y_test=y_te)
    (out_dir / "data_info.json").write_text(json.dumps(info, indent=2))
    return prepared


def _init_worker(prepared: str, cfg: dict):
    global _DATA, _CFG
    d = np.load(prepared)
    _DATA = {k: d[k] for k in d.files}
    _CFG = cfg


def run_one(task: tuple[str, int, str]) -> str:
    name, run, out_path = task
    cfg, data = _CFG, _DATA
    seed = int(cfg.get("seed0", 0)) * 1000 + run
    spec = {**cfg["algorithms"], **cfg.get("baselines", {})}[name]

    if name in cfg.get("baselines", {}):
        res = run_baseline(name, spec, data, run, seed)
    else:
        mc = cfg["model"]
        net = FlatMLP(data["X_opt"].shape[1], mc["hidden"])
        cw = balanced_class_weights(data["y_opt"]) if mc.get("class_weighted", True) else None
        fit = Fitness(net, data["X_opt"], data["y_opt"], mc.get("l2_lambda", 1e-3), cw)
        b = cfg["budget"]
        opt_spec = dict(spec)
        ps = opt_spec.pop("param_scale", mc.get("param_scale"))   # None | {"type": "glorot", "mult": c}
        scale = glorot_scale(net, ps.get("mult", 1.0)) if ps else None
        obj = ScaledObjective(fit, scale) if scale is not None else fit
        opt = build(opt_spec, b["pop_size"], b["iterations"], cfg.get("bounds", (-1.0, 1.0)))
        rng = np.random.default_rng(seed)  # same seed for every algorithm in run `run` -> paired design
        t0 = time.time()
        r = opt.run(obj, net.dim, rng)
        if scale is not None:
            r.best_x = r.best_x * scale       # back to network weights
        elapsed = time.time() - t0
        ce, reg = fit.components(r.best_x)
        pv = net.predict_proba(r.best_x, data["X_val"])[:, 1]
        pt = net.predict_proba(r.best_x, data["X_test"])[:, 1]
        res = {"algorithm": name, "run": run, "seed": seed, "spec": spec,
               "best_fitness": r.best_f, "fitness_ce": ce, "fitness_l2": reg,
               "history": [float(h) for h in r.history], "n_evals": fit.n_evals,
               "training_time": elapsed, "dim": net.dim,
               "val": classification_metrics(data["y_val"], (pv >= 0.5).astype(int), pv),
               "test": classification_metrics(data["y_test"], (pt >= 0.5).astype(int), pt),
               "test_proba": [round(float(p), 6) for p in pt],
               "extra": {k: v for k, v in r.extra.items() if k != "population"}}
    res["seed"] = seed
    Path(out_path).write_text(json.dumps(res))
    return f"{name} run {run}: test acc={res['test']['accuracy']:.4f} mcc={res['test']['mcc']:.4f} " \
           f"t={res['training_time']:.1f}s"


def write_meta(cfg: dict, out_dir: Path):
    import scipy, sklearn
    meta = {"package_version": __version__, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
