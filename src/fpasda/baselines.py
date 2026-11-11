"""Reference classifiers trained on the same features and split as the metaheuristics."""
from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .metrics import classification_metrics


def _evaluate(name, run, predict, proba, data, t0, extra=None):
    out = {"algorithm": name, "run": run, "training_time": time.time() - t0,
           "val": classification_metrics(data["y_val"], predict(data["X_val"]), proba(data["X_val"])),
           "test": classification_metrics(data["y_test"], predict(data["X_test"]), proba(data["X_test"]))}
    out["test_proba"] = [round(float(p), 6) for p in proba(data["X_test"])]
    out.update(extra or {})
    return out


def run_baseline(name: str, spec: dict, data: dict, run: int, seed: int) -> dict:
    kind = spec["type"]
    X, y = data["X_opt"], data["y_opt"]
    t0 = time.time()
    if kind == "majority":
        cls = int(np.bincount(y).argmax())
        return _evaluate(name, run, lambda Z: np.full(len(Z), cls),
                         lambda Z: np.full(len(Z), float(cls)), data, t0)
    if kind == "logreg":
        m = LogisticRegression(C=spec.get("C", 1.0), class_weight="balanced", max_iter=5000)
    elif kind == "mlp_backprop":
        m = MLPClassifier(hidden_layer_sizes=tuple(spec.get("hidden", (128, 64, 32))),
                          solver=spec.get("solver", "adam"),
                          learning_rate_init=spec.get("lr", 1e-3),
                          momentum=spec.get("momentum", 0.9),
                          alpha=spec.get("alpha", 1e-3),
                          batch_size=spec.get("batch_size", 64),
                          max_iter=spec.get("epochs", 150),
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                          random_state=seed)
    else:
        raise ValueError(kind)
    m.fit(X, y)
    extra = {}
    if kind == "mlp_backprop":
        extra = {"epochs_run": int(m.n_iter_), "history": [float(v) for v in m.loss_curve_]}
    return _evaluate(name, run, m.predict, lambda Z: m.predict_proba(Z)[:, 1], data, t0, extra)
