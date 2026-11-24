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
    out["val_proba"] = [round(float(p), 6) for p in proba(data["X_val"])]   # for threshold/calibration analyses
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
    elif kind == "mlp_weighted":
        return _mlp_weighted(name, spec, data, run, seed, t0)
    else:
        raise ValueError(kind)
    m.fit(X, y)
    extra = {}
    if kind == "mlp_backprop":
        extra = {"epochs_run": int(m.n_iter_), "history": [float(v) for v in m.loss_curve_]}
    return _evaluate(name, run, m.predict, lambda Z: m.predict_proba(Z)[:, 1], data, t0, extra)


def _mlp_weighted(name, spec, data, run, seed, t0):
    """Backpropagation reference trained on the same problem as the other methods: the same head
    (ReLU hidden layers, two-unit softmax), class-weighted cross-entropy, Adam, and early stopping on the
    patient-grouped validation subset (weighted validation loss, patience `patience`, best epoch kept)."""
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    X = torch.tensor(data["X_opt"], dtype=torch.float32); y = torch.tensor(data["y_opt"], dtype=torch.long)
    Xv = torch.tensor(data["X_val"], dtype=torch.float32); yv = torch.tensor(data["y_val"], dtype=torch.long)
    cnt = np.bincount(data["y_opt"], minlength=2); w = torch.tensor(len(y) / (2.0 * cnt), dtype=torch.float32)
    sizes = [X.shape[1], *spec.get("hidden", [16]), 2]; layers = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if k < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=spec.get("lr", 1e-3), weight_decay=spec.get("weight_decay", 1e-4))
    loss = torch.nn.CrossEntropyLoss(weight=w); bs = spec.get("batch_size", 64)
    g = torch.Generator().manual_seed(seed); best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(spec.get("epochs", 150)):
        net.train()
        for idx in torch.randperm(len(y), generator=g).split(bs):
            opt.zero_grad(); loss(net(X[idx]), y[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss(net(Xv), yv))
        hist.append(vl)
        if vl < best - 1e-6:
            best, best_state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= spec.get("patience", 15):
                break
    net.load_state_dict(best_state); net.eval()

    def proba(Z):
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(Z, dtype=torch.float32)), 1)[:, 1].numpy().astype(float)
    return _evaluate(name, run, lambda Z: (proba(Z) >= 0.5).astype(int), proba, data, t0,
                     {"epochs_run": ep + 1, "history": hist})
