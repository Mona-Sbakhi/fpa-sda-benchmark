"""Classification metrics computed from one set of predictions, so every reported number
(tables, box plots, confusion-matrix annotations) derives from the same confusion matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_pos: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn

    def div(a, b):
        return float(a / b) if b else 0.0

    prec, rec, spec = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec, "recall": rec, "specificity": spec,
        "f1": div(2 * prec * rec, prec + rec),
        "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0,
    }
    if p_pos is not None and len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p_pos))
    return out
