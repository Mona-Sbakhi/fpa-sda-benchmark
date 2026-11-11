"""Statistics for paired multi-run comparisons (runs share seeds across algorithms)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def vargha_delaney_a(x, y) -> float:
    """P(X > Y) + 0.5 P(X = Y). 0.5 = no effect; >0.71 or <0.29 = large (Vargha & Delaney 2000)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def friedman(blocks: dict) -> dict:
    """blocks: {algorithm: values aligned by run}. Returns statistic, p, mean ranks."""
    names = list(blocks)
    M = np.column_stack([blocks[n] for n in names])
    stat, p = stats.friedmanchisquare(*M.T)
    ranks = stats.rankdata(-M, axis=1).mean(axis=0)  # rank 1 = best (higher metric)
    return {"statistic": float(stat), "p": float(p), "mean_rank": dict(zip(names, map(float, ranks))),
            "n_blocks": int(M.shape[0])}


def nemenyi(blocks: dict):
    import pandas as pd
    import scikit_posthocs as sp
    df = pd.DataFrame(blocks)
    return sp.posthoc_nemenyi_friedman(df.to_numpy()).set_axis(df.columns, axis=0).set_axis(df.columns, axis=1)


def wilcoxon_vs(ref, others: dict) -> dict:
    """Two-sided Wilcoxon signed-rank of `ref` against each paired competitor, Holm-adjusted."""
    raw, out = {}, {}
    for n, v in others.items():
        d = np.asarray(ref) - np.asarray(v)
        raw[n] = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(ref, v).pvalue)
    adj = holm(raw)
    for n, v in others.items():
        out[n] = {"p": raw[n], "p_holm": adj[n], "A": vargha_delaney_a(ref, v),
                  "median_diff": float(np.median(np.asarray(ref) - np.asarray(v)))}
    return out
