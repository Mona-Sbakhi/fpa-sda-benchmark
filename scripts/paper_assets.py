#!/usr/bin/env python3
"""Generate every table (LaTeX) and figure (vector PDF + 300-dpi PNG) used in the manuscript
directly from the per-run JSON files in results/.  Nothing is typed by hand.

    python scripts/paper_assets.py            # writes paper/tables/*.tex and paper/figures/*
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpasda.stats import bootstrap_ci, friedman, nemenyi, wilcoxon_vs  # noqa: E402

RES, TAB, FIG = ROOT / "results", ROOT / "paper" / "tables", ROOT / "paper" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})


def sci(p):
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def load(*dirs):
    runs = {}
    for d in dirs:
        for p in sorted((RES / d / "runs").glob("*/run_*.json")):
            r = json.loads(p.read_text()); runs.setdefault(r["algorithm"], {})[r["run"]] = r
    return runs


def vals(runs, a, m, ids=None):
    ids = sorted(runs[a]) if ids is None else ids
    return np.array([runs[a][i]["test"][m] if m != "training_time" else runs[a][i]["training_time"] for i in ids], float)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def tex_name(a):
    return a.replace("_", r"\_")


# ---------------------------------------------------------------- main CNN-head results (50 runs)
cnn = load("cnn_head", "cnn_redesign")
GF = ["FPA", "SDA", "Hybrid", "Hybrid-reinit", "Random", "PSO", "DE", "GA", "L-SHADE", "APH"]
MEMETIC = ["Adam-only", "Memetic-Hybrid", "Memetic-APH"]
GRAD = ["Majority", "LogReg", "MLP-SGD", "MLP-Adam"]
LABEL = {"Hybrid": "Hybrid FPA--SDA (transfer)", "Hybrid-reinit": "Hybrid FPA--SDA (re-init)",
         "Random": "Random search", "APH": "APH (redesigned hybrid)", "LogReg": "Logistic regression",
         "MLP-SGD": "MLP, backprop (SGD)", "MLP-Adam": "MLP, backprop (Adam)", "Majority": "Majority class",
         "Adam-only": "Adam from random init.", "Memetic-Hybrid": "Hybrid + Adam (memetic)",
         "Memetic-APH": "APH + Adam (memetic)"}
lab = lambda a: LABEL.get(a, a)  # noqa: E731
METRICS = [("accuracy", "Acc. (\\%)", 100), ("balanced_accuracy", "Bal. acc. (\\%)", 100),
           ("recall", "Sens. (\\%)", 100), ("specificity", "Spec. (\\%)", 100),
           ("mcc", "MCC", 1), ("auc", "AUC", 1)]


def results_table(runs, groups, caption, label, path, time=True):
    cols = "l" + "c" * (len(METRICS) + (1 if time else 0))
    head = " & ".join(["Method"] + [h for _, h, _ in METRICS] + (["Time (s)"] if time else []))
    lines = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + cols + "}", r"\toprule",
             head + r" \\", r"\midrule"]
    for gname, algos in groups:
        lines.append(r"\multicolumn{" + str(len(METRICS) + 1 + (1 if time else 0)) + r"}{l}{\textit{" + gname + r"}} \\")
        best = {m: max(vals(runs, a, m).mean() for a in algos if a in runs) for m, _, _ in METRICS}
        for a in algos:
            if a not in runs:
                continue
            cells = [lab(a)]
            for m, _, k in METRICS:
                v = vals(runs, a, m)
                s = f"{k*v.mean():.2f}" if k == 100 else f"{v.mean():.3f}"
                sd = f"{k*v.std(ddof=1):.2f}" if k == 100 else f"{v.std(ddof=1):.3f}"
                cell = f"{s}$\\pm${sd}"
                cells.append(r"\textbf{" + cell + "}" if np.isclose(v.mean(), best[m]) and len(algos) > 1 else cell)
            if time:
                cells.append(f"{vals(runs, a, 'training_time').mean():.1f}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}"]
    (TAB / path).write_text("\n".join(lines) + "\n")


results_table(cnn, [("Gradient-free optimisers (6,795 fitness evaluations each)", GF),
                    ("Metaheuristic + gradient (memetic) and gradient-only control", MEMETIC),
                    ("Reference classifiers trained by gradient methods", GRAD)],
              "Test-set performance on frozen ResNet-50 features (2048--16--2 head, $D=32{,}818$); mean$\\pm$SD over 50 independent runs with paired seeds. Best value per group in bold.",
              "tab:cnn_main", "cnn_main.tex")

# ---------------------------------------------------------------- Friedman / Nemenyi / Wilcoxon (MCC)
ids50 = list(range(50))
blocks = {a: vals(cnn, a, "mcc", ids50) for a in GF}
fr = friedman(blocks)
nm = nemenyi(blocks)
k, N = len(GF), len(ids50)
q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q * np.sqrt(k * (k + 1) / (6 * N))
order = sorted(GF, key=lambda a: fr["mean_rank"][a])
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Friedman test over the ten gradient-free optimisers (MCC, $N=50$ paired runs): "
         f"$\\chi^2_F={fr['statistic']:.2f}$, $p={sci(fr['p'])}$. Mean ranks (1 = best) and Nemenyi critical difference "
         f"$\\mathrm{{CD}}_{{0.05}}={CD:.3f}$. Nemenyi $p$-values are given against the original hybrid and against APH." + "}",
         r"\label{tab:friedman}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
         r"Optimiser & Mean rank & $p$ vs Hybrid & $p$ vs APH \\", r"\midrule"]
fmtp = lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}"  # noqa: E731
for a in order:
    p1 = "--" if a == "Hybrid" else fmtp(nm.loc[a, "Hybrid"]); p2 = "--" if a == "APH" else fmtp(nm.loc[a, "APH"])
    lines.append(f"{lab(a)} & {fr['mean_rank'][a]:.2f} & {p1} & {p2} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "friedman.tex").write_text("\n".join(lines) + "\n")
nm.to_csv(TAB / "nemenyi_mcc_cnn.csv")

wx_rows = []
for ref, others in [("Hybrid", ["FPA", "SDA", "Hybrid-reinit", "Random", "PSO", "DE", "GA", "L-SHADE", "APH"]),
                    ("SDA", ["FPA"]),
                    ("APH", ["L-SHADE", "GA", "PSO", "DE"]),
                    ("Memetic-Hybrid", ["Adam-only", "MLP-Adam"])]:
    w = wilcoxon_vs(vals(cnn, ref, "mcc", ids50), {o: vals(cnn, o, "mcc", ids50) for o in others})
    for o, s in w.items():
        wx_rows.append((ref, o, s))
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Pairwise Wilcoxon signed-rank tests on MCC (50 paired runs), Holm-adjusted within each block. "
         r"$\bar\Delta$: difference of means; $\tilde\Delta$: median of the 50 paired differences (first method minus second). "
         r"$\hat A$: Vargha--Delaney effect size ($>0.5$ favours the first method; $|\hat A-0.5|\ge 0.21$ is large).}",
         r"\label{tab:wilcoxon}", r"\footnotesize", r"\begin{tabular}{llrrcc}", r"\toprule",
         r"Method & vs & $\bar\Delta$ & $\tilde\Delta$ & $p_{\mathrm{Holm}}$ & $\hat A$ \\", r"\midrule"]
prev = None
for ref, o, s in wx_rows:
    if prev and prev != ref:
        lines.append(r"\midrule")
    dm = vals(cnn, ref, "mcc", ids50).mean() - vals(cnn, o, "mcc", ids50).mean()
    lines.append(f"{lab(ref) if ref != prev else ''} & {lab(o)} & {dm:+.3f} & {s['median_diff']:+.3f} & {fmtp(s['p_holm'])} & {s['A']:.2f} \\\\")
    prev = ref
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "wilcoxon.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- CD diagram (Demsar 2006 style)
fig, ax = plt.subplots(figsize=(7.2, 2.9))
lo, hi = 1, k
ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(-0.05, 1.05); ax.axis("off")
ax.hlines(0.85, lo, hi, color="k", lw=0.8)
for t in range(lo, hi + 1):
    ax.vlines(t, 0.85, 0.88, color="k", lw=0.8); ax.text(t, 0.91, str(t), ha="center", fontsize=8)
ax.hlines(0.98, lo, lo + CD, color="k", lw=1.5); ax.vlines([lo, lo + CD], 0.96, 1.0, color="k", lw=1.0)
ax.text(lo + CD / 2, 1.01, f"CD = {CD:.2f}", ha="center", fontsize=8)
ranks = [fr["mean_rank"][a] for a in order]
half = (len(order) + 1) // 2
for j, (a, r) in enumerate(zip(order, ranks)):
    left = j < half
    y = 0.58 - 0.12 * (j if left else j - half)
    xt = lo - 0.25 if left else hi + 0.25
    ax.plot([r, r, xt], [0.85, y, y], color="k", lw=0.7)
    ax.text(xt + (-0.03 if left else 0.03), y, f"{a} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=8)
# cliques: maximal groups whose rank span < CD
cliques, s = [], sorted(ranks)
for i in range(len(s)):
    j = max(jj for jj in range(i, len(s)) if s[jj] - s[i] < CD)
    if j > i and not any(c[0] <= i and c[1] >= j for c in cliques):
        cliques.append((i, j))
for n_, (i, j) in enumerate(cliques):
    yb = 0.80 - 0.045 * n_
    ax.hlines(yb, s[i] - 0.03, s[j] + 0.03, color="k", lw=2.5)
save(fig, "cd_diagram")
name_by_rank = {fr["mean_rank"][a]: a for a in GF}
CDGROUPS = "; ".join("\\{" + ", ".join(name_by_rank[r] for r in s[i:j + 1]) + "\\}" for i, j in cliques)

# ---------------------------------------------------------------- MCC box plots
fig, ax = plt.subplots(figsize=(7.2, 3.0))
show = order[::-1] + ["Memetic-APH", "MLP-Adam", "LogReg"]
data = [vals(cnn, a, "mcc") for a in show]
bp = ax.boxplot(data, orientation='horizontal', tick_labels=show, showmeans=True, widths=0.6,
                meanprops=dict(marker="D", markersize=3, markerfacecolor="white", markeredgecolor="k"),
                medianprops=dict(color="k"), flierprops=dict(markersize=2))
for patch_i, a in enumerate(show):
    bp["boxes"][patch_i].set_color("#555" if a in GF else "#1f5fa8")
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_xlabel("Test MCC (50 runs)")
save(fig, "mcc_boxplots")

# ---------------------------------------------------------------- convergence
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for a in ["FPA", "SDA", "Hybrid", "GA", "L-SHADE", "APH"]:
    H = np.array([cnn[a][i]["history"][:151] for i in sorted(cnn[a])])
    x = np.arange(H.shape[1]); med = np.median(H, 0)
    ax.plot(x, med, lw=1, label=a); ax.fill_between(x, np.quantile(H, .25, 0), np.quantile(H, .75, 0), alpha=0.15)
ax.axvline(75, color="grey", ls="--", lw=0.6); ax.text(76, ax.get_ylim()[1] * 0.9 if False else 0, "", fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("Iteration ($\\times$45 evaluations)"); ax.set_ylabel("Best fitness")
ax.legend(fontsize=6, frameon=False, ncol=2)
save(fig, "convergence")
Hh = np.median(np.log(np.array([cnn["Hybrid"][i]["history"][:151] for i in sorted(cnn["Hybrid"])])), 0)
SLOPE_PRE, SLOPE_POST = (Hh[50] - Hh[75]) / 25, (Hh[75] - Hh[100]) / 25

# ---------------------------------------------------------------- tau study
tau = load("cnn_tau_ablation")
ids20 = list(range(20))
T = [0, 25, 50, 75, 100, 125, 150]          # tau = 0 is SDA alone, tau = 150 is FPA alone (same seeds)
def tau_vals(t, m):
    if t == 0:
        return vals(cnn, "SDA", m, ids20)
    if t == 150:
        return vals(cnn, "FPA", m, ids20)
    return vals(tau, f"Hybrid-tau{t}", m, ids20)
d = [tau_vals(t, "mcc") for t in T]
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.boxplot(d, tick_labels=[("0\n(SDA)" if t == 0 else "150\n(FPA)" if t == 150 else str(t)) for t in T],
           widths=0.5, medianprops=dict(color="k"))
ax.plot(range(1, len(T) + 1), [v.mean() for v in d], "o-", color="#1f5fa8", ms=3, lw=1)
ax.tick_params(axis="x", labelsize=7)
ax.set_xlabel("Transition iteration $\\tau$ (of $T=150$)"); ax.set_ylabel("Test MCC (20 runs)")
save(fig, "tau_study")
inner = [25, 50, 75, 100, 125]
frt = friedman({t: tau_vals(t, "mcc") for t in inner})
# Page's L test for a monotone trend (alternative: MCC decreases as tau increases), all 7 settings
page = stats.page_trend_test(np.column_stack(d[::-1]))
wt = wilcoxon_vs(tau_vals(75, "mcc"), {t: tau_vals(t, "mcc") for t in T if t != 75})
adj = {}
for t0, t1 in zip(T[:-1], T[1:]):
    adj[(t0, t1)] = stats.wilcoxon(tau_vals(t0, "mcc"), tau_vals(t1, "mcc")).pvalue
from fpasda.stats import holm  # noqa: E402
adj_h = holm({f"{a_}-{b_}": p_ for (a_, b_), p_ in adj.items()})
n_mono = int(sum(np.all(np.diff(np.column_stack(d)[i]) < 0) for i in range(20)))
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Sensitivity of the hybrid to the transition iteration $\tau$ (20 paired runs; $\tau=0$ is SDA alone and $\tau=150$ FPA alone, same seeds). "
         f"Friedman over $\\tau\\in\\{{25,\\dots,125\\}}$: $\\chi^2_F={frt['statistic']:.2f}$, $p={sci(frt['p'])}$. "
         f"Page's trend test over all seven settings: $L={page.statistic:.0f}$, $p={sci(page.pvalue)}$. $p_{{\\mathrm{{Holm}}}}$: Wilcoxon against $\\tau=75$." + "}",
         r"\label{tab:tau}", r"\footnotesize", r"\begin{tabular}{l" + "c" * len(T) + "}", r"\toprule",
         r"$\tau$ & " + " & ".join(map(str, T)) + r" \\", r"\midrule"]
for m, nm_, kk in [("accuracy", "Accuracy (\\%)", 100), ("mcc", "MCC", 1), ("auc", "AUC", 1)]:
    vv = [tau_vals(t, m).mean() * kk for t in T]
    lines.append(nm_ + " & " + " & ".join(f"{v:.2f}" if kk == 100 else f"{v:.3f}" for v in vv) + r" \\")
lines.append("$p_{\\mathrm{Holm}}$ (MCC) & " + " & ".join("--" if t == 75 else fmtp(wt[t]["p_holm"]) for t in T) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "tau.tex").write_text("\n".join(lines) + "\n")
TAU_ADJ_MIN = min(adj_h.values())

# ---------------------------------------------------------------- ablation
ab = load("cnn_redesign_ablation", "cnn_redesign")
AB = ["APH", "APH-uniform-ops", "APH-no-crossover", "APH-plant-only", "L-SHADE", "OpenAI-ES", "ZO-SGD"]
ABL = {"APH": "APH (full)", "APH-uniform-ops": "-- adaptive operator selection (uniform)",
       "APH-no-crossover": "-- binomial crossover (full-vector moves)",
       "APH-plant-only": "-- current-to-$p$best operator (plant operators only)",
       "L-SHADE": "L-SHADE", "OpenAI-ES": "OpenAI-ES", "ZO-SGD": "ZO-SGD (SPSA)"}
w = wilcoxon_vs(vals(ab, "APH", "mcc", ids20), {a: vals(ab, a, "mcc", ids20) for a in AB[1:]})
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Component ablation of APH and cross-field gradient-free baselines (20 paired runs). "
         r"$\Delta$ and $p_{\mathrm{Holm}}$: Wilcoxon signed-rank test of APH against each variant on MCC.}",
         r"\label{tab:ablation}", r"\footnotesize", r"\begin{tabular}{lccrc}", r"\toprule",
         r"Variant & MCC & AUC & $\Delta$ vs APH & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for a in AB:
    m = vals(ab, a, "mcc", ids20); u = vals(ab, a, "auc", ids20)
    dd = "--" if a == "APH" else f"{-w[a]['median_diff']:+.3f}"; pp = "--" if a == "APH" else fmtp(w[a]["p_holm"])
    lines.append(f"{ABL[a]} & {m.mean():.3f} & {u.mean():.3f} & {dd} & {pp} \\\\")
    if a == "APH-plant-only":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "ablation.tex").write_text("\n".join(lines) + "\n")
share = {k_: np.mean([cnn["APH"][i]["extra"]["operator_share"][k_] for i in ids50]) for k_ in ["global", "local", "animal", "gravity"]}

# ---------------------------------------------------------------- PCA experiments (10 runs)
pr, po = load("pca_mlp_revised"), load("pca_mlp_original_setting")
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Original pipeline: PCA(1000) pixel features and a 1000--128--64--32--2 network ($D=138{,}530$); mean over 10 runs. "
         r"Raw PCA scores reproduce the original notebook; standardised scores are the variant in which every component has unit variance.}",
         r"\label{tab:pca}", r"\footnotesize", r"\begin{tabular}{llccc}", r"\toprule",
         r"Features & Method & Acc. (\%) & MCC & AUC \\", r"\midrule"]
for nm_, R, algos in [("Raw PCA", po, ["SDA", "Hybrid", "Random", "FPA"]),
                      ("Standardised PCA", pr, ["Hybrid", "Hybrid-reinit", "FPA", "SDA", "PSO", "DE", "GA", "Random", "LogReg", "MLP-SGD", "MLP-Adam", "Majority"])]:
    for j, a in enumerate(algos):
        lines.append(f"{nm_ if j == 0 else ''} & {lab(a)} & {100*vals(R, a, 'accuracy').mean():.1f} & {vals(R, a, 'mcc').mean():.3f} & {vals(R, a, 'auc').mean():.3f} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(TAB / "pca.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- confusion matrices (median-MCC run)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
for ax, a in zip(axes, ["Hybrid", "GA", "APH", "MLP-Adam"]):
    rs = sorted(cnn[a].values(), key=lambda r: r["test"]["mcc"]); r = rs[len(rs) // 2]["test"]
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=390)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > 200 else "black", fontsize=8)
    ax.set_title(f"{a}\nacc {(cm[0,0]+cm[1,1])/cm.sum():.3f}, MCC {r['mcc']:.3f}", fontsize=7)
    ax.set_xticks([0, 1], ["N", "P"], fontsize=7); ax.set_yticks([0, 1], ["N", "P"], fontsize=7)
save(fig, "confusion")

# ---------------------------------------------------------------- test-image bootstrap (sampling uncertainty of the fixed test set)
LABELS = np.load(RES / "labels.npz")          # committed, so a fresh clone can rebuild every table
y_te = LABELS["y_test"].astype(bool)
PRED = {a: np.array([np.array(cnn[a][i]["test_proba"]) >= 0.5 for i in sorted(cnn[a])]) for a in cnn}


def mcc_rows(P, y):
    tp = (P & y).sum(1); tn = (~P & ~y).sum(1); fp = (P & ~y).sum(1); fn = (~P & y).sum(1)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)
    return np.divide(tp * tn - fp * fn, den, out=np.zeros(len(P)), where=den > 0)


for a_ in ["Hybrid", "APH", "LogReg"]:   # stored metrics are reproduced from the stored probabilities
    assert np.allclose(mcc_rows(PRED[a_], y_te), vals(cnn, a_, "mcc")), a_
B_ = 2000
rng_b = np.random.default_rng(12345)
IDX = rng_b.integers(0, len(y_te), (B_, len(y_te)))
boot_mean = {}
for a_ in set(sum([[x, y] for x, y in [("APH", "GA"), ("APH", "L-SHADE"), ("Hybrid", "SDA"), ("Hybrid", "FPA"), ("L-SHADE", "GA"),
                                         ("LogReg", "L-SHADE"), ("MLP-Adam", "APH"), ("Memetic-APH", "MLP-Adam")]], [])):
    boot_mean[a_] = np.array([mcc_rows(PRED[a_][:, ix], y_te[ix]).mean() for ix in IDX])
CONTRASTS = [("APH", "GA"), ("APH", "L-SHADE"), ("L-SHADE", "GA"), ("Hybrid", "SDA"), ("Hybrid", "FPA"),
             ("LogReg", "L-SHADE"), ("MLP-Adam", "APH"), ("Memetic-APH", "MLP-Adam")]
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Uncertainty due to the fixed 624-image test set. For each contrast, test images are resampled with replacement (2,000 bootstrap samples) and the difference in run-averaged MCC is recomputed from the stored predictions; seeds are held fixed. "
         r"Intervals that include 0 indicate contrasts not resolved by this test set, even when they are significant across seeds (Table~\ref{tab:wilcoxon}).}",
         r"\label{tab:bootstrap}", r"\footnotesize", r"\begin{tabular}{llrc}", r"\toprule",
         r"Method & vs & $\bar\Delta$ MCC & 95\% CI (test images) \\", r"\midrule"]
BOOT = {}
for x, y in CONTRASTS:
    dd = boot_mean[x] - boot_mean[y]; obs = vals(cnn, x, "mcc").mean() - vals(cnn, y, "mcc").mean()
    lo_, hi_ = np.quantile(dd, [0.025, 0.975]); BOOT[(x, y)] = (obs, lo_, hi_)
    lines.append(f"{lab(x)} & {lab(y)} & {obs:+.3f} & [{lo_:+.3f}, {hi_:+.3f}] \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "bootstrap.tex").write_text("\n".join(lines) + "\n")
one = np.array([mcc_rows(PRED["LogReg"][:1, ix], y_te[ix])[0] for ix in IDX])
ONE_HALF = (np.quantile(one, 0.975) - np.quantile(one, 0.025)) / 2

# ---------------------------------------------------------------- validation vs test
VAL_ALGOS = [a_ for a_ in GF + MEMETIC + GRAD if a_ != "Majority"]
vm = [np.mean([cnn[a_][i]["val"]["mcc"] for i in cnn[a_]]) for a_ in VAL_ALGOS]
tm = [vals(cnn, a_, "mcc").mean() for a_ in VAL_ALGOS]
RHO = stats.spearmanr(vm, tm).statistic
LR_VAL_SPEC = np.mean([cnn["LogReg"][i]["val"]["specificity"] for i in cnn["LogReg"]])
LR_TEST_SPEC = vals(cnn, "LogReg", "specificity").mean()
LR_VAL_MCC = np.mean([cnn["LogReg"][i]["val"]["mcc"] for i in cnn["LogReg"]])

# ---------------------------------------------------------------- numbers quoted in the text
macros = {"CDvalue": f"{CD:.2f}", "FriedmanChi": f"{fr['statistic']:.1f}", "FriedmanP": sci(fr["p"]),
          "ShareAnimal": f"{100*share['animal']:.1f}", "ShareLocal": f"{100*share['local']:.1f}",
          "ShareGravity": f"{100*share['gravity']:.1f}", "ShareGlobal": f"{100*share['global']:.1f}",
          "SharePlant": f"{100*(1-share['animal']):.1f}", "CDGroups": CDGROUPS,
          "SlopePre": f"{SLOPE_PRE:.4f}", "SlopePost": f"{SLOPE_POST:.4f}", "SlopeRatio": f"{SLOPE_POST/SLOPE_PRE:.1f}",
          "PageL": f"{page.statistic:.0f}", "PageP": sci(page.pvalue), "TauMonoRuns": str(n_mono), "TauAdjMin": f"{TAU_ADJ_MIN:.2f}",
          "TauFriedmanP": sci(frt["p"]), "OneModelCI": f"{ONE_HALF:.3f}", "ValTestRho": f"{RHO:.3f}",
          "LRValSpec": f"{100*LR_VAL_SPEC:.1f}", "LRTestSpec": f"{100*LR_TEST_SPEC:.1f}", "LRValMCC": f"{LR_VAL_MCC:.3f}",
          "EvalsReinit": str(int(np.median([cnn['Hybrid-reinit'][i]['n_evals'] for i in cnn['Hybrid-reinit']]))),
          "EvalsTransfer": str(int(np.median([cnn['Hybrid'][i]['n_evals'] for i in cnn['Hybrid']])))}
for (x, y), (o_, l_, h_) in BOOT.items():
    key = "Boot" + "".join(ch for ch in (x + y).title() if ch.isalpha())
    macros[key] = f"{o_:+.3f} [{l_:+.3f}, {h_:+.3f}]"
for a in GF + MEMETIC + GRAD:
    key = "".join(ch for ch in a.title() if ch.isalpha())
    macros[f"MCC{key}"] = f"{vals(cnn, a, 'mcc').mean():.3f}"
    macros[f"ACC{key}"] = f"{100*vals(cnn, a, 'accuracy').mean():.1f}"
(TAB / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k_}}}{{{v}}}\n" for k_, v in macros.items()))
print("CD =", round(CD, 3), "Friedman", round(fr["statistic"], 2), fr["p"]); print(sorted(fr["mean_rank"].items(), key=lambda kv: kv[1]))
print("operator share", {k_: round(v, 3) for k_, v in share.items()})
print("wrote", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))

# ================================================================ additions after the third review
from fpasda.stats import holm as _holm  # noqa: E402
from fpasda.data import patient_id  # noqa: E402
X = {}                                    # extra macros


def fitv(a, ids=ids50):
    return np.array([cnn[a][i]["best_fitness"] for i in ids])


def valv(a, ids=ids50):
    return np.array([cnn[a][i]["val"]["mcc"] for i in ids])


# ---- three endpoints for the ten gradient-free optimisers
fr_fit = friedman({a: -fitv(a) for a in GF}); fr_val = friedman({a: valv(a) for a in GF})
GF8 = [a for a in GF if a not in ("L-SHADE", "APH")]
fr8 = friedman({a: vals(cnn, a, "mcc", ids50) for a in GF8})
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Three endpoints for the gradient-free optimisers (50 paired runs): final training fitness (the quantity optimised; median), validation MCC and test MCC (means), with mean Friedman ranks (1 = best). "
         f"Friedman tests: fitness $\\chi^2_F={fr_fit['statistic']:.1f}$, validation MCC $\\chi^2_F={fr_val['statistic']:.1f}$, test MCC $\\chi^2_F={fr['statistic']:.1f}$ (all $p<10^{{-40}}$)." + "}",
         r"\label{tab:endpoints}", r"\footnotesize", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Training fitness} & \multicolumn{2}{c}{Validation MCC} & \multicolumn{2}{c}{Test MCC} \\",
         r"Optimiser & median & rank & mean & rank & mean & rank \\", r"\midrule"]
for a in sorted(GF, key=lambda a: fr_fit["mean_rank"][a]):
    lines.append(f"{lab(a)} & {np.median(fitv(a)):.3f} & {fr_fit['mean_rank'][a]:.2f} & {valv(a).mean():.3f} & {fr_val['mean_rank'][a]:.2f} & "
                 f"{vals(cnn, a, 'mcc', ids50).mean():.3f} & {fr['mean_rank'][a]:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "endpoints.tex").write_text("\n".join(lines) + "\n")

PAIRS = [("Hybrid", "SDA"), ("Hybrid", "Hybrid-reinit"), ("Hybrid", "DE"), ("Hybrid", "FPA"), ("SDA", "FPA"), ("APH", "L-SHADE"), ("APH", "GA")]
raw = {"fit": {}, "val": {}, "test": {}}; win = {"fit": {}, "val": {}, "test": {}}
for x, y in PAIRS:
    for key, fx, better in [("fit", fitv, lambda u, v: u < v), ("val", valv, lambda u, v: u > v),
                            ("test", lambda a_: vals(cnn, a_, "mcc", ids50), lambda u, v: u > v)]:
        u, v = fx(x), fx(y); raw[key][(x, y)] = stats.wilcoxon(u, v).pvalue; win[key][(x, y)] = int(better(u, v).sum())
adjp = {k_: _holm({f"{x}|{y}": p_ for (x, y), p_ in raw[k_].items()}) for k_ in raw}
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Key contrasts on the three endpoints (50 paired runs). ``Wins'': number of runs in which the first method is better (lower fitness, higher MCC). $p$: Wilcoxon signed-rank, Holm-adjusted over the seven contrasts of each endpoint.}",
         r"\label{tab:contrasts}", r"\footnotesize", r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{Training fitness} & \multicolumn{2}{c}{Validation MCC} & \multicolumn{2}{c}{Test MCC} \\",
         r"Method & vs & wins & $p$ & wins & $p$ & wins & $p$ \\", r"\midrule"]
for x, y in PAIRS:
    cells = []
    for k_ in ["fit", "val", "test"]:
        cells += [f"{win[k_][(x, y)]}/50", fmtp(adjp[k_][f"{x}|{y}"])]
        X[f"Win{k_.title()}{''.join(c for c in (x + y).title() if c.isalpha())}"] = str(win[k_][(x, y)])
    lines.append(f"{lab(x)} & {lab(y)} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "contrasts.tex").write_text("\n".join(lines) + "\n")
X.update({"FriedmanEightChi": f"{fr8['statistic']:.1f}", "FriedmanEightP": sci(fr8["p"]),
          "FitHybrid": f"{np.median(fitv('Hybrid')):.2f}", "FitSda": f"{np.median(fitv('SDA')):.2f}", "FitDe": f"{np.median(fitv('DE')):.2f}",
          "FitAph": f"{np.median(fitv('APH')):.3f}", "FitLshade": f"{np.median(fitv('L-SHADE')):.3f}",
          "ValAph": f"{valv('APH').mean():.3f}", "ValLshade": f"{valv('L-SHADE').mean():.3f}",
          "ValHybrid": f"{valv('Hybrid').mean():.3f}", "ValSda": f"{valv('SDA').mean():.3f}"})
d_ = vals(cnn, "SDA", "mcc", ids50) - vals(cnn, "FPA", "mcc", ids50)
X["SdaFpaDelta"] = f"{d_.mean():+.3f}"; X["SdaFpaP"] = sci(stats.wilcoxon(vals(cnn, 'SDA', 'mcc', ids50), vals(cnn, 'FPA', 'mcc', ids50)).pvalue)

# ---- ablation table rebuilt: mean differences and test-image bootstrap intervals
PA = {a: np.array([np.array(ab[a][i]["test_proba"]) >= 0.5 for i in ids20]) for a in AB}
bm = {a: np.array([mcc_rows(PA[a][:, ix], y_te[ix]).mean() for ix in IDX]) for a in AB}
lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Component ablation of APH and cross-field gradient-free baselines (20 paired runs). $\bar\Delta$: mean MCC of the variant minus APH; CI: 95\% test-image bootstrap interval of $\bar\Delta$; "
         r"$p_{\mathrm{Holm}}$: Wilcoxon signed-rank test against APH across runs.}",
         r"\label{tab:ablation}", r"\footnotesize", r"\begin{tabular}{lccrcc}", r"\toprule",
         r"Variant & MCC & AUC & $\bar\Delta$ & CI & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for a in AB:
    m = vals(ab, a, "mcc", ids20); u = vals(ab, a, "auc", ids20)
    if a == "APH":
        lines.append(f"{ABL[a]} & {m.mean():.3f} & {u.mean():.3f} & -- & -- & -- \\\\"); continue
    dd = bm[a] - bm["APH"]; lo_, hi_ = np.quantile(dd, [0.025, 0.975])
    lines.append(f"{ABL[a]} & {m.mean():.3f} & {u.mean():.3f} & {m.mean() - vals(ab, 'APH', 'mcc', ids20).mean():+.3f} & [{lo_:+.3f}, {hi_:+.3f}] & {fmtp(w[a]['p_holm'])} \\\\")
    if a == "APH-plant-only":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "ablation.tex").write_text("\n".join(lines) + "\n")
X["AblUniformVsNoCx"] = f"{stats.wilcoxon(vals(ab, 'APH-uniform-ops', 'mcc', ids20), vals(ab, 'APH-no-crossover', 'mcc', ids20)).pvalue:.2f}"
for a, k_ in [("APH-uniform-ops", "AblUniform"), ("APH-no-crossover", "AblNoCx"), ("APH-plant-only", "AblPlant")]:
    X[k_] = f"{vals(ab, a, 'mcc', ids20).mean() - vals(ab, 'APH', 'mcc', ids20).mean():+.3f}"

# ---- single-model test-set uncertainty (median-MCC run of four methods)
halves = []
for a in ["LogReg", "Hybrid", "GA", "APH"]:
    order_ = np.argsort(vals(cnn, a, "mcc")); row = PRED[a][order_[len(order_) // 2]][None, :]
    b_ = np.array([mcc_rows(row[:, ix], y_te[ix])[0] for ix in IDX]); halves.append((np.quantile(b_, .975) - np.quantile(b_, .025)) / 2)
X["OneModelLo"], X["OneModelHi"] = f"{min(halves):.2f}", f"{max(halves):.2f}"

# ---- patient-ID overlap sensitivity
fc = LABELS
ptr_ = {patient_id(str(f_)) for f_ in fc["train_files"]}
keep = np.array([patient_id(str(f_)) not in ptr_ for f_ in fc["test_files"]])
ALLM = [a for a in GF + MEMETIC + GRAD if a != "Majority"]
full = [vals(cnn, a, "mcc").mean() for a in ALLM]
sub = [mcc_rows(PRED[a][:, keep], y_te[keep]).mean() for a in ALLM]
lr = PRED["LogReg"][0]
X.update({"NoOverlapN": str(int(keep.sum())), "NoOverlapRho": f"{stats.spearmanr(full, sub).statistic:.3f}",
          "LRSensOverlap": f"{100 * lr[~keep & y_te].mean():.1f}", "LRSensOther": f"{100 * lr[keep & y_te].mean():.1f}",
          "MlpSgdPcaAcc": f"{100 * vals(pr, 'MLP-SGD', 'accuracy').mean():.1f}"})

# ---- fine-tuning: official split and patient-grouped cross-validation
FT, CV = RES / "finetune_resnet50", RES / "finetune_cv"
if FT.exists() and CV.exists():
    R5 = [json.loads((FT / f"seed_{s_}.json").read_text()) for s_ in range(5)]
    cvf = np.load(CV / "cv_folds.npz"); fold = cvf["fold"]
    lab_tr = LABELS
    y_all = np.concatenate([lab_tr["y_train"], lab_tr["y_test"]]).astype(bool)
    from sklearn.metrics import roc_auc_score  # noqa: E402
    P3 = np.zeros((3, len(y_all)))
    for s_ in range(3):
        for k_ in range(5):
            r_ = json.loads((CV / f"seed{s_}_fold{k_}.json").read_text()); P3[s_, fold == k_] = r_["test_proba"]

    def cm(yb, pr_):
        tp = (pr_ & yb).sum(); tn = (~pr_ & ~yb).sum(); fp = (pr_ & ~yb).sum(); fn = (~pr_ & yb).sum()
        return [100 * (tp + tn) / len(yb), 100 * tp / (tp + fn), 100 * tn / (tn + fp),
                (tp * tn - fp * fn) / np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))]
    off = np.array([[100 * r_["test"]["accuracy"], 100 * r_["test"]["recall"], 100 * r_["test"]["specificity"], r_["test"]["mcc"]] for r_ in R5])
    offauc = [roc_auc_score(y_te, r_["test_proba"]) for r_ in R5]
    cvm = np.array([cm(y_all, P3[s_] >= 0.5) for s_ in range(3)]); cvauc = [roc_auc_score(y_all, P3[s_]) for s_ in range(3)]
    is_te = np.arange(len(y_all)) >= len(lab_tr["y_train"])
    fmt_ = lambda v_: f"{v_.mean():.2f}$\\pm${v_.std(ddof=1):.2f}"  # noqa: E731
    fmt3 = lambda v_: f"{np.mean(v_):.3f}$\\pm${np.std(v_, ddof=1):.3f}"  # noqa: E731
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{End-to-end fine-tuning of ResNet-50 by backpropagation (mean$\pm$SD over seeds). Official split: trained on the optimisation subset, epoch chosen on the validation subset, evaluated once on the official test folder (5 seeds). "
             r"Cross-validation: patient-grouped 5-fold over all 5,856 images, pooled out-of-fold predictions (3 seeds).}",
             r"\label{tab:finetune}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Protocol & Acc. (\%) & Sens. (\%) & Spec. (\%) & MCC & AUC \\", r"\midrule",
             f"Official test folder (624 images) & {fmt_(off[:,0])} & {fmt_(off[:,1])} & {fmt_(off[:,2])} & {fmt3(off[:,3])} & {fmt3(offauc)} \\\\",
             f"5-fold cross-validation (5,856 images) & {fmt_(cvm[:,0])} & {fmt_(cvm[:,1])} & {fmt_(cvm[:,2])} & {fmt3(cvm[:,3])} & {fmt3(cvauc)} \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "finetune.tex").write_text("\n".join(lines) + "\n")
    X.update({"FTOffAcc": f"{off[:,0].mean():.1f}", "FTOffSpec": f"{off[:,2].mean():.1f}", "FTOffMin": f"{off[:,0].min():.1f}", "FTOffMax": f"{off[:,0].max():.1f}",
              "FTCvAcc": f"{cvm[:,0].mean():.1f}", "FTCvSpec": f"{cvm[:,2].mean():.1f}", "FTCvMcc": f"{cvm[:,3].mean():.3f}",
              "FTCvAccTestFolder": f"{100*np.mean([((P3[s_][is_te] >= 0.5) == y_all[is_te]).mean() for s_ in range(3)]):.1f}",
              "FTCvAccTrainFolder": f"{100*np.mean([((P3[s_][~is_te] >= 0.5) == y_all[~is_te]).mean() for s_ in range(3)]):.1f}",
              "FTValAcc": f"{100*np.mean([r_['val']['accuracy'] for r_ in R5]):.1f}",
              "CvPatients": str(len(set(cvf["groups"])))})

    # head training inside the cross-validation (seed-0 CNN features): folds x runs
    HM = ["FPA@glorot", "SDA@glorot", "Hybrid@box", "Hybrid@glorot", "GA@glorot", "L-SHADE@glorot", "LogReg", "MLP-Adam"]
    HL = {"FPA@glorot": "FPA", "SDA@glorot": "SDA", "Hybrid@box": "Hybrid FPA--SDA, $[-1,1]$ box", "Hybrid@glorot": "Hybrid FPA--SDA",
          "GA@glorot": "GA", "L-SHADE@glorot": "L-SHADE", "LogReg": "Logistic regression", "MLP-Adam": "MLP, backprop (Adam)"}
    H = {}
    for k_ in range(5):
        for p_ in sorted((RES / f"cv_head_fold{k_}" / "runs").glob("*/run_*.json")):
            r_ = json.loads(p_.read_text()); H.setdefault(r_["algorithm"], {})[(k_, r_["run"])] = r_["test"]
    nrun = min(len(H.get(a, {})) for a in HM) // 5 if all(a in H for a in HM) else 0
    if nrun >= 2:
        keys = [(k_, r_) for k_ in range(5) for r_ in range(nrun)]
        hv = lambda a, m: np.array([H[a][kk][m] for kk in keys])  # noqa: E731
        frh = friedman({a: hv(a, "mcc") for a in HM if a not in ("LogReg", "MLP-Adam")})
        wh = wilcoxon_vs(hv("Hybrid@glorot", "mcc"), {a: hv(a, "mcc") for a in HM if a != "Hybrid@glorot"})
        lines = [r"\begin{table}[!t]", r"\centering",
                 r"\caption{Classifier head trained by each optimiser on fine-tuned features inside the cross-validation "
                 f"(5 folds $\\times$ {nrun} runs; mean$\\pm$SD over the {5*nrun} fold--run pairs). Gradient-free optimisers search the Glorot-scaled space unless stated. "
                 r"$p_{\mathrm{Holm}}$: Wilcoxon signed-rank test against the hybrid on MCC.}",
                 r"\label{tab:cvhead}", r"\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Method & Acc. (\%) & Spec. (\%) & MCC & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
        for a in sorted(HM, key=lambda a: -hv(a, "mcc").mean()):
            ac, sp, mc_ = 100 * hv(a, "accuracy"), 100 * hv(a, "specificity"), hv(a, "mcc")
            lines.append(f"{HL[a]} & {ac.mean():.2f}$\\pm${ac.std(ddof=1):.2f} & {sp.mean():.2f}$\\pm${sp.std(ddof=1):.2f} & {mc_.mean():.3f}$\\pm${mc_.std(ddof=1):.3f} & "
                         + ("--" if a == "Hybrid@glorot" else fmtp(wh[a]["p_holm"])) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TAB / "cvhead.tex").write_text("\n".join(lines) + "\n")
        accs = {a: 100 * hv(a, "accuracy").mean() for a in HM}
        X.update({"CvHeadRuns": str(nrun), "CvHeadN": str(5 * nrun), "CvHeadHybrid": f"{accs['Hybrid@glorot']:.1f}", "CvHeadHybridBox": f"{accs['Hybrid@box']:.1f}",
                  "CvHeadMin": f"{min(accs.values()):.1f}", "CvHeadMax": f"{max(accs.values()):.1f}", "CvHeadFpa": f"{accs['FPA@glorot']:.1f}",
                  "CvHeadLshade": f"{accs['L-SHADE@glorot']:.1f}", "CvHeadMlp": f"{accs['MLP-Adam']:.1f}", "CvHeadLogreg": f"{accs['LogReg']:.1f}",
                  "CvHeadFriedmanP": sci(frh["p"]) if frh["p"] < 1e-3 else f"{frh['p']:.3f}"})
        print("cv head: runs per fold", nrun, {a: round(v_, 2) for a, v_ in accs.items()}, {a: round(v_["p_holm"], 4) for a, v_ in wh.items()})
with open(TAB / "numbers.tex", "a") as fh:
    fh.write("".join(f"\\newcommand{{\\{k_}}}{{{v_}}}\n" for k_, v_ in X.items()))
print("extra macros:", {k_: v_ for k_, v_ in X.items() if not k_.startswith("Win")})

# ---------------------------------------------------------------- fit single-column tables to the text width
for name in ["wilcoxon", "bootstrap", "tau", "ablation", "pca", "friedman", "endpoints", "contrasts", "finetune", "cvhead"]:
    if not (TAB / f"{name}.tex").exists():
        continue
    f = TAB / f"{name}.tex"; txt = f.read_text()
    if "\\resizebox" not in txt:
        a_, b_ = txt.index("\\begin{tabular}"), txt.index("\\end{tabular}") + len("\\end{tabular}")
        f.write_text(txt[:a_] + "\\resizebox{\\linewidth}{!}{%\n" + txt[a_:b_] + "}" + txt[b_:])
