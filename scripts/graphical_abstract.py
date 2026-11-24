#!/usr/bin/env python3
"""Graphical abstract (Elsevier: landscape, at least 1328 x 531 px, readable at 13 x 5 cm)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1] if len(sys.argv) > 1 else "graphical_abstract"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GF, HY, REF = "#2a78d6", "#eb6834", "#9a9890"
mcc = [("Random search", .178, GF), ("FPA", .260, GF), ("FPA–SDA hybrid", .350, HY), ("SDA", .362, GF),
       ("DE", .402, GF), ("sep-CMA-ES", .432, GF), ("PSO", .498, GF), ("GA", .561, GF),
       ("APH (redesigned)", .581, GF), ("L-SHADE", .585, GF), ("MLP, backprop", .612, REF), ("Logistic regr.", .635, REF)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})
fig = plt.figure(figsize=(13.28, 5.31), facecolor=SURF)


def box(ax, x, y, w, h, title, lines, fc="#eef3fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec="#c9c8c2", lw=1))
    ax.text(x + 0.03, y + h - 0.05, title, weight="bold", fontsize=11.5, va="top")
    ax.text(x + 0.03, y + h - 0.12, "\n".join(lines), fontsize=9.8, va="top", color=INK, linespacing=1.45)


a1 = fig.add_axes([0.01, 0.04, 0.30, 0.90]); a1.axis("off"); a1.set_xlim(0, 1); a1.set_ylim(0, 1)
a1.text(0.0, 0.99, "Benchmark", weight="bold", fontsize=13.5, va="top")
box(a1, 0.015, 0.52, 0.965, 0.38, "Task", ["Pneumonia vs normal chest X-rays",
    "Frozen ResNet-50 features → 16-unit head",
    "32,818 weights, 6,795 fitness evaluations", "10 gradient-free optimisers, paired runs"])
box(a1, 0.015, 0.06, 0.965, 0.40, "Evaluation", ["Kermany (patient-grouped), 50 runs",
    "RSNA adults (14,013 images, by patient)", "Cross-dataset, 5× budget, second backbone",
    "Pre-registered external test (448 children)"], fc="#f3f2ee")

a2 = fig.add_axes([0.43, 0.13, 0.25, 0.76], facecolor=SURF)
names = [m[0] for m in mcc]; vals = [m[1] for m in mcc]
a2.barh(range(len(mcc)), vals, color=[m[2] for m in mcc], height=0.68, edgecolor=SURF, linewidth=2)
for i, v in enumerate(vals):
    a2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.8, color=INK2)
a2.set_yticks(range(len(mcc))); a2.set_yticklabels(names, fontsize=9.3)
a2.get_yticklabels()[2].set_weight("bold")
a2.set_xlim(0, 0.75); a2.set_xlabel("Test MCC, Kermany (mean of 50 runs)", fontsize=9.5)
for s in ("top", "right", "left"):
    a2.spines[s].set_visible(False)
a2.spines["bottom"].set_color("#c9c8c2"); a2.tick_params(axis="y", length=0); a2.xaxis.grid(True, color="#e6e5e0", lw=0.8)
a2.set_axisbelow(True)
a2.set_title("Same budget, same head", loc="left", weight="bold", fontsize=13.5, pad=10)
from matplotlib.patches import Patch
a2.legend(handles=[Patch(color=GF, label="gradient-free"), Patch(color=HY, label="FPA–SDA hybrid"),
                   Patch(color=REF, label="gradient-trained")], loc="lower right", fontsize=8.5, frameon=False)

a3 = fig.add_axes([0.71, 0.04, 0.28, 0.90]); a3.axis("off"); a3.set_xlim(0, 1); a3.set_ylim(0, 1)
a3.text(0.0, 0.99, "Findings", weight="bold", fontsize=13.5, va="top")
F = [("The hybrid is not better than SDA alone", "worse training fitness in all 50 runs;\nsame ranking on RSNA and at 5× budget"),
     ("Gains come from established components", "the redesign owes its MCC to current-to-pbest;\nplant operators alone: 0.38"),
     ("Protocol and representation dominate", "re-splitting: up to +13.5 accuracy points;\nfine-tuned features: all methods 0.52–0.57")]
y = 0.86
for t, s in F:
    a3.add_patch(FancyBboxPatch((0.015, y - 0.22), 0.965, 0.22, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#f3f2ee", ec="#c9c8c2", lw=1))
    a3.text(0.03, y - 0.04, t, weight="bold", fontsize=10.2, va="top")
    a3.text(0.03, y - 0.10, s, fontsize=9.2, va="top", color=INK2, linespacing=1.35)
    y -= 0.29
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=200, facecolor=SURF)
print("written", out)
