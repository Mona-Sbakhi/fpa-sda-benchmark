"""Study-design figure: data -> features -> optimisers under equal budget -> evaluation,
with the two phases of the sequential FPA-SDA hybrid. Writes paper/figures/workflow.{pdf,png}.

Every number in the figure is stated in the manuscript (Sections 3.1-3.11)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5})

GREY, BLUE, PLANT = "#555555", "#1f5fa8", "#3c7a3c"
FILL = {"data": "#f2f2f2", "feat": "#eef3fa", "opt": "#ffffff", "eval": "#f2f2f2"}

fig = plt.figure(figsize=(7.5, 4.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(2, 62)
ax.axis("off")


def box(x, y, w, h, fill, edge=GREY, lw=0.8, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw))


def head(x, y, w, n, title):
    ax.text(x + w / 2, y, f"{n}  {title}", ha="center", va="top", fontsize=8.5, fontweight="bold")


def lines(x, y, items, dy=2.55, size=7.2, color="black"):
    for i, t in enumerate(items):
        ax.text(x, y - i * dy, t, ha="left", va="top", fontsize=size, color=color)


def arrow(x0, y0, x1, y1, color=GREY, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


TOP, H = 60, 37          # top of the four stage columns and their height
cols = [(1, 21), (25, 22), (51, 26), (81, 18)]

# 1 data
x, w = cols[0]
box(x, TOP - H, w, H, FILL["data"])
head(x, TOP - 1.2, w, "1", "Data")
lines(x + 1.2, TOP - 5.2, ["Kermany, paediatric", "5,856 images", "(normal / pneumonia)"])
ax.text(x + 1.2, TOP - 14.2, "Official test folder", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 16.8, ["624 images, held out,", "used once per model"])
ax.text(x + 1.2, TOP - 23.0, "Patient-grouped split", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 25.6, ["optimisation  4,186", "validation      1,046"])
ax.text(x + 1.2, TOP - 31.4, "Replication: RSNA,", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 34.0, ["14,863 adult images"])

# 2 features
x, w = cols[1]
box(x, TOP - H, w, H, FILL["feat"])
head(x, TOP - 1.2, w, "2", "Representation")
ax.text(x + 1.2, TOP - 5.2, "(a) PCA of pixels", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 7.8, ["150×150 → 1,000 PCs", "net 1000–128–64–32–2", "D = 138,530"])
ax.text(x + 1.2, TOP - 16.4, "(b) Frozen ResNet-50", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 19.0, ["2,048 features", "head 2048–16–2, D = 32,818", "main benchmark"])
ax.text(x + 1.2, TOP - 27.6, "(c) Fine-tuned ResNet-50", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 30.2, ["5-fold patient-grouped CV"])

# 3 optimisers
x, w = cols[2]
box(x, TOP - H, w, H, FILL["opt"])
head(x, TOP - 1.2, w, "3", "Training the classifier")
lines(x + 1.2, TOP - 5.2, ["weights x ∈ [−1, 1]$^D$; minimise", "class-weighted cross-entropy + L2"], size=7.0)
ax.text(x + 1.2, TOP - 11.0, "Equal budget: 6,795 evaluations", fontsize=7.2, fontweight="bold", va="top")
ax.text(x + 1.2, TOP - 14.2, "Plant-inspired", fontsize=7.2, color=PLANT, fontweight="bold", va="top")
lines(x + 1.2, TOP - 16.8, ["FPA, SDA, hybrid FPA→SDA,", "redesigned hybrid (APH)"], color=PLANT, size=7.0)
ax.text(x + 1.2, TOP - 22.4, "Other gradient-free", fontsize=7.2, color=GREY, fontweight="bold", va="top")
lines(x + 1.2, TOP - 25.0, ["GA, PSO, DE, L-SHADE,", "sep-CMA-ES, random search,", "OpenAI-ES, ZO-SGD"], color=GREY, size=7.0)
ax.text(x + 1.2, TOP - 33.4, "Gradient: logistic regression, MLP", fontsize=7.0, color=BLUE, va="top")

# 4 evaluation
x, w = cols[3]
box(x, TOP - H, w, H, FILL["eval"])
head(x, TOP - 1.2, w, "4", "Evaluation")
lines(x + 1.2, TOP - 5.2, ["50 paired runs", "(20 on RSNA)"])
ax.text(x + 1.2, TOP - 11.0, "Three endpoints", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 13.6, ["training fitness", "validation MCC", "test MCC"])
ax.text(x + 1.2, TOP - 22.0, "Statistics", fontsize=7.2, fontweight="bold", va="top")
lines(x + 1.2, TOP - 24.6, ["Friedman, Nemenyi", "Wilcoxon–Holm, Â", "test bootstrap"], size=7.0)

for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
    arrow(x0 + w0 + 0.4, TOP - H / 2, x1 - 0.4, TOP - H / 2)

# bottom band: the sequential hybrid
yb, hb = 4.0, 13.0
ax.text(1, yb + hb + 1.3, "Sequential hybrid FPA→SDA (n = 45, T = 150 iterations)",
        fontsize=8.5, fontweight="bold", va="bottom")
box(1, yb, 42, hb, "#eef6ee", edge=PLANT)
ax.text(22, yb + hb - 1.4, "Phase 1: FPA,  t = 1 … τ  (τ = 75)", ha="center", va="top", fontsize=7.6,
        fontweight="bold", color=PLANT)
lines(2.4, yb + hb - 5.0, ["global pollination (p = 0.8): Lévy step towards best g*",
                           "local pollination: random step between two flowers",
                           "keep a candidate only if strictly better"], size=7.0)
box(46, yb + 1.0, 17, hb - 2.0, "#ffffff", edge=GREY)
ax.text(54.5, yb + hb - 2.4, "Transition", ha="center", va="top", fontsize=7.6, fontweight="bold")
lines(47.2, yb + hb - 5.2, ["transfer population", "or re-initialise", "around the elite"], dy=2.3, size=7.0)
box(66, yb, 33, hb, "#eef6ee", edge=PLANT)
ax.text(82.5, yb + hb - 1.4, "Phase 2: SDA,  t = τ+1 … T", ha="center", va="top", fontsize=7.6,
        fontweight="bold", color=PLANT)
lines(67.4, yb + hb - 5.0, ["wind (0.70): Lévy step towards g*",
                            "animal (0.15): step towards a random seed",
                            "gravity (0.15): small uniform perturbation"], size=7.0)
arrow(43.4, yb + hb / 2, 45.6, yb + hb / 2, color=PLANT)
arrow(63.4, yb + hb / 2, 65.6, yb + hb / 2, color=PLANT)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "workflow.pdf")
fig.savefig(OUT / "workflow.png", dpi=300)
print("wrote", OUT / "workflow.pdf")
