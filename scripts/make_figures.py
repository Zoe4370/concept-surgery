#!/usr/bin/env python3
"""Regenerate every figure in assets/figures/ from the saved results.

    python scripts/make_figures.py

The two data-driven figures (headline_results.png, probe_by_layer.png) read
paper/results_gpt2_mickey.json, so they always match the saved run. The
diagrams (pipeline, surgery_hook, module_dataflow) are drawn directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = json.loads((ROOT / "paper" / "results_gpt2_mickey.json").read_text())

INK, MUTED, GRID = "#1f2d3a", "#5b6b7a", "#e3e8ee"
BLUE, ORANGE, GREEN = "#334e68", "#d06a35", "#4f8b67"
BOX_FILL, BOX_EDGE, HOT_FILL, HOT_EDGE = "#eef2f7", "#33475b", "#fdeee4", "#d06a35"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, cx, cy, w, h, title, sub, hot=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                fc=HOT_FILL if hot else BOX_FILL,
                                ec=HOT_EDGE if hot else BOX_EDGE, lw=1.4))
    ax.text(cx, cy + h * 0.27, title, ha="center", va="center", fontsize=9.4,
            fontweight="bold", color=INK)
    ax.text(cx, cy - h * 0.14, sub, ha="center", va="center", fontsize=8.2,
            color=MUTED, linespacing=1.3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                 color=BOX_EDGE, lw=1.4, shrinkA=0, shrinkB=0))


def row_diagram(name, title, items, hot=None, note=None, width=10.0, height=2.7):
    n = len(items)
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.set_xlim(0, n); ax.set_ylim(0.0 if note else 0.14, 1); ax.axis("off")
    ax.text(n / 2, 0.95, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    bw, bh = 0.84, 0.56
    for i, (t, s) in enumerate(items):
        box(ax, i + 0.5, 0.56, bw, bh, t, s, hot=(i == hot))
        if i < n - 1:
            arrow(ax, (i + 0.5 + bw / 2 + 0.005, 0.56), (i + 1.5 - bw / 2 - 0.005, 0.56))
    if note:
        ax.text(n / 2, 0.12, note, ha="center", va="center", fontsize=9,
                style="italic", color=MUTED)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


# ---- diagrams -------------------------------------------------------------
row_diagram("pipeline.png", "Pipeline: map, fit, select, ablate, prove", [
    ("1  Map", "capture last-token\nresidual states at\ncandidate layers"),
    ("2  Fit", "difference-in-\nmeans, LEACE,\nINLP directions"),
    ("3  Select", "rank layer and\nestimator by probe\ncollapse and NLL"),
    ("4  Ablate", "install reversible\nprojection hooks,\nweights untouched"),
    ("5  Prove", "probe every layer,\nNLL, locality KL,\ngenerations"),
], width=10.0, height=2.5)

row_diagram("surgery_hook.png", "Inference-time surgery: one hook, no weight changes", [
    ("Input", "tokens +\nembeddings"),
    ("Layers 1 to l", "unchanged\nblocks"),
    ("Hidden state h", "residual stream\nat layer l"),
    ("Projection hook", "h' = h - (h . v) v\n(orthogonal)"),
    ("Layers l+1 to L", "unchanged\nblocks"),
    ("Output", "next-token\nlogits"),
], hot=3, note="v is a unit concept direction (or k stacked directions). Removing the hook restores the original model exactly.",
    width=11.5, height=3.0)

# module data flow: top row left to right, bottom row right to left
mods = [("data.py", "concept and neutral\nprompt sets"),
        ("capture.py", "forward hooks record\nactivations"),
        ("core_math.py", "difference-in-means,\nPCA, INLP, LEACE"),
        ("pipeline.py", "layer x estimator\nselection"),
        ("ablate.py", "reversible projection\nhooks"),
        ("evaluate.py", "probe, NLL, KL,\ngenerations"),
        ("results.json", "metrics and\nconfiguration")]
pos = [(0.5, 0.72), (1.5, 0.72), (2.5, 0.72), (3.5, 0.72), (3.5, 0.26), (2.5, 0.26), (1.5, 0.26)]
fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=200)
ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
ax.text(2, 0.985, "Module data flow", ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
bw, bh = 0.86, 0.34
for (t, s), (cx, cy) in zip(mods, pos):
    box(ax, cx, cy, bw, bh, t, s, hot=(t == "ablate.py"))
for i in range(len(pos) - 1):
    (x0, y0), (x1, y1) = pos[i], pos[i + 1]
    if y0 == y1:
        sgn = 1 if x1 > x0 else -1
        arrow(ax, (x0 + sgn * (bw / 2 + 0.01), y0), (x1 - sgn * (bw / 2 + 0.01), y1))
    else:
        arrow(ax, (x0, y0 - bh / 2 - 0.01), (x1, y1 + bh / 2 + 0.01))
fig.savefig(OUT / "module_dataflow.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close(fig)

# ---- data-driven figures --------------------------------------------------
layers = [4, 5, 6, 8, 9, 10]
series = {k: [RES[k][f"probe_acc_layer{l}"] for l in layers]
          for k in ("baseline", "ablated_single", "ablated_multi")}

fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)
ax.plot(range(6), series["baseline"], color=BLUE, marker="o", ms=5.5, lw=2.0, label="Baseline (no edit)")
ax.plot(range(6), series["ablated_single"], color=ORANGE, marker="s", ms=5.5, lw=2.0, label="Single-layer edit at layer 5")
ax.plot(range(6), series["ablated_multi"], color=GREEN, marker="^", ms=6, lw=2.0, ls="--", label="Multi-layer edit")
ax.axhline(0.5, color="#9aa5b1", lw=1.0, ls=":")
ax.text(5.0, 0.515, "chance (0.5)", ha="right", va="bottom", fontsize=8.5, color=MUTED)
ax.annotate("0.058\n(edited layer)", xy=(1, series["ablated_single"][1]), xytext=(1.35, 0.13),
            fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.annotate("0.942 at layer 10", xy=(5, series["ablated_single"][5]), xytext=(3.55, 0.70),
            fontsize=9, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.set_xticks(range(6)); ax.set_xticklabels([str(l) for l in layers])
ax.set_ylim(-0.02, 1.08); ax.set_xlim(-0.2, 5.2)
ax.set_xlabel("Layer where the probe is measured"); ax.set_ylabel("Cross-validated probe accuracy")
ax.set_title("The layer-5 collapse does not persist", fontsize=12.5, fontweight="bold", color=INK, pad=10)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.savefig(OUT / "probe_by_layer.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close(fig)

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=200, gridspec_kw={"width_ratios": [1, 1.25]})
vals = [RES["baseline"]["probe_acc_layer5"], RES["ablated_single"]["probe_acc_layer5"]]
a.bar(["Before", "After edit\n(layer 5)"], vals, color=[BLUE, ORANGE], width=0.55)
a.axhline(0.5, color="#9aa5b1", lw=1.0, ls=":"); a.text(1.42, 0.515, "chance", ha="right", fontsize=8.5, color=MUTED)
for i, v in enumerate(vals):
    a.text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold", color=INK)
a.set_ylim(0, 1.12); a.set_ylabel("Probe accuracy at layer 5")
a.set_title("Local probe collapse", fontsize=11, fontweight="bold", color=INK)
nll = [RES["baseline"]["neutral_nll"], RES["ablated_single"]["neutral_nll"], RES["ablated_multi"]["neutral_nll"]]
bars = b.bar(["Baseline", "Single-layer\nedit", "Multi-layer\nedit"], nll, color=[BLUE, ORANGE, GREEN], width=0.55)
for i, v in enumerate(nll):
    lab = f"{v:.3f}" if i == 0 else f"{v:.3f}\n(+{v - nll[0]:.3f})"
    b.text(i, v + 0.06, lab, ha="center", fontweight="bold", color=INK, fontsize=9.5)
b.set_ylim(0, max(nll) * 1.28); b.set_ylabel("Neutral NLL (nats per token)")
b.set_title("Collateral cost on neutral text", fontsize=11, fontweight="bold", color=INK)
for ax_ in (a, b):
    ax_.grid(axis="y", color=GRID, lw=0.8); ax_.set_axisbelow(True)
    for s in ("top", "right"): ax_.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.5)
fig.savefig(OUT / "headline_results.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.close(fig)
print("figures written to", OUT)
