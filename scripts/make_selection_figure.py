#!/usr/bin/env python3
"""Draw assets/figures/selection_candidates.png from the saved results.

    python scripts/make_selection_figure.py

Shows the composite selection score (|probe - 0.5| + max(0, delta NLL), lower is
better) for every layer and estimator candidate. Difference-in-means and the
LEACE-derived eraser tie at every layer in the saved run, so they share a bar.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "paper" / "results_gpt2_mickey.json").read_text())
INK, MUTED, GRID = "#1f2d3a", "#5b6b7a", "#e3e8ee"
BLUE, ORANGE = "#334e68", "#d06a35"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

rows = RES["selection"]
layers = sorted({r["layer"] for r in rows})
score = {(r["layer"], r["method"]): r["score"] for r in rows}
chosen = (RES["chosen"]["layer"], RES["chosen"]["method"])
tie_gap = max(abs(score[(l, "leace")] - score[(l, "mean_diff")]) for l in layers)
assert tie_gap < 5e-4, "mean_diff and leace no longer tie; split the bars"

fig, ax = plt.subplots(figsize=(8.2, 3.9))
w = 0.36
xs = range(len(layers))
for i, l in enumerate(layers):
    a = ax.bar(i - w / 2, score[(l, "leace")], w, color=ORANGE if (l, "leace") == chosen else BLUE)
    b = ax.bar(i + w / 2, score[(l, "inlp")], w, color="#a9b8c8")
    if (l, "leace") == chosen:
        ax.annotate(f"selected\n{score[(l, 'leace')]:.3f}", (i - w / 2, score[(l, "leace")]),
                    xytext=(i - w / 2 - 0.1, score[(l, "leace")] + 0.3), ha="center", fontsize=8.6,
                    color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1),
                    bbox=dict(fc="white", ec="none", pad=1.5))
ax.set_xticks(list(xs), [str(l) for l in layers])
ax.set_xlabel("Candidate layer")
ax.set_ylabel("Composite score (lower is better)")
ax.set_ylim(0, 1.45)
ax.set_title("Layer 5 with the LEACE-derived eraser has the lowest composite score",
             fontsize=11, fontweight="bold", color=INK, pad=10)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=BLUE, label="Difference-in-means and LEACE-derived (tied)"),
                   Patch(color="#a9b8c8", label="INLP")], frameon=False, loc="upper right", fontsize=8.8)
fig.tight_layout()
out = ROOT / "assets" / "figures" / "selection_candidates.png"
fig.savefig(out, dpi=200)
print(f"Wrote {out}")
