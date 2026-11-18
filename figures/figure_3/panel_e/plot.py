"""Figure 3e: feature 12456 on every token of "no" in the stimulus. It is active on discourse
negation and silent inside "there's no X". Left, activations per use with medians (Mann-Whitney,
descriptive); right, the three highest-activation contexts of each use, target word in bold, each
with its activation as a bar.

data.csv: one row per token of "no": its word index, use, activation and transcript context;
``context_list_order`` marks the six contexts listed on the right.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv", keep_default_na=False)
GROUPS = [("discourse negation", "discourse negation", S.SPARSE),
          ("inside there's no X", "inside “there’s no X”", S.CTRL)]
act = {k: D[D.use == k].sort_values("activation", ascending=False) for k, *_ in GROUPS}
u = mannwhitneyu(act[GROUPS[0][0]].activation, act[GROUPS[1][0]].activation, alternative="greater")
amax = float(D.activation.max())

fig = S.figure(183, 46)
S.note(fig, f"all {len(D)} tokens of “no”; Mann–Whitney {S.p_str(u.pvalue, 1)}, descriptive", 2, 1.5,
       size=S.T_SMALL)
ax = S.ax_mm(fig, 32, 10, 50, 24)
rng = np.random.default_rng(1)
for k0, (key, lab, c) in enumerate(GROUPS):
    k = len(GROUPS) - 1 - k0
    v = act[key].activation.to_numpy()
    ax.scatter(v, np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), s=14, color=c, alpha=0.9,
               lw=0.5, edgecolors="white", zorder=3)
    ax.plot([np.median(v)] * 2, [k - 0.30, k + 0.30], color=S.INK, lw=1.2, zorder=4)
    ax.text(-0.04, k + 0.10, lab, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=S.T_SMALL, color=c if c != S.CTRL else S.INK2, fontweight="bold")
    ax.text(-0.04, k - 0.10, f"n = {len(v)}, median {np.median(v):.1f}",
            transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=S.T_SMALL, color=S.INK2)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(-1.5, amax * 1.08)
ax.set_xlabel("activation", labelpad=1)
S.despine(ax, left=False)

# the transcript, typeset, each line with its activation as a bar
ax_t = S.ax_mm(fig, 90, 5, 91, 37)
ax_t.set_xlim(0, 91)
ax_t.set_ylim(37, 0)
ax_t.axis("off")
ax_t.text(0, 1.2, "transcript context", fontsize=S.T_SMALL, color=S.INK2, va="center", fontstyle="italic")
ax_t.text(91, 1.2, "activation", fontsize=S.T_SMALL, color=S.INK2, va="center", ha="right",
          fontstyle="italic")
ax_t.plot([0, 91], [2.9, 2.9], color=S.RULE, lw=0.45)
r_ = fig.canvas.get_renderer()
yy = 5.6
for key, lab, c in GROUPS:
    ax_t.text(0, yy, lab, fontsize=S.T_SMALL, color=c if c != S.CTRL else S.INK2, fontweight="bold",
              va="center")
    yy += 4.0
    for _, r in D[(D.use == key) & D.context_list_order.ne("")].sort_values("context_list_order").iterrows():
        t1 = ax_t.text(2.0, yy, f"… {r.context_before} ", fontsize=S.T_ANNOT, color=S.INK, va="center")
        bb = t1.get_window_extent(r_).transformed(ax_t.transData.inverted())
        t2 = ax_t.text(bb.x1, yy, r.word, fontsize=S.T_ANNOT, va="center", fontweight="bold",
                       color=c if c != S.CTRL else S.INK)
        bb2 = t2.get_window_extent(r_).transformed(ax_t.transData.inverted())
        ax_t.text(bb2.x1, yy, f" {r.context_after} …", fontsize=S.T_ANNOT, color=S.INK, va="center")
        ax_t.add_patch(plt.Rectangle((72.0, yy - 0.9), 12.0 * r.activation / amax, 1.8,
                                     fc=c if c != S.CTRL else S.CTRL_L, ec="none", zorder=1))
        ax_t.text(91, yy, f"{r.activation:.1f}", fontsize=S.T_ANNOT, va="center", ha="right",
                  color=c if c != S.CTRL else S.INK2, fontweight="bold")
        yy += 4.0
    yy += 2.2
S.save(fig, __file__)
