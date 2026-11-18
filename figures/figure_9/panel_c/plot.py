"""Figure 9c: sparsity dose-response. Sparse minus dense total out-of-fold r in IFG for every
released layer-12 dictionary of one width, which differ only in sparsity: 16,384 features in both
cohorts, 65,536 in the podcast cohort (open circles). Top axis, the fraction of the stream's
variance the 16,384-feature dictionary leaves unexplained (FVU). Points, median per-electrode
differences; bars, 95% intervals (10,000 participant-clustered draws); series offset horizontally
for legibility.

data.csv: per family and nominal L0, the median difference and its interval.
slopes.csv: per family, the slope per doubling of L0, its interval and P.
fvu.csv: per nominal L0, the 16,384-feature dictionary's FVU.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
SL = pd.read_csv(HERE / "slopes.csv").set_index("cohort").slope_per_doubling
FVU = pd.read_csv(HERE / "fvu.csv").sort_values("nominal_l0")
FAM = [("Podcast, 16k", "o", "-", True), ("Podcast, 65k", "o", (0, (3, 2)), False),
       ("Brain Treebank, 16k", "D", (0, (1, 1.5)), True)]

fig, ax = S.panel(64, 60, box=(13, 12, 48, 38))
S.note(fig, "fraction of variance unexplained, 16k", 13, 1.0, size=S.T_SMALL)
ax.axhline(0, color=S.INK2, lw=0.6, zorder=1)
for i, (lab, mk, ls, filled) in enumerate(FAM):
    g = D[D.cohort == lab].sort_values("nominal_l0")
    x = np.log2(g.nominal_l0.to_numpy(float)) + 0.06 * (i - 1)
    for xi, lo, hi in zip(x, g.lo, g.hi):
        S.interval(ax, xi, lo, hi, S.SPARSE_L, lw=0.7, z=2)
    ax.plot(x, g["median"], ls=ls, color=S.SPARSE, lw=1.0, zorder=3)
    ax.plot(x, g["median"], mk, ls="none", mfc=S.SPARSE if filled else "white", mec=S.SPARSE, mew=0.8,
            ms=3.4 if mk == "o" else 3.0, zorder=4)
ax.set_xticks(np.log2([25, 50, 100, 200, 400]))
ax.set_xticklabels(["25", "50", "100", "200", "400"])
ax.set_xlabel("L0, features per token", labelpad=1)
ax.set_xlim(np.log2(17), np.log2(560))
ax.set_ylim(-0.085, 0.006)
ax.axhspan(-0.085, 0, color=S.DENSE_FIELD, lw=0, zorder=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else S.num(v, "g")))
ax.set_ylabel("sparse − dense, total r", labelpad=1.5)
S.despine(ax)
handles = [Line2D([], [], color=S.SPARSE, ls=ls, marker=mk, lw=1.0, mfc=S.SPARSE if filled else "white",
                  mec=S.SPARSE, mew=0.8, ms=3.4 if mk == "o" else 3.0) for (_, mk, ls, filled) in FAM]
labels = [f"{'Podcast' if l.startswith('Podcast') else 'Film'} {l.split(', ')[1]}   {S.num(SL[l])}"
          for (l, *_r) in FAM]
leg = ax.legend(handles, labels, loc="lower right", fontsize=S.T_SMALL, frameon=False, handlelength=2.6,
                borderaxespad=0.3, title="slope per doubling of L0", title_fontsize=S.T_SMALL, labelspacing=0.35)
leg.get_title().set_color(S.INK2)
top = ax.secondary_xaxis("top")
top.set_xticks(np.log2(FVU.nominal_l0.to_numpy(float)))
top.set_xticklabels([f"{v:.2f}" for v in FVU.fvu], fontsize=S.T_SMALL, color=S.INK2)
top.tick_params(length=2, color=S.RULE, pad=1)
top.spines["top"].set_visible(True)
top.spines["top"].set_color(S.RULE)
ax.text(0.13, 0.03, "dense better", transform=ax.transAxes, fontsize=S.T_SMALL, color=S.DENSE, ha="left",
        va="bottom")
S.save(fig, __file__)
