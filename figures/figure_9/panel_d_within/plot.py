"""Figure 9d (left, within type; right, between types: the same script in ../panel_d_between):
the within-type and between-type components of the sparse-minus-dense difference across the
16,384-feature dictionaries, in both cohorts (circles, podcast; diamonds, film). Points, median
per-electrode differences; bars, 95% intervals (10,000 participant-clustered draws); series offset
horizontally for legibility.

data.csv: per cohort and nominal L0, the median difference and its interval.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
COMP = HERE.name.split("_")[-1]
D = pd.read_csv(HERE / "data.csv")
FAM = [(0, "Podcast, 16k", "o", "-"), (2, "Brain Treebank, 16k", "D", (0, (1, 1.5)))]

fig, ax = S.panel(46, 54, box=(13, 6, 30, 38))
ax.text(0.5, 1.04, "within type" if COMP == "within" else "between types", transform=ax.transAxes, ha="center",
        va="bottom", fontsize=S.T_SMALL, color=S.INK, fontweight="bold")
ax.axhline(0, color=S.INK2, lw=0.6, zorder=1)
for i, lab, mk, ls in FAM:
    g = D[D.cohort == lab].sort_values("nominal_l0")
    x = np.log2(g.nominal_l0.to_numpy(float)) + 0.06 * (i - 1)
    for xi, lo, hi in zip(x, g.lo, g.hi):
        S.interval(ax, xi, lo, hi, S.SPARSE_L, lw=0.7, z=2)
    ax.plot(x, g["median"], ls=ls, color=S.SPARSE, lw=1.0, zorder=3)
    ax.plot(x, g["median"], mk, ls="none", mfc=S.SPARSE, mec=S.SPARSE, mew=0.8, ms=3.4 if mk == "o" else 3.0,
            zorder=4)
ax.set_xticks(np.log2([25, 50, 100, 200, 400]))
ax.set_xticklabels(["25", "50", "100", "200", "400"])
ax.set_xlabel("L0, features per token", labelpad=1)
ax.set_xlim(np.log2(17), np.log2(560))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else S.num(v, "g")))
if COMP == "within":
    ax.set_ylabel("sparse − dense", labelpad=1.5)
lo_, hi_ = ax.get_ylim()
ax.axhspan(lo_, 0, color=S.DENSE_FIELD, lw=0, zorder=0)
ax.set_ylim(lo_, hi_)
S.despine(ax)
ax.text(0.97, 0.03, "dense better", transform=ax.transAxes, fontsize=S.T_SMALL, color=S.DENSE, ha="right",
        va="bottom")
S.save(fig, __file__)
