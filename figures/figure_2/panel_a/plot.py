"""Figure 2a: out-of-fold correlation at the 135 IFG electrodes of 85 principal components of the
sparse feature space against the 85-column classical linguistic band (lags 0-500 ms, acoustic,
surprisal and lexical bands removed within fold). One point per electrode (blue above identity,
grey below); diamonds, participant means sized by electrode count; margins, distributions with
their means.

data.csv: per electrode, the participant and the two out-of-fold correlations.
contrast.csv: the difference of the means, its participant-clustered interval and P, the ratio.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
E = pd.read_csv(HERE / "data.csv")
C = pd.read_csv(HERE / "contrast.csv").iloc[0]
x, y = E["classical-85"].to_numpy(), E["SAE PCA-85"].to_numpy()
lim = (-0.12, 0.44)

fig = S.figure(78, 76)
ax0, ay0, aw = 13.0, 9.0, 54.0
ax = S.ax_mm(fig, ax0, ay0, aw, aw)
axt = S.ax_mm(fig, ax0, ay0 - 7.0, aw, 6.2, sharex=ax)
axr = S.ax_mm(fig, ax0 + aw + 0.8, ay0, 6.2, aw, sharey=ax)
ax.fill_between(lim, lim, [lim[1], lim[1]], color=S.SPARSE_XL, alpha=0.45, lw=0, zorder=0)
ax.plot(lim, lim, color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=1)
above = y > x
ax.scatter(x[~above], y[~above], s=8, color=S.CLASSICAL, alpha=0.85, lw=0.25, edgecolors="white", zorder=3)
ax.scatter(x[above], y[above], s=8, color=S.SPARSE, alpha=0.80, lw=0.25, edgecolors="white", zorder=3)
sm = E.groupby("subject").agg(n=("subject", "size"), c=("classical-85", "mean"), s=("SAE PCA-85", "mean"))
ax.scatter(sm.c, sm.s, s=7 + 1.0 * sm.n, facecolor="white", edgecolor=S.INK, lw=0.65, zorder=4, marker="D")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ticks = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.xaxis.set_major_formatter(S.minus_formatter("{:g}"))
ax.yaxis.set_major_formatter(S.minus_formatter("{:g}"))
ax.set_xlabel("classical linguistics, out-of-fold r", labelpad=1)
ax.set_ylabel("sparse code, out-of-fold r", labelpad=1)
S.despine(ax)
ax.text(0.37, 0.355, "identity", rotation=45, fontsize=S.T_SMALL, color=S.INK2, ha="center", va="top",
        rotation_mode="anchor")
for a_, v, c in ((axt, x, S.CLASSICAL), (axr, y, S.SPARSE)):
    g = np.linspace(*lim, 300)
    d = gaussian_kde(v, bw_method=0.25)(g)
    if a_ is axt:
        a_.fill_between(g, 0, d, color=c, alpha=0.22, lw=0)
        a_.plot(g, d, color=c, lw=0.6)
        a_.axvline(v.mean(), color=c, lw=0.9, ymax=0.92)
        a_.set_ylim(0, d.max() * 1.05)
    else:
        a_.fill_betweenx(g, 0, d, color=c, alpha=0.22, lw=0)
        a_.plot(d, g, color=c, lw=0.6)
        a_.axhline(v.mean(), color=c, lw=0.9, xmax=0.92)
        a_.set_xlim(0, d.max() * 1.05)
    a_.axis("off")
ax.text(-0.105, 0.425, f"{C.ratio:.1f}×", fontsize=15, fontweight="bold", color=S.SPARSE, ha="left", va="top")
ax.text(-0.105, 0.352, f"mean r {x.mean():.3f} → {y.mean():.3f}", fontsize=S.T_ANNOT, ha="left", va="top")
ax.text(-0.105, 0.322, f"Δr {S.num(C['diff'])} [{S.num(C.lo)}, {S.num(C.hi)}]", fontsize=S.T_SMALL,
        color=S.INK2, ha="left", va="top")
ax.text(-0.105, 0.294, S.p_str(C.p), fontsize=S.T_SMALL, color=S.INK2, ha="left", va="top")
ax.text(0.425, -0.035, f"{int(above.sum())} of {len(E)} electrodes\nabove identity", fontsize=S.T_SMALL,
        color=S.SPARSE, ha="right", va="center", linespacing=1.05)
ax.scatter([0.236], [-0.093], s=16, facecolor="white", edgecolor=S.INK, lw=0.65, marker="D", zorder=5)
ax.text(0.252, -0.093, "participant mean", fontsize=S.T_SMALL, color=S.INK2, va="center")
S.save(fig, __file__)
