"""Figure 4a: capacity. IFG mean out-of-fold r (lags 0-500 ms, acoustic, surprisal and lexical
bands removed within fold) as the sparse space is reduced by PCA to k dimensions, against the dense
stream at 300 principal components (line) and at its full 2,304 dimensions (open marker).

data.csv: per arm, the number of dimensions and the mean r over the 135 IFG electrodes.
statistics.csv: k at 95% of the maximum, and the full 4,933 minus 300 difference with its interval
and participant-clustered P.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
T = pd.read_csv(HERE / "statistics.csv").iloc[0]
sweep = D[D.arm.str.match(r"^SAE PCA-\d+$")].sort_values("k")
gem = float(D.loc[D.arm == "Gemma resid PCA-300", "mean_r"].iloc[0])
full = D[D.arm == "Gemma dense raw-2304"].iloc[0]

fig, ax = S.panel(56, 48, box=(13, 3, 40, 35))
for yy in (0.06, 0.08, 0.10, 0.12, 0.14):
    ax.axhline(yy, color=S.CTRL_XL, lw=0.4, zorder=0)
ax.axhline(gem, color=S.DENSE, lw=S.LW_DATA, zorder=2)
ax.text(18.5, gem + 0.0016, "dense, 300", fontsize=S.T_SMALL, color=S.DENSE, ha="left", va="bottom",
        fontweight="bold")
ax.plot(sweep.k, sweep.mean_r, "-", color=S.SPARSE, lw=S.LW_DATA, zorder=3)
ax.plot(sweep.k, sweep.mean_r, "o", color=S.SPARSE, ms=3.2, mec="white", mew=0.4, zorder=4)
kst = int(T.k_star)
rk = float(sweep.loc[sweep.k == kst, "mean_r"].iloc[0])
ax.annotate(f"95% of maximum\nat k = {kst}", xy=(kst, rk - 0.002), xytext=(kst, 0.090),
            fontsize=S.T_SMALL, color=S.INK2, ha="center", va="top", linespacing=1.0,
            arrowprops=dict(arrowstyle="-", lw=0.45, color=S.INK2))
ax.plot([full.k], [full.mean_r], "o", ms=3.4, mfc="white", mec=S.DENSE, mew=0.9, zorder=5)
ax.text(full.k, full.mean_r + 0.0045, "dense, all 2,304", fontsize=S.T_SMALL, color=S.DENSE, ha="center",
        va="bottom")
ax.text(sweep.k.iloc[0] * 1.05, float(sweep.mean_r.iloc[0]) - 0.005, "sparse", fontsize=S.T_SMALL,
        color=S.SPARSE, ha="left", va="top", fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(18, 7000)
ax.set_xticks([25, 100, 300, 1000, 4933])
ax.set_xticklabels(["25", "100", "300", "1,000", "4,933"])
ax.minorticks_off()
ax.set_xlabel("dimensions retained, k", labelpad=1)
ax.set_ylim(0.04, 0.16)
ax.set_ylabel("IFG mean r, out of fold", labelpad=1.5)
ax.text(6500, 0.0505, f"4,933 vs 300: {S.num(T.primary_diff)}", ha="right", va="bottom",
        fontsize=S.T_SMALL, color=S.INK)
S.rich_text(ax, 6500, 0.044, S.p_str(T.primary_p), ax.transData, ha="right", va="bottom",
            fontsize=S.T_SMALL, color=S.INK)
S.despine(ax)
S.save(fig, __file__)
