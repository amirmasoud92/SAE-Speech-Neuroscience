"""Figure 6b: concentration over electrodes. Lorenz curves of each basis's positive
peak unique variance over all 1,264 electrodes (electrodes ranked by effect; cumulative share of
the total effect); dashed, uniform.

data.csv: per electrode, the peak unique variance of the sparse and dense bases.
top_decile.csv: the share of each basis's total effect carried by the top decile of electrodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def lorenz(v):
    """Cumulative share of the total effect against cumulative share of electrodes."""
    s = np.sort(np.clip(v, 0, None))[::-1]
    return np.arange(1, len(s) + 1) / len(s), np.cumsum(s) / max(s.sum(), 1e-12)


S.apply()
D = pd.read_csv(HERE / "data.csv")
tds = {}
fig, ax = S.panel(58, 72, box=(10, 15, 44, 47))
for col, name, c in (("dense_peak_unique_dr2", "dense", S.DENSE), ("sparse_peak_unique_dr2", "sparse", S.SPARSE)):
    x, y = lorenz(D[col].to_numpy())
    ax.plot(x, y, lw=S.LW_DATA, color=c)
    tds[name] = float(np.interp(0.10, x, y))
S.note(fig, f"top decile of electrodes carries\n{100 * tds['sparse']:.0f}% (sparse) and "
            f"{100 * tds['dense']:.0f}% (dense)", 2, 1.0, size=S.T_SMALL)
ax.plot([0, 1], [0, 1], color=S.RULE, lw=S.LW_REF, ls=(0, (3, 2)), zorder=0)
ax.text(0.30, 0.47, "sparse", color=S.SPARSE, fontsize=S.T_SMALL, fontweight="bold")
ax.text(0.06, 0.74, "dense", color=S.DENSE, fontsize=S.T_SMALL, fontweight="bold")
ax.text(0.62, 0.50, "uniform", rotation=38, color=S.INK2, fontsize=S.T_SMALL)
ax.set_xlabel("share of electrodes (ranked)", labelpad=1.5)
ax.set_ylabel("share of the total effect", labelpad=2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
S.despine(ax)
S.save(fig, __file__)
