"""Figure 3b: principal-component coherence by component rank (median and interquartile range
per rank bin), against the sparse-feature median and chance (the median of the random 20-word
sets).

data.csv: per rank bin, the median and quartiles of principal-component coherence.
The two reference lines are read from ../panel_a/data.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
A = pd.read_csv(HERE.parent / "panel_a" / "data.csv")
rand = A[A.group == "random"].value.to_numpy(float)
sae_med = float(np.nanmedian(A[A.group == "SAE-300"].value))
q1r, floor, q3r = np.percentile(rand, [25, 50, 75])

fig, ax = S.panel(52, 46, box=(11, 3, 39, 31))
xs = np.arange(len(D))
ax.axhspan(q1r, q3r, color=S.FIELD, lw=0, zorder=0)
ax.fill_between(xs, D.q25, D.q75, color=S.DENSE, alpha=0.16, lw=0)
ax.plot(xs, D["median"], "-", color=S.DENSE, lw=S.LW_DATA)
ax.plot(xs, D["median"], "o", color=S.DENSE, ms=3.0, mec="white", mew=0.4)
ax.axhline(sae_med, lw=S.LW_DATA, color=S.SPARSE)
ax.axhline(floor, lw=S.LW_REF, ls=(0, (3, 2)), color=S.INK2)
ax.text(5.3, sae_med + 0.0025, "sparse median", fontsize=S.T_SMALL, color=S.SPARSE, ha="right",
        va="bottom", fontweight="bold")
ax.text(5.3, floor - 0.0025, "chance", fontsize=S.T_SMALL, color=S.INK2, ha="right", va="top")
ax.text(2.25, 0.490, "principal\ncomponents", fontsize=S.T_SMALL, color=S.DENSE, ha="left",
        va="bottom", fontweight="bold", linespacing=1.0)
ax.set_xticks(xs)
ax.set_xticklabels([b.replace("-", "–") for b in D.rank_bin], fontsize=S.T_SMALL, rotation=35,
                   ha="right", rotation_mode="anchor")
ax.set_xlim(-0.35, 5.35)
ax.set_ylim(0.425, 0.540)
ax.set_xlabel("component rank", labelpad=0.5)
ax.set_ylabel("coherence", labelpad=1.5)
S.despine(ax)
S.save(fig, __file__)
