"""Figure 2d (right): the pre-specified statistic, the change in the median unique sparse-band
variance in IFG when the syntactic band is added, at each electrode's own peak and over 0-500 ms,
against the 20% tolerance (shaded beyond); grey, the 4-band peak-lag convention, which is biased
toward a loss and not pre-specified.

data.csv: per lag convention, the medians of the 4- and 5-band models and the percentage change.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").set_index("convention")
ROWS = [("own peak", "own peak", True), ("mean, 0-500 ms", "mean, 0–500 ms", True),
        ("4-band peak lag", "4-band peak lag,\nnot pre-specified", False)]

fig, ax = S.panel(50, 44, box=(20, 6, 27, 29))
ax.axvspan(-45, -20, color=S.DENSE_FIELD, lw=0, zorder=0)
ax.axvline(-20, color=S.DENSE, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.axvline(0, color=S.RULE, lw=S.LW_REF, zorder=0)
for k, (key, lab, pre) in enumerate(ROWS):
    v = float(D.loc[key, "pct_change"])
    ax.plot([0, v], [k, k], color=S.SPARSE if pre else S.CTRL, lw=0.9, zorder=2)
    S.dot(ax, v, k, S.SPARSE if pre else S.CTRL, filled=pre, size=4.0)
    ax.text(v / 2 if pre else v, k - 0.22, f"{S.num(v, '.1f')}%", fontsize=S.T_SMALL, ha="center", va="bottom",
            color=S.INK if pre else S.INK2, fontweight="bold" if pre else "normal")
    ax.text(-0.04, k, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=S.T_SMALL,
            color=S.INK if pre else S.INK2, linespacing=1.0)
ax.text(-32.5, -0.70, "beyond\ntolerance", fontsize=S.T_SMALL, color=S.DENSE, ha="center", va="top",
        linespacing=1.0)
ax.set_xlim(-45, 3)
ax.set_ylim(len(ROWS) - 0.4, -1.05)
ax.set_yticks([])
ax.set_xticks([-40, -20, 0])
ax.set_xticklabels(["−40", "−20", "0"])
ax.set_xlabel("change in median (%)", labelpad=1)
S.despine(ax, left=False)
S.note(fig, "pre-specified statistic", 20, 1.0, size=S.T_SMALL)
S.save(fig, __file__)
