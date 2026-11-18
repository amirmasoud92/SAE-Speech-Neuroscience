"""Figure 6c: leave-one-participant-out inter-subject correlation of the
word-locked (cleaned) targets against lag, averaged over electrodes in STG, MTG and IFG.

data.csv: per region and lag, the mean leave-one-out inter-subject correlation.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")

fig, ax = S.panel(56, 72, box=(13, 15, 36, 47))
S.note(fig, "leave-one-out, by lag", 2, 1.0, size=S.T_SMALL)
for r in ["STG", "MTG", "IFG"]:
    g = D[D.roi == r].sort_values("lag_s")
    if not len(g):
        continue
    ax.plot(1000 * g.lag_s, g.isc_loo, lw=S.LW_DATA, color=S.REGION[r])
    ax.text(2050, float(g.isc_loo.iloc[-1]), r, color=S.REGION[r], fontsize=S.T_SMALL, fontweight="bold",
            va="center")
ax.axvline(0, lw=S.LW_REF, ls=(0, (2, 2)), color=S.INK2)
ax.text(80, 0.395, "word onset", fontsize=S.T_SMALL, color=S.INK2, va="top")
ax.set_xlim(-2000, 2000)
ax.set_xticks([-2000, 0, 2000])
ax.set_ylim(0, 0.40)
ax.set_xlabel("lag from word onset (ms)", labelpad=1.5)
ax.set_ylabel("inter-subject correlation", labelpad=1.5)
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
S.despine(ax)
S.save(fig, __file__)
