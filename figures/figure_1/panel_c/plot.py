"""Figure 1c: electrodes per participant by region; the bracket marks the two participants that
supply the largest share of the electrodes.

data.csv: number of electrodes per participant and region.
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
D = pd.read_csv(HERE / "data.csv")
ORDER = ["IFG", "STG", "MTG", "TMP", "EAC", "MFG", "PRC", "other"]
piv = D.pivot_table(index="subject", columns="roi", values="n", aggfunc="sum").reindex(columns=ORDER,
                                                                                      fill_value=0)
piv = piv.loc[piv.sum(1).sort_values(ascending=False).index]
total = int(piv.to_numpy().sum())
top2 = piv.sum(1).nlargest(2)
subs = piv.index.tolist()

fig, ax = S.panel(46, 54, box=(7, 3, 27, 33))
left = np.zeros(len(subs))
ypos = np.arange(len(subs))
for r in ORDER:
    v = piv[r].to_numpy()
    ax.barh(ypos, v, 0.70, left=left, color=S.REGION[r], lw=0, ec="#B9B4AD" if r == "other" else "none")
    left += v
ax.set_ylim(len(subs) - 0.5, -0.6)
ax.set_yticks(ypos)
ax.set_yticklabels([s.replace("sub-", "") for s in subs], fontsize=S.T_SMALL)
ax.tick_params(axis="y", length=0, pad=1.2)
ax.set_xlim(0, 250)
ax.set_xticks([0, 100, 200])
ax.set_xlabel("electrodes", labelpad=1)
S.despine(ax, left=False)
ax.plot([left[0] + 6, left[0] + 9, left[0] + 9, left[1] + 6], [-0.34, -0.34, 1.34, 1.34], color=S.INK,
        lw=0.5)
ax.text(left[0] + 11, 0.5, f"{int(top2.sum())} of\n{total:,}\n({100 * top2.sum() / total:.0f}%)",
        fontsize=S.T_SMALL, va="center", linespacing=1.05)
for i, r in enumerate(ORDER):                                  # colour key
    col, row = i % 4, i // 4
    x0, y0 = 6.0 + col * 9.6, 44.0 + row * 3.4
    fig.add_artist(plt.Rectangle(S.fxy(fig, x0, y0 + 0.8), 1.7 / 46, 1.7 / 54, transform=fig.transFigure,
                                 fc=S.REGION[r], ec="#9A958E" if r == "other" else "none", lw=0.25))
    S.note(fig, r, x0 + 2.3, y0 - 0.15, color=S.INK, size=S.T_SMALL - 0.3)
S.save(fig, __file__)
