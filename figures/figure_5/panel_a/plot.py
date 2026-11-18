"""Figure 5a: per participant, the percentage of admissible language-network
electrodes whose held-out R2 improves when the sparse band is pruned from all 4,933 live features to
the 300 with the highest stimulus variance (a criterion that uses no neural data); bars, 95%
Jeffreys intervals; dots sized by electrode count (printed right); dashed, parity. The sign test
counts untied participants with at least three electrodes.

data.csv: per participant, the electrode count, the percentage improving and its P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").sort_values("n", ascending=False).reset_index(drop=True)
con = D[D.n >= 3]
n_fav, n_unt = int((con.win > 50).sum()), int((con.win != 50).sum())
p_sign = float(sps.binomtest(n_fav, n_unt).pvalue)

fig, ax = S.panel(92, 72, box=(18, 9, 66, 52))
S.note(fig, f"{n_fav} of {len(con)} participants favour the 300 pruned features; median {con.win.median():.0f}%, "
            f"sign test $P$ = {p_sign:.3f}", 2, 1.0, size=S.T_SMALL)
ypos = np.arange(len(D))[::-1]
tr = ax.get_yaxis_transform()
for i, r in D.iterrows():
    lo, hi = sps.beta.ppf([0.025, 0.975], r.win / 100 * r.n + 0.5, (1 - r.win / 100) * r.n + 0.5)
    S.interval(ax, ypos[i], 100 * lo, 100 * hi, S.SPARSE_L, horizontal=True, lw=1.0)
    ax.plot([r.win], [ypos[i]], "o", ms=2.4 + 0.45 * np.sqrt(r.n), color=S.SPARSE, mec="white", mew=0.4, zorder=4)
    ax.text(-0.03, ypos[i], r.subject.replace("sub-", ""), transform=tr, ha="right", va="center",
            fontsize=S.T_SMALL, color=S.INK)
    ax.text(1.02, ypos[i], f"{int(r.n)}", transform=tr, ha="left", va="center", fontsize=S.T_SMALL, color=S.INK2)
ax.text(1.02, len(D) - 0.3, "n", transform=tr, ha="left", va="center", fontsize=S.T_SMALL, color=S.INK2,
        fontstyle="italic")
ax.text(-0.03, len(D) - 0.3, "participant", transform=tr, ha="right", va="center", fontsize=S.T_SMALL,
        color=S.INK2, fontstyle="italic")
ax.axvline(50, color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=1)
ax.set_yticks([])
ax.set_xlim(0, 100)
ax.set_ylim(-0.7, len(D) - 0.1)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("electrodes where 300 pruned features beat all 4,933 (%)", labelpad=1)
S.despine(ax, left=False)
S.save(fig, __file__)
