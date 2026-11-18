"""Figure 4d: an independent design. Per participant, the percentage of language-network
electrodes at which 300 sparse features beat 300 principal components of the same stream; bars,
95% Jeffreys intervals; dots sized by electrode count (printed right). The sign test counts the
untied participants with at least three electrodes.

data.csv: per participant, the electrode count, the percentage favouring sparse and its P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").sort_values("n", ascending=False).reset_index(drop=True)
con = D[D.n >= 3]
n_fav, n_untied = int((con.win > 50).sum()), int((con.win != 50).sum())
p = float(binomtest(n_fav, n_untied).pvalue)

fig, ax = S.panel(58, 64, box=(10, 9, 30, 46))
S.note(fig, f"{n_fav} of {n_untied} untied participants favour sparse;\nsign test {S.p_str(p)}", 2, 1.0,
       size=S.T_SMALL)
ypos = np.arange(len(D))[::-1]
for i, r in D.iterrows():
    lo, hi = sps.beta.ppf([0.025, 0.975], r.win / 100 * r.n + 0.5, (1 - r.win / 100) * r.n + 0.5)
    S.interval(ax, ypos[i], 100 * lo, 100 * hi, S.SPARSE_L, horizontal=True, lw=0.9)
    ax.plot([r.win], [ypos[i]], "o", ms=2.4 + 0.45 * np.sqrt(r.n), color=S.SPARSE, mec="white", mew=0.4,
            zorder=4)
    ax.text(-0.05, ypos[i], r.subject.replace("sub-", ""), transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=S.T_SMALL, color=S.INK)
    ax.text(1.04, ypos[i], f"{int(r.n)}", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=S.T_SMALL, color=S.INK2)
ax.text(1.04, len(D) - 0.25, "n", transform=ax.get_yaxis_transform(), ha="left", va="center",
        fontsize=S.T_SMALL, color=S.INK2, fontstyle="italic")
ax.axvline(50, color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=1)
ax.set_yticks([])
ax.set_xlim(0, 100)
ax.set_ylim(-0.7, len(D) - 0.3)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("electrodes favouring\nsparse (%)", labelpad=1, linespacing=1.0)
S.despine(ax, left=False)
S.save(fig, __file__)
