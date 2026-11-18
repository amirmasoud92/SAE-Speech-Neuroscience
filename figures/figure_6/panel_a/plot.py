"""Figure 6a: per participant, the share of each basis's total positive peak unique
variance that falls in language regions (sparse code against 300 principal components of the dense
stream), one line per participant; Wilcoxon signed-rank test on the paired shares.

data.csv: per participant, the language-region share of each basis.
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
D = pd.read_csv(HERE / "data.csv")
sp, dn = D.sparse_share_language, D.dense_share_language
p = float(sps.wilcoxon(sp, dn).pvalue)
N_PCS = 300

fig, ax = S.panel(58, 72, box=(15, 15, 36, 47))
S.note(fig, f"dense higher in {int((dn > sp).sum())} of {len(D)};\npaired median "
            f"{S.num(np.median(sp - dn), '+.3f')}, $P$ = {p:.4f}", 2, 1.0, size=S.T_SMALL)
for a, b in zip(sp, dn):
    ax.plot([0, 1], [a, b], color=S.CTRL_L, lw=0.8, zorder=1)
ax.scatter(np.zeros(len(D)), sp, s=12, color=S.SPARSE, zorder=3, lw=0)
ax.scatter(np.ones(len(D)), dn, s=12, color=S.DENSE, zorder=3, lw=0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["sparse", f"dense\n({N_PCS} PCs)"], linespacing=1.0)
for tl, c in zip(ax.get_xticklabels(), (S.SPARSE, S.DENSE)):
    tl.set_color(c)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(0.1, 0.8)
ax.set_ylabel("share of the effect in language regions", labelpad=1.5)
S.despine(ax, bottom=False)
S.save(fig, __file__)
