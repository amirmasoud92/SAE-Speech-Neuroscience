"""Figure 7c: the within-type signal by lag. Out-of-fold within-type correlation of three
representations (300 principal components each) against lag; solid, the per-fold form; dashed,
the pooled form (sparse, GPT-2 XL), which is biased against word identity. Dots, each profile's
peak.

data.csv: per lag, the within-type r of each representation (``*_within_fold``, per-fold form;
others, pooled form).
statistics.csv: the cross-participant within-type and between-type correlations and their
permutation and circular-shift P values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
PR = pd.read_csv(HERE / "data.csv")
A = pd.read_csv(HERE / "statistics.csv").iloc[0]
PROF = [("sae300_within_fold", "sae300", "sparse", S.SPARSE),
        ("prior_within_fold", "prior", "GPT-2 XL", S.PRIOR),
        ("gemma_within_fold", None, "Gemma-2-2B dense", S.DENSE)]

fig, ax = S.panel(148, 54, box=(14, 6, 89, 38))
S.note(fig, f"cross-participant within-type correlation {S.num(A.within_isc)}, $P$ = {A.within_p:.4f}",
       2, 1.0, size=S.T_SMALL)
S.ref(ax, 0, horizontal=False)
ax.axhline(0, color=S.INK2, lw=S.LW_REF, zorder=1)
for col, pooled, lab, c in PROF:
    if pooled:
        ax.plot(PR.lag_ms, PR[pooled], lw=0.8, color=c, ls=(0, (3, 2)), alpha=0.7, zorder=3)
    ax.plot(PR.lag_ms, PR[col], lw=S.LW_DATA, color=c, zorder=4)
    i = int(np.argmax(PR[col].to_numpy()))
    ax.plot([PR.lag_ms.iloc[i]], [PR[col].iloc[i]], "o", ms=3.2, color=c, mec="white", mew=0.5, zorder=5)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-0.07, 0.105)
ax.set_xticks([-2000, -1000, 0, 1000, 2000])
ax.set_yticks([-0.05, 0, 0.05, 0.10])
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
ax.yaxis.set_major_formatter(S.minus_formatter("{:.2f}"))
ax.set_xlabel("lag from word onset (ms)", labelpad=1)
ax.set_ylabel("within-type r", labelpad=1.5)
S.despine(ax)
XK = 107.0
for k, (col, _, lab, c) in enumerate(PROF):
    i = int(np.argmax(PR[col].to_numpy()))
    yk = 8.0 + 7.0 * k
    S.note(fig, f"{lab}, 300 PCs", XK, yk, va="center", size=S.T_SMALL, color=c, fontweight="bold")
    S.note(fig, f"peak {S.num(PR[col].iloc[i], '+.3f')} at {PR.lag_ms.iloc[i]:+.0f} ms", XK, yk + 3.0,
           va="center", size=S.T_SMALL)
W, H = fig._mm
for yl, kw in ((31.5, dict(lw=S.LW_DATA)), (35.1, dict(lw=0.8, ls=(0, (3, 2))))):
    fig.add_artist(Line2D([XK / W, (XK + 4.6) / W], [1 - yl / H] * 2, transform=fig.transFigure,
                          color=S.INK2, **kw))
S.note(fig, "per-fold form", XK + 6.0, 31.5, va="center", size=S.T_SMALL)
S.note(fig, "pooled form (sparse, GPT-2 XL),", XK + 6.0, 35.1, va="center", size=S.T_SMALL)
S.note(fig, "biased against word identity", XK + 6.0, 38.1, va="center", size=S.T_SMALL)
S.save(fig, __file__)
