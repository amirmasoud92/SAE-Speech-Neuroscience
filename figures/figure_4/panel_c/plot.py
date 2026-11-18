"""Figure 4c: denoised targets. IFG mean out-of-fold r of the dense and sparse arms when the
neural targets are averaged over 1, 2, 4 and 8 participants (dots, individual subsets; lines,
means; subset counts in brackets).

data.csv: per participant subset, the number averaged, the mean r of each arm and their difference.
statistics.csv: the sparse-minus-dense gap at eight participants, its interval and moving-block
bootstrap P.
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
T = pd.read_csv(HERE / "statistics.csv").iloc[0]
agg = D.groupby("n_subjects").agg(gemma=("gemma", "mean"), sae=("sae", "mean"),
                                  k=("subset", "size")).reset_index()

fig, ax = S.panel(72, 48, box=(12, 3, 58, 35))
rng = np.random.default_rng(2)
for _, r in D.iterrows():
    j = rng.uniform(-0.06, 0.06)
    x0 = np.log2(r.n_subjects)
    ax.plot([x0 - 0.13 + j], [r.gemma], "o", ms=1.9, color=S.DENSE_L, alpha=0.9, zorder=2)
    ax.plot([x0 + 0.13 + j], [r.sae], "o", ms=1.9, color=S.SPARSE_L, alpha=0.9, zorder=2)
lx = np.log2(agg.n_subjects)
for off, col, c in ((-0.13, "gemma", S.DENSE), (0.13, "sae", S.SPARSE)):
    ax.plot(lx + off, agg[col], "-", color=c, lw=S.LW_DATA, zorder=3)
    ax.plot(lx + off, agg[col], "o", color=c, ms=3.4, mec="white", mew=0.4, zorder=4)
ax.text(lx.iloc[1] - 0.25, agg.gemma.iloc[1] + 0.028, "dense", fontsize=S.T_SMALL, color=S.DENSE,
        ha="right", fontweight="bold", bbox=S.TEXT_BG, zorder=6)
ax.text(lx.iloc[2] + 0.30, agg.sae.iloc[2] - 0.034, "sparse", fontsize=S.T_SMALL, color=S.SPARSE,
        ha="left", fontweight="bold", bbox=S.TEXT_BG, zorder=6)
ax.set_xticks(lx)
ax.set_xticklabels([f"{int(n)} ({int(k)})" for n, k in zip(agg.n_subjects, agg.k)])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.45, 3.45)
ax.set_ylim(-0.04, 0.36)
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_xlabel("participants averaged (subsets)", labelpad=1)
ax.set_ylabel("IFG mean r, out of fold", labelpad=1.5)
ax.axhline(0, color=S.RULE, lw=S.LW_REF, zorder=0)
S.rich_text(ax, 3.42, 0.006, f"gap at 8: {S.num(T.diff_n8)}, {S.p_str(T.p_n8)}", ax.transData,
            ha="right", va="bottom", fontsize=S.T_SMALL, color=S.INK)
S.despine(ax, bottom=False)
S.save(fig, __file__)
