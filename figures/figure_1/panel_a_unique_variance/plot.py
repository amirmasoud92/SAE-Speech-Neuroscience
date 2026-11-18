"""Figure 1a (encoding model): held-out R2 at one IFG electrode across the 161 lags, with all four
bands and without the sparse band; the shaded gap is the sparse band's unique variance.

data.csv: lag (s) and the two held-out R2 profiles of the example electrode.
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
lms = D.lag_s.to_numpy() * 1000.0
full, wo = D.r2_full.to_numpy(), D.r2_without_sparse.to_numpy()

fig, ax = S.panel(40, 26, box=(3, 6, 34, 13))
ax.fill_between(lms, wo, full, where=full >= wo, color=S.SPARSE, alpha=0.85, lw=0, zorder=2)
ax.plot(lms, full, color=S.INK, lw=0.7, zorder=3)
ax.plot(lms, wo, color=S.CTRL, lw=0.7, zorder=3)
ax.set_xlim(-2000, 2000)
ax.set_ylim(0, max(full.max(), wo.max()) * 1.08)
ax.set_xticks([-2000, 0, 2000])
ax.set_xticklabels(["−2,000", "0", "+2,000"], fontsize=S.T_SMALL)
ax.set_xlabel("lag (ms)", fontsize=S.T_SMALL, labelpad=0.5)
ax.set_yticks([])
ax.tick_params(axis="x", length=1.5, pad=1)
S.despine(ax, left=False, offset_mm=0.4)
ax.axvline(0, color=S.RULE, lw=S.LW_REF, zorder=0)
ipk = int(np.argmax(full - wo))
ax.annotate("unique ΔR²", xy=(lms[ipk] + 50, (full[ipk] + wo[ipk]) / 2), xytext=(950, full.max() * 0.98),
            fontsize=S.T_SMALL, color=S.SPARSE, fontweight="bold", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.45, color=S.SPARSE, shrinkA=0.5, shrinkB=0.5))
ax.text(-1950, full.max() * 0.98, "all bands", fontsize=S.T_SMALL - 0.3, va="center")
ax.text(-1950, full.max() * 0.66, "without\nsparse", fontsize=S.T_SMALL - 0.3, color=S.INK2,
        va="center", linespacing=0.95)
S.note(fig, f"held-out R², one IFG electrode ({D.subject.iloc[0]}, {D.ch.iloc[0]})", 2, 1.5,
       size=S.T_SMALL)
S.save(fig, __file__)
