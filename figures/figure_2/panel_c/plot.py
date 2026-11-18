"""Figure 2c: each band's unique variance as a share of the full model's held-out R2 at each
electrode's peak lag, for the IFG electrodes the full model explains (R2 > 0.01); bars, medians.
Held-out shares can fall below 0 or exceed 100%; points beyond -25% or 125% are drawn at the edge.

data.csv: per electrode, the full-model peak R2, each band's unique variance and its share.
medians.csv: the median share of each band in IFG.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
F = pd.read_csv(HERE / "data.csv")
MED = pd.read_csv(HERE / "medians.csv").set_index("band").median_fraction_IFG
BANDS = [("sae", "sparse", S.SPARSE), ("acoustic", "acoustic", S.BAND_ACOUSTIC),
         ("lexical", "lexical", S.BAND_LEXICAL), ("surprisal", "surprisal", S.BAND_SURPRISAL)]

fig, ax = S.panel(86, 38, box=(20, 6, 62, 22))
for i, (b, nm, c) in enumerate(BANDS):
    v = 100 * F[f"frac_{b}"].dropna().to_numpy()
    vc = np.clip(v, -25, 125)
    ax.scatter(vc, i + S.beeswarm_offsets(vc, width=0.36, nbins=40), s=3.0, color=c, alpha=0.75, lw=0,
               zorder=2)
    med = 100 * float(MED[b])
    ax.plot([med, med], [i - 0.42, i + 0.42], color=S.INK, lw=1.2, zorder=4, solid_capstyle="butt")
    ax.text(med + 2.5, i - 0.30, f"{S.num(med, '.1f')}%", fontsize=S.T_SMALL, ha="left", va="center",
            bbox=S.TEXT_BG, zorder=5, fontweight="bold" if b == "sae" else "normal")
    ax.text(-0.03, i, nm, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=S.T_SMALL,
            color=c if b != "surprisal" else S.INK2, fontweight="bold" if b == "sae" else "normal")
ax.axvline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.axvline(100, lw=S.LW_REF, color=S.RULE, zorder=0, ls=(0, (3, 2)))
ax.set_yticks([])
ax.set_xlim(-25, 125)
ax.set_ylim(len(BANDS) - 0.5, -0.5)
ax.set_xticks([-25, 0, 25, 50, 75, 100, 125])
ax.set_xticklabels(["", "0", "", "50", "", "100", ""])
ax.set_xlabel("share of full-model R² (%)", labelpad=1)
S.despine(ax, left=False)
S.note(fig, f"{len(F)} IFG electrodes with full-model R² > 0.01; bar, median", 2, 1.5, size=S.T_SMALL)
S.save(fig, __file__)
