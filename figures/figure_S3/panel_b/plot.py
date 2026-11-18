"""Supplementary Figure 3 (a, sparse feature band; b, acoustic mel-spectrogram band; the same
script draws each panel from its folder name). Peak unique held-out variance across lags by region;
each point is one electrode; bars, interquartile range; black ticks, medians; half-violins, the
full distribution.

data.csv: per electrode, its participant, region and peak unique variance.
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
P = HERE.name.split("_")[-1]
lab, col, ymax = {"a": ("Sparse feature band", S.SPARSE, 0.14),
                  "b": ("Acoustic band (mel spectrogram)", S.BAND_ACOUSTIC, 0.50)}[P]
ORDER = ["IFG", "STG", "MFG", "MTG", "EAC", "TMP", "PRC"]
rng = np.random.default_rng(0)

fig, ax = S.panel(92, 66, box=(13, 5, 76, 52))
S.note(fig, lab, 13, 1.0, size=S.T_SMALL, color=S.INK, fontweight="bold")
for i, r in enumerate(ORDER):
    v = D[D.roi == r].peak_delta_r2.to_numpy()
    S.half_violin(ax, i + 0.06, v, col, width=0.34, alpha=0.26, bw=0.3)
    ax.scatter(i - 0.16 + rng.uniform(-0.08, 0.08, len(v)), v, s=1.6, color=col, alpha=0.5, lw=0, zorder=2)
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([i + 0.02, i + 0.02], [q1, q3], color=col, lw=2.0, zorder=3, solid_capstyle="butt")
    ax.plot([i - 0.04, i + 0.08], [md, md], color=S.INK, lw=1.0, zorder=4)
ax.axhline(0.0, color=S.RULE, lw=S.LW_REF, zorder=0)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=S.T_SMALL, fontweight="bold")
for tl in ax.get_xticklabels():
    tl.set_color(S.REGION[tl.get_text()] if tl.get_text() in S.LANGUAGE else S.INK2)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, len(ORDER) - 0.4)
ax.set_ylim(-0.005, ymax)
ax.set_ylabel("peak unique ΔR²", labelpad=1.5)
S.despine(ax, bottom=False)
S.save(fig, __file__)
