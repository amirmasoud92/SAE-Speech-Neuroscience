"""Figure 10d (left, raw; right, reliability-corrected, post hoc: the same script in
../panel_d_corrected): agreement across regions. The partial Spearman rho between podcast and film
feature importance for every pair of regions (rows, podcast region; columns, film region);
same-region cells outlined; language regions in bold.

data.csv: per podcast region and film region, the partial rho.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
RAW = HERE.name.endswith("raw")
D = pd.read_csv(HERE / "data.csv")
regs = list(dict.fromkeys(D.podcast))
M = D.pivot(index="podcast", columns="film", values="value").loc[regs, regs]
vmax = 0.15 if RAW else 0.45

fig, ax = S.panel(50, 50, box=(13, 6, 27, 27))
im = ax.imshow(M.to_numpy(float), cmap=S.SPARSE_CMAP, vmin=0, vmax=vmax)
ax.set_xticks(range(len(regs)))
ax.set_xticklabels(regs, rotation=90, fontsize=S.T_SMALL)
ax.set_yticks(range(len(regs)))
ax.set_yticklabels(regs, fontsize=S.T_SMALL)
for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    r = tl.get_text()
    if r in S.LANGUAGE:
        tl.set_color(tuple(np.array(to_rgb(S.REGION[r])) * 0.75))
        tl.set_fontweight("bold")
    else:
        tl.set_color(S.INK2)
ax.tick_params(length=0, pad=1.5)
for sp in ax.spines.values():
    sp.set_visible(False)
for i in range(len(regs)):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=S.INK, lw=0.7))
ax.text(0.5, 1.03, "raw" if RAW else "reliability-corrected, post hoc", transform=ax.transAxes, ha="center",
        va="bottom", fontsize=S.T_SMALL, color=S.INK)
cax = S.ax_mm(fig, 41.3, 6, 1.3, 27)
cb = fig.colorbar(im, cax=cax)
cb.set_ticks([0, vmax / 2, vmax])
cb.ax.tick_params(labelsize=S.T_SMALL, length=1.5, pad=1)
cb.outline.set_visible(False)
ax.set_ylabel("podcast region", labelpad=1.5)
ax.set_xlabel("film region", labelpad=1.5)
S.save(fig, __file__)
