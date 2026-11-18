"""Figure 4e: the same electrodes. Peak unique variance of the sparse and dense bases at all
1,264 electrodes, binned on log axes (colour, electrodes per cell); dashed, identity; Spearman rho.

data.csv: per electrode, its participant, region and the peak unique variance of each basis.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
xs, ys = D.sparse_peak_unique_dr2.to_numpy(), D.dense_peak_unique_dr2.to_numpy()
rho = float(sps.spearmanr(xs, ys).statistic)
KEY = (1, 10, 100)                                  # electrodes per cell: key ticks and colour range

fig, ax = S.panel(50, 50, box=(11, 3, 36, 36))
assert (xs > 0).all() and (ys > 0).all()
lim = (5e-4, 0.25)
le = np.log10(lim)
hb = ax.hexbin(xs, ys, xscale="log", yscale="log", gridsize=17, extent=(le[0], le[1], le[0], le[1]),
               mincnt=1, cmap=S.SEQ_CMAP, norm=LogNorm(vmin=1, vmax=KEY[-1]), linewidths=0.15,
               edgecolors="white", zorder=2)
counts = hb.get_array()
assert int(counts.sum()) == len(xs) and counts.max() <= KEY[-1]       # every electrode binned, in key
ax.plot(lim, lim, color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=3)
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.minorticks_off()
for axis in (ax.xaxis, ax.yaxis):
    axis.set_ticks([1e-3, 1e-2, 1e-1])
    axis.set_ticklabels([S.pow10(e) for e in (-3, -2, -1)])
ax.text(0.04, 0.96, f"ρ = {rho:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=S.T_ANNOT,
        color=S.INK, fontweight="bold")
ax.set_xlabel("sparse, peak unique ΔR²", labelpad=1, color=S.SPARSE)
ax.set_ylabel("dense, peak unique ΔR²", labelpad=1, color=S.DENSE)
S.despine(ax)
kx = ax.inset_axes([0.50, 0.105, 0.44, 0.045])
kx.imshow(np.linspace(0, np.log10(KEY[-1]), 256)[None, :], aspect="auto", cmap=S.SEQ_CMAP,
          extent=(0, np.log10(KEY[-1]), 0, 1), interpolation="bilinear")
kx.set_yticks([])
kx.set_xticks(np.log10(KEY))
kx.set_xticklabels([f"{k:,}" for k in KEY])
kx.tick_params(axis="x", length=1.0, width=S.LW_AX, labelsize=S.T_SMALL, pad=0.8, color=S.INK2)
for sp_ in kx.spines.values():
    sp_.set_visible(False)
kx.set_title("electrodes per cell", fontsize=S.T_SMALL, color=S.INK2, pad=1.2)
S.save(fig, __file__)
