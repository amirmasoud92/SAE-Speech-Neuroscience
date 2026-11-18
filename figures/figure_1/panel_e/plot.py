"""Figure 1e: peak unique sparse-band variance at every electrode, unthresholded, on the fsaverage
pial surface; colour and size encode magnitude from 0 to the 99th centile; the four language
regions are outlined. Electrodes are drawn at the nearest surface vertex (display only).

data.csv: per electrode, the peak unique sparse-band variance, its region and MNI coordinates.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import cm, colors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import brain as A  # noqa: E402
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
xyz = D[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
pk = D.peak_delta_sae.to_numpy(float)
vmax = float(np.percentile(pk, 99))

fig = S.figure(132, 64)
for hemi, (x, y, w, h) in (("left", (0, 2, 90, 60)), ("right", (90, 19, 41, 28))):
    ax = S.ax_mm(fig, x, y, w, h)
    A.lateral(ax, hemi, tint=(), rim=S.LANGUAGE, rim_alpha=0.85)
    m = (xyz[:, 0] < 0) if hemi == "left" else (xyz[:, 0] >= 0)
    P = A.snap(xyz[m], hemi)
    v = np.clip(pk[m], 0, vmax)
    o = np.argsort(v)
    sz = (1.0 + 24.0 * (v[o] / vmax) ** 1.5) * (1.0 if hemi == "left" else 0.6)
    ax.scatter(P[o, 0], P[o, 1], s=sz, c=v[o], cmap=S.SPARSE_CMAP, vmin=0, vmax=vmax, lw=0.22,
               edgecolors="white", zorder=5)
    S.note(fig, "L" if hemi == "left" else "R", x + 3 if hemi == "left" else x + 1,
           y + 6 if hemi == "left" else y - 0.5, size=S.T_SMALL, fontweight="bold")
cax = S.ax_mm(fig, 100, 54.0, 24, 1.5)
cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0.0, vmax), cmap=S.SPARSE_CMAP), cax=cax,
                  orientation="horizontal")
cb.set_ticks([0.0, vmax / 2, vmax])
cb.ax.set_xticklabels(["0", f"{vmax / 2:.3f}", f"{vmax:.3f}"], fontsize=S.T_SMALL)
cb.ax.tick_params(length=1.5, pad=1, width=0.4)
cb.outline.set_visible(False)
S.note(fig, "peak unique ΔR² (to 99th centile)", 112, 50.6, ha="center", size=S.T_SMALL)
S.save(fig, __file__)
