"""Supplementary Figure 7a: electrodes whose peak over lags of a closed-form, fixed-penalty proxy
of the sparse band's unique variance survives family-wise control (the 95th percentile of the
maximum over 654 electrodes x 161 lags in 1,000 circular shifts of the stimulus), on the fsaverage
pial surface (nearest vertex; display only). Coloured, language-region survivors; black,
control-region survivors; grey, not surviving; outlines, IFG, STG, MTG and TMP.

data.csv: per electrode, its participant, region, MNI coordinates, observed peak, peak lag,
family-wise P and whether it survives.
Requires nilearn (fsaverage surface and Destrieux atlas, fetched on first use).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import brain as A  # noqa: E402
import style as S  # noqa: E402

S.apply()
E = pd.read_csv(HERE / "data.csv")
sv = E[E.survives.astype(bool)]

fig = S.figure(183, 72)
S.note(fig, f"{len(sv)} of {len(E)} electrodes survive family-wise control", 2, 1.0, size=S.T_SMALL, color=S.INK,
       fontweight="bold")
for hemi, (x, y, w, h) in (("left", (2, 7, 92, 62)), ("right", (98, 25, 46, 31))):
    ax = S.ax_mm(fig, x, y, w, h)
    A.lateral(ax, hemi, A.ROI_MAP, tint=(), rim=S.LANGUAGE, rim_alpha=0.9)
    sub = E[(E.mni_x < 0) if hemi == "left" else (E.mni_x >= 0)]
    P = A.snap(sub[["mni_x", "mni_y", "mni_z"]].to_numpy(float), hemi)
    k = 1.0 if hemi == "left" else 0.6
    on = sub.survives.astype(bool).to_numpy()
    ax.scatter(P[~on, 0], P[~on, 1], s=2.0 * k, color=S.CTRL, lw=0, alpha=0.75, zorder=4)
    cols = [S.REGION[r] if r in S.LANGUAGE else S.INK for r in sub.roi[on]]
    ax.scatter(P[on, 0], P[on, 1], s=11.0 * k, c=cols, lw=0.35, edgecolors="white", zorder=5)
    S.note(fig, "L" if hemi == "left" else "R", x + 2 if hemi == "left" else x + 1, y + 2 if hemi == "left" else y - 1,
           size=S.T_SMALL, fontweight="bold")
kx, ky = 150.0, 30.0
kax = S.ax_mm(fig, kx, ky, 32, 21)
kax.axis("off")
kax.set_xlim(0, 1)
kax.set_ylim(0, 1)
items = [(S.REGION[r], f"{r} survivor", 11.0, "white") for r in S.LANGUAGE] + \
        [(S.INK, "control-region survivor", 11.0, "white"), (S.CTRL, "not surviving", 2.0, "none")]
for i, (c, lab, sz, ec) in enumerate(items):
    yy = 1 - (i + 0.5) / len(items)
    kax.scatter([0.06], [yy], s=sz, color=c, edgecolors=ec, lw=0.35)
    kax.text(0.16, yy, lab, fontsize=S.T_SMALL, color=S.INK, va="center", ha="left")
S.note(fig, "outlines: IFG, STG, MTG, TMP", kx, ky + 24, size=S.T_SMALL)
S.save(fig, __file__)
