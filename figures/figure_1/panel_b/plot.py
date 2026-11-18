"""Figure 1b: all analysis-eligible electrodes on the fsaverage pial surface, coloured by assigned
region, with the language-network Destrieux parcels tinted; electrodes are drawn at the nearest
surface vertex (display only).

data.csv: participant, electrode, region and MNI coordinates of every analysed electrode.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import brain as A  # noqa: E402
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
xyz = D[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
roi = D.roi.to_numpy()
rank = {r: i for i, r in enumerate(["other", "PRC", "MFG", "EAC", "TMP", "MTG", "STG", "IFG"])}


def darker(c, f=0.78):
    return tuple(np.array(to_rgb(c)) * f)


fig = S.figure(90, 52)
for hemi, (x, y, w, h) in (("left", (1, 4, 60, 42)), ("right", (60, 26, 27, 18.5))):
    ax = S.ax_mm(fig, x, y, w, h)
    cent, froi = A.lateral(ax, hemi, tint=S.LANGUAGE, rim=(), tint_alpha=0.22)
    m = (xyz[:, 0] < 0) if hemi == "left" else (xyz[:, 0] >= 0)
    P = A.snap(xyz[m], hemi)
    rr = roi[m]
    k_ = 1.0 if hemi == "left" else 0.75
    for kk in np.argsort([rank[r] for r in rr]):
        r = rr[kk]
        if r in S.LANGUAGE:
            ax.plot(*P[kk], "o", ms=2.2 * k_, mfc=S.REGION[r], mec="white", mew=0.25, zorder=5)
        elif r in ("EAC", "MFG", "PRC"):
            ax.plot(*P[kk], "o", ms=1.5 * k_, mfc=S.REGION[r], mec="none", zorder=4)
        else:
            ax.plot(*P[kk], "o", ms=1.1 * k_, mfc="white", mec="#9A958E", mew=0.25, zorder=3)
    if hemi == "left":
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        Wd, Ht = x1 - x0, y1 - y0
        ax.set_ylim(y0 - 10, y1)
        spots = {"IFG": (x0 + 0.07 * Wd, y0 + 0.10 * Ht), "TMP": (None, y0 - 5),
                 "MTG": (x0 + 0.70 * Wd, y0 + 0.04 * Ht), "STG": (x0 + 0.90 * Wd, y0 + 0.80 * Ht)}
        for r, (lx, ly) in spots.items():
            a = A.roi_anchor(cent, froi, r)
            ax.annotate(r, xy=a, xytext=(a[0] - 14 if lx is None else lx, ly), fontsize=S.T_ANNOT,
                        fontweight="bold", color=darker(S.REGION[r]), ha="center", va="center",
                        zorder=8, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=0.45, color=S.INK2, shrinkA=2.0, shrinkB=0))
    S.note(fig, "L" if hemi == "left" else "R", x + 2 if hemi == "left" else x + w - 1,
           y + 4 if hemi == "left" else y - 0.5, size=S.T_SMALL, fontweight="bold",
           ha="left" if hemi == "left" else "right")
S.note(fig, f"{len(D):,} electrodes, {D.subject.nunique()} participants; "
            f"{int(np.isin(roi, S.LANGUAGE).sum())} in the four language regions", 2, 48,
       size=S.T_SMALL)
S.save(fig, __file__)
