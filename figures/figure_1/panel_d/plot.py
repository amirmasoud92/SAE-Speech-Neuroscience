"""Figure 1d: median peak unique variance by region for the sparse (blue) and acoustic (green)
bands, with open markers for the sparse band's mean over 0-500 ms (triangles: medians at or below
zero, drawn at the axis floor); bars, 95% participant-clustered bootstrap intervals; control
regions shaded; logarithmic axis. The mixed-effects model (language against control networks,
per-participant random intercept) is printed below.

data.csv: per region and statistic, the median, its interval and the electrode count.
mixed_model.csv: the language-versus-control coefficient (SD) and P, peak and window statistics.
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
D = pd.read_csv(HERE / "data.csv")
MM = pd.read_csv(HERE / "mixed_model.csv").set_index("statistic")
ROI_ORDER = ["IFG", "STG", "MTG", "TMP", "MFG", "EAC", "PRC"]
FLOOR, xlo, xhi = 1.2e-4, 7e-5, 0.25
nr = len(ROI_ORDER)
row_h = 4.9

fig, ax = S.panel(62, 66, box=(22, 9, 36.5, nr * row_h))
ax.set_xscale("log")
ax.set_xlim(xlo, xhi)
ax.set_ylim(nr - 0.5, -0.5)
ax.axhspan(3.5, nr - 0.5, color=S.FIELD, lw=0, zorder=0)
for i in range(nr):
    ax.axhline(i, color=S.CTRL_XL, lw=0.35, zorder=0.2)
offs = {"SAE": -0.24, "SAE mean 0-500 ms": 0.0, "acoustic": 0.24}
for i, r in enumerate(ROI_ORDER):
    for band, col in (("SAE", S.SPARSE), ("SAE mean 0-500 ms", S.SPARSE), ("acoustic", S.BAND_ACOUSTIC)):
        q = D[(D.roi == r) & (D.band == band)].iloc[0]
        y = i + offs[band]
        lo, hi = max(q.ci_lo, xlo * 1.05), q.ci_hi
        if hi > xlo:
            ax.plot([lo, hi], [y, y], color=S.SPARSE_L if band == "SAE mean 0-500 ms" else col, lw=0.8,
                    solid_capstyle="butt", zorder=2)
        if q.median_peak_dr2 > FLOOR:
            ax.plot([q.median_peak_dr2], [y], "o", ms=3.0 if band != "SAE mean 0-500 ms" else 2.8,
                    mfc="white" if band == "SAE mean 0-500 ms" else col, mec=col, mew=0.7, zorder=4)
        else:
            ax.plot([xlo * 1.12], [y], "<", ms=2.8, mfc="white", mec=S.SPARSE, mew=0.7, zorder=4,
                    clip_on=False)
    c = tuple(np.array(to_rgb(S.REGION[r])) * 0.78) if r in S.LANGUAGE else S.INK2
    ax.text(-0.32, i, r, transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=S.T_SMALL, fontweight="bold", color=c)
    ax.text(-0.04, i, f"{int(D[D.roi == r].n.iloc[0])}", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=S.T_SMALL - 0.3, color=S.INK2)
ax.text(-0.04, -0.75, "n", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=S.T_SMALL - 0.3, color=S.INK2, fontstyle="italic")
ax.set_yticks([])
ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
ax.set_xticklabels([S.pow10(e) for e in (-4, -3, -2, -1)])
ax.xaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10) * 0.1, numticks=12))
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("median unique ΔR²", labelpad=1)
S.despine(ax, left=False)
ax.text(xhi * 0.92, nr - 0.62, "control", ha="right", va="bottom", fontsize=S.T_SMALL, color=S.INK2,
        fontstyle="italic")
for kx, ky, mk, fc, ec, lab in ((2.0, 1.6, "o", S.SPARSE, S.SPARSE, "sparse, peak"),
                                (28.0, 1.6, "o", "white", S.SPARSE, "sparse, mean 0–500 ms"),
                                (2.0, 4.6, "o", S.BAND_ACOUSTIC, S.BAND_ACOUSTIC, "acoustic, peak"),
                                (28.0, 4.6, "<", "white", S.SPARSE, "median ≤ 0")):
    kax = S.ax_mm(fig, kx, ky - 1.0, 2.0, 2.0)
    kax.plot([0.5], [0.5], mk, ms=3.0 if mk == "o" else 2.8, mfc=fc, mec=ec, mew=0.7)
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    kax.axis("off")
    S.note(fig, lab, kx + 2.2, ky, va="center", size=S.T_SMALL - 0.3, color=S.INK)
y0 = 9 + nr * row_h + 9.5
S.note(fig, "language vs control, mixed model", 2.0, y0, size=S.T_SMALL)
for k, (st, lab) in enumerate((("peak", "peak"), ("mean 0-500 ms", "0–500 ms"))):
    b, p = MM.loc[st, "beta_sd"], MM.loc[st, "p"]
    e = int(np.floor(np.log10(p)))
    S.note(fig, rf"{lab}: $\beta$ = {S.num(b, '+.3f')} SD, $P$ = "
                f"{p / 10 ** e:.{2 if st == 'peak' else 1}f} × {S.pow10(e)}",
           2.0, y0 + 2.9 * (k + 1), size=S.T_SMALL, color=S.INK)
S.save(fig, __file__)
