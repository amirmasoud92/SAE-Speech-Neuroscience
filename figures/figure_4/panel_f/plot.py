"""Figure 4f: the equivalence bound. The sparse-minus-dense difference in IFG mean out-of-fold r
(300 dimensions each) with 90% (thick) and 95% (thin) participant-clustered intervals, against the
margin Delta (shaded), which is the gap between the two reference arms drawn above on the same
scale: GPT-2 XL at 0 and the Gemma-2-2B dense stream at +Delta.

data.csv: the margin, the two reference means, the observed difference, its intervals, the
non-superiority and TOST P values and the electrode and participant counts.
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
a = pd.read_csv(HERE / "data.csv").iloc[0]
delta = float(a.margin_delta)
XL = (-0.028, 0.024)
axx, axw = 4.0, 74.0
mm_per = axw / (XL[1] - XL[0])

fig = S.figure(82, 66)
S.note(fig, f"IFG mean r, 300 dimensions each; n = {int(a.n_electrodes)} electrodes, "
            f"{int(a.n_subjects)} participants", 2, 1.0, size=S.T_SMALL)
dax = S.ax_mm(fig, axx, 5.0, axw, 10.0)
dax.set_xlim(*XL)
dax.set_ylim(0, 1)
dax.axis("off")
dax.plot([0, delta], [0.30, 0.30], color=S.INK2, lw=0.5, zorder=2)
for xv, v, c, lab in ((0.0, a.gpt2xl_mean_r, S.PRIOR, "GPT-2 XL"),
                      (delta, a.gemma_mean_r, S.DENSE, "Gemma-2-2B dense")):
    dax.plot([xv], [0.30], "o", ms=4.0, color=c, mec="white", mew=0.5, zorder=3)
    dax.text(xv, 0.55, f"{lab}\nr = {v:.4f}", fontsize=S.T_SMALL, color=c, ha="center", va="bottom",
             linespacing=1.0)
dax.text(delta / 2, 0.20, f"Δ = {delta:.4f}", fontsize=S.T_SMALL, color=S.INK, ha="center", va="top",
         fontweight="bold")
S.note(fig, "the margin is the gap\nbetween the two\nreference arms", axx, 6.0, size=S.T_SMALL,
       color=S.INK2, linespacing=1.05)
ax = S.ax_mm(fig, axx, 16.0, axw, 24.0)
ax.axvspan(-delta, delta, color=S.SPARSE_XL, alpha=0.55, lw=0, zorder=0)
ax.axvspan(XL[0], 0, color=S.DENSE_FIELD, alpha=0.6, lw=0, zorder=-1)
for xv in (-delta, delta):
    ax.axvline(xv, color=S.INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
for xv in (0.0, delta):                              # the derivation drops onto the axis
    fig.add_artist(plt.Line2D(*zip(S.fxy(fig, axx + (xv - XL[0]) * mm_per, 12.0),
                                   S.fxy(fig, axx + (xv - XL[0]) * mm_per, 16.0)),
                              transform=fig.transFigure, color=S.INK2, lw=0.45, ls=(0, (1, 1.2))))
ax.axvline(0, color=S.INK2, lw=0.6, zorder=1)
S.interval(ax, 0, a.ci95_lo, a.ci95_hi, S.SPARSE_L, horizontal=True, lw=1.2, z=3)
ax.plot([a.ci90_lo, a.ci90_hi], [0, 0], color=S.SPARSE, lw=5.0, zorder=4, solid_capstyle="butt")
ax.plot([a.observed_diff], [0], "o", color="white", ms=6.2, mec=S.SPARSE, mew=1.4, zorder=5)
ax.text(a.observed_diff + 0.0012, 0.26, S.num(a.observed_diff), fontsize=S.T_ANNOT, color=S.INK,
        ha="left", va="bottom", fontweight="bold", zorder=6)
for xb in (a.ci90_lo, a.ci90_hi):
    ax.text(xb, -0.30, S.num(xb), fontsize=S.T_SMALL, color=S.SPARSE, ha="center", va="top", zorder=6)
ax.text(-delta + 0.0005, 0.93, "−Δ", fontsize=S.T_ANNOT, color=S.INK2, ha="left", va="top")
ax.text(delta + 0.0005, 0.93, "+Δ", fontsize=S.T_ANNOT, color=S.INK2, ha="left", va="top")
ax.text(0.0, -0.36, "← dense better", transform=ax.transAxes, fontsize=S.T_SMALL, color=S.DENSE,
        ha="left", va="top")
ax.text(1.0, -0.36, "sparse better →", transform=ax.transAxes, fontsize=S.T_SMALL, color=S.SPARSE,
        ha="right", va="top")
ax.set_yticks([])
ax.set_ylim(-1.0, 1.0)
ax.set_xlim(*XL)
ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
ax.set_xticklabels(["−0.02", "−0.01", "0", "0.01", "0.02"])
ax.set_xlabel("difference in r", labelpad=1)
S.despine(ax, left=False)
ns = "$P$ < 0.0001" if a.p_nonsuperiority < 1e-4 else f"$P$ = {a.p_nonsuperiority:.4f}"
for k, (head, rest) in enumerate([
        ("No gain", f"upper 90% bound {S.num(a.ci90_hi)}, below zero and +Δ ({ns})"),
        ("Loss not excluded", f"lower 90% bound {S.num(a.ci90_lo)} beyond −Δ, TOST $P$ = {a.tost_p:.2f}")]):
    S.note(fig, head, axx, 53.0 + 3.3 * k, color=S.INK, size=S.T_SMALL, fontweight="bold")
    S.note(fig, rest, axx + 19.0, 53.0 + 3.3 * k, color=S.INK, size=S.T_SMALL)
S.note(fig, "bars, 90% (thick) and 95% (thin) participant-clustered intervals", axx, 60.0, size=S.T_SMALL)
S.save(fig, __file__)
