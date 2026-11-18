"""Figure 7b (forest): Spearman correlations over the 300 pooled features with 95% bootstrap
intervals over features; filled markers, the two pre-specified tests; grey band, the pre-specified
|rho| < 0.15 floor. (The schematic drawn above the forest in the paper summarises the same
correlations and has no script.)

data.csv: per relationship, its group, label, rho, interval and whether it is a pre-specified test.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
rows = []
for g, d in D.groupby("group", sort=False):
    rows.append((g, None))
    rows += [(r.label, r) for r in d.itertuples()]
BOX = 38.0

fig, ax = S.panel(76, 62, box=(35, 3, BOX, BOX))
ax.axvspan(-0.15, 0.15, color=S.FIELD, lw=0, zorder=0)
ax.axvline(0, color=S.INK2, lw=0.6, zorder=1)
tl, tv = -10.0 / BOX, -1.5 / BOX                        # label and value columns, axes fraction
for k, (lab, r) in enumerate(rows):
    yy = len(rows) - 1 - k
    tr = ax.get_yaxis_transform()
    if r is None:
        ax.text(tl, yy, lab, transform=tr, ha="right", va="center", fontsize=S.T_SMALL, color=S.INK,
                fontweight="bold")
        continue
    test = bool(r.prespecified_test)
    S.interval(ax, yy, r.ci_lo, r.ci_hi, S.SPARSE if test else S.SPARSE_L, horizontal=True,
               lw=1.1 if test else 0.8, z=3)
    ax.plot([r.rho], [yy], "o", ms=4.0 if test else 3.3, mfc=S.SPARSE if test else "white", mec=S.SPARSE,
            mew=0.8, zorder=4)
    ax.text(tl, yy, lab, transform=tr, ha="right", va="center", fontsize=S.T_SMALL,
            color=S.INK if test else S.INK2)
    ax.text(tv, yy, S.num(r.rho, "+.3f"), transform=tr, ha="right", va="center", fontsize=S.T_SMALL,
            color=S.INK if test else S.INK2, fontweight="bold" if test else "normal")
ax.set_xlim(-0.95, 0.95)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks([])
ax.set_xticks([-0.5, 0, 0.5])
ax.xaxis.set_major_formatter(S.minus_formatter("{:g}"))
ax.set_xlabel("Spearman ρ", labelpad=1)
S.despine(ax, left=False)
kd = S.ax_mm(fig, 2.0, 50.0, 3.4, 3.0)
kd.set_xlim(0, 1)
kd.set_ylim(0, 1)
kd.axis("off")
kd.plot([0.5], [0.5], "o", ms=4.0, mfc=S.SPARSE, mec=S.SPARSE, mew=0.8)
S.note(fig, "pre-specified test", 6.5, 51.5, va="center", size=S.T_SMALL)
kf = S.ax_mm(fig, 2.0, 54.4, 3.4, 2.2)
kf.set_facecolor(S.FIELD)
kf.set_xticks([])
kf.set_yticks([])
for sp_ in kf.spines.values():
    sp_.set_visible(False)
S.note(fig, "|ρ| < 0.15, the floor", 6.5, 55.5, va="center", size=S.T_SMALL)
S.save(fig, __file__)
