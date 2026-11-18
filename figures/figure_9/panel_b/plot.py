"""Figure 9b: the reconstruction error alone (dense stream minus the dictionary's
reconstruction; open markers), total and within-type out-of-fold r in IFG, in both cohorts;
horizontal bars, the dense stream's median. Circles, podcast; diamonds, film.

data.csv: per cohort and component, the error's median r, its 95% interval, P and the dense
stream's median r.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
fig, b = S.panel(60, 46, box=(13, 7, 43, 29))
xt, xl = [], []
for j, (coh, mk, chip) in enumerate((("Podcast", "o", "Podcast · ECoG"), ("Brain Treebank", "D", "Film · sEEG"))):
    for k, (row, lab) in enumerate((("error, total", "total"), ("error, within type", "within\ntype"))):
        x = j * 2.5 + k
        r = D[(D.cohort == coh) & (D.row == row)].iloc[0]
        b.plot([x - 0.30, x + 0.30], [r.dense_median] * 2, color=S.DENSE, lw=1.8, solid_capstyle="butt", zorder=2)
        S.interval(b, x, r.lo, r.hi, S.DENSE_L, lw=0.9, z=3)
        b.plot([x], [r["median"]], mk, ms=3.8 if mk == "o" else 3.4, mfc="white", mec=S.DENSE, mew=0.9, zorder=4)
        xt.append(x)
        xl.append(lab)
    b.text(j * 2.5 + 0.5, 1.04, chip, transform=b.get_xaxis_transform(), ha="center", va="bottom",
           fontsize=S.T_SMALL, color=S.INK, fontweight="bold")
b.axhline(0, color=S.RULE, lw=S.LW_REF, zorder=1)
b.set_xticks(xt)
b.set_xticklabels(xl, fontsize=S.T_SMALL, linespacing=1.0)
b.tick_params(axis="x", length=0)
b.set_xlim(-0.6, 4.1)
b.set_ylim(-0.005, 0.14)
b.set_yticks([0, 0.05, 0.10])
b.set_ylabel("IFG r, out of fold", labelpad=1.5)
S.despine(b, bottom=False)
p0 = D[(D.cohort == "Podcast") & (D.row == "error, total")].iloc[0]
b.text(0.36, p0.dense_median, "dense", color=S.DENSE, fontsize=S.T_SMALL, ha="left", va="center", fontweight="bold")
b.text(0.14, p0["median"], "error", color=S.DENSE, fontsize=S.T_SMALL, ha="left", va="center")
S.note(fig, "open, error; bars, dense stream", 2, 1.0, size=S.T_SMALL)
S.save(fig, __file__)
