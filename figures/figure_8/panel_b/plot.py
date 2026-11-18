"""Figure 8b: between and within word types. Within Gemma-2-2B at 300 components, the
sparse-minus-dense difference split into its between-word-type and within-word-type components
(type means within test fold, types occurring at least four times) and their difference, in the
podcast cohort (135 electrodes, circles; training-only fits; filled, without and open, with timing
and neighbouring-word controls) and the film cohort (144 contacts, diamonds); bars, 95% intervals.

data.csv: per cohort, condition and component, the median difference, its interval, P values and
the electrode and participant counts.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
F = pd.read_csv(HERE / "data.csv")
COMP = {"between": 0.0, "within": 1.0, "within - between": 2.0}
LAB = {"between": "between", "within": "within", "within - between": "within −\nbetween"}

fig, d = S.panel(76, 60, box=(15, 8, 52, 38))
d.axhspan(-0.092, 0, color=S.DENSE_FIELD, lw=0, zorder=0)
d.axvspan(2.5, 2.9, color="white", lw=0, zorder=0.1)
d.axhline(0, color=S.INK2, lw=0.6, zorder=1)
xs, labels = [], []
for cohort, x0, mk in (("podcast", 0.0, "o"), ("film", 3.3, "D")):
    sub_ = F[F.cohort == cohort]
    for comp, dx in COMP.items():
        col = S.INK if comp == "within - between" else S.SPARSE
        conds = (("without", -0.14, True), ("with ", 0.14, False)) if cohort == "podcast" else ((None, 0.0, True),)
        for cond_, off, filled in conds:
            q = sub_[sub_.component == comp]
            r = (q if cond_ is None else q[q.condition.str.startswith(cond_)]).iloc[0]
            x = x0 + dx + off
            S.interval(d, x, r.ci95_lo, r.ci95_hi, col, lw=0.9)
            d.plot([x], [r["median"]], mk, color=col, mfc=col if filled else "white", mec=col, mew=0.9,
                   ms=3.8 if mk == "o" else 3.4, zorder=3)
            if comp == "within - between":
                side = 0 if cohort == "film" else (1 if filled else -1)
                d.text(x + 0.06 * side, r.ci95_lo - 0.003, S.num(r["median"], "+.4f"), fontsize=S.T_SMALL,
                       color=S.INK, ha={0: "center", 1: "right", -1: "left"}[side], va="top")
        xs.append(x0 + dx)
        labels.append(LAB[comp])
for xc, lab in ((1.0, "Podcast · ECoG"), (4.3, "Film · sEEG")):
    d.text(xc, 1.03, lab, transform=d.get_xaxis_transform(), ha="center", va="bottom", fontsize=S.T_SMALL,
           color=S.INK, fontweight="bold")
d.set_xlim(-0.5, 5.75)
d.set_ylim(-0.092, 0.075)
d.set_yticks([-0.08, -0.04, 0, 0.04])
d.set_yticklabels(["−0.08", "−0.04", "0", "0.04"])
d.set_xticks(xs)
d.set_xticklabels(labels, fontsize=S.T_SMALL, linespacing=1.0)
d.tick_params(axis="x", length=0)
d.set_ylabel("r difference", labelpad=1.5)
S.despine(d, bottom=False)
d.text(-0.42, -0.090, "dense better", fontsize=S.T_SMALL, color=S.DENSE, ha="left", va="bottom")
S.note(fig, "Gemma-2-2B IFG, sparse − dense, 300 components", 2, 1.0, size=S.T_SMALL)
S.note(fig, "podcast, training-only fits: filled, without and open, with timing and", 2, 54.0, size=S.T_SMALL)
S.note(fig, "neighbouring-word controls", 2, 56.6, size=S.T_SMALL)
S.save(fig, __file__)
