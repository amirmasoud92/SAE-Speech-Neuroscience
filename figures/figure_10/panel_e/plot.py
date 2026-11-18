"""Figure 10e: acoustic control in early auditory cortex (EAC). The change in EAC's peak unique
sparse-band variance when acoustic control is weakened (podcast) or strengthened (film), with
participant-clustered 95% intervals; arrows, the direction an acoustic explanation predicts.

data.csv: per comparison, the predicted direction, counts, median change, interval and P.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def pfmt(p):
    return ("$P$ < 0.0001" if p < 1e-4 else f"$P$ = {p:.4f}" if p < 0.001 else
            f"$P$ = {p:.3f}" if p < 0.1 else f"$P$ = {p:.2f}")


S.apply()
R = pd.read_csv(HERE / "data.csv")
LAB = {"podcast, weaker acoustic control": "podcast,\nweaker\ncontrol",
       "film, stronger acoustic control": "film,\nstronger\ncontrol"}

fig, e = S.panel(42, 60, box=(14, 6, 24, 38))
S.note(fig, "arrows, the acoustic\naccount's prediction", 2, 1.0, size=S.T_SMALL, linespacing=1.0)
e.axhline(0, color=S.INK2, lw=0.6)
for i, r in R.iterrows():
    S.interval(e, i, r.lo * 1e3, r.hi * 1e3, S.REGION["EAC"], lw=1.1)
    e.plot([i], [r.median_diff * 1e3], "o", color=S.REGION["EAC"], ms=4.0, mec="white", mew=0.5, zorder=3)
    sgn = 1 if r.predicted_direction == "rise" else -1
    e.annotate("", xy=(i - 0.32, 0.45 * sgn + 0.6), xytext=(i - 0.32, 0.6),
               arrowprops=dict(arrowstyle="-|>", color=S.INK2, lw=0.6, mutation_scale=6))
    e.text(i, 2.30, S.num(r.median_diff * 1e3, '+.2f'), ha="center", va="bottom", fontsize=S.T_SMALL, color=S.INK)
    S.rich_text(e, i, 2.12, pfmt(r.p), e.transData, ha="center", va="center", fontsize=S.T_SMALL, color=S.INK)
e.set_xticks(range(len(R)))
e.set_xticklabels([LAB[c] for c in R.comparison], fontsize=S.T_SMALL, linespacing=1.0)
e.tick_params(axis="x", length=0)
e.set_xlim(-0.6, len(R) - 0.4)
e.set_ylim(-0.4, 2.5)
e.set_yticks([0, 0.5, 1.0, 1.5])
e.set_ylabel("Δ peak unique ΔR² (×10⁻³)", labelpad=1.5)
S.despine(e, bottom=False)
S.save(fig, __file__)
