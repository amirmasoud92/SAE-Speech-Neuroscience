"""Figure 10c: the same features in both cohorts. For every sparse feature, its IFG importance in
the podcast cohort against that in the film cohort, as ranks with both cohorts' activation
densities regressed out (binned; line, least-squares fit). The partial correlation and the 95th
percentile of its circular-shift null are printed beneath.

data.csv: per feature, its IFG importance and log activation density in each cohort.
statistics.csv: the partial Spearman rho, its null 95th percentile and P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def resid_rank(y, c1, c2):
    Z = np.column_stack([np.ones(len(y)), sps.rankdata(c1), sps.rankdata(c2)])
    r = sps.rankdata(y)
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def pfmt(p):
    return ("$P$ < 0.0001" if p < 1e-4 else f"$P$ = {p:.4f}" if p < 0.001 else
            f"$P$ = {p:.3f}" if p < 0.1 else f"$P$ = {p:.2f}")


S.apply()
F = pd.read_csv(HERE / "data.csv")
T = pd.read_csv(HERE / "statistics.csv").iloc[0]
rp = resid_rank(F.importance_podcast, F.log_density_podcast, F.log_density_film)
rf = resid_rank(F.importance_film, F.log_density_podcast, F.log_density_film)

fig, c = S.panel(50, 56, box=(6, 5, 38, 35))
S.note(fig, f"IFG; {int(T.n_features):,} features, density partialled", 2, 1.0, size=S.T_SMALL)
c.hexbin(rp, rf, gridsize=24, cmap=S.SPARSE_CMAP, mincnt=1, linewidths=0.15, edgecolors="white")
fit = np.polyfit(rp, rf, 1)
xx = np.array([rp.min(), rp.max()])
c.plot(xx, np.polyval(fit, xx), color=S.INK, lw=0.9)
c.set_xlabel("podcast importance rank", labelpad=1)
c.set_ylabel("film importance rank", labelpad=1.5)
c.set_xticks([])
c.set_yticks([])
for sp in c.spines.values():
    sp.set_visible(True)
    sp.set_color(S.RULE)
S.note(fig, f"partial ρ = {S.num(T.partial_rho, '+.3f')}, {pfmt(T.p_partial)}", 6, 45.0, color=S.INK,
       size=S.T_SMALL, fontweight="bold")
S.note(fig, f"shift null, 95th percentile {S.num(T.partial_null_95, '+.3f')}", 6, 48.0, size=S.T_SMALL)
S.save(fig, __file__)
