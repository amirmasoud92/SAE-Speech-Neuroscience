"""Figure 10f: coherence and importance. Feature coherence against cross-cohort importance (the
mean of each feature's podcast and film importance ranks), both as ranks with the two cohorts'
activation densities regressed out, in deciles of importance (means with 95% intervals,
1.96 s.e.m.).

data.csv: per decile, the mean importance rank, the mean coherence rank and its s.e.m.
statistics.csv: the feature-level partial Spearman rho, its null interval and two-sided P.
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
D = pd.read_csv(HERE / "data.csv")
T = pd.read_csv(HERE / "statistics.csv").iloc[0]

fig, g = S.panel(52, 62, box=(14, 9, 34, 38))
S.note(fig, "both cohorts, density-partialled ranks", 2, 1.0, size=S.T_SMALL)
S.note(fig, f"partial ρ = {S.num(T.partial_rho, '+.3f')}, {pfmt(T.p_two_sided)}, two-sided", 2, 4.0,
       size=S.T_SMALL, color=S.INK)
g.axhline(0, color=S.INK2, lw=0.6)
for r in D.itertuples():
    S.interval(g, r.importance, r.coherence - 1.96 * r.se, r.coherence + 1.96 * r.se, S.SPARSE_L, lw=0.9)
g.plot(D.importance, D.coherence, "o", color=S.SPARSE, ms=3.6, mec="white", mew=0.4, zorder=3)
g.set_xlabel("cross-cohort importance\n(rank | density, deciles)", labelpad=1, linespacing=1.0)
g.set_ylabel("coherence (rank | density)", labelpad=1.5)
g.set_xlim(-2500, 2500)
g.set_xticks([-2000, 0, 2000])
g.set_xticklabels(["−2,000", "0", "2,000"])
g.set_yticks([-200, 0, 200, 400])
g.set_yticklabels(["−200", "0", "200", "400"])
S.despine(g)
S.save(fig, __file__)
