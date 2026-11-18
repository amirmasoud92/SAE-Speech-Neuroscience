"""Figure 7a (right): the pre-specified test. The ranks of causal necessity (KL) and of neural
importance, each with the ranks of activation density (and a constant) regressed out; their
correlation is the partial Spearman rho, below the 0.15 floor. Line, least-squares fit.

data.csv: per feature, activation density, coherence, neural importance and the causal scores.
statistics.csv: per causal score, the raw and density-partialled Spearman rho and their P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def resid_ranks(x, z):
    """x's ranks with z's ranks (and a constant) regressed out."""
    xr = sps.rankdata(x)
    Z = np.column_stack([sps.rankdata(z), np.ones(len(z))])
    q, _ = np.linalg.qr(Z)
    return xr - q @ (q.T @ xr)


S.apply()
D = pd.read_csv(HERE / "data.csv")
st = pd.read_csv(HERE / "statistics.csv")
r = st[st.score.str.contains("KL")].iloc[0]
rk = resid_ranks(D.causal_kl.to_numpy(float), D.log_density.to_numpy(float))
ri = resid_ranks(D.importance.to_numpy(float), D.log_density.to_numpy(float))
assert abs(float(np.corrcoef(rk, ri)[0, 1]) - r.primary_partial_rho) < 1e-9

fig, ax = S.panel(56, 70, box=(13, 22, 38, 38))
S.note(fig, "density partialled (pre-specified)", 13, 9.2, color=S.INK, size=S.T_SMALL, fontweight="bold")
S.note(fig, f"partial ρ = {S.num(r.primary_partial_rho, '+.3f')}, {S.p_str(r.primary_p, 3)}", 13, 12.4,
       color=S.INK, size=S.T_SMALL)
S.note(fig, "below the 0.15 floor", 13, 15.6, size=S.T_SMALL)
ax.scatter(rk, ri, s=6.0, color=S.SPARSE_L, lw=0.2, edgecolors="white", zorder=3)
f2 = np.polyfit(rk, ri, 1)
xx = np.array([rk.min(), rk.max()])
ax.plot(xx, np.polyval(f2, xx), color=S.INK, lw=0.9, zorder=4)
S.ref(ax, 0)
S.ref(ax, 0, horizontal=False)
ax.set_xlabel("KL rank | density", labelpad=1)
ax.set_ylabel("importance rank | density", labelpad=0.5)
ax.set_xticks([-100, 0, 100])
ax.set_yticks([-100, 0, 100])
ax.xaxis.set_major_formatter(S.minus_formatter("{:g}"))
ax.yaxis.set_major_formatter(S.minus_formatter("{:g}"))
ax.set_xlim(-195, 195)
ax.set_ylim(-195, 195)
S.despine(ax)
S.save(fig, __file__)
