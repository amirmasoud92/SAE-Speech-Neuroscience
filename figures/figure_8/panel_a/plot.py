"""Figure 8a: every layer of Gemma-2-2B. Top, out-of-fold correlation at the 135 IFG electrodes
of the dense residual stream and of its sparse decomposition (GemmaScope, width 16,384, L0
closest to 100), each at 300 principal components; bottom, their difference with 95%
participant-clustered bootstrap intervals (filled, Holm-corrected P < 0.05 over layers).

data.csv: per layer, the two mean r values, the difference, its interval and Holm-corrected P.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").sort_values("layer")
L = D.layer.to_numpy()
sig = (D.d_sae_dense_p_holm < 0.05).to_numpy()
n_sparse = int(((D.d_sae_dense > 0) & (D.d_sae_dense_p_holm < 0.05)).sum())

fig = S.figure(116, 62)
S.note(fig, f"300 components each, IFG; Holm-corrected: dense ahead at {int(sig.sum())} of {len(sig)} "
            f"layers, sparse at {n_sparse}", 2, 1.0, size=S.T_SMALL)
a1 = S.ax_mm(fig, 15, 6.0, 88, 25)
a2 = S.ax_mm(fig, 15, 34.0, 88, 17)
for ax_ in (a1, a2):
    ax_.axvspan(11.5, 12.5, color=S.FIELD, lw=0, zorder=0)
a1.text(12.7, 0.0615, "primary layer", fontsize=S.T_SMALL, color=S.INK2, ha="left", va="bottom")
a1.fill_between(L, D.sae300_r, D.dense_r, where=D.dense_r >= D.sae300_r, color=S.DENSE_FIELD, lw=0, zorder=1)
a1.plot(L, D.dense_r, "-", color=S.DENSE, lw=S.LW_DATA, zorder=3)
a1.plot(L, D.sae300_r, "-", color=S.SPARSE, lw=S.LW_DATA, zorder=3)
a1.text(25.6, float(D.dense_r.iloc[-1]) + 0.003, "dense", color=S.DENSE, fontsize=S.T_SMALL, va="center",
        fontweight="bold")
a1.text(25.6, float(D.sae300_r.iloc[-1]) - 0.004, "sparse", color=S.SPARSE, fontsize=S.T_SMALL, va="center",
        fontweight="bold")
a1.set_ylabel("IFG r", labelpad=1.5)
a1.set_xlim(-0.5, 25.5)
a1.set_ylim(0.06, 0.15)
a1.set_yticks([0.06, 0.09, 0.12, 0.15])
a1.set_xticks([0, 5, 10, 15, 20, 25])
a1.tick_params(labelbottom=False)
S.despine(a1)
a2.axhline(0, color=S.INK2, lw=0.6)
a2.fill_between(L, D.d_sae_dense_lo, D.d_sae_dense_hi, color=S.SPARSE, alpha=0.15, lw=0)
a2.plot(L, D.d_sae_dense, "-", color=S.SPARSE, lw=0.9)
a2.plot(L[sig], D.d_sae_dense[sig], "o", color=S.SPARSE, ms=2.9, mec="white", mew=0.4)
a2.plot(L[~sig], D.d_sae_dense[~sig], "o", color="white", mec=S.SPARSE, ms=2.9, mew=0.7)
a2.set_ylabel("sparse − dense", labelpad=1.5)
a2.set_xlabel("layer", labelpad=1)
a2.set_xlim(-0.5, 25.5)
a2.set_ylim(-0.032, 0.010)
a2.set_xticks([0, 5, 10, 15, 20, 25])
a2.set_yticks([-0.03, -0.02, -0.01, 0])
a2.set_yticklabels(["−0.03", "−0.02", "−0.01", "0"])
S.despine(a2)
S.rich_text(a2, 25.4, -0.031, "filled, Holm $P$ < 0.05", a2.transData, ha="right", va="bottom",
            fontsize=S.T_SMALL, color=S.INK2)
S.save(fig, __file__)
