"""Figure 2b: the sparse-versus-classical contrast as unique variance inside the joint banded model,
at three lag conventions; points, medians across the IFG electrodes (grey, classical band; blue,
sparse band), with the difference, its participant-clustered P and the number of participants
favouring the sparse band beneath each.

data.csv: per lag convention, the two medians, their difference, P and the participant count.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").set_index("convention")
CONV = ["at the SAE peak", "own peak", "mean over 0-500 ms"]
LAB = {"at the SAE peak": "at the sparse band's peak", "own peak": "at each band's peak",
       "mean over 0-500 ms": "mean over 0–500 ms"}

fig, ax = S.panel(100, 36, box=(34, 5, 62, 21))
for i, cv in enumerate(CONV):
    r = D.loc[cv]
    ax.plot([r.median_classical, r.median_sae], [i, i], color=S.SPARSE_XL, lw=3.2, zorder=2,
            solid_capstyle="round")
    S.dot(ax, r.median_classical, i, S.CLASSICAL, size=4.0, edge="white", mew=0.5)
    S.dot(ax, r.median_sae, i, S.SPARSE, size=4.0, edge="white", mew=0.5)
    k, n = r.subjects_favouring_sae.split("/")
    ax.text(r.median_classical, i + 0.40, f"Δ {S.num(r.median_diff)}, $P$ = {r.p_cluster_boot:.4f}, "
            f"{k} of {n} participants", fontsize=S.T_SMALL, color=S.INK2, va="center", ha="left")
    ax.text(-0.02, i, LAB[cv], transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=S.T_SMALL)
ax.set_xlim(-0.0004, 0.0112)
ax.set_ylim(len(CONV) - 0.3, -0.62)
ax.set_yticks([])
ax.axvline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.set_xticks([0, 0.005, 0.010])
ax.set_xticklabels(["0", "0.005", "0.010"])
ax.set_xlabel("IFG median unique ΔR²", labelpad=1)
S.despine(ax, left=False)
S.note(fig, "classical", 66, 1.5, color=S.CLASSICAL, size=S.T_SMALL)
S.note(fig, "sparse", 82, 1.5, color=S.SPARSE, size=S.T_SMALL)
S.save(fig, __file__)
