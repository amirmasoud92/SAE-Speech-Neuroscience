"""Figure 10b: film-cohort H2. The fraction of admissible contacts at which 20 features capture
more than a share tau of the full dictionary's held-out variance, against tau, for the learned
dictionary (blue), a density-matched random dictionary and a column-shuffled control (greys);
solid, language network; dashed, IFG.

data.csv: per scope, arm and tau, the fraction of contacts above tau.
statistics.csv: per scope, the test at tau = 0.5 (percentages, participants favouring the learned
dictionary, one-sided sign-test P) and the contact counts.
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
H = pd.read_csv(HERE / "statistics.csv").set_index("scope")
hl = H.loc["language"]
ARMS = (("real", S.SPARSE, S.LW_DATA, "learned"), ("random_dict", S.CTRL, 0.9, "random dictionary"),
        ("shuffled", S.CTRL_L, 0.9, "column-shuffled"))

fig, b = S.panel(60, 64, box=(13, 8, 40, 35))
S.note(fig, f"at τ = 0.5: {hl.pct_real:.1f}% vs {hl.pct_random:.1f}%, {int(hl.n_real_higher)} of "
            f"{int(hl.n_untied)}, {pfmt(hl.p_sign_one_sided)}", 2, 1.0, size=S.T_SMALL)
b.axvspan(0.48, 0.52, color=S.FIELD, lw=0, zorder=0)
b.text(0.5, 0.84, "τ = 0.5", fontsize=S.T_SMALL, color=S.INK2, ha="center", va="bottom")
ends = {}
for scope, ls in (("language", "-"), ("IFG", (0, (3, 2)))):
    for arm, col, lw, _ in ARMS:
        g = D[(D.scope == scope) & (D.arm == arm)].sort_values("tau")
        b.plot(g.tau, g.fraction, ls=ls, color=col, lw=lw)
        ends[(scope, arm)] = float(g.fraction.iloc[-1])
b.text(0.92, ends[("language", "real")], " language", color=S.INK2, fontsize=S.T_SMALL, va="center")
b.text(0.92, ends[("IFG", "real")], " IFG", color=S.INK2, fontsize=S.T_SMALL, va="center")
b.set_xlim(-0.02, 1.13)
b.set_xticks([0, 0.25, 0.5, 0.75])
b.set_xlabel("share threshold τ", labelpad=1)
b.set_ylabel("fraction of contacts above τ", labelpad=1.5)
b.set_ylim(-0.02, 0.85)
S.despine(b)
for k, (_, col, lw, lab) in enumerate(ARMS):
    y = 53.0 + 2.9 * k
    ax_k = S.ax_mm(fig, 2.0, y - 0.6, 4.4, 1.2)
    ax_k.plot([0, 1], [0.5, 0.5], color=col, lw=lw)
    ax_k.set_xlim(0, 1)
    ax_k.axis("off")
    S.note(fig, lab, 7.4, y, va="center", size=S.T_SMALL, color=S.SPARSE if k == 0 else S.INK2)
S.note(fig, f"solid, language ({int(hl.n_electrodes_conf)});\ndashed, IFG ({int(H.loc['IFG', 'n_electrodes_conf'])})",
       32.0, 51.8, size=S.T_SMALL, linespacing=1.15)
S.save(fig, __file__)
