"""Figure 3d: the H2 test. Percentage of admissible language-network electrodes at which 20
features chosen on training folds capture more than 50% of the full dictionary's held-out
variance, for the learned dictionary, a random dictionary matched in per-feature activation
density and a column-shuffled control. Lines join each participant's three values (markers sized
by electrode count; open, the two exploratory participants); black bars, pooled over the
confirmatory participants.

data.csv: per participant and arm, the electrode count and the percentage above threshold.
test.csv: the learned-minus-random difference, its interval and the one-sided sign-test P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
T = pd.read_csv(HERE / "test.csv").iloc[0]
ARMS = [("share_real", "learned", S.SPARSE), ("share_random_dict", "random\ndictionary", S.CTRL),
        ("share_shuffled", "column-\nshuffled", S.CTRL_L)]
xs = np.array([0.0, 1.0, 2.0])

conf = D[~D.exploratory.astype(bool)]
n_conf = int(conf.drop_duplicates("subject").n.sum())
# pooled = electrode-weighted mean over the confirmatory participants
pooled = [float(np.average(conf[conf.arm == a].pct_above_tau, weights=conf[conf.arm == a].n))
          for a, *_ in ARMS]

fig, ax = S.panel(88, 54, box=(11, 3, 46, 37))
for sub, g in D.groupby("subject"):
    g = g.set_index("arm").loc[[a for a, *_ in ARMS]]
    expl, n = bool(g.exploratory.iloc[0]), float(g.n.iloc[0])
    ax.plot(xs, g.pct_above_tau, color=S.CTRL_L, lw=0.6, zorder=1, ls=(0, (2, 1.5)) if expl else "-")
    for x_, v, (_, _, c) in zip(xs, g.pct_above_tau, ARMS):
        ax.scatter([x_], [v], s=5 + 1.5 * n, facecolor="white" if expl else c,
                   edgecolor=c if c != S.CTRL_L else S.CTRL, lw=0.6, zorder=3)
for x_, v in zip(xs, pooled):
    ax.plot([x_ - 0.27, x_ + 0.27], [v, v], color=S.INK, lw=1.5, zorder=4, solid_capstyle="butt")
    ax.text(x_ + 0.31, v, f"{v:.1f}%", fontsize=S.T_ANNOT, color=S.INK, va="center", ha="left",
            fontweight="bold", bbox=S.TEXT_BG, zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=S.T_SMALL, linespacing=1.0)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.4, 2.75)
ax.set_ylim(-4, 82)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_ylabel("electrodes (%)", labelpad=1.5)
S.despine(ax, bottom=False)

kx = 60.0
S.note(fig, "learned − random", kx, 4.0, color=S.INK, size=S.T_SMALL, fontweight="bold")
S.note(fig, f"{S.num(T['diff'], '+.3f')} [{S.num(T.diff_ci_lo, '+.3f')}, {S.num(T.diff_ci_hi, '+.3f')}]",
       kx, 7.2, color=S.INK, size=S.T_SMALL)
S.note(fig, f"one-sided sign test\n$P$ = {T.p_sign_one_sided:.4f}", kx, 10.6, size=S.T_SMALL,
       linespacing=1.15)
S.note(fig, f"bars: pooled over {conf.subject.nunique()}\nconfirmatory participants,\n"
            f"n = {n_conf} electrodes", kx, 18.2, size=S.T_SMALL, linespacing=1.15)
kax = S.ax_mm(fig, kx, 29.0, 26, 9)
kax.set_xlim(0, 26)
kax.set_ylim(9, 0)
kax.axis("off")
kax.scatter([1.0], [1.5], s=12, facecolor=S.SPARSE, edgecolor=S.SPARSE, lw=0.6)
kax.text(2.6, 1.5, "participant;\nsize, electrodes", fontsize=S.T_SMALL, color=S.INK2, va="center",
         linespacing=1.0)
kax.scatter([1.0], [6.4], s=12, facecolor="white", edgecolor=S.SPARSE, lw=0.6)
kax.text(2.6, 6.4, "exploratory", fontsize=S.T_SMALL, color=S.INK2, va="center")
S.save(fig, __file__)
