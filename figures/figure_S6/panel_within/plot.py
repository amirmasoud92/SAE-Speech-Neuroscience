"""Supplementary Figure 6 (left, between types; right, within types; the same script draws each half
from its folder name). Out-of-fold predictions at the 135 IFG electrodes of the sparse code and
GPT-2 XL layer 24, each reduced to 300 principal components, decomposed into between-type and
within-type components over word types occurring at least four times (within-type components use
within-fold type means). Points and half-violins, electrodes; bars, medians. Beneath, sparse minus
GPT-2 XL with its participant-clustered bootstrap interval (10,000 resamples) and P.

data.csv: per representation and electrode, the component's out-of-fold r.
statistics.csv: the sparse-minus-GPT-2 XL estimate, interval, P and counts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def p_str(p):
    return "< 0.0001" if p < 1e-4 else f"{float(f'{p:.2g}'):g}"


S.apply()
COMP = HERE.name.split("_")[-1]
E = pd.read_csv(HERE / "data.csv")
r = pd.read_csv(HERE / "statistics.csv").iloc[0]
title, ylim, yt = {"between": ("between types", (0, 0.75), [0, 0.25, 0.5, 0.75]),
                   "within": ("within types", (-0.1, 0.35), [-0.1, 0, 0.1, 0.2, 0.3])}[COMP]
REPS = [("sparse", "sparse\ncode", S.SPARSE), ("GPT-2 XL", "GPT-2 XL\nL24", S.PRIOR)]

fig, ax = S.panel(44, 66, box=(12, 6, 30, 36))
ax.axhline(0, color=S.RULE, lw=S.LW_REF, zorder=0)
rng = np.random.default_rng(0)
for i, (key, lab, c) in enumerate(REPS):
    v = E[E.representation == key].r.to_numpy(float)
    v = v[np.isfinite(v)]
    S.half_violin(ax, i + 0.04, v, c, width=0.36, alpha=0.30)
    ax.scatter(i - 0.16 + rng.uniform(-0.09, 0.09, len(v)), v, s=2.0, color=c, alpha=0.55, lw=0, zorder=2)
    md = float(np.median(v))
    ax.plot([i - 0.28, i + 0.04], [md, md], color=S.INK, lw=1.2, zorder=4)
    ax.text(i, -0.04, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=S.T_SMALL,
            color=S.INK, linespacing=1.05)
    ax.text(i, -0.20, S.num(md, "+.3f"), transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=S.T_SMALL, color=S.INK2)
ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=S.T_ANNOT,
        fontweight="bold", color=S.INK)
S.note(fig, f"sparse − GPT-2 XL {S.num(r.estimate, '+.4f')}", 22, 54.0, ha="center", size=S.T_SMALL)
S.note(fig, f"[{S.num(r.ci95_lo, '+.4f')}, {S.num(r.ci95_hi, '+.4f')}], $P$ = {p_str(r.p_two_sided)}", 22, 57.2,
       ha="center", size=S.T_SMALL)
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(*ylim)
ax.set_yticks(yt)
ax.set_yticklabels([S.num(y, "g") if y else "0" for y in yt])
ax.set_ylabel("out-of-fold r", labelpad=2)
S.despine(ax, bottom=False)
ax.spines["bottom"].set_visible(False)
S.save(fig, __file__)
