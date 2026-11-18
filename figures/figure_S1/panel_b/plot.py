"""Supplementary Figure 1 (a, achieved against nominal L0; b, fraction of variance unexplained
against achieved L0; c, features active on at least 25 of 5,130 words against achieved L0). The
same script draws each panel from its folder name. Five released GemmaScope dictionaries at
Gemma-2-2B layer 12: circles, 16,384 features (joined in order of L0); square, 32,768; triangle,
65,536; filled, the dictionary used throughout.

data.csv: per dictionary, its width, nominal L0, achieved L0 per token, FVU and live-feature count.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
P = HERE.name.split("_")[-1]
XC, YC, XL, YL = {"a": ("nominal_l0", "token_l0_mean", "nominal L0", "achieved L0 (per token)"),
                  "b": ("token_l0_mean", "fvu", "achieved L0", "fraction of variance unexplained"),
                  "c": ("token_l0_mean", "n_features_ge25_words", "achieved L0", "features active on ≥ 25 words")}[P]
MK = {"16k": "o", "32k": "s", "65k": "^"}

fig, ax = S.panel(60, 72, box=(14, 20, 42, 40))
if P == "a":
    ax.plot([0, 215], [0, 215], color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=0)
    ax.text(150, 140, "identity", rotation=45, fontsize=S.T_SMALL, color=S.INK2, ha="center", va="top",
            rotation_mode="anchor")
f16 = D[D.width == "16k"].sort_values(XC)
ax.plot(f16[XC], f16[YC], color=S.SPARSE_L, lw=0.8, zorder=1)
for _, r in D.iterrows():
    ax.plot([r[XC]], [r[YC]], MK[r.width], ms=4.8 if r.primary else 4.2, mfc=S.SPARSE if r.primary else "white",
            mec=S.SPARSE, mew=0.8, zorder=3)
ax.set_xlabel(XL, labelpad=1)
ax.set_ylabel(YL, labelpad=1.5)
ax.margins(0.1)
if P == "a":
    ax.set_xlim(0, 215)
    ax.set_ylim(0, 230)
S.despine(ax)
for k, (m_, filled, lab) in enumerate((("o", False, "16,384 features (L0 22, 176)"),
                                       ("o", True, "16,384, L0 82 (used throughout)"),
                                       ("s", False, "32,768, L0 76"), ("^", False, "65,536, L0 72"))):
    x, y = 2.0, 2.0 + 3.4 * k
    kax = S.ax_mm(fig, x, y - 1.5, 3.0, 3.0)
    kax.set_xlim(0, 1)
    kax.set_ylim(0, 1)
    kax.axis("off")
    kax.plot([0.5], [0.5], m_, ms=3.8, mfc=S.SPARSE if filled else "white", mec=S.SPARSE, mew=0.8)
    S.note(fig, lab, x + 3.4, y, va="center", size=S.T_SMALL)
S.save(fig, __file__)
