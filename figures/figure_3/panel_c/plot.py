"""Figure 3c: coherence (median and interquartile range) of matched-recipe TopK autoencoders
(16,384 latents, k = 92) trained on Gemma-2-2B and on a randomly initialised copy of the model,
at initialisation and after 25, 50 and 100 million tokens, with GemmaScope for reference; grey
band, chance. Brackets, the two pre-specified comparisons at 100 M (Cliff's delta).

data.csv: per dictionary and checkpoint, the median, quartiles and number of scored features.
comparisons.csv: Cliff's delta and the one-sided Mann-Whitney P of the two comparisons.
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").set_index("key")
C = pd.read_csv(HERE / "comparisons.csv")
d_rand, d_init = (float(C.cliffs_delta[C.comparison.str.contains(s)].iloc[0])
                  for s in ("random-model", "initialisation"))
TOKENS = [0, 25, 50, 100]
XG = -26.0                                           # GemmaScope, the reference column

fig, ax = S.panel(82, 50, box=(12, 3, 54, 36))
rs = D.loc["random_sets"]
ax.axhspan(rs.q25, rs.q75, color=S.FIELD, lw=0, zorder=0)
ax.axhline(rs["median"], color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=0.5)
ax.text(102, rs.q25 - 0.0015, "chance", fontsize=S.T_SMALL, color=S.INK2, ha="right", va="top")
g = D.loc["gemmascope"]
ax.plot([XG, XG], [g.q25, g.q75], color=S.SPARSE_D, lw=1.6, solid_capstyle="butt", zorder=3)
ax.plot([XG], [g["median"]], "D", ms=3.4, color=S.SPARSE_D, mec="white", mew=0.4, zorder=4)
ax.plot([XG + 4, 104], [g["median"]] * 2, color=S.SPARSE_D, lw=0.45, ls=(0, (1, 1.4)), zorder=0.6)
med = {}
for model, c in (("trained", S.SPARSE), ("random", S.INK2)):
    q = D.loc[[f"{model}_{t}M" for t in TOKENS]]
    med[model] = q["median"].to_numpy()
    ax.fill_between(TOKENS, q.q25, q.q75, color=c, alpha=0.13, lw=0, zorder=1)
    ax.plot(TOKENS, q["median"], color=c, lw=S.LW_DATA, zorder=3)
    ax.plot(TOKENS[1:], q["median"].iloc[1:], "o", ms=3.4, color=c, mec="white", mew=0.4, zorder=4)
    ax.plot([0], [q["median"].iloc[0]], "o", ms=3.4, mfc="white", mec=c, mew=0.8, zorder=4)
ax.text(62, med["trained"][2] + 0.006, "on the trained model", fontsize=S.T_SMALL, color=S.SPARSE,
        ha="center", va="bottom", fontweight="bold")
ax.text(62, med["random"][2] - 0.006, "on a random model", fontsize=S.T_SMALL, color=S.INK2,
        ha="center", va="top", fontweight="bold")
t100, t0, r100 = med["trained"][3], med["trained"][0], med["random"][3]
ax.plot([0, 107], [t0, t0], color=S.SPARSE_L, lw=0.5, ls=(0, (2, 1.5)), zorder=0.7, clip_on=False)
for xo, ylo, yhi, d_ in ((107, t0, t100, d_init), (120, r100, t100, d_rand)):
    ax.plot([xo - 2.0, xo, xo, xo - 2.0], [yhi, yhi, ylo, ylo], color=S.INK, lw=0.5, clip_on=False)
    ax.text(xo + 2.2, (ylo + yhi) / 2, f"δ = {d_:.2f}", rotation=90, fontsize=S.T_SMALL, color=S.INK,
            ha="left", va="center", clip_on=False)
ax.set_xlim(-34, 104)
ax.set_ylim(0.43, 0.585)
ax.set_xticks([XG] + TOKENS)
ax.set_xticklabels(["Gemma-\nScope", "init.", "25", "50", "100"], linespacing=1.0)
ax.get_xticklabels()[0].set_color(S.SPARSE_D)
ax.set_xlabel("training tokens (millions)", labelpad=1)
ax.set_ylabel("coherence", labelpad=1.5)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
S.despine(ax)
S.save(fig, __file__)
