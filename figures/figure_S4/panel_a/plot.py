"""Supplementary Figure 4 (a, raw Spearman correlations; b, partialled on log activation density
and coherence; the same script draws each panel from its folder name). Correlation over the 300
pooled features between each score (neural importance over all 135 IFG electrodes; causal KL;
causal change in log-probability) and the lexical profile of the feature's top-20 activating words;
filled bars, uncorrected P < 0.05.

data.csv: per score and property, rho and P (b: also the FDR q and whether it survives).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
T = pd.read_csv(HERE / "data.csv")
RAW = HERE.name.endswith("_a")
vcol, pcol = ("raw_rho", "raw_p") if RAW else ("partial_rho", "p")
PROPS = ["is_function", "concreteness", "zipf", "n_char"]
PLAB = ["function word", "concreteness", "log frequency", "word length"]
SCORES = [("neural importance", "neural importance (cortex)", S.SPARSE),
          ("causal necessity (KL)", "causal KL (model)", S.INK),
          ("causal necessity (delta log-prob)", "Δ log-probability (model)", "#8C8C8C")]
if RAW:
    sub = f"{int((T[pcol] < 0.05).sum())} of {len(T)} coefficients at $P$ < 0.05 (filled)"
else:
    sub = (f"{int(T.survives_fdr.sum())} of {len(T)} survive false-discovery correction;\n"
           "filled, uncorrected $P$ < 0.05")

fig, ax = S.panel(92, 78, box=(15, 9, 74, 50))
S.note(fig, sub, 2, 1.0, size=S.T_SMALL)
xs, w = np.arange(len(PROPS)), 0.26
for si, (sc, _, c) in enumerate(SCORES):
    q = T[T.score == sc].set_index("property").loc[PROPS]
    for xb, yb, pb in zip(xs + (si - 1) * w, q[vcol], q[pcol]):
        ax.bar(xb, yb, w * 0.86, color=c if pb < 0.05 else "white", edgecolor=c, lw=0.8, zorder=3)
ax.axhline(0, color=S.INK2, lw=0.6, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels(PLAB)
ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.42, 0.40)
ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_yticklabels(["−0.4", "−0.2", "0", "0.2", "0.4"])
ax.set_ylabel("Spearman ρ with the property", labelpad=1.5)
S.despine(ax, bottom=False)
kax = S.ax_mm(fig, 15, 66.5, 74, 9)
kax.set_xlim(0, 74)
kax.set_ylim(9, 0)
kax.axis("off")
for k, (_, lab, c) in enumerate(SCORES):
    kax.add_patch(plt.Rectangle((0, k * 3 + 0.4), 2.6, 1.8, color=c, lw=0))
    kax.text(3.6, k * 3 + 1.3, lab, fontsize=S.T_SMALL, color=S.INK, va="center")
S.save(fig, __file__)
