"""Figure 1a (sparse code): the same ten words in the 16,384-feature sparse code; each tick is an
active feature (opacity scaled by the word's maximum activation) and the number active is given
per word.

data.csv: one row per active (word, feature) pair with its activation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
N_FEATURES = 16384
n = int(D.word_row.max()) + 1
tgt = int(D[D.is_target].word_row.iloc[0])

fig = S.figure(30, 31)
ax = S.ax_mm(fig, 1, 3, 22, 25)
segs, cols = [], []
for i, g in D.groupby("word_row"):
    v = g.activation.to_numpy() / g.activation.max()
    for j, a in zip(g.feature.to_numpy(), v):
        segs.append([(j, i + 0.12), (j, i + 0.88)])
        cols.append((*to_rgb(S.SPARSE), 0.30 + 0.70 * a))
    ax.text(N_FEATURES * 1.025, i + 0.5, f"{len(g)}", va="center", ha="left", fontsize=S.T_SMALL,
            color=S.INK if i == tgt else S.INK2, fontweight="bold" if i == tgt else "normal")
ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4))
ax.set_xlim(0, N_FEATURES)
ax.set_ylim(n, 0)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
for i in range(n + 1):
    ax.axhline(i, color=S.CTRL_XL, lw=0.3, zorder=0)
ax.add_patch(plt.Rectangle((0, tgt), N_FEATURES, 1, fc="none", ec=S.INK, lw=0.6, clip_on=False,
                           zorder=4))
ax.text(N_FEATURES * 1.025, -0.25, "active", fontsize=S.T_SMALL - 0.3, color=S.INK2, ha="left",
        va="bottom")
S.save(fig, __file__)
