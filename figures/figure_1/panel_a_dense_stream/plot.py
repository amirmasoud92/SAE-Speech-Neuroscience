"""Figure 1a (dense stream): ten words in the 2,304-dimensional Gemma-2-2B layer-12 residual
stream, shown as absolute values; the boxed word is the one marked in the spectrogram.

data.csv: one row per word (in transcript order), one column per residual dimension.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
R = D.filter(like="dim_").to_numpy(float)
labels = D.word.astype(str).tolist()
tgt = int(np.flatnonzero(D.is_target.to_numpy())[0])
n = len(D)

fig = S.figure(40, 31)
ax = S.ax_mm(fig, 15, 3, 22, 25)
cm = LinearSegmentedColormap.from_list("dn", ["#FFFFFF", S.DENSE_XL, S.DENSE_L, S.DENSE, S.DENSE_D])
M = np.abs(R)
ax.imshow(np.clip(M / np.percentile(M, 90), 0, 1) ** 0.8, aspect="auto", cmap=cm,
          interpolation="antialiased", rasterized=True, extent=[0, R.shape[1], n, 0])
ax.set_yticks(np.arange(n) + 0.5)
ax.set_yticklabels(labels, fontsize=S.T_SMALL)
ax.get_yticklabels()[tgt].set_fontweight("bold")
ax.tick_params(axis="y", length=0, pad=1.5)
ax.set_xticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
for i in range(1, n):
    ax.axhline(i, color="white", lw=0.45, zorder=3)
ax.add_patch(plt.Rectangle((0, tgt), R.shape[1], 1, fc="none", ec=S.INK, lw=0.6, clip_on=False,
                           zorder=4))
S.save(fig, __file__)
