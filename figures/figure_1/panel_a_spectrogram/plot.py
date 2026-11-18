"""Figure 1a (speech): the stimulus mel spectrogram around one word (boxed) and the word timeline.

data.csv: log-mel power, one row per mel bin, one column per 10-ms frame (time in s from the
boxed word's onset). words.csv: the words overlapping the window, with onset and offset in s.
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
W = pd.read_csv(HERE / "words.csv")
spec = D.drop(columns="mel_bin").to_numpy(float)
t = np.array([float(c) for c in D.columns[1:]])
pre, span = -t[0], t[-1] - t[0] + 0.01
target = W[W.is_target].iloc[0]

fig = S.figure(48, 34)
ax = S.ax_mm(fig, 9, 4, 36, 21)
cm = LinearSegmentedColormap.from_list("ac", ["#FFFFFF", "#D3DFD2", S.ACOUSTIC_L, S.BAND_ACOUSTIC,
                                              "#24382A"])
lo, hi = np.percentile(spec, [30, 99.5])
ax.imshow(np.clip(spec, lo, hi), aspect="auto", origin="lower", cmap=cm,
          extent=[-pre, span - pre, 0, 80], interpolation="bilinear", rasterized=True)
ax.add_patch(plt.Rectangle((0, 0), target.end_s, 80, fc="none", ec=S.INK, lw=0.6, zorder=4))
ax.text(target.end_s / 2, 81.5, target.word, ha="center", va="bottom", fontsize=S.T_SMALL,
        fontstyle="italic", clip_on=False)
ax.set_xlim(-pre, span - pre)
ax.set_ylim(0, 80)
ax.set_xticks([])
ax.set_yticks([0, 80])
ax.set_ylabel("mel band", labelpad=0.5, fontsize=S.T_SMALL)
ax.tick_params(labelsize=S.T_SMALL)
S.despine(ax, bottom=False, offset_mm=0.6)

tx = S.ax_mm(fig, 9, 25.8, 36, 3.0)
for k, w in W.reset_index(drop=True).iterrows():
    a, b = max(w.start_s, -pre), min(w.end_s, span - pre)
    tx.add_patch(plt.Rectangle((a, 0), b - a, 1, ec="white", lw=0.4,
                               fc=S.INK if w.is_target else ("#E3E0DA" if k % 2 else "#EEECE8")))
tx.set_xlim(-pre, span - pre)
tx.set_ylim(0, 1)
tx.axis("off")
S.note(fig, "words", 8.2, 27.3, ha="right", va="center", size=S.T_SMALL)
S.note(fig, "0", 9 + 36 * pre / span, 29.4, ha="center", size=S.T_SMALL)
S.note(fig, "1 s", 9 + 36 * (pre + 1.0) / span, 29.4, ha="center", size=S.T_SMALL)
S.save(fig, __file__)
