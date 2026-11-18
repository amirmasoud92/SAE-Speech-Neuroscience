"""Supplementary Figure 5: coherence advantage over principal components across five dictionaries.
Sparse directions (of 300) more coherent than the median principal component, in five
independently trained GemmaScope dictionaries at layer 12, grouped by width and labelled by nominal
L0 (filled, the dictionary used throughout); horizontal line, the 4 of 300 principal components
above the sparse median.

data.csv: per dictionary, its plotting position, the count of such directions, live features and
the median coherence of its sparse directions and of the principal components.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
PCS_ABOVE_SPARSE_MEDIAN = 4                         # stated in the paper; not in data.csv

fig, ax = S.panel(88, 62, box=(16, 8, 66, 40))
S.note(fig, "sparse directions above the median principal component", 2, 1.0, size=S.T_SMALL)
s16 = D[D.dictionary.str.startswith("w16k")].sort_values("x")
ax.plot(s16.x, s16.usable_directions, color=S.SPARSE_L, lw=0.8, zorder=1)
for _, r in D.iterrows():
    main = r.dictionary == "w16k_l0-82"
    ax.plot([r.x, r.x], [0, r.usable_directions], color=S.CTRL_XL, lw=0.6, zorder=0)
    ax.plot([r.x], [r.usable_directions], "o", ms=5.0, mfc=S.SPARSE if main else "white", mec=S.SPARSE, mew=0.9,
            zorder=3)
    ax.text(r.x, r.usable_directions + 9, f"{int(r.usable_directions)}", fontsize=S.T_SMALL, ha="center",
            va="bottom", color=S.INK if main else S.INK2, fontweight="bold" if main else "normal")
ax.axhline(PCS_ABOVE_SPARSE_MEDIAN, color=S.DENSE, lw=1.0, zorder=2)
ax.text(4.75, 14, f"principal components above the\nsparse median: {PCS_ABOVE_SPARSE_MEDIAN}", fontsize=S.T_SMALL,
        color=S.DENSE, ha="right", va="bottom", linespacing=1.05)
ax.set_xticks(D.x.tolist())
ax.set_xticklabels([d.split("-")[1] for d in D.dictionary])
ax.tick_params(axis="x", length=0)
tr = ax.get_xaxis_transform()
for xc, lab in ((1.0, "16k"), (3.35, "32k"), (4.35, "65k")):
    ax.text(xc, -0.20, lab, transform=tr, ha="center", va="top", fontsize=S.T_SMALL, color=S.INK2)
ax.plot([-0.25, 2.25], [-0.17, -0.17], transform=tr, color=S.INK2, lw=0.45, clip_on=False)
ax.text(-0.6, -0.025, "L0", transform=tr, ha="right", va="top", fontsize=S.T_SMALL, color=S.INK2)
ax.text(-0.6, -0.20, "width", transform=tr, ha="right", va="top", fontsize=S.T_SMALL, color=S.INK2)
ax.set_xlim(-0.5, 4.8)
ax.set_ylim(-22, 330)
ax.set_yticks([0, 100, 200, 300])
ax.set_ylabel("directions (of 300)", labelpad=2)
S.despine(ax, bottom=False)
S.save(fig, __file__)
