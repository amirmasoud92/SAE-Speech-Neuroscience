"""Figure 8c (sparse minus dense, 300 components) and 6d (sparse minus the classical linguistic
band at its own width; the same script in ../panel_d). Median per-electrode r difference with 95%
participant-clustered intervals (10,000 draws) across dictionaries, model sizes, model families
and the film cohort; filled, each group's primary row; diamonds, the film cohort; the side of zero
where the sparse code loses is tinted.

data.csv: per row, its group, label, whether it is the group's primary row, the median difference
and its interval.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
R = pd.read_csv(HERE / "data.csv")
GROUP_NAMES = {"Podcast": "Podcast, primary", "Dictionary": "Another dictionary",
               "Model size": "Larger model", "Model family": "Another model family",
               "Cohort": "Film cohort (Brain Treebank)"}
SETUP = {"c": dict(lose=S.DENSE_FIELD, xlim=(-0.032, 0.008), ticks=[-0.03, -0.02, -0.01, 0],
                   lose_text="dense better", lose_color=S.DENSE, title="Sparse − dense, 300 components"),
         "d": dict(lose=S.FIELD, xlim=(-0.022, 0.09), ticks=[0, 0.02, 0.04, 0.06, 0.08],
                   lose_text="classical\nbetter", lose_color=S.INK2, title="Sparse − classical, matched width")}
cfg = SETUP[HERE.name.split("_")[-1]]

ys, heads, y, prev = [], [], 0.0, None                   # a header slot above every group
for g in R.group:
    if g != prev:
        y -= 0.35 if prev is not None else 0.0
        heads.append((g, y))
        y -= 1.0
        prev = g
    ys.append(y)
    y -= 1.0
ys = np.array(ys)

fig, ax = S.panel(102, 80, box=(46, 5, 52, 66))
S.note(fig, cfg["title"], 2, 1.0, size=S.T_SMALL, color=S.INK, fontweight="bold")
ax.axvspan(cfg["xlim"][0], 0, color=cfg["lose"], lw=0, zorder=0)
ax.axvline(0, color=S.INK2, lw=0.6, zorder=1)
tr = ax.get_yaxis_transform()
for g, yh in heads:
    ax.text(-0.04, yh, GROUP_NAMES[g], transform=tr, ha="right", va="center", fontsize=S.T_SMALL,
            color=S.INK, fontweight="bold")
    if g != heads[0][0]:
        ax.plot([-0.85, 1.0], [yh + 0.68] * 2, transform=tr, color=S.RULE, lw=0.4, clip_on=False, zorder=5)
for yi, r in zip(ys, R.itertuples()):
    film = r.group == "Cohort"
    S.interval(ax, yi, r.ci95_lo, r.ci95_hi, S.SPARSE if r.primary else S.SPARSE_L, horizontal=True,
               lw=1.1 if r.primary else 0.8, z=2)
    ax.plot([r.diff], [yi], "D" if film else "o", ms=3.4 if film else 3.8, mfc=S.SPARSE if r.primary else "white",
            mec=S.SPARSE, mew=0.8, zorder=3)
    ax.text(-0.04, yi, r.label, transform=tr, ha="right", va="center", fontsize=S.T_SMALL,
            color=S.INK if r.primary else S.INK2)
ax.text(cfg["xlim"][0] + 0.0005, 0.25, cfg["lose_text"], fontsize=S.T_SMALL, color=cfg["lose_color"], ha="left",
        va="center", linespacing=1.0)
ax.set_yticks([])
ax.set_xlim(*cfg["xlim"])
ax.set_xticks(cfg["ticks"])
ax.set_xticklabels([S.num(v, "g") if v else "0" for v in cfg["ticks"]])
ax.set_ylim(ys.min() - 0.8, 0.6)
ax.set_xlabel("r difference", labelpad=1)
S.despine(ax, left=False)
S.save(fig, __file__)
