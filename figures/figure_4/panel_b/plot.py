"""Figure 4b: every arm on panel a's scale. IFG mean out-of-fold r for the classical linguistic
band (85 dimensions), GPT-2 XL, the sparse code and the Gemma dense stream (300 principal
components each), with the participant-clustered contrasts.

data.csv: per arm, its mean r.
statistics.csv: the three contrasts (difference, interval, P).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").set_index("arm")
T = pd.read_csv(HERE / "statistics.csv").set_index("comparison")
ARMS = [("classical", S.CLASSICAL), ("GPT-2 XL", S.PRIOR), ("sparse", S.SPARSE), ("Gemma dense", S.DENSE)]

fig, ax = S.panel(50, 48, box=(4, 3, 44, 35))
for yy in (0.06, 0.08, 0.10, 0.12, 0.14):
    ax.axhline(yy, color=S.CTRL_XL, lw=0.4, zorder=0)
for i, (lab, c) in enumerate(ARMS):
    v = float(D.loc[lab, "mean_r"])
    S.dot(ax, i, v, c, size=5.0, edge="white", mew=0.5)
    ax.text(i + 0.17, v, f"{v:.3f}", fontsize=S.T_SMALL, color=S.INK, ha="left", va="center")
    ax.text(i, -0.04, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=S.T_SMALL,
            color=c)


def bracket(x0, x1, y, row, col):
    r = T.loc[row]
    ax.plot([x0, x0, x1, x1], [y - 0.0022, y, y, y - 0.0022], color=col, lw=0.5)
    S.rich_text(ax, (x0 + x1) / 2, y + 0.0012, f"{S.num(r['diff'])}, {S.p_str(r.p)}", ax.transData,
                ha="center", va="bottom", fontsize=S.T_SMALL, color=col)


bracket(2, 3, 0.150, "sparse - Gemma dense", S.INK)
bracket(1, 2, 0.136, "sparse - GPT-2 XL", S.INK2)
r = T.loc["sparse 85 - classical 85"]
ax.text(0.0, 0.0775, "vs classical, 85 vs 85:", ha="left", va="bottom", fontsize=S.T_SMALL, color=S.INK2)
S.rich_text(ax, 0.0, 0.0705, f"{S.num(r['diff'])}, {S.p_str(r.p)}", ax.transData, ha="left", va="bottom",
            fontsize=S.T_SMALL, color=S.INK2)
ax.set_xlim(-0.5, 3.6)
ax.set_ylim(0.04, 0.16)
ax.set_xticks([])
ax.set_yticks([0.06, 0.08, 0.10, 0.12, 0.14])
ax.set_yticklabels([])
S.despine(ax, bottom=False)
S.save(fig, __file__)
