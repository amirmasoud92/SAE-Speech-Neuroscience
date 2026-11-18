"""Figure 4g (left, IFG) and (right, STG; the same script in ../panel_g_STG): sparse-minus-dense
total out-of-fold r against lag, 300 components each, training-only fits, without (thin) and with
(thick) timing and neighbouring-word controls; shading, pointwise 95% participant-clustered
bootstrap bands; bar, a cluster resolved by an exact participant-level sign-flip test (P < 0.05,
sparse below dense).

data.csv: per condition and lag, the median difference and its band, and the electrode count.
clusters.csv: the significant clusters (may be empty).
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
ROI = HERE.name.split("_")[-1]
D = pd.read_csv(HERE / "data.csv")
CL = pd.read_csv(HERE / "clusters.csv")

fig, ax = S.panel(92, 50, box=(13, 6, 72, 33))
S.note(fig, ROI, 13, 1.0, size=S.T_ANNOT, fontweight="bold", color=S.REGION[ROI])
S.note(fig, f"{int(D.n_electrodes.iloc[0])} electrodes", 20, 1.4, size=S.T_SMALL, color=S.INK2)
ax.axhspan(-0.04, 0, color=S.DENSE_FIELD, lw=0, zorder=0)
for cond, lw, c in (("without controls", 0.7, S.SPARSE_L), ("with controls", 1.5, S.SPARSE)):
    b = D[D.condition == cond].sort_values("lag_ms")
    ax.fill_between(b.lag_ms, b.band_lo, b.band_hi, color=c, alpha=0.20, lw=0, zorder=1)
    ax.plot(b.lag_ms, b.median_diff, color=c, lw=lw, zorder=3)
for r in CL.itertuples():
    ax.plot([r.lag_start_ms, r.lag_stop_ms], [-0.036, -0.036], color=S.INK, lw=2.0, solid_capstyle="butt",
            zorder=4)
    S.rich_text(ax, (r.lag_start_ms + r.lag_stop_ms) / 2, -0.0335,
                f"{S.num(r.lag_start_ms, '+.0f')} to {S.num(r.lag_stop_ms, '+.0f')} ms, {S.p_str(r.p_cluster)}",
                ax.transData, ha="center", va="bottom", fontsize=S.T_SMALL, color=S.INK)
ax.axhline(0, color=S.INK2, lw=S.LW_REF, zorder=0.5)
ax.axvline(0, color=S.RULE, lw=S.LW_REF, zorder=0)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-0.04, 0.02)
ax.set_xticks([-2000, -1000, 0, 1000, 2000])
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
ax.set_yticks([-0.04, -0.02, 0, 0.02])
ax.set_yticklabels(["−0.04", "−0.02", "0", "0.02"])
ax.set_xlabel("lag from word onset (ms)", labelpad=1)
ax.set_ylabel("r difference", labelpad=1.5)
ax.text(1950, 0.019, "sparse better", ha="right", va="top", fontsize=S.T_SMALL, color=S.SPARSE)
ax.text(1950, -0.0392, "dense better", ha="right", va="bottom", fontsize=S.T_SMALL, color=S.DENSE)
S.despine(ax)
S.save(fig, __file__)
