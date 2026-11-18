"""Figure 2d (left): mean unique sparse-band variance in IFG against lag, without and with a
79-column syntactic band; shading, s.e.m. across electrodes.

data.csv: per model and lag, the mean unique variance and its s.e.m. over IFG electrodes.
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
D = pd.read_csv(HERE / "data.csv")
N_ELECTRODES = 135                                 # IFG electrodes in both models (paper)

fig, ax = S.panel(74, 44, box=(13, 5, 50, 30))
pk = {}
for lab, c, lw in (("4-band (no syntax)", S.SPARSE_L, 1.5), ("5-band (+ 79 syntax columns)", S.SPARSE, S.LW_DATA)):
    g = D[D.model == lab].sort_values("lag_s")
    lms, m, se = g.lag_s.to_numpy() * 1000, g["mean"].to_numpy(), g["sem"].to_numpy()
    ax.fill_between(lms, m - se, m + se, color=c, alpha=0.22, lw=0)
    ax.plot(lms, m, color=c, lw=lw)
    pk[lab] = (float(lms[int(np.argmax(m))]), float(m.max()))
(l4, m4), (l5, m5) = pk["4-band (no syntax)"], pk["5-band (+ 79 syntax columns)"]
ax.text(l4 + 200, m4 * 1.02, f"without syntax, peak {l4:+.0f} ms", fontsize=S.T_SMALL, color=S.SPARSE_L,
        va="center", ha="left", fontweight="bold")
ax.text(800, m5 * 0.76, f"with syntax,\npeak {l5:+.0f} ms", fontsize=S.T_SMALL, color=S.SPARSE, va="center",
        ha="left", fontweight="bold", linespacing=1.05)
ax.axhline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.axvline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.set_xlim(-1000, 1500)
ax.set_ylim(-0.001, m4 * 1.15)
ax.set_xticks([-1000, -500, 0, 500, 1000, 1500])
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
ax.set_xlabel("lag from word onset (ms)", labelpad=1)
ax.set_ylabel("IFG mean unique ΔR²", labelpad=1.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"))
S.despine(ax)
S.note(fig, f"mean ± s.e.m. over {N_ELECTRODES} electrodes", 2, 1.0, size=S.T_SMALL)
S.save(fig, __file__)
