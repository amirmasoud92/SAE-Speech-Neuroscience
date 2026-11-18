"""Figure 1f: mean unique sparse-band variance against lag in IFG, STG and early auditory cortex;
shading, s.e.m. across electrodes; dashed line, the IFG peak.

data.csv: per region and lag, the mean unique variance, its s.e.m. and the electrode count.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")

fig, ax = S.panel(48, 58, box=(10, 4, 32, 44))
prof = {}
for r in ["EAC", "STG", "IFG"]:
    g = D[D.roi == r].sort_values("lag_s")
    lms, mu, se = g.lag_s.to_numpy() * 1000, g.mean_delta_sae.to_numpy(), g["sem"].to_numpy()
    prof[r] = (lms, mu, int(g.n_electrodes.iloc[0]))
    ax.fill_between(lms, mu - se, mu + se, color=S.REGION[r], alpha=0.20, lw=0)
    ax.plot(lms, mu, color=S.REGION[r], lw=S.LW_DATA)
lms, mu, _ = prof["IFG"]
pk_lag, peak = float(lms[int(np.argmax(mu))]), float(mu.max())
ax.axvline(pk_lag, lw=S.LW_REF, ls=(0, (2, 2)), color=S.INK2, zorder=0)
ax.axhline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.axvline(0, lw=S.LW_REF, color=S.RULE, zorder=0)
ax.text(pk_lag + 100, peak * 1.06, f"{pk_lag:+.0f} ms", fontsize=S.T_ANNOT, va="bottom",
        fontweight="bold")
for r, (x, y, ha) in {"IFG": (550, 0.0128, "left"), "STG": (-320, 0.0106, "right"),
                      "EAC": (-550, 0.0036, "right")}.items():
    ax.text(x, y, r, fontsize=S.T_SMALL, color=tuple(np.array(to_rgb(S.REGION[r])) * 0.78),
            fontweight="bold", ha=ha, va="center", bbox=S.TEXT_BG)
    ax.text(x, y - 0.0012, f"n = {prof[r][2]}", fontsize=S.T_SMALL - 0.3, color=S.INK2, ha=ha,
            va="center", bbox=S.TEXT_BG)
ax.set_xlim(-2000, 2000)
ax.set_xticks([-2000, -1000, 0, 1000, 2000])
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
ax.set_ylim(-0.0025, peak * 1.22)
ax.set_xlabel("lag from word onset (ms)", labelpad=1)
ax.set_ylabel("mean unique ΔR², sparse", labelpad=1.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip(".") if v else "0"))
S.despine(ax)
S.save(fig, __file__)
