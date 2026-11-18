"""Figure 2e: IFG median peak unique sparse-band variance for five independently trained
dictionaries (widths 16k to 65k, labelled by nominal L0); filled, the dictionary used throughout.

data.csv: per dictionary, width, nominal L0, the IFG median peak and the lag of the peak of the
mean profile; ``x`` is the plotting position.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
lo, hi = D.peak_of_mean_lag_ms.min(), D.peak_of_mean_lag_ms.max()
lag_txt = f"peak lag {lo:+.0f} ms for all five" if lo == hi else f"peak lag {lo:+.0f} to {hi:+.0f} ms"

fig, ax = S.panel(58, 46, box=(15, 6, 40, 30))
s16 = D[D.width == "16k"].sort_values("x")
ax.plot(s16.x, s16.ifg_median_peak, color=S.SPARSE_L, lw=0.8, zorder=1)
for _, r in D.iterrows():
    main = "main" in r.dictionary
    ax.plot([r.x, r.x], [0.0075, r.ifg_median_peak], color=S.CTRL_L, lw=0.5, zorder=0)
    ax.plot([r.x], [r.ifg_median_peak], "o", ms=4.4, mfc=S.SPARSE if main else "white", mec=S.SPARSE,
            mew=0.8, zorder=3)
    below = r.x == 0.0
    ax.text(r.x + (0.14 if below else 0), r.ifg_median_peak + (-0.00040 if below else 0.00034),
            S.num(r.ifg_median_peak, ".4f"), fontsize=S.T_SMALL, color=S.INK if main else S.INK2,
            ha="left" if below else "center", va="top" if below else "bottom",
            fontweight="bold" if main else "normal")
ax.set_xticks(D.x.tolist())
ax.set_xticklabels([str(int(v)) for v in D.l0])
ax.tick_params(axis="x", length=0)
for xc, lab in ((1.0, "16k"), (3.35, "32k"), (4.35, "65k")):
    ax.text(xc, -0.21, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=S.T_SMALL,
            color=S.INK2)
ax.plot([-0.25, 2.25], [-0.18, -0.18], transform=ax.get_xaxis_transform(), color=S.INK2, lw=0.45,
        clip_on=False)
ax.text(-0.75, -0.04, "L0", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=S.T_SMALL,
        color=S.INK2)
ax.text(-0.75, -0.21, "width", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=S.T_SMALL,
        color=S.INK2)
ax.set_xlim(-0.5, 4.8)
ax.set_ylim(0.0075, 0.0158)
ax.set_yticks([0.008, 0.010, 0.012, 0.014])
ax.set_ylabel("median peak ΔR²", labelpad=1.5)
S.despine(ax, bottom=False)
S.note(fig, f"IFG median peak; {lag_txt}", 2, 1.0, size=S.T_SMALL)
S.save(fig, __file__)
