"""Supplementary Figure 2 (a-f; one region per panel: IFG, STG, MTG, TMP, EAC, PRC). The same
script draws each panel; the region is read from data.csv. Mean unique held-out variance (full
model minus the model with that band removed) against lag from word onset for the sparse feature
band (blue) and word surprisal (grey); shading, +/- 1 s.e.m. across the region's electrodes. The
acoustic band is omitted by design: its delay taps span the full epoch.

data.csv: per band and lag, the mean unique variance, its s.e.m. and the electrode and participant
counts. The values are the stored source data of the paper's figure, re-rendered unchanged.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv", float_precision="round_trip")
roi = str(D.roi.iloc[0])
BANDS = [("sae", S.SPARSE, S.LW_DATA), ("surprisal", S.BAND_LEXICAL, 0.8)]

fig, ax = S.panel(64, 54, box=(11, 12, 46, 30))
S.note(fig, roi, 11, 2.0, va="center", size=S.T_LABEL, fontweight="bold",
       color=S.REGION[roi] if roi in S.LANGUAGE else S.INK2)
S.note(fig, f"{int(D.n_electrodes.iloc[0])} electrodes, {int(D.n_subjects.iloc[0])} participants", 57, 2.0,
       ha="right", va="center", size=S.T_SMALL)
for k, (lab, col) in enumerate((("sparse band", S.SPARSE), ("word surprisal", S.BAND_LEXICAL))):
    kx = S.ax_mm(fig, 11 + 24 * k, 5.8, 4.4, 1.0)
    kx.plot([0, 1], [0.5, 0.5], color=col, lw=S.LW_DATA)
    kx.axis("off")
    S.note(fig, lab, 16.4 + 24 * k, 6.3, va="center", size=S.T_SMALL, color=col)
for band, col, lw in BANDS:
    q = D[D.band == band].sort_values("lag_s")
    lms, m, se = q.lag_s * 1000, q.mean_delta_r2, q["sem"]
    ax.fill_between(lms, m - se, m + se, color=col, alpha=0.2, lw=0)
    ax.plot(lms, m, color=col, lw=lw)
ax.axvline(0.0, color=S.RULE, lw=S.LW_REF, zorder=0)
ax.axhline(0.0, color=S.RULE, lw=S.LW_REF, zorder=0)
ax.set_xlim(-2000, 2000)
ax.set_ylim(-0.0025, 0.021)
ax.set_xticks([-2000, -1000, 0, 1000, 2000])
ax.xaxis.set_major_formatter(S.minus_formatter("{:+,.0f}"))
ax.set_yticks([0, 0.01, 0.02])
ax.set_yticklabels(["0", "0.01", "0.02"])
ax.set_xlabel("lag from word onset (ms)", labelpad=1)
ax.set_ylabel("unique ΔR²", labelpad=1.5)
S.despine(ax)
S.save(fig, __file__)
