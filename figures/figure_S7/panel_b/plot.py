"""Supplementary Figure 7b: survivors of the family-wise test by region and participant
(survivors / electrodes; shading, fraction surviving), with region and participant totals and the
number of participants with at least one survivor per region. Above, the median peak lag of the IFG
survivors and the proxy's rank correlation with the banded model at the peak.

data.csv: per region and participant, the survivor and electrode counts.
proxy_validity.csv: Spearman rho between the proxy and the banded model over electrodes.
The median IFG peak lag is read from ../panel_a/data.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
B = pd.read_csv(HERE / "data.csv")
E = pd.read_csv(HERE.parent / "panel_a" / "data.csv")
rho_pk = float(pd.read_csv(HERE / "proxy_validity.csv").set_index("proxy_vs_banded_model").spearman_rho["peak"])
ifg_lag = float(E[E.survives.astype(bool) & (E.roi == "IFG")].peak_lag_ms.median())
ROWS = ("IFG", "STG", "MTG", "TMP", "EAC", "MFG", "PRC")
subs = sorted(B.subject.unique())
n_all = B.pivot(index="roi", columns="subject", values="n_electrodes").reindex(index=ROWS, columns=subs).fillna(0)
n_sv = B.pivot(index="roi", columns="subject", values="n_survivors").reindex(index=ROWS, columns=subs).fillna(0)
VMAX = 0.75

fig = S.figure(183, 60)
S.note(fig, f"survivors / electrodes; shading, fraction surviving; median peak lag of IFG survivors "
            f"{S.num(ifg_lag, '+.0f')} ms; proxy against banded model, rank correlation {rho_pk:.2f} (peak)",
       2, 1.0, size=S.T_SMALL)
ax = S.ax_mm(fig, 24, 12, 150, 38)


def cell(ci, ri, s_, n_, bold=False):
    s_, n_ = int(s_), int(n_)
    if n_ == 0:
        ax.text(ci, ri, "–", ha="center", va="center", fontsize=S.T_SMALL, color=S.RULE)
        return
    f = s_ / n_
    ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, lw=0.6, ec="white", fc=S.SPARSE_CMAP(min(f / VMAX, 1.0)),
                               zorder=1))
    ax.text(ci, ri, f"{s_}/{n_}", ha="center", va="center", fontsize=S.T_SMALL, color="white" if f > 0.42 else S.INK,
            zorder=2, fontweight="bold" if bold else "normal")


for ri, r in enumerate(ROWS):
    for ci, s in enumerate(subs):
        cell(ci, ri, n_sv.loc[r, s], n_all.loc[r, s])
    cell(len(subs) + 0.25, ri, n_sv.loc[r].sum(), n_all.loc[r].sum(), bold=True)
    ax.text(len(subs) + 0.95, ri, f"{int((n_sv.loc[r] > 0).sum())} of {int((n_all.loc[r] > 0).sum())}", ha="left",
            va="center", fontsize=S.T_SMALL, color=S.INK2)
for ci, s in enumerate(subs):
    cell(ci, len(ROWS) + 0.25, n_sv[s].sum(), n_all[s].sum(), bold=True)
cell(len(subs) + 0.25, len(ROWS) + 0.25, n_sv.values.sum(), n_all.values.sum(), bold=True)
ax.axhline(3.5, color=S.INK2, lw=0.6, zorder=3)
ax.set_xlim(-0.5, len(subs) + 2.3)
ax.set_ylim(len(ROWS) + 0.75, -0.5)
ax.set_yticks(list(range(len(ROWS))) + [len(ROWS) + 0.25])
ax.set_yticklabels(list(ROWS) + ["all"])
for t, r in zip(ax.get_yticklabels(), list(ROWS) + ["all"]):
    t.set_color(S.REGION[r] if r in S.LANGUAGE else S.INK2)
    t.set_fontweight("bold")
ax.set_xticks(list(range(len(subs))) + [len(subs) + 0.25, len(subs) + 1.35])
ax.set_xticklabels([s.replace("sub-", "") for s in subs] + ["all", "participants"], fontsize=S.T_SMALL)
ax.xaxis.tick_top()
ax.tick_params(length=0, pad=2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.text(-1.55, 1.5, "language", rotation=90, ha="center", va="center", fontsize=S.T_SMALL, color=S.INK2)
ax.text(-1.55, 5.0, "control", rotation=90, ha="center", va="center", fontsize=S.T_SMALL, color=S.INK2)
S.save(fig, __file__)
