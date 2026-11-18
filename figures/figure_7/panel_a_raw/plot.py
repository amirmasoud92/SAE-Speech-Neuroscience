"""Figure 7a (left): causal necessity against neural importance for the 300 pooled sparse
features, each coloured by activation density (words it is active on, of 5,130). Causal
necessity: the change in Gemma-2-2B's next-token distribution (KL) when the feature is ablated at
every word it is active on. Neural importance: the nuisance-controlled circular-shift |z| averaged
over all 135 IFG electrodes. Line, least-squares fit in the plotted space.

data.csv: per feature, activation density, coherence, neural importance and the causal scores.
statistics.csv: per causal score, the raw and density-partialled Spearman rho and their P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
st = pd.read_csv(HERE / "statistics.csv")
r = st[st.score.str.contains("KL")].iloc[0]
N_WORDS, TICKS = 5130, (25, 100, 1000, 4000)
kl, imp, ld = D.causal_kl.to_numpy(float), D.importance.to_numpy(float), D.log_density.to_numpy(float)
lo, hi = float(np.nanmin(ld)), float(np.nanmax(ld))

fig, ax = S.panel(56, 70, box=(13, 22, 38, 38))
S.note(fig, f"activation density, words active (of {N_WORDS:,})", 13, 1.0, size=S.T_SMALL)
kax = S.ax_mm(fig, 13, 4.2, 30.0, 1.7)
kax.imshow(np.linspace(lo, hi, 256)[None, :], aspect="auto", cmap=S.SEQ_CMAP, extent=(lo, hi, 0, 1),
           interpolation="bilinear")
kax.set_yticks([])
kax.set_xticks(np.log10(TICKS))
kax.set_xticklabels([f"{t:,}" for t in TICKS])
kax.tick_params(axis="x", length=1.0, width=S.LW_AX, labelsize=S.T_SMALL, pad=0.8, color=S.INK2)
for sp_ in kax.spines.values():
    sp_.set_visible(False)
pexp = int(np.floor(np.log10(r.raw_p)))
S.note(fig, "raw", 13, 12.4, color=S.INK, size=S.T_SMALL, fontweight="bold")
S.note(fig, f"ρ = {S.num(r.raw_rho, '+.3f')}, $P$ = {r.raw_p / 10 ** pexp:.1f} × {S.pow10(pexp)}", 13, 15.6,
       color=S.INK, size=S.T_SMALL)

o = np.argsort(ld)
ax.scatter(kl[o], imp[o], c=ld[o], cmap=S.SEQ_CMAP, vmin=lo, vmax=hi, s=6.0, lw=0.2, edgecolors="white",
           zorder=3)
ok = np.isfinite(kl) & np.isfinite(imp) & (kl > 0)
fit = np.polyfit(np.log10(kl[ok]), imp[ok], 1)
xs = np.linspace(np.log10(kl[ok]).min(), np.log10(kl[ok]).max(), 50)
ax.plot(10 ** xs, np.polyval(fit, xs), color=S.INK, lw=0.9, zorder=4)
ax.set_xscale("log")
ax.set_xticks([1e-4, 1e-2, 1e0])
ax.set_xticklabels([S.pow10(e) for e in (-4, -2, 0)])
ax.minorticks_off()
ax.set_xlabel("causal necessity, KL (nats)", labelpad=1)
ax.set_ylabel("neural importance |z|", labelpad=1.5)
ax.set_ylim(0.6, 3.7)
ax.set_yticks([1, 2, 3])
S.despine(ax)
S.save(fig, __file__)
