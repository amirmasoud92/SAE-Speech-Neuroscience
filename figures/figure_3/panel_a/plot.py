"""Figure 3a: semantic coherence (mean pairwise cosine in GPT-2 XL layer 24 of each direction's
top 20 distinct word types; no neural data) for the scorable sparse features, the positive pole
of 300 principal components of the same stream, 300 features of a random dictionary matched in
per-feature activation density, and 300 random 20-word sets (chance: grey band, interquartile
range; dashed line, median). Densities, points and interquartile bars with medians.

data.csv: one row per direction, with its group and coherence.
The statistics are computed here from data.csv: sparse versus principal components, two-sided
Mann-Whitney; principal components versus chance, one-sided; sparse versus the random dictionary,
Cliff's delta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv")
V = {g: d.value.to_numpy(float) for g, d in D.groupby("group")}
V = {g: v[np.isfinite(v)] for g, v in V.items()}
GROUPS = [("SAE-300", "sparse features", "", S.SPARSE, S.SPARSE),
          ("PCA +pole", "principal components", "positive pole; ", S.DENSE, S.DENSE),
          ("random dictionary", "random dictionary", "density-matched; ", S.CTRL, S.CTRL),
          ("random", "random 20-word sets", "", S.CTRL_L, S.CTRL)]

p_sp = float(mannwhitneyu(V["SAE-300"], V["PCA +pole"], alternative="two-sided").pvalue)
p_ch = float(mannwhitneyu(V["PCA +pole"], V["random"], alternative="greater").pvalue)
u = mannwhitneyu(V["SAE-300"], V["random dictionary"]).statistic
delta = 2 * u / (len(V["SAE-300"]) * len(V["random dictionary"])) - 1

fig, ax = S.panel(122, 52, box=(40, 4, 78, 39))
xlo, xhi = 0.37, 0.80
q1r, mdr, q3r = np.percentile(V["random"], [25, 50, 75])
ax.axvspan(q1r, q3r, color=S.FIELD, lw=0, zorder=0)
ax.axvline(mdr, color=S.INK2, lw=S.LW_REF, ls=(0, (3, 2)), zorder=0.5)
ax.text(mdr, 4.02, "chance", fontsize=S.T_SMALL, color=S.INK2, ha="center", va="bottom")
grid = np.linspace(xlo, xhi, 400)
rng = np.random.default_rng(0)
for k, (key, lab, sub, c, bar) in enumerate(GROUPS):
    v = V[key]
    y0 = len(GROUPS) - 1 - k + 0.30
    d = gaussian_kde(v, bw_method=0.22)(grid)
    d = d / d.max() * 0.62
    inside = (grid >= v.min() - 0.01) & (grid <= v.max() + 0.01)
    ax.fill_between(grid[inside], y0, y0 + d[inside], color=c, alpha=0.30, lw=0, zorder=2)
    ax.plot(grid[inside], y0 + d[inside], color=c, lw=0.7, zorder=2.2)
    ax.plot([grid[inside][0], grid[inside][-1]], [y0, y0], color=c, lw=0.45, zorder=2.1)
    ax.scatter(v, y0 - 0.17 + rng.uniform(-0.075, 0.075, len(v)), s=1.4, color=c, alpha=0.65, lw=0,
               zorder=2)
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [y0, y0], color=bar, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([md, md], [y0 - 0.07, y0 + 0.30], color=S.INK, lw=1.0, zorder=4)
    tr = ax.get_yaxis_transform()
    ax.text(-0.03, y0 + 0.36, lab, transform=tr, ha="right", va="bottom", fontsize=S.T_SMALL,
            color=S.INK, fontweight="bold")
    ax.text(-0.03, y0 + 0.30, f"{sub}median {md:.3f}, n = {len(v)}", transform=tr, ha="right",
            va="top", fontsize=S.T_SMALL, color=S.INK2)
S.rich_text(ax, xhi - 0.004, 2.60, f"vs sparse: $P$ = {S.sci(p_sp)}", ax.transData, ha="right",
            va="center", fontsize=S.T_SMALL, color=S.INK)
S.rich_text(ax, xhi - 0.004, 2.35, f"vs chance: $P$ = {p_ch:.2f}", ax.transData, ha="right",
            va="center", fontsize=S.T_SMALL, color=S.INK2)
ax.text(xhi - 0.004, 1.60, f"vs sparse: Cliff's δ = {delta:.2f}", ha="right", va="center",
        fontsize=S.T_SMALL, color=S.INK)
ax.set_xlim(xlo, xhi)
ax.set_ylim(-0.12, 4.0)
ax.set_yticks([])
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_xlabel("coherence (mean pairwise cosine)", labelpad=1)
S.despine(ax, left=False)
S.save(fig, __file__)
