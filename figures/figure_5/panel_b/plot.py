"""Figure 5b: admissible language-network electrodes per participant for the two
comparisons: the diagnostic (pruned 300 against all 4,933 sparse features; blue) and the primary
comparison (grey), which also requires the comparison arm's held-out R2 to be positive.

data.csv: per participant, the admissible electrode counts of the two comparisons.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

S.apply()
D = pd.read_csv(HERE / "data.csv").sort_values("n_diagnostic", ascending=False).reset_index(drop=True)
nd, npm = D.n_diagnostic.to_numpy(float), D.n_primary.to_numpy(float)
top2 = D.nlargest(2, "n_primary")

fig, ax = S.panel(80, 72, box=(10, 9, 67, 52))
S.note(fig, f"{top2.subject.iloc[0]} and {top2.subject.iloc[1]} supply "
            f"{top2.n_primary.sum() / npm.sum():.0%} of the primary comparison", 2, 1.0, size=S.T_SMALL)
xs = np.arange(len(D))
ax.bar(xs - 0.19, nd, 0.36, color=S.SPARSE, lw=0, zorder=3)
ax.bar(xs + 0.19, npm, 0.36, color=S.CTRL, lw=0, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels([s.replace("sub-", "") for s in D.subject])
ax.tick_params(axis="x", length=0)
ax.set_xlabel("participant", labelpad=1.5)
ax.set_ylabel("admissible language electrodes", labelpad=2)
ax.set_ylim(0, 72)
S.despine(ax, bottom=False)
ax.text(4.0, 52, f"diagnostic, n = {int(nd.sum())}", color=S.SPARSE, fontsize=S.T_SMALL, fontweight="bold")
ax.text(4.0, 45, f"primary, n = {int(npm.sum())}", color=S.INK2, fontsize=S.T_SMALL, fontweight="bold")
S.save(fig, __file__)
