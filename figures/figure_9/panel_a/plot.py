"""Figure 9a: within-type signal relative to the dense stream. The within-word-type component of
each arm's prediction minus that of the dense stream (Gemma-2-2B layer 12, IFG, 300 components per
arm; type means within test fold; types occurring at least four times): the sparse code of the
primary dictionary (filled), the dictionary's reconstruction of the stream (open) and, in the
podcast cohort only, an exploratory arm pooling the sparse code over a word's tokens by the mean
(grey). Circles, podcast; diamonds, film. Reconstruction rows report the Holm-corrected P.

data.csv: per cohort and arm, the median difference, its 95% interval, P and (reconstruction
rows) the Holm-corrected P.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402


def pfmt(p):
    return "$P$ < 0.0001" if p < 1e-4 else f"$P$ = {p:.4f}" if p < 0.01 else f"$P$ = {p:.2f}"


S.apply()
D = pd.read_csv(HERE / "data.csv")
ROWS = [("Podcast · ECoG", None, None), ("sparse code", "Podcast", "o"), ("reconstruction", "Podcast", "o"),
        ("mean-pooled sparse code", "Podcast", "o"), ("Film · sEEG", None, None),
        ("sparse code", "Brain Treebank", "D"), ("reconstruction", "Brain Treebank", "D")]

fig, a = S.panel(118, 50, box=(36, 3, 34, 38))
a.axvspan(-0.024, 0, color=S.DENSE_FIELD, lw=0, zorder=0)
a.axvline(0, color=S.INK2, lw=0.6, zorder=1)
nr = len(ROWS)
tr = a.get_yaxis_transform()
for k, (lab, coh, mk) in enumerate(ROWS):
    y = nr - 1 - k
    if coh is None:
        a.text(-1.0, y, lab, transform=tr, ha="left", va="center", fontsize=S.T_SMALL, color=S.INK,
               fontweight="bold")
        continue
    r = D[(D.cohort == coh) & (D.row == lab)].iloc[0]
    expl = bool(r.exploratory)
    col = S.CTRL if expl else S.SPARSE
    filled = lab != "reconstruction"
    S.interval(a, y, r.lo, r.hi, col, horizontal=True, lw=0.9)
    a.plot([r["median"]], [y], mk, ms=3.8 if mk == "o" else 3.4, mfc=col if filled else "white", mec=col,
           mew=0.9, zorder=3)
    a.text(-0.92, y, lab + ("\nexploratory" if expl else ""), transform=tr, ha="left", va="center",
           fontsize=S.T_SMALL, color=S.INK2 if expl else S.INK, linespacing=1.0)
    p = pfmt(r.p_holm) + " (Holm)" if np.isfinite(r.p_holm) else pfmt(r.p)
    S.rich_text(a, 1.05, y, f"{S.num(r['median'])}, {p}", tr, ha="left", va="center", fontsize=S.T_SMALL,
                color=S.INK2 if expl else S.INK)
a.set_yticks([])
a.set_ylim(-0.6, nr - 0.4)
a.set_xlim(-0.024, 0.003)
a.set_xticks([-0.02, -0.01, 0])
a.set_xticklabels(["−0.02", "−0.01", "0"])
a.set_xlabel("r difference", labelpad=1)
S.despine(a, left=False)
a.text(-0.0235, -0.45, "dense better", fontsize=S.T_SMALL, color=S.DENSE, ha="left", va="bottom")
S.save(fig, __file__)
