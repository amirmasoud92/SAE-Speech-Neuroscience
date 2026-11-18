"""Figure 10a: film cohort (Brain Treebank, sEEG), peak unique sparse-band variance (lags
0-500 ms) by region with participant-clustered 95% intervals; language regions filled in their
anatomical colours, control regions open. Beneath, the language-versus-control mixed-model
contrast, pre-specified and post hoc without early auditory cortex (EAC).

data.csv: per region, the mean peak unique variance, its interval and the contact count.
statistics.csv: the two mixed-model contrasts (beta in outcome SD, P).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
import style as S  # noqa: E402

ORDER = ["STG", "EAC", "IFG", "MTG", "TMP", "MFG", "PRC"]


def pfmt(p):
    return ("$P$ < 0.0001" if p < 1e-4 else f"$P$ = {p:.4f}" if p < 0.001 else
            f"$P$ = {p:.3f}" if p < 0.1 else f"$P$ = {p:.2f}")


def darker(c, f=0.75):
    return tuple(np.array(to_rgb(c)) * f)


S.apply()
B = pd.read_csv(HERE / "data.csv").set_index("roi")
T = pd.read_csv(HERE / "statistics.csv")
rois = [r for r in ORDER if r in B.index]
row_h = 5.0

fig, a = S.panel(62, 62, box=(16, 4, 27, row_h * len(rois)))
a.set_ylim(len(rois) - 0.5, -0.5)
for yi, r in enumerate(rois):
    b = B.loc[r]
    col, lang = S.REGION[r], r in S.LANGUAGE
    a.axhline(yi, color=S.CTRL_XL, lw=0.35, zorder=0.2)
    S.interval(a, yi, b.lo * 1e3, b.hi * 1e3, col if lang else S.INK2, horizontal=True, lw=1.1)
    a.plot([b.mean_peak * 1e3], [yi], "o", mfc=col if lang else "white", mec=col if lang else S.INK2, mew=0.9,
           ms=3.8, zorder=3)
    a.text(-0.08, yi, r, transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=S.T_SMALL,
           fontweight="bold", color=darker(col) if lang else S.INK2)
    a.text(1.06, yi, f"{int(b.n)}", transform=a.get_yaxis_transform(), ha="left", va="center",
           fontsize=S.T_SMALL, color=S.INK2)
a.text(1.06, -0.75, "n", transform=a.get_yaxis_transform(), ha="left", va="bottom", fontsize=S.T_SMALL,
       color=S.INK2, fontstyle="italic")
a.set_yticks([])
a.set_xlim(0, 20)
a.set_xticks([0, 10, 20])
a.set_xlabel("peak unique ΔR² (×10⁻³)", labelpad=1)
S.despine(a, left=False)
pre = T[T.contrast.str.endswith("pre-specified")].iloc[0]
post = T[T.contrast.str.endswith("post hoc")].iloc[0]
y0 = 4 + row_h * len(rois) + 11.2
S.note(fig, "language vs control, mixed model", 2.5, y0, size=S.T_SMALL)
S.note(fig, f"pre-specified: {S.num(pre.beta_sd, '+.2f')} SD, {pfmt(pre.p)}", 2.5, y0 + 2.9, color=S.INK,
       size=S.T_SMALL)
S.note(fig, f"without EAC, post hoc: {S.num(post.beta_sd, '+.2f')} SD, {pfmt(post.p)}", 2.5, y0 + 5.8,
       size=S.T_SMALL)
S.save(fig, __file__)
