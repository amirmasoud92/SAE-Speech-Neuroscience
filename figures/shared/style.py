"""Common type, colour, line and axis settings for the figure panels.

- Colour carries one meaning in every panel: blue, the sparse code; terracotta, the dense
  stream; mulberry, GPT-2 XL; warm slate, classical linguistics; sage green, the acoustic band;
  warm greys, controls and nulls. Regions have their own colours (``REGION``).
- Type: Arial, then Liberation Sans, Microsoft Sans Serif, DejaVu Sans; sizes are final print
  sizes. Exponents are Unicode superscripts.
- PDF text stays editable (TrueType, fonttype 42).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
PT_PER_MM = 72 / 25.4

# ---- colour tokens ----------------------------------------------------------------------
SPARSE_D, SPARSE, SPARSE_L, SPARSE_XL = "#172C5E", "#27468F", "#8C9FCB", "#E2E7F2"
DENSE_D, DENSE, DENSE_L, DENSE_XL = "#7C3516", "#B8582E", "#E0A486", "#F5E3D9"
PRIOR, PRIOR_L = "#6E2A48", "#BC90A3"              # GPT-2 XL
CLASSICAL, CLASSICAL_L = "#6B655E", "#BDB7B0"
CTRL, CTRL_L, CTRL_XL = "#A8A39C", "#D5D1CB", "#EEECE8"
INK, INK2, RULE = "#1C1F22", "#5D636A", "#CED2D7"
FIELD = "#F4F2EE"                                  # neutral wash
DENSE_FIELD = "#FBEFE8"                            # the side of zero where the sparse code loses
BAND_ACOUSTIC, ACOUSTIC_L = "#5A8A5E", "#AFC8B0"
BAND_LEXICAL, BAND_SURPRISAL = "#8F8982", "#C2BCB5"
REGION = {"IFG": "#99326A", "STG": "#228A84", "MTG": "#7DB8B1", "TMP": "#C39235",
          "EAC": "#6E6E6E", "MFG": "#9C9C9C", "PRC": "#C4C4C4", "other": "#E4E2DE"}
LANGUAGE = ("IFG", "STG", "MTG", "TMP")

SPARSE_CMAP = LinearSegmentedColormap.from_list(
    "sparse_seq", ["#F3F5FA", "#C9D2E8", "#7F95C6", SPARSE, "#101F45"])
# the same ramp without its near-white end, for marks that must stay visible (counts, densities)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "sparse_seq_marks", ["#C3CDE6", "#7F95C6", SPARSE, "#101F45"])

TEXT_BG = dict(boxstyle="square,pad=0.06", fc="white", ec="none", alpha=0.85)

# ---- type scale (pt, final size) and geometry ---------------------------------------------
T_LETTER, T_HEAD, T_LABEL, T_TICK, T_ANNOT, T_SMALL = 8.0, 6.5, 6.5, 6.0, 6.0, 5.5
LW_AX, LW_DATA, LW_CI, LW_REF = 0.45, 1.0, 0.8, 0.45
SPINE_OFFSET_MM = 1.0
MINUS = "−"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "Microsoft Sans Serif", "DejaVu Sans"],
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": T_LABEL, "axes.labelsize": T_LABEL, "axes.titlesize": T_HEAD,
        "xtick.labelsize": T_TICK, "ytick.labelsize": T_TICK, "legend.fontsize": T_ANNOT,
        "axes.linewidth": LW_AX, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2, "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": LW_DATA, "lines.markersize": 3.5, "lines.solid_capstyle": "butt",
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "axes.unicode_minus": True,
        "hatch.linewidth": 0.4, "text.color": INK,
    })


# ---- figures ------------------------------------------------------------------------------
def panel(width_mm: float, height_mm: float, box=(12.0, 3.0, None, None)):
    """A figure holding one axes. ``box`` = (left, top, width, height) in mm; width and height
    default to the rest of the figure less a 10-mm bottom and 3-mm right margin."""
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)
    x, y, w, h = box
    w = width_mm - x - 3.0 if w is None else w
    h = height_mm - y - 10.0 if h is None else h
    return fig, ax_mm(fig, x, y, w, h)


def figure(width_mm: float, height_mm: float):
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)
    return fig


def ax_mm(fig, x: float, y: float, w: float, h: float, **kw):
    """Axes whose box starts x mm from the left and y mm from the top."""
    W, H = fig._mm
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def fxy(fig, x: float, y: float):
    W, H = fig._mm
    return x / W, 1 - y / H


def note(fig, s: str, x: float, y: float, ha: str = "left", va: str = "top",
         color: str = INK2, size: float = T_ANNOT, **kw) -> None:
    """Text at (x, y) mm from the top-left corner of the figure."""
    W, H = fig._mm
    if needs_rich(s):
        rich_text(fig, x / W, 1 - y / H, s, "figure fraction", ha=ha, va=va, fontsize=size,
                  color=color, **kw)
        return
    fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def save(fig, script: str, name: str = "panel") -> Path:
    """Write ``<name>.pdf`` next to the panel script, or into the folder named by the
    environment variable ``PANEL_OUT``; ``PANEL_FORMAT`` (for example ``png``) changes the
    format."""
    here = Path(script).resolve().parent
    out = Path(os.environ["PANEL_OUT"]) if "PANEL_OUT" in os.environ else here
    out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(here.parts[-2:]) if "PANEL_OUT" in os.environ else name
    p = out / f"{tag}.{os.environ.get('PANEL_FORMAT', 'pdf')}"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---- text --------------------------------------------------------------------------------
_ITALIC = {"$P$": "P", r"$\beta$": "β", r"$\rho$": "ρ", r"$\delta$": "δ", "$n$": "n",
           "$k$": "k", "$r$": "r", "$z$": "z"}
_SUPCHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
_SUP = str.maketrans("0123456789-−+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺")


def needs_rich(s: str) -> bool:
    """Mathtext has no glyph fallback, so a string holding both $...$ and a Unicode exponent is
    set as plain pieces instead."""
    return "$" in s and any(c in _SUPCHARS for c in s)


def rich_text(target, x: float, y: float, s: str, xycoords, ha: str = "left",
              va: str = "top", fontsize: float = T_ANNOT, color: str = INK, **kw):
    """Text with its italic symbols as italic pieces, packed on one baseline."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
    pieces, carry = [], ""
    for p in [q for q in re.split(r"(\$[^$]+\$)", s) if q]:
        italic = p.startswith("$")
        t = carry + (_ITALIC[p] if italic else p)
        carry = t[len(t.rstrip(" ")):]
        pieces.append((t.rstrip(" "), italic))
    props = dict(fontsize=fontsize, color=color, **kw)
    kids = [TextArea(t, textprops=dict(props, fontstyle="italic") if it else props)
            for t, it in pieces if t]
    box = HPacker(children=kids, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(box, (x, y), xycoords=xycoords, frameon=False, pad=0,
                        box_alignment=({"left": 0.0, "center": 0.5, "right": 1.0}[ha],
                                       {"top": 1.0, "center": 0.5, "bottom": 0.0}[va]),
                        annotation_clip=False)
    target.add_artist(ab)
    return ab


def text(ax, x, y, s, **kw):
    """``ax.text`` that also handles italic symbols next to Unicode exponents."""
    if needs_rich(s):
        kw.setdefault("fontsize", T_ANNOT)
        kw.setdefault("color", INK)
        return rich_text(ax, x, y, s, kw.pop("transform", ax.transData), **kw)
    return ax.text(x, y, s, **kw)


def sup(x) -> str:
    return str(x).translate(_SUP)


def pow10(e: int) -> str:
    return "1" if e == 0 else "10" + sup(e)


def num(v: float, spec: str = "+.4f") -> str:
    """A number with a true minus sign."""
    return format(v, spec).replace("-", MINUS)


def fmt_p(p: float, sig: int = 2) -> str:
    s = f"{p:.{sig}g}"
    if "e" in s:
        s = f"{p:.{sig - 1 - int(np.floor(np.log10(p)))}f}"
    return s


def p_str(p: float, sig: int = 2) -> str:
    """$P$ with mantissa x 10^e below 1e-4."""
    if not np.isfinite(p):
        return ""
    if p <= 0:
        return "$P$ < " + pow10(-4)
    if p < 0.99995e-4:
        e = int(np.floor(np.log10(p)))
        m = round(p / 10 ** e, 1)
        if m >= 10:
            m, e = m / 10, e + 1
        return f"$P$ = {m:.1f} × {pow10(e)}"
    return rf"$P$ = {fmt_p(p, sig)}"


def sci(v: float, digits: int = 1) -> str:
    e = int(np.floor(np.log10(abs(v))))
    return f"{v / 10 ** e:.{digits}f} × {pow10(e)}"


def minus_formatter(fmt: str = "{:g}"):
    return plt.FuncFormatter(lambda v, _: fmt.format(v).replace("-", MINUS) if v != 0 else "0")


# ---- axes and marks ---------------------------------------------------------------------
def despine(ax, left: bool = True, bottom: bool = True, offset_mm: float = SPINE_OFFSET_MM):
    """Spines detached from the data by ``offset_mm``; hidden spines lose their ticks."""
    for side, show, axis in (("left", left, ax.yaxis), ("bottom", bottom, ax.xaxis)):
        sp = ax.spines[side]
        sp.set_visible(show)
        if show and offset_mm:
            props = [dict(color=l.get_color(), fontsize=l.get_fontsize(),
                          fontweight=l.get_fontweight(), rotation=l.get_rotation(),
                          ha=l.get_ha(), va=l.get_va()) for l in axis.get_ticklabels()]
            sp.set_position(("outward", offset_mm * PT_PER_MM))
            for l, p in zip(axis.get_ticklabels(), props):
                l.update(p)
    if not left:
        ax.tick_params(axis="y", length=0)
    if not bottom:
        ax.tick_params(axis="x", length=0)


def interval(ax, x, lo, hi, color, lw=LW_CI, horizontal=False, z=2, alpha=1.0):
    if horizontal:
        ax.plot([lo, hi], [x, x], color=color, lw=lw, solid_capstyle="butt", zorder=z,
                alpha=alpha)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=z,
                alpha=alpha)


def dot(ax, x, y, color, filled=True, size=3.6, marker="o", z=3, mew=0.8, edge=None):
    ax.plot([x], [y], marker, ms=size, color=color, mfc=color if filled else "white",
            mec=edge or color, mew=mew, zorder=z, ls="none")


def ref(ax, v, horizontal=True, **kw):
    kw = {"color": RULE, "lw": LW_REF, "zorder": 0.5, **kw}
    (ax.axhline if horizontal else ax.axvline)(v, **kw)


def half_violin(ax, x, vals, color, width=0.32, side="right", alpha=0.35, bw=0.25,
                horizontal=False, lw=0.0, ec=None):
    """A one-sided kernel density."""
    from scipy.stats import gaussian_kde
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return
    grid = np.linspace(v.min(), v.max(), 256)
    d = gaussian_kde(v, bw_method=bw)(grid)
    d = d / d.max() * width
    sgn = 1 if side == "right" else -1
    if horizontal:
        ax.fill_between(grid, x, x + sgn * d, color=color, alpha=alpha, lw=0, zorder=1)
        if lw:
            ax.plot(grid, x + sgn * d, color=ec or color, lw=lw, zorder=1.2)
    else:
        ax.fill_betweenx(grid, x, x + sgn * d, color=color, alpha=alpha, lw=0, zorder=1)
        if lw:
            ax.plot(x + sgn * d, grid, color=ec or color, lw=lw, zorder=1.2)


def beeswarm_offsets(vals, width=0.32, nbins=None):
    """Deterministic swarm offsets: points binned along the value axis spread sideways."""
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return np.zeros(0)
    nbins = nbins or max(8, int(np.sqrt(n) * 2.2))
    edges = np.linspace(np.nanmin(v), np.nanmax(v) + 1e-12, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    off = np.zeros(n)
    cmax = max(np.bincount(idx, minlength=nbins).max(), 1)
    step = width / max(cmax / 2, 1)
    for b in range(nbins):
        ii = np.flatnonzero(idx == b)
        ii = ii[np.argsort(v[ii])]
        k = np.arange(len(ii))
        off[ii] = ((k + 1) // 2) * step * np.where(k % 2 == 0, 1, -1)
    return np.clip(off, -width, width)
