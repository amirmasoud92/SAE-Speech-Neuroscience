"""Per-participant tally: 300 sparse features against 300 principal components.

* one lag per electrode: the peak of the unique sparse-band variance in the variance partition;
* admissible where the full-band held-out R2 is positive in **both** arms at that lag;
* per electrode, the arms are compared on R2_top20, the held-out R2 of the top 20 features
  selected inside each training fold;
* per participant and scope (IFG; the four language regions), ``win`` is the percentage of
  admissible electrodes where the first arm is higher, and ``p`` a two-sided Wilcoxon
  signed-rank test on the paired differences (at least three electrodes).

``primary``    : the fixed 300-feature pool against 300 principal components of the stream.
``diagnostic`` : the fixed 300-feature pool against the full 4,933-feature band.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import load_config

K = 20


def tally(cfg, root: Path, metric: str = "r2top") -> pd.DataFrame:
    lang = list(cfg.raw["rois"]["language_rois"]) if "rois" in cfg.raw else None
    if lang is None:
        lang = ["IFG", "STG", "MTG", "TMP"]
    rows = []
    for sub in cfg.subjects:
        v = np.load(root / "encoding" / f"{sub}_variance_partition.npz", allow_pickle=True)
        m = np.load(root / "basis_fits" / f"matched_sci_{sub}.npz", allow_pickle=True)
        d = np.load(root / "basis_fits" / f"dense_sci_{sub}.npz", allow_pickle=True)
        s = np.load(root / "basis_fits" / f"sci_{sub}.npz", allow_pickle=True)
        for z in (m, d, s):
            if list(z["ch_names"].astype(str)) != list(v["ch_names"].astype(str)):
                raise SystemExit(f"{sub}: channel order differs between the fits and the partition")
        dl = v["delta_sae"]
        li = np.nanargmax(np.where(np.isfinite(dl), dl, -np.inf), 1)
        r = np.arange(len(li))
        roi = v["roi"].astype(str)
        key = (lambda z, p="": z[f"{p}sci_{K}"]) if metric == "sci" else \
              (lambda z, p="": z[f"{p}r2_top{K}"])
        arms = {"matched": (m["r2_all"][r, li], key(m)[r, li]),
                "dense": (d["r2_all"][r, li], key(d)[r, li]),
                "full": (s["real__r2_all"][r, li], key(s, "real__")[r, li])}
        for cmp, other in (("primary", "dense"), ("diagnostic", "full")):
            a_all, a = arms["matched"]
            b_all, b = arms[other]
            adm = (a_all > 0) & (b_all > 0) & np.isfinite(a_all) & np.isfinite(b_all)
            for scope, sel in (("IFG", roi == "IFG"), ("language", np.isin(roi, lang))):
                k = adm & sel
                n = int(k.sum())
                diff = a[k] - b[k]
                win = float(100 * (diff > 0).mean()) if n else np.nan
                p = float(sps.wilcoxon(diff).pvalue) if n >= 3 else np.nan
                rows.append({"sub": sub, "cmp": cmp, "scope": scope, "n": n, "win": win,
                             "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = tally(cfg, cfg.paths.results_root)
    o = cfg.results("concentration", "a61_tally.csv")
    T.to_csv(o, index=False)
    for c in ("primary", "diagnostic"):
        for sc in ("language", "IFG"):
            u = T[(T.cmp == c) & (T.scope == sc)]
            v = u[u.n >= 3]
            print(f"{c:10s} {sc:8s}: {int(u.n.sum())} electrodes, {len(v)} subjects with "
                  f">= 3, {int((v.win > 50).sum())} favour the matched pool, median win "
                  f"{v.win.median():.0f}%")
    print("written:", o)


if __name__ == "__main__":
    main()
