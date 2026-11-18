"""Per-electrode held-out r in IFG for single-representation arms.

Every arm is fitted on the IFG 0-500 ms window target with exact masked ridge, under one of
two conventions:

* ``LEG`` - global z-scored PCA and one penalty per arm on pooled held-out r (the convention of
  the matched-capacity analyses);
* ``TO`` - training-only: PCA and penalty inside the outer training folds.

Writes ``controls/window_arms/<convention>_<arm>.csv`` (one column of r per electrode); contrasts
are computed by ``controls/window_contrasts.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent


def raw_arm(cfg, ctx, name):
    """(raw matrix, capacity k) for a named arm; k None means no PCA."""
    wi = ctx["wi"]
    z = lambda: np.load(cfg.results("controls", "features", "word_arms.npz"))  # noqa
    if name == "classical-85":
        from feature_extraction.feature_bands import classical_features
        C, _ = classical_features(cfg, wi, ctx["words"]["word"])
        return np.asarray(C, np.float64), None
    base, k = name.rsplit("-", 1)
    k = int(k)
    if base == "SAE":
        return M.sae_band(cfg), k
    if base == "Gemma dense":
        return M.gemma_dense(cfg, wi), k
    if base == "GPT-2 XL":
        return M.gpt2xl(cfg, wi), k
    if base == "Static embedding":
        return z()["static"][wi].astype(np.float64), k
    if base.startswith("pool:"):                       # pool:sae_last, pool:dense_max ...
        return z()[base[5:]][wi].astype(np.float64), k
    if base == "Matryoshka":
        f = cfg.results("features") / "gemma2-2b_L12_matryoshka32k.hdf5"
        with h5py.File(f, "r") as h:
            F = h["sae_word_max"][...][wi].astype(np.float64)
        return F[:, (F > 0).sum(0) >= int(cfg.sae["min_active_words"])], k
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", choices=("LEG", "TO"), required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("controls", "window_arms")
    out.mkdir(parents=True, exist_ok=True)
    ctx = PS.context(cfg)
    Y, V, subj, meta = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    for name in args.arms:
        f = out / f"{args.conv}_{name.replace(':', '_').replace(' ', '_')}.csv"
        if f.exists():
            print(f"  {name}: exists, skipped")
            continue
        X, k = raw_arm(cfg, ctx, name)
        if args.conv == "LEG":
            Xr = X if k is None else M.zpca(X, k)
            r, lam = PS.oof_legacy(Xr, ctx, Y, V, exact=True)
            lam = [lam]
        else:
            P, Yr, lams = PS.oof_to(X, k, ctx, Y, V, cfg, exact=True)
            r, lam = M.colcorr(P, Yr, V), lams.tolist()
        pd.DataFrame({"subject": subj, "ch": meta.ch, "r": r}).to_csv(f, index=False)
        json.dump({"arm": name, "conv": args.conv, "n_raw_columns": int(X.shape[1]),
                   "k": k, "lambdas": lam, "mean_r": float(r.mean())},
                  open(f.with_suffix(".json"), "w"), indent=1)
        print(f"  {args.conv} {name:28s} mean r {r.mean():+.4f}  lambda {lam}", flush=True)


if __name__ == "__main__":
    main()
