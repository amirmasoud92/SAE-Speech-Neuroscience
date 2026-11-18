"""Within-type comparison and lag-resolved comparison of the sparse and dense arms.

Training-only convention throughout (``training_only.oof_to``).

* ``--part window``: IFG 0-500 ms window target, exact masked ridge; sparse, Gemma dense,
  GPT-2 XL and static-embedding arms at 300 components each; within- and between-type r per
  electrode (within-fold type means) and electrode statistics.
* ``--part lags --roi R``: the region's 161-lag targets (training-mean imputation); sparse and
  dense arms (plus GPT-2 XL in IFG); total r and within-type r per electrode and lag; exact
  participant-level sign-flip cluster tests and bootstrap bands on sparse minus dense.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent
K = 300


def arms(cfg, ctx, with_prior=True, with_static=True):
    wi = ctx["wi"]
    a = {"SAE": M.sae_band(cfg), "Gemma dense": M.gemma_dense(cfg, wi)}
    if with_prior:
        a["GPT-2 XL"] = M.gpt2xl(cfg, wi)
    if with_static:
        z = np.load(cfg.results("controls", "features", "word_arms.npz"))
        a["Static embedding"] = z["static"][wi].astype(np.float64)
    return a


def part_window(cfg, out: Path) -> None:
    ctx = PS.context(cfg)
    Y, V, subj, meta = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    res, lam = {}, {}
    for name, X in arms(cfg, ctx).items():
        print(f"  {name}", flush=True)
        P, Yr, lams = PS.oof_to(X, K, ctx, Y, V, cfg, exact=True)
        within, between, info = PS.within_between(P, Yr, V, ctx)
        res[name] = dict(total=M.colcorr(P, Yr, V), within=within, between=between)
        lam[name] = lams.tolist()
        print(f"    total r {res[name]['total'].mean():+.4f} | within median "
              f"{np.nanmedian(within):+.4f} | between median {np.nanmedian(between):+.4f}",
              flush=True)
    per = pd.DataFrame({"subject": subj, "ch": meta.ch})
    for a in res:
        for c in ("total", "within", "between"):
            per[f"{a}|{c}"] = res[a][c]
    per.to_csv(out / "a1_window_per_electrode.csv", index=False)
    rows = []
    for i, (a, b) in enumerate([("SAE", "Gemma dense"), ("SAE", "GPT-2 XL"),
                                ("Gemma dense", "GPT-2 XL")]):
        for j, c in enumerate(("within", "between", "total")):
            st, _ = PS.electrode_stat(res[a][c] - res[b][c], subj, cfg.seed + 1400 + 10 * i + j)
            rows.append({"contrast": f"{a} - {b}", "component": c, **st})
    for a in res:
        st, _ = PS.electrode_stat(res[a]["within"], subj, cfg.seed + 1450)
        rows.append({"contrast": f"{a} vs 0", "component": "within", **st})
    T = pd.DataFrame(rows)
    T["n_types"], T["n_tokens"] = info["n_types"], info["n_tokens"]
    T.to_csv(out / "a1_window_stats.csv", index=False)
    json.dump(lam, open(out / "a1_window_lambdas.json", "w"), indent=1)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(T[["contrast", "component", "estimate", "ci95_lo", "ci95_hi", "p_two_sided",
                 "n_electrodes", "n_participants"]].to_string(index=False))


def part_lags(cfg, roi: str, out: Path) -> None:
    ctx = PS.context(cfg)
    Y, V, subj, chans, n_l = PS.roi_lag_targets(cfg, roi)
    n_e = len(subj)
    lags_ms = cfg.lags.centres_s * 1000
    print(f"{roi}: {n_e} electrodes from {len(np.unique(subj))} participants x {n_l} lags, "
          f"{100 * (~V).mean():.2f}% masked", flush=True)
    is_ifg = roi == "IFG"
    store = {}
    for name, X in arms(cfg, ctx, with_prior=is_ifg, with_static=is_ifg).items():
        print(f"  {name}", flush=True)
        P, Yr, lams = PS.oof_to(X, K, ctx, Y, V, cfg, exact=False)
        store[f"{name}|total"] = M.colcorr(P, Yr, V).reshape(n_e, n_l)
        if is_ifg:
            within, between, info = PS.within_between(P, Yr, V, ctx)
            store[f"{name}|within"] = within.reshape(n_e, n_l)
            store[f"{name}|between"] = between.reshape(n_e, n_l)
        store[f"{name}|lambdas"] = lams
        del P, Yr
    np.savez(out / f"profiles_{roi}.npz", subj=subj, chans=chans, lags_ms=lags_ms,
             **{k.replace(" ", "_"): v for k, v in store.items()})
    comps = ["total"] + (["within"] if is_ifg else [])
    others = ["Gemma dense"] + (["GPT-2 XL"] if is_ifg else [])
    clusters, bands = [], []
    for comp in comps:
        for ref in others:
            D = store[f"SAE|{comp}"] - store[f"{ref}|{comp}"]
            C, t_obs, meta = PS.signflip_cluster(D, subj, lags_ms)
            if len(C) == 0:
                C = pd.DataFrame([{"direction": "none", "p_cluster": np.nan}])
            C.insert(0, "contrast", f"SAE - {ref}")
            C.insert(1, "component", comp)
            C.insert(2, "roi", roi)
            for k, v in meta.items():
                C[k] = v
            C["n_electrodes"] = n_e
            clusters.append(C)
            med, lo, hi = PS.boot_bands(D, subj, cfg.seed + 1600)
            bands.append(pd.DataFrame({"roi": roi, "contrast": f"SAE - {ref}",
                                       "component": comp, "lag_ms": lags_ms, "t": t_obs,
                                       "median_diff": med, "band_lo": lo, "band_hi": hi}))
            print(f"  [{comp}] SAE - {ref}: {len(C)} cluster row(s)")
            with pd.option_context("display.width", 200):
                print(C.drop(columns=["roi"]).to_string(index=False))
    pd.concat(clusters).to_csv(out / f"clusters_{roi}.csv", index=False)
    pd.concat(bands).to_csv(out / f"bands_{roi}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("window", "lags"), required=True)
    ap.add_argument("--roi", default="IFG")
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("controls", "within_type_window" if args.part == "window" else "lag_resolved")
    out.mkdir(parents=True, exist_ok=True)
    if args.part == "window":
        part_window(cfg, out)
    else:
        part_lags(cfg, args.roi, out)


if __name__ == "__main__":
    main()
