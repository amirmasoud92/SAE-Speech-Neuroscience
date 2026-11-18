"""Covariance-form and kernel-form solvers must agree on podcast data.

Run before any film-cohort fit. On one podcast participant, with identical inputs:

* unique sparse variance from ``banded_ridge.fit_banded`` (kernel form) against
  ``covariance_ridge.delta_r2_primal``; the design is the film estimator's (sparse band at 300
  components, surprisal, acoustic, lexical), with targets at the 21 lags from 0 to 500 ms;
* R2(all) and R2(top-20) from ``banded_ridge.semantic_concentration`` against
  ``covariance_ridge.semantic_concentration_primal``, fitting the full live sparse band at each
  electrode's peak lag.

Pass: correlation >= 0.999 and median |difference| < 1e-4 for unique variance and R2(all);
R2(top-20) uses identical code in both and must agree exactly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from encoding.banded_ridge import fit_banded, semantic_concentration
from feature_extraction.feature_bands import Design
from encoding.covariance_ridge import (delta_r2_primal, semantic_concentration_cov,
                                       semantic_concentration_primal)
from encoding.fit_encoding_model import analysed_words, stage_design, stage_targets
from preprocessing import neural_targets as D

LAGS = slice(80, 101)            # centres 0.000 ... 0.500 s


def compare(a, b, name):
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    med = float(np.median(np.abs(a[ok] - b[ok])))
    mx = float(np.max(np.abs(a[ok] - b[ok])))
    return {"quantity": name, "n": int(ok.sum()), "corr": r, "median_abs_diff": med,
            "max_abs_diff": mx, "pass": bool(r >= 0.999 and med < 1e-4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--dtype", choices=("float32", "float64", "cov", "sci_cov", "cov_recheck"), default="float32",
                    help="float64: kernel and primal forms in "
                         "float64, H1 only, all folds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    out = cfg.results("film_h1_h2")
    out.mkdir(parents=True, exist_ok=True)

    words = analysed_words(cfg)
    des = stage_design(cfg, words)
    table = D.load_electrode_table(cfg)
    Y, ch, V = stage_targets(cfg, args.sub, words, table)
    onsets = words["start"].to_numpy()
    n_w, n_ch, _ = Y.shape
    Yl = np.ascontiguousarray(Y[:, :, LAGS]).reshape(n_w, -1)
    Vl = None if V is None else np.ascontiguousarray(V[:, :, LAGS]).reshape(n_w, -1)

    # ---- H1: identical SAE-300 design, kernel versus primal ----------------------------
    sae300 = M.zpca(des.band("sae"), 300).astype(np.float32)
    bands = [("sae", sae300)] + [(b, des.band(b)) for b in ("surprisal", "acoustic",
                                                          "lexical")]
    X = np.concatenate([x for _, x in bands], 1).astype(np.float32)
    off = np.cumsum([0] + [x.shape[1] for _, x in bands])
    d300 = Design(X, {b: slice(int(off[i]), int(off[i + 1])) for i, (b, _) in
                      enumerate(bands)}, {b: np.arange(x.shape[1]) for b, x in bands})
    if args.dtype == "cov_recheck":     # torch eigh, against the saved reference
        ref = np.load(out / f"solver_validation_{args.sub}_cov_arrays.npz")
        c = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="cov", form="cov")
        rows = [compare(ref["himalaya"], c["delta_sae"],
                        "H1 delta_sae, covariance form (torch eigh) vs himalaya float64"),
                compare(ref["cov"], c["delta_sae"],
                        "H1 delta_sae, torch eigh vs numpy eigh")]
        T = pd.DataFrame(rows)
        T.to_csv(out / f"solver_validation_{args.sub}_cov_recheck.csv", index=False)
        print(T.to_string(index=False))
        ok = bool(T["pass"].all())
        (out / "solver_gate_cov.json").write_text(json.dumps(
            {"sub": args.sub, "pass": ok, "rows": rows, "eigh": "torch"}, indent=2))
        print("SOLVER GATE (covariance form, torch eigh):", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if args.dtype == "cov":             # float64 covariance form
        c = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="cov", form="cov")
        h = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="himalaya64",
                            form="primal", dtype=np.float64, backend="torch")
        rows = [compare(h["delta_sae"], c["delta_sae"],
                        "H1 delta_sae, covariance form vs himalaya float64"),
                compare(h["full"], c["full"], "H1 R2(full), covariance form vs himalaya float64")]
        T = pd.DataFrame(rows)
        T.to_csv(out / f"solver_validation_{args.sub}_cov.csv", index=False)
        np.savez(out / f"solver_validation_{args.sub}_cov_arrays.npz", himalaya=h["delta_sae"],
                 cov=c["delta_sae"])
        print(T.to_string(index=False))
        ok = bool(T["pass"].all())
        (out / "solver_gate_cov.json").write_text(json.dumps(
            {"sub": args.sub, "pass": ok, "rows": rows}, indent=2))
        print("SOLVER GATE (covariance form):", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if args.dtype == "float64":
        k64 = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="kernel64",
                              form="kernel", dtype=np.float64)
        p64 = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="primal64",
                              form="primal", dtype=np.float64)
        rows = [compare(k64["delta_sae"], p64["delta_sae"],
                        "H1 delta_sae (SAE-300, 0-500 ms), float64"),
                compare(k64["full"], p64["full"], "H1 R2(full), float64")]
        T = pd.DataFrame(rows)
        T.to_csv(out / f"solver_validation_{args.sub}_float64.csv", index=False)
        print(T.to_string(index=False))
        ok = bool(T["pass"].all())
        (out / "solver_gate_float64.json").write_text(json.dumps(
            {"sub": args.sub, "pass": ok, "rows": rows}, indent=2))
        print("SOLVER GATE (float64):", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if args.dtype == "sci_cov":
        kf = None
    else:
        kf = fit_banded(cfg, d300, Yl, onsets, label="kernel full", valid=Vl)
    if kf is not None:
        kr = fit_banded(cfg, d300.without("sae"), Yl, onsets, label="kernel minus-sae",
                        valid=Vl)
        k_delta = kf.r2() - kr.r2()
        p = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label="primal")
        rows = [compare(k_delta, p["delta_sae"], "H1 delta_sae (SAE-300, 0-500 ms)"),
                compare(kf.r2(), p["full"], "H1 R2(full)")]

    # ---- H2: full live band at the delta_sae peak lag ----------------------------------
    vp = np.load(cfg.results("encoding", f"{args.sub}_variance_partition.npz"),
                 allow_pickle=True)
    d = vp["delta_sae"]
    peak = np.nanargmax(np.where(np.isfinite(d), d, -np.inf), axis=1)
    r_ = np.arange(n_ch)
    Yp = np.ascontiguousarray(Y[:, r_, peak])
    Vp = None if V is None else np.ascontiguousarray(V[:, r_, peak])
    ks = semantic_concentration(cfg, des.only(["sae"]), Yp, onsets, band="sae", valid=Vp,
                                top_k=[20])
    if args.dtype == "sci_cov":         # concentration fit in covariance form
        cs = semantic_concentration_cov(cfg, des.band("sae"), Yp, onsets, top_k=[20],
                                        valid=Vp)
        rows = [compare(ks["r2_all"], cs["r2_all"], "H2 R2(all), covariance form vs kernel"),
                compare(ks["r2_top20"], cs["r2_top20"], "H2 R2(top-20), covariance form vs kernel")]
        T = pd.DataFrame(rows)
        T.to_csv(out / f"solver_validation_{args.sub}_sci_cov.csv", index=False)
        print(T.to_string(index=False))
        ok = bool(T["pass"].all())
        (out / "solver_gate_sci_cov.json").write_text(json.dumps(
            {"sub": args.sub, "pass": ok, "rows": rows}, indent=2))
        print("SOLVER GATE (SCI covariance form):", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    ps = semantic_concentration_primal(cfg, des.band("sae"), Yp, onsets, top_k=[20],
                                       valid=Vp)
    rows += [compare(ks["r2_all"], ps["r2_all"], "H2 R2(all), full band"),
             compare(ks["r2_top20"], ps["r2_top20"], "H2 R2(top-20)")]
    T = pd.DataFrame(rows)
    T.to_csv(out / f"solver_validation_{args.sub}.csv", index=False)
    print(T.to_string(index=False))
    ok = bool(T["pass"].all())
    (out / "solver_gate.json").write_text(json.dumps(
        {"sub": args.sub, "pass": ok, "rows": rows}, indent=2))
    print("SOLVER GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
