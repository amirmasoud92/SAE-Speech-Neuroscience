"""H1 and H2 in the film cohort.

Per participant:

* **H1.** Banded ridge in covariance form (``covariance_ridge``), with bands sae (z-PCA-300 of
  the live sparse features), surprisal, lexical, acoustic (the released per-word mel-spectral
  vectors) and scene (visual scalars, speaker change and per-session intercepts). Targets are
  the cleaned responses at the 21 lags from 0 to 500 ms; the output is the unique sparse
  variance per contact and lag.
* **H2.** ``basis_comparison_fits.analysis_sci`` in covariance form: the live sparse band, a
  density-matched random dictionary and a column-shuffled control, each contact read at the
  peak lag of its unique sparse variance.

The statistics are the primary cohort's functions (``statistics.mixed_model``,
``statistics.cluster_bootstrap``, ``statistics.admissible_sci`` and
``semantic_concentration._report``). The solver-agreement check must have passed. Writes
``film_h1_h2/``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from film_cohort.replication import load_subject
from utils.config import load_config
from encoding.covariance_ridge import delta_r2_primal, semantic_concentration_cov

log = logging.getLogger("a41")
LANG, CTRL = ("IFG", "STG", "MTG", "TMP"), ("EAC", "PRC", "MFG")


def random_dictionary_tiesafe(resid: np.ndarray, densities: np.ndarray,
                              rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """``banded_ridge.random_dictionary_features`` with float32 threshold ties resolved.

    Same draw of the random encoder from the same generator state, same threshold for
    every feature. Only a feature whose float32 projection ties at its threshold, so that
    exact density matching is impossible, is recomputed from the same weights in float64.
    Where there is no tie the result is identical to the podcast function.
    """
    n_words, d_model = resid.shape
    densities = np.asarray(densities).astype(int)
    W = rng.standard_normal((d_model, densities.size)).astype(np.float32) / np.sqrt(d_model)
    pre = resid @ W
    feats = np.zeros_like(pre, dtype=np.float32)
    fixed = []
    for j, n_j in enumerate(densities):
        if n_j <= 0:
            continue
        v = pre[:, j]
        thr = np.partition(v, n_words - n_j - 1)[n_words - n_j - 1]
        col = np.maximum(v - thr, 0.0)
        if int((col > 0).sum()) != n_j:
            v64 = resid.astype(np.float64) @ W[:, j].astype(np.float64)
            t64 = np.partition(v64, n_words - n_j - 1)[n_words - n_j - 1]
            col = np.maximum(v64 - t64, 0.0).astype(np.float32)
            fixed.append(j)
        feats[:, j] = col
    if not np.array_equal((feats > 0).sum(0), densities):
        raise AssertionError("density matching failed even after float64 tie resolution")
    return feats, fixed


def fit_participant(cfg, sub: str, base: Path, dest: Path) -> None:
    S = load_subject(cfg, sub, base)
    z = np.load(base / "targets" / f"{sub}.npz", allow_pickle=True)
    Y = z["Y"].astype(np.float32)                        # (n, contacts, 21), cleaned
    V = z["V"].astype(bool)
    n, n_ch, n_lags = Y.shape
    onsets = S["W"]["t_global"].to_numpy()
    P = S["parts"]
    bands = [("sae", M.zpca(S["sae"], 300).astype(np.float32))] + [
        (b, np.asarray(P[b], np.float32)) for b in ("surprisal", "acoustic", "lexical",
                                                    "scene")]      # podcast order + scene
    log.info("%s: %d words, %d contacts, bands %s", sub, n, n_ch,
             {b: x.shape[1] for b, x in bands})
    # the exact float64 estimator, covariance form
    vp = delta_r2_primal(cfg, bands, Y.reshape(n, -1), onsets, "sae",
                         valid=V.reshape(n, -1), label=sub, form="cov")
    d = vp["delta_sae"].reshape(n_ch, n_lags)
    peak = np.nanargmax(np.where(np.isfinite(d), d, -np.inf), axis=1)
    log.info("%s H1: peak delta_sae median %.5f", sub, float(np.median(d.max(1))))

    # ---- H2 (basis_comparison_fits.analysis_sci, covariance form) -----------------------
    r_ = np.arange(n_ch)
    Yp, Vp = np.ascontiguousarray(Y[:, r_, peak]), np.ascontiguousarray(V[:, r_, peak])
    rng = np.random.default_rng(cfg.seed)
    real = np.asarray(S["sae"], np.float32)
    dens = (real > 0).sum(0)
    rand, fixed = random_dictionary_tiesafe(np.asarray(S["resid"], np.float32), dens, rng)
    if fixed:
        log.info("%s: %d random-dictionary features resolved in float64", sub,
                 len(fixed))
    shuf = real.copy()
    for j in range(shuf.shape[1]):
        rng.shuffle(shuf[:, j])
    sci = {}
    for name, Xb in (("real", real), ("random_dict", rand), ("shuffled", shuf)):
        res = semantic_concentration_cov(cfg, Xb, Yp, onsets, valid=Vp)
        for k, v in res.items():
            full = np.full((n_ch, n_lags), np.nan, np.float32)
            full[r_, peak] = v
            sci[f"{name}__{k}"] = full
        log.info("%s %s: r2_all median %.5f", sub, name, float(np.nanmedian(res["r2_all"])))
    np.savez_compressed(dest, delta_sae=d, full=vp["full"].reshape(n_ch, n_lags),
                        roi=S["C"]["roi"].to_numpy().astype(str),
                        contact=S["C"].iloc[:, 0].astype(str).to_numpy(),
                        peak=peak, n_sae=real.shape[1], n_words=n,
                        achieved_l0=float((real > 0).sum(1).mean()), **sci)


def h1_stats(cfg, R: dict, out: Path) -> dict:
    from inference.statistics import cluster_bootstrap, mixed_model
    rows = []
    for sub, z in R.items():
        for i, roi in enumerate(z["roi"].astype(str)):
            rows.append({"subject": sub, "roi": roi, "obs_peak": float(z["delta_sae"][i].max()),
                         "peak_lag_ms": 25.0 * int(z["peak"][i])})
    meta = pd.DataFrame(rows)
    res = {"mixed_model_roi": mixed_model(meta, outcome="obs_peak", fixed="roi")}
    sub_ = meta[meta.roi.isin(LANG + CTRL)].copy()
    sub_["network"] = np.where(sub_.roi.isin(LANG), "language", "control")
    res["mixed_model_network"] = mixed_model(sub_, outcome="obs_peak", fixed="network")
    boot = []
    for roi, g in meta.groupby("roi"):
        if g.subject.nunique() < 3:
            continue
        pt, lo, hi = cluster_bootstrap(g.obs_peak.to_numpy(), g.subject.to_numpy(),
                                       n_resamples=int(cfg.stats["bootstrap"]["n_resamples"]),
                                       ci=float(cfg.stats["bootstrap"]["ci"]), seed=cfg.seed)
        boot.append({"roi": roi, "network": "language" if roi in LANG else "control",
                     "n": len(g), "n_subjects": int(g.subject.nunique()),
                     "mean_peak": pt, "ci_lo": lo, "ci_hi": hi})
    meta.to_csv(out / "h1_film_peaks.csv", index=False)
    pd.DataFrame(boot).sort_values("mean_peak", ascending=False).to_csv(
        out / "h1_film_bootstrap.csv", index=False)
    (out / "h1_film_models.json").write_text(json.dumps(res, indent=2, default=str))
    net = res["mixed_model_network"]
    b = float(net["params"]["C(network)[T.language]"])
    p = float(net["pvalues"]["C(network)[T.language]"])
    verdict = "SUPPORTED" if (b > 0 and p < 0.05) else "NOT SUPPORTED"
    print(f"H1 film: language vs control beta {b:+.3f} SD, P = {p:.3g} "
          f"({(sub_.network == 'language').sum()} language, "
          f"{(sub_.network == 'control').sum()} control contacts) -> {verdict}")
    print(pd.DataFrame(boot).sort_values("mean_peak", ascending=False).to_string(index=False))
    return {"beta": b, "p": p, "verdict": verdict}


def h2_stats(cfg, R: dict, out: Path) -> pd.DataFrame:
    from interpretability.semantic_concentration import _report
    from inference.statistics import admissible_sci
    frames = []
    for sub, z in R.items():
        d = z["delta_sae"]
        adm = admissible_sci(z["real__r2_all"], z["random_dict__r2_all"], z["real__sci_20"],
                             z["random_dict__sci_20"], d)
        li, rows = adm["lag_idx"], np.arange(d.shape[0])
        df = pd.DataFrame({"subject": sub, "roi": z["roi"].astype(str),
                           "admissible": adm["admissible"], "peak_lag_ms": 25.0 * li})
        for arm in ("real", "random_dict", "shuffled"):
            num, den = z[f"{arm}__r2_top20"][rows, li], z[f"{arm}__r2_all"][rows, li]
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"share_{arm}"] = np.where(den > 0, num / den, np.nan)
            df[f"r2_all_{arm}"], df[f"r2_top_{arm}"] = den, num
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(out / "sci_film_all_contacts_k20.csv", index=False)
    df = allc[allc.admissible].drop(columns="admissible")
    df.to_csv(out / "sci_film_shares_k20.csv", index=False)
    lg = logging.getLogger("a41.h2")
    rows = []
    for scope, sel in (("IFG", df.roi == "IFG"), ("language", df.roi.isin(LANG))):
        print(f"\n--- H2 film, {scope} (tau 0.5, k 20) ---")
        r = _report(lg, df[sel], 0.5, cfg.seed)
        n_adm = int(sel.sum())
        n_all = int((allc.roi == "IFG").sum() if scope == "IFG"
                    else allc.roi.isin(LANG).sum())
        sign_ok = r.get("p_sign_one_sided", 1.0) < 0.05
        pool_ok = r.get("diff_ci_lo", -1.0) > 0
        verdict = ("SUPPORTED" if sign_ok and pool_ok else
                   "NOT SUPPORTED" if not sign_ok and not pool_ok else "MIXED")
        rows.append({"scope": scope, "tau": 0.5, "k": 20, "n_admissible": n_adm,
                     "n_contacts": n_all, **r, "verdict": verdict})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    T = pd.DataFrame(rows)
    T.to_csv(out / "h2_film_summary_k20.csv", index=False)
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    out = cfg.results("film_h1_h2")
    # H1 needs the float64 solver check; H2 the H2 rows of the solver check
    g64 = out / "solver_gate_float64.json"
    g32 = json.loads((out / "solver_gate.json").read_text())
    h2_ok = all(r["pass"] for r in g32["rows"] if r["quantity"].startswith("H2"))
    gcov = out / "solver_gate_cov.json"
    gsci = out / "solver_gate_sci_cov.json"
    if not (g64.exists() and json.loads(g64.read_text())["pass"] and h2_ok
            and gcov.exists() and json.loads(gcov.read_text())["pass"]
            and gsci.exists() and json.loads(gsci.read_text())["pass"]):
        raise SystemExit("solver check has not passed - no film fit is run")
    base = cfg.results("film_replication")
    (out / "fits").mkdir(parents=True, exist_ok=True)
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: (base / "targets" / f"{s}.npz").stat().st_size)
    if args.only:
        subs = [s for s in subs if s in args.only]
    for sub in subs:
        f = out / "fits" / f"{sub}.npz"
        if not f.exists():
            fit_participant(cfg, sub, base, f)
            print("done", sub, flush=True)
    if args.only:
        return
    R = {s: dict(np.load(out / "fits" / f"{s}.npz", allow_pickle=True)) for s in subs}
    h1 = h1_stats(cfg, R, out)
    h2 = h2_stats(cfg, R, out)
    (out / "a41_verdicts.json").write_text(json.dumps(
        {"H1": h1, "H2": h2[["scope", "verdict"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
