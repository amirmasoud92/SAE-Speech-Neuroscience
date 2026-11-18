"""Timing and neighbouring-word controls.

Two bands are added to the nuisance projection inside every fold:

- **timing:** ``feature_extraction.timing_features.timing_features``, 90 stimulus-only columns
  (log gaps to the four preceding and four following word onsets, durations of words n-1, n
  and n+1, local word rate, and their squares and pairwise products), z-scored over words;
- **neighbour content:** Gemma-2-2B static input embeddings of words n-2, n-1 and n+1 in
  chronological order, each projected on 50 principal components fitted on the training
  fold's own-word embeddings.

``--part window`` reruns the IFG 300 vs 300 contrast, the word-type decomposition and the
85 vs 85 classical comparison; ``--part lags --roi R`` reruns the lag-resolved difference.
Writes ``controls/timing_neighbour/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import neural_targets as D
from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config
from controls.sign_flip_and_electrode_tests import signflip
from feature_extraction.timing_features import timing_features

HERE = Path(__file__).parent
N_PC = 50
OFFSETS = (-2, -1, 1)


_ORIG_FOLD_PCA = PS.fold_pca


def cached_fold_pca(cache_dir: Path):
    """Exact ``training_only.fold_pca``, memoised on disk by (input fingerprint, training
    rows, k): each fold basis is computed once and shared by every analysis here. Results are
    identical to the uncached function."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def f(Xraw, tr, k):
        X = np.ascontiguousarray(np.asarray(Xraw, dtype=np.float64))
        key = hashlib.sha1(np.asarray(X.shape).tobytes() + X[:: max(1, len(X) // 64)].tobytes()
                           + np.asarray(tr).tobytes() + str(k).encode()).hexdigest()[:20]
        fn = cache_dir / f"{key}.npy"
        if fn.exists():
            return np.load(fn)
        out = _ORIG_FOLD_PCA(X, tr, k)
        tmp = fn.with_name(fn.stem + f".{os.getpid()}.tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, fn)                      # atomic: parallel processes share the cache
        return out
    return f


def c6_context(cfg):
    """The training-only context, with the extended nuisance matrix per outer fold."""
    ctx = PS.context(cfg)
    words = D.load_words(cfg)                                  # all 5,136, row order
    on, en = words.start.to_numpy(float), words.end.to_numpy(float)
    T = timing_features(on, np.zeros(len(words)), en - on, quadratic=True)
    T = np.asarray(T, np.float64)
    T = (T - T.mean(0)) / np.where(T.std(0) > 1e-12, T.std(0), 1.0)
    static = np.load(cfg.results("controls", "features", "word_arms.npz"))["static"].astype(np.float64)
    order = np.lexsort((words.word_idx.to_numpy(), on))       # chronological
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    nb = {}
    for k in OFFSETS:
        q = pos + k
        nb[k] = np.where((q >= 0) & (q < len(order)), order[np.clip(q, 0, len(order) - 1)], -1)
    wi = ctx["wi"]
    folds_n = []
    for tr, _ in ctx["folds"]:
        Xtr = static[wi[tr]]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        # top-50 principal axes of the training rows: eigenvectors of the covariance
        # (identical to the leading right singular vectors, and far cheaper here)
        Z = (Xtr - mu) / sd
        ev, U = np.linalg.eigh(Z.T @ Z)
        B = U[:, ::-1][:, :N_PC]
        cols = []
        for k in OFFSETS:
            idx = nb[k][wi]
            Z = np.zeros((len(wi), N_PC))
            ok = idx >= 0
            Z[ok] = ((static[idx[ok]] - mu) / sd) @ B
            cols.append(Z)
        folds_n.append(np.column_stack([ctx["nuis"], T[wi]] + cols))
    ctx["nuis_fold"] = folds_n
    info = {"timing_columns": int(T.shape[1]), "neighbour_columns": N_PC * len(OFFSETS),
            "base_nuisance_columns": int(ctx["nuis"].shape[1]),
            "total_nuisance_columns": int(folds_n[0].shape[1]),
            "neighbour_missing": {str(k): int((nb[k][wi] < 0).sum()) for k in OFFSETS}}
    return ctx, info


def part_window(cfg, out):
    ctx, info = c6_context(cfg)
    json.dump(info, open(out / "c6_bands.json", "w"), indent=1)
    print(json.dumps(info), flush=True)
    Y, V, subj, meta = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    res = {}
    for name, X, k in (("SAE", M.sae_band(cfg), 300), ("Gemma dense", M.gemma_dense(cfg, ctx["wi"]), 300)):
        P, Yr, lams = PS.oof_to(X, k, ctx, Y, V, cfg, exact=True)
        w, b, _ = PS.within_between(P, Yr, V, ctx)
        res[name] = {"total": M.colcorr(P, Yr, V), "within": w, "between": b}
        print(f"  {name}: total {res[name]['total'].mean():+.4f}", flush=True)
    from feature_extraction.feature_bands import classical_features
    C, _ = classical_features(cfg, ctx["wi"], ctx["words"]["word"])
    for name, X, k in (("classical-85", np.asarray(C, np.float64), None), ("SAE-85", M.sae_band(cfg), 85)):
        P, Yr, _ = PS.oof_to(X, k, ctx, Y, V, cfg, exact=True)
        res[name] = {"total": M.colcorr(P, Yr, V)}
        print(f"  {name}: total {res[name]['total'].mean():+.4f}", flush=True)
    per = pd.DataFrame({"subject": subj, "ch": meta.ch})
    for a in res:
        for c_, v in res[a].items():
            per[f"{a}|{c_}"] = v
    per.to_csv(out / "c6_window_per_electrode.csv", index=False)

    # with and without: "without" is the existing training-only output of the same code
    E0 = pd.read_csv(cfg.results("controls", "within_type_window", "a1_window_per_electrode.csv"))
    W0 = {a: pd.read_csv(cfg.results("controls", "window_arms", f"TO_{a}.csv")).r.to_numpy()
          for a in ("classical-85", "SAE-85")}
    assert (E0.ch.to_numpy() == per.ch.to_numpy()).all()
    quantities = {
        "IFG 300 vs 300 total r: SAE - dense": (
            E0["SAE|total"] - E0["Gemma dense|total"], per["SAE|total"] - per["Gemma dense|total"]),
        "C2 within-type gap": (
            E0["SAE|within"] - E0["Gemma dense|within"], per["SAE|within"] - per["Gemma dense|within"]),
        "C2 between-type gap": (
            E0["SAE|between"] - E0["Gemma dense|between"], per["SAE|between"] - per["Gemma dense|between"]),
        "C2 interaction (within gap - between gap)": (
            (E0["SAE|within"] - E0["Gemma dense|within"]) - (E0["SAE|between"] - E0["Gemma dense|between"]),
            (per["SAE|within"] - per["Gemma dense|within"]) - (per["SAE|between"] - per["Gemma dense|between"])),
        "LLM vs classical, 85 vs 85: SAE - classical": (
            pd.Series(W0["SAE-85"] - W0["classical-85"]), per["SAE-85|total"] - per["classical-85|total"]),
    }
    rows = []
    for i, (lab, (x0, x1)) in enumerate(quantities.items()):
        for cond, x in (("without controls", x0), ("with timing + neighbour controls", x1)):
            x = np.asarray(x, float)
            st, _ = PS.electrode_stat(x, subj, cfg.seed + 4500 + i)
            sf = signflip(x, subj, "t")
            rows.append({"quantity": lab, "condition": cond, "median": st["estimate"],
                         "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                         "p_bootstrap": st["p_two_sided"], "p_signflip_exact": sf["p_exact"],
                         "participants_negative": sf["n_participants_negative"],
                         "n_electrodes": st["n_electrodes"], "n_participants": st["n_participants"]})
    T = pd.DataFrame(rows)
    T.to_csv(out / "c6_window.csv", index=False)
    with pd.option_context("display.width", 220):
        print(T.to_string(index=False))


def part_lags(cfg, roi, out):
    ctx, _ = c6_context(cfg)
    Y, V, subj, chans, n_l = PS.roi_lag_targets(cfg, roi)
    n_e = len(subj)
    lags_ms = cfg.lags.centres_s * 1000
    store = {}
    for name, X in (("SAE", M.sae_band(cfg)), ("Gemma dense", M.gemma_dense(cfg, ctx["wi"]))):
        P, Yr, _ = PS.oof_to(X, 300, ctx, Y, V, cfg, exact=False)
        store[name] = M.colcorr(P, Yr, V).reshape(n_e, n_l)
        del P, Yr
        print(f"  {roi} {name} done", flush=True)
    D_ = store["SAE"] - store["Gemma dense"]
    C, t_obs, meta = PS.signflip_cluster(D_, subj, lags_ms)
    if len(C) == 0:
        C = pd.DataFrame([{"direction": "none", "p_cluster": np.nan}])
    C.insert(0, "roi", roi)
    for k_, v in meta.items():
        C[k_] = v
    C["n_electrodes"] = n_e
    C["condition"] = "with timing + neighbour controls"
    med, lo, hi = PS.boot_bands(D_, subj, cfg.seed + 1600)
    np.savez(out / f"c6_profiles_{roi}.npz", subj=subj, lags_ms=lags_ms,
             SAE=store["SAE"], dense=store["Gemma dense"])
    C.to_csv(out / f"c6_clusters_{roi}.csv", index=False)
    pd.DataFrame({"roi": roi, "lag_ms": lags_ms, "t": t_obs, "median_diff": med, "band_lo": lo,
                  "band_hi": hi}).to_csv(out / f"c6_bands_{roi}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(C.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("window", "lags"), required=True)
    ap.add_argument("--roi", default="IFG")
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("controls", "timing_neighbour")
    out.mkdir(parents=True, exist_ok=True)
    PS.fold_pca = cached_fold_pca(out / "pca_cache")
    if args.part == "window":
        part_window(cfg, out)
    else:
        part_lags(cfg, args.roi, out)


if __name__ == "__main__":
    main()
