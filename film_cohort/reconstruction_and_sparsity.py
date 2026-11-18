"""Reconstruction error and sparsity dose-response in the film cohort.

The replication pipeline unchanged (``replication.load_subject``: targets, nuisance,
per-participant ridge on the session-concatenated clock, per-region penalty chosen on the mean
held-out r over that region's contacts of all participants), with these arms:

* D - Gemma-2-2B layer 12, mean pooled;
* R, E - reconstruction and error of the width-16,384, L0 = 82 dictionary, mean pooled;
* S22 ... S445 - the width-16,384 family, max pooled, live on >= 25 of the participant's words.

All z-scored, PCA-300 over each participant's analysed words. Film feature files store only each
film's ever-active dictionary columns; they are placed back into dictionary space by ``sae_ids``
before the live-word filter. IFG is the primary region; every other eligible region is
Holm-corrected across regions.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "9")
os.environ.setdefault("MKL_NUM_THREADS", "9")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from film_cohort.replication import MIN_ACTIVE, MIN_OCC, boot, holm, load_subject, pick
from utils.config import load_config
from word_types.decomposition import decompose

L0S = [22, 41, 82, 176, 445]
ARMS = ["D", "R", "E"] + [f"S{l}" for l in L0S]


def mech_features(feat_dir: Path, W: pd.DataFrame) -> dict:
    """Per-word D, R, E and the five dictionaries for one subject's analysed words."""
    n = len(W)
    films = list(W.groupby("film", sort=False))
    out = {k: np.zeros((n, 2304)) for k in ("D", "R", "E")}
    for film, g in films:
        idx, w = np.asarray(g.index), g["w"].to_numpy()
        with h5py.File(feat_dir / f"btb_{film}.hdf5", "r") as h:
            out["D"][idx] = h["resid_word"][...][w]
            out["R"][idx] = h["recon_word"][...][w]
            out["E"][idx] = h["error_word"][...][w]
    for l in L0S:                                      # one dictionary at a time
        F = np.zeros((n, 16384), np.float32)
        for film, g in films:
            idx, w = np.asarray(g.index), g["w"].to_numpy()
            with h5py.File(feat_dir / f"btb_{film}.hdf5", "r") as h:
                grp = h[f"w16k_l0-{l}"]
                F[np.ix_(idx, grp["sae_ids"][...])] = grp["sae_word_max"][...][w]
        out[f"S{l}"] = F[:, (F > 0).sum(0) >= MIN_ACTIVE].astype(np.float64)
    return out


def fit(sub: str, cfg_path: str, dest: str) -> str:
    cfg = load_config(Path(cfg_path))
    base = cfg.results("film_replication")
    S = load_subject(cfg, sub, base)
    X = mech_features(cfg.results("reconstruction_and_sparsity", "features"), S["W"])
    Y, V, nuis = S["Y"], S["V"], S["nuis"]
    folds = M.folds_for(cfg, S["W"]["t_global"].to_numpy())
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in S["W"]["word"]])
    fid = np.full(len(Y), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    R = {"_roi": S["C"]["roi"].to_numpy(),
         "resid_check": float(np.abs(X["D"] - S["resid"]).max())}
    for name in ARMS:
        Xa = M.zpca(X[name], 300)
        R[f"k_{name}"] = X[name].shape[1]
        P, Yr, _ = M.oof(Xa, nuis, Y, V, folds, exact=True)
        R[f"r_{name}"] = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
        P, Yr, _ = M.oof(Xa, nuis, Y, V, folds, exact=False)
        R[f"a28r_{name}"] = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
        R[f"dec_{name}"] = np.stack([np.stack(decompose(P[li], Yr, V, types, MIN_OCC, fid)[:2])
                                     for li in range(len(P))])          # (L, 2, E)
        print(f"    {sub} {name:5s} k={X[name].shape[1]:5d} best mean r "
              f"{R[f'r_{name}'].mean(1).max():+.4f}", flush=True)
    np.savez(dest, **R)
    return sub


def region(roi: str, R: dict, subs: list, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {s: np.flatnonzero(R[s]["_roi"] == roi) for s in subs}
    cols = {s: c for s, c in cols.items() if len(c)}
    subj = np.concatenate([[s] * len(c) for s, c in cols.items()])

    def stack(key):
        return np.concatenate([R[s][key][..., c] for s, c in cols.items()], axis=-1)

    r = {a: pick(stack(f"r_{a}"))[0] for a in ARMS}
    comp = {a: stack(f"dec_{a}")[pick(stack(f"a28r_{a}"))[1]] for a in ARMS}   # (2, E)
    btw = {a: comp[a][0] for a in ARMS}
    wtn = {a: comp[a][1] for a in ARMS}

    rows = []

    def add(test, x, k, hyp):
        ok = np.isfinite(x)
        m, lo, hi, p, _ = boot(x[ok], subj[ok], seed + k)
        med = pd.Series(x[ok]).groupby(subj[ok]).median()
        rows.append({"roi": roi, "test": test, "median": m, "lo": lo, "hi": hi, "p": p,
                     "hypothesis": hyp, "n_contacts": int(ok.sum()),
                     "n_subjects": int(len(med)), "subjects_positive": int((med > 0).sum())})

    add("A39a E total r", r["E"], 1, "> 0")
    add("A39b within(R) - within(D)", wtn["R"] - wtn["D"], 2, "< 0")
    add("A39c within(E)", wtn["E"], 3, "> 0")
    add("A39d within(R) - within(S)", wtn["R"] - wtn["S82"], 4, "descriptive")
    add("A39d between(R) - between(S)", btw["R"] - btw["S82"], 5, "descriptive")
    add("between(R) - between(D)", btw["R"] - btw["D"], 6, "descriptive")
    add("within(S) - within(D) (A34 reproduction)", wtn["S82"] - wtn["D"], 7, "descriptive")
    add("total r R - D", r["R"] - r["D"], 8, "descriptive")
    add("total r E - S", r["E"] - r["S82"], 9, "descriptive")
    T39 = pd.DataFrame(rows)
    prim = T39.test.str.match(r"A39[abc]")
    T39["p_holm"] = np.nan
    T39.loc[prim, "p_holm"] = holm(T39.loc[prim, "p"].to_numpy())
    a_, b_, c_ = (T39.set_index("test").loc[k] for k in
                  ("A39a E total r", "A39b within(R) - within(D)", "A39c within(E)"))
    if a_.lo > 0 and b_.hi < 0 and c_.lo > 0:           # the pre-specified rule, as in the podcast
        T39["verdict"] = "SUPPORTED"
    elif b_.hi < 0 and not c_.lo > 0:
        T39["verdict"] = "PARTIAL (reconstruction loses context; error does not recover it)"
    elif not b_.hi < 0:
        T39["verdict"] = "NOT SUPPORTED (A39b interval includes 0)"
    else:
        T39["verdict"] = "NOT SUPPORTED (A39a fails)"
    for a in ARMS:
        T39[f"mean_r_{a}"] = float(r[a].mean())

    x = np.log2(np.array(L0S, float))
    xc = x - x.mean()
    srows = []
    for c, arr in (("d_sae_dense", np.stack([r[f"S{l}"] - r["D"] for l in L0S])),
                   ("d_within", np.stack([wtn[f"S{l}"] - wtn["D"] for l in L0S])),
                   ("d_between", np.stack([btw[f"S{l}"] - btw["D"] for l in L0S]))):
        slope = (xc[:, None] * (arr - np.nanmean(arr, 0))).sum(0) / (xc ** 2).sum()
        ok = np.isfinite(slope)
        m, lo, hi, p, _ = boot(slope[ok], subj[ok], seed + 20 + len(srows))
        med = pd.Series(slope[ok]).groupby(subj[ok]).median()
        srows.append({"roi": roi, "family": "16k", "component": c,
                      "predicted": "< 0" if c == "d_between" else "> 0", "median": m,
                      "lo": lo, "hi": hi, "p": p, "subjects_positive": int((med > 0).sum()),
                      "n_subjects": int(len(med)),
                      **{f"median_{c}_L0_{l}": float(np.nanmedian(arr[i]))
                         for i, l in enumerate(L0S)}})
    T40 = pd.DataFrame(srows)
    T40["p_holm"] = holm(T40.p.to_numpy())
    per = {"subj": subj, **{f"{a}__r": r[a] for a in ARMS},
           **{f"{a}__within": wtn[a] for a in ARMS}, **{f"{a}__between": btw[a] for a in ARMS}}
    return T39, T40, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg.results("film_replication")
    out = cfg.results("reconstruction_and_sparsity", "btb")
    (out / "fits").mkdir(parents=True, exist_ok=True)
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: int(s.split("_")[1]))
    todo = sorted((s for s in subs if not (out / "fits" / f"{s}.npz").exists()),
                  key=lambda s: -(base / "targets" / f"{s}.npz").stat().st_size)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fit, s, str(args.config), str(out / "fits" / f"{s}.npz"))
                for s in todo]
        for fu in as_completed(futs):
            print("done", fu.result(), flush=True)
    R = {s: dict(np.load(out / "fits" / f"{s}.npz", allow_pickle=True)) for s in subs}
    worst = max(float(R[s]["resid_check"]) for s in subs)
    print(f"dense stream vs A38 features: max |diff| {worst:.2e}")

    gates = json.loads((base / "results" / "a38_gates.json").read_text())
    rois = ["IFG"] + [r for r in gates["eligible_regions"] if r != "IFG"]
    T39, T40 = [], []
    for j, roi in enumerate(rois):
        a, b, per = region(roi, R, subs, cfg.seed + 3900 + 100 * j)
        T39.append(a)
        T40.append(b)
        np.savez(out / f"a39_a40_contacts_{roi}.npz", **per)          # per-contact values
    T39, T40 = pd.concat(T39, ignore_index=True), pd.concat(T40, ignore_index=True)
    for T, key in ((T39, "test"), (T40, "component")):
        T["p_holm_regions"] = np.nan
        sec = T.roi != "IFG"
        for k in T.loc[sec, key].unique():
            m = sec & (T[key] == k)
            T.loc[m, "p_holm_regions"] = holm(T.loc[m, "p"].to_numpy())
    T39.to_csv(out / "a39_btb.csv", index=False)
    T40.to_csv(out / "a40_btb.csv", index=False)

    # reproduction of the replication's IFG result (D = Gemma-300, S82 = SAE-300)
    A38 = pd.read_csv(base / "results" / "a38_regions.csv").set_index("roi")
    i39 = T39[T39.roi == "IFG"].iloc[0]
    print(f"IFG reproduction: D {i39.mean_r_D:+.4f} vs A38 Gemma-300 "
          f"{A38.loc['IFG', 'mean_r_gemma300']:+.4f}; S82 {i39.mean_r_S82:+.4f} vs A38 "
          f"SAE-300 {A38.loc['IFG', 'mean_r_sae300']:+.4f}")
    fmt = {"float_format": lambda v: f"{v:+.4f}"}
    print("\nA39 (IFG primary; others secondary)\n" + T39[
        ["roi", "test", "median", "lo", "hi", "p", "p_holm", "p_holm_regions",
         "subjects_positive", "n_subjects"]].to_string(index=False, **fmt))
    print("\nA39 verdicts:", T39.groupby("roi", sort=False).verdict.first().to_dict())
    print("\nA40 slopes on log2(L0), 16k family\n" + T40[
        ["roi", "component", "predicted", "median", "lo", "hi", "p", "p_holm",
         "p_holm_regions", "subjects_positive", "n_subjects"]].to_string(index=False, **fmt))


if __name__ == "__main__":
    main()
