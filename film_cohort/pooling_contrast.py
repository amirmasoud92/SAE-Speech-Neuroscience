"""Pooling against re-coding in the film cohort.

The machinery of ``film_cohort/reconstruction_and_sparsity.py``: z-PCA-300, exact masked ridge for
total r, and the within-fold word-type decomposition with imputed cells, with the penalty chosen
per region on total r. S_mean is the mean-pooled code (``film_cohort/mean_pooled_features.py``),
kept where live on >= 25 of the participant's words.

Contrasts per region (IFG primary; other regions Holm-corrected): pooling,
within(S_mean) - within(S); basis, within(R) - within(S_mean); within(S_mean) - within(D); and
total r, S_mean - D.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "9")
os.environ.setdefault("MKL_NUM_THREADS", "9")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from film_cohort.replication import MIN_ACTIVE, MIN_OCC, boot, holm, load_subject, pick
from utils.config import load_config
from word_types.decomposition import decompose


def smean(feat_dir: Path, W: pd.DataFrame) -> np.ndarray:
    n = len(W)
    F = np.zeros((n, 16384), np.float32)
    for film, g in W.groupby("film", sort=False):
        idx, w = np.asarray(g.index), g["w"].to_numpy()
        with h5py.File(feat_dir / f"btb_{film}.hdf5", "r") as h:
            grp = h["w16k_l0-82_mean"]
            F[np.ix_(idx, grp["sae_ids"][...])] = grp["sae_word_mean"][...][w]
    return F[:, (F > 0).sum(0) >= MIN_ACTIVE].astype(np.float64)


def fit(sub: str, cfg_path: str, dest: str) -> str:
    cfg = load_config(Path(cfg_path))
    S = load_subject(cfg, sub, cfg.results("film_replication"))
    X = M.zpca(smean(cfg.results("reconstruction_and_sparsity", "features"), S["W"]), 300)
    Y, V, nuis = S["Y"], S["V"], S["nuis"]
    folds = M.folds_for(cfg, S["W"]["t_global"].to_numpy())
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in S["W"]["word"]])
    fid = np.full(len(Y), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=True)
    r = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
    P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=False)
    a28r = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
    dec = np.stack([np.stack(decompose(P[li], Yr, V, types, MIN_OCC, fid)[:2])
                    for li in range(len(P))])
    np.savez(dest, r_Smean=r, a28r_Smean=a28r, dec_Smean=dec)
    print(f"done {sub}", flush=True)
    return sub


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = load_config(cfg_path)
    base = cfg.results("reconstruction_and_sparsity", "btb")
    (base / "fits_a47").mkdir(exist_ok=True)
    subs = sorted((p.stem for p in (base / "fits").glob("sub_*.npz")),
                  key=lambda s: -(cfg.results("film_replication", "targets") / f"{s}.npz").stat().st_size)
    todo = [s for s in subs if not (base / "fits_a47" / f"{s}.npz").exists()]
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(fit, todo, [str(cfg_path)] * len(todo),
                    [str(base / "fits_a47" / f"{s}.npz") for s in todo]))
    R = {}
    for s in subs:
        R[s] = dict(np.load(base / "fits" / f"{s}.npz", allow_pickle=True))
        R[s].update(dict(np.load(base / "fits_a47" / f"{s}.npz")))
    import json
    gates = json.loads((cfg.results("film_replication", "results") / "a38_gates.json").read_text())
    rois = ["IFG"] + [r for r in gates["eligible_regions"] if r != "IFG"]
    rows = []
    for j, roi in enumerate(rois):
        cols = {s: np.flatnonzero(R[s]["_roi"] == roi) for s in subs}
        cols = {s: c for s, c in cols.items() if len(c)}
        subj = np.concatenate([[s] * len(c) for s, c in cols.items()])

        def stack(key):
            return np.concatenate([R[s][key][..., c] for s, c in cols.items()], axis=-1)

        arms = ("D", "R", "S82", "Smean")
        r = {a: pick(stack(f"r_{a}"))[0] for a in arms}
        comp = {a: stack(f"dec_{a}")[pick(stack(f"a28r_{a}"))[1]] for a in arms}
        tests = (("pooling: within(S_mean) - within(S)", comp["Smean"][1] - comp["S82"][1]),
                 ("basis: within(R) - within(S_mean)", comp["R"][1] - comp["Smean"][1]),
                 ("within(S_mean) - within(D)", comp["Smean"][1] - comp["D"][1]),
                 ("total r S_mean - D", r["Smean"] - r["D"]))
        for k, (lab, x) in enumerate(tests):
            ok = np.isfinite(x)
            m, lo, hi, p, _ = boot(x[ok], subj[ok], cfg.seed + 4700 + 100 * j + k)
            rows.append({"roi": roi, "test": lab, "median": m, "lo": lo, "hi": hi, "p": p,
                         "n_contacts": int(ok.sum()), "n_subjects": len(cols),
                         "mean_r_Smean": float(r["Smean"].mean())})
    T = pd.DataFrame(rows)
    T["p_holm_regions"] = np.nan
    for lab in T.test.unique():
        m = (T.roi != "IFG") & (T.test == lab)
        T.loc[m, "p_holm_regions"] = holm(T.loc[m, "p"].to_numpy())
    T.to_csv(base / "a47_film.csv", index=False)
    print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
