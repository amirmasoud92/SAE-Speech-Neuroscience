"""Reconstruction error and sparsity dose-response (primary cohort).

The machinery of ``capacity_matching/matched_capacity.py`` and
``word_types.decomposition.decompose``, identical for every arm:

* total held-out r: exact masked ridge, one penalty per arm chosen on mean r;
* word-type decomposition: masked cells at the training mean, one penalty per arm chosen on
  total r of that fit, within-fold type means, types with at least four occurrences.

Arms (z-scored, PCA-300): D dense, R reconstruction, E error, and S for each of the ten
released layer-12 dictionaries (widths 16,384 and 65,536). Contrasts (Holm-corrected): the
error's total r above zero, the reconstruction's within-type component below the dense
stream's, and the error's within-type component above zero. Dose-response: the
per-electrode slope of each sparse-minus-dense difference on log2 of nominal L0, tested by
the participant-clustered bootstrap. Writes ``reconstruction_and_sparsity/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FAMILIES = {"16k": [22, 41, 82, 176, 445], "65k": [21, 38, 72, 141, 297]}
MIN_OCC, N_BOOT = 4, 10000


def shared(cfg, out: Path) -> Path:
    from capacity_matching import matched_capacity as M
    from encoding.fit_encoding_model import analysed_words
    f = out / "_shared.npz"
    if f.exists():
        return f
    w = analysed_words(cfg).reset_index(drop=True)
    wi = w["word_idx"].to_numpy()
    Y, V, subj, _ = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    folds = M.folds_for(cfg, w["start"].to_numpy(float))
    fid = np.full(len(w), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    np.savez(f, Y=Y, V=V, subj=subj, nuis=M.nuisance(cfg), wi=wi, fid=fid, types=types,
             gemma=M.gemma_dense(cfg, wi), n_folds=len(folds),
             **{f"tr{k}": tr for k, (tr, te) in enumerate(folds)},
             **{f"te{k}": te for k, (tr, te) in enumerate(folds)})
    return f


def arm_matrix(name: str, feat: Path, z, min_active: int) -> np.ndarray:
    wi = z["wi"]
    if name == "D":
        return z["gemma"]
    with h5py.File(feat, "r") as h:
        if name == "R":
            return h["recon_word"][...][wi].astype(np.float64)
        if name == "E":
            return h["error_word"][...][wi].astype(np.float64)
        F = h[name]["sae_word_max"][...][wi].astype(np.float64)
    return F[:, (F > 0).sum(0) >= min_active]


def run_arm(name: str, shared_f: str, feat: str, min_active: int) -> dict:
    z = np.load(shared_f, allow_pickle=True)
    return fit_matrix(name, arm_matrix(name, Path(feat), z, min_active), z)


def fit_matrix(name: str, Xraw: np.ndarray, z) -> dict:
    from capacity_matching import matched_capacity as M
    from word_types.decomposition import decompose
    Y, V, nuis, fid, types = z["Y"], z["V"], z["nuis"], z["fid"], z["types"]
    folds = [(z[f"tr{k}"], z[f"te{k}"]) for k in range(int(z["n_folds"]))]
    X = M.zpca(Xraw, 300)
    P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=True)
    r_exact = np.array([M.colcorr(P[k], Yr, V) for k in range(len(P))])
    P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=False)
    r_imp = np.array([M.colcorr(P[k], Yr, V) for k in range(len(P))])
    li = int(np.argmax(r_imp.mean(1)))                  # penalty on total r, per arm
    b, wn, _, _ = decompose(P[li], Yr, V, types, MIN_OCC, fid)
    return {"name": name, "r_exact": r_exact, "between": b, "within": wn,
            "lambda_decomp": float(M.LAMBDAS[li])}


def boot(x, subj, seed):
    from capacity_matching import matched_capacity as M
    m, lo, hi, p, _ = M.cluster_boot(np.asarray(x, float), subj, np.random.default_rng(seed),
                                     N_BOOT)
    return {"median": m, "lo": lo, "hi": hi, "p": p}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    from utils.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("reconstruction_and_sparsity")
    out.mkdir(parents=True, exist_ok=True)
    feat = out / "features" / "podcast.hdf5"
    sf = shared(cfg, out)
    z = np.load(sf, allow_pickle=True)
    subj = z["subj"]
    min_active = int(cfg.sae["min_active_words"])
    names = ["D", "R", "E"] + [f"w{w}_l0-{l}" for w, ls in FAMILIES.items() for l in ls]

    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_arm, n, str(sf), str(feat), min_active): n for n in names}
        for fu in as_completed(futs):
            o = fu.result()
            o["r"] = o["r_exact"][int(np.argmax(o["r_exact"].mean(1)))]
            res[o["name"]] = o
            print(f"  {o['name']:14s} mean r {o['r'].mean():+.4f}  median within "
                  f"{np.nanmedian(o['within']):+.4f}  between {np.nanmedian(o['between']):+.4f}",
                  flush=True)
    np.savez(out / "a39_a40_arms.npz", **{f"{n}__{k}": v for n, o in res.items()
                                          for k, v in o.items() if k != "name"})

    # ---- reconstruction error ---------------------------------------------------------
    D, R, E, S = res["D"], res["R"], res["E"], res["w16k_l0-82"]
    ok = lambda *a: np.all([np.isfinite(x) for x in a], axis=0)      # noqa: E731
    rows = []

    def add(test, x, seed, **kw):
        m = ok(x)
        rows.append({"test": test, **boot(x[m], subj[m], seed), **kw})

    add("A39a E total r", E["r"], cfg.seed + 391, hypothesis="> 0")
    add("A39b within(R) - within(D)", R["within"] - D["within"], cfg.seed + 392,
        hypothesis="< 0")
    add("A39c within(E)", E["within"], cfg.seed + 393, hypothesis="> 0")
    add("A39d within(R) - within(S)", R["within"] - S["within"], cfg.seed + 394,
        hypothesis="descriptive")
    add("A39d between(R) - between(S)", R["between"] - S["between"], cfg.seed + 395,
        hypothesis="descriptive")
    add("between(R) - between(D)", R["between"] - D["between"], cfg.seed + 396,
        hypothesis="descriptive")
    add("within(S) - within(D) (A34 reproduction)", S["within"] - D["within"],
        cfg.seed + 397, hypothesis="descriptive")
    add("total r R - D", R["r"] - D["r"], cfg.seed + 398, hypothesis="descriptive")
    add("total r E - S", E["r"] - S["r"], cfg.seed + 399, hypothesis="descriptive")
    T39 = pd.DataFrame(rows)
    prim = T39.test.str.match(r"A39[abc]")
    T39["p_holm"] = np.nan
    T39.loc[prim, "p_holm"] = holm(T39.loc[prim, "p"].to_numpy())
    a, b_, c = (T39.set_index("test").loc[k] for k in
                ("A39a E total r", "A39b within(R) - within(D)", "A39c within(E)"))
    if a.lo > 0 and b_.hi < 0 and c.lo > 0:
        v = "SUPPORTED: the context lost by sparsification lives in the reconstruction error"
    elif b_.hi < 0 and not c.lo > 0:
        v = ("PARTIAL: the reconstruction loses context, but the error term does not recover "
             "it on its own")
    elif not b_.hi < 0:
        v = "NOT SUPPORTED: the loss is not attributable to the discarded error"
    else:
        v = "NOT SUPPORTED as specified (A39a fails)"
    T39["verdict"] = v
    T39.to_csv(out / "a39_podcast.csv", index=False)
    print("\nA39\n" + T39.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---- sparsity dose-response -------------------------------------------------------
    drows, srows = [], []
    with h5py.File(feat, "r") as h:
        attrs = {g: dict(h[g].attrs) for g in h.keys() if g.startswith("w")}
    for fam, l0s in FAMILIES.items():
        keys = [f"w{fam}_l0-{l}" for l in l0s]
        x = np.log2(np.array(l0s, float))
        comp = {"d_sae_dense": np.stack([res[k]["r"] - D["r"] for k in keys]),
                "d_within": np.stack([res[k]["within"] - D["within"] for k in keys]),
                "d_between": np.stack([res[k]["between"] - D["between"] for k in keys])}
        for k, l0 in zip(keys, l0s):
            drows.append({"family": fam, "nominal_l0": l0,
                          "token_l0": attrs[k]["token_l0"], "fvu": attrs[k]["fvu"],
                          "sae_mean_r": float(res[k]["r"].mean()),
                          **{f"median_{c}": float(np.nanmedian(comp[c][keys.index(k)]))
                             for c in comp}})
        for c, arr in comp.items():
            xc = x - x.mean()
            slope = (xc[:, None] * (arr - np.nanmean(arr, 0))).sum(0) / (xc ** 2).sum()
            srows.append({"family": fam, "component": c,
                          "predicted": "> 0" if c != "d_between" else "< 0",
                          **boot(slope[np.isfinite(slope)], subj[np.isfinite(slope)],
                                 cfg.seed + 400 + len(srows))})
    T40 = pd.DataFrame(srows)
    for fam in FAMILIES:
        m = T40.family == fam
        T40.loc[m, "p_holm"] = holm(T40.loc[m, "p"].to_numpy())
    pd.DataFrame(drows).to_csv(out / "a40_dictionaries_podcast.csv", index=False)
    T40.to_csv(out / "a40_slopes_podcast.csv", index=False)
    print("\nA40 per dictionary\n" + pd.DataFrame(drows).to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nA40 slopes on log2(L0)\n" + T40.to_string(index=False,
                                                        float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
