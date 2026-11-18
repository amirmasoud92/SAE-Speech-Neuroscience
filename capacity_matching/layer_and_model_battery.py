"""The matched-capacity comparisons at every layer and for other dictionaries and models.

Each row is one (model, layer, dictionary): Gemma-2-2B layers 0-25, a Matryoshka BatchTopK
dictionary at layer 12, Gemma-2-9B layers 9, 20 and 31, and Llama-3.1-8B layers 8, 16 and 24
(JumpReLU and TopK-50 rules). Targets, masks, nuisance bands, folds, ridge and bootstrap are
those of ``capacity_matching/matched_capacity.py`` and identical in every row; only the
representation changes. Per row:

* dense z-PCA-300 and sparse z-PCA-300, exact masking - the sparse-versus-dense contrast, with
  the dense reference always the same model and layer's own residual stream;
* sparse z-PCA-85 against the classical 85-column band, exact masking;
* usable directions: of the 300 highest-variance live features, how many exceed the median
  coherence (GPT-2 XL layer 24) of 300 principal components of that row's residual stream.

Layers are Holm-corrected. Rows run in parallel processes on shared inputs. Writes
``layer_and_model_battery/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
POOL = 300


def shared_inputs(cfg, out: Path) -> Path:
    """Compute once what every row needs, and save it."""
    from capacity_matching import matched_capacity as M
    from feature_extraction.feature_bands import classical_features
    from encoding.fit_encoding_model import analysed_words

    f = out / "_shared.npz"
    if f.exists():
        return f
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, _ = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)
    C, _ = classical_features(cfg, wi, w["word"])
    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    with h5py.File(g / "features.hdf5", "r") as h:
        G = h["layer-24"][...]
    gt = pd.read_csv(g / "transcript.tsv", sep="\t")
    E = pd.DataFrame(G).groupby(gt["word_idx"].to_numpy()).mean().reindex(wi).to_numpy()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    tok = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    np.savez(f, Y=Y, V=V, subj=subj, nuis=nuis, wi=wi, C=np.asarray(C, np.float64),
             E=E.astype(np.float32), tok=tok,
             **{f"tr{k}": tr for k, (tr, te) in enumerate(folds)},
             **{f"te{k}": te for k, (tr, te) in enumerate(folds)}, n_folds=len(folds))
    return f


def run_row(row: dict, shared: str, seed: int, min_active: int, pca_seed: int) -> dict:
    from capacity_matching import matched_capacity as M
    from interpretability.coherence import coherence
    from sklearn.decomposition import PCA

    z = np.load(shared, allow_pickle=True)
    Y, V, subj, nuis, wi = z["Y"], z["V"], z["subj"], z["nuis"], z["wi"]
    folds = [(z[f"tr{k}"], z[f"te{k}"]) for k in range(int(z["n_folds"]))]
    with h5py.File(row["resid_file"], "r") as h:
        R = h["resid_word"][...][wi].astype(np.float64)
    with h5py.File(row["sae_file"], "r") as h:
        F = h["sae_word_max"][...][wi].astype(np.float64)
    F = F[:, (F > 0).sum(0) >= min_active]
    rng = np.random.default_rng(seed)

    def fit(Xr):
        P, Yr, _ = M.oof(np.ascontiguousarray(Xr), nuis, Y, V, folds, exact=True)
        r, li = M.best_lambda_r(P, Yr, V)
        return r, float(M.LAMBDAS[li])

    r_dense, l_dense = fit(M.zpca(R, 300))
    r_sae, l_sae = fit(M.zpca(F, 300))
    r_sae85, _ = fit(M.zpca(F, 85))
    out = {**{k: v for k, v in row.items() if not k.endswith("_file")},
           "n_live": int(F.shape[1]), "dense_r": float(r_dense.mean()),
           "sae300_r": float(r_sae.mean()), "sae85_r": float(r_sae85.mean()),
           "dense_lambda": l_dense, "sae300_lambda": l_sae}
    m, lo, hi, p, _ = M.cluster_boot(r_sae - r_dense, subj, rng)
    out.update(d_sae_dense=m, d_sae_dense_lo=lo, d_sae_dense_hi=hi, d_sae_dense_p=p)
    out["_r_sae85"] = r_sae85

    # usable directions against this row's own residual stream
    E, tok = z["E"].astype(np.float64), z["tok"]
    # the seed of dictionary_coherence, so the layer-12 row must reproduce its count
    P = PCA(n_components=POOL, random_state=pca_seed).fit_transform(R.astype(np.float32))
    c_pca = np.concatenate([np.asarray(coherence(P, E, tok)),
                            np.asarray(coherence(-P, E, tok))])
    med_p = float(np.nanmedian(c_pca))
    pool = np.sort(np.argsort(-F.var(0))[:POOL])
    coh = np.asarray(coherence(F[:, pool], E, tok))
    out.update(usable_directions=int(np.nansum(coh > med_p)),
               n_scored=int(np.isfinite(coh).sum()), pca_median_coherence=med_p,
               sae_median_coherence=float(np.nanmedian(coh)))
    return out


def rows_for(cfg, which: list[str]) -> list[dict]:
    feat = cfg.results("layer_and_model_battery", "features")
    rows = []
    if "gemma2-2b" in which:
        for f in sorted(feat.glob("gemma2-2b_L*_w16k_l0-*.hdf5"),
                        key=lambda p: int(p.name.split("_L")[1].split("_")[0])):
            with h5py.File(f, "r") as h:
                a = dict(h.attrs)
            rows.append({"model": "gemma2-2b", "family": "GemmaScope JumpReLU",
                         "layer": int(a["layer"]), "n_layers": 26,
                         "dictionary": f"w16k L0={int(a['nominal_l0'])}",
                         "token_l0": float(a["token_l0_mean"]), "fvu": float(a["fvu"]),
                         "resid_file": str(f), "sae_file": str(f)})
    if "matryoshka" in which:
        main = cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5"
        mat = cfg.results("features") / "gemma2-2b_L12_matryoshka32k.hdf5"
        rows.append({"model": "gemma2-2b", "family": "Matryoshka BatchTopK", "layer": 12,
                     "n_layers": 26, "dictionary": "w32k L0~40", "token_l0": np.nan,
                     "fvu": np.nan, "resid_file": str(main), "sae_file": str(mat)})
    for key, fam in (("llama3.1-8b", "Llama Scope"),                 # released JumpReLU rule
                     ("llama3.1-8b-topk", "Llama Scope (TopK-50)")):  # training rule
        if key not in which:
            continue
        for f in sorted(feat.glob(f"{key}_L*_w32k_l0-*.hdf5"),
                        key=lambda p: int(p.name.split("_L")[1].split("_")[0])):
            with h5py.File(f, "r") as h:
                a = dict(h.attrs)
            rows.append({"model": "llama3.1-8b", "family": fam,
                         "layer": int(a["layer"]), "n_layers": 32,
                         "dictionary": f"LXR-8x (32k) L0={int(a['nominal_l0'])}",
                         "token_l0": float(a["token_l0_mean"]), "fvu": float(a["fvu"]),
                         "resid_file": str(f), "sae_file": str(f)})
    if "gemma2-9b" in which:
        for f in sorted(feat.glob("gemma2-9b_L*_w16k_l0-*.hdf5"),
                        key=lambda p: int(p.name.split("_L")[1].split("_")[0])):
            with h5py.File(f, "r") as h:
                a = dict(h.attrs)
            rows.append({"model": "gemma2-9b", "family": "GemmaScope JumpReLU",
                         "layer": int(a["layer"]), "n_layers": 42,
                         "dictionary": f"w16k L0={int(a['nominal_l0'])}",
                         "token_l0": float(a["token_l0_mean"]), "fvu": float(a["fvu"]),
                         "resid_file": str(f), "sae_file": str(f)})
    return rows


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    from capacity_matching import matched_capacity as M
    from utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE.parent / "config" / "config.yaml")
    ap.add_argument("--which", nargs="+",
                    default=["gemma2-2b", "matryoshka", "gemma2-9b", "llama3.1-8b",
                             "llama3.1-8b-topk"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="",
                    help="suffix for the output table, e.g. _llama for rows run separately")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg.results("layer_and_model_battery")
    out.mkdir(parents=True, exist_ok=True)
    shared = shared_inputs(cfg, out)
    min_active = int(cfg.sae["min_active_words"])

    # the classical arm does not change across rows: fit once
    z = np.load(shared, allow_pickle=True)
    folds = [(z[f"tr{k}"], z[f"te{k}"]) for k in range(int(z["n_folds"]))]
    P, Yr, _ = M.oof(z["C"], z["nuis"], z["Y"], z["V"], folds, exact=True)
    r_cla, _ = M.best_lambda_r(P, Yr, z["V"])
    subj = z["subj"]
    print(f"classical-85 mean r {r_cla.mean():+.4f}", flush=True)

    rows = rows_for(cfg, args.which)
    print(f"{len(rows)} rows", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_row, r, str(shared), cfg.seed + 35 + i, min_active, cfg.seed): r
                for i, r in enumerate(rows)}
        for fu in as_completed(futs):
            o = fu.result()
            r85 = o.pop("_r_sae85")
            rng = np.random.default_rng(cfg.seed + 350 + int(o["layer"]))
            m, lo, hi, p, _ = M.cluster_boot(r85 - r_cla, subj, rng)
            o.update(classical_r=float(r_cla.mean()), ratio_sae85_classical=float(
                r85.mean() / r_cla.mean()), d_sae_classical=m, d_sae_classical_lo=lo,
                d_sae_classical_hi=hi, d_sae_classical_p=p)
            res.append(o)
            print(f"  {o['model']} L{o['layer']:>2} {o['family'][:10]:10s} dense "
                  f"{o['dense_r']:+.4f} sae300 {o['sae300_r']:+.4f} d {o['d_sae_dense']:+.4f} "
                  f"p {o['d_sae_dense_p']:.4f} | ratio {o['ratio_sae85_classical']:.1f} | "
                  f"usable {o['usable_directions']}/{o['n_scored']}", flush=True)
    T = pd.DataFrame(res).sort_values(["model", "family", "layer"]).reset_index(drop=True)
    sweep = (T.model == "gemma2-2b") & (T.family == "GemmaScope JumpReLU")
    T["d_sae_dense_p_holm"] = np.nan
    T.loc[sweep, "d_sae_dense_p_holm"] = holm(T.loc[sweep, "d_sae_dense_p"].to_numpy())
    T.to_csv(out / f"a35_a36_battery{args.tag}.csv", index=False)
    print("written:", out / f"a35_a36_battery{args.tag}.csv")
    if not sweep.any():                    # the layer criteria are read on the 2B sweep only
        return

    s = T[sweep]
    crit = {
        "layers": int(len(s)),
        "sae_above_dense_holm": int(((s.d_sae_dense > 0) & (s.d_sae_dense_p_holm < 0.05)).sum()),
        "dense_above_sae_holm": int(((s.d_sae_dense < 0) & (s.d_sae_dense_p_holm < 0.05)).sum()),
        "sae85_beats_classical": int(((s.d_sae_classical > 0)
                                      & (s.d_sae_classical_p < 0.05)).sum()),
        "usable_above_half": int((s.usable_directions > s.n_scored / 2).sum()),
    }
    crit["general_no_gain"] = crit["sae_above_dense_holm"] == 0
    crit["general_tenfold"] = crit["sae85_beats_classical"] > crit["layers"] / 2
    crit["general_interpretable"] = crit["usable_above_half"] > crit["layers"] / 2
    pd.DataFrame([crit]).to_csv(out / "a35_criteria.csv", index=False)
    print(pd.Series(crit).to_string())


if __name__ == "__main__":
    main()
