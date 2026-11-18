"""Per-feature table for the fixed 300-feature pool: density, coherence and neural importance.

* ``density_words`` - the number of analysed words the feature is active on.
* ``coherence`` - the feature's semantic coherence (``interpretability/coherence.py``) in
  GPT-2 XL layer-24 space; NaN where fewer than five distinct words are available.
* ``importance_z`` - neural importance: the feature's correlation with an electrode's
  0-500 ms window response after surprisal, acoustic and lexical bands are projected out,
  expressed as |z| against 200 circular shifts of the response, then averaged over all IFG
  electrodes. Masked tokens are set to the electrode's valid-token mean after
  residualisation, and the mask rotates with the series in every shift.

The table is keyed by raw dictionary feature id. Writes ``feature_table/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from interpretability.coherence import coherence
from encoding.fit_encoding_model import analysed_words

N_SHIFTS = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--n-pool", type=int, default=300,
                    help="number of features, taken by stimulus variance (300 in the paper)")
    ap.add_argument("--out", default="feature_table_ifg.csv", help="written under feature_table/")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)

    w = analysed_words(cfg).reset_index(drop=True)
    wi = w["word_idx"].to_numpy()
    tok = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    X = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)["X"]
    ids = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)["ids_sae"]
    Xs = X[:, 0:4933]

    pool = np.sort(np.argsort(-Xs.var(0))[:args.n_pool])   # the variance pool
    S = Xs[:, pool]
    dens = (S > 0).sum(0)
    raw = np.asarray(ids)[pool].astype(int)
    print(f"pool: {len(raw)} features, density {dens.min()}-{dens.max()} words")

    # ---- coherence, in an independent embedding space, no neural data ----------------
    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    with h5py.File(g / "features.hdf5", "r") as h:
        G = h["layer-24"][...]
    gt = pd.read_csv(g / "transcript.tsv", sep="\t")
    E = pd.DataFrame(G).groupby(gt["word_idx"].to_numpy()).mean().reindex(wi).to_numpy()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    coh = np.asarray(coherence(S, E, tok))
    print(f"coherence computed, {int(np.isnan(coh).sum())} NaN")

    # ---- neural importance over all IFG electrodes ----------------------------------
    Yw, Vw, subj_w, meta_w = Mt.ifg_targets(cfg, "window")
    sig = meta_w.assign(subject=subj_w)
    nuis = np.column_stack([X[:, 4933:4934], X[:, 4934:6294], X[:, 6294:6297],
                            np.ones(len(X), dtype=np.float32)])
    Q, _ = np.linalg.qr(nuis.astype(np.float64))

    def res(A):
        A = A.astype(np.float64)
        return A - Q @ (Q.T @ A)

    Sc = res(S)
    Sc -= Sc.mean(0)
    nx = np.sqrt((Sc ** 2).sum(0))
    nx[nx == 0] = np.inf
    sh = rng.choice(np.arange(172, len(w) - 172), N_SHIFTS, replace=False)

    def importance(yc, v=None):
        if v is not None:
            yc = np.where(v, yc - yc[v].mean(), 0.0)
        else:
            yc = yc - yc.mean()
        r = (Sc.T @ yc) / (nx * np.sqrt((yc ** 2).sum()))
        Yn = np.stack([np.roll(yc, int(s)) for s in sh], 1)
        Yn -= Yn.mean(0)
        rn = (Sc.T @ Yn) / np.outer(nx, np.sqrt((Yn ** 2).sum(0)))
        return np.abs((r - rn.mean(1)) / (rn.std(1, ddof=1) + 1e-12))

    imp = []
    for j in range(Yw.shape[1]):
        v = Vw[:, j]
        y = np.where(v, Yw[:, j], Yw[v, j].mean())
        imp.append(importance(res(y[:, None])[:, 0], v))
    per_el = np.stack(imp)
    # per-electrode importance, so that leave-one-participant-out can re-average it
    np.savez(cfg.results("feature_table", args.out.replace(".csv", "_per_electrode.npz")),
             importance=per_el, subject=np.asarray(sig.subject, dtype=object),
             feature_id=np.asarray(ids)[pool].astype(int))
    imp = per_el.mean(0)
    print(f"importance over {len(sig)} IFG electrodes, median |z| {np.median(imp):.3f}")

    out = pd.DataFrame({"feature_id": raw, "pool_row": np.arange(len(raw)),
                        "density_words": dens.astype(int), "coherence": coh,
                        "importance_z": imp})
    p = cfg.results("feature_table", args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    print("written:", p)


if __name__ == "__main__":
    main()
