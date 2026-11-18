"""The participant-averaging ladder with a Matryoshka arm.

The procedure of ``capacity_matching/denoising_ladder.py`` exactly (same participant subsets,
fold-wise target construction, z-scored PCA-300 and penalty on pooled held-out r), with the
Matryoshka BatchTopK dictionary (width 32,768, layer 12; max-pooled, live columns) added as an
arm. Its eight-participant contrast with the dense arm uses the same moving-block bootstrap
draws as the sparse contrast.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats as sps

from capacity_matching import matched_capacity as M
from utils.config import load_config
from capacity_matching.denoising_ladder import (BLOCK, LADDER, N_BOOT, N_SUBSET, fold_series,
                                                subject_series)
from encoding.fit_encoding_model import analysed_words

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    rng = np.random.default_rng(cfg.seed + 31)
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, _ = M.ifg_targets(cfg, "window")
    names, groups = subject_series(Y, V, subj)
    n_s = len(names)
    subsets, meta = [], []
    for n in LADDER:
        if n == n_s:
            ss = [tuple(range(n_s))]
        elif n == 1:
            ss = [(i,) for i in range(n_s)]
        else:
            allc = list(combinations(range(n_s), n))
            pick = rng.choice(len(allc), min(N_SUBSET, len(allc)), replace=False)
            ss = [allc[i] for i in pick]
        for s in ss:
            subsets.append(s)
            meta.append({"n_subjects": n, "subset": "+".join(names[i] for i in s)})
    M_ = pd.DataFrame(meta)
    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_matryoshka32k.hdf5", "r") as h:
        F = h["sae_word_max"][...][wi].astype(np.float64)
    F = F[:, (F > 0).sum(0) >= int(cfg.sae["min_active_words"])]
    arms = {"Gemma dense": M.zpca(M.gemma_dense(cfg, wi), 300),
            "SAE": M.zpca(M.sae_band(cfg), 300),
            "Matryoshka": M.zpca(F, 300)}
    n_t = len(subsets)
    P = {a: np.zeros((len(M.LAMBDAS), len(w), n_t)) for a in arms}
    Yr = np.zeros((len(w), n_t))
    Vt = np.zeros((len(w), n_t), bool)
    for f, (tr, te) in enumerate(folds):
        S, SV = fold_series(groups, names, nuis, tr)
        T = np.column_stack([S[:, list(s)].mean(1) for s in subsets])
        TV = np.column_stack([SV[:, list(s)].all(1) for s in subsets])
        for a, Xr in arms.items():
            Pa, Ya, _ = M.oof(Xr, nuis, T, TV, [(tr, te)])
            P[a][:, te] = Pa[:, te]
            if a == "SAE":
                Yr[te] = Ya[te]
        Vt[te] = TV[te]
        print(f"  fold {f + 1}/10", flush=True)
    res, pick = {}, {}
    for a in arms:
        res[a], pick[a] = M.best_lambda_r(P[a], Yr, Vt)
    M_["gemma"], M_["sae"], M_["matryoshka"] = res["Gemma dense"], res["SAE"], res["Matryoshka"]
    M_["diff"] = M_.sae - M_.gemma
    M_["diff_matryoshka"] = M_.matryoshka - M_.gemma
    j = int(M_.index[M_.n_subjects == n_s][0])
    v = Vt[:, j]
    pg = P["Gemma dense"][pick["Gemma dense"], v, j]
    ps = P["SAE"][pick["SAE"], v, j]
    pm = P["Matryoshka"][pick["Matryoshka"], v, j]
    yg = Yr[v, j]
    n_w = len(yg)
    nb = int(np.ceil(n_w / BLOCK))
    bs, bm = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        st = rng.integers(0, n_w - BLOCK, nb)
        ix = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n_w]
        rg = M.colcorr(pg[ix, None], yg[ix, None])[0]
        bs[b] = M.colcorr(ps[ix, None], yg[ix, None])[0] - rg
        bm[b] = M.colcorr(pm[ix, None], yg[ix, None])[0] - rg
    pub = pd.read_csv(cfg.results("denoising_ladder", "a31_primary.csv")).iloc[0]
    obs8 = float(M_.loc[j, "diff"])
    p8 = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    assert abs(obs8 - pub.diff_n8) < 1e-6 and abs(p8 - pub.p_n8) < 1e-9, (obs8, p8)
    rows = []
    for arm, col, bb in (("SAE", "diff", bs), ("Matryoshka", "diff_matryoshka", bm)):
        rho, p_rho = sps.spearmanr(M_.n_subjects, M_[col])
        o = float(M_.loc[j, col])
        rows.append({"arm": arm, "contrast": f"{arm} - Gemma dense",
                     **{f"mean_diff_n{n}": float(M_[M_.n_subjects == n][col].mean())
                        for n in LADDER},
                     f"diff_n{n_s}": o, "ci95_lo": float(np.percentile(bb, 2.5)),
                     "ci95_hi": float(np.percentile(bb, 97.5)),
                     "p_two_sided": float(2 * min((bb <= 0).mean(), (bb >= 0).mean())),
                     "test": "moving-block bootstrap (block 100 words, 10,000), two-sided",
                     "trend_rho": float(rho), "trend_p": float(p_rho),
                     "r_n1": float(M_[M_.n_subjects == 1][arm.lower() if arm != 'SAE' else 'sae'].mean()),
                     f"r_n{n_s}": float(M_.loc[j, arm.lower() if arm != 'SAE' else 'sae']),
                     "dense_r_n1": float(M_[M_.n_subjects == 1].gemma.mean()),
                     f"dense_r_n{n_s}": float(M_.loc[j, "gemma"]),
                     "n_participants": n_s, "n_target_series": n_t})
    out = cfg.results("controls", "matryoshka")
    out.mkdir(parents=True, exist_ok=True)
    M_.to_csv(out / "a2_ladder_series.csv", index=False)
    T = pd.DataFrame(rows)
    T.to_csv(out / "a2_ladder.csv", index=False)
    print(T.T.to_string())


if __name__ == "__main__":
    main()
