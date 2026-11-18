"""Sparse and dense arms against targets averaged over 1, 2, 4 and 8 participants.

Both representations - Gemma-2-2B's residual stream and the sparse code of that stream, each
at z-scored PCA-300 - are refitted against IFG targets averaged over participant subsets
(up to 12 random subsets per rung) with the base model held constant. Every step that touches
the target is fitted on training rows inside each fold: per-participant nuisance regression,
per-participant standardisation and the averaging itself.

A participant's series is valid at a word where at least half its IFG electrodes are valid;
an averaged target is valid only where every participant in the subset is valid. The contrast
at eight participants is tested with a moving-block bootstrap (100-token blocks, 10,000
resamples). Writes ``denoising_ladder/``.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from capacity_matching import matched_capacity as M
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

LADDER = [1, 2, 4, 8]
N_SUBSET = 12
BLOCK = 100
N_BOOT = 10000


def subject_series(Y, V, subj):
    """Per-participant IFG series inputs: electrode columns and masks, grouped."""
    names = list(dict.fromkeys(subj))
    return names, {s: (Y[:, subj == s], V[:, subj == s]) for s in names}


def fold_series(groups, names, nuis, tr):
    """Build every participant's series with training-row statistics only."""
    S, SV = [], []
    Ntr = nuis[tr]
    G = Ntr.T @ Ntr + 1e-8 * np.eye(Ntr.shape[1])
    for s in names:
        Ye, Ve = groups[s]
        w = Ve[tr].astype(np.float64)
        n = np.maximum(w.sum(0), 1.0)
        mu = (Ye[tr] * w).sum(0) / n
        sd = np.sqrt((((Ye[tr] - mu) ** 2) * w).sum(0) / n) + 1e-12
        Z = np.where(Ve, (Ye - mu) / sd, 0.0)
        cnt = Ve.sum(1)
        x = Z.sum(1) / np.maximum(cnt, 1)
        v = cnt >= 0.5 * Ve.shape[1]
        rows = tr[v[tr]]
        b = np.linalg.solve(nuis[rows].T @ nuis[rows] + 1e-8 * np.eye(nuis.shape[1]),
                            nuis[rows].T @ x[rows])
        x = x - nuis @ b
        m, s_ = x[rows].mean(), x[rows].std() + 1e-12
        S.append((x - m) / s_)
        SV.append(v)
    return np.column_stack(S), np.column_stack(SV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 31)

    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, _ = M.ifg_targets(cfg, "window")
    names, groups = subject_series(Y, V, subj)
    n_s = len(names)
    print(f"{n_s} subjects with IFG electrodes: {', '.join(names)}")

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
    print(f"{len(subsets)} target series across the ladder {LADDER}")

    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)
    arms = {"Gemma dense": M.zpca(M.gemma_dense(cfg, wi), 300),
            "SAE": M.zpca(M.sae_band(cfg), 300)}

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
        res[a], li = M.best_lambda_r(P[a], Yr, Vt)
        pick[a] = li
        print(f"  {a}: lambda {M.LAMBDAS[li]:g}")
    M_["gemma"], M_["sae"] = res["Gemma dense"], res["SAE"]
    M_["diff"] = M_.sae - M_.gemma
    M_["valid_frac"] = Vt.mean(0)

    print("\nDENOISING LADDER (fold-wise target construction)")
    print(f"{'n':>3} {'subsets':>8} {'dense':>9} {'SAE':>9} {'SAE-dense':>11}")
    for n in LADDER:
        g = M_[M_.n_subjects == n]
        print(f"{n:>3} {len(g):>8} {g.gemma.mean():>9.4f} {g.sae.mean():>9.4f} "
              f"{g['diff'].mean():>+11.4f}")
    rho, p_rho = sps.spearmanr(M_.n_subjects, M_["diff"])
    print(f"trend of (SAE - dense) with n: Spearman rho = {rho:+.3f}, P = {p_rho:.4f}")

    j = int(M_.index[M_.n_subjects == n_s][0])
    v = Vt[:, j]
    pg = P["Gemma dense"][pick["Gemma dense"], v, j]
    ps = P["SAE"][pick["SAE"], v, j]
    yg = Yr[v, j]
    n_w = len(yg)
    nb = int(np.ceil(n_w / BLOCK))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        st = rng.integers(0, n_w - BLOCK, nb)
        ix = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n_w]
        boot[b] = (M.colcorr(ps[ix, None], yg[ix, None])[0]
                   - M.colcorr(pg[ix, None], yg[ix, None])[0])
    obs8 = float(M_.loc[M_.n_subjects == n_s, "diff"].iloc[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p8 = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    print(f"n = {n_s} rung: SAE - dense = {obs8:+.4f} [{lo:+.4f}, {hi:+.4f}]  P = {p8:.4f}")

    o = cfg.results("denoising_ladder")
    o.mkdir(parents=True, exist_ok=True)
    M_.to_csv(o / "a31_ladder.csv", index=False)
    pd.DataFrame([{"rho_trend": float(rho), "p_trend": float(p_rho), "diff_n8": obs8,
                   "ci_lo": lo, "ci_hi": hi, "p_n8": p8,
                   "gemma_n1": float(M_[M_.n_subjects == 1].gemma.mean()),
                   "gemma_n8": float(M_[M_.n_subjects == n_s].gemma.mean()),
                   "sae_n1": float(M_[M_.n_subjects == 1].sae.mean()),
                   "sae_n8": float(M_[M_.n_subjects == n_s].sae.mean()),
                   "valid_frac_n8": float(Vt[:, j].mean())}]).to_csv(
        o / "a31_primary.csv", index=False)
    print("written:", o / "a31_primary.csv")


if __name__ == "__main__":
    main()
