"""Shared machinery for the control and robustness analyses.

Data handling is reused from ``capacity_matching/matched_capacity.py`` (targets, masks,
folds, nuisance set, z-scored PCA, masked ridge). These analyses use a **training-only**
convention:

* z-scoring and PCA are fitted on the outer training rows and applied to the test rows;
* the penalty is chosen per arm and per outer fold by an inner contiguous-block
  cross-validation on the outer training rows (three folds, same embargo), on mean inner r;
* held-out predictions are standardised per target within each test fold before pooling.

Also holds inference used by several analyses: the participant-clustered electrode statistic
and the exact participant-level sign-flip cluster test over lags, which enumerates every sign
assignment.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

from capacity_matching import matched_capacity as M
from encoding.cross_validation import BlockedTimeSeriesSplit

N_BOOT = 10000
MIN_OCC = 4


# --------------------------------------------------------------------------- inputs

def context(cfg):
    """Analysed words, onsets, word index, word types, outer folds, fold id and nuisance."""
    from encoding.fit_encoding_model import analysed_words
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    folds = M.folds_for(cfg, onsets)
    fid = np.full(len(w), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    return dict(words=w, onsets=onsets, wi=wi, types=types, folds=folds, fid=fid,
                nuis=M.nuisance(cfg))


def roi_lag_targets(cfg, roi: str, step: int = 1):
    """All-lag targets for one ROI: Y (n_words, n_elec * n_lags) standardised per column
    over valid cells, V, and per-electrode subject and channel labels."""
    blocks, vblocks, subj, chans = [], [], [], []
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        keep = np.flatnonzero(v["roi"].astype(str) == roi)
        if len(keep) == 0:
            continue
        z = np.load(cfg.cache(cfg.digest, f"targets_{sub}.npz"), allow_pickle=True)
        blocks.append(z["Y"][:, keep, ::step].astype(np.float32))
        vblocks.append(z["V"][:, keep, ::step])
        subj += [sub] * len(keep)
        chans += [str(c) for c in v["ch_names"][keep]]
        del z
    Y = np.concatenate(blocks, axis=1)
    V = np.concatenate(vblocks, axis=1)
    n_w, n_e, n_l = Y.shape
    w = V.astype(np.float32)
    nv = np.maximum(w.sum(0), 1.0)
    mu = (Y * w).sum(0) / nv
    sd = np.sqrt((((Y - mu) ** 2) * w).sum(0) / nv) + 1e-12
    Y = np.where(V, (Y - mu) / sd, 0.0).astype(np.float64).reshape(n_w, n_e * n_l)
    return Y, V.reshape(n_w, n_e * n_l), np.array(subj), np.array(chans), n_l


# ------------------------------------------------------------------ fold-wise design

def fold_pca(Xraw: np.ndarray, tr: np.ndarray, k: int) -> np.ndarray:
    """z-score and PCA fitted on training rows; every row projected with those loadings."""
    Xraw = np.asarray(Xraw, dtype=np.float64)
    mu = Xraw[tr].mean(0)
    sd = Xraw[tr].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xraw - mu) / sd
    _, _, Vt = np.linalg.svd(Z[tr], full_matrices=False)
    return np.ascontiguousarray(Z @ Vt[:k].T)


def fold_z(Xraw: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """z-scoring fitted on training rows, with no rotation: the full-width raw arms."""
    Xraw = np.asarray(Xraw, dtype=np.float64)
    mu = Xraw[tr].mean(0)
    sd = Xraw[tr].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return np.ascontiguousarray((Xraw - mu) / sd)


def _ridge_all(Xp, nuis, Y, V, tr, te, lambdas):
    """Imputed (training-mean) masked ridge for every penalty; returns predictions
    (n_lam, n_te, n_t) and residualised test targets (n_te, n_t)."""
    w = V[tr].astype(np.float64)
    mu = (Y[tr] * w).sum(0) / np.maximum(w.sum(0), 1)
    Ytr = np.where(V[tr], Y[tr], mu)
    Yte = np.where(V[te], Y[te], mu)
    Ntr, Nte = nuis[tr], nuis[te]
    G = Ntr.T @ Ntr + 1e-8 * np.eye(Ntr.shape[1])
    bx = np.linalg.solve(G, Ntr.T @ Xp[tr])
    by = np.linalg.solve(G, Ntr.T @ Ytr)
    Xtr, Xte = Xp[tr] - Ntr @ bx, Xp[te] - Nte @ bx
    Ytr, Yte = Ytr - Ntr @ by, Yte - Nte @ by
    mx = Xtr.mean(0)
    Xtr, Xte = Xtr - mx, Xte - mx
    my = Ytr.mean(0)
    Ytr, Yte = Ytr - my, Yte - my
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ Ytr
    XV = Xte @ Vt.T
    P = np.stack([XV @ ((s / (s ** 2 + lam))[:, None] * UtY) for lam in lambdas])
    return P, Yte


def inner_lambda(Xp, nuis, Y, V, tr, onsets, cfg, lambdas=M.LAMBDAS):
    """Penalty chosen on the outer training rows only: inner contiguous-block CV, mean
    over inner folds and target columns of the per-fold held-out r."""
    inner = BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_inner_folds"]),
                                   embargo_s=float(cfg.cv["embargo_s"])).split(onsets[tr])
    score = np.zeros(len(lambdas))
    for itr, ite in inner:
        a, b = tr[itr], tr[ite]
        P, Yte = _ridge_all(Xp, nuis, Y, V, a, b, lambdas)
        v = V[b]
        for li in range(len(lambdas)):
            score[li] += float(np.nanmean(M.colcorr(P[li], Yte, v)))
    return float(lambdas[int(np.argmax(score))]), score


def _standardise(P, V):
    """z-score each prediction column over its valid rows (one test fold)."""
    w = V.astype(np.float64)
    n = np.maximum(w.sum(0), 1.0)
    mu = (P * w).sum(0) / n
    sd = np.sqrt((((P - mu) ** 2) * w).sum(0) / n)
    return (P - mu) / np.where(sd > 1e-12, sd, 1.0)


def oof_to(Xraw, k, ctx, Y, V, cfg, exact: bool = True, verbose: str = "", lambdas=None):
    """Training-only out-of-fold predictions for one arm.

    ``k``: number of training-fold principal components; ``"z"`` for training-fold
    z-scoring with no PCA; None for the raw matrix as given.
    ``lambdas``: the inner-CV penalty grid; default ``matched_capacity.LAMBDAS`` (a wider
    grid is used only as a labelled sensitivity check).

    Returns P (n_words, n_targets), standardised within each test fold; Yr, the residualised
    test targets; and the penalty chosen in each outer fold.
    """
    nuis, onsets = ctx["nuis"], ctx["onsets"]
    n_w, n_t = Y.shape
    P = np.zeros((n_w, n_t))
    Yr = np.zeros((n_w, n_t))
    lams = []
    for f, (tr, te) in enumerate(ctx["folds"]):
        # a fold-specific nuisance set (bands with training-fold PCA) when supplied
        nuis_f = ctx["nuis_fold"][f] if "nuis_fold" in ctx else nuis
        if isinstance(k, str) and k == "z":
            Xp = fold_z(Xraw, tr)
        elif k is not None:
            Xp = fold_pca(Xraw, tr, k)
        else:
            Xp = np.asarray(Xraw, np.float64)
        lam, _ = inner_lambda(Xp, nuis_f, Y, V, tr, onsets, cfg,
                              **({} if lambdas is None else {"lambdas": np.asarray(lambdas)}))
        lams.append(lam)
        if exact:
            Pf, Yf, _ = M.oof(Xp, nuis_f, Y, V, [(tr, te)], lambdas=np.array([lam]),
                              exact=True)
            P[te], Yr[te] = Pf[0][te], Yf[te]
        else:
            Pf, Yte = _ridge_all(Xp, nuis_f, Y, V, tr, te, np.array([lam]))
            P[te], Yr[te] = Pf[0], Yte
        P[te] = _standardise(P[te], V[te])
        if verbose:
            print(f"    {verbose} fold {f + 1}/{len(ctx['folds'])}: lambda {lam:g}",
                  flush=True)
    return P, Yr, np.array(lams)


def oof_legacy(Xr, ctx, Y, V, exact: bool = True):
    """The convention of the main matched-capacity analyses: global PCA supplied by the
    caller, one penalty per arm on pooled held-out r (``matched_capacity.best_lambda_r``)."""
    P, Yr, _ = M.oof(np.ascontiguousarray(Xr), ctx["nuis"], Y, V, ctx["folds"], exact=exact)
    r, li = M.best_lambda_r(P, Yr, V)
    return r, float(M.LAMBDAS[li])


# --------------------------------------------------------- word-type decomposition

def within_between(P, Yr, V, ctx):
    """Between-type and within-type r per target column: ``decomposition.decompose`` with
    within-fold type means, types with >= MIN_OCC tokens."""
    from word_types.decomposition import decompose
    between, within, n_tok, n_types = decompose(P, Yr, V, ctx["types"], MIN_OCC,
                                                fid=ctx["fid"])
    return within, between, dict(n_types=int(n_types), n_tokens=int(n_tok))


# ------------------------------------------------------------------------ inference

def electrode_stat(diff, subj, seed, n_boot=N_BOOT):
    """Median electrode difference, participant-clustered bootstrap 95% and 90% CIs,
    two-sided P; also the bootstrap draws."""
    ok = np.isfinite(diff)
    m, lo, hi, p, b = M.cluster_boot(diff[ok], subj[ok], np.random.default_rng(seed), n_boot)
    lo90, hi90 = np.percentile(b, [5, 95])
    return dict(estimate=m, ci95_lo=lo, ci95_hi=hi, ci90_lo=float(lo90),
                ci90_hi=float(hi90), p_two_sided=p, test="participant-clustered bootstrap "
                f"of the median electrode difference ({n_boot} resamples), two-sided",
                n_electrodes=int(ok.sum()), n_participants=int(len(np.unique(subj[ok])))), b


def _clusters(t, thresh):
    """Contiguous same-sign supra-threshold runs: list of (start, stop, mass, sign)."""
    out = []
    for sign in (1, -1):
        above = sign * t > thresh
        i = 0
        while i < len(t):
            if above[i]:
                j = i
                while j + 1 < len(t) and above[j + 1]:
                    j += 1
                out.append((i, j + 1, float(t[i:j + 1].sum()), sign))
                i = j + 1
            else:
                i += 1
    return out


def signflip_cluster(D, subj, lags_ms):
    """Exact participant-level sign-flip cluster test on a (n_elec, n_lags) difference.

    Statistic: one-sample t over participant-mean profiles; threshold |t| > t(0.975, n-1);
    cluster mass = sum of t; null = max |mass| over all 2^n sign assignments.
    """
    uq = list(dict.fromkeys(subj))
    S = np.stack([np.nanmean(D[subj == u], 0) for u in uq])          # (n_subj, n_lags)
    n = len(uq)
    thresh = float(sps.t.ppf(0.975, n - 1))

    def tstat(X):
        m = X.mean(0)
        s = X.std(0, ddof=1)
        return m / (s / np.sqrt(n) + 1e-15)

    t_obs = tstat(S)
    obs = _clusters(t_obs, thresh)
    signs = np.array(list(itertools.product((1, -1), repeat=n)), dtype=np.float64)
    null = np.empty(len(signs))
    for i, sg in enumerate(signs):
        c = _clusters(tstat(S * sg[:, None]), thresh)
        null[i] = max([abs(x[2]) for x in c], default=0.0)
    rows = []
    for st, sp, mass, sign in obs:
        rows.append({"direction": "SAE > dense" if sign > 0 else "SAE < dense",
                     "lag_start_ms": float(lags_ms[st]), "lag_stop_ms": float(lags_ms[sp - 1]),
                     "n_lags": sp - st, "mass": mass,
                     "p_cluster": float((null >= abs(mass)).mean()),
                     "mean_diff_in_cluster": float(np.nanmean(S[:, st:sp]))})
    return pd.DataFrame(rows), t_obs, dict(n_participants=n, n_assignments=len(signs),
                                           t_threshold=thresh,
                                           p_floor=float(2 / len(signs)))


def boot_bands(D, subj, seed, n_boot=N_BOOT):
    """Pointwise 95% band of the electrode-median difference at each lag,
    participant-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    uq = np.unique(subj)
    by = {u: np.flatnonzero(subj == u) for u in uq}
    B = np.empty((n_boot, D.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(uq, len(uq), True)])
        B[b] = np.nanmedian(D[idx], 0)
    return np.nanmedian(D, 0), np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)
