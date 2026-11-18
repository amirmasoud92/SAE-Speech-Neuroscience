"""Peak-lag targets and a nuisance-projected ridge pass over a grid of PCA ranks.

``load_targets`` reads each electrode's high-gamma at its own peak lag from the variance
partition; ``run_folds`` fits, inside each training fold, the nuisance projection, centring
and an eigendecomposition-based ridge, and accumulates held-out error over a grid of numbers
of principal components and penalties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_GRID = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 200, 320, 512, 800, 1280, 2000, 3200]
LAMBDAS = np.array([1e2, 1e3, 1e4, 1e5])
N_ACOUSTIC_PC = 100


def load_targets(cfg, sig: pd.DataFrame, n_words: int) -> np.ndarray:
    """Each electrode's high-gamma at its own peak lag, as one (n_words, n_elec) matrix."""
    Y = np.full((n_words, len(sig)), np.nan, dtype=np.float64)
    for sub, g in sig.groupby("subject"):
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        names = list(v["ch_names"].astype(str))
        Ys = np.load(cfg.cache(cfg.digest, f"targets_{sub}.npz"), allow_pickle=True)["Y"]
        for pos, e in zip(g.index.to_numpy(), g.itertuples()):
            lag = int(np.argmin(np.abs(cfg.lags.centres_s - e.lag_s)))
            Y[:, pos] = Ys[:, names.index(e.ch), lag]
        del Ys
    assert np.isfinite(Y).all(), "missing target values"
    return Y


def run_folds(Xs, nuis, Yall, folds, ks, n_curve, want_full, lambdas=None):
    """One pooled out-of-fold pass. Everything fitted here is fitted on training rows only.

    Returns the summed squared error over the k-grid for the first ``n_curve`` targets and,
    if asked, the full-rank out-of-fold predictions for every target.
    """
    lambdas = LAMBDAS if lambdas is None else np.asarray(lambdas, dtype=float)
    n_tot = Yall.shape[1]
    sse = np.zeros((len(ks), len(lambdas), n_curve))
    full = (np.zeros((len(lambdas), len(Xs), n_tot), dtype=np.float32)
            if want_full else None)
    # the fold-residualised test targets, so a caller can score any target against the
    # same quantity the model was actually asked to predict
    yres = np.zeros((len(Xs), n_tot), dtype=np.float32) if want_full else None

    for fi, (tr, te) in enumerate(folds):
        # --- nuisance removal, fitted on train rows and applied to test rows ------------
        Ntr, Nte = nuis[tr], nuis[te]
        G = Ntr.T @ Ntr + 1e-8 * np.eye(Ntr.shape[1])
        bx = np.linalg.solve(G, Ntr.T @ Xs[tr])
        by = np.linalg.solve(G, Ntr.T @ Yall[tr])
        Xtr, Xte = Xs[tr] - Ntr @ bx, Xs[te] - Nte @ bx
        Ytr, Yte = Yall[tr] - Ntr @ by, Yall[te] - Nte @ by
        mu = Xtr.mean(0)
        Xtr, Xte = Xtr - mu, Xte - mu
        ym = Ytr.mean(0)
        Ytr, Yte = Ytr - ym, Yte - ym

        ev, U = np.linalg.eigh(Xtr @ Xtr.T)
        ev, U = ev[::-1], U[:, ::-1]
        keep = ev > ev.max() * 1e-10
        ev, U = ev[keep], U[:, keep]
        s = np.sqrt(ev)
        Zte = Xte @ ((Xtr.T @ U) / s)
        UtY = U.T @ Ytr

        for li, lam in enumerate(lambdas):
            coef = (s[:, None] * UtY) / (ev[:, None] + lam)
            P, prev = np.zeros((len(te), n_tot)), 0
            for ki, k in enumerate(ks + [len(s)]):
                k = min(k, len(s))
                if k > prev:
                    P += Zte[:, prev:k] @ coef[prev:k]
                    prev = k
                if ki < len(ks):
                    sse[ki, li] += ((Yte[:, :n_curve] - P[:, :n_curve]) ** 2).sum(0)
            if want_full:
                full[li, te] = P.astype(np.float32)
                if li == 0:
                    yres[te] = Yte.astype(np.float32)
            if li == 0:
                # the residualised test targets are needed for the R2 denominator
                sse_tss = (Yte[:, :n_curve] ** 2).sum(0)
                tss = sse_tss if fi == 0 and li == 0 else tss + sse_tss
        print(f"  fold {fi + 1}/{len(folds)}: rank {len(s)}", flush=True)
    return sse, full, tss, yres
