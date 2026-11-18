"""Double-precision covariance-form twins of ``banded_ridge.fit_banded`` and
``banded_ridge.semantic_concentration``.

Used for the film cohort, where a participant can have more than 50,000 words and a kernel
of that size per band is impractical. The functions run the same himalaya solvers in primal
(covariance) form and are otherwise the podcast procedure:

- the same blocked outer folds, embargo and fold standardisation, including the treatment of
  masked cells (training mean, excluded from scoring);
- the same random search over band weights, penalties and inner folds, with the same seeds
  (``cfg.seed + fold``);
- the same top-k selection and batched subset ridge.

Banded fits use himalaya ``BandedRidgeCV``, the primal twin of ``MultipleKernelRidgeCV``,
with the same Dirichlet candidates for a given random state. With a single band every
candidate weight is 1, so the fit reduces to a per-target penalty chosen by inner
cross-validation (``RidgeCV``). Agreement with the kernel forms on podcast data is checked by
``encoding/solver_agreement.py``.
"""
from __future__ import annotations

import logging

import numpy as np

from utils.config import Config
from encoding.cross_validation import BlockedTimeSeriesSplit
from encoding.banded_ridge import FoldStandardiser, _batched_subset_ridge, pooled_r2

log = logging.getLogger("covariance_ridge")


def _params(cfg: Config, random_search: bool) -> dict:
    m = cfg.model
    p = dict(n_targets_batch=int(m["n_targets_batch"]),
             n_alphas_batch=int(m["n_alphas_batch"]),
             n_targets_batch_refit=int(m["n_targets_batch_refit"]), progress_bar=False)
    if random_search:
        p.update(n_iter=int(m["n_iter"]), alphas=cfg.alphas)
    else:
        p.pop("progress_bar")
    return p


def _folds(cfg: Config, onsets_s: np.ndarray):
    return list(BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_outer_folds"]),
                                       embargo_s=float(cfg.cv["embargo_s"]))
                .split(onsets_s))


def fit_banded_primal(cfg: Config, bands: list[tuple[str, np.ndarray]], Y: np.ndarray,
                      onsets_s: np.ndarray, valid: np.ndarray | None = None,
                      seed: int | None = None, label: str = "", form: str = "primal",
                      dtype=np.float32, backend: str | None = None) -> dict:
    """Banded ridge. Returns y_pred, y_true (fold-standardised), valid, r2.

    ``form="kernel"`` is ``encoding.fit_banded``'s estimator (``ColumnKernelizer`` +
    ``MultipleKernelRidgeCV``), here with a selectable precision. In float32 the kernel
    form is inexact at some targets, while both forms agree exactly in float64.
    """
    from himalaya.backend import get_backend, set_backend
    from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
    from himalaya.ridge import BandedRidgeCV
    from sklearn.pipeline import make_pipeline

    seed = cfg.seed if seed is None else seed
    set_backend(backend or cfg.model["backend"], on_error="warn")
    be = get_backend()
    X = np.concatenate([x for _, x in bands], axis=1).astype(np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    edges = np.cumsum([0] + [x.shape[1] for _, x in bands])
    n_words, n_t = Y.shape
    y_pred = np.full((n_words, n_t), np.nan, dtype=np.float64)
    y_true = np.full((n_words, n_t), np.nan, dtype=np.float64)
    split = lambda A: [np.ascontiguousarray(A[:, a:b])            # noqa: E731
                       for a, b in zip(edges[:-1], edges[1:])]
    for k, (tr, te) in enumerate(_folds(cfg, onsets_s)):
        xs = FoldStandardiser().fit(X[tr])
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        ys = FoldStandardiser().fit(Y[tr], vtr)
        # FoldStandardiser returns float32; recompute in the requested precision
        Xtr, Xte = ((A - xs.mean_) / xs.scale_ for A in (X[tr], X[te]))
        Xtr[:, xs.dead_], Xte[:, xs.dead_] = 0.0, 0.0
        Ytr = (Y[tr] - ys.mean_) / ys.scale_
        Yte = (Y[te] - ys.mean_) / ys.scale_
        Ytr[:, ys.dead_], Yte[:, ys.dead_] = 0.0, 0.0
        if valid is not None:
            Ytr[~vtr], Yte[~vte] = 0.0, 0.0
        Xtr, Xte, Ytr = (np.ascontiguousarray(A, dtype=dtype) for A in (Xtr, Xte, Ytr))
        if form == "primal":
            model = BandedRidgeCV(groups="input", solver="random_search",
                                  solver_params=_params(cfg, True),
                                  cv=int(cfg.cv["n_inner_folds"]), random_state=seed + k)
            model.fit(split(Xtr), Ytr)
            pred = model.predict(split(Xte))
        else:
            model = make_pipeline(
                ColumnKernelizer([(b, Kernelizer(kernel="linear"), slice(int(a), int(c)))
                                  for (b, _), a, c in zip(bands, edges[:-1], edges[1:])]),
                MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                                      solver_params=_params(cfg, True),
                                      cv=int(cfg.cv["n_inner_folds"]), random_state=seed + k))
            model.fit(Xtr, Ytr)
            pred = model.predict(Xte)
        y_pred[te] = be.to_numpy(pred)
        y_true[te] = Yte
        log.info("%s fold %d: train %d test %d", label, k + 1, len(tr), len(te))
    return {"y_pred": y_pred, "y_true": y_true, "valid": valid,
            "r2": pooled_r2(y_true, y_pred, valid)}


def _eigh(M: np.ndarray):
    """Symmetric eigendecomposition in float64 via torch's CPU LAPACK.

    Numerically the same routine as ``np.linalg.eigh`` (syevd), but linked to MKL rather
    than numpy's OpenBLAS, which is about 8x slower here.
    """
    import torch
    lam, W = torch.linalg.eigh(torch.from_numpy(np.ascontiguousarray(M, np.float64)))
    return lam.numpy(), W.numpy()


def banded_ridge_cov(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                     edges: np.ndarray, alphas: np.ndarray, n_iter: int, n_inner: int,
                     random_state: int, return_choice: bool = False):
    """himalaya's banded random search (``solve_group_ridge_random_search``), float64,
    computed from cross-products.

    The same candidates are used: Dirichlet samples alternating concentrations 0.1 and
    1.0, with the first set to equal weights. The same rules apply:
    - contiguous ``KFold(n_inner)`` inner splits;
    - a score of minus the mean squared error per target;
    - the mean over splits plus ``1e-10 log(alpha)``, argmax over alphas;
    - a candidate replaces the best only if it scores higher by more than float64 eps;
    - refit on the whole training fold.

    Only the linear algebra differs. Each split's ridge solutions and validation losses
    come from X'X, X'Y and the validation cross-products, so the cost scales with the
    number of features, not the number of words. Exact up to rounding when features are
    fewer than samples.
    """
    from himalaya.backend import set_backend
    from himalaya.kernel_ridge._random_search import generate_dirichlet_samples
    from sklearn.model_selection import KFold

    set_backend("numpy")
    Xtr, Ytr, Xte = (np.asarray(A, np.float64) for A in (Xtr, Ytr, Xte))
    n_sp, T = len(edges) - 1, Ytr.shape[1]
    band_of = np.repeat(np.arange(n_sp), np.diff(edges))
    gammas = np.array(generate_dirichlet_samples(n_iter, n_sp, [0.1, 1.0], random_state),
                      dtype=np.float64)
    gammas[0] = 1.0 / n_sp
    alphas = np.asarray(alphas, np.float64)
    p_ = Xtr.shape[1]
    splits = []
    for tr, va in KFold(n_inner).split(Xtr):
        A, B = Xtr[tr], Xtr[va]
        # validation loss from cross-products (cheap when targets are many and features
        # few, as in H1) or from direct validation predictions (cheap when features are
        # many and targets few, as in the SCI band); both exact
        direct = p_ * p_ >= len(alphas) * len(va) * T
        splits.append((A.T @ A, A.T @ Ytr[tr], None if direct else B.T @ B,
                       None if direct else B.T @ Ytr[va], (Ytr[va] ** 2).sum(0), len(va),
                       (B, Ytr[va]) if direct else None))
    eps = np.finfo(np.float64).eps
    best = np.full(T, -np.inf)
    best_i = np.zeros(T, int)
    best_a = np.ones(T)
    slope = (np.log(alphas) * 1e-10)[:, None]
    for i, g in enumerate(gammas):
        d = np.sqrt(g[band_of])
        sc = np.zeros((len(splits), len(alphas), T))
        for j, (G, C, Gv, Cv, yy, nv, dv) in enumerate(splits):
            lam, W = _eigh(d[:, None] * G * d[None, :])
            lam = np.clip(lam, 0.0, None)
            P = W.T @ (d[:, None] * C)
            if dv is None:
                Qv = W.T @ (d[:, None] * Cv)
                H = W.T @ (d[:, None] * Gv * d[None, :]) @ W
            else:
                BW = dv[0] @ (d[:, None] * W)                 # validation rows, eigenbasis
            for a, al in enumerate(alphas):
                b = P / (lam + al)[:, None]
                if dv is None:
                    loss = yy - 2.0 * (b * Qv).sum(0) + (b * (H @ b)).sum(0)
                else:
                    loss = ((dv[1] - BW @ b) ** 2).sum(0)
                sc[j, a] = -loss / nv
        sm = sc.mean(0) + slope
        am = sm.argmax(0)
        cvs = sm[am, np.arange(T)]
        m = cvs > best + eps
        best[m], best_i[m], best_a[m] = cvs[m], i, alphas[am[m]]
    # refit on the whole training fold, one decomposition per candidate in use
    G, C = Xtr.T @ Xtr, Xtr.T @ Ytr
    pred = np.empty((Xte.shape[0], T))
    for i in np.unique(best_i):
        d = np.sqrt(gammas[i][band_of])
        lam, W = _eigh(d[:, None] * G * d[None, :])
        lam = np.clip(lam, 0.0, None)
        for al in np.unique(best_a[best_i == i]):
            t = np.flatnonzero((best_i == i) & (best_a == al))
            coef = d[:, None] * (W @ ((W.T @ (d[:, None] * C[:, t])) / (lam + al)[:, None]))
            pred[:, t] = Xte @ coef
    if return_choice:
        return pred, best_i, best_a
    return pred


def fit_banded_cov(cfg: Config, bands: list[tuple[str, np.ndarray]], Y: np.ndarray,
                   onsets_s: np.ndarray, valid: np.ndarray | None = None,
                   seed: int | None = None, label: str = "") -> dict:
    """``fit_banded_primal`` with the covariance-form solver (float64, CPU)."""
    seed = cfg.seed if seed is None else seed
    X = np.concatenate([x for _, x in bands], axis=1).astype(np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    edges = np.cumsum([0] + [x.shape[1] for _, x in bands])
    n_words, n_t = Y.shape
    y_pred = np.full((n_words, n_t), np.nan)
    y_true = np.full((n_words, n_t), np.nan)
    for k, (tr, te) in enumerate(_folds(cfg, onsets_s)):
        xs = FoldStandardiser().fit(X[tr])
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        ys = FoldStandardiser().fit(Y[tr], vtr)
        Xtr, Xte = ((A - xs.mean_) / xs.scale_ for A in (X[tr], X[te]))
        Xtr[:, xs.dead_], Xte[:, xs.dead_] = 0.0, 0.0
        Ytr = (Y[tr] - ys.mean_) / ys.scale_
        Yte = (Y[te] - ys.mean_) / ys.scale_
        Ytr[:, ys.dead_], Yte[:, ys.dead_] = 0.0, 0.0
        if valid is not None:
            Ytr[~vtr], Yte[~vte] = 0.0, 0.0
        y_pred[te] = banded_ridge_cov(Xtr, Ytr, Xte, edges, cfg.alphas,
                                      int(cfg.model["n_iter"]), int(cfg.cv["n_inner_folds"]),
                                      seed + k)
        y_true[te] = Yte
        log.info("%s fold %d: train %d test %d", label, k + 1, len(tr), len(te))
    return {"y_pred": y_pred, "y_true": y_true, "valid": valid,
            "r2": pooled_r2(y_true, y_pred, valid)}


def delta_r2_primal(cfg: Config, bands: list[tuple[str, np.ndarray]], Y: np.ndarray,
                    onsets_s: np.ndarray, band: str, valid: np.ndarray | None = None,
                    label: str = "", form: str = "primal", dtype=np.float32,
                    backend: str | None = None) -> dict:
    """``encoding.variance_partition`` for one band: R2(full) - R2(full without band)."""
    if form == "cov":                       # exact, float64
        fit = lambda b_, lab: fit_banded_cov(cfg, b_, Y, onsets_s, valid,  # noqa: E731
                                             label=lab)
    else:
        fit = lambda b_, lab: fit_banded_primal(cfg, b_, Y, onsets_s, valid,  # noqa: E731
                                                label=lab, form=form, dtype=dtype,
                                                backend=backend)
    full = fit(bands, f"{label} full")
    red = fit([b for b in bands if b[0] != band], f"{label} minus-{band}")
    return {"full": full["r2"], f"delta_{band}": full["r2"] - red["r2"],
            f"r2_without_{band}": red["r2"]}


def semantic_concentration_cov(cfg: Config, Xband: np.ndarray, Y: np.ndarray,
                               onsets_s: np.ndarray, top_k: list[int] | None = None,
                               valid: np.ndarray | None = None) -> dict:
    """``banded_ridge.semantic_concentration`` with the full-band fit in float64 covariance
    form.

    The podcast fits the band as one kernel by random search. With one band every
    candidate weight is 1, so ``banded_ridge_cov`` with a single band and a single
    candidate is that fit, exactly, in float64. The top-k step is the podcast's code,
    unchanged.
    """
    top_k = top_k or list(cfg.mono["top_k"])
    n_words, n_targets = Y.shape
    X64 = np.asarray(Xband, np.float64)
    Y64 = np.asarray(Y, np.float64)
    full_pred = np.full((n_words, n_targets), np.nan)
    y_std = np.full((n_words, n_targets), np.nan)
    top_pred = {k: np.full((n_words, n_targets), np.nan, dtype=np.float32) for k in top_k}
    edges = np.array([0, X64.shape[1]])
    for f, (tr, te) in enumerate(_folds(cfg, onsets_s)):
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        xs = FoldStandardiser().fit(X64[tr])
        ys = FoldStandardiser().fit(Y64[tr], vtr)
        Xtr, Xte = ((A - xs.mean_) / xs.scale_ for A in (X64[tr], X64[te]))
        Xtr[:, xs.dead_], Xte[:, xs.dead_] = 0.0, 0.0
        Ytr = (Y64[tr] - ys.mean_) / ys.scale_
        Yte = (Y64[te] - ys.mean_) / ys.scale_
        Ytr[:, ys.dead_], Yte[:, ys.dead_] = 0.0, 0.0
        if valid is not None:
            Ytr[~vtr], Yte[~vte] = 0.0, 0.0
        y_std[te] = Yte
        full_pred[te] = banded_ridge_cov(Xtr, Ytr, Xte, edges, cfg.alphas, 1,
                                         int(cfg.cv["n_inner_folds"]), cfg.seed + f)
        # top-k: identical to encoding.semantic_concentration (float32 inputs)
        X32, Xte32 = Xtr.astype(np.float32), Xte.astype(np.float32)
        Y32 = Ytr.astype(np.float32)
        a = X32 - X32.mean(0)
        b = Y32 - Y32.mean(0)
        na = np.sqrt((a ** 2).sum(0))
        nb = np.sqrt((b ** 2).sum(0))
        na[na == 0] = np.inf
        nb[nb == 0] = np.inf
        corr = np.abs((a.T @ b) / np.outer(na, nb))
        for k in top_k:
            kk = min(k, corr.shape[0])
            idx = np.argpartition(-corr, kk - 1, axis=0)[:kk]
            top_pred[k][te] = _batched_subset_ridge(
                X32, Y32, Xte32, idx, cfg.alphas, int(cfg.cv["n_inner_folds"]),
                n_jobs=cfg.n_jobs_subset_ridge)
        log.info("SCI (covariance form) fold %d done", f + 1)
    out = {"r2_all": pooled_r2(y_std, full_pred, valid)}
    for k in top_k:
        out[f"r2_top{k}"] = pooled_r2(y_std, top_pred[k], valid)
        denom = out["r2_all"].copy()
        denom[denom <= 0] = np.nan
        out[f"sci_{k}"] = out[f"r2_top{k}"] / denom
    return out


def semantic_concentration_primal(cfg: Config, Xband: np.ndarray, Y: np.ndarray,
                                  onsets_s: np.ndarray, top_k: list[int] | None = None,
                                  valid: np.ndarray | None = None) -> dict:
    """``encoding.semantic_concentration`` with the full-band fit in primal form."""
    from himalaya.backend import get_backend, set_backend
    from himalaya.ridge import RidgeCV

    top_k = top_k or list(cfg.mono["top_k"])
    set_backend(cfg.model["backend"], on_error="warn")
    be = get_backend()
    n_words, n_targets = Y.shape
    full_pred = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    y_std = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    top_pred = {k: np.full((n_words, n_targets), np.nan, dtype=np.float32) for k in top_k}
    for f, (tr, te) in enumerate(_folds(cfg, onsets_s)):
        xs = FoldStandardiser().fit(Xband[tr])
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        ys = FoldStandardiser().fit(Y[tr], vtr)
        Xtr, Xte = xs.transform(Xband[tr]), xs.transform(Xband[te])
        Ytr, Yte = ys.transform(Y[tr], vtr), ys.transform(Y[te], vte)
        y_std[te] = Yte
        model = RidgeCV(alphas=cfg.alphas, cv=int(cfg.cv["n_inner_folds"]),
                        solver_params=_params(cfg, False))
        model.fit(Xtr, Ytr)
        full_pred[te] = be.to_numpy(model.predict(Xte)).astype(np.float32)

        # top-k: identical to encoding.semantic_concentration
        a = Xtr - Xtr.mean(0)
        b = Ytr - Ytr.mean(0)
        na = np.sqrt((a ** 2).sum(0))
        nb = np.sqrt((b ** 2).sum(0))
        na[na == 0] = np.inf
        nb[nb == 0] = np.inf
        corr = np.abs((a.T @ b) / np.outer(na, nb))
        for k in top_k:
            kk = min(k, corr.shape[0])
            idx = np.argpartition(-corr, kk - 1, axis=0)[:kk]
            top_pred[k][te] = _batched_subset_ridge(
                Xtr, Ytr, Xte, idx, cfg.alphas, int(cfg.cv["n_inner_folds"]),
                n_jobs=cfg.n_jobs_subset_ridge)
        log.info("SCI (primal) fold %d done", f + 1)
    out = {"r2_all": pooled_r2(y_std, full_pred, valid)}
    for k in top_k:
        out[f"r2_top{k}"] = pooled_r2(y_std, top_pred[k], valid)
        denom = out["r2_all"].copy()
        denom[denom <= 0] = np.nan
        out[f"sci_{k}"] = out[f"r2_top{k}"] / denom
    return out
