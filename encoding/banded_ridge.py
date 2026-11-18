"""Banded-ridge encoding models, variance partitioning and the concentration statistic.

High-gamma at each electrode and lag is predicted by multiple-kernel ridge regression in
which each feature band receives its own weight (himalaya ``MultipleKernelRidgeCV``, random
search over 20 kernel-weight samples and 12 penalties from 1e-3 to 1e8, chosen by three-fold
inner cross-validation). Unique variance of a band (delta R2) is the loss in held-out R2 when
that band is removed.

The fold loop is explicit, so that every fitted quantity - per-channel neural means and SDs,
per-feature means and SDs, PCA bases, top-k feature selection, ridge penalties and band
weights - is estimated on the training fold alone. Scores use held-out predictions **pooled
across folds and correlated once**. Masked target cells enter training at the training mean
and are excluded from every score.

The concentration statistic for a dictionary is R2(top-k features) / R2(all features), with
the top k features selected inside each training fold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from himalaya.backend import get_backend, set_backend
from joblib import Parallel, delayed
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from sklearn.pipeline import make_pipeline

from utils.config import Config
from encoding.cross_validation import BlockedTimeSeriesSplit
from feature_extraction.feature_bands import Design

log = logging.getLogger("encoding")


# --------------------------------------------------------------------------- helpers

class FoldStandardiser:
    """Z-score fitted on a training fold and applied unchanged to the test fold.

    Columns with zero training variance (an SAE feature that happens to fire on no
    training word) are mapped to exactly zero rather than dividing by zero.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.dead_: np.ndarray | None = None

    def fit(self, X: np.ndarray, valid: np.ndarray | None = None) -> "FoldStandardiser":
        """``valid``: optional bool mask; statistics use valid cells only."""
        if valid is None:
            self.mean_ = X.mean(0)
            sd = X.std(0)
        else:
            w = valid.astype(np.float64)
            n = np.maximum(w.sum(0), 1.0)
            self.mean_ = (X * w).sum(0) / n
            sd = np.sqrt((((X - self.mean_) ** 2) * w).sum(0) / n)
        self.dead_ = sd < 1e-12
        self.scale_ = np.where(self.dead_, 1.0, sd)
        return self

    def transform(self, X: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Masked cells are set to exactly 0, the training mean in standardised units."""
        out = (X - self.mean_) / self.scale_
        out[:, self.dead_] = 0.0
        if valid is not None:
            out[~valid] = 0.0
        return out.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, valid).transform(X, valid)


def to_numpy(x: object) -> np.ndarray:
    """Bring a himalaya result back to host memory.

    The ``torch_cuda`` backend returns CUDA tensors, which ``np.asarray`` refuses to
    convert. Every value crossing back out of himalaya goes through here.
    """
    try:
        return np.asarray(get_backend().to_numpy(x))
    except Exception:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)


def pooled_r(y_true: np.ndarray, y_pred: np.ndarray,
             valid: np.ndarray | None = None) -> np.ndarray:
    """Column-wise Pearson r between two (n_samples, n_targets) arrays.

    ``valid``: optional bool mask; masked cells are excluded from scoring.
    """
    if valid is not None:
        w = valid.astype(np.float64)
        n = np.maximum(w.sum(0), 1.0)
        a = (y_true - (y_true * w).sum(0) / n) * w
        b = (y_pred - (y_pred * w).sum(0) / n) * w
        denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        out = np.zeros(a.shape[1], dtype=np.float64)
        ok = denom > 0
        out[ok] = (a[:, ok] * b[:, ok]).sum(0) / denom[ok]
        return out
    a = y_true - y_true.mean(0)
    b = y_pred - y_pred.mean(0)
    denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    out = np.zeros(a.shape[1], dtype=np.float64)
    ok = denom > 0
    out[ok] = (a[:, ok] * b[:, ok]).sum(0) / denom[ok]
    return out


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray,
              valid: np.ndarray | None = None) -> np.ndarray:
    """Column-wise coefficient of determination, 1 - SS_res / SS_tot.

    Can be negative when the model predicts worse than the held-out mean, which is the
    honest outcome for an uninformative model and is not clipped. ``valid``:
    optional bool mask; masked cells are excluded from both sums.
    """
    if valid is not None:
        w = valid.astype(np.float64)
        n = np.maximum(w.sum(0), 1.0)
        ss_res = (((y_true - y_pred) ** 2) * w).sum(0)
        ss_tot = (((y_true - (y_true * w).sum(0) / n) ** 2) * w).sum(0)
        out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
        ok = ss_tot > 0
        out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
        return out
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    ok = ss_tot > 0
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


# --------------------------------------------------------------------------- fitting

@dataclass
class FitResult:
    """Held-out predictions and diagnostics for one model over all folds."""
    y_pred: np.ndarray                       # (n_words, n_targets) pooled held-out
    y_true: np.ndarray                       # (n_words, n_targets) fold-standardised
    fold_of: np.ndarray                      # (n_words,) which fold each word was tested in
    n_train: list[int] = field(default_factory=list)
    deltas: list[np.ndarray] = field(default_factory=list)   # band weights per fold
    best_alphas: list[np.ndarray] = field(default_factory=list)
    valid: np.ndarray | None = None          # (n_words, n_targets) bool

    def r(self) -> np.ndarray:
        return pooled_r(self.y_true, self.y_pred, self.valid)

    def r2(self) -> np.ndarray:
        return pooled_r2(self.y_true, self.y_pred, self.valid)


def _make_model(cfg: Config, band_names: list[str], n_inner: int, seed: int):
    m = cfg.model
    return MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=m["solver"],
        solver_params=dict(n_iter=int(m["n_iter"]),
                           alphas=cfg.alphas,
                           n_targets_batch=int(m["n_targets_batch"]),
                           n_alphas_batch=int(m["n_alphas_batch"]),
                           n_targets_batch_refit=int(m["n_targets_batch_refit"])),
        cv=n_inner,
        random_state=seed,
    )


class FoldPCA:
    """PCA on one band, fitted on the training fold only.

    Used for the dense-embedding baseline. The SAE was trained on a different corpus and
    has never seen this transcript, so fitting the dense basis on the full transcript
    would give it an advantage the sparse basis does not have. Refitting per fold removes
    the asymmetry, at the cost of a slightly different basis in each fold - which is the
    correct trade, since the comparison between the two bases is the whole point.
    """

    def __init__(self, n_components: int, seed: int = 0) -> None:
        self.n_components = n_components
        self.seed = seed

    def fit_transform(self, raw_tr: np.ndarray,
                      raw_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.decomposition import PCA
        n = min(self.n_components, raw_tr.shape[0], raw_tr.shape[1])
        p = PCA(n_components=n, svd_solver="randomized", random_state=self.seed)
        return (p.fit_transform(raw_tr).astype(np.float32),
                p.transform(raw_te).astype(np.float32))


def make_deferred_designer(base: Design, deferred: dict[str, tuple[np.ndarray, FoldPCA]]):
    """Build a per-fold design where some bands are fitted inside the training fold.

    Parameters
    ----------
    base : Design
        Bands that need no fold-wise fitting.
    deferred : mapping band -> (raw features (n_words, d_raw), transformer)

    Returns
    -------
    callable ``(train_idx, test_idx) -> (Xtr, Xte, slices)``
    """
    def designer(tr: np.ndarray, te: np.ndarray):
        parts_tr, parts_te, slices, off = [], [], {}, 0
        for b in base.slices:
            blk = base.band(b)
            parts_tr.append(blk[tr]); parts_te.append(blk[te])
            slices[b] = slice(off, off + blk.shape[1]); off += blk.shape[1]
        for b, (raw, tf) in deferred.items():
            a, c = tf.fit_transform(raw[tr], raw[te])
            parts_tr.append(a); parts_te.append(c)
            slices[b] = slice(off, off + a.shape[1]); off += a.shape[1]
        return (np.ascontiguousarray(np.concatenate(parts_tr, axis=1), dtype=np.float32),
                np.ascontiguousarray(np.concatenate(parts_te, axis=1), dtype=np.float32),
                slices)
    return designer


def fit_banded(cfg: Config, design: Design, Y: np.ndarray, onsets_s: np.ndarray,
               seed: int | None = None, label: str = "",
               fold_designer=None, valid: np.ndarray | None = None) -> FitResult:
    """Fit a banded-ridge encoding model with chronologically blocked CV.

    Parameters
    ----------
    design : Design
        (n_words, n_features) with one slice per band.
    Y : (n_words, n_targets) float32
        Flattened neural targets. For the main analysis, ``n_targets = n_channels * n_lags``
        with C-order flattening so ``Y.reshape(n_words, n_channels, n_lags)`` recovers it.
    onsets_s : (n_words,) word onset times, used to build the folds.

    valid : optional (n_words, n_targets) bool
        Artifact mask from ``artifact_cleaning``. Target statistics use valid training cells
        only; masked training cells enter the fit at exactly the training mean (zero after
        standardisation); masked cells are excluded from scoring. With ~3% of cells masked
        this zero-imputation attenuates coefficients slightly relative to an exact masked
        ridge - toward the null, never away from it - and the size of that attenuation is
        measured where the exact solution is cheap (the matched-capacity analyses).

    Returns
    -------
    FitResult
    """
    seed = cfg.seed if seed is None else seed
    set_backend(cfg.model["backend"], on_error="warn")

    splitter = BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_outer_folds"]),
                                      embargo_s=float(cfg.cv["embargo_s"]))
    band_names = list(design.slices)
    n_words, n_targets = Y.shape

    res = FitResult(y_pred=np.full((n_words, n_targets), np.nan, dtype=np.float32),
                    y_true=np.full((n_words, n_targets), np.nan, dtype=np.float32),
                    fold_of=np.full(n_words, -1, dtype=np.int16))

    for k, (tr, te) in enumerate(splitter.split(onsets_s)):
        if fold_designer is None:
            raw_tr, raw_te, slices = design.X[tr], design.X[te], design.slices
        else:
            raw_tr, raw_te, slices = fold_designer(tr, te)
        band_names = list(slices)

        xs = FoldStandardiser().fit(raw_tr)
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        ys = FoldStandardiser().fit(Y[tr], vtr)
        Xtr, Xte = xs.transform(raw_tr), xs.transform(raw_te)
        Ytr, Yte = ys.transform(Y[tr], vtr), ys.transform(Y[te], vte)

        kernelizers = [(b, Kernelizer(kernel="linear"), slices[b])
                       for b in band_names]
        model = make_pipeline(ColumnKernelizer(kernelizers),
                              _make_model(cfg, band_names, int(cfg.cv["n_inner_folds"]),
                                          seed + k))
        model.fit(Xtr, Ytr)
        pred = to_numpy(model.predict(Xte)).astype(np.float32)

        res.y_pred[te] = pred
        res.y_true[te] = Yte
        res.fold_of[te] = k
        res.n_train.append(len(tr))
        mkr = model[-1]
        res.deltas.append(to_numpy(getattr(mkr, "deltas_", np.empty(0))))
        res.best_alphas.append(to_numpy(getattr(mkr, "best_alphas_", np.empty(0))))
        log.info("%s fold %d/%d: train=%d test=%d", label or "model", k + 1,
                 splitter.n_splits, len(tr), len(te))

    if (res.fold_of < 0).any():
        raise AssertionError("some words were never in a test fold")
    res.valid = valid
    return res


def variance_partition(cfg: Config, design: Design, Y: np.ndarray, onsets_s: np.ndarray,
                       bands: list[str] | None = None,
                       keep_full_fit: dict | None = None,
                       valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Unique held-out variance per band.

    For each band b, ``delta_r2[b] = R2(full) - R2(full without b)``, both evaluated on
    pooled held-out predictions. Shared variance is *not* assigned to any band; it is
    recoverable as ``R2(full) - sum(delta_r2)`` and reported separately.

    Returns
    -------
    dict with ``"full"`` (n_targets,) and one ``"delta_<band>"`` entry per band, plus
    ``"shared"``.
    """
    bands = bands or list(design.slices)
    out: dict[str, np.ndarray] = {}

    full = fit_banded(cfg, design, Y, onsets_s, label="full", valid=valid)
    out["full"] = full.r2()
    out["full_r"] = full.r()
    if keep_full_fit is not None:
        # Held-out predictions are expensive to regenerate (hours) and cheap to store, so
        # any further metric can be computed without refitting.
        keep_full_fit["y_pred"] = full.y_pred
        keep_full_fit["y_true"] = full.y_true
        keep_full_fit["fold_of"] = full.fold_of
        keep_full_fit["n_train"] = np.asarray(full.n_train)
        keep_full_fit["deltas"] = np.stack(full.deltas) if full.deltas else np.empty(0)
        keep_full_fit["best_alphas"] = (np.stack(full.best_alphas)
                                        if full.best_alphas else np.empty(0))
        if valid is not None:
            keep_full_fit["valid"] = valid

    for b in bands:
        reduced = fit_banded(cfg, design.without(b), Y, onsets_s, label=f"minus-{b}",
                             valid=valid)
        out[f"delta_{b}"] = out["full"] - reduced.r2()
        out[f"r2_without_{b}"] = reduced.r2()
        log.info("delta R2 for band %-10s: median %.5f, p95 %.5f", b,
                 float(np.nanmedian(out[f"delta_{b}"])),
                 float(np.nanpercentile(out[f"delta_{b}"], 95)))

    out["shared"] = out["full"] - sum(out[f"delta_{b}"] for b in bands)
    return out


# --------------------------------------------------- semantic concentration index

def _subset_ridge_chunk(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                        cols: np.ndarray, c0: int, c1: int, alphas: np.ndarray,
                        inner_edges: np.ndarray) -> np.ndarray:
    """One chunk of :func:`_batched_subset_ridge`: targets ``[c0, c1)``.

    Split out of the loop body unchanged so that it can be dispatched to a worker
    process. It reads only its own columns of ``Ytr`` and its own ``cols``, and returns
    its own slice of the prediction matrix, so chunks share no state and their results
    do not depend on how they are scheduled.
    """
    k = cols.shape[0]
    n_train = Xtr.shape[0]
    eye = np.eye(k, dtype=np.float64)
    n_inner = len(inner_edges) - 1

    # Xtr[:, cols] is (n_train, k, m); we want (m, n_train, k) so that every
    # target is a separate slab in a batched solve.
    A = np.transpose(Xtr[:, cols], (2, 0, 1)).astype(np.float64)
    B = np.transpose(Xte[:, cols], (2, 0, 1)).astype(np.float64)
    y = Ytr[:, c0:c1].T.astype(np.float64)[:, :, None]    # (m, n_train, 1)
    m = c1 - c0

    # ---- inner CV to pick alpha per target -------------------------------
    err = np.zeros((len(alphas), m))
    for i in range(n_inner):
        lo, hi = inner_edges[i], inner_edges[i + 1]
        sel = np.ones(n_train, bool); sel[lo:hi] = False
        At, Av = A[:, sel], A[:, ~sel]
        yt, yv = y[:, sel], y[:, ~sel]
        G = At.transpose(0, 2, 1) @ At
        r = At.transpose(0, 2, 1) @ yt
        for ai, a in enumerate(alphas):
            beta = np.linalg.solve(G + a * eye, r)
            err[ai] += ((Av @ beta - yv) ** 2).sum(axis=(1, 2))
    best = alphas[np.argmin(err, axis=0)]                 # (m,)

    # ---- refit on the whole training fold --------------------------------
    G = A.transpose(0, 2, 1) @ A
    r = A.transpose(0, 2, 1) @ y
    beta = np.linalg.solve(G + best[:, None, None] * eye, r)   # (m, k, 1)
    return (B @ beta)[:, :, 0].T.astype(np.float32)


def _batched_subset_ridge(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
                          idx: np.ndarray, alphas: np.ndarray, n_inner: int,
                          chunk: int = 512, n_jobs: int = 1) -> np.ndarray:
    """Ridge on a *different* small feature subset per target, solved in batch.

    Each target ``t`` gets its own ``k`` columns ``idx[:, t]``. Because ``k`` is small
    (5-20), each target's normal equations are a ``k x k`` system, and thousands of them
    solve together with ``np.linalg.solve`` on a stacked array. Fitting a separate kernel
    ridge per target - which is what a naive implementation does - would mean thousands of
    model fits per fold and is not tractable.

    Alpha is chosen per target by an inner contiguous split of the training fold.

    Parameters
    ----------
    Xtr, Xte : (n_train, n_features), (n_test, n_features) standardised
    Ytr : (n_train, n_targets) standardised
    idx : (k, n_targets) int - column indices per target
    alphas : (n_alphas,) ridge penalties to search
    n_inner : number of contiguous inner folds for alpha selection

    Returns
    -------
    (n_test, n_targets) float32 predictions
    """
    k, n_targets = idx.shape
    n_train = Xtr.shape[0]
    out = np.empty((Xte.shape[0], n_targets), dtype=np.float32)

    inner_edges = np.linspace(0, n_train, n_inner + 1).astype(int)
    spans = [(c0, min(c0 + chunk, n_targets)) for c0 in range(0, n_targets, chunk)]

    if n_jobs == 1:
        for c0, c1 in spans:
            out[:, c0:c1] = _subset_ridge_chunk(
                Xtr, Ytr, Xte, idx[:, c0:c1], c0, c1, alphas, inner_edges)
        return out

    # Processes, not threads: the dominant cost is the gather in ``Xtr[:, cols]`` and
    # the float64 copies that follow it, and NumPy advanced indexing holds the GIL, so
    # a thread pool would serialise on exactly the expensive part. ``max_nbytes`` lets
    # joblib memory-map the large read-only inputs once per call instead of pickling
    # them per task.
    parts = Parallel(n_jobs=n_jobs, max_nbytes="16M")(
        delayed(_subset_ridge_chunk)(
            Xtr, Ytr, Xte, idx[:, c0:c1], c0, c1, alphas, inner_edges)
        for c0, c1 in spans)
    for (c0, c1), part in zip(spans, parts):
        out[:, c0:c1] = part
    return out


def semantic_concentration(cfg: Config, design: Design, Y: np.ndarray,
                           onsets_s: np.ndarray, band: str = "sae",
                           top_k: list[int] | None = None,
                           valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """SCI_k = R2(top-k features of `band`) / R2(all features of `band`), held out.

    The top-k features are chosen **inside each training fold** by the magnitude of their
    univariate correlation with the training target, so no test-fold information enters
    the selection. The denominator uses the same folds and the same band.

    ``valid``: optional (n_words, n_targets) mask. Targets are standardised on
    valid training cells, masked training cells enter every fit (and the top-k selection)
    at the training mean, and masked test cells are excluded from every score.

    Returns
    -------
    dict mapping ``"sci_<k>"`` -> (n_targets,), plus ``"r2_all"`` and ``"r2_top<k>"``.
    """
    top_k = top_k or list(cfg.mono["top_k"])
    splitter = BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_outer_folds"]),
                                      embargo_s=float(cfg.cv["embargo_s"]))
    set_backend(cfg.model["backend"], on_error="warn")

    n_words, n_targets = Y.shape
    band_only = design.only([band])
    full_pred = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    y_std = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    top_pred = {k: np.full((n_words, n_targets), np.nan, dtype=np.float32) for k in top_k}

    for f, (tr, te) in enumerate(splitter.split(onsets_s)):
        xs = FoldStandardiser().fit(band_only.X[tr])
        vtr = None if valid is None else valid[tr]
        vte = None if valid is None else valid[te]
        ys = FoldStandardiser().fit(Y[tr], vtr)
        Xtr, Xte = xs.transform(band_only.X[tr]), xs.transform(band_only.X[te])
        Ytr, Yte = ys.transform(Y[tr], vtr), ys.transform(Y[te], vte)
        y_std[te] = Yte

        model = make_pipeline(
            ColumnKernelizer([(band, Kernelizer(kernel="linear"),
                               slice(0, Xtr.shape[1]))]),
            _make_model(cfg, [band], int(cfg.cv["n_inner_folds"]), cfg.seed + f))
        model.fit(Xtr, Ytr)
        full_pred[te] = to_numpy(model.predict(Xte)).astype(np.float32)

        # Univariate |r| between each feature and each target, on the training fold only.
        a = Xtr - Xtr.mean(0)
        b = Ytr - Ytr.mean(0)
        na = np.sqrt((a ** 2).sum(0))
        nb = np.sqrt((b ** 2).sum(0))
        na[na == 0] = np.inf
        nb[nb == 0] = np.inf
        corr = np.abs((a.T @ b) / np.outer(na, nb))        # (n_features, n_targets)

        for k in top_k:
            kk = min(k, corr.shape[0])
            idx = np.argpartition(-corr, kk - 1, axis=0)[:kk]     # (k, n_targets)
            top_pred[k][te] = _batched_subset_ridge(
                Xtr, Ytr, Xte, idx, cfg.alphas, int(cfg.cv["n_inner_folds"]),
                n_jobs=cfg.n_jobs_subset_ridge)
        log.info("SCI fold %d/%d done", f + 1, splitter.n_splits)

    out = {"r2_all": pooled_r2(y_std, full_pred, valid)}
    for k in top_k:
        out[f"r2_top{k}"] = pooled_r2(y_std, top_pred[k], valid)
        denom = out["r2_all"].copy()
        # SCI is undefined where the full band explains no held-out variance.
        denom[denom <= 0] = np.nan
        out[f"sci_{k}"] = out[f"r2_top{k}"] / denom
    return out


def random_dictionary_features(resid: np.ndarray, densities: np.ndarray,
                               rng: np.random.Generator) -> np.ndarray:
    """Density-matched control dictionary: random encoder, per-feature thresholds.

    This is the primary control for H2. A learned SAE and this share the input, the
    dictionary width, and - after this function - the *per-feature* activation density,
    so they differ only in whether the dictionary was trained.

    Parameters
    ----------
    resid : (n_words, d_model) residual-stream activations
    densities : (d_sae,) int - for each real feature, the number of words on which it is
        active. Feature ``j`` of the control is thresholded to fire on exactly
        ``densities[j]`` words.

    Returns
    -------
    (n_words, d_sae) float32, non-negative, whose per-feature active-word counts equal
    ``densities`` exactly, and whose mean word-level L0 therefore matches the real
    dictionary's exactly as well.

    Why per-feature and not one global threshold
    --------------------------------------------
    A single global threshold matched to the *mean* L0 matches the mean and nothing else.
    The columns of a random projection have heterogeneous variance, so a global threshold
    gives a very uneven density distribution, with many control features active on only a
    few words. A feature firing on one or two words becomes a large leverage spike once
    standardised within a fold, so ridge fits it in training and fails out of sample, and a
    higher concentration statistic for the learned dictionary would then follow from the
    control's regression collapsing rather than from learned structure. Matching density
    per feature removes the asymmetry at source: every control feature inherits its real
    counterpart's density and so clears the same ``min_active_words`` floor.
    """
    n_words, d_model = resid.shape
    densities = np.asarray(densities).astype(int)
    d_sae = densities.size
    if densities.min() < 0 or densities.max() >= n_words:
        raise ValueError(f"densities must lie in [0, {n_words}); got "
                         f"[{densities.min()}, {densities.max()}]")

    W = rng.standard_normal((d_model, d_sae)).astype(np.float32) / np.sqrt(d_model)
    pre = resid @ W
    feats = np.zeros_like(pre, dtype=np.float32)

    for j, n_j in enumerate(densities):
        if n_j <= 0:
            continue
        # Threshold at the (n_j + 1)-th largest value, so exactly n_j entries exceed it.
        v = pre[:, j]
        thr = np.partition(v, n_words - n_j - 1)[n_words - n_j - 1]
        feats[:, j] = np.maximum(v - thr, 0.0)

    got = (feats > 0).sum(0)
    if not np.array_equal(got, densities):
        bad = int((got != densities).sum())
        raise AssertionError(
            f"per-feature density mismatch on {bad} of {d_sae} features - ties in the "
            f"random projection are the only possible cause and should not occur")
    return feats
