"""Phase-randomised surrogate nulls for the unique-variance statistics.

* **Hyperparameters are re-selected for every surrogate.** Band weights and ridge penalties are
  searched afresh on each surrogate's training folds, exactly as for the real data.
* **Kernels are computed once per fold and reused across surrogates.** A surrogate changes only
  ``Y``; the band kernels ``K_b = X_b X_b^T`` depend only on ``X`` and the fold indices.
* **Surrogates are generated in the log domain.** Fourier phase randomisation is applied to
  ``log(envelope)`` and exponentiated back, giving a strictly positive surrogate whose
  log-domain power spectrum matches the original, which is then windowed and logged by the
  same function used for the real targets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from utils.config import Config, LagGrid
from himalaya.backend import set_backend

from encoding.cross_validation import BlockedTimeSeriesSplit, block_circular_shift, phase_randomise
from preprocessing.neural_targets import SubjectData, build_targets
from encoding.banded_ridge import FoldStandardiser, pooled_r2, to_numpy
from feature_extraction.feature_bands import Design

log = logging.getLogger("nulls")


# --------------------------------------------------------------- surrogate signals

def phase_randomised_subject(sd: SubjectData, rng: np.random.Generator) -> SubjectData:
    """A surrogate recording with the log-envelope power spectrum preserved.

    Parameters
    ----------
    sd : SubjectData
        Real recording; ``sd.data`` is (n_channels, n_times) strictly positive.
    rng : numpy Generator

    Returns
    -------
    SubjectData with surrogate ``data``, strictly positive, same shape and dtype.
    """
    if (sd.data <= 0).any():
        raise ValueError(f"{sd.subject}: non-positive envelope samples")
    surro = np.exp(phase_randomise(np.log(sd.data.astype(np.float64)), rng))
    if not np.isfinite(surro).all() or (surro <= 0).any():
        raise ValueError(f"{sd.subject}: surrogate is not strictly positive and finite")
    return SubjectData(sd.subject, surro.astype(np.float32), sd.ch_names, sd.sfreq,
                       sd.coords_mm)


def circular_shift_design(design: Design, onsets_s: np.ndarray, shift_s: float,
                          total_s: float) -> Design:
    """Realign the whole design matrix to the wrong words by a circular time shift.

    Preserves the autocorrelation of both the features and the targets while destroying
    their correspondence. Assumption-free relative to phase randomisation, which assumes
    the surrogate's Gaussian-ish spectral structure is the right null.
    """
    idx = block_circular_shift(onsets_s, shift_s, total_s)
    return Design(np.ascontiguousarray(design.X[idx]), dict(design.slices),
                  dict(design.feature_ids))


# ------------------------------------------------------------- cached-kernel fitting

@dataclass
class FoldKernels:
    """Band kernels for one fold. Depends on X and the split only, never on Y."""
    train: np.ndarray          # (n_train,) indices
    test: np.ndarray           # (n_test,) indices
    k_train: np.ndarray        # (n_bands, n_train, n_train)
    k_test: np.ndarray         # (n_bands, n_test, n_train)
    bands: list[str]


def precompute_kernels(cfg: Config, design: Design,
                       onsets_s: np.ndarray) -> list[FoldKernels]:
    """Compute every fold's band kernels once.

    The feature standardiser is fitted on each training fold, exactly as in the real fit,
    so the kernels are the same matrices ``fit_banded`` would build internally.
    """
    splitter = BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_outer_folds"]),
                                      embargo_s=float(cfg.cv["embargo_s"]))
    bands = list(design.slices)
    out: list[FoldKernels] = []
    for tr, te in splitter.split(onsets_s):
        xs = FoldStandardiser().fit(design.X[tr])
        Xtr, Xte = xs.transform(design.X[tr]), xs.transform(design.X[te])
        ktr = np.stack([Xtr[:, design.slices[b]] @ Xtr[:, design.slices[b]].T
                        for b in bands]).astype(np.float32)
        kte = np.stack([Xte[:, design.slices[b]] @ Xtr[:, design.slices[b]].T
                        for b in bands]).astype(np.float32)
        out.append(FoldKernels(tr, te, ktr, kte, bands))
    log.info("precomputed kernels for %d folds, %d bands, %.1f GB",
             len(out), len(bands),
             sum(f.k_train.nbytes + f.k_test.nbytes for f in out) / 1e9)
    return out


def _fit_precomputed(cfg: Config, fk: FoldKernels, Ytr: np.ndarray,
                     drop: str | None, seed: int) -> np.ndarray:
    """Fit banded ridge on precomputed kernels with a full hyperparameter search.

    ``drop`` removes one band, for the reduced model of a variance partition.
    Returns test-set predictions, (n_test, n_targets).
    """
    from himalaya.kernel_ridge import MultipleKernelRidgeCV

    keep = [i for i, b in enumerate(fk.bands) if b != drop]
    m = cfg.model
    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=m["solver"],
        solver_params=dict(n_iter=int(m["n_iter"]), alphas=cfg.alphas,
                           n_targets_batch=int(m["n_targets_batch"]),
                           n_alphas_batch=int(m["n_alphas_batch"]),
                           n_targets_batch_refit=int(m["n_targets_batch_refit"])),
        cv=int(cfg.cv["n_inner_folds"]),
        random_state=seed,
    )
    model.fit(fk.k_train[keep], Ytr)
    return to_numpy(model.predict(fk.k_test[keep])).astype(np.float32)


def delta_r2_one_pass(cfg: Config, kernels: list[FoldKernels], Y: np.ndarray,
                      bands: str | list[str], seed: int) -> dict[str, np.ndarray]:
    """Unique held-out variance of one or more bands, for one target set.

    The full model is fitted **once** and shared across every requested band, so asking
    for two bands costs three fits per fold rather than four.

    Parameters
    ----------
    Y : (n_words, n_targets) float32, unstandardised targets.
    bands : band name, or list of them.

    Returns
    -------
    dict mapping band -> (n_targets,) float64, ``R2(full) - R2(full without band)``,
    pooled over folds.
    """
    # The backend is global state in himalaya. Without this the null silently runs on
    # numpy/CPU while the real fits ran on CUDA - identical numbers, but roughly thirty
    # times slower, with no warning of any kind.
    set_backend(cfg.model["backend"], on_error="warn")

    band_list = [bands] if isinstance(bands, str) else list(bands)
    n_words, n_targets = Y.shape
    y_true = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    p_full = np.full((n_words, n_targets), np.nan, dtype=np.float32)
    p_red = {b: np.full((n_words, n_targets), np.nan, dtype=np.float32)
             for b in band_list}

    for k, fk in enumerate(kernels):
        ys = FoldStandardiser().fit(Y[fk.train])
        Ytr, Yte = ys.transform(Y[fk.train]), ys.transform(Y[fk.test])
        y_true[fk.test] = Yte
        p_full[fk.test] = _fit_precomputed(cfg, fk, Ytr, None, seed + k)
        for b in band_list:
            p_red[b][fk.test] = _fit_precomputed(cfg, fk, Ytr, b, seed + k)

    r2_full = pooled_r2(y_true, p_full)
    return {b: r2_full - pooled_r2(y_true, p_red[b]) for b in band_list}


def run_phase_null(cfg: Config, design: Design, sd: SubjectData, onsets_s: np.ndarray,
                   band: str, n_perm: int, lags: LagGrid | None = None,
                   seed: int | None = None,
                   kernels: list[FoldKernels] | None = None) -> np.ndarray:
    """Null distribution of ``delta_r2`` for ``band`` under phase-randomised neural data.

    Returns
    -------
    (n_perm, n_channels, n_lags) float32
    """
    lags = lags or cfg.lags
    seed = cfg.seed if seed is None else seed
    kernels = kernels if kernels is not None else precompute_kernels(cfg, design, onsets_s)
    n_ch = len(sd.ch_names)

    out = np.empty((n_perm, n_ch, lags.n_lags), dtype=np.float32)
    for i in range(n_perm):
        rng = np.random.default_rng([seed, i])
        surro = phase_randomised_subject(sd, rng)
        Ys = build_targets(cfg, surro, onsets_s, lags)
        d = delta_r2_one_pass(cfg, kernels, Ys.reshape(len(onsets_s), -1), band,
                              seed + 1000 * (i + 1))[band]
        out[i] = d.reshape(n_ch, lags.n_lags)
        log.info("%s phase-null %d/%d: delta R2 median %.6f p95 %.6f", sd.subject,
                 i + 1, n_perm, float(np.median(out[i])),
                 float(np.percentile(out[i], 95)))
    return out
