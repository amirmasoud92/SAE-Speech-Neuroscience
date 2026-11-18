"""Shared machinery for the analyses that fit a single representation at matched capacity.

* **Targets.** Each IFG electrode is read as its cleaned response averaged over lags 0 to
  +500 ms (21 lags), valid only where all 21 lags are valid. Reading each electrode at the
  peak lag of a fitted model (``rule="peak_reduced"`` or ``"peak_full"``) is available as a
  sensitivity analysis.
* **Nuisance.** Surprisal, lexical and acoustic bands (the last reduced to 100 principal
  components) are projected out of design and target with coefficients estimated on training
  rows.
* **Capacity.** Every representation is z-scored per dimension and reduced by PCA, fitted
  without reference to neural data, so that all arms enter with the same dimensionality.
* **Exact masked ridge.** In each fold, targets are grouped by their pattern of valid
  training rows; within a group the nuisance regression and the ridge are fitted on exactly
  those rows. Where the design is wider than the training set the dual solve per pattern is
  too costly, and masked training cells instead enter at the training mean
  (``exact=False``).
* **Penalty.** One ridge penalty per representation, chosen from 1e1 to 1e6 on the pooled
  held-out correlation.
* **Inference.** Electrode-level contrasts use a bootstrap clustered by participant
  (``cluster_boot``).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from encoding.cross_validation import BlockedTimeSeriesSplit
from feature_extraction.feature_bands import dense_residual
from capacity_matching.projected_targets import N_ACOUSTIC_PC

LAMBDAS = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
WINDOW_S = (0.0, 0.5)


# --------------------------------------------------------------------------- targets

def ifg_targets(cfg, rule: str = "window", roi: str = "IFG", vp_root=None,
                digest: str | None = None):
    """Per-electrode targets for one ROI across all subjects.

    rule : "window"      mean over lags in WINDOW_S (primary, no selection)
           "peak_full"   lag of the full model's peak R2 (sensitivity)
           "peak_reduced" lag of the reduced model without the sparse band (sensitivity)
    vp_root : results directory holding encoding/ (for the peak rules)
    digest : target cache to read; default the current config's

    Returns Y (n_words, n_elec), V (n_words, n_elec) bool, subj (n_elec,), meta DataFrame
    """
    from pathlib import Path
    digest = digest or cfg.digest
    vp_root = Path(vp_root) if vp_root is not None else cfg.paths.results_root
    lags = cfg.lags.centres_s
    win = np.flatnonzero((lags >= WINDOW_S[0] - 1e-9) & (lags <= WINDOW_S[1] + 1e-9))
    cols, vals, subj, meta = [], [], [], []
    for sub in cfg.subjects:
        v = np.load(vp_root / "encoding" / f"{sub}_variance_partition.npz",
                    allow_pickle=True)
        keep = np.flatnonzero(v["roi"].astype(str) == roi)
        if len(keep) == 0:
            continue
        z = np.load(cfg.cache(digest, f"targets_{sub}.npz"), allow_pickle=True)
        Y = z["Y"]
        V = z["V"] if "V" in z.files else np.ones(Y.shape, bool)
        if rule == "window":
            for i in keep:
                cols.append(Y[:, i, win].astype(np.float64).mean(1))
                vals.append(V[:, i, win].all(1))
                meta.append({"subject": sub, "ch": str(v["ch_names"][i]), "lag": "0-500ms"})
        else:
            key = "full" if rule == "peak_full" else "r2_without_sae"
            q = v[key]
            peak = np.nanargmax(np.where(np.isfinite(q), q, -np.inf), 1)
            for i in keep:
                cols.append(Y[:, i, peak[i]].astype(np.float64))
                vals.append(V[:, i, peak[i]])
                meta.append({"subject": sub, "ch": str(v["ch_names"][i]),
                             "lag": float(lags[peak[i]])})
        subj += [sub] * len(keep)
        del Y, V
    return (np.column_stack(cols), np.column_stack(vals), np.array(subj),
            pd.DataFrame(meta))


def isolated_partition(iso_cfg, sub, roi: str = "IFG"):
    """Variance partition of an isolated (ROI-restricted) run, or None where the main run
    shows the subject has no electrode in ``roi`` - such a subject writes no file.
    A file missing for a subject that does have ``roi`` electrodes is an unfinished run and
    raises."""
    f = iso_cfg.results("encoding", f"{sub}_variance_partition.npz")
    if f.exists():
        return np.load(f, allow_pickle=True)
    from utils.config import load_config
    from pathlib import Path
    main = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    v = np.load(main.results("encoding", f"{sub}_variance_partition.npz"),
                allow_pickle=True)
    if (v["roi"].astype(str) == roi).any():
        raise FileNotFoundError(f"{f} missing although {sub} has {roi} electrodes")
    return None


# ------------------------------------------------------------------ design and arms

def load_design(cfg):
    d = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)
    return d["X"], json.loads(str(d["slices"]))


def nuisance(cfg):
    """Surprisal, lexical, 100 acoustic PCs, intercept - identical in every analysis."""
    X, sl = load_design(cfg)
    A = X[:, sl["acoustic"][0]:sl["acoustic"][1]].astype(np.float64)
    A -= A.mean(0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return np.column_stack([X[:, sl["surprisal"][0]:sl["surprisal"][1]],
                            X[:, sl["lexical"][0]:sl["lexical"][1]],
                            A @ Vt[:N_ACOUSTIC_PC].T,
                            np.ones(len(X))]).astype(np.float64)


def zpca(M: np.ndarray, k: int | None = None, zscore: bool = True) -> np.ndarray:
    """PCA scores of M after optional per-dimension z-scoring."""
    M = np.asarray(M, dtype=np.float64)
    M = M - M.mean(0)
    if zscore:
        sd = M.std(0)
        M = M / np.where(sd > 1e-12, sd, 1.0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    S = U * s
    return S if k is None else S[:, :k]


def sae_band(cfg) -> np.ndarray:
    X, sl = load_design(cfg)
    return X[:, sl["sae"][0]:sl["sae"][1]].astype(np.float64)


def gemma_dense(cfg, word_idx) -> np.ndarray:
    return dense_residual(cfg, word_idx).astype(np.float64)


def gpt2xl(cfg, word_idx) -> np.ndarray:
    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    with h5py.File(g / "features.hdf5", "r") as h:
        G = h["layer-24"][...]
    gt = pd.read_csv(g / "transcript.tsv", sep="\t")
    return pd.DataFrame(G).groupby(gt["word_idx"].to_numpy()).mean().reindex(
        word_idx).to_numpy().astype(np.float64)


def folds_for(cfg, onsets):
    return list(BlockedTimeSeriesSplit(n_splits=int(cfg.cv["n_outer_folds"]),
                                       embargo_s=float(cfg.cv["embargo_s"])).split(onsets))


# --------------------------------------------------------------------------- ridge

def _patterns(Vtr: np.ndarray):
    """Group target columns by their pattern of valid training rows."""
    keys = {}
    packed = np.packbits(Vtr, axis=0)
    for j in range(Vtr.shape[1]):
        keys.setdefault(packed[:, j].tobytes(), []).append(j)
    return [np.array(v) for v in keys.values()]


def oof(Xr, nuis, Y, V, folds, lambdas=LAMBDAS, exact: bool = True):
    """Out-of-fold ridge predictions with fold-wise nuisance regression.

    Returns P (n_lambdas, n_words, n_targets), Yr (residualised test targets), and the
    number of distinct mask patterns fitted. Masked cells are excluded exactly when
    ``exact`` and the design fits the primal; otherwise they enter at the training mean.
    """
    Xr = np.asarray(Xr, dtype=np.float64)
    n_w, n_e = Y.shape
    k = Xr.shape[1]
    P = np.zeros((len(lambdas), n_w, n_e))
    Yr = np.zeros((n_w, n_e))
    n_pat = 0
    for tr, te in folds:
        groups = _patterns(V[tr]) if exact else [np.arange(n_e)]
        n_pat += len(groups)
        for J in groups:
            rows = tr[V[tr][:, J[0]]] if exact else tr
            Ntr, Nte = nuis[rows], nuis[te]
            G = Ntr.T @ Ntr + 1e-8 * np.eye(Ntr.shape[1])
            bx = np.linalg.solve(G, Ntr.T @ Xr[rows])
            Yg = Y[:, J]
            if not exact:                       # masked cells at the training mean
                Vg = V[:, J]
                w = Vg[tr].astype(np.float64)
                mu = (Yg[tr] * w).sum(0) / np.maximum(w.sum(0), 1)
                Yg = np.where(Vg, Yg, mu)
            by = np.linalg.solve(G, Ntr.T @ Yg[rows])
            Xtr, Xte = Xr[rows] - Ntr @ bx, Xr[te] - Nte @ bx
            Ytr, Yte = Yg[rows] - Ntr @ by, Yg[te] - Nte @ by
            mx = Xtr.mean(0)
            Xtr, Xte = Xtr - mx, Xte - mx
            my = Ytr.mean(0)
            Ytr, Yte = Ytr - my, Yte - my
            if k <= len(rows):
                XtX, XtY = Xtr.T @ Xtr, Xtr.T @ Ytr
                I = np.eye(k)
                for li, lam in enumerate(lambdas):
                    P[li][np.ix_(te, J)] = Xte @ np.linalg.solve(XtX + lam * I, XtY)
            else:
                ev, U = np.linalg.eigh(Xtr @ Xtr.T)
                ev, U = ev[::-1], U[:, ::-1]
                keep = ev > ev.max() * 1e-10
                ev, U = ev[keep], U[:, keep]
                s = np.sqrt(ev)
                Zte = Xte @ ((Xtr.T @ U) / s)
                UtY = U.T @ Ytr
                for li, lam in enumerate(lambdas):
                    P[li][np.ix_(te, J)] = Zte @ ((s[:, None] * UtY) / (ev[:, None] + lam))
            Yr[np.ix_(te, J)] = Yte
    return P, Yr, n_pat


def colcorr(a, b, v=None):
    """Column-wise correlation over valid rows."""
    if v is None:
        v = np.ones(a.shape, bool)
    w = v.astype(np.float64)
    n = np.maximum(w.sum(0), 1.0)
    a = (a - (a * w).sum(0) / n) * w
    b = (b - (b * w).sum(0) / n) * w
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def best_lambda_r(P, Yr, V):
    """Per-arm lambda chosen on the mean held-out r across electrodes.

    The training-only alternative, with the penalty chosen inside each training fold, is in
    ``controls/training_only.py``.
    """
    rs = np.array([colcorr(P[li], Yr, V) for li in range(P.shape[0])])
    li = int(np.argmax(rs.mean(1)))
    return rs[li], li


def cluster_boot(diff, subj, rng, n_boot=10000):
    """Median difference with a subject-clustered bootstrap, two-sided P."""
    uq = np.unique(subj)
    by = {u: diff[subj == u] for u in uq}
    b = np.array([np.median(np.concatenate([by[u] for u in rng.choice(uq, len(uq), True)]))
                  for _ in range(n_boot)])
    return (float(np.median(diff)), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float(2 * min((b <= 0).mean(), (b >= 0).mean())),
            b)
