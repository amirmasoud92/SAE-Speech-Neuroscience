"""Stimulus-only word-timing features.

Per word, within a session: log gaps to the first to fourth preceding and following word
onsets, log durations of words n-1, n and n+1, and the log local word rate (10-word moving
mean of the onset gaps); with ``quadratic``, their squares and pairwise products are added
(12 + 12 + 66 = 90 columns). Values are clipped to [0.02, 10] s before the log, and missing
values at sequence edges are filled with the column median.
"""
from __future__ import annotations

import numpy as np

N_GAPS = 4


def timing_features(onset_s: np.ndarray, session: np.ndarray, duration_s: np.ndarray, quadratic: bool = True) -> np.ndarray:
    """(n_words, 12) or, with ``quadratic``, (n_words, 12 + 12 + 66) stimulus-only timing features."""
    n = len(onset_s); X = np.full((n, 12), np.nan)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s); order = idx[np.argsort(onset_s[idx], kind="stable")]
        t = onset_s[order]; dur = np.asarray(duration_s, float)[order]
        F = np.full((len(order), 12), np.nan)
        for k in range(1, N_GAPS + 1):
            F[k:, k - 1] = t[k:] - t[:-k]                      # gap to the k-th preceding onset
            F[:-k, N_GAPS + k - 1] = t[k:] - t[:-k]            # gap to the k-th following onset
        F[:, 8] = dur; F[1:, 9] = dur[:-1]; F[:-1, 10] = dur[1:]
        gaps = np.r_[np.diff(t), np.median(np.diff(t))] if len(t) > 1 else np.array([0.3])
        F[:, 11] = np.convolve(gaps, np.ones(10) / 10, mode="same")
        X[order] = F
    X = np.log(np.clip(X, 0.02, 10.0))
    med = np.nanmedian(X, 0); X = np.where(np.isfinite(X), X, med)
    if not quadratic:
        return X.astype(np.float32)
    cols = [X, X ** 2] + [X[:, i] * X[:, j] for i in range(12) for j in range(i + 1, 12)]
    return np.column_stack(cols).astype(np.float32)
