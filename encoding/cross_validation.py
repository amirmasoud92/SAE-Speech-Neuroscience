"""Chronologically blocked cross-validation with an embargo.

The narrative and the neural signal are both strongly autocorrelated, and the +/-2 s epochs
of neighbouring words overlap in the same samples, so random K-fold is invalid. Two guards
are used:

1. **Contiguous blocks in stimulus time.** Folds are contiguous spans of the narrative, split
   on word *onset time*. The transcript is stored in speaker-turn order and some rows run
   backwards in time, so splitting on row index is not the same as splitting on time.
2. **An embargo.** Training words whose onset falls within ``embargo_s`` (10 s) of any
   test-block word are dropped from the training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class BlockedTimeSeriesSplit:
    """Contiguous chronological K-fold with a symmetric temporal embargo.

    Parameters
    ----------
    n_splits : int
        Number of contiguous folds. With 1,800 s of narrative and ``n_splits=10`` each
        test block spans about 180 s.
    embargo_s : float
        Training words within this many seconds of the test block's time span are
        excluded from training. Must be at least the half-width of the epoch window.
    """

    n_splits: int = 10
    embargo_s: float = 10.0

    def split(self, onsets_s: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` index arrays into ``onsets_s``.

        Parameters
        ----------
        onsets_s : (n_words,) float array
            Word onset times in seconds. Need not be sorted; ordering is handled here.

        Yields
        ------
        train_idx, test_idx : int arrays
            Indices into the *original* ``onsets_s`` order, so they can be applied
            directly to feature and target matrices built in that same order.
        """
        onsets_s = np.asarray(onsets_s, dtype=float)
        if onsets_s.ndim != 1:
            raise ValueError("onsets_s must be 1-D")
        n = onsets_s.size
        if self.n_splits < 2 or self.n_splits > n:
            raise ValueError(f"n_splits must be in [2, {n}], got {self.n_splits}")

        order = np.argsort(onsets_s, kind="stable")
        sorted_onsets = onsets_s[order]
        edges = np.linspace(0, n, self.n_splits + 1).astype(int)

        for i in range(self.n_splits):
            lo, hi = edges[i], edges[i + 1]
            test_idx = order[lo:hi]
            t_start, t_end = sorted_onsets[lo], sorted_onsets[hi - 1]

            too_close = ((onsets_s >= t_start - self.embargo_s)
                         & (onsets_s <= t_end + self.embargo_s))
            is_test = np.zeros(n, dtype=bool)
            is_test[test_idx] = True
            train_idx = np.flatnonzero(~is_test & ~too_close)
            yield train_idx, test_idx

    def get_n_splits(self, *_: object) -> int:
        return self.n_splits


def block_circular_shift(onsets_s: np.ndarray, shift_s: float,
                         total_s: float) -> np.ndarray:
    """Circularly shift the stimulus in chronological order, as a true permutation.

    Used as a null model: it preserves the full autocorrelation structure of both the
    features and the targets while destroying their correspondence.

    The shift is applied by rolling words in chronological *rank*, not by shifting onset
    times and re-matching. Time-shift-and-match looks more natural but is not a bijection
    on irregularly spaced events: several words map onto the same nearest neighbour while
    others are never selected, so the surrogate feature matrix ends up with duplicated
    rows, missing rows, and therefore different column marginals from the real design.
    That would bias the null. Rolling by rank keeps every word exactly once.

    Parameters
    ----------
    onsets_s : (n_words,) float array
        Word onsets in seconds. Need not be sorted.
    shift_s : float
        Requested shift in seconds, converted to a whole number of words at the corpus's
        mean word rate.
    total_s : float
        Length of the recording in seconds.

    Returns
    -------
    (n_words,) int array
        A permutation. Applying it to the feature matrix realigns features to the wrong
        words while leaving every feature's marginal distribution untouched.
    """
    onsets_s = np.asarray(onsets_s, dtype=float)
    n = onsets_s.size
    if n == 0:
        raise ValueError("onsets_s is empty")
    if total_s <= 0:
        raise ValueError("total_s must be positive")

    order = np.argsort(onsets_s, kind="stable")
    k = int(round(shift_s / (total_s / n))) % n
    idx = np.empty(n, dtype=np.int64)
    idx[order] = np.roll(order, -k)
    return idx


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomised surrogate with the power spectrum preserved.

    Parameters
    ----------
    x : (n_channels, n_times) float array
        Continuous signal. Randomisation is applied independently per channel.
    rng : numpy Generator
        Seeded random source.

    Returns
    -------
    (n_channels, n_times) float array
        Surrogate with the same per-channel amplitude spectrum and mean, and random
        phases. Real-valued by construction (DC and, for even length, Nyquist are
        left untouched).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    amp = np.abs(spec)

    n_free = spec.shape[-1] - 1                       # exclude DC
    if n % 2 == 0:
        n_free -= 1                                   # exclude Nyquist too
    phases = rng.uniform(0, 2 * np.pi, size=(x.shape[0], n_free))

    new = np.empty_like(spec)
    new[..., 0] = spec[..., 0]                        # DC preserved
    new[..., 1:1 + n_free] = amp[..., 1:1 + n_free] * np.exp(1j * phases)
    if n % 2 == 0:
        new[..., -1] = spec[..., -1]                  # Nyquist preserved
    return np.fft.irfft(new, n=n, axis=-1)
