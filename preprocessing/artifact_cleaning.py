"""Artifact masks and slow-drift removal for the word x lag high-gamma targets.

The released high-gamma derivative keeps prolonged artifact episodes and slow baseline
drift. Under contiguous cross-validation a test block can then sit at a baseline offset
relative to the training data, so both are removed before any model is fitted.
Detection and drift estimation run on a 10-Hz version of each electrode's log envelope, and
the result is transferred to every word x lag cell by the time its window occupies.

1. **10-Hz grid.** Mean of the 512-Hz envelope in 100-ms bins, then log.
2. **CV blocks in time.** The ten contiguous word blocks of ``BlockedTimeSeriesSplit`` are
   converted to time spans, split at the midpoint between the last word of one block and
   the first of the next. Every estimate below is computed **within a block**.
3. **Detection trend.** Centred running median of 1-s means over 121 s, within block.
4. **Per-electrode artifact seconds** on the detrended 1-s means: window SD above 4x the
   electrode's median window SD, or window mean beyond 5 robust SD (1.4826 x MAD).
5. **Common-mode artifact seconds**: across the participant's electrodes, median SD ratio
   above 2.5 or median |robust z| above 3; applied to all of that participant's electrodes.
6. **Mask** = union of 4 and 5, dilated by 1 s on each side.
7. **Drift**: centred 120-s running mean of valid 10-Hz samples, within block.
8. **Transfer to cells.** Cell (word w, lag l) covers ``[onset_w + lag_l, +200 ms)``. It is
   valid only if every 10-Hz bin it overlaps is valid, and it is corrected by subtracting
   the drift at the window's centre.
9. **Residual-offset guard.** A block x electrode x lag set whose held-out mean still
   departs from the training mean by more than 0.5 training standard deviations is masked.

The drift estimate never crosses a fold boundary. Artifact *detection* is normalised by a
recording-wide robust noise scale, because an artifact can last most of a block; this
dependency is neural-only and carries no stimulus information.

The target definition is ``log(mean of the 200-ms window)``; the drift estimate is in the
same log units.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

RATE = 10                     # Hz, detection/drift grid
DETECT_MEDIAN_S = 121         # running median for detection, seconds
SD_RATIO_ELECTRODE = 4.0      # per-electrode window-SD ratio threshold
Z_ELECTRODE = 5.0             # per-electrode robust-z threshold on window means
SD_RATIO_COMMON = 2.5         # common-mode window-SD ratio threshold
Z_COMMON = 3.0                # common-mode robust-z threshold
DILATE_S = 1                  # seconds of dilation either side
DETREND_HALF_S = 60           # half-width of the 120-s running mean
MIN_VALID_S = 5               # seconds of valid data needed for a local drift estimate


def block_time_edges(onsets_s: np.ndarray, n_splits: int, total_s: float) -> np.ndarray:
    """Time boundaries (s) of the contiguous CV blocks, len n_splits + 1.

    Mirrors ``cv.BlockedTimeSeriesSplit``: words sorted by onset, split into equal-count
    blocks; a boundary sits at the midpoint between the last onset of one block and the
    first onset of the next.
    """
    t = np.sort(np.asarray(onsets_s, dtype=float))
    n = t.size
    idx = np.linspace(0, n, n_splits + 1).astype(int)
    edges = np.empty(n_splits + 1)
    edges[0], edges[-1] = 0.0, float(total_s)
    for b in range(1, n_splits):
        edges[b] = 0.5 * (t[idx[b] - 1] + t[idx[b]])
    return edges


def to_grid(data: np.ndarray, sfreq: float, rate: int = RATE) -> np.ndarray:
    """(n_ch, n_times) envelope -> (n_bins, n_ch) log of 1/rate-s bin means.

    Bin k spans samples ``[round(k sfreq / rate), round((k + 1) sfreq / rate))``, so it is the
    interval [k / rate, (k + 1) / rate) s to within one sample whatever the sampling rate.

    A fixed bin of ``round(sfreq / rate)`` samples would be 51 samples (99.6 ms) at
    512 Hz, so a grid built that way runs 0.39% fast against the clock that every consumer
    assumes; the bin edges are therefore computed from the sample times.
    """
    n_bins = int(np.floor(data.shape[1] * rate / sfreq + 1e-9))
    edges = np.rint(np.arange(n_bins + 1) * sfreq / rate).astype(np.int64)
    cs = np.zeros((data.shape[0], data.shape[1] + 1))
    np.cumsum(data, axis=1, out=cs[:, 1:])
    x = (cs[:, edges[1:]] - cs[:, edges[:-1]]) / np.diff(edges)
    if (x <= 0).any():
        raise ValueError("non-positive bin mean before log")
    return np.log(x).T.astype(np.float64)


def clean_grid(Y: np.ndarray, block_bins: np.ndarray, rate: int = RATE,
               mask: bool = True, detrend: bool = True
               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Detect artifacts and estimate drift on one participant's 10-Hz log envelope.

    Parameters
    ----------
    Y : (n_bins, E) log envelope for all of one participant's electrodes.
    block_bins : (n_blocks + 1,) bin indices of the CV block boundaries.

    Returns
    -------
    trend : (n_bins, E) drift to subtract (zeros if ``detrend`` is False)
    valid : (n_bins, E) bool, False on artifact bins (all True if ``mask`` is False)
    report : per-electrode counts
    """
    n, E = Y.shape
    W = rate
    wins, wmean, wsd = [], [], []
    for b in range(len(block_bins) - 1):
        a, z = int(block_bins[b]), int(block_bins[b + 1])
        ws = [(s, min(s + W, z)) for s in range(a, z, W)]
        mu = np.stack([Y[s:e].mean(0) for s, e in ws])
        sd = np.stack([Y[s:e].std(0) for s, e in ws])
        med = median_filter(mu, size=(DETECT_MEDIAN_S, 1), mode="nearest")
        wins += ws
        wmean.append(mu - med)
        wsd.append(sd)
    wmean, wsd = np.concatenate(wmean), np.concatenate(wsd)
    rob = 1.4826 * np.median(np.abs(wmean - np.median(wmean, 0)), 0) + 1e-9
    zmean = wmean / rob
    rsd = wsd / (np.median(wsd, 0) + 1e-9)
    flag_e = (rsd > SD_RATIO_ELECTRODE) | (np.abs(zmean) > Z_ELECTRODE)
    cm = (np.median(rsd, 1) > SD_RATIO_COMMON) | (np.median(np.abs(zmean), 1) > Z_COMMON)
    flag = binary_dilation(flag_e | cm[:, None],
                           structure=np.ones((2 * DILATE_S + 1, 1), bool))

    valid = np.ones((n, E), bool)
    if mask:
        for k, (s, e) in enumerate(wins):
            valid[s:e] = ~flag[k]

    trend = np.zeros((n, E))
    if detrend:
        H = DETREND_HALF_S * rate
        for b in range(len(block_bins) - 1):
            a, z = int(block_bins[b]), int(block_bins[b + 1])
            y, v = Y[a:z], valid[a:z].astype(np.float64)
            cs = np.vstack([np.zeros((1, E)), np.cumsum(y * v, 0)])
            cn = np.vstack([np.zeros((1, E)), np.cumsum(v, 0)])
            i = np.arange(z - a)
            lo, hi = np.clip(i - H, 0, z - a), np.clip(i + H + 1, 0, z - a)
            num, den = cs[hi] - cs[lo], cn[hi] - cn[lo]
            tr = num / np.maximum(den, 1)
            tr[den < MIN_VALID_S * rate] = np.nan
            bm = (y * v).sum(0) / np.maximum(v.sum(0), 1)
            trend[a:z] = np.where(np.isfinite(tr), tr, bm[None])

    report = {"masked_bins": (~valid).sum(0), "common_mode_s": cm.sum(),
              "electrode_flag_s": flag_e.sum(0)}
    return trend, valid, report


def cells_from_grid(onsets_s: np.ndarray, lag_starts_s: np.ndarray, window_s: float,
                    trend: np.ndarray, valid: np.ndarray, rate: int = RATE
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Transfer grid drift and validity to every word x lag cell.

    Returns
    -------
    drift : (n_words, E, n_lags) float32 - drift at each window's centre
    ok    : (n_words, E, n_lags) bool - every overlapped bin valid
    """
    n_bins, E = trend.shape
    t0 = np.asarray(onsets_s)[:, None] + np.asarray(lag_starts_s)[None]    # (W, L)
    centre = (t0 + window_s / 2.0) * rate - 0.5          # bin k is centred at (k + .5)/rate
    k0 = np.clip(np.floor(centre).astype(int), 0, n_bins - 1)
    k1 = np.clip(k0 + 1, 0, n_bins - 1)
    frac = np.clip(centre - np.floor(centre), 0.0, 1.0)
    drift = (trend[k0] * (1 - frac)[..., None] + trend[k1] * frac[..., None])  # (W,L,E)

    first = np.clip(np.floor(t0 * rate).astype(int), 0, n_bins - 1)
    last = np.clip(np.ceil((t0 + window_s) * rate).astype(int) - 1, 0, n_bins - 1)
    # validity over [first, last] via prefix sums of invalid bins
    bad = np.vstack([np.zeros((1, E)), np.cumsum(~valid, 0)])               # (n_bins+1, E)
    n_bad = bad[last + 1] - bad[first]                                      # (W, L, E)
    ok = n_bad == 0
    return (np.ascontiguousarray(drift.transpose(0, 2, 1)).astype(np.float32),
            np.ascontiguousarray(ok.transpose(0, 2, 1)))


OFFSET_SD = 0.5               # the block-offset guard's threshold, fixed before cleaning ran


def mask_residual_offsets(Y: np.ndarray, V: np.ndarray, onsets_s: np.ndarray,
                          n_splits: int, embargo_s: float,
                          threshold: float = OFFSET_SD) -> tuple[np.ndarray, int]:
    """Mask electrode x lag cells whose test block still sits at a baseline offset.

    Artifact detection compares each second with a 121-s running median *within the
    block*, which follows any shift longer than about 60 s, so an artifact lasting minutes
    can leave a block at a baseline offset after cleaning. This guard (labelled post hoc in
    the paper) masks, for that block, any cell set whose train-standardised test-block mean
    exceeds 0.5 SD. It acts only where detection fails.

    Parameters
    ----------
    Y : (n_words, n_ch, n_lags) cleaned targets
    V : (n_words, n_ch, n_lags) bool validity mask, modified in place

    Returns
    -------
    V, and the number of (block, electrode, lag) sets masked.
    """
    from encoding.cross_validation import BlockedTimeSeriesSplit
    folds = list(BlockedTimeSeriesSplit(n_splits, embargo_s).split(onsets_s))
    n_masked = 0
    for li in range(Y.shape[2]):
        y = Y[:, :, li].astype(np.float64)
        v = V[:, :, li]
        for tr, te in folds:
            w = v[tr].astype(np.float64)
            n = np.maximum(w.sum(0), 1.0)
            mu = (y[tr] * w).sum(0) / n
            sd = np.sqrt((((y[tr] - mu) ** 2) * w).sum(0) / n) + 1e-12
            wt = v[te].astype(np.float64)
            off = (((y[te] - mu) / sd) * wt).sum(0) / np.maximum(wt.sum(0), 1.0)
            bad = np.abs(off) > threshold
            if bad.any():
                V[np.ix_(te, np.flatnonzero(bad), [li])] = False
                n_masked += int(bad.sum())
    return V, n_masked


def clean_subject(data: np.ndarray, sfreq: float, onsets_s: np.ndarray,
                  lag_starts_s: np.ndarray, window_s: float, n_splits: int,
                  mask: bool = True, detrend: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """End-to-end: continuous envelope -> per-cell drift and validity for one subject."""
    grid = to_grid(data, sfreq)
    total_s = data.shape[1] / sfreq
    edges_s = block_time_edges(onsets_s, n_splits, total_s)
    block_bins = np.clip(np.rint(edges_s * RATE).astype(int), 0, grid.shape[0])
    trend, valid, rep = clean_grid(grid, block_bins, mask=mask, detrend=detrend)
    drift, ok = cells_from_grid(onsets_s, lag_starts_s, window_s, trend, valid)
    rep["masked_frac_grid"] = float((~valid).mean())
    rep["masked_frac_cells"] = float((~ok).mean())
    return drift, ok, rep
