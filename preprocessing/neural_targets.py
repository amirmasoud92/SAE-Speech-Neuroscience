"""Neural targets and word/sample alignment.

Turns the continuous 512-Hz high-gamma envelope and the word onset times into a
``(n_words, n_channels, n_lags)`` target tensor.

Conventions
-----------
* A lag ``l`` is the **left edge** of an averaging window: the target is the mean signal
  over ``[onset + l, onset + l + window)``, log-transformed.
* Sample indices are computed with ``round``, not ``int``.
* A word is analysable only if every lag's window lies fully inside the recording.

Electrodes enter if they are marked good, are of type ECoG, are present in the
preprocessed derivative and carry an MNI coordinate (``preprocessing/electrode_table.py``);
the names listed in ``config.yaml`` under ``electrodes.exclude_names`` are removed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from utils.config import Config, LagGrid

mne.set_log_level("ERROR")
log = logging.getLogger("data")


@dataclass(frozen=True)
class SubjectData:
    """High-gamma recording for one subject, restricted to analysis-eligible channels."""
    subject: str
    data: np.ndarray          # (n_channels, n_times) float32, Hilbert envelope in volts
    ch_names: list[str]
    sfreq: float
    coords_mm: np.ndarray     # (n_channels, 3) MNI152NLin2009aSym, millimetres

    def __post_init__(self) -> None:
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError("data rows and ch_names disagree")
        if self.coords_mm.shape != (len(self.ch_names), 3):
            raise ValueError("coords_mm shape disagrees with ch_names")


def load_electrode_table(cfg: Config) -> pd.DataFrame:
    """Analysis-eligible electrodes, with the config's exclusions applied.

    Requires ``preprocessing/electrode_table.py`` to have been run first.
    """
    f = cfg.paths.results_root / "electrodes" / "electrodes_master.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found - run preprocessing.electrode_table first")
    tab = pd.read_csv(f)

    elec = cfg.electrodes
    keep = tab["analysis_eligible"].to_numpy(bool).copy()
    if elec.get("require_status_good", True):
        keep &= (tab["status"] == "good").to_numpy()
    if elec.get("require_in_derivative", True):
        keep &= tab["in_derivative"].to_numpy(bool)
    if elec.get("require_localised", True):
        keep &= tab["in_electrodes_tsv"].to_numpy(bool)

    n_before = int(keep.sum())
    for sub, names in (elec.get("exclude_names") or {}).items():
        drop = (tab["subject"] == sub) & tab["name"].isin(list(names))
        if int(drop.sum()) != len(names):
            raise ValueError(
                f"exclude_names[{sub}] lists {len(names)} electrodes but "
                f"{int(drop.sum())} matched - a name is wrong")
        keep &= ~drop.to_numpy()
    log.info("electrodes: %d eligible, %d after named exclusions", n_before, int(keep.sum()))

    out = tab[keep].reset_index(drop=True)
    missing = set(cfg.subjects) - set(out["subject"])
    if missing:
        raise ValueError(f"no eligible electrodes for {sorted(missing)}")
    return out


def load_subject(cfg: Config, subject: str, table: pd.DataFrame) -> SubjectData:
    """Load one subject's high-gamma derivative, unit-corrected and channel-selected.

    The unit fix matters: ``sub-01``'s EDF leaves the physical-dimension field blank, so
    MNE applies no microvolt-to-volt conversion and its values are ~1.0008e6 times larger
    than every other subject's. Fold-wise z-scoring would hide this,
    but the divisor is applied anyway so absolute-unit reporting is correct.
    """
    want = table.loc[table["subject"] == subject, "name"].astype(str).tolist()
    raw = mne.io.read_raw_fif(cfg.paths.subject_highgamma(subject), preload=True,
                              verbose=False)

    missing = [c for c in want if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{subject}: electrodes in the table are absent from the "
                         f"derivative: {missing}")
    raw.pick(want)
    if raw.ch_names != want:
        raise AssertionError(f"{subject}: channel order changed during pick")

    data = raw.get_data().astype(np.float32)
    scale = float((cfg.neural.get("unit_fix") or {}).get(subject, 1.0))
    if scale != 1.0:
        log.warning("%s: applying unit correction x%.3g", subject, scale)
        data = data * np.float32(scale)

    if not np.isfinite(data).all():
        raise ValueError(f"{subject}: non-finite samples in the derivative")
    if (data <= 0).any():
        raise ValueError(f"{subject}: non-positive envelope samples - log would fail")

    sf = float(raw.info["sfreq"])
    if abs(sf - cfg.sfreq) > 1e-6:
        raise ValueError(f"{subject}: sfreq {sf} != configured {cfg.sfreq}")

    coords = np.array([c["loc"][:3] for c in raw.info["chs"]], dtype=float) * 1000.0
    log.info("%s: %d channels x %d samples (%.1f s)", subject, data.shape[0],
             data.shape[1], data.shape[1] / sf)
    return SubjectData(subject, data, list(raw.ch_names), sf, coords)


def onset_samples(onsets_s: np.ndarray, sfreq: float) -> np.ndarray:
    """Word onsets in seconds -> sample indices. Rounds, never truncates."""
    return np.rint(np.asarray(onsets_s, dtype=float) * sfreq).astype(np.int64)


def analysable_mask(onsets_s: np.ndarray, sfreq: float, n_times: int,
                    lags: LagGrid) -> np.ndarray:
    """Words whose every lag window lies fully inside the recording.

    Returns
    -------
    (n_words,) bool array
    """
    start = onset_samples(onsets_s, sfreq)
    win = int(round(lags.window_s * sfreq))
    first = start + int(np.floor(lags.centres_s.min() * sfreq))
    last = start + int(np.ceil(lags.centres_s.max() * sfreq)) + win
    return (first >= 0) & (last <= n_times)


def prefix_sums(data: np.ndarray) -> np.ndarray:
    """(n_channels, n_times + 1) float64 prefix sums, with a leading zero column.

    Computed once per subject and reused across all lags; recomputing it per lag would
    dominate the runtime of :func:`build_targets`.
    """
    cs = np.empty((data.shape[0], data.shape[1] + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(data, axis=1, out=cs[:, 1:])
    return cs


def window_average(data: np.ndarray, starts: np.ndarray, n_win: int,
                   cs: np.ndarray | None = None) -> np.ndarray:
    """Mean of ``data`` over ``[s, s + n_win)`` for every start in ``starts``.

    Parameters
    ----------
    data : (n_channels, n_times) float array
    starts : (n_events,) int array of window start samples
    n_win : int, window length in samples
    cs : optional (n_channels, n_times + 1) prefix sums from :func:`prefix_sums`.
        Pass it to avoid recomputing the cumulative sum on every call.

    Returns
    -------
    (n_events, n_channels) float32

    Uses prefix sums, so cost is O(n_times + n_events * n_channels) rather than
    O(n_events * n_win * n_channels).
    """
    if n_win < 1:
        raise ValueError("n_win must be >= 1")
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("starts is empty")
    if starts.min() < 0 or starts.max() + n_win > data.shape[1]:
        raise IndexError("a window falls outside the recording")

    if cs is None:
        cs = prefix_sums(data)
    out = (cs[:, starts + n_win] - cs[:, starts]) / float(n_win)
    return np.ascontiguousarray(out.T.astype(np.float32))


def build_targets(cfg: Config, sd: SubjectData, onsets_s: np.ndarray,
                  lags: LagGrid | None = None) -> np.ndarray:
    """Neural target tensor for one subject.

    Parameters
    ----------
    onsets_s : (n_words,) word onset times in seconds, already restricted to analysable
        words (see :func:`analysable_mask`).

    Returns
    -------
    (n_words, n_channels, n_lags) float32
        Window-averaged high-gamma envelope, log-transformed if configured.
        **Not standardised** - z-scoring happens inside training folds only.
    """
    lags = lags or cfg.lags
    starts0 = onset_samples(onsets_s, sd.sfreq)
    n_win = int(round(lags.window_s * sd.sfreq))
    centres = np.rint(lags.centres_s * sd.sfreq).astype(np.int64)

    cs = prefix_sums(sd.data)
    out = np.empty((len(onsets_s), len(sd.ch_names), lags.n_lags), dtype=np.float32)
    for i, off in enumerate(centres):
        out[:, :, i] = window_average(sd.data, starts0 + off, n_win, cs=cs)

    if cfg.neural["transform"] == "log":
        if (out <= 0).any():
            raise ValueError(f"{sd.subject}: non-positive window average before log")
        np.log(out, out=out)
    if not np.isfinite(out).all():
        raise ValueError(f"{sd.subject}: non-finite targets")
    return out


def load_words(cfg: Config) -> pd.DataFrame:
    """The master transcript, with the row-order caveat made explicit.

    Returns a frame with columns ``word``, ``start``, ``end``, and ``word_idx`` (the
    original row index, which is the key every stimulus feature file joins on).

    Row order is **speaker-turn order, not chronological**: 18 rows run backwards in time.
    Rows are returned in their original order so that ``word_idx`` stays aligned with the
    feature files; anything time-based must sort by ``start`` itself.
    """
    t = pd.read_csv(cfg.paths.transcript)
    if list(t.columns) != ["word", "start", "end"]:
        raise ValueError(f"unexpected transcript columns: {list(t.columns)}")
    t = t.reset_index().rename(columns={"index": "word_idx"})
    n_back = int((np.diff(t["start"].to_numpy()) < 0).sum())
    if n_back:
        log.info("transcript: %d rows run backwards in time (expected: 18) - all "
                 "time-based operations sort by `start`", n_back)
    return t
