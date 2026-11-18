"""Typed configuration loaded from ``config/config.yaml``.

Every analysis parameter lives in the YAML file and reaches the code through this module.
Relative paths in the file are resolved against the repository root; the environment
variable ``PODCAST_BIDS_ROOT`` overrides the location of the dataset.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Paths:
    bids_root: Path
    results_root: Path
    cache_root: Path

    def subject_highgamma(self, sub: str) -> Path:
        return (self.bids_root / "derivatives" / "ecogprep" / sub / "ieeg"
                / f"{sub}_task-podcast_desc-highgamma_ieeg.fif")

    def subject_broadband(self, sub: str) -> Path:
        return (self.bids_root / "derivatives" / "ecogprep" / sub / "ieeg"
                / f"{sub}_task-podcast_ieeg.fif")

    @property
    def transcript(self) -> Path:
        return self.bids_root / "stimuli" / "podcast_transcript.csv"

    @property
    def spectrogram(self) -> Path:
        return self.bids_root / "stimuli" / "spectral" / "spectrogram.hdf5"


@dataclass(frozen=True)
class LagGrid:
    """The lag axis, in seconds relative to word onset.

    ``centres`` are the *left edges* of each averaging window: the target at lag l is the
    mean neural signal over [onset + l, onset + l + window].
    """
    tmin_s: float
    tmax_s: float
    step_ms: float
    window_ms: float

    @property
    def centres_s(self) -> np.ndarray:
        n = int(round((self.tmax_s - self.tmin_s) / (self.step_ms / 1000.0))) + 1
        return self.tmin_s + np.arange(n) * (self.step_ms / 1000.0)

    @property
    def n_lags(self) -> int:
        return len(self.centres_s)

    @property
    def window_s(self) -> float:
        return self.window_ms / 1000.0


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    paths: Paths
    seed: int
    subjects: list[str]
    lags: LagGrid
    ablation_lags: LagGrid

    # -- convenience accessors into the raw dict -------------------------------------
    @property
    def electrodes(self) -> dict[str, Any]:
        return self.raw["electrodes"]

    @property
    def neural(self) -> dict[str, Any]:
        return self.raw["neural"]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]

    @property
    def sae(self) -> dict[str, Any]:
        return self.raw["sae"]

    @property
    def bands(self) -> dict[str, Any]:
        return self.raw["bands"]

    @property
    def cv(self) -> dict[str, Any]:
        return self.raw["cv"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def mono(self) -> dict[str, Any]:
        return self.raw["monosemanticity"]

    @property
    def nulls(self) -> dict[str, Any]:
        return self.raw["nulls"]

    @property
    def stats(self) -> dict[str, Any]:
        return self.raw["stats"]

    @property
    def runtime(self) -> dict[str, Any]:
        """Execution settings - worker counts and the like.

        Absent from :attr:`COMPUTE_SECTIONS` by design, so nothing here can re-key the
        cache. Defaults to an empty dict so that a config written before this section
        existed still loads.
        """
        return self.raw.get("runtime", {})

    @property
    def n_jobs_subset_ridge(self) -> int:
        """Worker processes for the subset-ridge target loop. 1 selects the sequential
        path exactly, which is what the equivalence test pins."""
        return int(self.runtime.get("n_jobs_subset_ridge", 1))

    @property
    def sfreq(self) -> float:
        return float(self.neural["sfreq_hz"])

    @property
    def alphas(self) -> np.ndarray:
        lo, hi, n = self.model["alphas_log10"]
        return np.logspace(lo, hi, int(n))

    #: Sections that determine the *content* of a cached artefact. Anything outside this
    #: list changes how results are presented or tested after the fact, never what was
    #: computed, so it must not invalidate the cache.
    COMPUTE_SECTIONS: tuple[str, ...] = (
        "subjects", "electrodes", "neural", "lags", "llm", "sae", "bands",
        "comparison_models", "cv", "model", "monosemanticity", "nulls",
    )

    @property
    def digest(self) -> str:
        """Cache key: a hash of only the parameters that affect what gets computed.

        Deliberately **excludes** presentation and statistics settings, so that a cosmetic
        edit never invalidates a fit. The seed is included, since it does change the numbers.
        The key is coarse - it errs toward recomputation rather than toward silently
        reusing a stale fit.
        """
        subset = {k: self.raw[k] for k in self.COMPUTE_SECTIONS if k in self.raw}
        subset["seed"] = self.seed
        blob = yaml.safe_dump(subset, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def full_digest(self) -> str:
        """Hash of the entire config, including presentation. Logged for provenance so
        the exact file that produced a run is still identifiable."""
        blob = yaml.safe_dump(self.raw, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def cache(self, *parts: str) -> Path:
        p = self.paths.cache_root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def results(self, *parts: str) -> Path:
        p = self.paths.results_root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path) -> Config:
    """Read a configuration file, validate it, and return a typed :class:`Config`."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    proj = raw["project"]
    repo = Path(__file__).resolve().parents[1]

    def resolve(value, env=None):
        p = Path(os.environ.get(env, value) if env else value)
        return p if p.is_absolute() else repo / p

    results_root = resolve(proj["results_root"])
    cache = Path(proj["cache_root"])
    paths = Paths(bids_root=resolve(proj["bids_root"], "PODCAST_BIDS_ROOT"),
                  results_root=results_root,
                  cache_root=cache if cache.is_absolute() else results_root / cache)

    lg = raw["lags"]
    lags = LagGrid(tmin_s=float(lg["tmin_s"]), tmax_s=float(lg["tmax_s"]),
                   step_ms=float(lg["step_ms"]), window_ms=float(raw["neural"]["window_ms"]))
    ablation = LagGrid(tmin_s=float(lg["tmin_s"]), tmax_s=float(lg["tmax_s"]),
                       step_ms=float(lg["ablation_step_ms"]),
                       window_ms=float(lg["ablation_window_ms"]))

    cfg = Config(raw=raw, paths=paths, seed=int(proj["seed"]),
                 subjects=list(raw["subjects"]), lags=lags, ablation_lags=ablation)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail loudly at load time rather than deep inside a fit."""
    if not cfg.paths.bids_root.exists():
        raise FileNotFoundError(f"bids_root does not exist: {cfg.paths.bids_root}")
    if len(cfg.subjects) != len(set(cfg.subjects)):
        raise ValueError("duplicate entries in subjects")
    for sub in cfg.subjects:
        f = cfg.paths.subject_highgamma(sub)
        if not f.exists():
            raise FileNotFoundError(f"missing derivative for {sub}: {f}")

    if cfg.lags.tmax_s <= cfg.lags.tmin_s:
        raise ValueError("lags.tmax_s must exceed lags.tmin_s")
    if cfg.lags.window_ms <= 0 or cfg.lags.step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")

    embargo = float(cfg.cv["embargo_s"])
    half_epoch = max(abs(cfg.lags.tmin_s), abs(cfg.lags.tmax_s)) + cfg.lags.window_s
    if embargo < half_epoch:
        raise ValueError(
            f"cv.embargo_s ({embargo} s) is shorter than the epoch half-width "
            f"({half_epoch:.3f} s). Training and test epochs would share samples.")
    if cfg.cv["order_by"] != "word_onset_time":
        raise ValueError("cv.order_by must be 'word_onset_time' - the transcript is "
                         "stored in speaker-turn order, so row index is not time order")
    if cfg.cv.get("shuffle", False):
        raise ValueError("cv.shuffle must be false for a time-series design")

    if cfg.neural["transform"] not in {"log", "none"}:
        raise ValueError("neural.transform must be 'log' or 'none'")

    rois = set(cfg.electrodes["roi_map"])
    for key in ("language_rois", "control_rois"):
        unknown = set(cfg.electrodes[key]) - rois
        if unknown:
            raise ValueError(f"electrodes.{key} references undefined ROIs: {sorted(unknown)}")
    if set(cfg.electrodes["language_rois"]) & set(cfg.electrodes["control_rois"]):
        raise ValueError("language_rois and control_rois overlap")
