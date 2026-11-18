"""The banded-ridge encoding model of the primary cohort.

Stages
------
``design``   build and cache the stimulus design matrix (participant-independent)
``targets``  build and cache the cleaned neural target tensor for each participant
``encode``   fit the banded-ridge models and the variance partition per participant

Every stage caches under ``<results>/cache/<config digest>/``; the digest is a hash of the
parameters that determine what is computed, so a changed configuration never reuses a stale
cache. Writes ``encoding/<participant>_variance_partition.npz`` (held-out R2 of the full
model and of each reduced model, unique variance of each band, per electrode and lag).

Example
-------
    python -m encoding.fit_encoding_model --config config/config.yaml --stage all
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import neural_targets as D
from utils.config import Config, load_config
from encoding.banded_ridge import variance_partition
from feature_extraction.feature_bands import Design, build_design
from utils.run_logging import log_data_versions, setup_logging, stage

log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- words

def analysed_words(cfg: Config) -> pd.DataFrame:
    """The words that survive the lag grid at the recording boundaries.

    All subjects share the same 512 Hz sampling rate and 921,600-sample length, so this
    mask is subject-independent. That is asserted rather than assumed.
    """
    words = D.load_words(cfg)
    n_times = None
    for sub in cfg.subjects:
        import mne
        info = mne.io.read_raw_fif(cfg.paths.subject_highgamma(sub), preload=False,
                                   verbose=False)
        if n_times is None:
            n_times = info.n_times
        elif info.n_times != n_times:
            raise ValueError(f"{sub} has {info.n_times} samples, expected {n_times} - "
                             "the analysable-word mask would differ between subjects")
    mask = D.analysable_mask(words["start"].to_numpy(), cfg.sfreq, n_times, cfg.lags)
    out = words[mask].reset_index(drop=True)
    log.info("words: %d of %d analysable over lags %.2f..%.2f s (+%d ms window)",
             len(out), len(words), cfg.lags.tmin_s, cfg.lags.tmax_s, cfg.lags.window_ms)
    return out


# --------------------------------------------------------------------------- stages

def stage_design(cfg: Config, words: pd.DataFrame) -> Design:
    f = cfg.cache(cfg.digest, "design.npz")
    if f.exists():
        z = np.load(f, allow_pickle=True)
        slices = {k: slice(int(a), int(b)) for k, (a, b) in
                  json.loads(str(z["slices"])).items()}
        ids = {k: z[f"ids_{k}"] for k in slices}
        log.info("design: loaded cache %s", f.name)
        return Design(z["X"], slices, ids)

    des = build_design(cfg, words)
    np.savez_compressed(
        f, X=des.X,
        slices=json.dumps({k: [v.start, v.stop] for k, v in des.slices.items()}),
        **{f"ids_{k}": v for k, v in des.feature_ids.items()})
    log.info("design: cached to %s (%.1f MB)", f.name, f.stat().st_size / 1e6)
    return des


def stage_targets(cfg: Config, sub: str, words: pd.DataFrame,
                  table: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Return ``(Y, ch_names, V)``: ``Y`` (n_words, n_channels, n_lags) and, when target
    cleaning is configured, the matching bool validity mask ``V``; else None."""
    f = cfg.cache(cfg.digest, f"targets_{sub}.npz")
    if f.exists():
        z = np.load(f, allow_pickle=True)
        V = z["V"] if "V" in z.files else None
        log.info("targets %s: loaded cache %s %s%s", sub, f.name, z["Y"].shape,
                 "" if V is None else f", {100 * (~V).mean():.2f}% masked")
        return z["Y"], list(z["ch_names"]), V

    sd = D.load_subject(cfg, sub, table)
    onsets = words["start"].to_numpy()
    Y = D.build_targets(cfg, sd, onsets)
    V = None
    extra = {}
    clean = cfg.neural.get("clean")
    if clean:
        from preprocessing import artifact_cleaning as CT
        drift, V, rep = CT.clean_subject(
            sd.data, sd.sfreq, onsets, cfg.lags.centres_s, cfg.lags.window_s,
            int(cfg.cv["n_outer_folds"]), mask=bool(clean["mask"]),
            detrend=bool(clean["detrend"]))
        Y -= drift
        frac0 = float((~V).mean())
        V, n_off = CT.mask_residual_offsets(Y, V, onsets, int(cfg.cv["n_outer_folds"]),
                                            float(cfg.cv["embargo_s"]))
        extra = {"V": V, "masked_frac_grid": rep["masked_frac_grid"],
                 "common_mode_s": rep["common_mode_s"],
                 "masked_frac_procedure": frac0, "n_offset_sets_masked": n_off}
        log.info("targets %s: residual-offset guard masked %d block x electrode x lag "
                 "sets (+%.2f%% of cells)", sub, n_off, 100 * ((~V).mean() - frac0))
        log.info("targets %s: cleaned - %.2f%% of cells masked, %d s common-mode",
                 sub, 100 * rep["masked_frac_cells"], int(rep["common_mode_s"]))
    np.savez(f, Y=Y, ch_names=np.asarray(sd.ch_names, dtype=object),
             lags_s=cfg.lags.centres_s, **extra)
    log.info("targets %s: %s cached (%.0f MB)", sub, Y.shape, f.stat().st_size / 1e6)
    return Y, sd.ch_names, V


def stage_encode(cfg: Config, sub: str, des: Design, Y: np.ndarray,
                 ch_names: list[str], words: pd.DataFrame,
                 table: pd.DataFrame, V: np.ndarray | None = None,
                 rois: list[str] | None = None, bands: list[str] | None = None) -> None:
    """``rois`` restricts the fit to electrodes in those regions, and ``bands`` to the
    deltas requested - both for configurations whose results are read only in IFG."""
    if rois:
        meta0 = table[table["subject"] == sub].set_index("name").loc[ch_names]
        keep = np.flatnonzero(meta0["roi"].isin(rois).to_numpy())
        if keep.size == 0:
            log.info("%s: no electrodes in %s - skipped", sub, rois)
            return
        Y = Y[:, keep]
        V = None if V is None else V[:, keep]
        ch_names = [ch_names[i] for i in keep]
        log.info("%s: restricted to %d electrodes in %s", sub, keep.size, rois)
    n_words, n_ch, n_lags = Y.shape
    Yflat = Y.reshape(n_words, n_ch * n_lags)
    Vflat = None if V is None else V.reshape(n_words, n_ch * n_lags)

    full_fit: dict = {}
    vp = variance_partition(cfg, des, Yflat, words["start"].to_numpy(), bands=bands,
                            keep_full_fit=full_fit, valid=Vflat)

    pred_path = cfg.cache(cfg.digest, f"fullfit_{sub}.npz")
    np.savez(pred_path, ch_names=np.asarray(ch_names, dtype=object),
             lags_s=cfg.lags.centres_s, n_channels=n_ch, n_lags=n_lags, **full_fit)
    log.info("%s: cached full-model held-out predictions (%.0f MB)", sub,
             pred_path.stat().st_size / 1e6)

    out = cfg.results("encoding", f"{sub}_variance_partition.npz")
    meta = table[table["subject"] == sub].set_index("name").loc[ch_names]
    np.savez_compressed(
        out,
        ch_names=np.asarray(ch_names, dtype=object),
        lags_s=cfg.lags.centres_s,
        roi=meta["roi"].to_numpy().astype(object),
        destrieux=meta["destrieux"].to_numpy().astype(object),
        mni=meta[["mni_x", "mni_y", "mni_z"]].to_numpy(float),
        parcel_dist_mm=meta["destrieux_dist_mm"].to_numpy(float),
        **{k: v.reshape(n_ch, n_lags) for k, v in vp.items()})
    log.info("%s: wrote %s", sub, out.name)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--stage", choices=["design", "targets", "encode", "all"],
                    default="all")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--rois", nargs="*", default=None,
                    help="fit only electrodes in these ROIs (variant configurations)")
    ap.add_argument("--bands", nargs="*", default=None,
                    help="compute unique variance only for these bands")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subs = args.subjects or cfg.subjects
    unknown = set(subs) - set(cfg.subjects)
    if unknown:
        raise SystemExit(f"unknown subjects: {sorted(unknown)}")

    log_ = setup_logging(cfg.paths.results_root / "logs", "pipeline", cfg.digest)
    log_.info("full config digest (incl. presentation): %s", cfg.full_digest)
    log_data_versions(log_, cfg.paths.bids_root)
    log_.info("stage=%s subjects=%s", args.stage, ",".join(subs))
    np.random.seed(cfg.seed)

    with stage(log_, "load words + electrode table"):
        words = analysed_words(cfg)
        table = D.load_electrode_table(cfg)

    des = None
    if args.stage in ("design", "encode", "all"):
        with stage(log_, "build design matrix"):
            des = stage_design(cfg, words)

    for sub in subs:
        if args.stage in ("targets", "encode", "all"):
            with stage(log_, f"targets {sub}"):
                Y, ch, V = stage_targets(cfg, sub, words, table)
        if args.stage in ("encode", "all"):
            with stage(log_, f"encode {sub}"):
                stage_encode(cfg, sub, des, Y, ch, words, table, V=V,
                             rois=args.rois, bands=args.bands)

    log_.info("pipeline finished")


if __name__ == "__main__":
    main()
