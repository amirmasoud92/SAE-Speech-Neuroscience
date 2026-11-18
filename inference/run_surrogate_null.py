"""Surrogate null distributions for the unique-variance statistics.

Every hyperparameter is re-searched on every surrogate; only the band kernels are reused.
Surrogates are written incrementally, so an interrupted run resumes from what is on disk.

Example:
    python -m inference.run_surrogate_null --subjects sub-08 --band sae --n-perm 200
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from preprocessing import neural_targets as D
from utils.config import Config, load_config
from utils.run_logging import log_data_versions, setup_logging
from inference.surrogate_null import delta_r2_one_pass, phase_randomised_subject, precompute_kernels
from encoding.fit_encoding_model import analysed_words, stage_design

log = logging.getLogger("surrogate_null")


def null_path(cfg: Config, sub: str, band: str) -> Path:
    return cfg.cache(cfg.digest, f"null_{band}_{sub}.npy")


def load_existing(path: Path, n_ch: int, n_lags: int) -> np.ndarray:
    """Whatever surrogates are already on disk, as (k, n_ch, n_lags)."""
    if not path.exists():
        return np.empty((0, n_ch, n_lags), dtype=np.float32)
    arr = np.load(path)
    if arr.shape[1:] != (n_ch, n_lags):
        raise ValueError(f"{path.name} has shape {arr.shape}, expected (*, {n_ch}, "
                         f"{n_lags}) - delete it if the config changed")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--band", type=str, default=None,
                    help="band whose unique variance gets a null; default: config")
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="time a single surrogate and stop")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subs = args.subjects or cfg.subjects
    bands = [args.band] if args.band else list(cfg.nulls["phase_randomised_neural"]["bands"])
    n_perm = args.n_perm or int(cfg.nulls["phase_randomised_neural"]["n_permutations"])

    log_ = setup_logging(cfg.paths.results_root / "logs", "nulls", cfg.digest)
    log_.info("full config digest (incl. presentation): %s", cfg.full_digest)
    log_data_versions(log_, cfg.paths.bids_root)
    log_.info("bands=%s subjects=%s n_perm=%d", bands, ",".join(subs), n_perm)
    log_.info("hyperparameters are re-searched per surrogate; kernels are cached "
              "(exact: they do not depend on Y)")

    words = analysed_words(cfg)
    table = D.load_electrode_table(cfg)
    des = stage_design(cfg, words)
    onsets = words["start"].to_numpy()

    # Surrogates are the OUTER loop and subjects the inner one, so at any moment every
    # subject has the same number of completed surrogates. Over a run this long that
    # matters: an interruption at the halfway point leaves a usable lower-resolution null
    # for the whole study, rather than complete nulls for four subjects and none for the
    # other five. Kernels are recomputed per subject per surrogate, which costs ~11 s
    # against a ~3.7 h surrogate - under 1% - and avoids holding 9 x 3.7 GB in memory.
    target = 1 if args.benchmark else n_perm

    def completed(sub: str, n_ch: int) -> int:
        return min(len(load_existing(null_path(cfg, sub, b), n_ch, cfg.lags.n_lags))
                   for b in bands)

    subject_data = {}
    for sub in subs:
        subject_data[sub] = D.load_subject(cfg, sub, table)
    have = {s: completed(s, len(subject_data[s].ch_names)) for s in subs}
    log_.info("surrogates already on disk: %s", have)

    start_i = min(have.values())
    for i in range(start_i, target):
        t_surro = time.perf_counter()
        for sub in subs:
            if have[sub] > i:
                continue                      # this subject already has surrogate i
            sd = subject_data[sub]
            n_ch, n_lags = len(sd.ch_names), cfg.lags.n_lags
            kernels = precompute_kernels(cfg, des, onsets)

            t1 = time.perf_counter()
            rng = np.random.default_rng([cfg.seed, i])
            surro = phase_randomised_subject(sd, rng)
            Ys = D.build_targets(cfg, surro, onsets)
            d = delta_r2_one_pass(cfg, kernels, Ys.reshape(len(onsets), -1),
                                  bands, cfg.seed + 1000 * (i + 1))
            dt = time.perf_counter() - t1

            for b in bands:
                prev = list(load_existing(null_path(cfg, sub, b), n_ch, n_lags))[:i]
                np.save(null_path(cfg, sub, b),
                        np.stack(prev + [d[b].reshape(n_ch, n_lags).astype(np.float32)]))
            have[sub] = i + 1
            log_.info("surrogate %d/%d %s: %.1f s | %s", i + 1, target, sub, dt,
                      " ".join(f"{b}: med {np.median(d[b]):+.6f} "
                               f"p95 {np.percentile(d[b], 95):.6f}" for b in bands))
            del kernels
            if args.benchmark:
                est = dt * n_perm / 3600.0
                print(f"\n{sub}: {dt:.1f} s per surrogate for bands {bands} "
                      f"-> {n_perm} surrogates = {est:.1f} h for this subject")
                return
        log_.info("=== surrogate %d/%d complete for all %d subjects in %.1f min ===",
                  i + 1, target, len(subs), (time.perf_counter() - t_surro) / 60.0)
    log_.info("null run finished")


if __name__ == "__main__":
    main()
