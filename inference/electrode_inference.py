"""Electrode-level inference from observed unique-variance maps and surrogate nulls.

Max-statistic permutation test: each surrogate contributes one maximum TFCE value over every
electrode and lag, so the resulting p-values control the family-wise error over the whole map.
Secondary: per-electrode Benjamini-Hochberg FDR, with a generalised-Pareto tail fit beyond the
permutation floor. The number of surrogates used is recorded in every output.

``load_observed`` reads the observed maps (one row per analysed electrode) and is used by the
group-level analyses.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import Config, load_config
from utils.run_logging import setup_logging, stage
from inference.statistics import (cluster_bootstrap, fdr, gpd_tail_pvalue, max_statistic_pvalues,
                                  mixed_model, peak_lag, tfce_1d, _shared_hmax)

log = logging.getLogger("inference")


def load_observed(cfg: Config, band: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Observed delta-R2 for every electrode, plus its metadata.

    Returns
    -------
    obs : (n_electrodes, n_lags) float
    meta : DataFrame with subject, electrode, roi, mni coordinates
    """
    maps, rows = [], []
    for sub in cfg.subjects:
        f = cfg.paths.results_root / "encoding" / f"{sub}_variance_partition.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run encoding.fit_encoding_model --stage encode")
        z = np.load(f, allow_pickle=True)
        maps.append(z[f"delta_{band}"])
        for i, ch in enumerate(z["ch_names"]):
            rows.append(dict(subject=sub, electrode=str(ch), roi=str(z["roi"][i]),
                             destrieux=str(z["destrieux"][i]),
                             mni_x=z["mni"][i, 0], mni_y=z["mni"][i, 1],
                             mni_z=z["mni"][i, 2]))
    return np.concatenate(maps, axis=0), pd.DataFrame(rows)


def load_null(cfg: Config, band: str) -> tuple[np.ndarray, int]:
    """Surrogate nulls concatenated over subjects in the same electrode order.

    Only the surrogates present for *every* subject are used, so the electrode axis is
    always complete. Returns ``(n_perm, n_electrodes, n_lags)`` and the count used.
    """
    per_subject = []
    for sub in cfg.subjects:
        f = cfg.cache(cfg.digest, f"null_{band}_{sub}.npy")
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run inference.run_surrogate_null")
        per_subject.append(np.load(f))
    n_perm = min(a.shape[0] for a in per_subject)
    if n_perm == 0:
        raise ValueError(f"no surrogates on disk for band '{band}'")
    return np.concatenate([a[:n_perm] for a in per_subject], axis=1), n_perm


# A surrogate whose unique variance exceeds the total variance of the signal is a diverged
# fit, not a draw from the null: delta R2 > 1 requires the reduced model to be worse than
# predicting the mean by more than the entire variance, and through the shared TFCE ceiling
# a single such draw can drive most electrodes to zero. The bound is 1.0 because that is
# arithmetically impossible for a converged fit; a lower bound would make the exclusion
# depend on the observed values being tested.
DEGENERATE_BOUND = 1.0


def drop_degenerate(null: np.ndarray, bound: float = DEGENERATE_BOUND):
    """Remove diverged surrogate fits. Returns (kept, n_dropped, dropped_maxima)."""
    per_draw = np.nanmax(null.reshape(null.shape[0], -1), axis=1)
    keep = per_draw <= bound
    if not keep.any():
        raise ValueError(f"every surrogate exceeds delta R2 {bound} - the null is unusable")
    frac = float((~keep).mean())
    if frac > 0.05:
        raise ValueError(
            f"{100 * frac:.1f}% of surrogates exceed delta R2 {bound}. This requires the "
            "null be regenerated rather than filtered above 5%: the fitting procedure "
            "itself is unreliable at that rate.")
    return null[keep], int((~keep).sum()), per_draw[~keep]


def infer(cfg: Config, band: str, min_perm: int = 1) -> pd.DataFrame:
    """Per-electrode inference for one band."""
    obs, meta = load_observed(cfg, band)
    null, n_perm = load_null(cfg, band)
    # diverged draws are removed before the shared TFCE ceiling is computed
    null, n_dropped, dropped = drop_degenerate(null)
    n_perm = null.shape[0]
    if n_dropped:
        log.warning("dropped %d of %d surrogates as diverged fits (max delta R2 "
                    "%s); %d retained", n_dropped, n_dropped + n_perm,
                    ", ".join(f"{d:.3f}" for d in dropped), n_perm)
    else:
        log.info("no diverged surrogates in band %s", band)
    if null.shape[1] != obs.shape[0]:
        raise ValueError(f"null has {null.shape[1]} electrodes, observed has "
                         f"{obs.shape[0]} - electrode order has diverged")
    if n_perm < min_perm:
        raise SystemExit(f"only {n_perm} surrogates available, --min-perm is {min_perm}")

    floor = 1.0 / (1.0 + n_perm)
    log.info("band=%s: %d electrodes x %d lags, %d surrogates (empirical p floor %.5f)",
             band, obs.shape[0], obs.shape[1], n_perm, floor)
    if n_perm < 200:
        log.warning("PARTIAL NULL: %d of 200 surrogates. Results are provisional and "
                    "the p-value floor is %.4f.", n_perm, floor)

    p_fwer, null_max = max_statistic_pvalues(obs, null)

    # TFCE on a grid shared with the null, so per-electrode statistics are comparable.
    hmax = _shared_hmax(obs, null)
    obs_tfce = tfce_1d(obs, hmax=hmax)
    null_tfce = tfce_1d(null.reshape(-1, null.shape[-1]), hmax=hmax).reshape(null.shape)
    obs_stat = obs_tfce.max(axis=-1)
    null_stat = null_tfce.max(axis=-1)                    # (n_perm, n_electrodes)

    p_emp = (1.0 + (null_stat >= obs_stat[None, :]).sum(0)) / (1.0 + n_perm)
    gpd = [gpd_tail_pvalue(obs_stat[i], null_stat[:, i],
                           n_exceedances=int(cfg.stats["gpd"]["n_exceedances"]),
                           min_exceedances=int(cfg.stats["gpd"]["min_exceedances"]),
                           gof_alpha=float(cfg.stats["gpd"]["goodness_of_fit_alpha"]))
           for i in range(obs.shape[0])]
    p_tail = np.array([g[0] for g in gpd])
    used_gpd = np.array([g[1] for g in gpd])
    reject, q = fdr(p_tail, alpha=float(cfg.stats["alpha"]))

    lags = cfg.lags.centres_s
    meta = meta.assign(
        band=band, n_perm=n_perm, p_floor=floor,
        a27_dropped=n_dropped, a27_bound=DEGENERATE_BOUND, a27_hmax=hmax,
        obs_peak=obs.max(axis=1), obs_peak_lag_s=peak_lag(obs, lags),
        obs_tfce=obs_stat,
        null_median=np.median(null.max(axis=-1), axis=0),
        p_fwer=p_fwer, sig_fwer=p_fwer <= float(cfg.stats["alpha"]),
        p_empirical=p_emp, p_fdr_input=p_tail, p_from_gpd_tail=used_gpd,
        q_fdr=q, sig_fdr=reject)

    log.info("band=%s: %d/%d significant by max-statistic FWER, %d by FDR "
             "(%d p-values from the Pareto tail)", band,
             int(meta.sig_fwer.sum()), len(meta), int(meta.sig_fdr.sum()),
             int(used_gpd.sum()))
    return meta


def group_summary(cfg: Config, meta: pd.DataFrame, band: str) -> dict:
    """ROI effects and subject-level uncertainty on the observed peak."""
    out: dict = {}
    try:
        out["mixed_model_roi"] = mixed_model(meta, outcome="obs_peak", fixed="roi")
    except Exception as exc:  # noqa: BLE001
        out["mixed_model_roi"] = {"error": str(exc)}

    lang = set(cfg.electrodes["language_rois"])
    ctrl = set(cfg.electrodes["control_rois"])
    sub = meta[meta.roi.isin(lang | ctrl)].copy()
    sub["network"] = np.where(sub.roi.isin(lang), "language", "control")
    try:
        out["mixed_model_network"] = mixed_model(sub, outcome="obs_peak",
                                                 fixed="network")
    except Exception as exc:  # noqa: BLE001
        out["mixed_model_network"] = {"error": str(exc)}

    out["roi_bootstrap"] = {}
    for roi, g in meta.groupby("roi"):
        if g.subject.nunique() >= 3:
            point, lo, hi = cluster_bootstrap(
                g.obs_peak.to_numpy(), g.subject.to_numpy(),
                n_resamples=int(cfg.stats["bootstrap"]["n_resamples"]),
                ci=float(cfg.stats["bootstrap"]["ci"]), seed=cfg.seed)
            out["roi_bootstrap"][roi] = dict(n=len(g), n_subjects=int(g.subject.nunique()),
                                             mean=point, ci_low=lo, ci_high=hi)

    out["significant_by_roi"] = (meta.groupby("roi")
                                 .agg(n=("electrode", "size"),
                                      n_sig_fwer=("sig_fwer", "sum"),
                                      n_sig_fdr=("sig_fdr", "sum"))
                                 .to_dict("index"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--band", nargs="*", default=None)
    ap.add_argument("--min-perm", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bands = args.band or list(cfg.nulls["phase_randomised_neural"]["bands"])
    log_ = setup_logging(cfg.paths.results_root / "logs", "inference", cfg.digest)
    log_.info("full config digest (incl. presentation): %s", cfg.full_digest)

    for band in bands:
        with stage(log_, f"inference {band}"):
            meta = infer(cfg, band, args.min_perm)
            meta.to_csv(cfg.results("electrode_inference", f"{band}_electrodes.csv"),
                        index=False)
            summary = group_summary(cfg, meta, band)
            summary["n_perm"] = int(meta.n_perm.iloc[0])
            summary["provisional"] = bool(meta.n_perm.iloc[0] < 200)
            (cfg.results("electrode_inference", f"{band}_group.json")
             .write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8"))

            print(f"\n=== {band}: {int(meta.n_perm.iloc[0])} surrogates "
                  f"{'(PROVISIONAL)' if summary['provisional'] else ''} ===")
            print(meta.groupby("roi").agg(
                n=("electrode", "size"),
                median_peak=("obs_peak", "median"),
                n_sig_fwer=("sig_fwer", "sum"),
                n_sig_fdr=("sig_fdr", "sum")).sort_values(
                    "median_peak", ascending=False).to_string(
                        float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
