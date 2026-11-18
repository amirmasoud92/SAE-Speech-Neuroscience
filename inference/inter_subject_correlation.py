"""Inter-subject correlation of the word-locked targets.

All participants heard the same stimulus, so the stimulus-locked variance shared across brains
can be estimated by inter-subject correlation, computed on the word-locked targets the model
predicts (words x channels x lags).

* **Region level.** Average within region per participant, then correlate each participant
  against the mean of the others (leave-one-out) and pairwise.
* **Spatially matched.** For each electrode, the mean signal of other participants'
  electrodes within 15 mm in MNI space.

Inter-subject correlation captures only variance shared across brains and is not a
repeat-trial noise ceiling. Writes ``inter_subject_correlation/``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import Config, load_config
from utils.run_logging import setup_logging

log = logging.getLogger("a13")

MATCH_MM = 15.0
ROIS = ("IFG", "STG", "MTG", "TMP", "EAC", "MFG", "PRC")


def zscore(a, axis=0):
    a = a - a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True)
    return np.divide(a, s, out=np.zeros_like(a), where=s > 0)


def corr_cols(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between each column of A and the vector b."""
    Az, bz = zscore(A), zscore(b.reshape(-1, 1))
    return (Az * bz).mean(axis=0)


def collect(cfg: Config):
    """ROI-mean signals per subject, and per-electrode signals at each electrode's own
    peak SAE lag, from the cached targets."""
    roi_sig, elec = {}, []
    for sub in cfg.subjects:
        z = np.load(cfg.cache(cfg.digest, f"targets_{sub}.npz"), allow_pickle=True)
        Y = z["Y"]                                        # (words, ch, lags)
        if "V" in z.files:
            # masked cells enter at their own (electrode, lag) valid-cell mean, so
            # they add nothing to any correlation; an ROI mean then averages every electrode
            Vm = z["V"]
            n = np.maximum(Vm.sum(0, keepdims=True), 1)
            mu = np.where(Vm, Y, 0.0).sum(0, keepdims=True) / n
            Y = np.where(Vm, Y, mu).astype(np.float32)
            del Vm
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        if not np.array_equal(z["ch_names"].astype(str), v["ch_names"].astype(str)):
            raise SystemExit(f"{sub}: channel order differs between targets and partition")
        roi = v["roi"].astype(str)
        d = v["delta_sae"]
        li = np.nanargmax(np.where(np.isfinite(d), d, -np.inf), axis=1)

        roi_sig[sub] = {r: Y[:, roi == r, :].mean(axis=1)
                        for r in ROIS if (roi == r).sum() > 0}   # (words, lags)
        elec.append(pd.DataFrame({
            "subject": sub, "ch": v["ch_names"].astype(str), "roi": roi,
            "peak_lag_i": li, "peak_sae": np.nanmax(d, axis=1),
            "mni_x": v["mni"][:, 0], "mni_y": v["mni"][:, 1], "mni_z": v["mni"][:, 2],
            "sig": list(Y[np.arange(Y.shape[0])[:, None],
                          np.arange(Y.shape[1])[None, :], li[None, :]].T)}))
        log.info("%s: %d channels, ROI signals for %s", sub, Y.shape[1],
                 sorted(roi_sig[sub]))
        del Y, z
    return roi_sig, pd.concat(elec, ignore_index=True)


def roi_ceiling(cfg, roi_sig, log_):
    rows = []
    for r in ROIS:
        subs = [s for s in cfg.subjects if r in roi_sig[s]]
        if len(subs) < 3:
            continue
        n_lags = roi_sig[subs[0]][r].shape[1]
        for li in range(n_lags):
            M = np.column_stack([roi_sig[s][r][:, li] for s in subs])   # (words, subs)
            Mz = zscore(M)
            for k, s in enumerate(subs):
                others = np.delete(Mz, k, axis=1).mean(axis=1)
                loo = float(corr_cols(Mz[:, [k]], others)[0])
                pair = float(np.mean([corr_cols(Mz[:, [k]], Mz[:, j])[0]
                                      for j in range(len(subs)) if j != k]))
                rows.append({"roi": r, "lag_i": li, "subject": s,
                             "isc_loo": loo, "isc_pairwise": pair})
    d = pd.DataFrame(rows)
    log_.info("--- A13 ROI-level ISC, at each ROI's best lag ---")
    best = []
    for r, g in d.groupby("roi"):
        by_lag = g.groupby("lag_i").isc_loo.mean()
        li = int(by_lag.idxmax())
        gg = g[g.lag_i == li]
        best.append({"roi": r, "lag_i": li, "n_subjects": gg.subject.nunique(),
                     "isc_loo": gg.isc_loo.mean(), "isc_pairwise": gg.isc_pairwise.mean(),
                     "ceiling_r2_loo": gg.isc_loo.mean() ** 2,
                     "ceiling_r2_pairwise": gg.isc_pairwise.mean() ** 2})
        log_.info("  %-4s best lag idx %3d  ISC loo %+.3f (r2 %.4f)  pairwise %+.3f "
                  "(r2 %.4f)  n=%d", r, li, gg.isc_loo.mean(), gg.isc_loo.mean() ** 2,
                  gg.isc_pairwise.mean(), gg.isc_pairwise.mean() ** 2,
                  gg.subject.nunique())
    return d, pd.DataFrame(best)


def electrode_ceiling(cfg, elec, log_):
    xyz = elec[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    subj = elec.subject.to_numpy()
    sig = np.column_stack(elec.sig.to_list())            # (words, electrodes)
    D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    D[subj[:, None] == subj[None, :]] = np.inf
    out = np.full(len(elec), np.nan)
    n_match = np.zeros(len(elec), int)
    Z = zscore(sig)
    for i in range(len(elec)):
        m = np.flatnonzero(D[i] <= MATCH_MM)
        n_match[i] = len(m)
        if len(m) >= 2:
            out[i] = float(corr_cols(Z[:, [i]], Z[:, m].mean(axis=1))[0])
    elec = elec.assign(isc_matched=out, n_matched=n_match)
    ok = np.isfinite(out)
    log_.info("--- A13 electrode-level ISC, %.0f mm matching ---", MATCH_MM)
    log_.info("  %d of %d electrodes have >=2 cross-subject neighbours",
              int(ok.sum()), len(elec))
    for r in ROIS:
        g = elec[(elec.roi == r) & np.isfinite(elec.isc_matched)]
        if len(g) >= 5:
            log_.info("  %-4s n=%3d  median ISC %+.3f  ceiling r2 %.4f  "
                      "median peak dR2_SAE %.5f  => %.0f%% of ceiling", r, len(g),
                      g.isc_matched.median(), g.isc_matched.median() ** 2,
                      g.peak_sae.median(),
                      100 * g.peak_sae.median() / max(g.isc_matched.median() ** 2, 1e-9))
    return elec.drop(columns=["sig"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_ = setup_logging(cfg.paths.results_root / "logs", "a13", cfg.digest)

    roi_sig, elec = collect(cfg)
    out = cfg.results("inter_subject_correlation", "")
    out.mkdir(parents=True, exist_ok=True)

    full, best = roi_ceiling(cfg, roi_sig, log_)
    full.to_csv(out / "a13_roi_isc_by_lag.csv", index=False)

    e = electrode_ceiling(cfg, elec, log_)
    e.to_csv(out / "a13_electrode_isc.csv", index=False)

    # headline: unique SAE variance as a fraction of the ROI ceiling
    rows = []
    for _, b in best.iterrows():
        g = e[e.roi == b.roi]
        rows.append({**b.to_dict(), "median_peak_dr2_sae": float(g.peak_sae.median()),
                     "pct_of_ceiling_loo": 100 * float(g.peak_sae.median()) /
                     max(b.ceiling_r2_loo, 1e-9),
                     "pct_of_ceiling_pairwise": 100 * float(g.peak_sae.median()) /
                     max(b.ceiling_r2_pairwise, 1e-9)})
    hz = pd.DataFrame(rows)
    hz.to_csv(out / "a13_normalised.csv", index=False)
    log_.info("--- A13 headline: unique SAE variance against the ROI ceiling ---")
    for _, r in hz.iterrows():
        log_.info("  %-4s dR2 %.5f  ceiling(loo) %.4f -> %5.1f%%   "
                  "ceiling(pairwise) %.4f -> %5.1f%%", r.roi, r.median_peak_dr2_sae,
                  r.ceiling_r2_loo, r.pct_of_ceiling_loo,
                  r.ceiling_r2_pairwise, r.pct_of_ceiling_pairwise)
    print("written:", out)


if __name__ == "__main__":
    main()
