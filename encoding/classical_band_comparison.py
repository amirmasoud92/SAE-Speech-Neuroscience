"""Sparse band against the classical linguistic band inside the joint banded model.

Fits the banded model with the 85-column classical band added (``config/config_classical.yaml``)
and compares, in IFG, the sparse band's unique variance with the classical band's at three lag
conventions: at the sparse band's peak, at each band's own peak, and as the mean over 0-500 ms.
Tested per participant and pooled with a participant-clustered bootstrap.

The classical band: 35 part-of-speech tags, 43 dependency relations and a stopword flag from the
dataset's parse, concreteness (Brysbaert et al., 2014), valence, arousal and dominance (Warriner
et al., 2013) and two norm-availability indicators. Frequency, length and duration stay in the
always-present lexical band, so both sides control them identically. Writes ``classical_band/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import load_config

CONVENTIONS = ["own peak", "at the SAE peak", "mean over 0-500 ms"]
N_BOOT = 10000


def load(cfg, sub):
    """None for a subject without IFG electrodes (the variant run is IFG-only)."""
    from capacity_matching.matched_capacity import isolated_partition
    return isolated_partition(cfg, sub)


def measure(dA, dB, lags, how):
    """Return (a, b) per electrode under one lag convention, never measuring B at A's peak
    unless that is the convention being tested."""
    win = (lags >= 0.0) & (lags <= 0.5)
    fin = lambda M: np.where(np.isfinite(M), M, -np.inf)
    if how == "own peak":
        return fin(dA).max(1), fin(dB).max(1)
    if how == "at the SAE peak":
        i = fin(dA).argmax(1)
        r = np.arange(len(dA))
        return dA[r, i], dB[r, i]
    return np.nanmean(dA[:, win], 1), np.nanmean(dB[:, win], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config_classical.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)

    rows, per_sub = [], []
    for sub in cfg.subjects:
        v = load(cfg, sub)
        if v is None:
            continue
        roi = v["roi"].astype(str)
        lags = v["lags_s"]
        ifg = roi == "IFG"
        if ifg.sum() == 0:
            continue
        for how in CONVENTIONS:
            a, b = measure(v["delta_sae"][ifg], v["delta_classical"][ifg], lags, how)
            per_sub.append({"subject": sub, "convention": how, "n": int(ifg.sum()),
                            "sae": float(np.median(a)), "classical": float(np.median(b)),
                            "diff": float(np.median(a - b))})
            for j in range(int(ifg.sum())):
                rows.append({"subject": sub, "convention": how,
                             "sae": a[j], "classical": b[j], "diff": a[j] - b[j]})
    E = pd.DataFrame(rows)
    S = pd.DataFrame(per_sub)

    out_rows = []
    for how in CONVENTIONS:
        e = E[E.convention == how]
        subs = e.subject.unique()
        obs = float(np.median(e["diff"]))
        # cluster bootstrap: resample subjects, not electrodes, so a subject contributing
        # many electrodes cannot manufacture a narrow interval
        boot = np.empty(N_BOOT)
        by = {s: e[e.subject == s]["diff"].to_numpy() for s in subs}
        for i in range(N_BOOT):
            pick = rng.choice(subs, len(subs), replace=True)
            boot[i] = np.median(np.concatenate([by[s] for s in pick]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_boot = float((np.sum(boot <= 0) + 1) / (N_BOOT + 1))
        w = sps.wilcoxon(e["sae"], e["classical"], alternative="greater")
        s = S[S.convention == how]
        n_win = int((s["diff"] > 0).sum())
        out_rows.append({"convention": how, "n_electrodes": len(e),
                         "n_subjects": len(subs),
                         "median_sae": float(np.median(e["sae"])),
                         "median_classical": float(np.median(e["classical"])),
                         "median_diff": obs, "ci_lo": lo, "ci_hi": hi,
                         "p_cluster_boot": p_boot, "p_wilcoxon": float(w.pvalue),
                         "subjects_favouring_sae": f"{n_win}/{len(s)}"})

    R = pd.DataFrame(out_rows)
    o = cfg.results("classical_band")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "a14_primary.csv", index=False)
    S.to_csv(o / "a14_per_subject.csv", index=False)
    E.to_csv(o / "a14_electrodes.csv", index=False)

    print("\nA14 primary - IFG, SAE unique vs classical unique\n" + "=" * 78)
    for _, r in R.iterrows():
        print(f"{r.convention:<20s} SAE {r.median_sae:.5f}  classical "
              f"{r.median_classical:.5f}  diff {r.median_diff:+.5f} "
              f"[{r.ci_lo:+.5f}, {r.ci_hi:+.5f}]  p_boot {r.p_cluster_boot:.4f}  "
              f"subjects {r.subjects_favouring_sae}")
    print("=" * 78)
    holds = bool((R.p_cluster_boot < 0.05).all())
    print(f"H-A14 {'HOLDS at all three conventions' if holds else 'DOES NOT hold at all three conventions'}")
    print("written:", o / "a14_primary.csv")


if __name__ == "__main__":
    main()
