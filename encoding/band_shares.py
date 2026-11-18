"""Each band's unique variance as a share of the full model's held-out R2.

Each electrode is read at the lag where its full model peaks; the unique variance of each
band there is divided by that peak R2. Only electrodes with a full-model peak R2 above 0.01
are kept, and medians are reported for the language network and for IFG. Held-out shares can
fall below 0 or exceed 100%. Writes ``band_shares/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config

LANG = ("IFG", "STG", "MTG", "TMP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    rows = []
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        roi = v["roi"].astype(str)
        ch = v["ch_names"].astype(str)
        full = v["full"]
        bands = [k[len("delta_"):] for k in v.keys() if k.startswith("delta_")]
        fin = np.where(np.isfinite(full), full, -np.inf)
        peak = fin.argmax(1)
        r = np.arange(len(ch))
        rec = {"subject": sub, "ch": ch, "roi": roi,
               "full_peak": full[r, peak], "peak_lag_s": v["lags_s"][peak]}
        for b in bands:
            rec[f"delta_{b}"] = v[f"delta_{b}"][r, peak]
        rows.append(pd.DataFrame(rec))
    D = pd.concat(rows, ignore_index=True)
    bands = [c[len("delta_"):] for c in D.columns if c.startswith("delta_")]
    print(f"{len(D)} electrodes, bands: {bands}")

    # Only electrodes the full model actually explains can carry an interpretable
    # fraction. A negative or ~zero denominator makes the ratio meaningless, not large.
    D["explainable"] = D.full_peak > 0.01
    print(f"{int(D.explainable.sum())} electrodes with full-model peak R2 > 0.01 "
          f"({100 * D.explainable.mean():.0f}%)")

    lang = D[D.roi.isin(LANG) & D.explainable]
    out = []
    for b in bands:
        frac = (lang[f"delta_{b}"] / lang.full_peak).replace([np.inf, -np.inf], np.nan)
        ifg = lang[lang.roi == "IFG"]
        fifg = (ifg[f"delta_{b}"] / ifg.full_peak).replace([np.inf, -np.inf], np.nan)
        out.append({"band": b, "n_language": int(frac.notna().sum()),
                    "median_delta": float(lang[f"delta_{b}"].median()),
                    "median_full": float(lang.full_peak.median()),
                    "median_fraction_language": float(frac.median()),
                    "median_fraction_IFG": float(fifg.median())})
    R = pd.DataFrame(out).sort_values("median_fraction_language", ascending=False)

    print(f"\nA21a - unique variance as a fraction of the full model, language ROIs "
          f"(n = {len(lang)})\n" + "=" * 78)
    print(f"{'band':<14s} {'median dR2':>12s} {'% of full':>11s} {'% of full (IFG)':>17s}")
    for _, x in R.iterrows():
        print(f"{x.band:<14s} {x.median_delta:>12.5f} "
              f"{100 * x.median_fraction_language:>10.1f}% "
              f"{100 * x.median_fraction_IFG:>16.1f}%")
    print("=" * 78)
    print(f"median full-model peak R2 in language cortex: {lang.full_peak.median():.4f}")

    o = cfg.results("band_shares")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "a21a_band_fractions.csv", index=False)
    D.to_csv(o / "a21a_electrodes.csv", index=False)
    print("written:", o / "a21a_band_fractions.csv")


if __name__ == "__main__":
    main()
