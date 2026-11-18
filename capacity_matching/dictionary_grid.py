"""Neural results across the five dictionaries of the sensitivity grid.

For each dictionary (width 16,384 at L0 = 22, 82 and 176; width 32,768 at L0 = 76; width
65,536 at L0 = 72), from its own variance partition
(``config/config_dict_*.yaml``):

* IFG median peak unique variance;
* the temporal consistency ratio, peak of the mean profile over the mean of electrode peaks;
* the lag of the peak of the mean profile.

Each is also expressed as a fraction of the full model's held-out R2, which is comparable
across dictionaries. Writes ``dictionary_grid/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from inference.statistics import temporal_consistency

VARIANTS = [
    ("w16k L0=82 (main)", "config.yaml"),
    ("w16k L0=22 (sparser)", "config_dict_w16k_l0_22.yaml"),
    ("w16k L0=176 (denser)", "config_dict_w16k_l0_176.yaml"),
    ("w32k L0=76", "config_dict_w32k_l0_76.yaml"),
    ("w65k L0=72", "config_dict_w65k_l0_72.yaml"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1] / "config")
    args = ap.parse_args()

    rows = []
    for label, name in VARIANTS:
        path = args.code / name
        if not path.exists():
            print(f"  {label:<24s} config missing, skipped")
            continue
        cfg = load_config(path)
        D, F, roi, lags = [], [], [], None
        missing = False
        for sub in cfg.subjects:
            try:
                v = M.isolated_partition(cfg, sub)
            except FileNotFoundError:
                missing = True
                break
            if v is None:                   # no IFG electrodes: IFG-only run wrote no file
                continue
            D.append(v["delta_sae"])
            F.append(v["full"])
            roi.append(v["roi"].astype(str))
            lags = v["lags_s"]
        if missing or not D:
            print(f"  {label:<24s} fit not finished, skipped")
            continue
        D, F = np.vstack(D), np.vstack(F)
        roi = np.concatenate(roi)
        ifg = roi == "IFG"
        tc = temporal_consistency(D[ifg], lags)
        peak = np.nanmax(np.where(np.isfinite(D[ifg]), D[ifg], -np.inf), 1)
        fpk = np.nanmax(np.where(np.isfinite(F[ifg]), F[ifg], -np.inf), 1)
        ok = fpk > 0.01
        rows.append({"dictionary": label, "digest": cfg.digest,
                     "width": cfg.sae["primary"]["width"], "l0": cfg.sae["primary"]["l0"],
                     "n_ifg": int(ifg.sum()),
                     "ifg_median_peak": float(np.median(peak)),
                     "ifg_pct_of_full": float(100 * np.median(peak[ok] / fpk[ok]))
                     if ok.any() else np.nan,
                     "temporal_consistency": float(tc["ratio"]),
                     "peak_of_mean_lag_ms": float(tc["peak_of_mean_lag_s"] * 1000)})
        print(f"  {label:<24s} done")

    if not rows:
        raise SystemExit("no completed dictionary fits found")
    R = pd.DataFrame(rows)

    print("\nA23b - the neural claims, once per dictionary\n" + "=" * 88)
    print(f"{'dictionary':<24s} {'n':>4s} {'median dR2':>11s} {'% of full':>10s} "
          f"{'consistency':>12s} {'peak lag':>10s}")
    for _, r in R.iterrows():
        print(f"{r.dictionary:<24s} {r.n_ifg:>4d} {r.ifg_median_peak:>11.5f} "
              f"{r.ifg_pct_of_full:>9.1f}% {r.temporal_consistency:>12.3f} "
              f"{r.peak_of_mean_lag_ms:>8.0f} ms")
    print("=" * 88)

    if len(R) > 1:
        lag_spread = float(R.peak_of_mean_lag_ms.max() - R.peak_of_mean_lag_ms.min())
        frac_spread = float(R.ifg_pct_of_full.max() - R.ifg_pct_of_full.min())
        print(f"peak lag spans {lag_spread:.0f} ms; the fraction of the full model spans "
              f"{frac_spread:.1f} percentage points across {len(R)} dictionaries")
        print("VERDICT: " + (
            "the neural findings are ROBUST to dictionary width and sparsity"
            if lag_spread <= 100 and frac_spread <= 15 else
            "the neural findings DEPEND on the dictionary - report the dependence, and "
            "check whether it tracks L0 rather than width"))

    o = load_config(args.code / "config.yaml").results("dictionary_grid")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "a23b_neural_side.csv", index=False)
    print("written:", o / "a23b_neural_side.csv")


if __name__ == "__main__":
    main()
