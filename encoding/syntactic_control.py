"""Does adding a syntactic band reduce the sparse band's unique variance in IFG?

Compares the 4-band variance partition with a 5-band partition that adds the dataset's
79-column syntactic band (``config/config_syntactic.yaml``), over the IFG electrodes both
contain, under three lag conventions:

* **own peak** - each model read at its own peak lag;
* **mean, 0-500 ms** - no lag selection;
* **4-band peak lag** - both models read at the 4-band model's peak, a convention biased
  toward a loss because the 5-band model is evaluated away from its own peak (reported, not
  part of the tolerance criterion).

The statistic is the change in the median unique sparse-band variance, judged against a
20% tolerance. Writes ``syntactic_control/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from capacity_matching.matched_capacity import isolated_partition
from inference.statistics import temporal_consistency

THRESHOLD_PCT = -20.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--a5-config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config_syntactic.yaml")
    args = ap.parse_args()
    cfg, a5 = load_config(args.config), load_config(args.a5_config)
    lags = cfg.lags.centres_s

    d4, d5 = [], []
    for sub in cfg.subjects:
        f5 = isolated_partition(a5, sub)
        if f5 is None:
            continue
        f4 = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                     allow_pickle=True)
        i4 = f4["roi"].astype(str) == "IFG"
        i5 = f5["roi"].astype(str) == "IFG"
        if list(f4["ch_names"][i4]) != list(f5["ch_names"][i5]):
            raise SystemExit(f"{sub}: IFG electrodes differ between the two runs")
        d4.append(f4["delta_sae"][i4])
        d5.append(f5["delta_sae"][i5])
    d4, d5 = np.vstack(d4), np.vstack(d5)
    r = np.arange(len(d4))
    fin = lambda M: np.where(np.isfinite(M), M, -np.inf)
    li4, li5 = fin(d4).argmax(1), fin(d5).argmax(1)
    win = (lags >= 0.0) & (lags <= 0.5)
    conv = [("4-band peak lag", d4[r, li4], d5[r, li4], False),
            ("own peak", d4[r, li4], d5[r, li5], True),
            ("mean, 0-500 ms", d4[:, win].mean(1), d5[:, win].mean(1), True)]
    rows = []
    for name, a, b, unbiased in conv:
        m4, m5 = float(np.median(a)), float(np.median(b))
        rows.append({"convention": name, "unbiased": unbiased, "median_4band": m4,
                     "median_5band": m5, "pct_change": 100.0 * (m5 - m4) / abs(m4),
                     "n_electrodes": len(a)})
    T = pd.DataFrame(rows)
    tc4, tc5 = temporal_consistency(d4, lags), temporal_consistency(d5, lags)
    T["peak_of_mean_lag_ms_4band"] = 1000 * float(tc4["peak_of_mean_lag_s"])
    T["peak_of_mean_lag_ms_5band"] = 1000 * float(tc5["peak_of_mean_lag_s"])
    ok = T.loc[T.unbiased, "pct_change"].min() > THRESHOLD_PCT
    T["verdict"] = ("SUPPORTED: both unbiased conventions change by less than 20%" if ok
                    else "NOT SUPPORTED: an unbiased convention exceeds the 20% reduction")
    print(T.to_string(index=False))
    o = cfg.results("syntactic_control")
    o.mkdir(parents=True, exist_ok=True)
    T.to_csv(o / "a5_summary.csv", index=False)
    print("written:", o / "a5_summary.csv")


if __name__ == "__main__":
    main()
