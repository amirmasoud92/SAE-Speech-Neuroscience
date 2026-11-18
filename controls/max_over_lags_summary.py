"""Summary of the max-over-lags circular-shift null.

For each region and for the language-minus-control contrast, the observed median (across the
region's electrodes) of the proxy statistic is compared with its distribution over shifts:

* **peak over lags**: P is the share of shifts at least as large (one-sided, +1 corrected);
* **mean over 0-500 ms**: the fixed-window statistic.

The proxy's validity is its electrode-wise rank correlation with the banded model's peak and
window unique variance. Writes ``controls/fixed_window/a8_shift_null_summary.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import load_config

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    o = cfg.results("controls", "fixed_window")
    z = np.load(o / "shift_null.npz", allow_pickle=True)
    n = int(z["done"])
    roi, subj = z["roi"].astype(str), z["subj"].astype(str)
    PK, WN = z["null_peak"][:n], z["null_win"][:n]
    pk0, wn0 = z["obs_peak"], z["obs_win"]
    lang = set(cfg.electrodes["language_rois"])
    groups = {r: roi == r for r in sorted(set(roi))}
    groups["language network"] = np.isin(roi, list(lang))
    groups["control network"] = ~groups["language network"]
    rows = []
    for g, m in groups.items():
        for lab, obs, null in (("peak over lags", pk0, PK), ("mean 0-500 ms", wn0, WN)):
            o_ = float(np.median(obs[m]))
            nl = np.median(null[:, m], 1)
            rows.append({"group": g, "statistic": lab, "observed_median": o_,
                         "null_mean": float(nl.mean()), "null_95th": float(np.percentile(nl, 95)),
                         "bias_corrected": o_ - float(nl.mean()),
                         "p_one_sided": float((1 + (nl >= o_).sum()) / (n + 1)),
                         "n_electrodes": int(m.sum()), "n_participants": int(len(set(subj[m]))),
                         "n_shifts": n})
    for lab, obs, null in (("peak over lags", pk0, PK), ("mean 0-500 ms", wn0, WN)):
        L, C = groups["language network"], groups["control network"]
        o_ = float(np.median(obs[L]) - np.median(obs[C]))
        nl = np.median(null[:, L], 1) - np.median(null[:, C], 1)
        rows.append({"group": "language - control", "statistic": lab, "observed_median": o_,
                     "null_mean": float(nl.mean()), "null_95th": float(np.percentile(nl, 95)),
                     "bias_corrected": o_ - float(nl.mean()),
                     "p_one_sided": float((1 + (nl >= o_).sum()) / (n + 1)),
                     "n_electrodes": int(L.sum() + C.sum()), "n_participants": int(len(set(subj))),
                     "n_shifts": n})
    T = pd.DataFrame(rows)
    # validity of the proxy against the banded model, same electrodes
    from inference.electrode_inference import load_observed
    obs, meta = load_observed(cfg, "sae")
    lags = cfg.lags.centres_s
    win = (lags >= -1e-9) & (lags <= 0.5 + 1e-9)
    keep = meta.roi.isin(set(roi)).to_numpy()
    bp, bw = np.nanmax(obs[keep], 1), np.nanmean(obs[keep][:, win], 1)
    assert len(bp) == len(pk0) and (meta.roi[keep].to_numpy() == roi).all()
    T.attrs = {}
    val = pd.DataFrame([{"check": "proxy vs banded, peak (Spearman over electrodes)",
                         "rho": sps.spearmanr(bp, pk0)[0]},
                        {"check": "proxy vs banded, 0-500 ms mean (Spearman)",
                         "rho": sps.spearmanr(bw, wn0)[0]}])
    T.to_csv(o / "a8_shift_null_summary.csv", index=False)
    val.to_csv(o / "a8_proxy_validity.csv", index=False)
    with pd.option_context("display.width", 220):
        print(T.round(5).to_string(index=False))
        print(val.to_string(index=False))


if __name__ == "__main__":
    main()
