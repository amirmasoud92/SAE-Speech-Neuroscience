"""Film effect sizes as a share of the full stimulus model.

The procedure of ``encoding/band_shares.py`` applied to the film fits (no fitting): each contact
is read at the lag where its full model peaks, the sparse band's unique variance there is
divided by that peak R2, and only contacts with a full-model peak R2 above 0.01 are kept.
Medians by region carry participant-clustered 95% intervals (10,000 draws). The podcast
comparison reads the primary partition over all 161 lags and over the 21 lags from 0 to
500 ms that the film cohort has.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from inference.statistics import cluster_bootstrap

REGIONS = ["IFG", "STG", "MTG", "TMP", "EAC", "MFG", "PRC"]


def table(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = []
    ok = rows[rows.full_peak > 0.01]
    for roi in REGIONS + ["language"]:
        g = ok[ok.roi.isin(["IFG", "STG", "MTG", "TMP"])] if roi == "language" else \
            ok[ok.roi == roi]
        if g.subject.nunique() < 3:
            continue
        frac = (g.delta_at_peak / g.full_peak).to_numpy()
        m, lo, hi = cluster_bootstrap(frac, g.subject.to_numpy(), n_resamples=10000,
                                      statistic=np.median, seed=seed)
        out.append({"roi": roi, "n": len(g), "n_subjects": int(g.subject.nunique()),
                    "n_contacts_total": int((rows.roi == roi).sum()) if roi != "language"
                    else int(rows.roi.isin(["IFG", "STG", "MTG", "TMP"]).sum()),
                    "median_full_peak": float(g.full_peak.median()),
                    "median_fraction": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("film_h1_h2")
    rows = []
    for f in sorted((out / "fits").glob("sub_*.npz")):
        z = np.load(f, allow_pickle=True)
        full, d = z["full"], z["delta_sae"]
        pk = np.nanargmax(np.where(np.isfinite(full), full, -np.inf), axis=1)
        r = np.arange(len(pk))
        for i, roi in enumerate(z["roi"].astype(str)):
            rows.append({"subject": f.stem, "roi": roi, "full_peak": float(full[i, pk[i]]),
                         "delta_at_peak": float(d[i, pk[i]]), "peak_lag_ms": 25.0 * pk[i]})
    film = pd.DataFrame(rows)
    pod = []
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        for lab, sl in (("161", slice(None)), ("0_500", slice(80, 101))):
            full, d = v["full"][:, sl], v["delta_sae"][:, sl]
            pk = np.nanargmax(np.where(np.isfinite(full), full, -np.inf), axis=1)
            for i, roi in enumerate(v["roi"].astype(str)):
                pod.append({"reading": lab, "subject": sub, "roi": roi,
                            "full_peak": float(full[i, pk[i]]),
                            "delta_at_peak": float(d[i, pk[i]])})
    pod = pd.DataFrame(pod)
    T = pd.concat([table(film, cfg.seed + 46).assign(cohort="film (0-500 ms)"),
                   table(pod[pod.reading == "161"], cfg.seed + 46).assign(
                       cohort="podcast (161 lags)"),
                   table(pod[pod.reading == "0_500"], cfg.seed + 46).assign(
                       cohort="podcast (0-500 ms)")], ignore_index=True)
    T.to_csv(out / "a46_fractions.csv", index=False)
    film.to_csv(out / "a46_film_contacts.csv", index=False)
    print(T[["cohort", "roi", "n", "n_subjects", "median_full_peak", "median_fraction",
             "ci_lo", "ci_hi"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
