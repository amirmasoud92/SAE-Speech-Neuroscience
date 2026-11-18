"""Participant-level sign-flip tests and electrode-level family-wise tests.

No model fits; reads stored per-electrode outputs.

* Exact participant-level sign-flip tests of the within-type and between-type sparse-minus-dense
  differences and of their interaction, with participants weighted equally.
* Two one-sided tests of the window contrasts against the equivalence margin.
* Lag-resolved clusters of the sparse-minus-dense difference.
* Electrode-level tests under the closed-form proxy statistic: an electrode survives if its
  peak over lags exceeds the 95th percentile of the maximum over all language and control
  electrodes and lags in each of 1,000 circular shifts of the stimulus.

Writes ``controls/sign_flip_and_electrode_tests/``.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent


def signflip(per_elec, subj, stat="t"):
    """Exact two-sided sign-flip P over all 2^n participant sign assignments.
    stat 't': one-sample t of participant means; 'median': mean of participant medians."""
    uq = list(dict.fromkeys(subj))
    if stat == "t":
        S = np.array([np.nanmean(per_elec[subj == u]) for u in uq])
    else:
        S = np.array([np.nanmedian(per_elec[subj == u]) for u in uq])
    n = len(S)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    X = signs * S
    if stat == "t":
        val = X.mean(1) / (X.std(1, ddof=1) / np.sqrt(n))
        obs = S.mean() / (S.std(ddof=1) / np.sqrt(n))
    else:
        val, obs = X.mean(1), S.mean()
    p = float((np.abs(val) >= abs(obs) - 1e-12).mean())
    return dict(statistic=stat, observed=float(obs), p_exact=p, n_participants=n,
                n_assignments=len(signs), p_floor=float(2 / len(signs)),
                n_participants_negative=int((S < 0).sum()),
                participant_values=";".join(f"{u}:{v:+.4f}" for u, v in zip(uq, S)))


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    R = cfg.results("controls")
    out = R / "sign_flip_and_electrode_tests"
    out.mkdir(exist_ok=True)

    # ------------------------------------- participant-level sign flips, word types
    E = pd.read_csv(R / "within_type_window" / "a1_window_per_electrode.csv")
    W = pd.read_csv(R / "within_type_window" / "a1_window_stats.csv")
    subj = E.subject.to_numpy()
    dw = (E["SAE|within"] - E["Gemma dense|within"]).to_numpy()
    db = (E["SAE|between"] - E["Gemma dense|between"]).to_numpy()
    inter = dw - db
    pub = W[(W.contrast == "SAE - Gemma dense") & (W.component == "within")].iloc[0]
    # the stored bootstrap P must be reproduced (same seed as within_type_and_lag_resolved)
    st_w, _ = PS.electrode_stat(dw, subj, cfg.seed + 1400)
    assert abs(st_w["p_two_sided"] - pub.p_two_sided) < 1e-12 and abs(st_w["estimate"] - pub.estimate) < 1e-12
    rows = []
    for lab, x in (("C1 within-type SAE - dense (0-500 ms)", dw),
                   ("C2 between-type SAE - dense (0-500 ms)", db),
                   ("C2 interaction: within gap - between gap", inter)):
        st, _ = PS.electrode_stat(x, subj, cfg.seed + 1400 if lab.startswith("C1") else
                                  (cfg.seed + 1401 if "between-type" in lab else cfg.seed + 4201))
        for s_ in ("t", "median"):
            sf = signflip(x, subj, s_)
            rows.append({"check": lab, "bootstrap_median": st["estimate"],
                         "bootstrap_ci95_lo": st["ci95_lo"], "bootstrap_ci95_hi": st["ci95_hi"],
                         "bootstrap_p": st["p_two_sided"], **sf,
                         "n_electrodes": st["n_electrodes"]})
    C12 = pd.DataFrame(rows)
    C12.to_csv(out / "c1_c2_signflip.csv", index=False)

    # -------------------------------------------- two one-sided tests against the margin
    rows = []
    s0 = cfg.seed + 2000
    for conv in ("LEG", "TO"):
        for i, (a, b) in enumerate([("Matryoshka-85", "classical-85"),
                                    ("Matryoshka-300", "Gemma dense-300"),
                                    ("SAE-300", "Gemma dense-300")]):
            if "classical" in b:
                continue
            A = pd.read_csv(R / "window_arms" / f"{conv}_{a.replace(' ', '_')}.csv")
            B = pd.read_csv(R / "window_arms" / f"{conv}_{b.replace(' ', '_')}.csv")
            st, boot = PS.electrode_stat(A.r.to_numpy() - B.r.to_numpy(), A.subject.to_numpy(),
                                         s0 + 100 + i)
            for m in (0.0155,):
                pu, pl = float((boot >= m).mean()), float((boot <= -m).mean())
                rows.append({"convention": conv, "contrast": f"{a} - {b}", "margin": m,
                             "margin_status": "pre-specified", "estimate": st["estimate"],
                             "ci90_lo": st["ci90_lo"], "ci90_hi": st["ci90_hi"],
                             "p_upper": pu, "p_lower": pl, "tost_p": max(pu, pl),
                             "ci90_inside_margin": bool(st["ci90_lo"] > -m and st["ci90_hi"] < m)})
    C3 = pd.DataFrame(rows)
    ref = pd.read_csv(R / "matryoshka" / "a2_equivalence.csv")
    for r in C3[C3.margin == 0.0155].itertuples():           # same draws as the report
        q = ref[(ref.convention == r.convention) & (ref.contrast == r.contrast)].iloc[0]
        assert abs(q.ci90_lo - r.ci90_lo) < 1e-12 and abs(q.tost_p - r.tost_p) < 1e-12
    C3.to_csv(out / "c3_tost.csv", index=False)

    z = np.load(R / "fixed_window" / "shift_null.npz", allow_pickle=True)

    # ------------------------------------------------------------- lag-resolved clusters
    rows = []
    for roi in ("IFG", "STG", "MTG", "TMP"):
        C = pd.read_csv(R / "lag_resolved" / f"clusters_{roi}.csv")
        C = C[(C.contrast == "SAE - Gemma dense") & (C.component == "total") & (C.p_cluster < 0.05)]
        for r in C.itertuples():
            rows.append({"roi": roi, "direction": r.direction, "lag_start_ms": round(r.lag_start_ms),
                         "lag_end_ms": round(r.lag_stop_ms), "n_lags": r.n_lags, "mass": r.mass,
                         "mean_diff_in_cluster": r.mean_diff_in_cluster, "p_cluster": r.p_cluster,
                         "n_participants": r.n_participants, "n_assignments": r.n_assignments,
                         "n_electrodes": r.n_electrodes})
    pd.DataFrame(rows).to_csv(out / "c5_a6_clusters.csv", index=False)

    # -------------------------------------------- electrode-level family-wise tests
    n = int(z["done"])
    Mx = z["null_peak"][:n].max(1)                         # max over electrodes and lags
    thr = float(np.percentile(Mx, 95))
    pk = z["obs_peak"]
    p_fwer = (1 + (Mx[None, :] >= pk[:, None]).sum(1)) / (n + 1)
    roi, sj = z["roi"].astype(str), z["subj"].astype(str)
    lags = z["lags"] * 1000
    peak_lag = lags[np.argmax(z["obs_profile"], 1)]
    T = pd.DataFrame({"subject": sj, "roi": roi, "obs_peak": pk, "peak_lag_ms": peak_lag,
                      "p_fwer": p_fwer, "survives": p_fwer <= 0.05})
    T.to_csv(out / "a10_electrodes.csv", index=False)
    by_roi = T.groupby("roi").agg(n_electrodes=("survives", "size"),
                                  n_survivors=("survives", "sum"),
                                  n_participants=("subject", "nunique")).reset_index()
    by_subj = T.groupby("subject").agg(n_electrodes=("survives", "size"),
                                       n_survivors=("survives", "sum")).reset_index()
    surv_part = T[T.survives].groupby("roi").subject.nunique()
    by_roi["n_participants_with_survivor"] = by_roi.roi.map(surv_part).fillna(0).astype(int)
    by_roi.to_csv(out / "a10_by_roi.csv", index=False)
    by_subj.to_csv(out / "a10_by_participant.csv", index=False)
    json.dump({"n_shifts": n, "family": f"{len(T)} language and control electrodes x 161 lags",
               "threshold_95th_of_max": thr, "n_survivors": int(T.survives.sum()),
               "statistic": "closed-form proxy dR2, not the banded model"},
              open(out / "a10_summary.json", "w"), indent=1)

    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(C12.drop(columns=["participant_values"]).to_string(index=False))
        print(C3.to_string(index=False))
        print(pd.read_csv(out / "c5_a6_clusters.csv").to_string(index=False))
        print(f"electrode level: threshold {thr:.5f}; survivors {int(T.survives.sum())} of {len(T)}")
        print(by_roi.to_string(index=False))
        print(by_subj.to_string(index=False))


if __name__ == "__main__":
    main()
