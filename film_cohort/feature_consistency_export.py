"""Per-participant importances behind the cross-cohort feature consistency.

Recomputes the IFG importances of ``film_cohort/feature_consistency.py`` exactly (same
functions, seeds, shifts and entry rule) and keeps what is needed downstream:

- podcast: the observed |z| per IFG electrode, with participant labels, and the importance under
  200 outer shifts of at least 172 words (FFT route);
- film: per participant, the IFG contacts' summed |z| under the observed alignment and each of
  the 200 outer shifts, the live features and the activation counts.

The IFG statistic must be reproduced from the export before it is written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as sfft

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from film_cohort.feature_consistency import (N_OUTER, N_SHIFTS, centred, importance_direct,
                                             importance_fft, partial_spearman, resid_fn, response)
from encoding.fit_encoding_model import analysed_words

from film_cohort.replication import load_subject

REGIONS = ["IFG", "STG", "MTG", "EAC", "MFG", "PRC"]       # same order as feature_consistency: fixes the seeds


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("feature_consistency")
    w = analysed_words(cfg).reset_index(drop=True)
    d = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)
    X, pids = d["X"], np.asarray(d["ids_sae"]).astype(int)
    Xs = X[:, 0:4933]
    res_p = resid_fn(np.column_stack([X[:, 4933:4934], X[:, 4934:6294], X[:, 6294:6297],
                                      np.ones(len(X), dtype=np.float32)]))
    Sc_p, nx_p = centred(Xs, res_p)
    rng = np.random.default_rng(cfg.seed)
    sh_p = rng.choice(np.arange(172, len(w) - 172), N_SHIFTS, replace=False)
    Yw, Vw, subj, _ = Mt.ifg_targets(cfg, "window", roi="IFG")
    pod_obs = np.stack([importance_direct(Sc_p, nx_p, response(Yw[:, j], Vw[:, j], res_p),
                                          sh_p) for j in range(Yw.shape[1])])
    # podcast outer-shift null for coherence_and_importance (FFT route)
    n_p = len(w)
    g = np.random.default_rng(cfg.seed + 4500)
    outer_p = np.r_[0, g.choice(np.arange(172, n_p - 172), N_OUTER, replace=False)]
    ScF = sfft.rfft(Sc_p, axis=0, workers=-1)
    pod_null = np.zeros((N_OUTER + 1, len(pids)))
    for j in range(Yw.shape[1]):
        pod_null += importance_fft(ScF, n_p, response(Yw[:, j], Vw[:, j], res_p), sh_p,
                                   outer_p)
    pod_null /= Yw.shape[1]
    live_p = np.isfinite(nx_p)
    print(f"podcast IFG: {Yw.shape[1]} electrodes; FFT observed vs direct max |diff| "
          f"{np.abs(pod_null[0][live_p] - pod_obs.mean(0)[live_p]).max():.1e}", flush=True)
    del ScF

    base = cfg.results("film_replication")
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: int(s.split("_")[1]))
    pos = {f: i for i, f in enumerate(pids)}
    film = {}
    for i, sub in enumerate(subs):
        S = load_subject(cfg, sub, base)
        roi = S["C"]["roi"].to_numpy().astype(str)
        if not np.isin(roi, REGIONS).any():
            continue
        keep = np.isin(S["sae_ids"], pids)
        F = np.asarray(S["sae"], np.float64)[:, keep]
        fpos = np.array([pos[f] for f in S["sae_ids"][keep]])
        res_f = resid_fn(S["nuis"])
        Sc, nx = centred(F, res_f)
        n = len(F)
        t = S["W"]["t_global"].to_numpy()
        floor = int(round(60.0 * n / (t[-1] - t[0])))
        g = np.random.default_rng(cfg.seed + 4200 + i)
        sh = g.choice(np.arange(floor, n - floor), N_SHIFTS, replace=False)
        outer = np.r_[0, g.choice(np.arange(floor, n - floor), N_OUTER, replace=False)]
        cols = np.flatnonzero(roi == "IFG")
        if not len(cols):
            continue
        ScF = sfft.rfft(Sc, axis=0, workers=-1)
        s = np.zeros((N_OUTER + 1, len(fpos)))
        for j in cols:
            s += importance_fft(ScF, n, response(S["Y"][:, j], S["V"][:, j], res_f), sh,
                                outer)
        film[sub] = {"fpos": fpos, "sum": s, "n": len(cols), "act": (F > 0).sum(0),
                     "words": n}
        print(f"{sub}: {len(cols)} IFG contacts", flush=True)
        del ScF, Sc

    # reproduce the IFG statistic of feature_consistency from the export
    nF = len(pids)
    sums, cnt, act, held = np.zeros((N_OUTER + 1, nF)), np.zeros(nF), np.zeros(nF), np.zeros(nF)
    words = 0.0
    for v in film.values():
        sums[:, v["fpos"]] += v["sum"]
        cnt[v["fpos"]] += v["n"]
        held[v["fpos"]] += v["n"]
        act[v["fpos"]] += v["act"]
        words += v["words"]
    n_contacts = sum(v["n"] for v in film.values())
    dens_p = (Xs > 0).sum(0)
    elig = (held >= 0.5 * n_contacts) & (dens_p >= 25)
    impF = sums[:, elig] / cnt[elig]
    lf = np.log(np.maximum(act[elig], 0.5) / words)
    lp = np.log(dens_p)[elig]
    rho = partial_spearman(pod_obs.mean(0)[elig], impF[0], lp, lf)
    ref = pd.read_csv(out / "a42_consistency.csv").set_index("roi").loc["IFG"]
    print(f"IFG statistic reproduced: partial rho {rho:+.6f} (stored {ref.partial_rho:+.6f}), "
          f"{int(elig.sum())} features (stored {int(ref.n_features)})")
    if abs(rho - ref.partial_rho) > 1e-9 or int(elig.sum()) != int(ref.n_features):
        raise SystemExit("export does not reproduce the stored statistic - nothing written")
    np.savez_compressed(
        out / "a42_ifg_export.npz", pids=pids, elig=elig, dens_p=dens_p,
        pod_obs=pod_obs, pod_subj=np.asarray(subj).astype(str), pod_null=pod_null,
        film_subs=np.array(list(film)),
        **{f"film_{k}_{s}": v[k] for s, v in film.items()
           for k in ("fpos", "sum", "act")},
        **{f"film_n_{s}": v["n"] for s, v in film.items()},
        **{f"film_words_{s}": v["words"] for s, v in film.items()})
    print("written", out / "a42_ifg_export.npz")


if __name__ == "__main__":
    main()
