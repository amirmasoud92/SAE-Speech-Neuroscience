"""Do the same sparse features matter for IFG in both cohorts?

Both cohorts are read through one dictionary, so feature j is the same feature in each.
Per-feature neural importance is the quantity of ``interpretability/feature_table.py``: the
nuisance-controlled correlation of the feature with a contact's 0-500 ms window response, as
|z| against 200 circular shifts of the response, averaged over a region's contacts.

The film cohort needs 200 further outer shifts per contact, each with the inner 200. Every
shift is a lag of one circular cross-correlation, so each contact's full cross-correlation with
every feature is computed once by FFT and all shifts are read from it. Both routes are checked
against each other, and the podcast route against the stored 300-feature importances.

Primary: the partial Spearman correlation of podcast and film IFG importance, controlling log
activation density in both cohorts, against the outer-shift null. Also: the raw correlation,
the other regions (Holm), the region matrix and within-cohort split-half reliabilities. Writes
``feature_consistency/``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as sfft
from scipy import stats as sps

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

from film_cohort.replication import load_subject

REGIONS = ["IFG", "STG", "MTG", "EAC", "MFG", "PRC"]
N_SHIFTS, N_OUTER = 200, 200


def resid_fn(nuis):
    Q, _ = np.linalg.qr(np.asarray(nuis, np.float64))
    return lambda A: np.asarray(A, np.float64) - Q @ (Q.T @ np.asarray(A, np.float64))


def centred(F, res):
    Sc = res(F)
    Sc -= Sc.mean(0)
    nx = np.sqrt((Sc ** 2).sum(0))
    nx[nx == 0] = np.inf
    return Sc, nx


def response(y, v, res):
    """As in ``interpretability/feature_table.py``: fill masked, residualise, demean on valid,
    masked -> 0."""
    y = np.where(v, y, y[v].mean())
    yc = res(y[:, None])[:, 0]
    return np.where(v, yc - yc[v].mean(), 0.0)


def importance_direct(Sc, nx, yc, sh):
    """The ``importance`` of ``interpretability/feature_table.py``, verbatim."""
    r = (Sc.T @ yc) / (nx * np.sqrt((yc ** 2).sum()))
    Yn = np.stack([np.roll(yc, int(s)) for s in sh], 1)
    Yn -= Yn.mean(0)
    rn = (Sc.T @ Yn) / np.outer(nx, np.sqrt((Yn ** 2).sum(0)))
    return np.abs((r - rn.mean(1)) / (rn.std(1, ddof=1) + 1e-12))


def importance_fft(ScF, n, yc, sh, outer, chunk=25):
    """|z| for every outer shift (rows) and feature (columns), from one cross-correlation."""
    cc = sfft.irfft(ScF * np.conj(sfft.rfft(yc))[:, None], n=n, axis=0, workers=-1)
    out = np.empty((len(outer), cc.shape[1]))
    for c0 in range(0, len(outer), chunk):
        o = outer[c0:c0 + chunk]
        null = cc[(o[:, None] + sh[None, :]) % n]            # (chunk, n_shifts, F)
        out[c0:c0 + chunk] = np.abs((cc[o] - null.mean(1)) / (null.std(1, ddof=1) + 1e-300))
    return out


def rank(x):
    return sps.rankdata(x)


def partial_spearman(a, b, c1, c2):
    Z = np.column_stack([np.ones(len(a)), rank(c1), rank(c2)])
    ra = rank(a) - Z @ np.linalg.lstsq(Z, rank(a), rcond=None)[0]
    rb = rank(b) - Z @ np.linalg.lstsq(Z, rank(b), rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = cfg.results("feature_consistency")
    out.mkdir(parents=True, exist_ok=True)

    # ================================ podcast ==========================================
    w = analysed_words(cfg).reset_index(drop=True)
    d = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)
    X, pids = d["X"], np.asarray(d["ids_sae"]).astype(int)
    Xs = X[:, 0:4933]
    res_p = resid_fn(np.column_stack([X[:, 4933:4934], X[:, 4934:6294], X[:, 6294:6297],
                                      np.ones(len(X), dtype=np.float32)]))
    Sc_p, nx_p = centred(Xs, res_p)
    rng = np.random.default_rng(cfg.seed)
    sh_p = rng.choice(np.arange(172, len(w) - 172), N_SHIFTS, replace=False)   # as in feature_table
    dens_p = (Xs > 0).sum(0)
    P, Psub = {}, {}
    for roi in REGIONS:
        Yw, Vw, subj, _ = Mt.ifg_targets(cfg, "window", roi=roi)
        per = np.stack([importance_direct(Sc_p, nx_p, response(Yw[:, j], Vw[:, j], res_p),
                                          sh_p) for j in range(Yw.shape[1])])
        P[roi], Psub[roi] = per, np.asarray(subj)
        print(f"podcast {roi}: {per.shape[0]} electrodes, "
              f"{len(np.unique(subj))} participants", flush=True)

    # reproduction of the stored 300-feature IFG importances
    ft = pd.read_csv(cfg.results("feature_table", "feature_table_ifg.csv"))
    pos = {f: i for i, f in enumerate(pids)}
    rep = P["IFG"].mean(0)[[pos[f] for f in ft.feature_id]]
    d_rep = float(np.abs(rep - ft.importance_z.to_numpy()).max())
    print(f"stored IFG importance reproduced: max |diff| {d_rep:.2e}")
    if d_rep > 1e-6:
        raise SystemExit("podcast importance does not reproduce the stored values")

    # FFT route against the direct function, on one podcast electrode
    n_p = len(w)
    ScF_p = sfft.rfft(Sc_p, axis=0, workers=-1)
    yc = response(Yw[:, 0], Vw[:, 0], res_p)
    zf = importance_fft(ScF_p, n_p, yc, sh_p, np.array([0]))[0]
    live = np.isfinite(nx_p)
    d_fft = float(np.abs(zf[live] - importance_direct(Sc_p, nx_p, yc, sh_p)[live]).max())
    print(f"FFT route vs direct function: max |diff| {d_fft:.2e}")
    if d_fft > 1e-6:
        raise SystemExit("FFT importance does not match the direct function")
    del ScF_p

    # ================================ film ============================================
    base = cfg.results("film_replication")
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: int(s.split("_")[1]))
    nF = len(pids)
    acc = {r: {"sum": np.zeros((N_OUTER + 1, nF)), "cnt": np.zeros(nF),
               "act": np.zeros(nF), "words": 0.0, "n_contacts": 0,
               "per_sub": {}} for r in REGIONS}
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
        ScF = sfft.rfft(Sc, axis=0, workers=-1)
        for r in REGIONS:
            cols = np.flatnonzero(roi == r)
            if not len(cols):
                continue
            a = acc[r]
            s_sum = np.zeros(len(fpos))
            for j in cols:
                z = importance_fft(ScF, n, response(S["Y"][:, j], S["V"][:, j], res_f),
                                   sh, outer)
                a["sum"][:, fpos] += z
                s_sum += z[0]
            a["cnt"][fpos] += len(cols)
            a["n_contacts"] += len(cols)
            a["act"][fpos] += (F > 0).sum(0)
            a["words"] += n
            a["per_sub"][sub] = (fpos, s_sum, len(cols))
        print(f"{sub}: {n} words, {len(fpos)} shared live features, floor {floor} words, "
              f"contacts {dict(zip(*np.unique(roi, return_counts=True)))}", flush=True)
        del ScF, Sc

    # ================================ statistics ======================================
    lp = np.log(dens_p)
    rows, mats, ceil = [], {}, []
    for r in REGIONS:
        a = acc[r]
        if len(a["per_sub"]) < 3 or len(np.unique(Psub[r])) < 3:
            continue
        # entry rule: live in the podcast, and in participants holding >= half the contacts
        held = np.zeros(nF)
        for fpos, _, nc in a["per_sub"].values():
            held[fpos] += nc
        elig = (held >= 0.5 * a["n_contacts"]) & (dens_p >= 25)
        impP = P[r].mean(0)[elig]
        impF = (a["sum"][:, elig] / a["cnt"][elig])                  # (201, F)
        lf = np.log(np.maximum(a["act"][elig], 0.5) / a["words"])
        obs_p = partial_spearman(impP, impF[0], lp[elig], lf)
        obs_r = sps.spearmanr(impP, impF[0]).statistic
        null_p = np.array([partial_spearman(impP, impF[b], lp[elig], lf)
                           for b in range(1, N_OUTER + 1)])
        null_r = np.array([sps.spearmanr(impP, impF[b]).statistic
                           for b in range(1, N_OUTER + 1)])
        row = {"roi": r, "n_features": int(elig.sum()),
               "n_contacts_film": int(a["n_contacts"]),
               "n_participants_film": len(a["per_sub"]),
               "n_electrodes_podcast": int(P[r].shape[0]),
               "partial_rho": obs_p, "partial_null_mean": float(null_p.mean()),
               "partial_null_95": float(np.percentile(null_p, 95)),
               "p_partial": float((1 + (null_p >= obs_p).sum()) / (N_OUTER + 1)),
               "raw_rho": float(obs_r), "raw_null_mean": float(null_r.mean()),
               "raw_null_95": float(np.percentile(null_r, 95)),
               "p_raw": float((1 + (null_r >= obs_r).sum()) / (N_OUTER + 1))}
        # the 300-feature pool
        pool = np.isin(pids, ft.feature_id.to_numpy())[elig]
        if pool.sum() >= 30:
            row["pool_n"] = int(pool.sum())
            row["pool_partial_rho"] = partial_spearman(impP[pool], impF[0][pool],
                                                       lp[elig][pool], lf[pool])
            nb = np.array([partial_spearman(impP[pool], impF[b][pool], lp[elig][pool],
                                            lf[pool]) for b in range(1, N_OUTER + 1)])
            row["pool_p_partial"] = float((1 + (nb >= row["pool_partial_rho"]).sum())
                                          / (N_OUTER + 1))
        # within-cohort split-half ceilings (descriptive), averaged over balanced splits
        for cohort in ("podcast", "film"):
            if cohort == "podcast":
                us = np.unique(Psub[r])
                vec = {u: P[r][Psub[r] == u].sum(0)[elig] for u in us}
                cnt = {u: float((Psub[r] == u).sum()) for u in us}
            else:
                us = np.array(sorted(a["per_sub"]))
                vec, cnt = {}, {}
                for u in us:
                    fpos, s_sum, nc = a["per_sub"][u]
                    v_ = np.full(nF, np.nan)
                    v_[fpos] = s_sum
                    vec[u], cnt[u] = v_[elig], float(nc)
            half = len(us) // 2
            vals = []
            for A in itertools.combinations(us, half):
                if us[0] not in A:
                    continue                            # each split once
                B = [u for u in us if u not in A]

                def m(G):
                    s = np.nansum([vec[u] for u in G], 0)
                    c = np.sum([np.where(np.isfinite(vec[u]), cnt[u], 0) for u in G], 0)
                    return np.where(c > 0, s / np.maximum(c, 1), np.nan)
                x, y_ = m(A), m(B)
                ok = np.isfinite(x) & np.isfinite(y_)
                vals.append(partial_spearman(x[ok], y_[ok], lp[elig][ok], lf[ok]))
            row[f"splithalf_{cohort}"] = float(np.mean(vals))
            row[f"n_splits_{cohort}"] = len(vals)
        sb = {c: 2 * row[f"splithalf_{c}"] / (1 + row[f"splithalf_{c}"])
              for c in ("podcast", "film")}
        row["ceiling"] = float(np.sqrt(max(sb["podcast"], 0) * max(sb["film"], 0)))
        rows.append(row)
        mats[r] = (elig, impF[0], lf)
        print(f"[{r}] {row['n_features']} features: partial rho {obs_p:+.3f} "
              f"(null 95% {row['partial_null_95']:+.3f}, P {row['p_partial']:.4f}); raw "
              f"{obs_r:+.3f} (P {row['p_raw']:.4f}); split-half podcast "
              f"{row['splithalf_podcast']:+.3f}, film {row['splithalf_film']:+.3f}",
              flush=True)
    T = pd.DataFrame(rows)
    sec = T.roi != "IFG"
    T["p_partial_holm_regions"] = np.nan
    T.loc[sec, "p_partial_holm_regions"] = holm(T.loc[sec, "p_partial"].to_numpy())
    ifg = T[T.roi == "IFG"].iloc[0]
    T["verdict"] = ""
    T.loc[T.roi == "IFG", "verdict"] = (
        "SUPPORTED: the same features matter for IFG in both cohorts"
        if ifg.partial_rho > 0 and ifg.p_partial < 0.05 else
        "consistency carried by activation density" if ifg.p_raw < 0.05 else
        "no evidence of feature-level consistency")
    T.to_csv(out / "a42_consistency.csv", index=False)

    # region specificity: podcast region a against film region b (descriptive)
    M_ = pd.DataFrame(index=[f"podcast {a}" for a in mats], columns=[f"film {b}" for b in mats],
                      dtype=float)
    for a_ in mats:
        for b_ in mats:
            e = mats[a_][0] & mats[b_][0]
            ea = e[mats[b_][0]]
            impP = P[a_].mean(0)[e]
            impF = mats[b_][1][ea]
            M_.loc[f"podcast {a_}", f"film {b_}"] = partial_spearman(
                impP, impF, lp[e], mats[b_][2][ea])
    M_.to_csv(out / "a42_region_matrix.csv")
    json.dump({"published_reproduction_max_abs": d_rep, "fft_vs_direct_max_abs": d_fft},
              open(out / "a42_checks.json", "w"), indent=2)
    print("\n" + T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nregion matrix (partial Spearman)\n" + M_.round(3).to_string())
    print("\nVERDICT (IFG):", T.loc[T.roi == "IFG", "verdict"].iloc[0])


if __name__ == "__main__":
    main()
