"""Fixed-window mixed models, direct contrasts and leave-one-participant-out robustness.

CPU only, from stored outputs; every reused quantity must first reproduce its stored value.

* Mixed-effects models of unique sparse variance with a per-participant random intercept, on
  the peak statistic and on the mean over a fixed 0-500 ms window: language against control
  networks, and each region against early auditory cortex.
* Direct contrasts: Wald contrasts between mixed-model coefficients, within-participant
  bootstrap differences between regions, cohorts and scores.
* Leave-one-participant-out: every statistic recomputed with each participant left out.

Writes ``controls/{leave_one_out,fixed_window,contrasts}/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent
N_BOOT = 10000
IFG_DOMINANT = ("sub-03", "sub-09")


# ------------------------------------------------------------------ mixed model + Wald

def mm_fit(df, outcome, fixed):
    """``stats.mixed_model`` exactly (z-scored outcome, REML, optimiser order), returning
    the statsmodels fit so the fixed-effect covariance is available."""
    import statsmodels.formula.api as smf
    d = df[[outcome, fixed, "subject"]].dropna().copy()
    d["_z"] = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=1)
    model = smf.mixedlm(f"_z ~ C({fixed})", d, groups=d["subject"])
    fit = None
    for method in (None, "powell", "cg"):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                cand = model.fit(reml=True) if method is None else model.fit(reml=True,
                                                                              method=method)
            if bool(cand.converged):
                return cand
            fit = fit or cand
        except Exception:  # noqa: BLE001
            continue
    return fit


def wald(fit, plus, minus=None):
    names = list(fit.fe_params.index)
    c = np.zeros(len(names))
    c[names.index(plus)] = 1.0
    if minus:
        c[names.index(minus)] = -1.0
    cov = fit.cov_params().loc[names, names].to_numpy()
    est = float(c @ fit.fe_params.to_numpy())
    se = float(np.sqrt(c @ cov @ c))
    z = est / se
    return dict(estimate=est, se=se, ci95_lo=est - 1.96 * se, ci95_hi=est + 1.96 * se,
                z=z, p_two_sided=float(2 * sps.norm.sf(abs(z))),
                test="Wald z on the mixed-model fixed effects, two-sided")


def electrode_table(cfg):
    from inference.electrode_inference import load_observed
    obs, meta = load_observed(cfg, "sae")
    lags = cfg.lags.centres_s
    win = (lags >= -1e-9) & (lags <= 0.5 + 1e-9)
    meta = meta.assign(obs_peak=np.nanmax(obs, axis=1),
                       obs_win=np.nanmean(obs[:, win], axis=1)).reset_index(drop=True)
    lang, ctrl = set(cfg.electrodes["language_rois"]), set(cfg.electrodes["control_rois"])
    meta["network"] = np.where(meta.roi.isin(lang), "language",
                               np.where(meta.roi.isin(ctrl), "control", "other"))
    return obs, meta, int(win.sum())


# ------------------------------------------------------------- fixed-window mixed models

def a8_window(cfg, out):
    obs, meta, n_win = electrode_table(cfg)
    ifg = meta.roi == "IFG"
    assert abs(np.median(meta.obs_peak[ifg]) - 0.011484) < 5e-6
    rows = []
    rng = np.random.default_rng(cfg.seed + 801)
    for roi, g in meta.groupby("roi"):
        for stat in ("obs_peak", "obs_win"):
            m, lo, hi, p, _ = M.cluster_boot(g[stat].to_numpy(), g.subject.to_numpy(), rng,
                                             N_BOOT)
            rows.append({"roi": roi, "statistic": "peak over lags" if stat == "obs_peak"
                         else "mean 0-500 ms", "median": m, "ci95_lo": lo, "ci95_hi": hi,
                         "p_two_sided_vs_0": p, "n_electrodes": len(g),
                         "n_participants": g.subject.nunique()})
    R = pd.DataFrame(rows)
    R.to_csv(out / "a8_roi_medians.csv", index=False)
    models = []
    sub = meta[meta.network != "other"]
    for stat in ("obs_peak", "obs_win"):
        f = mm_fit(sub, stat, "network")
        w = wald(f, "C(network)[T.language]")
        models.append({"model": "language vs control (network)", "statistic": stat,
                       "contrast": "language - control", **w, "n_electrodes": int(f.nobs),
                       "n_participants": sub.subject.nunique()})
        g = mm_fit(meta, stat, "roi")
        for roi in ("IFG", "STG", "MTG", "TMP", "MFG", "PRC"):
            w = wald(g, f"C(roi)[T.{roi}]")
            models.append({"model": "ROI (reference EAC)", "statistic": stat,
                           "contrast": f"{roi} - EAC", **w, "n_electrodes": int(g.nobs),
                           "n_participants": meta.subject.nunique()})
    T = pd.DataFrame(models)
    pk = T[(T.statistic == "obs_peak")].set_index("contrast").estimate
    assert abs(pk["language - control"] - 0.5328) < 5e-4 and abs(pk["IFG - EAC"] - 1.0527) < 5e-4
    T.to_csv(out / "a8_mixed_models.csv", index=False)
    json.dump({"n_window_lags": n_win, "window": "0 to 500 ms inclusive"},
              open(out / "a8_window_definition.json", "w"))
    print(R.pivot(index="roi", columns="statistic", values="median").to_string())
    print(T[["model", "statistic", "contrast", "estimate", "ci95_lo", "ci95_hi",
             "p_two_sided"]].to_string(index=False))
    return meta


# --------------------------------------------------------------------- direct contrasts

def a9(cfg, out, meta):
    rows = []
    # (1) IFG - MFG (and STG - MFG) in the ROI mixed model, peak and window
    for stat in ("obs_peak", "obs_win"):
        g = mm_fit(meta, stat, "roi")
        for roi in ("IFG", "STG"):
            rows.append({"id": "A9.1", "inference": f"{roi} carries the effect, MFG does not",
                         "contrast": f"{roi} beta - MFG beta ({'peak' if stat == 'obs_peak' else 'window 0-500 ms'})",
                         **wald(g, f"C(roi)[T.{roi}]", "C(roi)[T.MFG]"),
                         "n_electrodes": int(g.nobs), "n_participants": meta.subject.nunique()})

    # (2) H2 in IFG vs the rest of the language network
    # the H2 set: admissible language-ROI electrodes of the confirmatory participants
    # (sub-03 and sub-09 are exploratory), tau = 0.5
    S = pd.read_csv(cfg.results("concentration", "sci_shares_k20.csv"))
    lang = set(cfg.electrodes["language_rois"])
    S = S[S.roi.isin(lang) & ~S.subject.isin(IFG_DOMINANT)].copy()
    S["d"] = ((S.share_real > 0.5).astype(float)
              - (S.share_random_dict > 0.5).astype(float))
    assert len(S) == 92 and abs(S.d.mean() - 0.336957) < 1e-5
    assert abs(S[S.roi == "IFG"].d.mean() - 0.384615) < 1e-5
    isifg = (S.roi == "IFG").to_numpy()
    d, subj = S.d.to_numpy(), S.subject.to_numpy()
    obs = d[isifg].mean() - d[~isifg].mean()
    rng = np.random.default_rng(cfg.seed + 902)
    uq = np.unique(subj)
    by = {u: np.flatnonzero(subj == u) for u in uq}
    bs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[u] for u in rng.choice(uq, len(uq), True)])
        a, b = isifg[idx], ~isifg[idx]
        if a.any() and b.any():
            bs.append(d[idx][a].mean() - d[idx][b].mean())
    bs = np.array(bs)
    perm = np.empty(N_BOOT)
    for i in range(N_BOOT):
        lab = isifg.copy()
        for u in uq:
            j = by[u]
            lab[j] = rng.permutation(isifg[j])
        perm[i] = d[lab].mean() - d[~lab].mean()
    rows.append({"id": "A9.2", "inference": "H2 supported at network scope, unresolved in IFG",
                 "contrast": "H2 effect (learned - random exceedance) in IFG - in non-IFG "
                             "language electrodes",
                 "estimate": obs, "ci95_lo": float(np.percentile(bs, 2.5)),
                 "ci95_hi": float(np.percentile(bs, 97.5)),
                 "p_two_sided": float((np.abs(perm) >= abs(obs) - 1e-12).mean()),
                 "test": "within-participant permutation of the IFG label (10,000), two-sided; "
                         "CI participant-clustered bootstrap",
                 "n_electrodes": len(S), "n_participants": len(uq),
                 "detail": f"IFG {d[isifg].mean():+.4f} (n={isifg.sum()}), non-IFG "
                           f"{d[~isifg].mean():+.4f} (n={(~isifg).sum()})"})

    # (3) coherence predicts causal necessity but not cortical importance
    F = pd.read_csv(cfg.results("causal", "a11_feature_table_ifg.csv"))   # importance over all IFG electrodes
    rk = lambda x: sps.rankdata(x)  # noqa: E731

    def rho(a, b):
        return float(np.corrcoef(rk(a), rk(b))[0, 1])
    c, kl, imp = F.coherence.to_numpy(), F.causal_kl.to_numpy(), F.importance.to_numpy()
    ok = np.isfinite(c) & np.isfinite(kl) & np.isfinite(imp)
    c, kl, imp = c[ok], kl[ok], imp[ok]
    r1, r2 = rho(c, kl), rho(c, imp)
    assert abs(r1 - 0.271054) < 1e-4 and abs(r2 - (-0.066)) < 5e-4, (r1, r2)
    bd = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, len(c), len(c))
        bd[i] = rho(c[j], kl[j]) - rho(c[j], imp[j])
    rows.append({"id": "A9.3", "inference": "coherence predicts what the model needs but not "
                 "what cortex weights", "contrast": "rho(coherence, KL) - rho(coherence, "
                 "IFG importance)", "estimate": r1 - r2,
                 "ci95_lo": float(np.percentile(bd, 2.5)), "ci95_hi": float(np.percentile(bd, 97.5)),
                 "p_two_sided": float(2 * min((bd <= 0).mean(), (bd >= 0).mean())),
                 "test": "bootstrap over features (10,000), two-sided",
                 "n_features": int(ok.sum()), "detail": f"{r1:+.3f} vs {r2:+.3f}"})

    # (4, 5) podcast vs film: lexical component and reconstruction step
    zp = np.load(cfg.results("reconstruction_and_sparsity", "a39_a40_arms.npz"), allow_pickle=True)
    sp = np.load(cfg.results("reconstruction_and_sparsity", "_shared.npz"), allow_pickle=True)["subj"]
    zf = np.load(cfg.results("reconstruction_and_sparsity", "btb", "a39_a40_contacts_IFG.npz"),
                 allow_pickle=True)
    sf = zf["subj"]
    pairs = {
        "A9.4": ("lexical (between-type) S - D resolved in film, not podcast",
                 zp["w16k_l0-82__between"] - zp["D__between"],
                 zf["S82__between"] - zf["D__between"]),
        "A9.5": ("reconstruction step within(R) - within(D) resolved in podcast, not film IFG",
                 zp["R__within"] - zp["D__within"], zf["R__within"] - zf["D__within"]),
    }
    for k, (lab, dp, df_) in pairs.items():
        stp, bp = PS.electrode_stat(dp, sp, cfg.seed + 950)
        stf, bf = PS.electrode_stat(df_, sf, cfg.seed + 951)
        diff = bf - bp
        rows.append({"id": k, "inference": lab, "contrast": "film IFG - podcast IFG",
                     "estimate": stf["estimate"] - stp["estimate"],
                     "ci95_lo": float(np.percentile(diff, 2.5)),
                     "ci95_hi": float(np.percentile(diff, 97.5)),
                     "p_two_sided": float(2 * min((diff <= 0).mean(), (diff >= 0).mean())),
                     "test": "difference of independent participant-clustered bootstraps "
                             "(10,000 each), two-sided",
                     "n_electrodes": f"{stp['n_electrodes']} podcast + {stf['n_electrodes']} film",
                     "n_participants": f"{stp['n_participants']} + {stf['n_participants']}",
                     "detail": f"podcast {stp['estimate']:+.4f} [{stp['ci95_lo']:+.4f}, "
                               f"{stp['ci95_hi']:+.4f}]; film {stf['estimate']:+.4f} "
                               f"[{stf['ci95_lo']:+.4f}, {stf['ci95_hi']:+.4f}]"})

    # (6) reproducible (cross-cohort) importance vs podcast-only importance, vs coherence
    from film_cohort.feature_consistency import partial_spearman
    A = pd.read_csv(cfg.results("feature_consistency", "a45_features.csv"))
    rn = lambda x: sps.rankdata(x) / len(x)  # noqa: E731
    cross = 0.5 * (rn(A.importance_podcast) + rn(A.importance_film))
    okf = np.isfinite(A.coherence_podcast).to_numpy()
    coh = A.coherence_podcast.to_numpy()[okf]
    cr, ip = cross[okf], A.importance_podcast.to_numpy()[okf]
    lp, lf = A.log_density_podcast.to_numpy()[okf], A.log_density_film.to_numpy()[okf]
    q1, q2 = partial_spearman(coh, cr, lp, lf), partial_spearman(coh, ip, lp, lf)
    assert abs(q1 - 0.062533) < 1e-4 and abs(q2 - 0.012935) < 1e-4, (q1, q2)
    bq = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, len(coh), len(coh))
        bq[i] = (partial_spearman(coh[j], cr[j], lp[j], lf[j])
                 - partial_spearman(coh[j], ip[j], lp[j], lf[j]))
    rows.append({"id": "A9.6", "inference": "coherence relates to reproducible importance, "
                 "not to podcast-only importance",
                 "contrast": "partial rho(coherence, cross-cohort importance) - partial "
                             "rho(coherence, podcast importance)", "estimate": q1 - q2,
                 "ci95_lo": float(np.percentile(bq, 2.5)), "ci95_hi": float(np.percentile(bq, 97.5)),
                 "p_two_sided": float(2 * min((bq <= 0).mean(), (bq >= 0).mean())),
                 "test": "bootstrap over features (10,000), two-sided",
                 "n_features": int(okf.sum()), "detail": f"{q1:+.4f} vs {q2:+.4f}"})

    # (7) H1: podcast (supported) vs film (not replicated)
    hp = json.load(open(cfg.results("group_models", "h1_group_models_sae.json")))
    hf = json.load(open(cfg.results("film_h1_h2", "h1_film_models.json")))
    k = "C(network)[T.language]"
    netf = hf.get("mixed_model_network", hf)
    bp_, sep = hp["mixed_model_network"]["params"][k], hp["mixed_model_network"]["bse"][k]
    bf_, sef = netf["params"][k], netf["bse"][k]
    z = (bp_ - bf_) / np.hypot(sep, sef)
    rows.append({"id": "A9.7", "inference": "H1 supported in podcast, not replicated in film",
                 "contrast": "language - control beta (SD): podcast - film",
                 "estimate": bp_ - bf_, "se": float(np.hypot(sep, sef)),
                 "ci95_lo": bp_ - bf_ - 1.96 * np.hypot(sep, sef),
                 "ci95_hi": bp_ - bf_ + 1.96 * np.hypot(sep, sef), "z": z,
                 "p_two_sided": float(2 * sps.norm.sf(abs(z))),
                 "test": "z test on two independent mixed-model coefficients (each in its "
                         "cohort's outcome SD), two-sided",
                 "detail": f"podcast {bp_:+.3f} (SE {sep:.3f}); film {bf_:+.3f} (SE {sef:.3f})"})
    T = pd.DataFrame(rows)
    T.to_csv(out / "a9_contrasts.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(T[["id", "contrast", "estimate", "ci95_lo", "ci95_hi", "p_two_sided"]]
              .to_string(index=False))


# ------------------------------------------------------- leave-one-participant-out

def a7(cfg, out, meta):
    S = pd.read_csv(cfg.results("leave_one_out", "a37_loso_summary.csv"))
    L = pd.read_csv(cfg.results("leave_one_out", "a37_loso_long.csv"))
    keep = ["H1 language vs control beta (SD)", "A32 SAE-85 minus classical-85", "A32 ratio",
            "A30 SAE-300 minus Gemma-300", "A33 90% upper bound (non-superiority if < Delta)",
            "A33 90% lower bound (equivalence needs > -Delta)", "H2 pooled difference",
            "H2 paired sign test (one-sided p)", "A11 partial rho (KL vs importance | density)",
            "A34 SAE-300 vs Gemma dense-300: d_within (within-fold)"]
    rows = []
    for st in keep:
        s = S[S.statistic == st].iloc[0]
        lg = L[L.statistic == st].set_index("dropped")
        rows.append({"statistic": st, "source": "leave-one-participant-out", "all": s.all_subjects,
                     "p_all": s.all_p, "loso_min": s.loso_min, "loso_max": s.loso_max,
                     "max_p": s.max_p, "sign_stable": s.sign_stable,
                     "decision_stable": s.decision_stable,
                     "changes_when_dropping": s.changes_when_dropping,
                     **{f"drop_{u}": lg.value.get(u, np.nan) for u in IFG_DOMINANT},
                     **{f"p_drop_{u}": lg.p.get(u, np.nan) for u in IFG_DOMINANT}})
    # new: IFG beta and IFG - MFG (ROI model), window H1, refitted per dropped participant
    for drop in [None] + list(cfg.subjects):
        m = meta if drop is None else meta[meta.subject != drop]
        for stat in ("obs_peak", "obs_win"):
            g = mm_fit(m, stat, "roi")
            for lab, (a, b) in {"IFG beta (ROI model, vs EAC)": ("C(roi)[T.IFG]", None),
                                "IFG beta - MFG beta": ("C(roi)[T.IFG]", "C(roi)[T.MFG]")}.items():
                w = wald(g, a, b)
                rows.append({"_stat": f"{lab} [{stat}]", "_drop": drop or "none",
                             "value": w["estimate"], "p": w["p_two_sided"]})
            sub = m[m.network != "other"]
            f = mm_fit(sub, stat, "network")
            w = wald(f, "C(network)[T.language]")
            rows.append({"_stat": f"H1 language vs control beta [{stat}]",
                         "_drop": drop or "none", "value": w["estimate"],
                         "p": w["p_two_sided"]})
    # the window contrast (training-only), 2,000 resamples per fold
    f1 = cfg.results("controls", "within_type_window", "a1_window_per_electrode.csv")
    if f1.exists():
        E = pd.read_csv(f1)
        for comp in ("within", "total"):
            d_all = (E[f"SAE|{comp}"] - E[f"Gemma dense|{comp}"]).to_numpy()
            for i, drop in enumerate([None] + list(cfg.subjects)):
                k = (E.subject != drop).to_numpy() if drop else np.ones(len(E), bool)
                if drop and k.all():
                    continue
                m, lo, hi, pp, _ = M.cluster_boot(d_all[k], E.subject.to_numpy()[k],
                                                  np.random.default_rng(cfg.seed + 7100 + i),
                                                  2000)
                rows.append({"_stat": f"A1 SAE-300 - Gemma dense-300, {comp} (TO, window)",
                             "_drop": drop or "none", "value": m, "p": pp})
    R = pd.DataFrame(rows)
    old = R[R.statistic.notna()].drop(columns=[c for c in R if c.startswith("_")] +
                                      ["value", "p"], errors="ignore")
    new = R[R._stat.notna()] if "_stat" in R else pd.DataFrame()
    summ = []
    for st, g in new.groupby("_stat", sort=False):
        a = g[g._drop == "none"].iloc[0]
        lo = g[g._drop != "none"]
        dec = (lo.p < 0.05) == (a.p < 0.05)
        sign = np.sign(lo.value) == np.sign(a.value)
        summ.append({"statistic": st, "source": "training-only window", "all": a.value, "p_all": a.p,
                     "loso_min": lo.value.min(), "loso_max": lo.value.max(),
                     "max_p": lo.p.max(), "sign_stable": bool(sign.all()),
                     "decision_stable": bool(dec.all()),
                     "changes_when_dropping": ",".join(lo._drop[~(dec & sign)]) or "-",
                     **{f"drop_{u}": float(lo[lo._drop == u].value.iloc[0]) for u in IFG_DOMINANT},
                     **{f"p_drop_{u}": float(lo[lo._drop == u].p.iloc[0]) for u in IFG_DOMINANT}})
    T = pd.concat([old, pd.DataFrame(summ)], ignore_index=True)
    T.to_csv(out / "a7_loso.csv", index=False)
    new.rename(columns={"_stat": "statistic", "_drop": "dropped"})[
        ["statistic", "dropped", "value", "p"]].to_csv(out / "a7_loso_new_long.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(T[["statistic", "all", "loso_min", "loso_max", "sign_stable", "decision_stable",
                 "changes_when_dropping", "drop_sub-03", "drop_sub-09"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["fixed_window", "contrasts", "leave_one_out"])
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    outs = {k: cfg.results("controls", k) for k in ("leave_one_out", "fixed_window", "contrasts")}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    meta = (a8_window(cfg, outs["fixed_window"]) if "fixed_window" in args.parts
            else electrode_table(cfg)[1])
    if "contrasts" in args.parts:
        a9(cfg, outs["contrasts"], meta)
    if "leave_one_out" in args.parts:
        a7(cfg, outs["leave_one_out"], meta)


if __name__ == "__main__":
    main()
