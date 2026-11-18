"""Leave-one-participant-out robustness of the headline statistics.

Every headline statistic is recomputed with each participant dropped in turn, from the saved
per-electrode outputs; the only refit is H1's mixed model. Per-fold P values use 2,000
participant-clustered bootstrap draws. A result is robust if its sign, and its significance
decision at 0.05 where it has one, are unchanged in every fold; otherwise the participant whose
removal changes it is named. Writes ``leave_one_out/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from interpretability.causal_vs_neural_importance import partial_spearman
from inference.electrode_inference import load_observed
from inference.statistics import (mixed_model, paired_subject_test, pooled_exceedance_test,
                                  temporal_consistency)

N_BOOT = 2000
TAU = 0.5
EXPLORATORY = ("sub-03", "sub-09")


def boot(diff, subj, seed):
    m, lo, hi, p, b = M.cluster_boot(diff, subj, np.random.default_rng(seed), N_BOOT)
    return m, lo, hi, p, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    R = cfg.paths.results_root
    folds = [None] + list(cfg.subjects)          # None = all subjects
    rows = []

    def add(stat, drop, value, p=np.nan, lo=np.nan, hi=np.nan, decision=None):
        rows.append({"statistic": stat, "dropped": drop or "none", "value": float(value),
                     "p": float(p), "lo": float(lo), "hi": float(hi),
                     "significant": (None if decision is None else bool(decision))})

    # ---- main encoding: IFG effect, timing, H1 --------------------------------------
    obs, meta = load_observed(cfg, "sae")
    lags = cfg.lags.centres_s
    meta = meta.assign(obs_peak=np.nanmax(obs, axis=1))
    lang, ctrl = set(cfg.electrodes["language_rois"]), set(cfg.electrodes["control_rois"])
    for d in folds:
        keep = (meta.subject != d).to_numpy() if d else np.ones(len(meta), bool)
        ifg = keep & (meta.roi == "IFG").to_numpy()
        add("IFG median peak delta_sae", d, np.median(obs[ifg].max(1)))
        tc = temporal_consistency(obs[ifg], lags)
        add("IFG timing agreement", d, tc["ratio"])
        add("IFG peak-of-mean lag (s)", d, tc["peak_of_mean_lag_s"])
        sub = meta[keep & meta.roi.isin(lang | ctrl).to_numpy()].copy()
        sub["network"] = np.where(sub.roi.isin(lang), "language", "control")
        mm = mixed_model(sub, outcome="obs_peak", fixed="network")
        k = "C(network)[T.language]"
        add("H1 language vs control beta (SD)", d, mm["params"][k], mm["pvalues"][k],
            decision=mm["pvalues"][k] < 0.05)

    # ---- matched-capacity analyses ------------------------------------------------------
    E32 = pd.read_csv(R / "classical_matched" / "a32_per_electrode.csv")
    E30 = pd.read_csv(R / "dimensionality_sweep" / "a30_per_electrode.csv")
    for i, d in enumerate(folds):
        e = E32[E32.subject != d] if d else E32
        m, lo, hi, p, _ = boot((e["SAE PCA-85"] - e["classical-85"]).to_numpy(),
                               e.subject.to_numpy(), cfg.seed + 3700 + i)
        add("A32 SAE-85 minus classical-85", d, m, p, lo, hi, p < 0.05)
        add("A32 ratio", d, e["SAE PCA-85"].mean() / e["classical-85"].mean())
        e = E30[E30.subject != d] if d else E30
        s = e["SAE PCA-300 exact"].to_numpy()
        g = e["Gemma resid PCA-300"].to_numpy()
        x = e["GPT-2 XL PCA-300"].to_numpy()
        sj = e.subject.to_numpy()
        m, lo, hi, p, b = boot(s - g, sj, cfg.seed + 3710 + i)
        add("A30 SAE-300 minus Gemma-300", d, m, p, lo, hi, p < 0.05)
        m2, lo2, hi2, p2, _ = boot(s - x, sj, cfg.seed + 3720 + i)
        add("A30 SAE-300 minus GPT-2 XL-300", d, m2, p2, lo2, hi2, p2 < 0.05)
        delta = g.mean() - x.mean()
        lo90, hi90 = np.percentile(b, [5, 95])
        add("A33 margin Delta", d, delta)
        add("A33 90% upper bound (non-superiority if < Delta)", d, hi90,
            decision=hi90 < delta)
        add("A33 90% lower bound (equivalence needs > -Delta)", d, lo90,
            decision=(lo90 > -delta) and (hi90 < delta))

    # ---- within-fold word-type decomposition --------------------------------------------
    D = pd.read_csv(R / "word_types" / "a28_decomposition.csv")
    D = D[(D.min_occ == 4) & (D.type_means == "within_fold")]
    piv = D.pivot_table(index=["subject", "ch"], columns="representation",
                        values=["between", "within"])
    sj_all = piv.index.get_level_values("subject").to_numpy()
    for tag, a, bb in [("A28 SAE band vs GPT-2 XL", "SAE", "prior"),
                       ("A34 SAE-300 vs Gemma dense-300", "sae300", "gemma")]:
        if ("within", a) not in piv.columns or ("within", bb) not in piv.columns:
            continue
        dw = (piv[("within", a)] - piv[("within", bb)]).to_numpy()
        db = (piv[("between", a)] - piv[("between", bb)]).to_numpy()
        for i, d in enumerate(folds):
            k = sj_all != d if d else np.ones(len(sj_all), bool)
            m, lo, hi, p, _ = boot(dw[k] - db[k], sj_all[k], cfg.seed + 3730 + i)
            add(f"{tag}: interaction (within-fold)", d, m, p, lo, hi, p < 0.05)
            m, lo, hi, p, _ = boot(dw[k], sj_all[k], cfg.seed + 3740 + i)
            add(f"{tag}: d_within (within-fold)", d, m, p, lo, hi, p < 0.05)
            m, lo, hi, p, _ = boot(db[k], sj_all[k], cfg.seed + 3750 + i)
            add(f"{tag}: d_between", d, m, p, lo, hi, p < 0.05)

    # ---- cross-participant agreement: pairs not involving the dropped participant ------
    pairs = pd.read_csv(R / "within_type_signal" / "a29a_pairs.csv")
    _, _, subj_ifg, _ = M.ifg_targets(cfg, "window")
    order = list(dict.fromkeys(subj_ifg))
    pa = pairs.pair.str.split("-", expand=True).astype(int)
    pairs["s1"] = [order[i] for i in pa[0]]
    pairs["s2"] = [order[i] for i in pa[1]]
    for d in folds:
        q = pairs[(pairs.s1 != d) & (pairs.s2 != d)] if d else pairs
        add("A29a within-type ISC (mean over pairs)", d, q.within.mean())
        add("A29a between-type ISC (mean over pairs)", d, q.between.mean())
        add("A29a pairs with positive within-type ISC (fraction)", d,
            (q.within > 0).mean())

    # ---- H2 pooled co-primary and paired test -----------------------------------------
    sh = pd.read_csv(R / "concentration" / "sci_shares_k20.csv")
    sh = sh[sh.roi.isin(lang) & ~sh.subject.isin(EXPLORATORY)]
    for d in [None] + sorted(sh.subject.unique()):
        q = sh[sh.subject != d] if d else sh
        pool = pooled_exceedance_test(q.share_real.to_numpy(), q.share_random_dict.to_numpy(),
                                      q.subject.to_numpy(), tau=TAU, seed=cfg.seed)
        add("H2 pooled difference", d, pool["diff"], lo=pool.get("diff_ci_lo", np.nan),
            hi=pool.get("diff_ci_hi", np.nan),
            decision=pool.get("diff_ci_lo", -1) > 0)
        per = q.groupby("subject").apply(lambda g: pd.Series({
            "real": (g.share_real > TAU).mean(), "ctrl": (g.share_random_dict > TAU).mean()}))
        pt = paired_subject_test(per.real.to_numpy(), per.ctrl.to_numpy())
        add("H2 paired sign test (one-sided p)", d, pt["n_real_higher"], p=pt["p_sign"],
            decision=pt["p_sign"] < 0.05)

    # ---- neural importance re-averaged without the participant -----------------------
    pe = np.load(R / "feature_table" / "feature_table_ifg_per_electrode.npz",
                 allow_pickle=True)
    T = pd.read_csv(R / "causal" / "a11_feature_table_ifg.csv").set_index("feature")
    fid = pe["feature_id"]
    T = T.loc[fid]
    for d in folds:
        k = pe["subject"] != d if d else np.ones(len(pe["subject"]), bool)
        imp = pe["importance"][k].mean(0)
        r, p, _ = partial_spearman(T.causal_kl.to_numpy(), imp, T.log_density.to_numpy())
        add("A11 partial rho (KL vs importance | density)", d, r, p, decision=p < 0.05)

    L = pd.DataFrame(rows)
    out = cfg.results("leave_one_out")
    out.mkdir(parents=True, exist_ok=True)
    L.to_csv(out / "a37_loso_long.csv", index=False)

    summ = []
    for stat, g in L.groupby("statistic", sort=False):
        full = g[g.dropped == "none"].iloc[0]
        f = g[g.dropped != "none"]
        sign_ok = bool((np.sign(f.value) == np.sign(full.value)).all())
        dec = g.significant.dropna()
        dec_ok = bool((f.significant == full.significant).all()) if len(dec) else None
        flips = f[(np.sign(f.value) != np.sign(full.value))
                  | ((f.significant != full.significant) if len(dec) else False)]
        infl = f.iloc[int(np.argmax(np.abs(f.value - full.value)))]
        summ.append({"statistic": stat, "all_subjects": full.value,
                     "all_p": full.p, "loso_min": f.value.min(), "loso_max": f.value.max(),
                     "max_p": f.p.max(), "sign_stable": sign_ok,
                     "decision_stable": dec_ok,
                     "robust": sign_ok and (dec_ok is not False),
                     "changes_when_dropping": ", ".join(flips.dropped) or "-",
                     "most_influential": infl.dropped})
    S = pd.DataFrame(summ)
    S.to_csv(out / "a37_loso_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("written:", out / "a37_loso_summary.csv")


if __name__ == "__main__":
    main()
