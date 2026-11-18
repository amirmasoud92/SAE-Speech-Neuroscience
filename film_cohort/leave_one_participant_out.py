"""Leave-one-participant-out robustness of the film-cohort and cross-cohort results.

A result is robust if its decision holds with every single participant removed; otherwise the
participant that changes it is named. Recomputed with one participant removed:

- film H2: both co-primary tests, IFG and language scopes;
- reconstruction-error contrasts: both cohorts, IFG;
- sparsity dose-response: the width-16,384 slopes, both cohorts, IFG;
- cross-cohort feature consistency: IFG partial Spearman against its outer-shift null, with the
  entry rule re-applied after each removal;
- film H1: the network coefficient, descriptive.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from film_cohort.feature_consistency import N_OUTER, partial_spearman
from inference.statistics import mixed_model

L16 = [22, 41, 82, 176, 445]
LANG, CTRL = ("IFG", "STG", "MTG", "TMP"), ("EAC", "PRC", "MFG")


def boot(x, subj, seed):
    ok = np.isfinite(x)
    m, lo, hi, p, _ = M.cluster_boot(x[ok], subj[ok], np.random.default_rng(seed), 10000)
    return m, lo, hi, p


def a39_decision(A, subj, seed, keys):
    a = boot(A[keys["E_r"]], subj, seed + 1)
    b = boot(A[keys["R_w"]] - A[keys["D_w"]], subj, seed + 2)
    c = boot(A[keys["E_w"]], subj, seed + 3)
    if a[1] > 0 and b[2] < 0 and c[1] > 0:
        return "SUPPORTED", (a[0], b[0], c[0])
    if b[2] < 0 and not c[1] > 0:
        return "PARTIAL", (a[0], b[0], c[0])
    return "NOT SUPPORTED", (a[0], b[0], c[0])


def a40_decisions(arr_by_l0, D, subj, seed):
    x = np.log2(np.array(L16, float))
    xc = x - x.mean()
    out = {}
    for k, (comp, pred) in enumerate((("r", 1), ("within", 1), ("between", -1))):
        arr = np.stack([arr_by_l0[l][comp] - D[comp] for l in L16])
        slope = (xc[:, None] * (arr - np.nanmean(arr, 0))).sum(0) / (xc ** 2).sum()
        m, lo, hi, p = boot(slope, subj, seed + k)
        out[comp] = ("supported" if (lo > 0 if pred > 0 else hi < 0) else
                     "contradicted" if (hi < 0 if pred > 0 else lo > 0) else "unresolved",
                     m)
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    rows = []

    # ---- H2 (film) ----------------------------------------------------------------
    from interpretability.semantic_concentration import _report
    out41 = cfg.results("film_h1_h2")
    allc = pd.read_csv(out41 / "sci_film_all_contacts_k20.csv")
    adm = allc[allc.admissible].drop(columns="admissible")
    lg = logging.getLogger("a44")
    for drop in [None] + sorted(adm.subject.unique()):
        d = adm if drop is None else adm[adm.subject != drop]
        for scope, sel in (("IFG", d.roi == "IFG"), ("language", d.roi.isin(LANG))):
            r = _report(lg, d[sel], 0.5, cfg.seed)
            s_ok, p_ok = r.get("p_sign_one_sided", 1) < 0.05, r.get("diff_ci_lo", -1) > 0
            v = "SUPPORTED" if s_ok and p_ok else "NOT SUPPORTED" if not (s_ok or p_ok) \
                else "MIXED"
            rows.append({"analysis": f"A41 H2 film, {scope}", "dropped": drop or "(none)",
                         "decision": v, "effect": r.get("diff", np.nan),
                         "p": r.get("p_sign_one_sided", np.nan)})

    # ---- H1 (film), descriptive -----------------------------------------------------
    pk = pd.read_csv(out41 / "h1_film_peaks.csv")
    for drop in [None] + sorted(pk.subject.unique()):
        d = pk if drop is None else pk[pk.subject != drop]
        d = d[d.roi.isin(LANG + CTRL)].copy()
        d["network"] = np.where(d.roi.isin(LANG), "language", "control")
        mm = mixed_model(d, outcome="obs_peak", fixed="network")
        b, p = mm["params"]["C(network)[T.language]"], mm["pvalues"]["C(network)[T.language]"]
        rows.append({"analysis": "A41 H1 film (descriptive)", "dropped": drop or "(none)",
                     "decision": "SUPPORTED" if (b > 0 and p < 0.05) else "NOT SUPPORTED",
                     "effect": b, "p": p})

    # ---- reconstruction error and sparsity dose-response ---------------------------
    m36 = cfg.results("reconstruction_and_sparsity")
    A = np.load(m36 / "a39_a40_arms.npz", allow_pickle=True)
    psubj = np.load(m36 / "_shared.npz", allow_pickle=True)["subj"].astype(str)
    pod = {"D": {c: A[f"D__{c}"] for c in ("r", "within", "between")}}
    pod.update({l: {c: A[f"w16k_l0-{l}__{c}"] for c in ("r", "within", "between")}
                for l in L16})
    pod_a39 = {"E_r": A["E__r"], "R_w": A["R__within"], "D_w": A["D__within"],
               "E_w": A["E__within"]}
    F = np.load(m36 / "btb" / "a39_a40_contacts_IFG.npz", allow_pickle=True)
    fsubj = F["subj"].astype(str)
    film = {"D": {c: F[f"D__{c}"] for c in ("r", "within", "between")}}
    film.update({l: {c: F[f"S{l}__{c}"] for c in ("r", "within", "between")} for l in L16})
    film_a39 = {"E_r": F["E__r"], "R_w": F["R__within"], "D_w": F["D__within"],
                "E_w": F["E__within"]}
    for coh, subj, a39, a40 in (("podcast", psubj, pod_a39, pod),
                                ("film", fsubj, film_a39, film)):
        for drop in [None] + sorted(np.unique(subj)):
            keep = np.ones(len(subj), bool) if drop is None else subj != drop
            v, eff = a39_decision({k: x[keep] for k, x in a39.items()}, subj[keep],
                                  cfg.seed + 440, {k: k for k in a39})
            rows.append({"analysis": f"A39 {coh} IFG", "dropped": drop or "(none)",
                         "decision": v, "effect": eff[1], "p": np.nan})
            dec = a40_decisions({l: {c: a40[l][c][keep] for c in a40[l]} for l in L16},
                                {c: a40["D"][c][keep] for c in a40["D"]}, subj[keep],
                                cfg.seed + 445)
            for comp, (v, m) in dec.items():
                rows.append({"analysis": f"A40 {coh} IFG 16k slope ({comp})",
                             "dropped": drop or "(none)", "decision": v, "effect": m,
                             "p": np.nan})

    # ---- cross-cohort feature consistency (from the export) ------------------------
    E = np.load(cfg.results("feature_consistency", "a42_ifg_export.npz"),
                allow_pickle=True)
    pids, dens_p = E["pids"], E["dens_p"]
    pod_obs, pod_subj = E["pod_obs"], E["pod_subj"].astype(str)
    fsubs = [str(s) for s in E["film_subs"]]
    nF = len(pids)

    def a42_stat(drop_pod=None, drop_film=None):
        po = pod_obs[pod_subj != drop_pod] if drop_pod else pod_obs
        sums, cnt, act = np.zeros((N_OUTER + 1, nF)), np.zeros(nF), np.zeros(nF)
        words, n_c = 0.0, 0
        for s in fsubs:
            if s == drop_film:
                continue
            fp = E[f"film_fpos_{s}"]
            sums[:, fp] += E[f"film_sum_{s}"]
            cnt[fp] += int(E[f"film_n_{s}"])
            act[fp] += E[f"film_act_{s}"]
            words += float(E[f"film_words_{s}"])
            n_c += int(E[f"film_n_{s}"])
        elig = (cnt >= 0.5 * n_c) & (dens_p >= 25)
        impF = sums[:, elig] / cnt[elig]
        lf = np.log(np.maximum(act[elig], 0.5) / words)
        lp = np.log(dens_p)[elig]
        ip = po.mean(0)[elig]
        obs = partial_spearman(ip, impF[0], lp, lf)
        null = np.array([partial_spearman(ip, impF[b], lp, lf) for b in range(1, N_OUTER + 1)])
        return obs, float((1 + (null >= obs).sum()) / (N_OUTER + 1)), int(elig.sum())

    for who, drops in (("podcast", [None] + sorted(np.unique(pod_subj))),
                       ("film", sorted(fsubs))):
        for drop in drops:
            obs, p, nf = a42_stat(drop_pod=drop if who == "podcast" else None,
                                  drop_film=drop if who == "film" else None)
            rows.append({"analysis": "A42 IFG partial rho",
                         "dropped": f"{who}:{drop}" if drop else "(none)",
                         "decision": "SUPPORTED" if (obs > 0 and p < 0.05) else
                         "NOT SUPPORTED", "effect": obs, "p": p, "n_features": nf})

    T = pd.DataFrame(rows)
    out = cfg.results("film_leave_one_out")
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "a44_loso_all.csv", index=False)
    summ = []
    for an, g in T.groupby("analysis", sort=False):
        ref = g[g.dropped == "(none)"].decision.iloc[0]
        changed = g[(g.dropped != "(none)") & (g.decision != ref)]
        summ.append({"analysis": an, "full_sample": ref,
                     "n_leave_outs": int((g.dropped != "(none)").sum()),
                     "robust": changed.empty,
                     "changes_when_dropping": ", ".join(f"{r.dropped} -> {r.decision}"
                                                        for r in changed.itertuples())})
    S = pd.DataFrame(summ)
    S.to_csv(out / "a44_loso_summary.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
