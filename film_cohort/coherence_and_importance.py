"""Are the features cortex weights reproducibly the more interpretable ones?

- **Features:** the IFG feature set of the cross-cohort consistency analysis.
- **Coherence:** ``interpretability.coherence.coherence`` on podcast words (and, as a
  sensitivity, on the films' words).
- **Cross-cohort importance:** the mean of the two cohorts' rank-normalised IFG importances.
- **Statistic:** the partial Spearman correlation of coherence with cross-cohort importance,
  controlling log density in both cohorts.
- **Null:** null shift b of the podcast is paired with null shift b of the film cohort, giving
  200 null cross-cohort importances; two-sided P = (1 + #|null| >= |observed|) / 201.

Writes ``feature_consistency/``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import load_config
from interpretability.coherence import coherence
from film_cohort.feature_consistency import N_OUTER, partial_spearman
from encoding.fit_encoding_model import analysed_words


def ranknorm(x):
    return sps.rankdata(x) / len(x)


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("feature_consistency")
    E = np.load(out / "a42_ifg_export.npz", allow_pickle=True)
    pids, dens_p = E["pids"], E["dens_p"]
    nF = len(pids)
    fsubs = [str(s) for s in E["film_subs"]]
    sums, cnt, act, words, n_c = np.zeros((N_OUTER + 1, nF)), np.zeros(nF), np.zeros(nF), 0.0, 0
    for s in fsubs:
        fp = E[f"film_fpos_{s}"]
        sums[:, fp] += E[f"film_sum_{s}"]
        cnt[fp] += int(E[f"film_n_{s}"])
        act[fp] += E[f"film_act_{s}"]
        words += float(E[f"film_words_{s}"])
        n_c += int(E[f"film_n_{s}"])
    elig = E["elig"]
    impF = sums[:, elig] / cnt[elig]                       # (201, F)
    impP = E["pod_null"][:, elig]                          # (201, F); row 0 observed
    lp = np.log(dens_p[elig])
    lf = np.log(np.maximum(act[elig], 0.5) / words)
    fid = pids[elig]
    print(f"{int(elig.sum())} A42 features; podcast observed check "
          f"{np.abs(impP[0] - E['pod_obs'].mean(0)[elig]).max():.1e}")

    # ---- coherence: podcast words (primary) ----------------------------------------
    w = analysed_words(cfg).reset_index(drop=True)
    wi = w["word_idx"].to_numpy()
    tok = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    X = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)["X"]
    S = X[:, 0:4933][:, elig]
    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    with h5py.File(g / "features.hdf5", "r") as h:
        G = h["layer-24"][...]
    gt = pd.read_csv(g / "transcript.tsv", sep="\t")
    Em = pd.DataFrame(G).groupby(gt["word_idx"].to_numpy()).mean().reindex(wi).to_numpy()
    Em = Em / (np.linalg.norm(Em, axis=1, keepdims=True) + 1e-12)
    coh_p = np.asarray(coherence(S, Em, tok))
    # stored values for the 300-feature pool must be reproduced
    ft = pd.read_csv(cfg.results("feature_table", "feature_table_ifg.csv"))
    mp = dict(zip(fid, coh_p))
    pub = ft[ft.feature_id.isin(fid)]
    dc = np.nanmax(np.abs(pub.coherence.to_numpy() - np.array([mp[f] for f in pub.feature_id])))
    print(f"coherence reproduces the stored 300-pool values: max |diff| {dc:.1e} "
          f"({len(pub)} features)")
    if dc > 1e-9:
        raise SystemExit("coherence does not reproduce the stored values")

    # ---- coherence: film words (sensitivity) ---------------------------------------
    fdir = cfg.results("film_replication", "features")
    wdir = cfg.results("film_replication", "words")
    Sf, Ef, tf = [], [], []
    for f in sorted(fdir.glob("*.hdf5")):
        with h5py.File(f, "r") as h:
            Sf.append(h["sae_word_max"][...][:, fid])
            Ef.append(h["gpt2xl_l24_word"][...])
        tf += [str(x).strip().lower().strip('.,!?;:"\'') for x in
               pd.read_pickle(wdir / f"{f.stem}.pkl")["word"]]
    Sf, Ef = np.concatenate(Sf), np.concatenate(Ef).astype(np.float64)
    Ef = Ef / (np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-12)
    coh_f = np.asarray(coherence(Sf, Ef, np.array(tf)))

    # ---- statistic and null --------------------------------------------------------
    def stat(coh, b):
        cross = 0.5 * (ranknorm(impP[b]) + ranknorm(impF[b]))
        ok = np.isfinite(coh)
        return partial_spearman(coh[ok], cross[ok], lp[ok], lf[ok])

    rows = []
    for lab, coh in (("podcast words (primary)", coh_p), ("film words (sensitivity)", coh_f)):
        obs = stat(coh, 0)
        null = np.array([stat(coh, b) for b in range(1, N_OUTER + 1)])
        p = float((1 + (np.abs(null) >= abs(obs)).sum()) / (N_OUTER + 1))
        # each cohort alone, descriptive
        ok = np.isfinite(coh)
        rP = partial_spearman(coh[ok], impP[0][ok], lp[ok], lf[ok])
        rF = partial_spearman(coh[ok], impF[0][ok], lp[ok], lf[ok])
        rows.append({"coherence_on": lab, "n_features": int(ok.sum()), "partial_rho": obs,
                     "null_2.5": float(np.percentile(null, 2.5)),
                     "null_97.5": float(np.percentile(null, 97.5)), "p_two_sided": p,
                     "podcast_only_rho": rP, "film_only_rho": rF})
    T = pd.DataFrame(rows)
    r0 = T.iloc[0]
    T["verdict"] = ("POSITIVE: the features cortex weights reproducibly across cohorts are "
                    "the more interpretable ones" if r0.partial_rho > 0 and r0.p_two_sided < 0.05
                    else "NEGATIVE" if r0.partial_rho < 0 and r0.p_two_sided < 0.05
                    else "NO RELATION (extends A11)")
    T.to_csv(out / "a45_coherence.csv", index=False)
    pd.DataFrame({"feature_id": fid, "coherence_podcast": coh_p, "coherence_film": coh_f,
                  "importance_podcast": impP[0], "importance_film": impF[0],
                  "log_density_podcast": lp, "log_density_film": lf}).to_csv(
        out / "a45_features.csv", index=False)
    print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("VERDICT:", T.verdict.iloc[0])


if __name__ == "__main__":
    main()
