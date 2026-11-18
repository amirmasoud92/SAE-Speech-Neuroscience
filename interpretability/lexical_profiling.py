"""Lexical profile of the two feature rankings.

Each feature of the fixed pool is given a quantitative profile of its top-20 distinct
activating words - function-word fraction, mean concreteness, frequency and length - and
causal necessity and neural importance are correlated with each property across the 300
features, raw and partialled on log activation density and coherence, with false-discovery-
rate correction. Writes ``lexical_profiling/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import load_config
from feature_extraction.lexical_properties import word_properties
from encoding.fit_encoding_model import analysed_words
from inference.statistics import fdr

TOP_K = 20
PROFILE = ["is_function", "concreteness", "zipf", "n_char"]
SCORES = [("importance", "neural importance"),
          ("causal_dlp_mag", "causal necessity (delta log-prob)"),
          ("causal_kl", "causal necessity (KL)")]


def partial_corr(x, y, Z):
    """Spearman correlation of x and y after removing Z from both, on ranks."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    xr, yr = sps.rankdata(x[ok]), sps.rankdata(y[ok])
    Zr = np.column_stack([sps.rankdata(Z[ok, j]) for j in range(Z.shape[1])]
                         + [np.ones(ok.sum())])
    q, _ = np.linalg.qr(Zr)
    xr = xr - q @ (q.T @ xr)
    yr = yr - q @ (q.T @ yr)
    r = float(np.corrcoef(xr, yr)[0, 1])
    n, k = int(ok.sum()), Zr.shape[1]
    t = r * np.sqrt(max(n - k - 1, 1) / max(1 - r ** 2, 1e-12))
    return r, float(2 * sps.t.sf(abs(t), max(n - k - 1, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--suffix", default="_ifg",
                    help="input/output suffix on the causal feature table (_ifg: neural "
                         "importance averaged over all IFG electrodes)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sfx = args.suffix

    w = analysed_words(cfg).reset_index(drop=True)
    props = word_properties(cfg, w)
    d = np.load(cfg.cache(cfg.digest, "design.npz"), allow_pickle=True)
    X, sl = d["X"], json.loads(str(d["slices"]))
    ids = np.asarray(d["ids_sae"]).astype(int)
    Xs = X[:, sl["sae"][0]:sl["sae"][1]]

    tab = pd.read_csv(cfg.results("causal", f"a11_feature_table{sfx}.csv"))
    print(f"{len(tab)} features carry both a neural importance and a causal score")

    # ---- profile each feature by the words it fires on --------------------------------
    rows = []
    for _, f in tab.iterrows():
        col = int(np.flatnonzero(ids == int(f.feature))[0])
        act = Xs[:, col]
        hit = np.flatnonzero(act > 0)
        if len(hit) == 0:
            continue
        # top-K distinct word types, strongest activation first
        order = hit[np.argsort(-act[hit])]
        seen, pick = set(), []
        for i in order:
            t = props.word.iloc[i]
            if t not in seen:
                seen.add(t)
                pick.append(i)
            if len(pick) == TOP_K:
                break
        p = props.iloc[pick]
        rows.append({"feature": int(f.feature), "n_top": len(pick),
                     "is_function": float(p.is_function.mean()),
                     "concreteness": float(np.nanmean(p.concreteness)),
                     "zipf": float(p.zipf.mean()), "n_char": float(p.n_char.mean()),
                     "top_words": " ".join(p.word.tolist()[:8])})
    P = pd.DataFrame(rows).merge(tab, on="feature")
    print(f"profiled {len(P)} features on their top-{TOP_K} distinct words")

    # ---- the test: does the profile differ between the two rankings? -------------------
    Z = np.column_stack([P.log_density.to_numpy(), P.coherence.to_numpy()])
    out = []
    for score, slabel in SCORES:
        for prop in PROFILE:
            r, p = partial_corr(P[prop].to_numpy(float), P[score].to_numpy(float), Z)
            out.append({"score": slabel, "property": prop, "partial_rho": r, "p": p})
    R = pd.DataFrame(out)
    rej, q = fdr(R.p.to_numpy(), alpha=0.05)
    R["q"], R["survives_fdr"] = q, rej

    print("\nA18 - partial Spearman, controlling log density and coherence\n" + "=" * 74)
    for slabel in R.score.unique():
        print(f"\n{slabel}")
        for _, r in R[R.score == slabel].iterrows():
            star = " *" if r.survives_fdr else ""
            print(f"   {r['property']:<14s} rho {r.partial_rho:+.3f}   p {r.p:.4f}   "
                  f"q {r.q:.4f}{star}")
    print("=" * 74)

    # ---- exemplars, printed after the statistics so they illustrate rather than argue --
    print(f"\ntop {8} features by neural importance")
    for _, r in P.nlargest(8, "importance").iterrows():
        print(f"   f{r.feature:<6d} imp {r.importance:5.2f}  fn {r.is_function:.2f}  "
              f"conc {r.concreteness:4.2f}  | {r.top_words}")
    print(f"\ntop {8} features by causal necessity (KL)")
    for _, r in P.nlargest(8, "causal_kl").iterrows():
        print(f"   f{r.feature:<6d} KL  {r.causal_kl:5.3f}  fn {r.is_function:.2f}  "
              f"conc {r.concreteness:4.2f}  | {r.top_words}")

    # ---- raw correlations, uncontrolled, printed because the control may remove the ----
    # ---- effect: function words ARE high-density, so partialling density partials -----
    # ---- part of what makes a word closed-class -------------------------------------
    raw = []
    for score, slabel in SCORES:
        for prop in PROFILE:
            ok = np.isfinite(P[prop]) & np.isfinite(P[score])
            rr, pp = sps.spearmanr(P[prop][ok], P[score][ok])
            raw.append({"score": slabel, "property": prop, "raw_rho": float(rr),
                        "raw_p": float(pp)})
    RAW = pd.DataFrame(raw)
    print("\nraw (uncontrolled) Spearman, for comparison")
    for slabel in RAW.score.unique():
        r_ = RAW[RAW.score == slabel]
        print(f"   {slabel:<34s} " + "  ".join(
            f"{x['property']} {x.raw_rho:+.3f}" for _, x in r_.iterrows()))

    o = cfg.results("lexical_profiling")
    o.mkdir(parents=True, exist_ok=True)
    P.to_csv(o / f"a18_feature_profiles{sfx}.csv", index=False)
    R.to_csv(o / f"a18_partial_correlations{sfx}.csv", index=False)
    RAW.to_csv(o / f"a18_raw_correlations{sfx}.csv", index=False)
    print("\nwritten:", o / f"a18_partial_correlations{sfx}.csv")


if __name__ == "__main__":
    main()
