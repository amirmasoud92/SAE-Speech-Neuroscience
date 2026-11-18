"""Word-type decomposition of out-of-fold predictions.

Restricted to word types with at least four occurrences, each IFG electrode's held-out
prediction is split against word type:

* **between** - correlation of per-type means of prediction and response: does the model know
  which word this is?
* **within** - correlation after removing per-type means: does it know which *occurrence*?

Two forms are computed. ``global`` pools type means over all folds; it compares an occurrence
predicted by one fitted model with occurrences predicted by nine others and is biased against
any representation that encodes word identity (a predictor that knows word identity and
nothing else scores -0.42 on it). ``within_fold`` computes type means inside each test fold,
where the same predictor scores exactly zero. A within-model contrast compares the sparse code
with Gemma-2-2B's own residual stream at 300 components each.

Targets come from ``matched_capacity.ifg_targets`` (cleaned, mean over 0-500 ms). Masked
tokens are excluded per electrode from type means and from both correlations. Writes
``word_types/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

MIN_OCC = 4
SENSITIVITY = (3, 6)
N_BOOT = 10000
ROI = "IFG"
MODES = ("global", "within_fold")


def _means(g, n_g, x, w):
    c = np.zeros((n_g, x.shape[1]))
    m = np.zeros((n_g, x.shape[1]))
    np.add.at(c, g, w)
    np.add.at(m, g, x * w)
    return m / np.maximum(c, 1), c


def decompose(pred, obs, valid, types, min_occ, fid=None):
    """(between, within) correlation per electrode, restricted to repeated word types.

    Word types qualify on their token count over the whole story (fixed across electrodes).
    Within an electrode, masked tokens are dropped: type means use that electrode's valid
    tokens only, a type with no valid token drops out of its between-type correlation, and
    the within-type correlation runs over valid tokens only.

    ``fid`` (test-fold id per token) switches the within-type component to **within-fold**
    type means: each occurrence is compared only with occurrences of the same word
    in the same test fold, over (type, fold) groups with at least two valid tokens. The
    pooled form (``fid=None``) pools type means over all folds, and so compares an
    occurrence predicted by one fitted model with occurrences predicted by nine others - a
    channel through which word identity alone produces a negative score (see
    ``lexical_oracle``). The between-type component is the same in both forms.
    """
    vc = pd.Series(types).value_counts()
    keep = np.isin(types, vc[vc >= min_occ].index)
    p, o, v, t = pred[keep], obs[keep], valid[keep], types[keep]
    idx = pd.factorize(pd.Series(t))[0]
    n_g = idx.max() + 1
    wv = v.astype(np.float64)
    pm, cnt = _means(idx, n_g, p, wv)
    om, _ = _means(idx, n_g, o, wv)
    between = Mt.colcorr(pm, om, cnt > 0)
    if fid is None:
        within = Mt.colcorr(p - pm[idx], o - om[idx], v)
    else:
        gid = pd.factorize(pd.Series(idx * 1000 + fid[keep]))[0]
        n_f = gid.max() + 1
        pf, cf = _means(gid, n_f, p, wv)
        of, _ = _means(gid, n_f, o, wv)
        within = Mt.colcorr(p - pf[gid], o - of[gid], v & (cf[gid] >= 2))
    return between, within, int(keep.sum()), n_g


def lexical_oracle(obs, valid, types, fid):
    """Prediction = the leave-fold-out type mean of the response: word identity only.

    Scored like a representation, it measures how much of a within-type score word identity
    alone can produce under each decomposition - the bias, not a property of cortex.
    """
    P = np.zeros_like(obs)
    tid = pd.factorize(pd.Series(types))[0]
    n_t = tid.max() + 1
    w = valid.astype(np.float64)
    tot, cnt = _means(tid, n_t, obs, w)
    tot = tot * np.maximum(cnt, 1)
    for k in np.unique(fid[fid >= 0]):
        r = fid == k
        tk, ck = _means(tid[r], n_t, obs[r], w[r])
        tk = tk * np.maximum(ck, 1)
        P[r] = ((tot - tk) / np.maximum(cnt - ck, 1))[tid[r]]
    return P


def interaction_stats(D, occ, mode, rng, a="SAE", b="prior"):
    s = D[(D.min_occ == occ) & (D.type_means == mode)].pivot_table(
        index=["subject", "ch"], columns="representation", values=["between", "within"])
    d_within = (s[("within", a)] - s[("within", b)]).to_numpy()
    d_between = (s[("between", a)] - s[("between", b)]).to_numpy()
    inter = d_within - d_between
    subs = s.index.get_level_values("subject").to_numpy()
    uq = np.unique(subs)
    by = {u: inter[subs == u] for u in uq}
    draws = [rng.choice(uq, len(uq), replace=True) for _ in range(N_BOOT)]
    boot = np.array([np.median(np.concatenate([by[u] for u in d])) for d in draws])
    comp = {}
    for lab, x in (("d_within", d_within), ("d_between", d_between)):
        bx = {u: x[subs == u] for u in uq}
        bb = np.array([np.median(np.concatenate([bx[u] for u in d])) for d in draws])
        comp.update({f"{lab}_lo": float(np.percentile(bb, 2.5)),
                     f"{lab}_hi": float(np.percentile(bb, 97.5)),
                     f"{lab}_p_two_sided": float(min(1.0, 2 * min((bb <= 0).mean(),
                                                                  (bb >= 0).mean())
                                                     + 2 / (N_BOOT + 1)))})
    return {**comp, "type_means": mode, "min_occ": occ, "n_electrodes": len(inter),
            "median_d_within": float(np.median(d_within)),
            "median_d_between": float(np.median(d_between)),
            "median_interaction": float(np.median(inter)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "p_one_sided": float((np.sum(boot <= 0) + 1) / (N_BOOT + 1)),
            "p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())
                                     + 2 / (N_BOOT + 1)))}


def verdict(pr, floor=0.01):
    if pr.p_one_sided < 0.05 and pr.median_interaction >= floor:
        return ("H-A28 SUPPORTED: the SAE increment is contextual - it distinguishes "
                "occurrences of the same word")
    if pr.p_one_sided < 0.05:
        return (f"H-A28 reaches significance but the interaction {pr.median_interaction:+.4f} "
                f"is below the pre-specified floor of {floor} - detectable but slight, not "
                "a dissociation")
    if pr.median_d_between > pr.median_d_within:
        return ("H-A28 REJECTED in the other direction: the SAE band is a better lexical "
                "index. A real result and a weaker story; report it as such")
    return ("H-A28 NOT SUPPORTED: the components do not differ. SAE is uniformly better "
            "with no cognitive specificity; make no interpretive claim")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--roi", default=ROI)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 28)

    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    Xsae = Mt.sae_band(cfg)
    Xpri = Mt.zpca(Mt.gpt2xl(cfg, wi), 300)
    nuis = Mt.nuisance(cfg)

    Yobs, V, subj, meta = Mt.ifg_targets(cfg, "window", roi=args.roi)
    Yobs = (Yobs - Yobs.mean(0)) / (Yobs.std(0, keepdims=True) + 1e-12)
    M = meta.assign(valid_frac=V.mean(0))
    print(f"{Yobs.shape[1]} {args.roi} electrodes, {M.subject.nunique()} subjects, "
          f"{len(w)} words, {100 * (~V).mean():.2f}% of cells masked")
    print(f"SAE {Xsae.shape}, prior {Xpri.shape}, nuisance {nuis.shape[1]} columns")

    folds = Mt.folds_for(cfg, onsets)
    fid = np.full(len(w), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    out = {}
    # the model the sparse code was built from, and the sparse code at the same capacity
    arms = [("SAE", Xsae), ("prior", Xpri),
            ("gemma", Mt.zpca(Mt.gemma_dense(cfg, wi), 300)), ("sae300", Mt.zpca(Xsae, 300))]
    for name, Xr in arms:
        print(f"  fitting {name} ...", flush=True)
        # masked cells at the training mean on both sides (the SAE band is too wide for
        # per-pattern exact fits); lambda chosen on total held-out r over valid cells
        P, Yres, _ = Mt.oof(Xr, nuis, Yobs, V, folds, exact=False)
        r, li = Mt.best_lambda_r(P, Yres, V)
        out[name] = (P[li], Yres, float(Mt.LAMBDAS[li]), float(r.mean()))
        print(f"    lambda {Mt.LAMBDAS[li]:g}, mean total r {r.mean():+.4f}")
    Yres = out["SAE"][1]
    out["lexical oracle"] = (lexical_oracle(Yres, V, types, fid), Yres, np.nan, np.nan)

    rows = []
    for name in ("SAE", "prior", "gemma", "sae300", "lexical oracle"):
        P, Yr, lam, _ = out[name]
        for mode in MODES:
            for occ in (MIN_OCC,) + SENSITIVITY:
                b, wn, n_tok, n_typ = decompose(P, Yr, V, types, occ,
                                                fid if mode == "within_fold" else None)
                for e in range(len(b)):
                    rows.append({**M.iloc[e].to_dict(), "representation": name,
                                 "type_means": mode, "min_occ": occ, "between": b[e],
                                 "within": wn[e], "lambda": lam, "n_tokens": n_tok,
                                 "n_types": n_typ})
    D = pd.DataFrame(rows)

    d0 = D[D.min_occ == MIN_OCC]
    print(f"\nA28 - decomposition at min_occ = {MIN_OCC} "
          f"({d0.n_types.iloc[0]} types, {d0.n_tokens.iloc[0]} tokens), medians")
    print("=" * 74)
    print(d0.groupby(["type_means", "representation"])[["between", "within"]]
          .median().to_string())

    R = pd.DataFrame([interaction_stats(D, occ, mode, rng)
                      for mode in MODES for occ in (MIN_OCC,) + SENSITIVITY])
    print("\nPRIMARY: [dWithin(SAE-prior)] - [dBetween(SAE-prior)] > 0")
    print("=" * 96)
    for _, r in R.iterrows():
        tag = ("  <-- POOLED FORM" if (r.min_occ == MIN_OCC and r.type_means == "global")
               else "  <-- WITHIN-FOLD FORM" if r.min_occ == MIN_OCC else "  (sensitivity)")
        print(f"  {r.type_means:11s} min_occ {int(r.min_occ)}: dWithin "
              f"{r.median_d_within:+.4f} | dBetween {r.median_d_between:+.4f} | interaction "
              f"{r.median_interaction:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
              f"p1={r.p_one_sided:.4f} p2={r.p_two_sided:.4f}{tag}")
    print("=" * 96)
    R["verdict"] = [verdict(r) for _, r in R.iterrows()]
    for mode in MODES:
        print(f"VERDICT ({mode}):", R[(R.type_means == mode) & (R.min_occ == MIN_OCC)]
              .verdict.iloc[0])

    o = cfg.results("word_types")
    o.mkdir(parents=True, exist_ok=True)
    D.to_csv(o / "a28_decomposition.csv", index=False)
    R.to_csv(o / "a28_primary.csv", index=False)

    # ---- contrasts against the base model the sparse code was built from ---------------
    rows34 = []
    for a, b, tag in [("sae300", "gemma", "SAE-300 vs Gemma dense-300 (A34 primary)"),
                      ("SAE", "gemma", "SAE band vs Gemma dense-300"),
                      ("SAE", "prior", "SAE band vs GPT-2 XL-300 (A28)")]:
        for mode in MODES:
            for occ in (MIN_OCC,) + SENSITIVITY:
                rows34.append({"contrast": tag, "a": a, "b": b,
                               **interaction_stats(D, occ, mode, rng, a, b)})
    R34 = pd.DataFrame(rows34)
    pr = R34[(R34.a == "sae300") & (R34.type_means == "within_fold")
             & (R34.min_occ == MIN_OCC)].iloc[0]
    ctx = ("sparsification COSTS contextual information" if pr.d_within_hi < 0 else
           "the within-model contrast does not resolve a contextual difference")
    lex = ("sparsification SHARPENS lexical identity" if pr.d_between_lo > 0 else
           "the within-model contrast does not resolve a lexical difference")
    R34["verdict_a34"] = f"{ctx}; {lex}"
    R34.to_csv(o / "a28_within_model.csv", index=False)
    print("\nA34 - against the model the sparse code was built from (within-fold, min_occ 4)")
    for _, r in R34[(R34.type_means == "within_fold") & (R34.min_occ == MIN_OCC)].iterrows():
        print(f"  {r.contrast:42s} dBetween {r.median_d_between:+.4f} "
              f"[{r.d_between_lo:+.4f}, {r.d_between_hi:+.4f}]  dWithin "
              f"{r.median_d_within:+.4f} [{r.d_within_lo:+.4f}, {r.d_within_hi:+.4f}]  "
              f"interaction {r.median_interaction:+.4f} p2={r.p_two_sided:.4f}")
    print("A34 VERDICT:", R34.verdict_a34.iloc[0])
    print("written:", o / "a28_primary.csv")


if __name__ == "__main__":
    main()
