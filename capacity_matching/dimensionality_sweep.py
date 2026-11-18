"""Predictivity of the sparse space against the number of principal components retained.

The z-scored sparse space is reduced to k = 25, 50, 100, 300, 600, 1,200, 2,400 and 4,933
principal components and fitted against the IFG window targets.

**Primary:** r(k = 4,933) - r(k = 300), participant-clustered bootstrap, two-sided.
**Reference arms** at matched capacity: Gemma-2-2B's own residual stream at 300 components
and GPT-2 XL layer 24 at 300 components.

The three 300-component arms are fitted with exact masking; the sweep uses training-mean
imputation of masked cells, and the effect of that approximation is measured at k = 300 and
written to the primary output. Reduction is always by principal components, never by
truncating features. Writes ``dimensionality_sweep/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

KS = [25, 50, 100, 300, 600, 1200, 2400, 4933]


def run(cfg, rule="window", targets_digest=None, vp_root=None, zscore=True, exact=True,
        sweep=True):
    rng = np.random.default_rng(cfg.seed + 30)
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, meta = M.ifg_targets(cfg, rule, vp_root=vp_root, digest=targets_digest)
    if targets_digest is not None and targets_digest != cfg.digest:
        V = np.ones_like(V)
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)

    S = M.zpca(M.sae_band(cfg), None, zscore=zscore)
    per, rows = {}, []

    def fit(name, Xr, k, ex):
        P, Yr, _ = M.oof(np.ascontiguousarray(Xr), nuis, Y, V, folds, exact=ex)
        r, li = M.best_lambda_r(P, Yr, V)
        per[name] = r
        rows.append({"arm": name, "k": k, "mean_r": float(r.mean()),
                     "median_r": float(np.median(r)), "lambda": float(M.LAMBDAS[li]),
                     "exact_mask": ex})
        print(f"  {name:22s} k={k:>4}  mean r {r.mean():+.4f}  (lambda {M.LAMBDAS[li]:g}"
              f"{'' if ex else ', imputed'})", flush=True)

    if sweep:
        for k in KS:
            kk = min(k, S.shape[1])
            fit(f"SAE PCA-{k}", S[:, :kk], kk, False)
    fit("SAE PCA-300 exact", S[:, :300], 300, exact)
    fit("Gemma resid PCA-300", M.zpca(M.gemma_dense(cfg, wi), 300, zscore=zscore), 300, exact)
    fit("GPT-2 XL PCA-300", M.zpca(M.gpt2xl(cfg, wi), 300, zscore=zscore), 300, exact)

    T = pd.DataFrame(rows)
    out = {"rule": rule, "zscore_pca": zscore, "exact_mask": exact,
           "masked_frac": float((~V).mean()), "n_electrodes": int(Y.shape[1])}
    if sweep:
        curve = T[T.arm.str.match(r"^SAE PCA-\d+$")]
        best = curve.mean_r.max()
        out["k_star"] = int(curve.loc[curve.mean_r >= 0.95 * best, "k"].min())
        out["best_mean_r"] = float(best)
        m, lo, hi, p, _ = M.cluster_boot(per["SAE PCA-4933"] - per["SAE PCA-300"], subj, rng)
        out.update(primary_diff=m, primary_lo=lo, primary_hi=hi, primary_p=p)
        out["imputation_effect_k300"] = float(per["SAE PCA-300 exact"].mean()
                                              - per["SAE PCA-300"].mean())
    for ref, key in [("Gemma resid PCA-300", "gemma"), ("GPT-2 XL PCA-300", "gpt2xl")]:
        m, lo, hi, p, _ = M.cluster_boot(per["SAE PCA-300 exact"] - per[ref], subj, rng)
        out.update({f"sae_minus_{key}": m, f"sae_minus_{key}_lo": lo,
                    f"sae_minus_{key}_hi": hi, f"sae_minus_{key}_p": p})
    pe = pd.DataFrame({k: v for k, v in per.items()}).assign(subject=subj)
    return out, T, pe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, T, pe = run(cfg)
    print("\n" + "=" * 82)
    print(f"saturation k* = {out['k_star']}   r(4933)-r(300) = {out['primary_diff']:+.4f} "
          f"[{out['primary_lo']:+.4f}, {out['primary_hi']:+.4f}]  P = {out['primary_p']:.3f}")
    print(f"SAE-300 - Gemma-300   = {out['sae_minus_gemma']:+.4f} "
          f"[{out['sae_minus_gemma_lo']:+.4f}, {out['sae_minus_gemma_hi']:+.4f}]  "
          f"P = {out['sae_minus_gemma_p']:.4f}")
    print(f"SAE-300 - GPT-2 XL-300 = {out['sae_minus_gpt2xl']:+.4f}  "
          f"P = {out['sae_minus_gpt2xl_p']:.3f}")
    print(f"imputation vs exact at k=300: {out['imputation_effect_k300']:+.5f}")
    print("=" * 82)
    o = cfg.results("dimensionality_sweep")
    o.mkdir(parents=True, exist_ok=True)
    T.to_csv(o / "a30_curve.csv", index=False)
    pd.DataFrame([out]).to_csv(o / "a30_primary.csv", index=False)
    pe.to_csv(o / "a30_per_electrode.csv", index=False)
    print("written:", o / "a30_primary.csv")


if __name__ == "__main__":
    main()
