"""Language-model features against hand-crafted linguistics at matched capacity.

Both arms have exactly 85 dimensions: 85 principal components of the sparse feature space
against the 85 classical columns (35 part-of-speech tags, 43 dependency relations, a stopword
flag, concreteness, valence, arousal, dominance and two norm-availability indicators).
Targets, masking, z-scored PCA and ridge come from ``capacity_matching/matched_capacity.py``.
The difference in mean IFG out-of-fold r is tested with a participant-clustered bootstrap.
Writes ``classical_matched/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from feature_extraction.feature_bands import classical_features
from encoding.fit_encoding_model import analysed_words


def run(cfg, rule="window", targets_digest=None, vp_root=None, zscore=True, exact=True,
        rng_seed=None):
    rng = np.random.default_rng(cfg.seed + 32 if rng_seed is None else rng_seed)
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, meta = M.ifg_targets(cfg, rule, vp_root=vp_root, digest=targets_digest)
    if targets_digest is not None and targets_digest != cfg.digest:
        V = np.ones_like(V)                     # uncleaned cache carries no mask
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)

    C, _ = classical_features(cfg, wi, w["word"])
    C = np.asarray(C, dtype=np.float64)
    k = C.shape[1]
    arms = {f"classical-{k}": C,
            f"SAE PCA-{k}": M.zpca(M.sae_band(cfg), k, zscore=zscore)}
    res = {}
    for name, Xr in arms.items():
        P, Yr, npat = M.oof(Xr, nuis, Y, V, folds, exact=exact)
        res[name], li = M.best_lambda_r(P, Yr, V)
        print(f"  {name:16s} mean r {res[name].mean():+.4f}  (lambda {M.LAMBDAS[li]:g}, "
              f"{npat} fold x pattern fits)", flush=True)
    sae, cla = res[f"SAE PCA-{k}"], res[f"classical-{k}"]
    m, lo, hi, p, _ = M.cluster_boot(sae - cla, subj, rng)
    out = {"comparison": f"SAE PCA-{k} - classical-{k}", "diff": m, "lo": lo, "hi": hi,
           "p": p, "sae_mean_r": float(sae.mean()), "classical_mean_r": float(cla.mean()),
           "ratio": float(sae.mean() / cla.mean()), "n_above": int((sae > cla).sum()),
           "n_electrodes": len(sae), "n_subjects": int(len(np.unique(subj))),
           "rule": rule, "zscore_pca": zscore, "exact_mask": exact,
           "masked_frac": float((~V).mean())}
    per = pd.DataFrame({f"classical-{k}": cla, f"SAE PCA-{k}": sae, "subject": subj})
    return out, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out, per = run(cfg)
    print("\n" + "=" * 78)
    print(f"A32  SAE-85 {out['sae_mean_r']:.4f}  classical-85 {out['classical_mean_r']:.4f}  "
          f"ratio {out['ratio']:.1f}x")
    print(f"     difference {out['diff']:+.4f} [{out['lo']:+.4f}, {out['hi']:+.4f}]  "
          f"P = {out['p']:.4f}   {out['n_above']}/{out['n_electrodes']} above identity")
    print("=" * 78)
    o = cfg.results("classical_matched")
    o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(o / "a32_primary.csv", index=False)
    per.to_csv(o / "a32_per_electrode.csv", index=False)
    print("written:", o / "a32_primary.csv")


if __name__ == "__main__":
    main()
