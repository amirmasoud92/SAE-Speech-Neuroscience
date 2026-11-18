"""Sensitivity analyses for the matched-capacity results.

1. **Lag rule.** The primary analyses read every IFG electrode as its mean response over
   0-500 ms. Here each electrode is instead read at the peak lag of the *reduced* model (the
   full model without the sparse band) and of the full model.
2. **Penalty-grid edge.** Arms that choose the largest penalty in the grid (1e6) are refitted
   on an extended grid (up to 1e9).
3. **Nested penalty choice.** The penalty is chosen inside each outer training fold.

Writes ``sensitivity/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from capacity_matching import dimensionality_sweep as A30
from capacity_matching import classical_matched as A32
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

EXT = np.array([1e5, 1e6, 1e7, 1e8, 1e9])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["peak", "penalty"],
                    choices=["peak", "penalty"],
                    help="'peak' needs the re-fitted variance partitions; 'penalty' (grid "
                         "edge and nested choice) does not")
    args = ap.parse_args()
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    o = cfg.results("sensitivity")
    o.mkdir(parents=True, exist_ok=True)

    # ---- 1. reduced-model peak lag ---------------------------------------------------
    # peak_full is read from the variance partition fitted on the cleaned targets
    rows1 = []
    for rule in (("peak_reduced", "peak_full") if "peak" in args.parts else ()):
        a32, _ = A32.run(cfg, rule=rule)
        a30, _, _ = A30.run(cfg, rule=rule, sweep=False)
        row = {"rule": rule,
               "a32_sae": a32["sae_mean_r"], "a32_classical": a32["classical_mean_r"],
               "a32_ratio": a32["ratio"], "a32_diff": a32["diff"], "a32_lo": a32["lo"],
               "a32_hi": a32["hi"], "a32_p": a32["p"], "a32_above": a32["n_above"],
               **{f"a30_{k}": v for k, v in a30.items() if k.startswith("sae_minus")}}
        rows1.append(row)
        print(pd.Series(row).to_string(), flush=True)
    if rows1:
        pd.DataFrame(rows1).to_csv(o / "d6102_peak_reduced.csv", index=False)
    if "penalty" not in args.parts:
        return

    # ---- 2. penalty-grid edge --------------------------------------------------------
    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    Y, V, subj, _ = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    nuis = M.nuisance(cfg)
    folds = M.folds_for(cfg, onsets)
    S = M.zpca(M.sae_band(cfg))
    arms = [("SAE PCA-300 exact", S[:, :300], True),
            ("Gemma resid PCA-300", M.zpca(M.gemma_dense(cfg, wi), 300), True),
            ("GPT-2 XL PCA-300", M.zpca(M.gpt2xl(cfg, wi), 300), True),
            ("SAE PCA-600", S[:, :600], False),
            ("SAE PCA-1200", S[:, :1200], False),
            ("SAE PCA-2400", S[:, :2400], False)]
    rows = []
    for name, Xr, ex in arms:
        P, Yr, _ = M.oof(np.ascontiguousarray(Xr), nuis, Y, V, folds, lambdas=EXT, exact=ex)
        rs = np.array([M.colcorr(P[li], Yr, V).mean() for li in range(len(EXT))])
        for lam, r in zip(EXT, rs):
            rows.append({"arm": name, "lambda": lam, "mean_r": float(r)})
        print(f"  {name:22s} " + "  ".join(f"{l:.0e}:{r:+.4f}" for l, r in zip(EXT, rs)),
              flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(o / "d6102_lambda_edge.csv", index=False)
    best = L.loc[L.groupby("arm").mean_r.idxmax()]
    at1e6 = L[L["lambda"] == 1e6].set_index("arm").mean_r
    gain = (best.set_index("arm").mean_r - at1e6).rename("gain_over_1e6")
    print("\nbest penalty per arm and the gain over the grid edge (1e6):")
    print(best.set_index("arm").join(gain).to_string())

    # ---- 3. nested penalty choice ------------------------------------------------------
    # The matched analyses give each arm one penalty, chosen on the mean held-out r across
    # electrodes. Here the penalty for each test fold is chosen on the other nine folds'
    # held-out predictions only, so no fold is scored by a penalty it helped choose.
    from feature_extraction.feature_bands import classical_features
    C, _ = classical_features(cfg, wi, w["word"])
    arms3 = [("classical-85", np.asarray(C, dtype=np.float64)),
             ("SAE PCA-85", S[:, :85]),
             ("SAE PCA-300 exact", S[:, :300]),
             ("Gemma resid PCA-300", M.zpca(M.gemma_dense(cfg, wi), 300)),
             ("GPT-2 XL PCA-300", M.zpca(M.gpt2xl(cfg, wi), 300))]
    # A fold's penalty changes the SCALE of its predictions (a 10x smaller penalty can give
    # several-fold larger predictions), and one correlation over predictions stitched
    # from folds of different scale falls even when every fold predicts equally well. Each
    # fold's predictions are therefore standardised within that fold, per electrode, before
    # pooling - for the shared and the nested penalty alike, so the two are comparable.
    def fold_standardised(Pk):
        Z = np.zeros_like(Pk)
        for _, te in folds:
            v = V[te].astype(np.float64)
            n = np.maximum(v.sum(0), 1.0)
            mu = (Pk[te] * v).sum(0) / n
            sd = np.sqrt((((Pk[te] - mu) ** 2) * v).sum(0) / n) + 1e-12
            Z[te] = (Pk[te] - mu) / sd
        return Z

    rows3 = []
    for name, Xr in arms3:
        P, Yr, _ = M.oof(np.ascontiguousarray(Xr), nuis, Y, V, folds)
        shared, li_shared = M.best_lambda_r(P, Yr, V)
        Pn = np.zeros_like(Yr)
        picks = []
        for f, (_, te) in enumerate(folds):
            other = np.concatenate([t for g, (_, t) in enumerate(folds) if g != f])
            li = int(np.argmax([M.colcorr(P[k][other], Yr[other], V[other]).mean()
                                for k in range(P.shape[0])]))
            Pn[te] = P[li][te]
            picks.append(float(M.LAMBDAS[li]))
        nested_raw = M.colcorr(Pn, Yr, V)
        shared_z = M.colcorr(fold_standardised(P[li_shared]), Yr, V)
        nested_z = M.colcorr(fold_standardised(Pn), Yr, V)
        rows3.append({"arm": name, "shared_lambda_mean_r": float(shared.mean()),
                      "nested_lambda_mean_r_unstandardised": float(nested_raw.mean()),
                      "shared_lambda_fold_standardised": float(shared_z.mean()),
                      "nested_lambda_mean_r": float(nested_z.mean()),
                      "difference": float(nested_z.mean() - shared_z.mean()),
                      "nested_picks": ";".join(f"{x:g}" for x in picks)})
        print(f"  {name:22s} shared {shared.mean():+.4f} | fold-standardised: shared "
              f"{shared_z.mean():+.4f} nested {nested_z.mean():+.4f} | unstandardised "
              f"nested {nested_raw.mean():+.4f}", flush=True)
    pd.DataFrame(rows3).to_csv(o / "d6102_nested_lambda.csv", index=False)


if __name__ == "__main__":
    main()
