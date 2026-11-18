"""Equivalence bound on the sparse-minus-dense difference.

The margin is the measured difference between the two reference arms,
Delta = r(Gemma-2-2B dense, 300 components) - r(GPT-2 XL, 300 components), which is blind to
the sparse-versus-dense contrast it bounds.

* **Non-superiority (one-sided):** is the sparse code's advantage over the dense stream
  smaller than Delta?
* **Equivalence (two one-sided tests):** does the 90% interval lie inside +/- Delta?

Intervals come from the participant-clustered bootstrap of the dimensionality sweep. Writes
``equivalence_bound/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config

N_BOOT = 10000


def clustered_boot(diff, subj, rng, n_boot=N_BOOT):
    """Resample subjects with replacement; statistic is the median over pooled electrodes."""
    uq = np.unique(subj)
    by = {u: diff[subj == u] for u in uq}
    b = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uq, len(uq), replace=True)
        b[i] = np.median(np.concatenate([by[u] for u in pick]))
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 33)

    P = pd.read_csv(cfg.results("dimensionality_sweep", "a30_per_electrode.csv"))
    subj = P["subject"].to_numpy()
    sae = P["SAE PCA-300 exact"].to_numpy(float)
    gem = P["Gemma resid PCA-300"].to_numpy(float)
    gpt = P["GPT-2 XL PCA-300"].to_numpy(float)

    # ---- the margin, fixed from the reference arms alone ---------------------------------
    delta = float(np.mean(gem) - np.mean(gpt))
    print(f"{len(P)} IFG electrodes, {P.subject.nunique()} subjects")
    print(f"MARGIN  Delta = mean r(Gemma dense) - mean r(GPT-2 XL) = "
          f"{np.mean(gem):.4f} - {np.mean(gpt):.4f} = {delta:.4f}")

    d = sae - gem
    obs = float(np.median(d))
    b = clustered_boot(d, subj, rng)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"\nOBSERVED  SAE-300 - Gemma-300 = {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ---- non-superiority: is the SAE advantage below +Delta? -----------------------------
    p_sup = float((b >= delta).mean())
    lo90, hi90 = np.percentile(b, [5.0, 95.0])
    sup_ok = hi90 < delta
    print(f"\nNON-SUPERIORITY (one-sided, H0: SAE - Gemma >= +{delta:.4f})")
    print(f"  90% interval upper bound {hi90:+.4f} against margin {delta:+.4f}  ->  "
          f"{'ESTABLISHED' if sup_ok else 'not established'}")
    print(f"  bootstrap p = {p_sup:.4f}")

    # ---- equivalence: does the whole interval sit inside +/- Delta? ----------------------
    p_lower = float((b <= -delta).mean())
    equiv_ok = (lo90 > -delta) and (hi90 < delta)
    print(f"\nEQUIVALENCE (TOST, H0: |SAE - Gemma| >= {delta:.4f})")
    print(f"  90% interval [{lo90:+.4f}, {hi90:+.4f}] against +/-{delta:.4f}  ->  "
          f"{'ESTABLISHED' if equiv_ok else 'NOT established'}")
    print(f"  p(upper) = {p_sup:.4f}   p(lower) = {p_lower:.4f}   "
          f"TOST p = {max(p_sup, p_lower):.4f}")

    if sup_ok and equiv_ok:
        v = (f"Sparsification is EQUIVALENT to no transform within +/-{delta:.4f}. The "
             "sparse code neither helps nor hurts by more than the base-model effect.")
    elif sup_ok:
        v = (f"NON-SUPERIORITY established - the sparse code's advantage over the dense "
             f"stream is bounded above by {hi90:+.4f}, well inside the {delta:.4f} margin. "
             "Equivalence is NOT established: the data are consistent with sparsification "
             f"COSTING up to {abs(lo90):.4f}: a gain is excluded, a loss is not.")
    else:
        v = ("Neither non-superiority nor equivalence established at this margin. The "
             "interval is too wide to bound the sparsity effect.")
    print("\nVERDICT:", v)

    o = cfg.results("equivalence_bound")
    o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"margin_delta": delta, "gemma_mean_r": float(np.mean(gem)),
                   "gpt2xl_mean_r": float(np.mean(gpt)),
                   "observed_diff": obs, "ci95_lo": float(lo), "ci95_hi": float(hi),
                   "ci90_lo": float(lo90), "ci90_hi": float(hi90),
                   "p_nonsuperiority": p_sup, "p_lower": p_lower,
                   "tost_p": float(max(p_sup, p_lower)),
                   "nonsuperiority_established": bool(sup_ok),
                   "equivalence_established": bool(equiv_ok),
                   "n_electrodes": int(len(P)), "n_subjects": int(P.subject.nunique()),
                   "verdict": v}]).to_csv(o / "a33_equivalence.csv", index=False)
    print("written:", o / "a33_equivalence.csv")


if __name__ == "__main__":
    main()
