"""Pooling against re-coding in the primary cohort (exploratory).

The reconstruction is an affine image of the mean-pooled sparse code, so S_mean -> R changes
only basis and capacity, and S -> S_mean changes only the pooling. Uses
``reconstruction_and_sparsity.fit_matrix`` and its bootstrap.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.config import load_config
from word_types.reconstruction_and_sparsity import boot, fit_matrix

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("reconstruction_and_sparsity")
    z = np.load(out / "_shared.npz", allow_pickle=True)
    subj, wi = z["subj"], z["wi"]
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
        F = h["sae_word_mean"][...][wi].astype(np.float64)
    F = F[:, (F > 0).sum(0) >= int(cfg.sae["min_active_words"])]
    sm = fit_matrix("S_mean", F, z)
    sm["r"] = sm["r_exact"][int(np.argmax(sm["r_exact"].mean(1)))]
    A = np.load(out / "a39_a40_arms.npz", allow_pickle=True)
    arm = {n: {k: A[f"{n}__{k}"] for k in ("r", "within", "between")}
           for n in ("D", "R", "w16k_l0-82")}
    S, R, D = arm["w16k_l0-82"], arm["R"], arm["D"]
    print(f"S_mean: {F.shape[1]} live features, mean r {sm['r'].mean():+.4f}, "
          f"median within {np.nanmedian(sm['within']):+.4f}, "
          f"between {np.nanmedian(sm['between']):+.4f}")
    rows = []
    for i, (test, x) in enumerate([
            ("pooling: within(S_mean) - within(S)", sm["within"] - S["within"]),
            ("basis: within(R) - within(S_mean)", R["within"] - sm["within"]),
            ("within(S_mean) - within(D)", sm["within"] - D["within"]),
            ("pooling: between(S_mean) - between(S)", sm["between"] - S["between"]),
            ("basis: between(R) - between(S_mean)", R["between"] - sm["between"]),
            ("total r S_mean - S", sm["r"] - S["r"]),
            ("total r S_mean - D", sm["r"] - D["r"])]):
        ok = np.isfinite(x)
        rows.append({"test": test, **boot(x[ok], subj[ok], cfg.seed + 1200 + i),
                     "status": "exploratory"})
    T = pd.DataFrame(rows)
    T.to_csv(out / "a39_pooling_explore_podcast.csv", index=False)
    print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
