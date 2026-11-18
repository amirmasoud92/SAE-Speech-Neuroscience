"""Stimulus-side properties of the five dictionaries of the sensitivity grid.

No neural data. For each dictionary:

* variance concentration - the share of total feature variance carried by the 300
  highest-variance live features;
* coherence of those 300 features (``interpretability/coherence.py``, GPT-2 XL layer 24);
* usable directions - how many of the 300 exceed the median coherence of 300 principal
  components of the residual stream.

Writes ``dictionary_grid/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils.config import load_config
from feature_extraction.feature_bands import dense_residual
from interpretability.coherence import coherence
from encoding.fit_encoding_model import analysed_words

POOL = 300


def load_dict(path: Path, word_idx: np.ndarray, min_active: int, key: str):
    """Load an SAE activation file exactly as `sae_features` does.

    The first version guessed the dataset name and took whichever key came first, which is
    `resid_word` - the 2,304-dimensional dense residual stream. Every dictionary then
    reported 2,303 live features with identical statistics, and the apparent finding that
    the usable-direction count varies with dictionary was an artefact of profiling the
    residual stream five times over. The key comes from the config now, and the rows are
    already word-indexed so no pooling is needed.
    """
    with h5py.File(path, "r") as h:
        F = h[key][...][word_idx].astype(np.float32)
    live = (F > 0).sum(0) >= min_active
    return F[:, live], np.flatnonzero(live)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    w = analysed_words(cfg).reset_index(drop=True)
    wi = w["word_idx"].to_numpy()
    tok = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    min_active = int(cfg.sae["min_active_words"])
    sae_key = ("sae_word_max" if cfg.llm["word_pooling"]["sae"] == "max"
               else "sae_word_mean")
    print(f"pooling key {sae_key}, min_active_words {min_active}")

    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    with h5py.File(g / "features.hdf5", "r") as h:
        G = h["layer-24"][...]
    gt = pd.read_csv(g / "transcript.tsv", sep="\t")
    E = pd.DataFrame(G).groupby(gt["word_idx"].to_numpy()).mean().reindex(wi).to_numpy()
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    del G
    print(f"reference embedding {E.shape} (GPT-2 XL layer 24)")

    # The baseline is PCA of the DENSE RESIDUAL STREAM at both poles - not PCA of the SAE
    # activations, which would be a different comparison.
    R = dense_residual(cfg, wi)
    P = PCA(n_components=POOL, random_state=cfg.seed).fit_transform(R)
    c_pca = np.concatenate([np.asarray(coherence(P, E, tok)),
                            np.asarray(coherence(-P, E, tok))])
    med_p = float(np.nanmedian(c_pca))
    print(f"PCA baseline from the residual stream {R.shape}: median coherence "
          f"{med_p:.4f} over {int(np.isfinite(c_pca).sum())} signed components")
    del R, P

    files = sorted(cfg.results("features").glob("gemma2-2b_L12_w*.hdf5"))
    rows = []
    for f in files:
        name = f.stem.replace("gemma2-2b_L12_", "")
        M, ids = load_dict(f, wi, min_active, sae_key)
        var = M.var(0)
        order = np.argsort(-var)
        pool = np.sort(order[:POOL])
        S = M[:, pool]
        share = float(var[pool].sum() / var.sum())

        coh = np.asarray(coherence(S, E, tok))
        usable = int(np.nansum(coh > med_p))

        rows.append({"dictionary": name, "n_live": int(M.shape[1]),
                     "top300_variance_share": share,
                     "median_coherence_sae": float(np.nanmedian(coh)),
                     "median_coherence_pca": med_p,
                     "usable_directions": usable, "pool": POOL,
                     "median_density_words": float(np.median((S > 0).sum(0)))})
        print(f"  {name:<18s} live {M.shape[1]:6d} | top-300 var {100 * share:4.1f}% | "
              f"coherence SAE {np.nanmedian(coh):.3f} vs PCA {med_p:.3f} | "
              f"usable {usable}/{POOL}")
        del M, S

    R = pd.DataFrame(rows)
    o = cfg.results("dictionary_grid")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "a23a_stimulus_side.csv", index=False)

    print("\nA23a summary\n" + "=" * 74)
    print(R.to_string(index=False))
    main_row = R[R.dictionary.str.contains("16k_l0-82")]
    if len(main_row):
        u = int(main_row.usable_directions.iloc[0])
        spread = int(R.usable_directions.max() - R.usable_directions.min())
        print(f"\nmain dictionary gives {u}/{POOL} usable directions; across all "
              f"{len(R)} dictionaries the count spans {spread}")
        print("VERDICT: " + ("the usable-direction advantage is ROBUST across dictionaries"
                             if spread <= 40 else
                             "the usable-direction count VARIES with dictionary - report "
                             "the dependence rather than a single number"))
    print("written:", o / "a23a_stimulus_side.csv")


if __name__ == "__main__":
    main()
