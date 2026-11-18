"""Per-feature coherence for every autoencoder and checkpoint.

The computations of ``evaluate_autoencoders.evaluate``: encode the normalised podcast
activations, max-pool them per word, keep the live features, take the 300-feature variance
pool and apply the coherence metric. Also saves the 300 random 20-type sets behind the chance
floor and GemmaScope's pool. No neural data. Writes
``controls/random_model/a3_coherence_by_checkpoint.npz``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

from pathlib import Path

import h5py
import numpy as np
import torch

from random_model_control.evaluate_autoencoders import encode, pool_words
from utils.config import load_config
from interpretability.coherence import K, coherence

HERE = Path(__file__).parent


def pool_coherence(F, E, tok, cfg):
    F = F[:, (F > 0).sum(0) >= int(cfg.sae["min_active_words"])]
    pool = np.sort(np.argsort(-F.var(0))[:300])
    return np.asarray(coherence(F[:, pool], E, tok), float)


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("controls", "random_model")
    z = np.load(cfg.results("layer_and_model_battery", "_shared.npz"), allow_pickle=True)
    E, tok, wi = z["E"].astype(np.float64), z["tok"], z["wi"]
    res = {}
    for which in ("trained", "random"):
        resid = np.load(out / which / "podcast_resid.npy")
        t2w = np.load(out / which / "podcast_tok2word.npy")
        for k in (0, 25, 50, 100):
            ck = torch.load(out / which / f"sae_{k}M.pt", weights_only=False)
            F = pool_words(encode(resid, ck), t2w, "max")[wi].astype(np.float64)
            res[f"{which}_{k}M"] = pool_coherence(F, E, tok, cfg)
            print(which, k, round(float(np.nanmedian(res[f"{which}_{k}M"])), 4), flush=True)
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
        Fg = h["sae_word_max"][...][wi].astype(np.float64)
    res["gemmascope"] = pool_coherence(Fg, E, tok, cfg)
    rng = np.random.default_rng(cfg.seed)
    types = {}
    for i in range(len(tok)):
        types.setdefault(tok[i], i)
    uniq = np.array(list(types.values()))
    res["random_sets"] = np.array([float((lambda V: (V @ V.T)[np.triu_indices(K, 1)].mean())(
        E[rng.choice(uniq, K, replace=False)])) for _ in range(300)])
    np.savez(out / "a3_coherence_by_checkpoint.npz", **res)
    print("written", out / "a3_coherence_by_checkpoint.npz")


if __name__ == "__main__":
    main()
