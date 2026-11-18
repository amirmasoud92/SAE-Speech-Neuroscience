"""The film sparse code mean-pooled per word.

Per film, as in ``feature_extraction/reconstruction_error_features.py``: the layer-12 token
activations are recomputed (``run_model``) and encoded with the width-16,384, L0 = 82
dictionary, pooled to words by the mean and added to the film's feature file as group
``w16k_l0-82_mean`` (``sae_word_mean`` on ever-active columns, and ``sae_ids``). The same run's
max pooling must reproduce the stored ``sae_word_max`` exactly.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from utils.config import load_config
from feature_extraction.layer_and_model_features import pool
from feature_extraction.language_model_features import (ExtractConfig, JumpReLUSAE, run_model,
                                                        tokenise_transcript)

log = logging.getLogger("a47")


@torch.no_grad()
def encode(resid, device):
    sae = JumpReLUSAE.from_hub("google/gemma-scope-2b-pt-res", 12, "16k", 82, device,
                               torch.float32)
    F = np.concatenate([sae.encode(torch.tensor(resid[c:c + 2048], device=device))
                        .cpu().numpy() for c in range(0, len(resid), 2048)])
    del sae
    torch.cuda.empty_cache()
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    from transformers import AutoTokenizer
    xcfg = ExtractConfig(device=args.device)
    tok = AutoTokenizer.from_pretrained(xcfg.model_id)
    fdir = cfg.results("reconstruction_and_sparsity", "features")
    for f in sorted(cfg.results("film_replication", "words").glob("*.pkl")):
        dest = fdir / f"btb_{f.stem}.hdf5"
        with h5py.File(dest, "r") as h:
            if "w16k_l0-82_mean" in h:
                continue
            ref_max, ref_ids = h["w16k_l0-82"]["sae_word_max"][...], h["w16k_l0-82"]["sae_ids"][...]
        words = pd.read_pickle(f)["word"].astype(str).tolist()
        ids, t2w = tokenise_transcript(words, tok)
        resid, _ = run_model(xcfg, ids, tok)
        F = encode(resid, args.device)
        fmax = pool(t2w, len(words), F, "max")
        dmax = float(np.abs(fmax[:, ref_ids] - ref_max).max())
        if dmax > 1e-4:
            raise SystemExit(f"{f.stem}: max pooling does not reproduce the stored features "
                             f"({dmax:.2e})")
        fmean = pool(t2w, len(words), F, "mean")
        ever = np.flatnonzero((fmean > 0).any(0))
        with h5py.File(dest, "a") as h:
            g = h.create_group("w16k_l0-82_mean")
            g.create_dataset("sae_word_mean", data=fmean[:, ever], compression="gzip")
            g.create_dataset("sae_ids", data=ever)
            g.attrs["max_pool_reproduction_max_abs"] = dmax
        log.info("%s: mean-pooled %d ever-active features; max-pool reproduction %.1e",
                 f.stem, len(ever), dmax)


if __name__ == "__main__":
    main()
