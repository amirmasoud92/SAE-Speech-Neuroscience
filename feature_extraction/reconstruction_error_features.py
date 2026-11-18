"""Dictionary reconstruction and error, and every released GemmaScope dictionary at layer 12.

Stimulus side only. Per stimulus (the podcast, and each film of the replication cohort),
from the layer-12 token activations x (1,024-token windows, stride 512, BOS, float32):

* for the primary dictionary (width 16,384, L0 = 82): the reconstruction
  x_hat = decode(encode(x)) and the error e = x - x_hat, both mean-pooled per word, so that
  dense = reconstruction + error holds exactly at word level (``resid_word`` is stored beside
  them and checked);
* for every dictionary of width 16,384 (L0 22, 41, 82, 176, 445) and, for the podcast, of
  width 65,536 (L0 21, 38, 72, 141, 297): the features, max-pooled per word, with the
  achieved token L0 and the fraction of variance unexplained.

Podcast token activations are the cached ones; film activations are recomputed per film by
``language_model_features.run_model``.
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

log = logging.getLogger("a39_a40")
REPO = "google/gemma-scope-2b-pt-res"
FAMILIES = {"16k": [22, 41, 82, 176, 445], "65k": [21, 38, 72, 141, 297]}
PRIMARY = ("16k", 82)


@torch.no_grad()
def encode_all(resid: np.ndarray, width: str, l0: int, device: str, recon: bool = False):
    sae = JumpReLUSAE.from_hub(REPO, 12, width, l0, device, torch.float32)
    feats, rec = [], []
    num = den = 0.0
    mu = resid.mean(0)
    for c0 in range(0, len(resid), 2048):
        x = torch.tensor(resid[c0:c0 + 2048], device=device)
        f = sae.encode(x)
        xh = sae.decode(f)
        num += float(((xh - x) ** 2).sum())
        den += float(((x - torch.tensor(mu, device=device)) ** 2).sum())
        feats.append(f.cpu().numpy())
        if recon:
            rec.append(xh.cpu().numpy())
    F = np.concatenate(feats)
    out = {"F": F, "token_l0": float((F > 0).sum(1).mean()), "fvu": num / den}
    if recon:
        out["R"] = np.concatenate(rec)
    del sae
    torch.cuda.empty_cache()
    return out


def stimulus(resid, t2w, n_words, families, out_path: Path, device: str) -> None:
    with h5py.File(out_path, "w") as h:
        h.create_dataset("resid_word", data=pool(t2w, n_words, resid, "mean"),
                         compression="gzip")
        for width, l0s in families.items():
            for l0 in l0s:
                is_primary = (width, l0) == PRIMARY
                o = encode_all(resid, width, l0, device, recon=is_primary)
                fw = pool(t2w, n_words, o["F"], "max")
                ever = np.flatnonzero((fw > 0).any(0))
                g = h.create_group(f"w{width}_l0-{l0}")
                g.create_dataset("sae_word_max", data=fw[:, ever], compression="gzip")
                g.create_dataset("sae_ids", data=ever)
                g.attrs.update(width=width, nominal_l0=l0, token_l0=o["token_l0"],
                               fvu=o["fvu"])
                if is_primary:
                    rw = pool(t2w, n_words, o["R"], "mean")
                    ew = pool(t2w, n_words, resid - o["R"], "mean")
                    h.create_dataset("recon_word", data=rw, compression="gzip")
                    h.create_dataset("error_word", data=ew, compression="gzip")
                    d = np.abs(h["resid_word"][...] - rw - ew).max()
                    h.attrs["identity_max_abs"] = float(d)
                    if d > 1e-3:
                        raise RuntimeError(f"dense != recon + error at word level ({d})")
                log.info("%s  w%s L0=%d: token L0 %.1f, FVU %.3f, %d ever-active", out_path.stem,
                         width, l0, o["token_l0"], o["fvu"], len(ever))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["podcast", "btb"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("reconstruction_and_sparsity", "features")
    out.mkdir(parents=True, exist_ok=True)

    if "podcast" in args.which and not (out / "podcast.hdf5").exists():
        with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
            resid, t2w = h["resid_token"][...], h["tok2word"][...]
            n_words = int(h.attrs["n_words"])
        stimulus(resid, t2w, n_words, FAMILIES, out / "podcast.hdf5", args.device)

    if "btb" in args.which:
        from transformers import AutoTokenizer
        xcfg = ExtractConfig(device=args.device)
        tok = AutoTokenizer.from_pretrained(xcfg.model_id)
        wdir = cfg.results("film_replication", "words")
        for f in sorted(wdir.glob("*.pkl")):
            dest = out / f"btb_{f.stem}.hdf5"
            if dest.exists():
                continue
            words = pd.read_pickle(f)["word"].astype(str).tolist()
            ids, t2w = tokenise_transcript(words, tok)
            resid, _ = run_model(xcfg, ids, tok)
            stimulus(resid, t2w, len(words), {"16k": FAMILIES["16k"]}, dest, args.device)


if __name__ == "__main__":
    main()
