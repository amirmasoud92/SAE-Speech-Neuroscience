"""Apply the dictionaries of the sensitivity grid to the cached layer-12 token residuals.

The dictionaries listed under ``sae.sensitivity`` in ``config.yaml`` (width 16,384 at
L0 = 22 and 176, width 32,768 at L0 = 76, width 65,536 at L0 = 72) encode the same
activations as the primary dictionary, so the Gemma forward pass of
``feature_extraction/language_model_features.py`` is reused and only the encoder is applied.
Word pooling and the live-feature rule are identical to the primary dictionary.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.config import load_config
from feature_extraction.language_model_features import JumpReLUSAE, aggregate_to_words
from utils.run_logging import setup_logging, stage

log = logging.getLogger("sae_variants")


def encode_variant(resid: np.ndarray, tok2word: np.ndarray, n_words: int,
                   surp_tok: np.ndarray, repo: str, layer: int, width: str, l0: int,
                   device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Encode cached token residuals with one GemmaScope dictionary.

    Parameters
    ----------
    resid : (n_tokens, d_model) float32 - residual stream after ``layer``
    tok2word : (n_tokens,) int - token to word index map
    surp_tok : (n_tokens,) float32 - token surprisal in bits

    Returns
    -------
    arrays : dict of word-level arrays ready to write
    stats : dict of scalar diagnostics
    """
    sae = JumpReLUSAE.from_hub(repo, layer, width, l0, device, torch.float32)
    with torch.no_grad():
        x = torch.tensor(resid, device=device, dtype=torch.float32)
        feats_t = sae.encode(x)
        recon = sae.decode(feats_t)
        fvu = float(((recon - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
        l0_tok = (feats_t > 0).float().sum(-1)
        token_l0 = float(l0_tok.mean())
        feats = feats_t.cpu().numpy().astype(np.float32)
    del feats_t, recon, x
    torch.cuda.empty_cache()

    resid_w, feat_max, feat_mean, surp_w = aggregate_to_words(
        tok2word, n_words, resid, feats, surp_tok)
    alive = feat_max > 0
    arrays = dict(resid_word=resid_w, sae_word_max=feat_max, sae_word_mean=feat_mean,
                  surprisal_word_bits=surp_w, tok2word=tok2word)
    stats = dict(layer=layer, width=width, nominal_l0=l0, d_sae=int(feat_max.shape[1]),
                 token_l0_mean=token_l0, fvu=fvu,
                 word_l0_mean=float(alive.sum(1).mean()),
                 n_features_ever_active=int(alive.any(0).sum()),
                 n_features_ge5_words=int((alive.sum(0) >= 5).sum()),
                 n_features_ge10_words=int((alive.sum(0) >= 10).sum()),
                 n_features_ge25_words=int((alive.sum(0) >= 25).sum()),
                 n_features_ge50_words=int((alive.sum(0) >= 50).sum()))
    return arrays, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or (cfg.paths.results_root / "features")
    out.mkdir(parents=True, exist_ok=True)
    log_ = setup_logging(cfg.paths.results_root / "logs", "sae_variants", cfg.digest)

    p = cfg.sae["primary"]
    src = out / f"gemma2-2b_L{p['layer']}_w{p['width']}_l0-{p['l0']}.hdf5"
    with h5py.File(src, "r") as h:
        if "resid_token" not in h:
            raise SystemExit(
                f"{src.name} has no 'resid_token' dataset. Re-run feature_extraction.language_model_features "
                f"with the current version, which caches token-level residuals.")
        resid = h["resid_token"][...]
        tok2word = h["tok2word"][...]
        surp_tok = h["surprisal_token_bits"][...]
        n_words = int(h.attrs["n_words"])
    log_.info("loaded %s: %s token residuals, %d words", src.name, resid.shape, n_words)

    all_stats = []
    for v in cfg.sae["sensitivity"]:
        if int(v["layer"]) != int(p["layer"]):
            log_.warning("skipping layer-%s variant: needs its own forward pass",
                         v["layer"])
            continue
        tag = f"L{v['layer']}_w{v['width']}_l0-{v['l0']}"
        dest = out / f"gemma2-2b_{tag}.hdf5"
        if dest.exists():
            log_.info("%s already exists, skipping", dest.name)
            continue
        with stage(log_, f"encode {tag}"):
            arrays, stats = encode_variant(
                resid, tok2word, n_words, surp_tok, cfg.sae["repo"],
                int(v["layer"]), str(v["width"]), int(v["l0"]), args.device)
            with h5py.File(dest, "w") as h:
                for k, arr in arrays.items():
                    h.create_dataset(k, data=arr, compression="gzip")
                h.attrs.update(dict(model=cfg.llm["model_id"], sae_repo=cfg.sae["repo"],
                                    n_words=n_words, n_tokens=len(tok2word), **stats))
            (out / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2))
        stats["tag"] = tag
        all_stats.append(stats)
        log_.info("%s: token L0 %.1f (nominal %d) | FVU %.4f | %d features on >=25 words",
                  tag, stats["token_l0_mean"], stats["nominal_l0"], stats["fvu"],
                  stats["n_features_ge25_words"])

    if all_stats:
        print("\n{:<22s} {:>10s} {:>8s} {:>8s} {:>12s}".format(
            "variant", "token L0", "nominal", "FVU", ">=25 words"))
        for s in all_stats:
            print("{:<22s} {:>10.1f} {:>8d} {:>8.4f} {:>12d}".format(
                s["tag"], s["token_l0_mean"], s["nominal_l0"], s["fvu"],
                s["n_features_ge25_words"]))


if __name__ == "__main__":
    main()
