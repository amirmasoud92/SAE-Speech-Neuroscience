"""Apply a Matryoshka BatchTopK dictionary to the cached layer-12 residual stream.

The dictionary is ``chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40`` (width
32,768) and hooks ``blocks.12.hook_resid_post``, the output of decoder layer 12 - the same
tensor as the primary extraction - so no language-model forward pass is needed. Word pooling
reuses ``aggregate_to_words`` from ``feature_extraction/language_model_features.py``.

Two differences from GemmaScope are handled explicitly:

* ``apply_b_dec_to_input`` is true, so the encoder sees ``x - b_dec``;
* the dictionary was trained with BatchTopK and converted to a JumpReLU threshold for
  inference, so the inference-time activation is the JumpReLU gate on the learned
  threshold.

Achieved L0 and the fraction of variance unexplained are printed as a check on both
points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from feature_extraction.language_model_features import aggregate_to_words

REPO = "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40"
HOOK = "snap/blocks.12.hook_resid_post"


def load_sae(device: str):
    cfg = json.load(open(hf_hub_download(repo_id=REPO, filename=f"{HOOK}/cfg.json")))
    w = load_file(hf_hub_download(repo_id=REPO, filename=f"{HOOK}/sae_weights.safetensors"))
    w = {k: v.to(device=device, dtype=torch.float32) for k, v in w.items()}
    return cfg, w


def encode(x: torch.Tensor, w: dict, apply_b_dec: bool) -> torch.Tensor:
    """JumpReLU encode. ``x``: (n, d_model) -> (n, d_sae)."""
    if apply_b_dec:
        x = x - w["b_dec"]
    pre = x @ w["W_enc"] + w["b_enc"]
    return torch.where(pre > w["threshold"], torch.relu(pre),
                       torch.zeros((), device=pre.device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("results/features/gemma2-2b_L12_w16k_l0-82.hdf5"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/features/gemma2-2b_L12_matryoshka32k.hdf5"))
    ap.add_argument("--min-active-words", type=int, default=25)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, w = load_sae(dev)
    steps = list(cfg["matryoshka_steps"])
    print(f"SAE {REPO}\n  hook {cfg['hook_name']}  d_in {cfg['d_in']}  d_sae {cfg['d_sae']}"
          f"\n  architecture {cfg['architecture']}  apply_b_dec_to_input "
          f"{cfg['apply_b_dec_to_input']}\n  matryoshka_steps {steps}")

    with h5py.File(args.source, "r") as h:
        resid = np.asarray(h["resid_token"])
        tok2word = np.asarray(h["tok2word"])
        surp_tok = np.asarray(h["surprisal_token_bits"])
        n_words = int(h.attrs["n_words"])
        src_layer = int(h.attrs["layer"])
    if src_layer != int(cfg["hook_layer"]):
        raise SystemExit(f"layer mismatch: cache is L{src_layer}, SAE is "
                         f"L{cfg['hook_layer']}")
    print(f"  residuals {resid.shape} from {args.source.name}, {n_words} words")

    x = torch.from_numpy(resid).to(dev)
    with torch.no_grad():
        feats = encode(x, w, bool(cfg["apply_b_dec_to_input"]))
        recon = feats @ w["W_dec"] + w["b_dec"]
        fvu = float(((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
        l0 = float((feats > 0).sum(1).float().mean())
    print(f"  achieved token L0 {l0:.1f} (release targets {cfg['activation_fn_kwargs']['k']})"
          f"   FVU {fvu:.4f}   (GemmaScope 16k L0-82 was 0.273)")
    if not (0.02 < fvu < 0.75):
        raise SystemExit(f"FVU {fvu:.3f} is implausible - the encode formula is probably "
                         f"wrong; refusing to write features")

    feats = feats.cpu().numpy()
    _, feat_max, feat_mean, surp_w = aggregate_to_words(
        tok2word, n_words, resid, feats, surp_tok)

    with h5py.File(args.out, "w") as h:
        h.attrs.update(dict(model=cfg["model_name"], layer=cfg["hook_layer"],
                            sae_repo=REPO, hook=cfg["hook_name"],
                            d_sae=cfg["d_sae"], fvu=fvu, token_l0_mean=l0,
                            matryoshka_steps=json.dumps(steps),
                            n_words=n_words, n_tokens=len(tok2word),
                            apply_b_dec_to_input=int(cfg["apply_b_dec_to_input"])))
        h.create_dataset("sae_word_max", data=feat_max, compression="gzip")
        h.create_dataset("sae_word_mean", data=feat_mean, compression="gzip")
        h.create_dataset("surprisal_word_bits", data=surp_w)
        h.create_dataset("tok2word", data=tok2word)


    usable = int(((feat_max > 0).sum(0) >= args.min_active_words).sum())
    print(f"  features active on >= {args.min_active_words} words: {usable}")
    print("written:", args.out)


if __name__ == "__main__":
    main()
