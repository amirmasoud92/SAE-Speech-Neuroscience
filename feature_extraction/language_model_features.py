"""Gemma-2-2B residual-stream activations, GemmaScope sparse features and word surprisal.

Stimulus side only: no neural data are read.

Context handling
----------------
Activations are taken with a sliding window: each window is ``context`` tokens long (1,024)
and only the final ``stride`` tokens (512) are kept, so every retained token after the
opening stretch of the transcript has at least ``context - stride`` tokens of left context;
tokens inside the opening stretch have the transcript before them. A BOS token is prepended
to every window, because the dictionaries were trained with one; BOS activations are never
retained. The residual stream is read at the output of decoder layer 12.

Token -> word aggregation
-------------------------
Words are joined with single spaces and tokenised as one string, so each token maps to
exactly one word. Sub-word tokens are mean-pooled within a word for the residual stream and
max-pooled for sparse features (a feature is active on a word if it fires on any of its
tokens). Word surprisal is the sum of token surprisals, -log2 P(word | context), in bits.

Writes ``gemma2-2b_L12_w16k_l0-82.hdf5`` (for the default dictionary) to ``--out``; the
encoding models read it from ``results/features/``.

Run:
    python -m feature_extraction.language_model_features --transcript data/ds005574/stimuli/podcast_transcript.csv --out results/features
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

log = logging.getLogger("llm_sae")


@dataclass
class ExtractConfig:
    model_id: str = "google/gemma-2-2b"
    layer: int = 12                 # SAE hooks the OUTPUT of model.model.layers[layer]
    sae_repo: str = "google/gemma-scope-2b-pt-res"
    sae_width: str = "16k"
    sae_l0: int = 82
    context: int = 1024             # tokens of context per window
    stride: int = 512               # tokens retained from the end of each window
    dtype: str = "float32"
    device: str = "cuda:0"
    max_weight_mem_per_gpu: str = "5500MiB"
    logit_chunk: int = 128          # rows per chunk when reducing the (L, 256k) logits
    seed: int = 0


class JumpReLUSAE(nn.Module):
    """GemmaScope JumpReLU sparse autoencoder.

    Parameters are loaded from the released ``params.npz``:
    ``W_enc`` (d_model, d_sae), ``b_enc`` (d_sae,), ``W_dec`` (d_sae, d_model),
    ``b_dec`` (d_model,), ``threshold`` (d_sae,).
    """

    def __init__(self, d_model: int, d_sae: int) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) residual-stream activations -> (..., d_sae) sparse features."""
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """f: (..., d_sae) -> (..., d_model) reconstruction."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    @classmethod
    def from_hub(cls, repo: str, layer: int, width: str, l0: int,
                 device: str, dtype: torch.dtype) -> "JumpReLUSAE":
        path = hf_hub_download(
            repo_id=repo,
            filename=f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz")
        p = np.load(path)
        sae = cls(p["W_enc"].shape[0], p["W_enc"].shape[1])
        sae.load_state_dict({k: torch.tensor(v) for k, v in p.items()})
        return sae.to(device=device, dtype=dtype).eval()


def tokenise_transcript(words: list[str],
                        tok: "PreTrainedTokenizerBase") -> tuple[list[int], np.ndarray]:
    """Tokenise the space-joined transcript, keeping a token -> word index map.

    Returns
    -------
    ids : list[int], length n_tokens
    tok2word : (n_tokens,) int array mapping each token to its word index
    """
    ids: list[int] = []
    tok2word: list[int] = []
    for i, w in enumerate(words):
        piece = w if i == 0 else " " + w
        wid = tok(piece, add_special_tokens=False)["input_ids"]
        if not wid:                       # pathological empty token
            wid = tok(" ", add_special_tokens=False)["input_ids"]
        ids.extend(wid)
        tok2word.extend([i] * len(wid))
    return ids, np.asarray(tok2word, dtype=np.int32)


@torch.no_grad()
def run_model(cfg: ExtractConfig, ids: list[int],
              tok: "PreTrainedTokenizerBase") -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window forward pass.

    Returns
    -------
    resid : (n_tokens, d_model) float32 - residual stream after block ``cfg.layer``
    surprisal : (n_tokens,) float32 - -log2 P(token | left context); NaN for the very
        first token, which has no context.
    """
    dtype = getattr(torch, cfg.dtype)
    if cfg.device == "auto":
        # gemma-2-2b in float32 is ~10.4 GB and each RTX 2080 SUPER has 8.6 GB, so the
        # model is sharded across both cards rather than downcast. Turing (sm_75) has no
        # native bf16, and fp16 risks overflow in Gemma-2's soft-capped logits.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype, attn_implementation="eager",
            device_map="balanced",
            # Cap the weights well below capacity: Gemma-2's 256k vocab makes the logits
            # tensor ~1 GB per 1k-token window, and that has to fit alongside them.
            max_memory={i: cfg.max_weight_mem_per_gpu
                        for i in range(torch.cuda.device_count())}).eval()
        in_device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=dtype,
            attn_implementation="eager").to(cfg.device).eval()
        in_device = torch.device(cfg.device)
    log.info("model loaded, dtype=%s, input device=%s", dtype, in_device)
    d_model = model.config.hidden_size
    bos = tok.bos_token_id

    n = len(ids)
    resid = np.full((n, d_model), np.nan, dtype=np.float32)
    surp = np.full(n, np.nan, dtype=np.float32)

    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[cfg.layer].register_forward_hook(hook)

    start = 0
    while start < n:
        stop = min(start + cfg.stride, n)
        ctx_start = max(0, stop - cfg.context)
        window = [bos] + ids[ctx_start:stop]
        x = torch.tensor([window], device=in_device)
        # use_cache=False: Gemma-2's HybridCache is allocated on a single device and
        # breaks under a sharded device_map. Nothing here generates, so the KV cache is
        # pure overhead anyway.
        out = model(x, use_cache=False)

        h = captured["h"][0].float().cpu().numpy()           # (1 + len, d_model)

        # Surprisal without materialising a second (L, 256k) tensor: log_softmax would
        # double the ~1 GB logits allocation and OOM an 8 GB card. Reduce in row chunks
        # instead, using -log2 p = (logsumexp(row) - row[target]) / ln 2.
        logits = out.logits[0]                               # (1 + len, vocab), on GPU
        targets = torch.tensor(window[1:], device=logits.device)
        n_pred = targets.numel()
        bits = torch.empty(n_pred, dtype=torch.float32, device=logits.device)
        for c0 in range(0, n_pred, cfg.logit_chunk):
            c1 = min(c0 + cfg.logit_chunk, n_pred)
            rows = logits[c0:c1].float()
            lse = torch.logsumexp(rows, dim=-1)
            tgt = rows.gather(1, targets[c0:c1, None]).squeeze(1)
            bits[c0:c1] = (lse - tgt) / float(np.log(2.0))
        bits_np = bits.cpu().numpy()

        # Position j in `window` (j >= 1) is token ctx_start + j - 1, and its surprisal
        # is predicted from position j - 1, i.e. bits_np[j - 1].
        keep_lo = start - ctx_start + 1                      # first window pos to retain
        keep_hi = stop - ctx_start + 1
        t_idx = np.arange(ctx_start + keep_lo - 1, ctx_start + keep_hi - 1)
        resid[t_idx] = h[keep_lo:keep_hi]
        surp[t_idx] = bits_np[keep_lo - 1:keep_hi - 1]

        del out, logits, rows, bits
        torch.cuda.empty_cache()
        start = stop
        log.info("window -> %d of %d tokens", stop, n)

    handle.remove()
    del model
    torch.cuda.empty_cache()
    return resid, surp


def aggregate_to_words(tok2word: np.ndarray, n_words: int, resid: np.ndarray,
                       feats: np.ndarray, surp: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool token-level arrays to word level.

    Returns
    -------
    resid_w : (n_words, d_model)   mean over the word's tokens
    feat_max : (n_words, d_sae)    max over the word's tokens
    feat_mean : (n_words, d_sae)   mean over the word's tokens
    surp_w : (n_words,)            sum over the word's tokens (bits)
    """
    d_model, d_sae = resid.shape[1], feats.shape[1]
    resid_w = np.zeros((n_words, d_model), dtype=np.float32)
    feat_max = np.zeros((n_words, d_sae), dtype=np.float32)
    feat_mean = np.zeros((n_words, d_sae), dtype=np.float32)
    surp_w = np.zeros(n_words, dtype=np.float32)
    for w in range(n_words):
        m = tok2word == w
        resid_w[w] = resid[m].mean(0)
        feat_max[w] = feats[m].max(0)
        feat_mean[w] = feats[m].mean(0)
        surp_w[w] = surp[m].sum()
    return resid_w, feat_max, feat_mean, surp_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", type=str, default="16k")
    ap.add_argument("--l0", type=int, default=82)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto' shards the model across all visible GPUs; or e.g. cuda:0, cpu")
    ap.add_argument("--sae-device", type=str, default="cuda:0")
    ap.add_argument("--max-gpu-weights", type=str, default="5500MiB",
                    help="per-GPU cap on model weights; headroom is needed for the "
                         "~1 GB logits tensor Gemma-2's 256k vocab produces")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = ExtractConfig(layer=args.layer, sae_width=args.width, sae_l0=args.l0,
                        context=args.context, stride=args.stride, device=args.device,
                        max_weight_mem_per_gpu=args.max_gpu_weights)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    words = pd.read_csv(args.transcript)["word"].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    ids, tok2word = tokenise_transcript(words, tok)
    log.info("%d words -> %d tokens (%.3f tok/word)", len(words), len(ids), len(ids) / len(words))

    resid, surp = run_model(cfg, ids, tok)
    assert not np.isnan(resid).any(), "residual stream has gaps"

    sae = JumpReLUSAE.from_hub(cfg.sae_repo, cfg.layer, cfg.sae_width, cfg.sae_l0,
                               args.sae_device, getattr(torch, cfg.dtype))
    with torch.no_grad():
        x = torch.tensor(resid, device=args.sae_device, dtype=getattr(torch, cfg.dtype))
        feats_t = sae.encode(x)
        recon = sae.decode(feats_t)
        # Fraction of variance unexplained, the standard GemmaScope reconstruction metric.
        fvu = ((recon - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
        l0_achieved = (feats_t > 0).float().sum(-1)
        feats = feats_t.cpu().numpy().astype(np.float32)
    log.info("token-level: achieved L0 mean=%.2f median=%.1f | FVU=%.4f | nominal L0=%d",
             float(l0_achieved.mean()), float(l0_achieved.median()), float(fvu), cfg.sae_l0)

    resid_w, feat_max, feat_mean, surp_w = aggregate_to_words(
        tok2word, len(words), resid, feats, surp)

    tag = f"L{cfg.layer}_w{cfg.sae_width}_l0-{cfg.sae_l0}"
    with h5py.File(args.out / f"gemma2-2b_{tag}.hdf5", "w") as h:
        # Token-level residuals are kept so that other SAEs at this layer can be applied
        # without re-running the model: the sensitivity variants (different L0, different
        # width) differ only in the dictionary, not in the activations they encode.
        h.create_dataset("resid_token", data=resid, compression="gzip")
        h.create_dataset("surprisal_token_bits", data=surp)
        h.create_dataset("resid_word", data=resid_w, compression="gzip")
        h.create_dataset("sae_word_max", data=feat_max, compression="gzip")
        h.create_dataset("sae_word_mean", data=feat_mean, compression="gzip")
        h.create_dataset("surprisal_word_bits", data=surp_w)
        h.create_dataset("tok2word", data=tok2word)
        h.attrs.update(dict(model=cfg.model_id, layer=cfg.layer, sae_repo=cfg.sae_repo,
                            width=cfg.sae_width, nominal_l0=cfg.sae_l0,
                            context=cfg.context, stride=cfg.stride,
                            token_l0_mean=float(l0_achieved.mean()), fvu=float(fvu),
                            n_tokens=len(ids), n_words=len(words)))

    alive = (feat_max > 0)
    stats = dict(
        tag=tag, n_words=len(words), n_tokens=len(ids), d_sae=int(feat_max.shape[1]),
        token_l0_mean=float(l0_achieved.mean()), fvu=float(fvu),
        word_l0_mean=float(alive.sum(1).mean()),
        n_features_ever_active=int(alive.any(0).sum()),
        n_features_ge5_words=int((alive.sum(0) >= 5).sum()),
        n_features_ge10_words=int((alive.sum(0) >= 10).sum()),
        n_features_ge25_words=int((alive.sum(0) >= 25).sum()),
        n_features_ge50_words=int((alive.sum(0) >= 50).sum()),
        surprisal_mean_bits=float(surp_w.mean()), surprisal_sd_bits=float(surp_w.std()),
    )
    (args.out / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
