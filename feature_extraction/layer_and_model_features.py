"""Stimulus features at every layer of Gemma-2-2B and for two further base models.

The procedure is that of ``feature_extraction/language_model_features.py``: the same
tokenisation of the space-joined transcript, 1,024-token windows with a 512-token stride and
a BOS token, the residual stream at the output of decoder layer L, mean pooling of the
residual and max pooling of sparse features over a word's tokens. Every requested layer is
hooked in one forward pass, and only the transformer body is run because no logits are
needed.

Dictionaries:

* Gemma-2-2B, layers 0-25: the GemmaScope residual dictionary of width 16,384 whose
  nominal L0 is closest to 100 (ties to the lower L0), used as released;
* Gemma-2-9B, layers 9, 20 and 31: its GemmaScope residual dictionaries (bfloat16 weights,
  residual cast to float32 before the encoder);
* Llama-3.1-8B, layers 8, 16 and 24: the Llama Scope dictionaries, applied both under
  their released JumpReLU rule and under the TopK-50 rule with which they were trained.

Layer 12 of Gemma-2-2B re-extracted here must reproduce the cached primary arrays; the check
is written to ``extraction_check.json`` and a failure stops the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import load_config
from feature_extraction.language_model_features import JumpReLUSAE, tokenise_transcript

log = logging.getLogger("layer_sweep")

MODELS = {
    "gemma2-2b": dict(model_id="google/gemma-2-2b", sae_repo="google/gemma-scope-2b-pt-res",
                      dtype="float32"),
    "gemma2-9b": dict(model_id="google/gemma-2-9b", sae_repo="google/gemma-scope-9b-pt-res",
                      dtype="bfloat16"),
    # Llama Scope, residual stream, 8x (32,768 features), trained with TopK k = 50
    "llama3.1-8b": dict(model_id="meta-llama/Llama-3.1-8B",
                        sae_repo="fnlp/Llama3_1-8B-Base-LXR-8x", dtype="bfloat16"),
}


class LlamaScopeSAE:
    """A Llama Scope residual SAE as SAELens v5.4.1's ``llama_scope`` loader runs it.

    The input is scaled by sqrt(d_model) / ``dataset_average_activation_norm.in`` (SAELens's
    ``norm_scaling_factor``, 5.278 at layer 16); pre = x_scaled W_enc^T + b_enc; a feature is
    relu(pre) where pre exceeds the scalar ``jump_relu_threshold``, else 0. The decoder bias
    is not subtracted from the input. ``decode`` returns the reconstruction in the model's
    own units, so the FVU is comparable with GemmaScope's.
    """

    def __init__(self, repo: str, layer: int, device: str):
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        folder = f"Llama3_1-8B-Base-L{layer}R-8x"
        hp = json.load(open(hf_hub_download(repo, f"{folder}/hyperparams.json")))
        if hp["hook_point_in"] != f"blocks.{layer}.hook_resid_post" or hp["act_fn"] != "jumprelu":
            raise ValueError(f"unexpected Llama Scope SAE: {hp['hook_point_in']}, {hp['act_fn']}")
        sd = load_file(hf_hub_download(repo, f"{folder}/checkpoints/final.safetensors"))
        f32 = dict(device=device, dtype=torch.float32)
        self.W_enc = sd["encoder.weight"].to(**f32).T.contiguous()      # (d_model, d_sae)
        self.b_enc = sd["encoder.bias"].to(**f32)
        self.W_dec = sd["decoder.weight"].to(**f32).T.contiguous()      # (d_sae, d_model)
        self.b_dec = sd["decoder.bias"].to(**f32)
        self.theta = float(hp["jump_relu_threshold"])
        self.scale = float(np.sqrt(hp["d_model"]) / hp["dataset_average_activation_norm"]["in"])
        self.top_k = int(hp["top_k"])

    def encode(self, x: torch.Tensor, rule: str = "jumprelu") -> torch.Tensor:
        """``jumprelu``: the released inference rule.
        ``topk``: the rule the dictionaries were trained with - the 50 features with the
        largest pre-activation x decoder norm, valued relu(pre)."""
        pre = (x * self.scale) @ self.W_enc + self.b_enc
        if rule == "jumprelu":
            return torch.relu(pre) * (pre > self.theta)
        dn = self.W_dec.norm(dim=1)
        idx = (pre * dn).topk(self.top_k, dim=-1).indices
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(pre.gather(-1, idx)))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return (f @ self.W_dec + self.b_dec) / self.scale


def canonical_l0(repo: str, layer: int, width: str, target: int = 100) -> int:
    """GemmaScope's canonical rule: the released L0 closest to 100, ties to the lower."""
    pat = re.compile(rf"layer_{layer}/width_{width}/average_l0_(\d+)/params\.npz")
    l0s = sorted({int(m[1]) for f in HfApi().model_info(repo).siblings
                  if (m := pat.match(f.rfilename))})
    if not l0s:
        raise SystemExit(f"{repo}: no width_{width} SAE at layer {layer}")
    return min(l0s, key=lambda v: (abs(v - target), v))


@torch.no_grad()
def residuals(model_id: str, dtype: str, ids: list[int], bos: int, layers: list[int],
              context: int, stride: int, device: str) -> dict[int, np.ndarray]:
    """Token-level residual stream after each block in ``layers`` (float32 on the host)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), attn_implementation="eager")
    if model_id.startswith("meta-llama/"):
        # nothing hooked depends on later blocks, so they are dropped before the model
        # reaches the GPU
        model.model.layers = model.model.layers[:max(layers) + 1]
        model.lm_head = torch.nn.Identity()           # never called: only model.model runs
    model = model.to(device)
    model.eval()
    d = model.config.hidden_size
    n = len(ids)
    out = {L: np.full((n, d), np.nan, dtype=np.float32) for L in layers}
    cap: dict[int, torch.Tensor] = {}
    handles = [model.model.layers[L].register_forward_hook(
        lambda _m, _i, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o))
        for L in layers]
    start = 0
    while start < n:
        stop = min(start + stride, n)
        ctx_start = max(0, stop - context)
        window = [bos] + ids[ctx_start:stop]
        model.model(torch.tensor([window], device=device), use_cache=False)
        keep_lo, keep_hi = start - ctx_start + 1, stop - ctx_start + 1
        t_idx = np.arange(start, stop)
        for L in layers:
            out[L][t_idx] = cap[L][0, keep_lo:keep_hi].float().cpu().numpy()
        cap.clear()
        start = stop
        log.info("window -> %d of %d tokens", stop, n)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return out


def pool(tok2word: np.ndarray, n_words: int, X: np.ndarray, how: str) -> np.ndarray:
    """Pool token rows to words: mean or max over each word's tokens.

    ``tokenise_transcript`` emits tokens in word order, so each word's tokens are one
    contiguous run and a segment reduction is exact.
    """
    if np.any(np.diff(tok2word) < 0) or np.unique(tok2word).size != n_words:
        raise ValueError("tokens are not contiguous per word")
    starts = np.flatnonzero(np.r_[True, np.diff(tok2word) != 0])
    if how == "mean":
        cnt = np.diff(np.r_[starts, len(tok2word)])[:, None]
        return (np.add.reduceat(X.astype(np.float64), starts, axis=0) / cnt).astype(np.float32)
    return np.maximum.reduceat(X, starts, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="gemma2-2b")
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--llama-rule", choices=("jumprelu", "topk"), default="jumprelu",
                    help="Llama Scope inference rule (jumprelu as released, topk as "
                         "the declared sensitivity)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(0)
    cfg = load_config(args.config)
    spec = MODELS[args.model]
    out_dir = cfg.results("layer_and_model_battery", "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    words = pd.read_csv(cfg.paths.transcript)["word"].astype(str).tolist()
    tok = AutoTokenizer.from_pretrained(spec["model_id"])
    ids, tok2word = tokenise_transcript(words, tok)
    log.info("%d words -> %d tokens", len(words), len(ids))
    t0 = time.time()
    R = residuals(spec["model_id"], spec["dtype"], ids, tok.bos_token_id, args.layers,
                  int(cfg.llm["context_tokens"]), int(cfg.llm["stride_tokens"]), args.device)
    log.info("forward passes done in %.0f s", time.time() - t0)

    checks = {}
    if args.model == "gemma2-2b" and 12 in args.layers:
        # the re-extraction check, before any other layer is written
        with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
            ref = h["resid_token"][...]
        new = R[12]
        rel = float(np.abs(new - ref).max() / (ref.std() + 1e-12))
        corr = float(np.corrcoef(new.ravel(), ref.ravel())[0, 1])
        checks = {"layer12_resid_token_max_abs_over_sd": rel,
                  "layer12_resid_token_corr": corr, "pass": bool(rel < 1e-3)}
        (out_dir / "extraction_check.json").write_text(json.dumps(checks, indent=2))
        log.info("layer-12 reproduction: max |d|/sd %.2e, r %.9f", rel, corr)
        if not checks["pass"]:
            raise SystemExit("layer 12 does not reproduce the cached arrays - stopping")

    for L in args.layers:
        if args.model == "llama3.1-8b":
            sae = LlamaScopeSAE(spec["sae_repo"], L, args.device)
            l0, args.width = sae.top_k, "32k"
        else:
            l0 = canonical_l0(spec["sae_repo"], L, args.width)
            sae = JumpReLUSAE.from_hub(spec["sae_repo"], L, args.width, l0, args.device,
                                       torch.float32)
        with torch.no_grad():
            x = torch.tensor(R[L], device=args.device)
            f = (sae.encode(x, rule=args.llama_rule) if args.model == "llama3.1-8b"
                 else sae.encode(x))
            fvu = float(((sae.decode(f) - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
            l0_tok = float((f > 0).float().sum(-1).mean())
            F = f.cpu().numpy()
        del sae, x, f
        torch.cuda.empty_cache()
        rw = pool(tok2word, len(words), R[L], "mean")
        fw = pool(tok2word, len(words), F, "max")
        ever = np.flatnonzero((fw > 0).any(0))
        tag = "-topk" if (args.model == "llama3.1-8b" and args.llama_rule == "topk") else ""
        name = f"{args.model}{tag}_L{L}_w{args.width}_l0-{l0}.hdf5"
        with h5py.File(out_dir / name, "w") as h:
            h.create_dataset("resid_word", data=rw, compression="gzip", compression_opts=4)
            h.create_dataset("sae_word_max", data=fw[:, ever], compression="gzip",
                             compression_opts=4)
            h.create_dataset("sae_ids", data=ever)
            h.attrs.update(dict(model=spec["model_id"], layer=L, width=args.width,
                                nominal_l0=l0, token_l0_mean=l0_tok, fvu=fvu,
                                dtype=spec["dtype"], n_words=len(words)))
        log.info("layer %d: SAE L0=%d, token L0 %.1f, FVU %.3f, %d features ever active -> %s",
                 L, l0, l0_tok, fvu, len(ever), name)
        del F, fw
    log.info("done")


if __name__ == "__main__":
    main()
