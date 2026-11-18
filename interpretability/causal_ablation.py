"""Per-feature causal necessity in Gemma-2-2B.

For each of the 300 features of the fixed pool, at every token where the feature is active,
its contribution ``a_f(t) * W_dec[f]`` is subtracted from the true layer-12 residual and the
rest of the model is run. The causal score is the change in the model's prediction of the
next token: the Kullback-Leibler divergence of the next-token distribution and the change in
the log-probability of the token that actually occurred.

Only the final position of each window is read: the final norm is hooked, the position
gathered, and the output head applied to it alone, with Gemma-2's logit soft-capping applied
explicitly; ``assert_readout_matches`` requires this readout to reproduce
``model(...).logits``. Windowing follows ``language_model_features.run_model`` exactly.

The pool is read from ``causal/a11_pool_raw_ids.npy``: the ``pool_ids`` array that
``encoding/basis_comparison_fits.py`` saves for the fixed pool.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocessing import neural_targets as D
from utils.config import load_config
from feature_extraction.language_model_features import JumpReLUSAE, tokenise_transcript
from utils.run_logging import setup_logging

log = logging.getLogger("a11")

MODEL_ID = "google/gemma-2-2b"
LAYER = 12
CONTEXT, STRIDE = 1024, 512


class Ablator:
    """Runs the model with a per-batch-element edit to the layer-12 residual."""

    def __init__(self, model, layer: int):
        self.model, self.softcap = model, model.config.final_logit_softcapping
        self.edit = None          # (B,) positions and (B, d_model) deltas, or None
        self.norm_out = None
        model.model.layers[layer].register_forward_hook(self._edit_hook)
        model.model.norm.register_forward_hook(self._norm_hook)

    def _edit_hook(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.edit is not None:
            pos, delta = self.edit
            h = h.clone()
            h[torch.arange(h.shape[0], device=h.device), pos] += delta.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _norm_hook(self, _m, _i, out):
        self.norm_out = out[0] if isinstance(out, tuple) else out

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Soft-capped logits from final-norm hidden states."""
        with torch.no_grad():
            z = self.model.lm_head(h).float()
            if self.softcap:
                z = self.softcap * torch.tanh(z / self.softcap)
        return z

    def norm_states(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) final-norm states for a clean forward."""
        self.edit = None
        with torch.no_grad():
            self.model.model(ids)
        return self.norm_out

    def logits_at(self, ids: torch.Tensor, pos: torch.Tensor,
                  delta: torch.Tensor | None) -> torch.Tensor:
        """(B, vocab) soft-capped logits at ``pos``, with the layer-12 edit applied."""
        self.edit = None if delta is None else (pos, delta)
        with torch.no_grad():
            self.model.model(ids)                       # decoder only; no lm_head
            h = self.norm_out[torch.arange(ids.shape[0], device=ids.device), pos]
            z = self.model.lm_head(h).float()
            if self.softcap:
                z = self.softcap * torch.tanh(z / self.softcap)
        self.edit = None
        return z


def assert_readout_matches(abl: Ablator, ids: torch.Tensor, pos: torch.Tensor) -> None:
    """The hooked readout must equal the model's own logits, or the run is meaningless."""
    mine = abl.logits_at(ids, pos, None)
    with torch.no_grad():
        ref = abl.model(ids).logits[torch.arange(ids.shape[0]), pos].float()
    err = (mine - ref).abs().max().item()
    log.info("readout check: max |hooked - model.logits| = %.3e", err)
    if err > 1e-2:
        raise SystemExit(f"readout mismatch {err:.3e}; refusing to run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--batch", type=int, default=16,
                    help="sized for the 1025-token windows, not the short first one")
    ap.add_argument("--limit-features", type=int, default=None)
    ap.add_argument("--pool", default="a11_pool_raw_ids.npy",
                    help="pool file under 07_interpretation")
    ap.add_argument("--tag", default="",
                    help="suffix for the checkpoint and output files, so an extension run "
                         "cannot overwrite the completed one")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_ = setup_logging(cfg.paths.results_root / "logs", "a11", cfg.digest)
    dev = "cuda"

    pool = np.load(cfg.results("causal", args.pool))
    if args.limit_features:
        pool = pool[:args.limit_features]
    log_.info("fixed 300-feature pool: %d features", len(pool))

    words = D.load_words(cfg)
    master = pd.read_csv(cfg.paths.transcript)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # tokenise_transcript takes a list of words; a DataFrame iterates its column names
    ids, tok2word = tokenise_transcript(master["word"].astype(str).tolist(), tok)
    log_.info("%d tokens, %d master words", len(ids), len(master))

    # token-level activations of the pool features, from the cached residuals
    feat_file = cfg.paths.results_root / "features" / "gemma2-2b_L12_w16k_l0-82.hdf5"
    with h5py.File(feat_file, "r") as h:
        resid_tok = np.asarray(h["resid_token"])
    sae = JumpReLUSAE.from_hub(cfg.sae["repo"], LAYER, cfg.sae["primary"]["width"],
                               cfg.sae["primary"]["l0"], dev, torch.float32)
    with torch.no_grad():
        acts = sae.encode(torch.from_numpy(resid_tok).to(dev))[:, pool].cpu().numpy()
    W_dec = sae.W_dec.detach()[pool]                     # (P, d_model)
    del sae
    torch.cuda.empty_cache()

    analysed = set(words["word_idx"].to_numpy().tolist())
    in_scope = np.array([w in analysed for w in tok2word])
    fi, ti = np.nonzero(acts.T > 0)
    keep = in_scope[ti] & (ti + 1 < len(ids))            # need a next token to score
    fi, ti = fi[keep], ti[keep]
    log_.info("%d (feature, token) ablations over analysed words", len(fi))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    abl = Ablator(model, LAYER)
    bos = tok.bos_token_id

    # which window owns each token, and at what position inside it
    owner = np.full(len(ids), -1, np.int64)
    wpos = np.full(len(ids), -1, np.int64)
    windows, start = [], 0
    while start < len(ids):
        stop = min(start + STRIDE, len(ids))
        ctx = max(0, stop - CONTEXT)
        w = len(windows)
        windows.append((ctx, stop))
        t = np.arange(start, stop)
        owner[t], wpos[t] = w, t - ctx + 1               # +1 for the BOS prefix
        start = stop
    log_.info("%d sliding windows", len(windows))

    dlp = np.full(len(fi), np.nan, np.float32)
    kl = np.full(len(fi), np.nan, np.float32)
    ckpt = cfg.results("causal", f"a11_checkpoint{args.tag}.npz")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done_windows = set()
    if ckpt.exists() and not args.restart:
        z = np.load(ckpt, allow_pickle=True)
        if len(z["dlp"]) == len(fi):
            dlp, kl = z["dlp"], z["kl"]
            done_windows = set(z["windows"].tolist())
            log_.info("resumed from checkpoint: %d windows already done",
                      len(done_windows))
        else:
            log_.info("checkpoint has %d rows, need %d - ignoring it",
                      len(z["dlp"]), len(fi))
    checked, t0, done = False, time.perf_counter(), 0

    for w, (ctx, stop) in enumerate(windows):
        sel = np.nonzero(owner[ti] == w)[0]
        if not len(sel) or w in done_windows:
            continue
        win = torch.tensor([[bos] + ids[ctx:stop]], device=dev)
        clean_norm = abl.norm_states(win).clone()      # (1, L, d_model), ~9 MB
        if not checked:
            assert_readout_matches(abl, win, torch.tensor([win.shape[1] - 2], device=dev))
            checked = True
        for b0 in range(0, len(sel), args.batch):
            idx = sel[b0:b0 + args.batch]
            pos = torch.tensor(wpos[ti[idx]], device=dev, dtype=torch.long)
            rep = win.expand(len(idx), -1)
            a = torch.tensor(acts[ti[idx], fi[idx]], device=dev, dtype=torch.float32)
            delta = -a[:, None] * W_dec[fi[idx]]
            z_abl = abl.logits_at(rep, pos, delta)
            z_cln = abl.head(clean_norm[0, pos])
            tgt = torch.tensor(np.asarray(ids)[ti[idx] + 1], device=dev, dtype=torch.long)
            lp_a = torch.log_softmax(z_abl, -1)
            lp_c = torch.log_softmax(z_cln, -1)
            dlp[idx] = (lp_a.gather(1, tgt[:, None]).squeeze(1)
                        - lp_c.gather(1, tgt[:, None]).squeeze(1)).cpu().numpy()
            kl[idx] = (lp_c.exp() * (lp_c - lp_a)).sum(-1).cpu().numpy()
            done += len(idx)
            if (b0 // args.batch) % 40 == 0 and b0:
                el0 = time.perf_counter() - t0
                log_.info("  window %d/%d: %d/%d of this window, %.1f abl/s",
                          w + 1, len(windows), b0 + len(idx), len(sel),
                          done / max(el0, 1e-9))
        np.savez_compressed(ckpt, dlp=dlp, kl=kl,
                            windows=np.array(sorted(done_windows | {w})))
        done_windows.add(w)
        el = time.perf_counter() - t0
        log_.info("window %d/%d: %d done, %.0f ablations/s, ETA %.1f min",
                  w + 1, len(windows), done, done / el, (len(fi) - done) / (done / el) / 60)

    out = pd.DataFrame({"feature": pool[fi], "token": ti, "word": tok2word[ti],
                        "activation": acts[ti, fi], "delta_logprob": dlp, "kl": kl})
    p = cfg.results("causal", f"a11_ablations{args.tag}.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    per = (out.groupby(["feature", "word"])[["delta_logprob", "kl"]].mean()
              .groupby("feature").mean().reset_index()
              .rename(columns={"delta_logprob": "causal_dlp", "kl": "causal_kl"}))
    per["n_words"] = out.groupby("feature")["word"].nunique().values
    per.to_csv(cfg.results("causal", f"a11_feature_scores{args.tag}.csv"),
               index=False)
    log_.info("causal_dlp: median %+.5f  causal_kl: median %.5f over %d features",
              per.causal_dlp.median(), per.causal_kl.median(), len(per))
    print("written:", p)


if __name__ == "__main__":
    main()
