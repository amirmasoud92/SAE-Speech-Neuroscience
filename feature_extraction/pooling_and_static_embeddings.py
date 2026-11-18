"""Word-level arms for the static-embedding control and the pooling analysis.

CPU only; reads cached weights and runs no forward pass.

* **Static embeddings.** The transcript is re-tokenised exactly as at extraction
  (``language_model_features.tokenise_transcript``); the token count and token-to-word map
  must equal those stored with the primary features. Each token takes its row of
  Gemma-2-2B's input embedding matrix, mean-pooled over the word's tokens.
* **Pooling.** Dense arms pool the stored layer-12 token residuals by mean, max and last
  token. Sparse arms apply the GemmaScope encoder (layer 12, width 16,384, L0 = 82) to the
  same residuals and pool the codes the same three ways, on the 4,933 live columns. Max
  pooling must reproduce the stored ``sae_word_max`` and mean pooling the stored
  ``resid_word``.

Writes ``controls/features/word_arms.npz`` (rows = transcript words).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.config import load_config

HERE = Path(__file__).parent


def pool(tok2word, n_words, X, how):
    starts = np.flatnonzero(np.r_[True, np.diff(tok2word) != 0])
    assert len(starts) == n_words and np.all(np.diff(tok2word) >= 0)
    if how == "mean":
        return np.add.reduceat(X, starts, axis=0) / np.diff(np.r_[starts, len(X)])[:, None]
    if how == "max":
        return np.maximum.reduceat(X, starts, axis=0)
    if how == "last":
        return X[np.r_[starts[1:], len(X)] - 1]
    raise ValueError(how)


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    from feature_extraction.language_model_features import tokenise_transcript
    from feature_extraction.feature_bands import sae_features
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    out = cfg.results("controls", "features")
    out.mkdir(parents=True, exist_ok=True)
    f = cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5"
    with h5py.File(f, "r") as h:
        R = h["resid_token"][...].astype(np.float32)
        t2w = h["tok2word"][...]
        stored_max = h["sae_word_max"][...]
        stored_mean_resid = h["resid_word"][...]
        n_words = int(h.attrs["n_words"])
    gate = {}

    # ---- static input embeddings ------------------------------------------------------
    words = pd.read_csv(cfg.paths.bids_root / "stimuli" / "podcast_transcript.csv")["word"]
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    ids, t2w_new = tokenise_transcript(words.astype(str).tolist(), tok)
    gate["tokens_match"] = bool(len(ids) == len(t2w) and np.array_equal(t2w_new, t2w))
    gate["n_tokens"] = len(ids)
    assert gate["tokens_match"], "re-tokenisation does not reproduce the stored token map"
    idx = json.load(open(hf_hub_download("google/gemma-2-2b",
                                         "model.safetensors.index.json")))["weight_map"]
    shard = hf_hub_download("google/gemma-2-2b", idx["model.embed_tokens.weight"])
    with safe_open(shard, "pt") as s:
        Ew = s.get_tensor("model.embed_tokens.weight")
    E = Ew[np.asarray(ids)].float().numpy()
    del Ew
    static = pool(t2w, n_words, E, "mean")
    gate["static_shape"] = list(static.shape)

    # ---- dense pooling ------------------------------------------------------------------
    dense = {h_: pool(t2w, n_words, R, h_) for h_ in ("mean", "max", "last")}
    gate["dense_mean_max_abs_err"] = float(np.abs(dense["mean"] - stored_mean_resid).max())

    # ---- sparse codes at token level ---------------------------------------------------
    p = np.load(hf_hub_download(cfg.sae["repo"],
                                "layer_12/width_16k/average_l0_82/params.npz"))
    pre = R @ p["W_enc"] + p["b_enc"]
    F = np.where(pre > p["threshold"], np.maximum(pre, 0), 0).astype(np.float32)
    del pre
    gate["token_l0_mean"] = float((F > 0).sum(1).mean())
    from encoding.fit_encoding_model import analysed_words
    wi = analysed_words(cfg).reset_index(drop=True)["word_idx"].to_numpy()
    _, live = sae_features(cfg, wi)
    sae = {h_: pool(t2w, n_words, F, h_)[:, live] for h_ in ("mean", "max", "last")}
    # JumpReLU is discontinuous at its threshold, so a pre-activation within float rounding
    # of it can land on either side on CPU and GPU. The gate is the relative error on
    # entries active in both, with any such threshold ties counted and reported.
    ref = stored_max[:, live]
    a, b = sae["max"] > 0, ref > 0
    both = a & b
    gate["sae_active_disagreements"] = int((a != b).sum())
    gate["sae_active_entries"] = int(b.sum())
    gate["sae_max_coactive_rel_err"] = float((np.abs(sae["max"][both] - ref[both])
                                              / ref[both]).max())
    assert gate["sae_max_coactive_rel_err"] < 1e-4, gate
    assert gate["sae_active_disagreements"] <= 1e-5 * gate["sae_active_entries"], gate
    gate["n_live"] = int(len(live))
    np.savez(out / "word_arms.npz", static=static.astype(np.float32),
             **{f"dense_{k}": v.astype(np.float32) for k, v in dense.items()},
             **{f"sae_{k}": v.astype(np.float32) for k, v in sae.items()}, live=live)
    json.dump(gate, open(out / "word_arms_gate.json", "w"), indent=1)
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
