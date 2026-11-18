"""Stimulus representations for the Brain Treebank films.

One HDF5 per film, rows = the film's whitespace words. No neural data.

* **Gemma-2-2B layer 12 and GemmaScope (width 16,384, L0 = 82)** by the primary pipeline's own
  functions (``tokenise_transcript``, ``run_model``, ``JumpReLUSAE``): windows of 1,024 tokens,
  stride 512, BOS, float32; residual mean-pooled, sparse features max-pooled, surprisal summed.
* **GPT-2 XL ``hidden_states[24]``** as in the Podcast dataset's release code with a 32-token
  context: each word tokenised as ``" " + word``; each token embedded at the last position of a
  33-token window (itself and its 32 predecessors), left-padded with id 0 and no attention
  mask; mean-pooled per word. ``--check-podcast`` first reproduces the release's own rows.
* **Syntax** as in the dataset's release code: spaCy ``en_core_web_lg``; one-hot fine tags and
  dependency labels and the stopword flag, pooled to words by the maximum over their tokens.
"""
from __future__ import annotations

import argparse
import json
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

log = logging.getLogger("film_stimulus_features")
GPT2_CTX = 32
GPT2_LAYER = 24


# ------------------------------------------------------------------------------ Gemma

def gemma_film(words: list[str], tok, xcfg: ExtractConfig, sae: JumpReLUSAE) -> dict:
    ids, t2w = tokenise_transcript(words, tok)
    resid, surp = run_model(xcfg, ids, tok)
    if np.isnan(resid).any():
        raise RuntimeError("residual stream has gaps")
    with torch.no_grad():
        feats = []
        for c0 in range(0, len(resid), 4096):
            x = torch.tensor(resid[c0:c0 + 4096], device=xcfg.device)
            feats.append(sae.encode(x).cpu().numpy().astype(np.float32))
        F = np.concatenate(feats)
    n = len(words)
    starts = np.flatnonzero(np.r_[True, np.diff(t2w) != 0])
    surp = np.nan_to_num(surp, nan=0.0)
    return {"resid_word": pool(t2w, n, resid, "mean"),
            "sae_word_max": pool(t2w, n, F, "max"),
            "surprisal_word_bits": np.add.reduceat(surp.astype(np.float64), starts).astype(
                np.float32),
            "n_tokens": len(ids), "token_l0": float((F > 0).sum(1).mean())}


# ---------------------------------------------------------------------------- GPT-2 XL

def gpt2_tokens(words: list[str], tok) -> tuple[list[int], np.ndarray]:
    ids, t2w = [], []
    for i, w in enumerate(words):
        t = tok.convert_tokens_to_ids(tok.tokenize(" " + w))
        ids += t
        t2w += [i] * len(t)
    return ids, np.asarray(t2w)


@torch.no_grad()
def gpt2_layer24(ids: list[int], model, device, batch: int = 128) -> np.ndarray:
    """``llm.multi_batch``: row i = tokens[max(0, i-32) : i+1], left-padded with 0."""
    n = len(ids)
    data = torch.zeros((n, GPT2_CTX + 1), dtype=torch.long)
    for i in range(n):
        ex = ids[max(0, i - GPT2_CTX): i + 1]
        data[i, -len(ex):] = torch.tensor(ex)
    out = np.empty((n, model.config.n_embd), dtype=np.float32)
    for b0 in range(0, n, batch):
        o = model(data[b0:b0 + batch].to(device), output_hidden_states=True)
        out[b0:b0 + batch] = o.hidden_states[GPT2_LAYER][:, -1, :].float().cpu().numpy()
    return out


def check_podcast(cfg, model, tok, device, n_tok: int = 600) -> dict:
    """Reproduce the release's own GPT-2 XL layer-24 rows for the first ``n_tok`` tokens."""
    g = cfg.paths.bids_root / "stimuli" / "gpt2-xl"
    words = pd.read_csv(cfg.paths.bids_root / "stimuli" / "podcast_transcript.csv")["word"]
    ids, _ = gpt2_tokens(words.astype(str).tolist(), tok)
    ref_t = pd.read_csv(g / "transcript.tsv", sep="\t")
    assert ref_t.token_id.tolist()[:n_tok] == ids[:n_tok], "tokenisation differs"
    mine = gpt2_layer24(ids, model, device)[:n_tok]
    with h5py.File(g / "features.hdf5", "r") as h:
        ref = h[f"layer-{GPT2_LAYER}"][:n_tok]
    d = np.abs(mine - ref).max() / ref.std()
    r = float(np.corrcoef(mine.ravel(), ref.ravel())[0, 1])
    return {"n_tokens": n_tok, "max_abs_diff_over_sd": float(d), "r": r}


# ------------------------------------------------------------------------------ syntax

def syntax_film(words: list[str], nlp) -> tuple[np.ndarray, list[str]]:
    from sklearn.preprocessing import LabelBinarizer
    from spacy.tokens import Doc
    tagE = LabelBinarizer().fit(nlp.get_pipe("tagger").labels)
    depE = LabelBinarizer().fit(nlp.get_pipe("parser").labels)
    toks, t2w = [], []
    for i, w in enumerate(words):
        for t in nlp.tokenizer(str(w) + " "):
            toks.append(t)
            t2w.append(i)
    doc = nlp(Doc(nlp.vocab, words=[t.text for t in toks],
                  spaces=[t.whitespace_ == " " for t in toks]))
    a = tagE.transform([t.tag_ for t in doc])
    b = depE.transform([t.dep_ for t in doc])
    c = np.array([[t.is_stop] for t in doc], dtype=np.int64)
    V = np.hstack([a, b, c])
    names = ([f"tag={x}" for x in tagE.classes_] + [f"dep={x}" for x in depE.classes_]
             + ["stop"])
    W = np.zeros((len(words), V.shape[1]), dtype=np.int8)
    np.maximum.at(W, np.asarray(t2w), V.astype(np.int8))
    return W, names


# -------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--films", nargs="*", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    wdir = cfg.results("film_replication", "words")
    out = cfg.results("film_replication", "features")
    out.mkdir(parents=True, exist_ok=True)
    films = args.films or sorted(p.stem for p in wdir.glob("*.pkl"))
    todo = [f for f in films if not (out / f"{f}.hdf5").exists()]
    log.info("%d films, %d to do", len(films), len(todo))
    if not todo:
        return
    words = {f: pd.read_pickle(wdir / f"{f}.pkl")["word"].astype(str).tolist() for f in todo}

    # GPT-2 XL first (smaller), with the release-reproduction check
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gtok = AutoTokenizer.from_pretrained("gpt2-xl")
    gmod = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(args.device).eval()
    chk = out / "gpt2xl_release_check.json"
    if not chk.exists():
        c = check_podcast(cfg, gmod, gtok, args.device)
        chk.write_text(json.dumps(c, indent=2))
        log.info("GPT-2 XL release check: %s", c)
        if c["max_abs_diff_over_sd"] > 1e-2:
            raise SystemExit("GPT-2 XL procedure does not reproduce the release - stopping")
    g2 = {}
    for f in todo:
        ids, t2w = gpt2_tokens(words[f], gtok)
        g2[f] = pool(t2w, len(words[f]), gpt2_layer24(ids, gmod, args.device), "mean")
        log.info("GPT-2 XL %s: %d words, %d tokens", f, len(words[f]), len(ids))
    del gmod
    torch.cuda.empty_cache()

    import spacy
    nlp = spacy.load("en_core_web_lg")

    from transformers import AutoTokenizer as AT
    xcfg = ExtractConfig(device=args.device)
    tok = AT.from_pretrained(xcfg.model_id)
    sae = JumpReLUSAE.from_hub(xcfg.sae_repo, xcfg.layer, xcfg.sae_width, xcfg.sae_l0,
                               args.device, torch.float32)
    for f in todo:
        G = gemma_film(words[f], tok, xcfg, sae)
        S, names = syntax_film(words[f], nlp)
        with h5py.File(out / f"{f}.hdf5", "w") as h:
            h.create_dataset("resid_word", data=G["resid_word"], compression="gzip")
            h.create_dataset("sae_word_max", data=G["sae_word_max"], compression="gzip")
            h.create_dataset("surprisal_word_bits", data=G["surprisal_word_bits"])
            h.create_dataset("gpt2xl_l24_word", data=g2[f], compression="gzip")
            h.create_dataset("syntax_word", data=S, compression="gzip")
            h.create_dataset("syntax_names", data=np.array(names, dtype="S"))
            h.attrs.update(dict(film=f, n_words=len(words[f]), gemma_tokens=G["n_tokens"],
                                token_l0_mean=G["token_l0"], layer=xcfg.layer,
                                sae=f"{xcfg.sae_repo} L{xcfg.layer} w{xcfg.sae_width} "
                                    f"l0-{xcfg.sae_l0}", context=xcfg.context,
                                stride=xcfg.stride, gpt2_context=GPT2_CTX,
                                spacy_model=f"en_core_web_lg {nlp.meta['version']}"))
        log.info("%s: %d words, %d Gemma tokens, token L0 %.1f", f, len(words[f]),
                 G["n_tokens"], G["token_l0"])


if __name__ == "__main__":
    main()
