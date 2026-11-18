"""One shuffled FineWeb token stream shared by the trained and the re-initialised model.

FineWeb ``sample-10BT``, first parquet shard. Documents are tokenised with the Gemma
tokenizer (no special tokens), joined with EOS and packed into chunks of 1,023 tokens, each
prefixed with BOS (context 1,024; the BOS position is never used for training). The chunk
order is shuffled once with the project seed.

Splits, in shuffled order:
- chunks 0 to 97,751: training (100,000,296 tokens);
- the next 2,000: held-out evaluation;
- the next 1,000: estimation of the activation normalisation.

Writes ``data/fineweb/a3_chunks.npy`` (int32, n_chunks x 1024) and a JSON manifest with the
split boundaries and the SHA-256 of the array.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SEED = 20260901
CTX = 1024
N_TRAIN, N_EVAL, N_NORM = 97_752, 2_000, 1_000
DATA = Path(__file__).resolve().parents[1] / "data" / "fineweb"


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    bos, eos = tok.bos_token_id, tok.eos_token_id
    need = (N_TRAIN + N_EVAL + N_NORM) * (CTX - 1)
    pf = pq.ParquetFile(DATA / "sample" / "10BT" / "000_00000.parquet")
    stream, n_tok, n_docs = [], 0, 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        for i in range(0, len(texts), 1000):
            enc = tok(texts[i:i + 1000], add_special_tokens=False)["input_ids"]
            for ids in enc:
                stream.append(np.asarray(ids + [eos], dtype=np.int32))
                n_tok += len(ids) + 1
                n_docs += 1
            if n_tok >= need * 1.02:
                break
        print(f"row group {rg}: {n_docs:,} documents, {n_tok / 1e6:.1f} M tokens", flush=True)
        if n_tok >= need * 1.02:
            break
    flat = np.concatenate(stream)
    n_chunks = len(flat) // (CTX - 1)
    chunks = flat[: n_chunks * (CTX - 1)].reshape(n_chunks, CTX - 1)
    rng = np.random.default_rng(SEED)
    chunks = chunks[rng.permutation(n_chunks)][: N_TRAIN + N_EVAL + N_NORM]
    out = np.concatenate([np.full((len(chunks), 1), bos, np.int32), chunks], 1)
    np.save(DATA / "a3_chunks.npy", out)
    sha = hashlib.sha256(out.tobytes()).hexdigest()
    man = {"source": "HuggingFaceFW/fineweb sample-10BT 000_00000.parquet",
           "tokenizer": "google/gemma-2-2b", "seed": SEED, "context": CTX,
           "bos_position_excluded": True, "documents_used": n_docs,
           "n_chunks": int(len(out)), "train": [0, N_TRAIN],
           "eval": [N_TRAIN, N_TRAIN + N_EVAL],
           "norm": [N_TRAIN + N_EVAL, N_TRAIN + N_EVAL + N_NORM],
           "train_tokens": N_TRAIN * (CTX - 1), "sha256": sha}
    json.dump(man, open(DATA / "a3_chunks.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
