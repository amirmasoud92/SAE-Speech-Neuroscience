"""Semantic coherence of a feature or direction.

For each column of ``scores`` (one value per word), the ``k`` = 20 highest-scoring distinct
word types are taken (deduplicating to types prevents the metric from rewarding detectors of
a single word), and coherence is their mean pairwise cosine similarity in the embedding
space ``E`` (rows L2-normalised; GPT-2 XL layer 24 in the paper). Fewer than five distinct
types give NaN. ``RANK_BINS`` are the component-rank bins used to summarise principal
components.
"""
from __future__ import annotations

import numpy as np

K = 20
RANK_BINS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 300)]


def coherence(scores, E, tok, k=K):
    out = []
    for j in range(scores.shape[1]):
        s = scores[:, j]
        seen, idx = set(), []
        for i in np.argsort(-s):
            if s[i] <= 0 and idx:
                break
            if tok[i] in seen:
                continue
            seen.add(tok[i]); idx.append(i)
            if len(idx) == k:
                break
        if len(idx) < 5:
            out.append(np.nan); continue
        V = E[idx]
        C = V @ V.T
        out.append(float(C[np.triu_indices(len(idx), 1)].mean()))
    return np.array(out)
