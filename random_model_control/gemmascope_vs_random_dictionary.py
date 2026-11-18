"""Coherence of GemmaScope against the density-matched random dictionary.

Mann-Whitney test and Cliff's delta between the coherence of GemmaScope's 300-feature pool and
that of the random dictionary. Stimulus features only. Writes
``controls/random_model/gemmascope_vs_h2_random_dict.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from random_model_control.evaluate_autoencoders import coherence_block, compare, random_word_floor
from utils.config import load_config

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    z = np.load(cfg.results("layer_and_model_battery", "_shared.npz"), allow_pickle=True)
    E, tok, wi = z["E"].astype(np.float64), z["tok"], z["wi"]
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
        Fg = h["sae_word_max"][...][wi].astype(np.float64)
        Rg = h["resid_word"][...][wi].astype(np.float64)
    gs = coherence_block(Fg, Rg, E, tok, cfg)
    H = json.load(open(cfg.results("controls", "random_model", "h2_random_dict_coherence.json")))
    a = gs["coh"][np.isfinite(gs["coh"])]
    b = np.asarray(H["coherence"], float)
    b = b[np.isfinite(b)]
    floor = random_word_floor(E, tok, cfg)
    cmp_ = compare(a, b, cfg.seed + 4402)
    out = {"comparison": "GemmaScope (w16k, L0 82) > H2 random-encoder dictionary",
           **{k: v for k, v in cmp_.items() if k != "passes"},
           "floor": floor,
           "gemmascope": {"median": float(np.median(a)), "median_minus_floor": float(np.median(a) - floor),
                          "pct_above_floor": float(100 * (a > floor).mean()), "n_scored": int(len(a))},
           "h2_random_dict": {"median": float(np.median(b)), "median_minus_floor": float(np.median(b) - floor),
                              "pct_above_floor": float(100 * (b > floor).mean()), "n_scored": int(len(b))},
           "status": "descriptive; changes no decision rule"}
    o = cfg.results("controls", "random_model", "gemmascope_vs_h2_random_dict.json")
    json.dump(out, open(o, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
