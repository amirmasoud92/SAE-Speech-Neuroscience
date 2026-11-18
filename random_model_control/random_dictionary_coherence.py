"""Coherence of the density-matched random dictionary used as the H2 control.

Rebuilds the control exactly as ``encoding/basis_comparison_fits.py`` does, then applies the
coherence metric (``evaluate_autoencoders.coherence_block``) and the random-word floor.
Stimulus features only. Writes ``controls/random_model/h2_random_dict_coherence.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from capacity_matching import matched_capacity as M
from random_model_control.evaluate_autoencoders import coherence_block, random_word_floor
from utils.config import load_config
from encoding.banded_ridge import random_dictionary_features
from feature_extraction.feature_bands import dense_residual

HERE = Path(__file__).parent


def main() -> None:
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    z = np.load(cfg.results("layer_and_model_battery", "_shared.npz"), allow_pickle=True)
    E, tok, wi = z["E"].astype(np.float64), z["tok"], z["wi"]
    resid = dense_residual(cfg, wi)
    real = M.sae_band(cfg)                                    # the design's 4,933 live features
    densities = (real > 0).sum(0)
    rand = random_dictionary_features(resid, densities, np.random.default_rng(cfg.seed))
    assert np.array_equal((rand > 0).sum(0), densities)      # density matched exactly
    floor = random_word_floor(E, tok, cfg)
    cb = coherence_block(rand.astype(np.float64), resid.astype(np.float64), E, tok, cfg)
    c = cb["coh"][np.isfinite(cb["coh"])]
    out = {"dictionary": "H2 random-encoder dictionary (density-matched)",
           "status": "descriptive; not in the decision rule",
           "n_live": cb["n_live"], "n_scored": int(len(c)),
           "median_coherence": float(np.median(c)), "floor": floor,
           "median_minus_floor": float(np.median(c) - floor),
           "pct_above_floor": float(100 * (c > floor).mean()),
           "trained_pca_median": cb["pca_median"],
           "pct_above_trained_pca_median": float(100 * (c > cb["pca_median"]).mean()),
           "coherence": [float(x) for x in cb["coh"]]}
    o = cfg.results("controls", "random_model")
    o.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(o / "h2_random_dict_coherence.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "coherence"}, indent=1))


if __name__ == "__main__":
    main()
