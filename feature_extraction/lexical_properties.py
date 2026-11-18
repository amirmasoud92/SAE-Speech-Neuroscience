"""Word-level lexical properties used to profile features.

Function-word status comes from an explicit word list that is written out with the results.
Concreteness comes from Brysbaert et al. (2014; 83.1% of tokens), affective ratings from
Warriner et al. (2013; 26.3% of tokens), frequency from ``wordfreq`` and length from the
token itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wordfreq

# Closed-class English. Determiners, pronouns, prepositions, conjunctions, auxiliaries,
# particles and the commonest discourse markers of spoken narrative.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their some any no every each
i me you he him she it we us they them who whom whose which what
am is are was were be been being do does did done have has had having
will would shall should can could may might must
of in on at by for with about against between into through during before after above
below to from up down out off over under again further then once here there when where
why how all both few more most other such only own same so than too very just
and or but nor yet because as if while although though unless until since whether
not n't no nope yeah yes okay ok well um uh like right now still also even
""".split())


def word_properties(cfg, words: pd.DataFrame) -> pd.DataFrame:
    """One row per analysed word with the properties the classical band draws on."""
    tok = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in words["word"]])
    norms = Path(__file__).resolve().parent.parent / "data" / "norms"
    conc = pd.read_csv(norms / "brysbaert_concreteness.txt", sep="\t")
    vad = pd.read_csv(norms / "warriner_vad.csv")
    cmap = dict(zip(conc.Word.astype(str).str.lower(), conc["Conc.M"].astype(float)))
    vmap = {n: dict(zip(vad.Word.astype(str).str.lower(), vad[c].astype(float)))
            for n, c in [("valence", "V.Mean.Sum"), ("arousal", "A.Mean.Sum"),
                         ("dominance", "D.Mean.Sum")]}

    out = pd.DataFrame({"word": tok})
    out["is_function"] = np.isin(tok, list(FUNCTION_WORDS)).astype(float)
    out["zipf"] = [wordfreq.zipf_frequency(t, "en") for t in tok]
    out["n_char"] = [len(t) for t in tok]
    out["concreteness"] = [cmap.get(t, np.nan) for t in tok]
    for n in vmap:
        out[n] = [vmap[n].get(t, np.nan) for t in tok]
    return out


def fill_with_indicators(df: pd.DataFrame, cols) -> tuple[np.ndarray, list[str]]:
    """Mean-fill the named columns and append a missingness indicator for each, so a
    coverage gap enters the model as a gap rather than as a value of zero."""
    mats, names = [], []
    for c in cols:
        v = df[c].to_numpy(float).copy()
        m = ~np.isfinite(v)
        if m.all():
            continue
        v[m] = np.nanmean(v)
        mats.append(v)
        names.append(c)
        if m.any():
            mats.append(m.astype(float))
            names.append(f"{c}_missing")
    return np.column_stack(mats), names
