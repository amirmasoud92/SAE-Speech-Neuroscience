"""Stimulus feature bands for the banded-ridge encoding model.

Bands
-----
``sae``        sparse features passing the activity filter (active on >= 25 of 5,130 words)
``surprisal``  Gemma-2-2B word surprisal in bits, one column
``acoustic``   80-bin log-mel spectrogram averaged in 200-ms windows at 17 delays from
               -2.0 to +2.0 s (1,360 columns), identical at every lag
``lexical``    Zipf frequency, character count and word duration

Optional bands, present when a configuration names them: ``classical`` (part-of-speech,
dependency and stopword indicators from the dataset's parse, concreteness, valence, arousal,
dominance and norm-availability indicators; 85 columns) and ``syntactic`` (the dataset's
one-hot syntactic features).

The acoustic band uses the finite-impulse-response form: the mel spectrogram sampled at a
fixed grid of delays spanning the whole epoch. At every lag the model may use acoustic
information from the entire epoch, and the design matrix stays constant across lags so one
fit covers all of them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


import h5py
import numpy as np
import pandas as pd

from utils.config import Config

log = logging.getLogger("features")


@dataclass(frozen=True)
class Design:
    """Concatenated design matrix with a named slice per band."""
    X: np.ndarray                        # (n_words, n_features) float32
    slices: dict[str, slice]
    feature_ids: dict[str, np.ndarray]   # band -> identifiers of its columns

    @property
    def n_words(self) -> int:
        return self.X.shape[0]

    def band(self, name: str) -> np.ndarray:
        return self.X[:, self.slices[name]]

    def without(self, name: str) -> "Design":
        """Drop one band, for variance partitioning."""
        keep = [b for b in self.slices if b != name]
        parts, slices, ids, off = [], {}, {}, 0
        for b in keep:
            blk = self.band(b)
            parts.append(blk)
            slices[b] = slice(off, off + blk.shape[1])
            ids[b] = self.feature_ids[b]
            off += blk.shape[1]
        return Design(np.concatenate(parts, axis=1), slices, ids)

    def only(self, names: list[str]) -> "Design":
        parts, slices, ids, off = [], {}, {}, 0
        for b in names:
            blk = self.band(b)
            parts.append(blk)
            slices[b] = slice(off, off + blk.shape[1])
            ids[b] = self.feature_ids[b]
            off += blk.shape[1]
        return Design(np.concatenate(parts, axis=1), slices, ids)


# --------------------------------------------------------------------------- bands

def sae_features(cfg: Config, word_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GemmaScope features for the analysed words, filtered by activity.

    Parameters
    ----------
    word_idx : (n_words,) indices into the master transcript for the analysed words.

    Returns
    -------
    F : (n_words, n_selected) float32
    ids : (n_selected,) int32 - the SAE dictionary indices retained

    The ``min_active_words`` threshold is a property of the *stimulus* and uses no neural
    data, so applying it once over all analysed words cannot leak the target. It is fixed
    a priori in the config; the resulting count is a result, not a tuned knob.
    """
    p = cfg.sae["primary"]
    tag = f"L{p['layer']}_w{p['width']}_l0-{p['l0']}"
    f = cfg.paths.results_root / "features" / f"gemma2-2b_{tag}.hdf5"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - run feature_extraction.language_model_features first")

    key = "sae_word_max" if cfg.llm["word_pooling"]["sae"] == "max" else "sae_word_mean"
    with h5py.File(f, "r") as h:
        F = h[key][...][word_idx].astype(np.float32)

    thresh = int(cfg.sae["min_active_words"])
    n_active = (F > 0).sum(0)
    ids = np.flatnonzero(n_active >= thresh).astype(np.int32)
    if ids.size == 0:
        raise ValueError(f"no SAE feature is active on >= {thresh} words")
    log.info("sae: %d of %d dictionary features active on >= %d of %d words",
             ids.size, F.shape[1], thresh, len(word_idx))
    return np.ascontiguousarray(F[:, ids]), ids


def surprisal_feature(cfg: Config, word_idx: np.ndarray) -> np.ndarray:
    """(n_words, 1) float32 - Gemma-2-2B word surprisal in bits."""
    p = cfg.sae["primary"]
    tag = f"L{p['layer']}_w{p['width']}_l0-{p['l0']}"
    f = cfg.paths.results_root / "features" / f"gemma2-2b_{tag}.hdf5"
    with h5py.File(f, "r") as h:
        s = h["surprisal_word_bits"][...][word_idx].astype(np.float32)
    if not np.isfinite(s).all():
        raise ValueError("non-finite surprisal values")
    return s[:, None]


def dense_residual(cfg: Config, word_idx: np.ndarray) -> np.ndarray:
    """(n_words, 2304) float32 - Gemma-2-2B layer-12 residual stream, for the dense
    baseline. PCA is applied inside training folds, not here."""
    p = cfg.sae["primary"]
    tag = f"L{p['layer']}_w{p['width']}_l0-{p['l0']}"
    f = cfg.paths.results_root / "features" / f"gemma2-2b_{tag}.hdf5"
    with h5py.File(f, "r") as h:
        return h["resid_word"][...][word_idx].astype(np.float32)


def acoustic_features(cfg: Config, onsets_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log-mel spectrogram sampled at a grid of delays around each word onset.

    Returns
    -------
    A : (n_words, n_delays * 80) float32
    ids : (n_delays * 80,) object array of "d{delay_ms}_mel{bin}" labels

    The spectrogram is 80 bins x 100 Hz x 1800 s; frame ``k`` is time ``k / 100`` s with
    zero offset (verified against the audio envelope, r = 0.68 at lag 0).
    """
    with h5py.File(cfg.paths.spectrogram, "r") as h:
        S = h["vectors"][...]                       # (80, 180000) float32, log-scaled
    n_bins, n_frames = S.shape
    fs = float(cfg.bands["acoustic"]["spectrogram_hz"])
    win = int(round(cfg.bands["acoustic"]["window_ms"] / 1000.0 * fs))
    delays = np.asarray(cfg.bands["acoustic"]["delays_s"], dtype=float)

    starts0 = np.rint(np.asarray(onsets_s) * fs).astype(np.int64)
    blocks, labels = [], []
    cs = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(S, axis=1, dtype=np.float64)],
                        axis=1)
    for d in delays:
        s = starts0 + int(round(d * fs))
        # Clamp at the edges rather than dropping words: the spectrogram is silence
        # outside the narrative, so the clamped value is a faithful stand-in.
        s = np.clip(s, 0, n_frames - win)
        blocks.append(((cs[:, s + win] - cs[:, s]) / win).T.astype(np.float32))
        labels += [f"d{int(round(d*1000)):+05d}ms_mel{b:02d}" for b in range(n_bins)]

    A = np.concatenate(blocks, axis=1)
    log.info("acoustic: %d delays x %d mel bins = %d features", len(delays), n_bins,
             A.shape[1])
    return np.ascontiguousarray(A), np.asarray(labels, dtype=object)


def lexical_features(words: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n_words, 3) float32 - Zipf frequency, character count, word duration in seconds.

    Out-of-corpus tokens (13 of 5,136: proper nouns like 'Chivvis', 'Dhuey') get
    ``zipf = 0``, which is what ``wordfreq`` returns and is the correct floor: they are
    rarer than anything in the corpus.
    """
    from wordfreq import zipf_frequency

    clean = (words["word"].astype(str).str.lower()
             .str.replace(r"[^a-z']", "", regex=True))
    zipf = np.array([zipf_frequency(w, "en") if w else 0.0 for w in clean],
                    dtype=np.float32)
    nchar = clean.str.len().to_numpy(dtype=np.float32)
    dur = (words["end"] - words["start"]).to_numpy(dtype=np.float32)
    n_oov = int((zipf == 0).sum())
    if n_oov:
        log.info("lexical: %d words out of the wordfreq corpus (zipf set to 0)", n_oov)
    return (np.column_stack([zipf, nchar, dur]).astype(np.float32),
            np.asarray(["zipf_frequency", "n_characters", "word_duration_s"], dtype=object))


def syntactic_features(cfg: Config, word_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_words, n_live) binary syntax from the dataset's own annotations.

    ``stimuli/syntactic/features.hdf5`` stores a (5305, 96) binary matrix with **one row per
    spaCy token, not per word**: the release's ``code/syntactic.py`` splits each transcript
    word with spaCy's tokenizer (``there's`` -> ``there`` + ``'s``, ``middle.`` -> ``middle``
    + ``.``), and ``stimuli/syntactic/transcript.tsv`` gives each token row its
    ``word_idx``. Columns: one-hot over the tagger's fine-grained tags and the parser's
    dependency labels, plus a stopword flag.

    **Rows are pooled to words by the maximum over the word's tokens** - the rule the SAE
    band uses - so a word carries every tag, relation and stopword flag of its tokens.
    Columns never set for any word are dropped.

    Rows must therefore be pooled to words; reading them as if they were words misaligns
    every word after the first one spaCy splits.

    These are stimulus properties and involve no neural data, so nothing here is fitted.
    """
    d = cfg.paths.bids_root / "stimuli" / "syntactic"
    f = d / "features.hdf5"
    if not f.exists():
        raise FileNotFoundError(f"syntactic features not found: {f}")
    with h5py.File(f, "r") as h:
        V = h["vectors"][...]
    if V.max() > 1 or V.min() < 0:
        raise ValueError(f"syntactic vectors are not binary: [{V.min()}, {V.max()}]")
    tok_word = pd.read_csv(d / "transcript.tsv", sep="\t")["word_idx"].to_numpy()
    if len(tok_word) != len(V) or np.any(np.diff(tok_word) < 0):
        raise ValueError("syntactic token table does not match the feature rows")

    W = np.zeros((int(tok_word.max()) + 1, V.shape[1]), dtype=V.dtype)
    np.maximum.at(W, tok_word, V)
    live = W.sum(0) > 0
    ids = np.asarray([f"syn{j:02d}" for j in np.flatnonzero(live)], dtype=object)
    return W[word_idx][:, live].astype(np.float32), ids


def classical_features(cfg: Config, word_idx, words) -> tuple[np.ndarray, np.ndarray]:
    """(n_words, 85) hand-crafted linguistics: syntax plus psycholinguistic norms.

    The baseline for language-model features: the features a psycholinguist would have
    chosen.

    Columns: the 79 one-hot POS and dependency indicators of ``syntactic_features``,
    Brysbaert concreteness, Warriner valence, arousal and dominance, and two missingness
    indicators.

    **Frequency and length are deliberately not here.** They are already in the ``lexical``
    band, which is present in every configuration, so putting them in this band as well
    would make each one's leave-one-band-out unique variance cancel the other's. Leaving
    them in ``lexical`` means they are controlled identically on both sides of the sparse
    against classical comparison.

    **Coverage is not uniform and this matters for interpretation.** Concreteness covers
    83.1% of tokens; the Warriner affective norms cover only 26.3%, because they are
    predominantly content words and this stimulus is running speech. Missing entries take
    the observed mean with a missingness indicator, so a gap never
    silently becomes a zero - but a band that is mean-filled for three quarters of its
    affective columns cannot carry much, and that is a limit of the norms, not a result.
    """
    syn, syn_ids = syntactic_features(cfg, word_idx)
    nrm, nrm_ids = norms_features(cfg, words)
    return (np.column_stack([syn, nrm]).astype(np.float32),
            np.asarray(list(syn_ids) + list(nrm_ids), dtype=object))


def norms_features(cfg: Config, words) -> tuple[np.ndarray, np.ndarray]:
    """(n_words, 6) the norms columns of the classical band: concreteness, valence,
    arousal, dominance, and the two missingness indicators. Mean-filled where a word is not
    normed. Separate from the syntax so that another cohort's parse can be combined with
    exactly these columns (the film cohort)."""
    norms = cfg.paths.bids_root.parent / "norms"
    if not norms.exists():
        norms = Path(__file__).resolve().parent.parent / "data" / "norms"
    conc = pd.read_csv(norms / "brysbaert_concreteness.txt", sep="	")
    vad = pd.read_csv(norms / "warriner_vad.csv")

    cmap = dict(zip(conc.Word.astype(str).str.lower(), conc["Conc.M"].astype(float)))
    vmap = {c: dict(zip(vad.Word.astype(str).str.lower(), vad[c].astype(float)))
            for c in ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]}
    tok = [str(x).strip().lower().strip('.,!?;:\"\'') for x in words]

    cols, ids = [], []
    c_raw = np.array([cmap.get(t, np.nan) for t in tok], dtype=float)
    have_c = np.isfinite(c_raw)
    c_raw[~have_c] = np.nanmean(c_raw)
    cols.append(c_raw)
    ids.append("concreteness")

    v_raw = np.column_stack([[vmap[c].get(t, np.nan) for t in tok] for c in vmap])
    have_v = np.isfinite(v_raw).all(1)
    for j in range(v_raw.shape[1]):
        col = v_raw[:, j]
        col[~np.isfinite(col)] = np.nanmean(col)
        cols.append(col)
    ids += ["valence", "arousal", "dominance"]

    cols.append((~have_c).astype(float))
    cols.append((~have_v).astype(float))
    ids += ["concreteness_missing", "vad_missing"]

    log.info("classical: concreteness on %.1f%% of tokens, VAD on %.1f%%",
             100 * have_c.mean(), 100 * have_v.mean())
    return np.column_stack(cols).astype(np.float32), np.asarray(ids, dtype=object)


def build_design(cfg: Config, words: pd.DataFrame) -> Design:
    """Assemble every band into one design matrix.

    ``words`` must already be restricted to the analysable words and carry ``word_idx``
    (master-transcript row) and ``start``.
    """
    idx = words["word_idx"].to_numpy()
    onsets = words["start"].to_numpy(float)

    sur = surprisal_feature(cfg, idx)
    aco, aco_ids = acoustic_features(cfg, onsets)
    lex, lex_ids = lexical_features(words)

    blocks = {}
    # Driven by the config rather than assumed. `sae` stays first when present, so every
    # design keeps the same column order.
    if "sae" in cfg.bands:
        blocks["sae"] = sae_features(cfg, idx)
    blocks["surprisal"] = (sur, np.asarray(["surprisal_bits"], dtype=object))
    blocks["acoustic"] = (aco, aco_ids)
    blocks["lexical"] = (lex, lex_ids)

    # Only present when the config asks for it (config_syntactic.yaml); the main config does
    # not, so this branch cannot change the four-band design.
    if "syntactic" in cfg.bands:
        blocks["syntactic"] = syntactic_features(cfg, idx)

    # Present only when the config asks for it (config_classical.yaml).
    if "classical" in cfg.bands:
        blocks["classical"] = classical_features(cfg, idx, words["word"].to_numpy())

    parts, slices, ids, off = [], {}, {}, 0
    for name, (blk, bid) in blocks.items():
        if blk.shape[0] != len(words):
            raise ValueError(f"band {name} has {blk.shape[0]} rows, expected {len(words)}")
        if not np.isfinite(blk).all():
            raise ValueError(f"band {name} contains non-finite values")
        parts.append(blk)
        slices[name] = slice(off, off + blk.shape[1])
        ids[name] = bid
        off += blk.shape[1]

    X = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
    log.info("design: %d words x %d features (%s)", X.shape[0], X.shape[1],
             ", ".join(f"{k}={v.stop - v.start}" for k, v in slices.items()))
    return Design(X, slices, ids)
