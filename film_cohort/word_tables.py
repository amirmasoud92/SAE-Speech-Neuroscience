"""Brain Treebank word units and per-participant word tables (stimulus side only).

The Brain Treebank transcripts are Universal Dependencies (UD) tokens: punctuation rows are
removed and clitics are separate, untimed rows. The primary cohort's unit is the whitespace word
with its punctuation, so the same unit is rebuilt:

1. each film's sentence strings (``sentence``, in ``sentence_idx`` order) are split on
   whitespace - that word list is exactly what the language models read;
2. each whitespace word is aligned, in order, to the UD tokens it contains by comparing
   lower-cased alphanumeric characters;
3. a word's onset is the ``start`` of its first timed UD token and its end the ``end`` of its
   last; a word with no timed token stays in the language-model context but is never analysed.

Gate: at least 98% of the timed UD tokens must fall inside aligned words.

Per participant, words are joined through their onset token to a word-to-neural-sample
alignment table (one row per timed UD token with its ``neural_sample``), which places each word
on its session's neural clock. That table is an input (``BTB_ALIGNMENT``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from utils.paths import BTB_ROOT, BTB_ALIGNMENT


BTB = BTB_ROOT
TRANSCRIPTS = BTB / "transcripts" / "transcripts"
ALIGN = BTB_ALIGNMENT
EXCLUDED = {"sub_5"}                                   # no atlas registration
COVERAGE_GATE = 0.98

# per-word nuisance released with the transcripts: the mel and delta-mel are 128-bin
# vectors, everything else a scalar
MEL = ["mel", "delta_mel"]
ACOUSTIC = ["rms", "delta_rms", "pitch", "delta_pitch", "magnitude", "delta_magnitude"]
VISUAL = ["max_mean_pixel_brightness", "mean_pixel_brightness", "min_mean_pixel_brightness",
          "max_mean_pixel_difference", "min_mean_pixel_difference", "max_global_magnitude",
          "max_global_angle", "max_vector_magnitude", "max_vector_angle",
          "max_mean_magnitude", "max_median_magnitude", "face_num"]


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def film_words(film: str) -> tuple[pd.DataFrame, dict]:
    """Whitespace words of one film, aligned to its UD tokens.

    Returns a table with one row per whitespace word (``w`` = its index in the film's word
    list) and the alignment statistics.
    """
    ud = pd.read_csv(TRANSCRIPTS / film / "features.csv").sort_values("word_idx")
    ud = ud.reset_index(drop=True)
    rows, n_bad = [], 0
    for sidx, g in ud.groupby("sentence_idx", sort=True):
        sent = str(g["sentence"].iloc[0])
        toks = list(g.index)
        j = 0
        for piece in sent.split():
            target = norm(piece)
            got, acc = [], ""
            if target:
                k = j
                while k < len(toks) and len(acc) < len(target):
                    t = norm(ud.at[toks[k], "text"])
                    if not target.startswith(acc + t):
                        break
                    acc += t
                    got.append(toks[k])
                    k += 1
                if acc == target:
                    j = k
                else:                                  # resynchronise, never guess
                    got = []
                    n_bad += 1
                    for skip in range(1, 4):
                        if j + skip < len(toks) and target.startswith(
                                norm(ud.at[toks[j + skip], "text"]) or "\0"):
                            j = j + skip
                            break
            timed = [t for t in got if np.isfinite(ud.at[t, "start"])]
            rows.append({"film": film, "sentence_idx": int(sidx), "word": piece,
                         "ud_rows": got, "aligned": bool(got),
                         "onset_row": timed[0] if timed else -1,
                         "start": ud.at[timed[0], "start"] if timed else np.nan,
                         "end": ud.at[timed[-1], "end"] if timed else np.nan,
                         "speaker": ud.at[got[0], "speaker"] if got else None})
    W = pd.DataFrame(rows)
    W.insert(1, "w", np.arange(len(W)))
    timed_ud = set(ud.index[np.isfinite(ud["start"])])
    covered = {t for g in W.loc[W.aligned, "ud_rows"] for t in g} & timed_ud
    # nuisance: mean over the word's timed UD tokens
    tt = [[t for t in g if t in timed_ud] for g in W.ud_rows]
    for c in ACOUSTIC + VISUAL:
        vals = ud[c].to_numpy(float)
        W[c] = [vals[g].mean() if g else np.nan for g in tt]
    for c in MEL:
        vec = {t: np.asarray(json.loads(ud.at[t, c]), dtype=np.float64) for t in timed_ud}
        W[c] = [np.mean([vec[t] for t in g], axis=0) if g else None for g in tt]
    stats = {"film": film, "ud_tokens": len(ud), "timed_ud": len(timed_ud),
             "whitespace_words": len(W), "unaligned_words": n_bad,
             "timed_words": int(np.isfinite(W.start).sum()),
             "coverage": len(covered) / max(len(timed_ud), 1)}
    return W, stats


def all_films(out: Path) -> dict[str, pd.DataFrame]:
    """Build (or load) every film's word table and write the G1a record."""
    out.mkdir(parents=True, exist_ok=True)
    films = sorted(p.name for p in TRANSCRIPTS.iterdir() if p.is_dir())
    tabs, stats = {}, []
    for f in films:
        p = out / f"{f}.pkl"
        if p.exists():
            tabs[f] = pd.read_pickle(p)
            stats.append(json.loads((out / f"{f}.json").read_text()))
            continue
        W, s = film_words(f)
        W.to_pickle(p)
        (out / f"{f}.json").write_text(json.dumps(s, indent=2))
        tabs[f] = W
        stats.append(s)
    S = pd.DataFrame(stats)
    cov = S.timed_ud.to_numpy() @ S.coverage.to_numpy() / S.timed_ud.sum()
    S.to_csv(out / "g1a_alignment.csv", index=False)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"G1a pooled coverage {cov:.4f} (gate {COVERAGE_GATE}) ->",
          "PASS" if cov >= COVERAGE_GATE else "FAIL")
    return tabs


def subject_words(sub: str, films: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analysable-candidate words of one subject, in recording order, on the neural clock.

    ``neural_sample`` is the onset token's sample index in the session file (raw rate).
    Boundary exclusions need the recording length and happen in the neural stage.
    """
    A = pd.read_csv(ALIGN)
    A = A[(A["sub"] == sub)].dropna(subset=["neural_sample"])
    out = []
    for sess in sorted(A.sess.unique(), key=lambda s: int(s.split("trial")[1])):
        a = A[A.sess == sess]
        film = a.film.iloc[0]
        W = films[film]
        ud = pd.read_csv(TRANSCRIPTS / film / "features.csv").sort_values("word_idx")
        ud = ud.reset_index(drop=True)
        key = pd.Series(ud.index, index=ud.start.round(4))
        key = key[~key.index.duplicated(keep=False)]
        ns = pd.Series(a.neural_sample.to_numpy(), index=a.start.round(4))
        ns = ns[~ns.index.duplicated(keep=False)]
        row2ns = {int(key[s]): float(v) for s, v in ns.items() if s in key.index}
        w = W[W.onset_row >= 0].copy()
        w["neural_sample"] = w.onset_row.map(row2ns)
        w = w.dropna(subset=["neural_sample"])
        w.insert(0, "sess", sess)
        w.insert(0, "sub", sub)
        out.append(w)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    tabs = all_films(cfg.results("film_replication", "words"))
