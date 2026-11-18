"""Cross-participant agreement within and between word types.

Participants heard the same narrative, so every token was measured once per brain. If
participants agree on which *occurrence* of a word drove IFG harder, a contextual signal
exists independently of any model.

* Each participant's series is the mean of its cleaned IFG window targets, with the acoustic,
  surprisal and lexical bands projected out; a token counts for a participant when at least
  half of its IFG electrodes are valid there, and every pair statistic uses tokens valid in
  both participants.
* Agreement is computed for every pair of the participants with IFG coverage, between types
  (the positive control) and within types.
* Nulls: occurrence order permuted within each word type for one participant of the pair
  (1,000 permutations), and a circular shift of one participant's whole series by at least
  10% of the story (1,000 shifts). Both statistics are tested against both nulls.

Writes ``within_type_signal/``.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

MIN_OCC = 4
N_PERM = 1000
ROI = "IFG"
MIN_SHIFT_FRAC = 0.10
MIN_VALID_ELEC = 0.5


def type_index(types, min_occ):
    vc = pd.Series(types).value_counts()
    keep = np.isin(types, vc[vc >= min_occ].index)
    idx = pd.factorize(pd.Series(types[keep]))[0]
    return keep, idx, int(idx.max() + 1)


def _corr(a, b, v):
    a, b = a[v], b[v]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))


def pair_stats(xa, va, xb, vb, idx, n_g):
    """(between, within) ISC for one pair over tokens valid in both subjects."""
    v = va & vb
    w = v.astype(np.float64)
    cnt = np.bincount(idx, weights=w, minlength=n_g)
    n = np.maximum(cnt, 1.0)
    ma = np.bincount(idx, weights=xa * w, minlength=n_g) / n
    mb = np.bincount(idx, weights=xb * w, minlength=n_g) / n
    return (_corr(ma, mb, cnt > 0),
            _corr(xa - ma[idx], xb - mb[idx], v))


def subject_series(cfg, roi, nuis):
    """One nuisance-residualised IFG value per token per subject, with a validity mask."""
    Y, V, subj, _ = Mt.ifg_targets(cfg, "window", roi=roi)
    names = list(dict.fromkeys(subj))
    S, SV, info = [], [], []
    for s in names:
        Ye, Ve = Y[:, subj == s], V[:, subj == s]
        w = Ve.astype(np.float64)
        n = np.maximum(w.sum(0), 1.0)
        mu = (Ye * w).sum(0) / n
        sd = np.sqrt((((Ye - mu) ** 2) * w).sum(0) / n) + 1e-12
        Z = np.where(Ve, (Ye - mu) / sd, 0.0)
        cnt = Ve.sum(1)
        x = Z.sum(1) / np.maximum(cnt, 1)
        v = cnt >= MIN_VALID_ELEC * Ve.shape[1]
        b = np.linalg.lstsq(nuis[v], x[v], rcond=None)[0]
        x = np.where(v, x - nuis @ b, 0.0)
        S.append(x)
        SV.append(v)
        info.append((s, int(Ve.shape[1]), float(v.mean())))
    return np.column_stack(S), np.column_stack(SV), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--roi", default=ROI)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--detrend", type=int, default=0, metavar="K",
                    help="sensitivity only: additionally remove K cosine/sine "
                         "pairs of slow drift over stimulus time. Drift is already removed "
                         "upstream within each CV block.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 29)

    w = analysed_words(cfg).reset_index(drop=True)
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    nuis = Mt.nuisance(cfg)
    if args.detrend:
        T = w["start"].to_numpy(float)
        T = T - T.min()
        dur = T.max()
        nuis = np.column_stack([nuis] + [f(2 * np.pi * k * T / dur)
                                         for k in range(1, args.detrend + 1)
                                         for f in (np.cos, np.sin)])
        print(f"SENSITIVITY: {args.detrend} slow-drift harmonics also removed")
    S, SV, info = subject_series(cfg, args.roi, nuis)
    n_w = S.shape[0]
    print(f"{S.shape[1]} subjects with {args.roi} electrodes: "
          + ", ".join(f"{s}({n}, {100 * f:.1f}% tokens valid)" for s, n, f in info))

    keep, idx, n_g = type_index(types, MIN_OCC)
    print(f"{n_g} word types with >= {MIN_OCC} occurrences, {int(keep.sum())} tokens "
          f"({100 * keep.mean():.1f}% of the corpus)")

    pairs = list(combinations(range(S.shape[1]), 2))
    obs = np.array([pair_stats(S[keep, a], SV[keep, a], S[keep, b], SV[keep, b], idx, n_g)
                    for a, b in pairs])
    obs_b, obs_w = obs[:, 0], obs[:, 1]
    print(f"\n{len(pairs)} subject pairs")
    print(f"  between-type ISC (positive control): mean {obs_b.mean():+.4f}, "
          f"median {np.median(obs_b):+.4f}, {int((obs_b > 0).sum())}/{len(pairs)} positive")
    print(f"  within-type  ISC                   : mean {obs_w.mean():+.4f}, "
          f"median {np.median(obs_w):+.4f}, {int((obs_w > 0).sum())}/{len(pairs)} positive")

    # null 1: permute type labels (between) / occurrences within type
    # (within). null 2: circular shift of subject b's whole series in token order.
    kpos = np.flatnonzero(keep)
    lo_sh = int(MIN_SHIFT_FRAC * n_w)
    null = {k: np.empty(args.n_perm) for k in ("perm_b", "perm_w", "circ_b", "circ_w")}
    for p in range(args.n_perm):
        acc = {k: [] for k in null}
        for a, b in pairs:
            xa, va = S[keep, a], SV[keep, a]
            xb, vb = S[keep, b], SV[keep, b]
            # between: relabel whole word types for subject b
            acc["perm_b"].append(_between_relabelled(xa, va, xb, vb, idx,
                                                     rng.permutation(n_g), n_g))
            # within: shuffle subject b's occurrences inside each word type
            order = np.lexsort((rng.random(len(idx)), idx))
            src = np.empty_like(order)
            src[np.argsort(idx, kind="stable")] = order
            acc["perm_w"].append(pair_stats(xa, va, xb[src], vb[src], idx, n_g)[1])
            # circular shift of the whole series, mask carried along
            k = int(rng.integers(lo_sh, n_w - lo_sh))
            xs, vs = np.roll(S[:, b], k)[kpos], np.roll(SV[:, b], k)[kpos]
            cb, cw = pair_stats(xa, va, xs, vs, idx, n_g)
            acc["circ_b"].append(cb)
            acc["circ_w"].append(cw)
        for k in null:
            null[k][p] = np.mean(acc[k])
        if (p + 1) % 200 == 0:
            print(f"    {p + 1}/{args.n_perm} permutations", flush=True)

    def pv(x, nl):
        return float((np.sum(nl >= x) + 1) / (args.n_perm + 1))

    res = {"between_p": pv(obs_b.mean(), null["perm_b"]),
           "within_p": pv(obs_w.mean(), null["perm_w"]),
           "between_p_circ": pv(obs_b.mean(), null["circ_b"]),
           "within_p_circ": pv(obs_w.mean(), null["circ_w"])}
    print("\n" + "=" * 86)
    for lab, o, kp, kc in [("between-type", obs_b.mean(), "perm_b", "circ_b"),
                           ("within-type ", obs_w.mean(), "perm_w", "circ_w")]:
        for tag, key in [("permutation", kp), ("circular   ", kc)]:
            nl = null[key]
            print(f"  {lab} ISC {o:+.4f}  {tag} null {nl.mean():+.4f} "
                  f"[{np.percentile(nl, 2.5):+.4f}, {np.percentile(nl, 97.5):+.4f}]  "
                  f"P = {pv(o, nl):.4f}")
    print("=" * 86)

    p_b, p_w = res["between_p"], res["within_p"]
    if p_b >= 0.05:
        v = ("CONTROL FAILED: brains do not agree on which words drive IFG. The method is "
             "broken and no part of this analysis may be interpreted.")
    elif p_w < 0.05 and res["within_p_circ"] < 0.05:
        v = ("A reliable WITHIN-WORD signal exists, surviving both the within-type "
             "permutation and the circular shift.")
    elif p_w < 0.05:
        v = ("Within-word agreement exceeds the within-type permutation but NOT the circular "
             "shift: it cannot be separated from shared slow structure.")
    else:
        v = ("No reliable within-word signal at this lag - report as a limitation, not a "
             "finding about either representation.")
    print("VERDICT:", v)

    tag = f"_detrend{args.detrend}" if args.detrend else ""
    o = cfg.results("within_type_signal")
    o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair": [f"{a}-{b}" for a, b in pairs],
                  "between": obs_b, "within": obs_w}).to_csv(
        o / f"a29a_pairs{tag}.csv", index=False)
    pd.DataFrame([{"n_subjects": S.shape[1], "n_pairs": len(pairs), "n_types": n_g,
                   "n_tokens": int(keep.sum()), "min_occ": MIN_OCC,
                   "between_isc": obs_b.mean(), "within_isc": obs_w.mean(), **res,
                   "circ_null_w_lo": float(np.percentile(null["circ_w"], 2.5)),
                   "circ_null_w_hi": float(np.percentile(null["circ_w"], 97.5)),
                   "perm_null_w_lo": float(np.percentile(null["perm_w"], 2.5)),
                   "perm_null_w_hi": float(np.percentile(null["perm_w"], 97.5)),
                   "n_perm": args.n_perm, "min_shift_frac": MIN_SHIFT_FRAC,
                   "detrend_harmonics": args.detrend, "verdict": v}]).to_csv(
        o / f"a29a_summary{tag}.csv", index=False)
    print("written:", o / f"a29a_summary{tag}.csv")


def _between_relabelled(xa, va, xb, vb, idx, perm, n_g):
    """Between-type ISC with subject b's type means assigned to permuted types."""
    v = va & vb
    w = v.astype(np.float64)
    cnt = np.bincount(idx, weights=w, minlength=n_g)
    n = np.maximum(cnt, 1.0)
    ma = np.bincount(idx, weights=xa * w, minlength=n_g) / n
    mb = np.bincount(idx, weights=xb * w, minlength=n_g) / n
    has = cnt > 0
    mbp, hp = mb[perm], has[perm]
    return _corr(ma, mbp, has & hp)


if __name__ == "__main__":
    main()
