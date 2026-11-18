"""Brain Treebank replication of the matched-capacity comparisons.

Labels used in the outputs:

* gates - ``G0`` coverage (participants with IFG contacts), ``G1`` word alignment and an
  alignment check in STG, ``G2`` positive control (sparse and dense arms predict IFG);
* hypotheses, Holm-corrected as one family - ``R1`` sparse code (85 components) against the
  classical band; ``R2`` sparse minus dense at 300 components against the equivalence margin;
  ``R3`` the within-type sparse-minus-dense difference and ``R3b`` the word-type interaction;
  ``R4`` sparse code against GPT-2 XL at 300 components;
* sensitivity - ``S1`` leave one participant out, ``S3`` Laplacian-referenced contacts only.

Every fit, score and statistic is the primary cohort's code (``matched_capacity.oof``,
``best_lambda_r``, ``colcorr``, ``cluster_boot``, ``zpca`` and ``decomposition.decompose``). What
the cohort forces: ridge is fitted per participant (participants watched different films), PCA
runs over each participant's analysed words, and the nuisance set is built from the released
per-word mel-spectral vectors because no audio is released.

Per participant, all analysed contacts are fitted at once (column-wise independent); each arm's
penalty is then chosen per region on the mean held-out r over that region's contacts of all
participants. Writes ``film_replication/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from utils.config import load_config
from feature_extraction.feature_bands import lexical_features, norms_features
from word_types.decomposition import decompose

N_BOOT = 10000
MIN_ACTIVE = 25
MIN_OCC, SENS_OCC = 4, (3, 6)
PODCAST_MARGIN = 0.011062                        # used only if this cohort's margin is <= 0
N_ACOUSTIC_PC = 100
ACOUSTIC = ["rms", "delta_rms", "pitch", "delta_pitch", "magnitude", "delta_magnitude"]
VISUAL = ["max_mean_pixel_brightness", "mean_pixel_brightness", "min_mean_pixel_brightness",
          "max_mean_pixel_difference", "min_mean_pixel_difference", "max_global_magnitude",
          "max_global_angle", "max_vector_magnitude", "max_vector_angle",
          "max_mean_magnitude", "max_median_magnitude", "face_num"]
EXACT_ARMS = ("saek", "classical", "sae300", "gemma300", "gpt2_300")
A28_ARMS = ("SAE", "prior", "gemma", "sae300")   # word-type decomposition arms, exact=False


# ------------------------------------------------------------------------------ inputs

def load_subject(cfg, sub: str, base: Path):
    z = np.load(base / "targets" / f"{sub}.npz", allow_pickle=True)
    W = pd.DataFrame.from_records(z["words"])
    C = pd.DataFrame.from_records(z["contacts"])
    Y, V = z["Y"].astype(np.float64).mean(2), z["V"].all(2)
    feats = {k: [] for k in ("resid", "sae", "surp", "gpt2", "syn")}
    for film, g in W.groupby("film", sort=False):
        with h5py.File(base / "features" / f"{film}.hdf5", "r") as h:
            w = g["w"].to_numpy()
            feats["resid"].append((g.index, h["resid_word"][...][w]))
            feats["sae"].append((g.index, h["sae_word_max"][...][w]))
            feats["surp"].append((g.index, h["surprisal_word_bits"][...][w]))
            feats["gpt2"].append((g.index, h["gpt2xl_l24_word"][...][w]))
            feats["syn"].append((g.index, h["syntax_word"][...][w]))
    F = {}
    for k, parts in feats.items():
        shape = (len(W),) + parts[0][1].shape[1:]
        a = np.zeros(shape, dtype=np.float64)
        for idx, x in parts:
            a[np.asarray(idx)] = x
        F[k] = a
    sae_ids = np.flatnonzero((F["sae"] > 0).sum(0) >= MIN_ACTIVE)   # dictionary ids
    sae = F["sae"][:, sae_ids]
    syn = F["syn"][:, F["syn"].sum(0) > 0]
    nrm, _ = norms_features(cfg, W["word"].to_numpy())
    classical = np.column_stack([syn, nrm]).astype(np.float64)

    lex, _ = lexical_features(W[["word", "start", "end"]])
    A = np.column_stack([np.log10(np.maximum(z["mel"], 1e-10)), z["delta_mel"]])
    A = A - A.mean(0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    spk = W["speaker"].astype(str).to_numpy()
    change = np.r_[1.0, (spk[1:] != spk[:-1]).astype(float)]
    change[np.r_[True, W.sess.to_numpy()[1:] != W.sess.to_numpy()[:-1]]] = 1.0
    sess = pd.get_dummies(W["sess"]).to_numpy(np.float64)
    nuis = np.column_stack([F["surp"][:, None], lex, A @ Vt[:N_ACOUSTIC_PC].T,
                            W[ACOUSTIC + VISUAL].to_numpy(np.float64), change, sess])
    if not np.isfinite(nuis).all():
        raise ValueError(f"{sub}: non-finite nuisance")
    Yz = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    out = {"W": W, "C": C, "Y": Yz, "V": V, "nuis": nuis, "sae": sae, "resid": F["resid"],
           "gpt2": F["gpt2"], "classical": classical, "sae_ids": sae_ids,
           # the nuisance pieces as separate bands for the banded model (H1)
           "parts": {"surprisal": F["surp"][:, None], "lexical": np.asarray(lex, float),
                     "acoustic": np.column_stack([A + 0.0, W[ACOUSTIC].to_numpy(float)]),
                     "scene": np.column_stack([W[VISUAL].to_numpy(float), change, sess])}}
    if "Ysh" in z.files:
        ysh, vsh = z["Ysh"].astype(np.float64).mean(2), z["Vsh"].all(2)
        out["Ysh"] = (ysh - ysh.mean(0)) / (ysh.std(0, keepdims=True) + 1e-12)
        out["Vsh"] = vsh
    return out


# ------------------------------------------------------------------------------- fits

def fit_subject(sub: str, S: dict, folds, rng_seed: int) -> dict:
    """Per-lambda scores for every arm, and word-type decompositions at every lambda."""
    Y, V, nuis = S["Y"], S["V"], S["nuis"]
    k_c = S["classical"].shape[1]
    exact = {"saek": M.zpca(S["sae"], k_c), "classical": S["classical"],
             "sae300": M.zpca(S["sae"], 300), "gemma300": M.zpca(S["resid"], 300),
             "gpt2_300": M.zpca(S["gpt2"], 300)}
    R = {"k_c": k_c, "n_live_sae": int(S["sae"].shape[1])}
    for name, X in exact.items():
        P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=True)
        R[f"r_{name}"] = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
        print(f"    {sub} {name:10s} k={X.shape[1]:4d} best mean r "
              f"{R[f'r_{name}'].mean(1).max():+.4f}", flush=True)
    # G1c: the same Gemma arm on targets at onset + 37 s (STG), and on the aligned targets
    # restricted to the words whose shifted windows stay in the recording
    if "Ysh" in S:
        stg = np.flatnonzero(S["C"]["roi"].to_numpy() == "STG")
        inrange = S["Vsh"].any(1)
        Va = V[:, stg] & inrange[:, None]
        for tag, Yt, Vt in (("aligned", Y[:, stg], Va), ("shifted", S["Ysh"], S["Vsh"])):
            P, Yr, _ = M.oof(exact["gemma300"], nuis, Yt, Vt, folds, exact=True)
            R[f"g1c_{tag}"] = np.array([M.colcorr(P[li], Yr, Vt) for li in range(len(P))])
    # word-type decomposition arms, masked cells at the training mean
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in S["W"]["word"]])
    fid = np.full(len(Y), -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    a28 = {"SAE": S["sae"], "prior": exact["gpt2_300"], "gemma": exact["gemma300"],
           "sae300": exact["sae300"]}
    for name, X in a28.items():
        P, Yr, _ = M.oof(X, nuis, Y, V, folds, exact=False)
        R[f"a28r_{name}"] = np.array([M.colcorr(P[li], Yr, V) for li in range(len(P))])
        for occ in (MIN_OCC,) + SENS_OCC:
            for mode in ("global", "within_fold"):
                bw = [decompose(P[li], Yr, V, types, occ,
                                fid if mode == "within_fold" else None)
                      for li in range(len(P))]
                R[f"dec_{name}_{mode}_{occ}"] = np.stack(
                    [np.stack([b[0], b[1]]) for b in bw])          # (L, 2, E)
        print(f"    {sub} A28 {name:7s} best mean r {R[f'a28r_{name}'].mean(1).max():+.4f}",
              flush=True)
    return R


# -------------------------------------------------------------------------- statistics

def boot(diff, subj, seed):
    return M.cluster_boot(diff, subj, np.random.default_rng(seed), N_BOOT)


def pick(rs: np.ndarray) -> tuple[np.ndarray, int]:
    """``best_lambda_r`` on stacked (L, E) scores."""
    li = int(np.argmax(rs.mean(1)))
    return rs[li], li


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def interaction(dw, db, subj, seed):
    """``decomposition.interaction_stats`` on per-contact component differences."""
    rng = np.random.default_rng(seed)
    uq = np.unique(subj)
    draws = [rng.choice(uq, len(uq), replace=True) for _ in range(N_BOOT)]
    out = {}
    for lab, x in (("interaction", dw - db), ("d_within", dw), ("d_between", db)):
        bx = {u: x[subj == u] for u in uq}
        bb = np.array([np.median(np.concatenate([bx[u] for u in d])) for d in draws])
        out.update({lab: float(np.median(x)), f"{lab}_lo": float(np.percentile(bb, 2.5)),
                    f"{lab}_hi": float(np.percentile(bb, 97.5)),
                    f"{lab}_p": float(min(1.0, 2 * min((bb <= 0).mean(), (bb >= 0).mean())
                                          + 2 / (N_BOOT + 1)))})
    return out


def sign_test(diff, subj, predicted_positive: bool):
    from scipy.stats import binomtest
    med = pd.Series(diff).groupby(subj).median()
    k = int(((med > 0) if predicted_positive else (med < 0)).sum())
    return k, len(med), float(binomtest(k, len(med), 0.5, alternative="greater").pvalue)


def region_stats(roi: str, rows: pd.DataFrame, R: dict, seed: int) -> dict:
    """All gates and hypotheses for one region, from the stored per-lambda scores."""
    subj = rows.subject.to_numpy()
    cols = {s: np.flatnonzero(R[s]["_roi"] == roi) for s in rows.subject.unique()}

    def stack(key):
        return np.concatenate([R[s][key][:, cols[s]] for s in cols], axis=1)

    out = {"roi": roi, "n_contacts": len(rows), "n_subjects": int(rows.subject.nunique())}
    r = {a: pick(stack(f"r_{a}")) for a in EXACT_ARMS}
    for a in EXACT_ARMS:
        out[f"mean_r_{a}"] = float(r[a][0].mean())
        out[f"lambda_{a}"] = float(M.LAMBDAS[r[a][1]])
    # G2 - positive control
    for a in ("sae300", "gemma300"):
        m, lo, hi, p, _ = boot(r[a][0], subj, seed + 1)
        out.update({f"g2_{a}": m, f"g2_{a}_lo": lo, f"g2_{a}_hi": hi})
    out["G2_pass"] = bool(out["g2_sae300_lo"] > 0 and out["g2_gemma300_lo"] > 0)
    # R1
    d1 = r["saek"][0] - r["classical"][0]
    m, lo, hi, p, _ = boot(d1, subj, seed + 2)
    out.update(R1_diff=m, R1_lo=lo, R1_hi=hi, R1_p=p, R1_ratio=float(
        r["saek"][0].mean() / r["classical"][0].mean()), R1_n_above=int((d1 > 0).sum()))
    out["R1_sign_k"], out["R1_sign_n"], out["R1_sign_p"] = sign_test(d1, subj, True)
    # R2 (equivalence bound) and R4
    d2 = r["sae300"][0] - r["gemma300"][0]
    m, lo, hi, p, b = boot(d2, subj, seed + 3)
    delta = float(r["gemma300"][0].mean() - r["gpt2_300"][0].mean())
    margin = delta if delta > 0 else PODCAST_MARGIN
    out.update(R2_diff=m, R2_lo=lo, R2_hi=hi, R2_p_two_sided=p, R2_delta_cohort=delta,
               R2_margin=margin, R2_hi90=float(np.percentile(b, 95)),
               R2_lo90=float(np.percentile(b, 5)), R2_p=float((b >= margin).mean()))
    out["R2_nonsuperiority"] = bool(out["R2_hi90"] < margin)
    out["R2_sign_k"], out["R2_sign_n"], out["R2_sign_p"] = sign_test(d2, subj, False)
    d4 = r["sae300"][0] - r["gpt2_300"][0]
    m, lo, hi, p, _ = boot(d4, subj, seed + 4)
    out.update(R4_diff=m, R4_lo=lo, R4_hi=hi, R4_p=p)
    # R3 (sparse vs dense within type) and R3b (interaction); lambda per arm on total r
    li = {a: pick(stack(f"a28r_{a}"))[1] for a in A28_ARMS}
    for occ in (MIN_OCC,) + SENS_OCC:
        for mode in ("within_fold", "global"):
            comp = {a: np.concatenate(
                [R[s][f"dec_{a}_{mode}_{occ}"][li[a]][:, cols[s]] for s in cols], axis=1)
                for a in A28_ARMS}                          # (2, E): between, within
            for tag, a, b in (("R3", "sae300", "gemma"), ("R3b", "SAE", "prior")):
                dw = comp[a][1] - comp[b][1]
                db = comp[a][0] - comp[b][0]
                ok = np.isfinite(dw) & np.isfinite(db)
                st = interaction(dw[ok], db[ok], subj[ok], seed + 5)
                key = f"{tag}_{mode}_{occ}"
                out.update({f"{key}_{k}": v for k, v in st.items()})
    return out


def verdicts(o: dict) -> dict:
    """Verdict wording for each hypothesis, as fixed in the analysis plan."""
    def word(direction_ok: bool, passed: bool, opposite: bool) -> str:
        if direction_ok and passed:
            return "replicated"
        if opposite:
            return "contradicted"
        return "inconclusive"
    p = holm([o["R1_p"], o["R2_p"], o["R3_within_fold_4_interaction_p"]])
    o["R1_p_holm"], o["R2_p_holm"], o["R3_p_holm"] = p
    o["R1_outcome"] = word(o["R1_diff"] > 0, p[0] < 0.05, o["R1_hi"] < 0)
    o["R2_outcome"] = word(True, o["R2_nonsuperiority"] and p[1] < 0.05,
                           o["R2_lo"] > o["R2_margin"])
    k = "R3_within_fold_4"
    o["R3_outcome"] = word(o[f"{k}_interaction"] < 0, p[2] < 0.05,
                           o[f"{k}_interaction_lo"] > 0)
    o["R3_context"] = ("sparsification costs contextual information"
                       if o[f"{k}_d_within_hi"] < 0 else
                       "the within-model contrast does not resolve a contextual difference")
    o["R3_lexical"] = ("sparsification sharpens lexical identity"
                       if o[f"{k}_d_between_lo"] > 0 else
                       "the within-model contrast does not resolve a lexical difference")
    kb = "R3b_within_fold_4"
    o["R3b_outcome"] = word(o[f"{kb}_interaction"] < 0, o[f"{kb}_interaction_p"] < 0.05,
                            o[f"{kb}_interaction_lo"] > 0)
    return o


# -------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--fit-only", nargs="*", default=None,
                    help="fit (and cache) only these subjects, then stop - a smoke test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = cfg.results("film_replication")
    res = base / "results"
    (res / "fits").mkdir(parents=True, exist_ok=True)
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: int(s.split("_")[1]))
    if args.fit_only:
        subs = [s for s in subs if s in args.fit_only]
    R, contacts = {}, []
    for i, sub in enumerate(subs):
        f = res / "fits" / f"{sub}.npz"
        S = None
        if args.refit or not f.exists():
            S = load_subject(cfg, sub, base)
            folds = M.folds_for(cfg, S["W"]["t_global"].to_numpy())
            print(f"{sub}: {len(S['W'])} words, {S['Y'].shape[1]} contacts, "
                  f"{100 * (~S['V']).mean():.2f}% masked, nuisance {S['nuis'].shape[1]}, "
                  f"SAE live {S['sae'].shape[1]}, classical {S['classical'].shape[1]}",
                  flush=True)
            fr = fit_subject(sub, S, folds, cfg.seed + 38 + i)
            fr["_roi"] = S["C"]["roi"].to_numpy()
            np.savez(f, **{k: v for k, v in fr.items()})
        z = np.load(f, allow_pickle=True)
        R[sub] = {k: z[k] for k in z.files}
        C = pd.DataFrame.from_records(np.load(base / "targets" / f"{sub}.npz",
                                              allow_pickle=True)["contacts"])
        V = np.load(base / "targets" / f"{sub}.npz", allow_pickle=True)["V"].all(2)
        C["subject"], C["valid_frac"] = sub, V.mean(0)
        contacts.append(C)
    if args.fit_only:
        print("fit-only: cached", subs)
        return
    C = pd.concat(contacts, ignore_index=True)

    # G0 and region eligibility (census rule applied after neural QC)
    per = C.groupby(["roi", "subject"]).size().unstack(fill_value=0)
    ifg = per.loc["IFG"] if "IFG" in per.index else pd.Series(dtype=int)
    G0 = int((ifg >= 5).sum())
    eligible = [r for r in per.index if int((per.loc[r] >= 3).sum()) >= 3]
    print(f"G0: {G0} subjects with >= 5 IFG contacts -> {'PASS' if G0 >= 5 else 'FAIL'}")
    print("eligible regions:", eligible, "| not eligible:",
          sorted(set(per.index) - set(eligible)))

    # G1b/G1c (STG)
    g1 = {}
    stg_subj = [s for s in subs if "g1c_aligned" in R[s]]
    if stg_subj:
        rows = C[C.roi == "STG"]
        cols = {s: np.flatnonzero(R[s]["_roi"] == "STG") for s in stg_subj}
        subj = np.concatenate([[s] * len(cols[s]) for s in stg_subj])
        al, li_a = pick(np.concatenate([R[s]["g1c_aligned"] for s in stg_subj], axis=1))
        sh, li_s = pick(np.concatenate([R[s]["g1c_shifted"] for s in stg_subj], axis=1))
        gem, _ = pick(np.concatenate([R[s]["r_gemma300"][:, cols[s]] for s in stg_subj],
                                     axis=1))
        m, lo, hi, p, _ = boot(gem, subj, cfg.seed + 380)
        g1.update(G1b_median_r=m, G1b_lo=lo, G1b_hi=hi, G1b_pass=bool(lo > 0))
        m, lo, hi, p, _ = boot(al - sh, subj, cfg.seed + 381)
        g1.update(G1c_aligned_mean_r=float(al.mean()), G1c_shifted_mean_r=float(sh.mean()),
                  G1c_diff=m, G1c_lo=lo, G1c_hi=hi, G1c_pass=bool(lo > 0))
        print("G1:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g1.items()})

    out = []
    for j, roi in enumerate(["IFG"] + [r for r in eligible if r != "IFG"]):
        rows = C[C.roi == roi]
        o = verdicts(region_stats(roi, rows, R, cfg.seed + 3800 + 10 * j))
        out.append(o)
        print(f"\n[{roi}] {o['n_contacts']} contacts / {o['n_subjects']} subjects | "
              f"G2 {'PASS' if o['G2_pass'] else 'FAIL'} (SAE-300 {o['g2_sae300']:+.4f} "
              f"[{o['g2_sae300_lo']:+.4f}], Gemma {o['g2_gemma300']:+.4f} "
              f"[{o['g2_gemma300_lo']:+.4f}])")
        print(f"  R1 SAE-{int(R[subs[0]]['k_c'])} - classical {o['R1_diff']:+.4f} "
              f"[{o['R1_lo']:+.4f}, {o['R1_hi']:+.4f}] p {o['R1_p']:.4f} ratio "
              f"{o['R1_ratio']:.2f} -> {o['R1_outcome']}")
        print(f"  R2 SAE-300 - Gemma-300 {o['R2_diff']:+.4f} [{o['R2_lo']:+.4f}, "
              f"{o['R2_hi']:+.4f}]; 90% upper {o['R2_hi90']:+.4f} vs margin "
              f"{o['R2_margin']:.4f} (cohort Delta {o['R2_delta_cohort']:+.4f}) -> "
              f"{o['R2_outcome']}")
        k = "R3_within_fold_4"
        print(f"  R3 interaction {o[f'{k}_interaction']:+.4f} [{o[f'{k}_interaction_lo']:+.4f},"
              f" {o[f'{k}_interaction_hi']:+.4f}] p {o[f'{k}_interaction_p']:.4f}; d_between "
              f"{o[f'{k}_d_between']:+.4f} [{o[f'{k}_d_between_lo']:+.4f}, "
              f"{o[f'{k}_d_between_hi']:+.4f}]; d_within {o[f'{k}_d_within']:+.4f} "
              f"[{o[f'{k}_d_within_lo']:+.4f}, {o[f'{k}_d_within_hi']:+.4f}] -> "
              f"{o['R3_outcome']}")
        print(f"  R3b (A28) interaction {o['R3b_within_fold_4_interaction']:+.4f} "
              f"[{o['R3b_within_fold_4_interaction_lo']:+.4f}, "
              f"{o['R3b_within_fold_4_interaction_hi']:+.4f}] -> {o['R3b_outcome']}")
        print(f"  R4 SAE-300 - GPT-2 XL-300 {o['R4_diff']:+.4f} [{o['R4_lo']:+.4f}, "
              f"{o['R4_hi']:+.4f}]")
    T = pd.DataFrame(out)
    # secondary regions: Holm across regions within each hypothesis
    sec = T.roi != "IFG"
    for h, col in (("R1", "R1_p"), ("R2", "R2_p"), ("R3", "R3_within_fold_4_interaction_p")):
        T[f"{h}_p_holm_regions"] = np.nan
        if sec.sum():
            T.loc[sec, f"{h}_p_holm_regions"] = holm(T.loc[sec, col].to_numpy())
    T.to_csv(res / "a38_regions.csv", index=False)
    C.to_csv(res / "a38_contacts.csv", index=False)
    (res / "a38_gates.json").write_text(json.dumps(
        {"G0_subjects_ge5_ifg": G0, "G0_pass": G0 >= 5, "eligible_regions": eligible,
         "not_eligible": sorted(set(per.index) - set(eligible)), **g1,
         "contacts_by_region_subject": per.to_dict()}, indent=2, default=str))

    # S1 (leave one subject out) and S3 (Laplacian only) for IFG R1-R3
    sens = []
    rows = C[C.roi == "IFG"]
    for drop in [None] + sorted(rows.subject.unique()):
        for lap_only in (False, True):
            if drop is None and lap_only is False:
                continue
            keep = rows.subject != drop if drop else pd.Series(True, index=rows.index)
            if lap_only:
                keep &= rows.ref == "lap"
            if drop and lap_only:
                continue
            R2_ = {}
            for s in rows.subject.unique():
                if s == drop:
                    continue
                m = R[s]["_roi"] == "IFG"
                lap = rows[rows.subject == s].ref.to_numpy() == "lap"
                d = dict(R[s])
                d["_roi"] = np.where(m, "IFG", "x").astype(object)
                if lap_only:
                    idx = np.flatnonzero(m)
                    d["_roi"][idx[~lap]] = "x"
                R2_[s] = d
            rr = rows[keep]
            o = verdicts(region_stats("IFG", rr, R2_, cfg.seed + 3900))
            sens.append({"analysis": "S3 Laplacian only" if lap_only else "S1 LOSO",
                         "dropped": drop or "-", **{k: o[k] for k in o if k.startswith(
                             ("R1_", "R2_", "R3_within_fold_4", "n_"))}})
    pd.DataFrame(sens).to_csv(res / "a38_sensitivity.csv", index=False)
    print("\nwritten:", res / "a38_regions.csv")


if __name__ == "__main__":
    main()
