"""Within-type predictivity across lags.

Within-type predictivity of each representation is computed at 41 lags from -2,000 to
+2,000 ms in 100-ms steps. Each arm's penalty is the one chosen on total held-out
predictivity in ``word_types/decomposition.py``.

Inference is cluster-based permutation over lags: a cluster-forming threshold from the null,
cluster mass as the statistic, and the null distribution of the maximum cluster mass, with
occurrence order shuffled within word type in the observed responses (1,000 permutations).
Both decompositions (``global`` and ``within_fold``) are computed; masked cells are excluded
from type means and correlations, observed and permuted. Writes ``within_type_signal/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as Mt
from utils.config import load_config
from encoding.fit_encoding_model import analysed_words

STEP = 4                                      # 25 ms grid -> 100 ms
MIN_OCC = 4
N_PERM = 1000
ROI = "IFG"


def inherited_lambdas(cfg):
    """Penalties from the word-type decomposition's output (one per representation)."""
    D = pd.read_csv(cfg.results("word_types", "a28_decomposition.csv"))
    return D.groupby("representation")["lambda"].first().to_dict()


def within_resid(M, V, g, n_g):
    """Remove per-group means (over valid tokens) from every column; masked cells -> 0."""
    w = V.astype(np.float64)
    cnt = np.zeros((n_g, M.shape[1]))
    m = np.zeros((n_g, M.shape[1]))
    np.add.at(cnt, g, w)
    np.add.at(m, g, M.astype(np.float64) * w)
    return np.where(V, M - (m / np.maximum(cnt, 1))[g], 0.0)


def groups_for(mode, idx, fid_k, Vk):
    """Group ids, group count and the effective validity for one decomposition."""
    if mode == "global":
        return idx, int(idx.max() + 1), Vk
    g = pd.factorize(pd.Series(idx * 1000 + fid_k))[0]
    n_g = int(g.max() + 1)
    cnt = np.zeros((n_g, Vk.shape[1]))
    np.add.at(cnt, g, Vk.astype(np.float64))
    return g, n_g, Vk & (cnt[g] >= 2)


def clusters(profile, thresh):
    """Contiguous runs above threshold; returns list of (start, stop, mass)."""
    above = profile > thresh
    out, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            out.append((i, j + 1, float(profile[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed + 291)
    LAMBDA = inherited_lambdas(cfg)
    print("penalties inherited from the decomposition:", LAMBDA)

    w = analysed_words(cfg).reset_index(drop=True)
    onsets, wi = w["start"].to_numpy(float), w["word_idx"].to_numpy()
    types = np.array([str(x).strip().lower().strip('.,!?;:"\'') for x in w.word])
    Xsae = Mt.sae_band(cfg)
    Xpri = Mt.zpca(Mt.gpt2xl(cfg, wi), 300)
    nuis = Mt.nuisance(cfg)

    blocks, vblocks = [], []
    lags = cfg.lags.centres_s[::STEP]
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        keep = np.flatnonzero(v["roi"].astype(str) == ROI)
        if len(keep) == 0:
            continue
        z = np.load(cfg.cache(cfg.digest, f"targets_{sub}.npz"), allow_pickle=True)
        blocks.append(z["Y"][:, keep, ::STEP].astype(np.float64))
        vblocks.append(z["V"][:, keep, ::STEP])
        del z
    Yall = np.concatenate(blocks, axis=1)
    Vall = np.concatenate(vblocks, axis=1)
    del blocks, vblocks
    n_w, n_e, n_l = Yall.shape
    wv = Vall.astype(np.float64)
    nv = np.maximum(wv.sum(0), 1.0)
    mu = (Yall * wv).sum(0) / nv
    sd = np.sqrt((((Yall - mu) ** 2) * wv).sum(0) / nv) + 1e-12
    flat = np.where(Vall, (Yall - mu) / sd, 0.0).reshape(n_w, n_e * n_l)
    Vflat = Vall.reshape(n_w, n_e * n_l)
    del Yall, Vall, wv
    print(f"{n_e} {ROI} electrodes x {n_l} lags ({lags[0]*1000:.0f} to {lags[-1]*1000:.0f} "
          f"ms), {100 * (~Vflat).mean():.2f}% of cells masked")

    vc = pd.Series(types).value_counts()
    keep_tok = np.isin(types, vc[vc >= MIN_OCC].index)
    idx = pd.factorize(pd.Series(types[keep_tok]))[0]
    n_g = int(idx.max() + 1)
    print(f"{n_g} types with >= {MIN_OCC} occurrences, {int(keep_tok.sum())} tokens")

    folds = Mt.folds_for(cfg, onsets)
    fid = np.full(n_w, -1)
    for k, (_, te) in enumerate(folds):
        fid[te] = k
    Vk0 = Vflat[keep_tok]
    MODES = ("global", "within_fold")

    rows, prof_store = [], {}
    # the model the sparse code was built from, and the sparse code at 300 components,
    # appended after the other arms so that their permutation draws are unchanged
    arms = [("SAE", Xsae), ("prior", Xpri),
            ("gemma", Mt.zpca(Mt.gemma_dense(cfg, wi), 300)), ("sae300", Mt.zpca(Xsae, 300))]
    for name, Xr in arms:
        lam = float(LAMBDA[name])
        print(f"  fitting {name} at lambda {lam:g} ...", flush=True)
        P, Yr, _ = Mt.oof(Xr, nuis, flat, Vflat, folds, lambdas=np.array([lam]), exact=False)
        Pk, Yk = P[0][keep_tok], Yr[keep_tok]
        del P, Yr
        for mode in MODES:
            g, n_g2, Vk = groups_for(mode, idx, fid[keep_tok], Vk0)
            Vkf = Vk.astype(np.float64)
            order = np.argsort(g, kind="stable")
            Rp = within_resid(Pk, Vk, g, n_g2)
            Ro = within_resid(Yk, Vk, g, n_g2)
            np_ = np.sqrt((Rp ** 2).sum(0)) + 1e-12
            no_ = np.sqrt((Ro ** 2).sum(0)) + 1e-12
            obs = ((Rp * Ro).sum(0) / (np_ * no_)).reshape(n_e, n_l).mean(0)
            key = name if mode == "global" else f"{name}_within_fold"
            prof_store[key] = obs
            Rp2, Ro2 = Rp ** 2, Ro ** 2

            null = np.empty((args.n_perm, n_l))
            for p in range(args.n_perm):
                src = np.empty_like(order)
                src[order] = np.lexsort((rng.random(len(g)), g))
                num = (Rp * Ro[src]).sum(0)
                den = (np.sqrt((Rp2 * Vkf[src]).sum(0)) * np.sqrt((Ro2[src] * Vkf).sum(0))
                       + 1e-12)
                null[p] = (num / den).reshape(n_e, n_l).mean(0)
                if (p + 1) % 250 == 0:
                    print(f"    {mode} {p + 1}/{args.n_perm}", flush=True)

            thresh = float(np.percentile(null, 95))
            obs_cl = clusters(obs, thresh)
            null_max = np.array([max([c[2] for c in clusters(null[p], thresh)], default=0.0)
                                 for p in range(args.n_perm)])
            print(f"    [{mode}] threshold {thresh:+.4f} | {len(obs_cl)} observed cluster(s)")
            base = {"representation": name, "type_means": mode, "lambda": lam,
                    "threshold": thresh}
            for st, sp, mass in obs_cl:
                pv = float((np.sum(null_max >= mass) + 1) / (args.n_perm + 1))
                rows.append({**base, "lag_start_ms": lags[st] * 1000,
                             "lag_stop_ms": lags[sp - 1] * 1000, "n_lags": sp - st,
                             "mass": mass, "peak_r": float(obs[st:sp].max()),
                             "peak_lag_ms": float(lags[st + int(np.argmax(obs[st:sp]))] * 1000),
                             "p_cluster": pv})
                print(f"      {lags[st]*1000:+.0f} to {lags[sp-1]*1000:+.0f} ms, mass "
                      f"{mass:.4f}, peak r {obs[st:sp].max():+.4f} at "
                      f"{lags[st + int(np.argmax(obs[st:sp]))]*1000:+.0f} ms, p = {pv:.4f}")
            if not obs_cl:
                rows.append({**base, "lag_start_ms": np.nan, "lag_stop_ms": np.nan,
                             "n_lags": 0, "mass": 0.0, "peak_r": float(obs.max()),
                             "peak_lag_ms": float(lags[int(np.argmax(obs))] * 1000),
                             "p_cluster": 1.0})
                print(f"      no supra-threshold cluster (max r {obs.max():+.4f} at "
                      f"{lags[int(np.argmax(obs))]*1000:+.0f} ms)")

    R = pd.DataFrame(rows)
    verdicts = {}
    print("\n" + "=" * 80)
    for mode in MODES:
        sig = R[(R.representation == "SAE") & (R.type_means == mode) & (R.p_cluster < 0.05)]
        if len(sig):
            b = sig.loc[sig.mass.idxmax()]
            v = (f"H-A29b SUPPORTED ({mode}): SAE within-word predictivity forms a cluster "
                 f"from {b.lag_start_ms:+.0f} to {b.lag_stop_ms:+.0f} ms, "
                 f"p = {b.p_cluster:.4f}.")
        else:
            v = (f"H-A29b NOT SUPPORTED ({mode}): no lag shows SAE within-word predictivity "
                 "above the cluster-corrected null.")
        verdicts[mode] = v
        print("VERDICT:", v)
    print("=" * 80)
    R["verdict"] = R.type_means.map(verdicts)

    o = cfg.results("within_type_signal")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "a29b_clusters.csv", index=False)
    pd.DataFrame({"lag_ms": lags * 1000, **{k: prof_store[k] for k in prof_store}}).to_csv(
        o / "a29b_profiles.csv", index=False)
    print("written:", o / "a29b_clusters.csv")


if __name__ == "__main__":
    main()
