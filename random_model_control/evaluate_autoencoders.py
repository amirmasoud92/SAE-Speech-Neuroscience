"""Evaluation of the trained-model and random-model autoencoders.

* ``podcast``: layer-12 activations of each model on the podcast transcript, computed as the
  primary extraction does (its tokenisation, BOS-prefixed windows of 1,024 with stride 512,
  eager attention, float32, output of block 12). The trained model must reproduce the stored
  ``resid_token``.
* ``eval``: for each checkpoint, the normalised podcast activations are encoded and max-pooled
  over each word's tokens, and the coherence metric is applied to a 300-feature variance pool
  of live features (top-20 distinct word types, GPT-2 XL layer-24 space), with 300 principal
  components of the same model's residual stream for comparison. At 100 M tokens also: IFG
  predictivity at 300 components and the comparison with GemmaScope.

Decision rule: coherence is attributed to learned structure only if the trained-model
autoencoder exceeds both the random-model autoencoder and its own initialisation with
one-sided Mann-Whitney P < 0.05 and Cliff's delta >= 0.33. Writes
``controls/random_model/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent
POOL = 300


def podcast(cfg, out: Path, which: str, device: str) -> np.ndarray:
    from feature_extraction.language_model_features import tokenise_transcript
    from random_model_control.train_topk_autoencoder import LAYER, load_model
    from transformers import AutoTokenizer
    f = out / which / "podcast_resid.npy"
    if f.exists():
        return np.load(f)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    words = pd.read_csv(cfg.paths.bids_root / "stimuli" / "podcast_transcript.csv")["word"]
    ids, t2w = tokenise_transcript(words.astype(str).tolist(), tok)
    base = load_model(which, device, torch.float32)
    cap = {}
    base.layers[LAYER].register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    n, ctx, stride, bos = len(ids), 1024, 512, tok.bos_token_id
    resid = np.full((n, base.config.hidden_size), np.nan, np.float32)
    start = 0
    with torch.no_grad():
        while start < n:
            stop = min(start + stride, n)
            c0 = max(0, stop - ctx)
            window = [bos] + ids[c0:stop]
            base(input_ids=torch.tensor([window], device=device), use_cache=False)
            h = cap["h"][0].float().cpu().numpy()
            lo, hi = start - c0 + 1, stop - c0 + 1
            resid[np.arange(c0 + lo - 1, c0 + hi - 1)] = h[lo:hi]
            start = stop
    assert not np.isnan(resid).any()
    (out / which).mkdir(parents=True, exist_ok=True)
    np.save(f, resid)
    np.save(out / which / "podcast_tok2word.npy", t2w)
    if which == "trained":
        with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
            ref = h["resid_token"][...]
        rel = float(np.abs(resid - ref).max() / np.abs(ref).max())
        corr = float(np.corrcoef(resid.ravel()[::97], ref.ravel()[::97])[0, 1])
        json.dump({"max_abs_rel_err": rel, "corr": corr}, open(out / "trained" / "podcast_gate.json", "w"))
        print(f"gate (trained vs stored resid_token): max |diff| / max |x| = {rel:.2e}, r = {corr:.8f}")
        assert rel < 1e-3 and corr > 0.99999, "trained podcast activations do not reproduce the extraction"
    return resid


def pool_words(X, t2w, how):
    from feature_extraction.pooling_and_static_embeddings import pool
    return pool(t2w, int(t2w.max()) + 1, X, how)


def encode(resid, ck):
    x = torch.from_numpy(resid) * ck["c"]
    with torch.no_grad():
        p = (x - ck["b_pre"]) @ ck["W_enc"] + ck["b_enc"]
        v, i = p.topk(int(ck["k"]), dim=-1)
        z = torch.zeros_like(p).scatter_(-1, i, torch.relu(v))
    return z.numpy()


def coherence_block(F, R, E, tok, cfg):
    """``layer_and_model_battery.run_row``'s statistic: per-feature coherence of the 300-feature variance
    pool of the live features, plus the coherence of 300 PCs of the residual R (both signs)."""
    from interpretability.coherence import coherence
    from sklearn.decomposition import PCA
    F = F[:, (F > 0).sum(0) >= int(cfg.sae["min_active_words"])]
    P = PCA(n_components=POOL, random_state=cfg.seed).fit_transform(R.astype(np.float32))
    c_pca = np.concatenate([np.asarray(coherence(P, E, tok)), np.asarray(coherence(-P, E, tok))])
    pool = np.sort(np.argsort(-F.var(0))[:POOL])
    coh = np.asarray(coherence(F[:, pool], E, tok))
    med = float(np.nanmedian(c_pca))
    return dict(coh=coh, c_pca=c_pca, pca_median=med, n_live=int(F.shape[1]),
                n_scored=int(np.isfinite(coh).sum()), usable=int(np.nansum(coh > med)),
                median_coherence=float(np.nanmedian(coh)))


def random_word_floor(E, tok, cfg):
    """The chance floor: median coherence of 300 random sets of K distinct word types,
    seeded with the configured seed."""
    from interpretability.coherence import K
    rng = np.random.default_rng(cfg.seed)
    types = {}
    for i in range(len(tok)):
        types.setdefault(tok[i], i)
    uniq = np.array(list(types.values()))
    c = np.array([float((lambda V: (V @ V.T)[np.triu_indices(K, 1)].mean())(
        E[rng.choice(uniq, K, replace=False)])) for _ in range(300)])
    return float(np.median(c))


def cliffs_delta(a, b):
    """Pr(A > B) - Pr(A < B), from sorted counts."""
    bs = np.sort(b)
    lt = np.searchsorted(bs, a, side="left")          # b < a
    le = np.searchsorted(bs, a, side="right")         # b <= a
    return float((lt.sum() - (len(b) - le).sum()) / (len(a) * len(b)))


def compare(a, b, seed, n_boot=10000):
    """One-sided Mann-Whitney (a > b), Cliff's delta and its bootstrap 95% CI over features."""
    rng = np.random.default_rng(seed)
    mw = sps.mannwhitneyu(a, b, alternative="greater")
    d = cliffs_delta(a, b)
    bd = np.array([cliffs_delta(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    return {"U": float(mw.statistic), "p_one_sided": float(mw.pvalue), "cliffs_delta": d,
            "delta_ci95_lo": float(lo), "delta_ci95_hi": float(hi),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "n_a": int(len(a)), "n_b": int(len(b)),
            "passes": bool(mw.pvalue < 0.05 and d >= 0.33)}


def evaluate(cfg, out: Path, skip_ifg: bool = False) -> None:
    z = np.load(cfg.results("layer_and_model_battery", "_shared.npz"), allow_pickle=True)
    E, tok, wi = z["E"].astype(np.float64), z["tok"], z["wi"]
    floor = random_word_floor(E, tok, cfg)
    res, words = {}, {}
    for which in ("trained", "random"):
        resid = np.load(out / which / "podcast_resid.npy")
        t2w = np.load(out / which / "podcast_tok2word.npy")
        R = pool_words(resid, t2w, "mean")[wi]
        words[which] = R
        for ck_path in sorted((out / which).glob("sae_*M.pt"), key=lambda p: int(p.stem[4:-1])):
            ck = torch.load(ck_path, weights_only=False)
            F = pool_words(encode(resid, ck), t2w, "max")[wi].astype(np.float64)
            cb = coherence_block(F, R, E, tok, cfg)
            res[(which, int(ck_path.stem[4:-1]))] = dict(cb, fvu=ck["fvu_eval"],
                                                       dead=ck["dead_frac_10M_window"],
                                                       never=ck["never_active_on_eval_frac"], F=F)
            print(f"  {which} {ck_path.stem}: FVU {ck['fvu_eval']:.4f}, usable "
                  f"{cb['usable']}/{cb['n_scored']}, median coherence {cb['median_coherence']:.4f}",
                  flush=True)
    with h5py.File(cfg.results("features") / "gemma2-2b_L12_w16k_l0-82.hdf5", "r") as h:
        Fg = h["sae_word_max"][...][wi].astype(np.float64)
        Rg = h["resid_word"][...][wi].astype(np.float64)
    gs = coherence_block(Fg, Rg, E, tok, cfg)
    final = max(k for (w, k) in res if w == "trained")
    tr, ra = res[("trained", final)], res[("random", final)]
    t0, r0 = res[("trained", 0)], res[("random", 0)]
    common = tr["pca_median"]

    def row(label, status, coh, own_med, extra=None):
        c = coh[np.isfinite(coh)]
        return {"dictionary": label, "status": status, "n_scored": int(len(c)),
                "median_coherence": float(np.median(c)), "median_minus_floor": float(np.median(c) - floor),
                "pct_above_floor": float(100 * (c > floor).mean()),
                "pct_above_trained_pca_median": float(100 * (c > common).mean()),
                "pct_above_own_pca_median": float(100 * (c > own_med).mean()), **(extra or {})}

    rows = [row("GemmaScope (w16k, L0 82)", "released", gs["coh"], gs["pca_median"])]
    for which, lab in (("trained", "matched-recipe SAE, trained model"), ("random", "matched-recipe SAE, random model")):
        for k in sorted(k for (w, k) in res if w == which and k > 0):
            r = res[(which, k)]
            rows.append(row(lab, f"{k} M tokens", r["coh"], r["pca_median"],
                            {"fvu_eval": r["fvu"], "dead_frac_10M_window": r["dead"],
                             "never_active_on_eval": r["never"]}))
    for which, lab in (("trained", "initialisation dictionary, trained model"),
                       ("random", "initialisation dictionary, random model")):
        r = res[(which, 0)]
        rows.append(row(lab, "step 0 (no training)", r["coh"], r["pca_median"],
                        {"fvu_eval": r["fvu"], "never_active_on_eval": r["never"]}))
    rows.append(row("PCA, trained model (300 PCs, both signs)", "reference", tr["c_pca"], tr["pca_median"]))
    rows.append(row("PCA, random model (300 PCs, both signs)", "extra row", ra["c_pca"], ra["pca_median"]))
    h2 = out / "h2_random_dict_coherence.json"           # descriptive only
    if h2.exists():
        H = json.load(open(h2))
        rows.append(row(H["dictionary"], "descriptive, not in the decision rule",
                        np.asarray(H["coherence"], float), tr["pca_median"]))
    T = pd.DataFrame(rows)
    T.to_csv(out / "a3_coherence_table.csv", index=False)
    pd.DataFrame({"trained_100M": tr["coh"], "random_100M": ra["coh"], "trained_init": t0["coh"],
                  "random_init": r0["coh"], "gemmascope": gs["coh"]}).to_csv(out / "a3_feature_coherence.csv", index=False)

    fin = lambda x: x[np.isfinite(x)]  # noqa: E731
    cmp_a = compare(fin(tr["coh"]), fin(ra["coh"]), cfg.seed + 4400)
    cmp_b = compare(fin(tr["coh"]), fin(t0["coh"]), cfg.seed + 4401)
    learned = cmp_a["passes"] and cmp_b["passes"]
    decision = ("ATTRIBUTED TO LEARNED STRUCTURE: the trained-model SAE beats both the random-model "
                "SAE and the trained-model initialisation dictionary (P < 0.05 and Cliff's delta >= 0.33 each)"
                if learned else
                "NOT ATTRIBUTED TO LEARNING: reframe coherence as a property of sparse overcomplete bases "
                "(failing comparison(s): " + ", ".join(n for n, c in (("vs random-model SAE", cmp_a),
                                                                    ("vs trained-model initialisation", cmp_b))
                                                       if not c["passes"]) + ")")
    rc = sps.mannwhitneyu(fin(tr["coh"]), fin(gs["coh"]), alternative="two-sided")
    S = {"final_checkpoint_M": final, "floor": floor,
         "rule": "one-sided Mann-Whitney P < 0.05 and Cliff's delta >= 0.33 for both comparisons",
         "comparison_vs_random_model_sae": cmp_a, "comparison_vs_trained_init": cmp_b,
         "learned_structure": bool(learned), "decision": decision,
         "recipe_check": {"test": "Mann-Whitney U, two-sided, trained matched-recipe vs GemmaScope",
                          "U": float(rc.statistic), "p": float(rc.pvalue), "cliffs_delta":
                          cliffs_delta(fin(tr["coh"]), fin(gs["coh"])),
                          "median_gemmascope": gs["median_coherence"],
                          "gemmascope_usable": gs["usable"], "gemmascope_scored": gs["n_scored"],
                          "gemmascope_pca_median": gs["pca_median"]}}
    json.dump(S, open(out / "a3_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    with pd.option_context("display.width", 250):
        print(T.to_string(index=False))
    if skip_ifg:
        return

    # ---- IFG predictivity at 300 PCs: five dictionaries and two dense streams -----------
    ctx = PS.context(cfg)
    Y, V, subj, meta = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    arms = {"trained-model SAE": tr["F"], "random-model SAE": ra["F"],
            "trained-model initialisation": t0["F"], "random-model initialisation": r0["F"],
            "trained dense": words["trained"], "random dense": words["random"]}
    pred = {}
    for name, X in arms.items():
        if "dense" not in name:
            X = X[:, (X > 0).sum(0) >= int(cfg.sae["min_active_words"])]
        rL, _ = PS.oof_legacy(M.zpca(X, 300), ctx, Y, V, exact=True)
        P, Yr, _ = PS.oof_to(X, 300, ctx, Y, V, cfg, exact=True)
        pred[("LEG", name)], pred[("TO", name)] = rL, M.colcorr(P, Yr, V)
        print(f"  IFG {name}: LEG {rL.mean():+.4f}  TO {pred[('TO', name)].mean():+.4f}", flush=True)
    for conv in ("LEG", "TO"):
        pred[(conv, "GemmaScope")] = pd.read_csv(cfg.results("controls", "window_arms", f"{conv}_SAE-300.csv")).r.to_numpy()
    pairs = [("trained-model SAE", "random-model SAE"),
             ("trained-model SAE", "trained-model initialisation"),
             ("GemmaScope", "trained-model SAE"),
             ("random-model SAE", "random-model initialisation"),
             ("trained dense", "random dense"),
             ("GemmaScope", "trained dense"), ("trained-model SAE", "trained dense"),
             ("trained-model initialisation", "trained dense"),
             ("random-model SAE", "random dense"), ("random-model initialisation", "random dense")]
    con = []
    for i, (a_, b_) in enumerate(pairs):
        for conv in ("LEG", "TO"):
            st, _ = PS.electrode_stat(pred[(conv, a_)] - pred[(conv, b_)], subj, cfg.seed + 4300 + i)
            con.append({"convention": conv, "contrast": f"{a_} - {b_}",
                        "mean_r_a": float(pred[(conv, a_)].mean()), "mean_r_b": float(pred[(conv, b_)].mean()), **st})
    C = pd.DataFrame(con)
    C.to_csv(out / "a3_ifg.csv", index=False)
    pd.DataFrame({f"{c}|{n}": v for (c, n), v in pred.items()}).assign(subject=subj).to_csv(
        out / "a3_ifg_per_electrode.csv", index=False)
    with pd.option_context("display.width", 220):
        print(C[["convention", "contrast", "mean_r_a", "mean_r_b", "estimate", "ci95_lo", "ci95_hi", "p_two_sided"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("podcast", "eval"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-ifg", action="store_true", help="smoke tests only")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="default results/controls/random_model")
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = args.out_root or cfg.results("controls", "random_model")
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "podcast":
        for which in ("trained", "random"):
            podcast(cfg, out, which, args.device)
            print(f"{which}: podcast activations written", flush=True)
    else:
        evaluate(cfg, out, skip_ifg=args.skip_ifg)


if __name__ == "__main__":
    main()
