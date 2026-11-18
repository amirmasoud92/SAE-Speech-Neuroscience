"""Sparse and dense bases in the banded model, and the concentration test (H2).

``dense``
    The sparse band against 300 principal components of the **same** Gemma-2-2B layer-12
    residual stream, fitted on the same folds with the same banded structure.

``sci``
    The concentration statistic for the learned dictionary against two controls: a random
    dictionary whose encoder is a random matrix with thresholds retuned to match each
    feature's activation density, and a column-shuffled dictionary that permutes word order
    within each feature column.

``dense_sci`` and ``matched_sci``
    The same statistic for 300 principal components of the residual stream, and for a fixed
    pool of the 300 live features with the highest variance across the stimulus (chosen
    without neural data; its feature ids are saved as ``pool_ids``).

Each electrode is fitted at the lag of its peak unique sparse-band variance. Writes
``basis_fits/``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import neural_targets as D
from utils.config import Config, load_config
from encoding.banded_ridge import FoldPCA, fit_banded, make_deferred_designer, random_dictionary_features, semantic_concentration, variance_partition
from feature_extraction.feature_bands import Design, dense_residual
from utils.run_logging import log_data_versions, setup_logging, stage
from encoding.fit_encoding_model import analysed_words, stage_design, stage_targets

log = logging.getLogger("tier1")


def _out(cfg: Config, name: str) -> Path:
    return cfg.results("basis_fits", name)


def _targets(Y: np.ndarray, V: np.ndarray | None, peak: np.ndarray | None):
    """Flatten targets for an SCI fit, optionally to one lag per electrode.

    Every consumer of these fits (``semantic_concentration``, ``participant_tally``) reads
    each electrode at a single lag - the argmax of delta_sae in the main variance partition -
    so with ``peak`` only those targets are fitted. The single-band ridge selects its
    penalty per target, so a target's result does not depend on which other targets are
    fitted alongside it. Outputs are written back into (n_ch, n_lags) arrays, NaN at every
    lag that was not fitted.

    Returns (Yf, Vf, back) where ``back`` maps a per-target vector to (n_ch, n_lags).
    """
    n_words, n_ch, n_lags = Y.shape
    if peak is None:
        Vf = None if V is None else V.reshape(n_words, n_ch * n_lags)
        return (Y.reshape(n_words, n_ch * n_lags), Vf,
                lambda v: v.reshape(n_ch, n_lags) if v.size == n_ch * n_lags else v)
    r = np.arange(n_ch)
    Yf = np.ascontiguousarray(Y[:, r, peak])
    Vf = None if V is None else np.ascontiguousarray(V[:, r, peak])

    def back(v):
        if v.size != n_ch:
            return v
        full = np.full((n_ch, n_lags), np.nan, dtype=np.float32)
        full[r, peak] = v
        return full
    return Yf, Vf, back


# --------------------------------------------------------------- 1.1 dense baseline

def analysis_dense(cfg: Config, sub: str, des: Design, Y: np.ndarray,
                   ch: list[str], words: pd.DataFrame, V: np.ndarray | None = None,
                   vp=None) -> None:
    """SAE band vs a matched dense-PCA band, both against the same nuisance bands.

    ``vp``: the participant's main variance partition. Its full model and its model
    without the SAE band are exactly the sparse full model and the shared nuisance-only
    reduced model this analysis would otherwise refit (same design, folds, targets and
    mask), so they are reused and only the dense full model is fitted.
    """
    n_words, n_ch, n_lags = Y.shape
    Yf = Y.reshape(n_words, n_ch * n_lags)
    Vf = None if V is None else V.reshape(n_words, n_ch * n_lags)
    onsets = words["start"].to_numpy()
    n_pcs = int(cfg.raw["comparison_models"]["dense_pca"]["n_components"])

    resid = dense_residual(cfg, words["word_idx"].to_numpy())
    nuisance = des.only(["surprisal", "acoustic", "lexical"])

    if vp is None:
        # Sparse: the real design, SAE band included.
        sparse_vp = variance_partition(cfg, des, Yf, onsets, bands=["sae"], valid=Vf)
        d_sae, full_sparse = sparse_vp["delta_sae"], sparse_vp["full"]
        reduced = fit_banded(cfg, nuisance, Yf, onsets, label="dense-minus",
                             valid=Vf).r2()
    else:
        if list(vp["ch_names"].astype(str)) != list(ch):
            raise SystemExit(f"{sub}: channels differ from the variance partition")
        d_sae = vp["delta_sae"].reshape(-1)
        full_sparse = vp["full"].reshape(-1)
        reduced = vp["r2_without_sae"].reshape(-1)

    # Dense: identical nuisance bands, SAE replaced by PCs fitted inside each fold.
    designer_full = make_deferred_designer(
        nuisance, {"dense": (resid, FoldPCA(n_pcs, seed=cfg.seed))})
    dense_full = fit_banded(cfg, nuisance, Yf, onsets, label="dense-full",
                            fold_designer=designer_full, valid=Vf)
    d_dense = dense_full.r2() - reduced

    np.savez_compressed(
        _out(cfg, f"dense_vs_sae_{sub}.npz"),
        ch_names=np.asarray(ch, dtype=object), lags_s=cfg.lags.centres_s,
        delta_sae=d_sae.reshape(n_ch, n_lags),
        delta_dense=d_dense.reshape(n_ch, n_lags),
        r2_full_sparse=full_sparse.reshape(n_ch, n_lags),
        r2_full_dense=dense_full.r2().reshape(n_ch, n_lags),
        r2_nuisance_only=reduced.reshape(n_ch, n_lags),
        n_pcs=n_pcs, reused_main_partition=vp is not None)
    log.info("%s dense-vs-sae: peak delta sae %.5f | peak delta dense %.5f",
             sub, float(np.nanmax(d_sae)), float(np.nanmax(d_dense)))


# ------------------------------------------------------------------------ 1.2 SCI

def analysis_sci(cfg: Config, sub: str, des: Design, Y: np.ndarray,
                 ch: list[str], words: pd.DataFrame, V: np.ndarray | None = None,
                 peak: np.ndarray | None = None) -> None:
    """SCI_k for the real dictionary and two sparsity-matched controls."""
    n_words, n_ch, n_lags = Y.shape
    Yf, Vf, back = _targets(Y, V, peak)
    onsets = words["start"].to_numpy()
    rng = np.random.default_rng(cfg.seed)

    real = des.only(["sae"])
    achieved_l0 = float((real.X > 0).sum(1).mean())
    d_sae = real.X.shape[1]
    log.info("%s: SAE band %d features, achieved word-level L0 %.1f",
             sub, d_sae, achieved_l0)

    resid = dense_residual(cfg, words["word_idx"].to_numpy())
    # Match the control to the real band feature by feature, not merely on mean L0: the
    # real band is filtered to >= min_active_words, so a globally thresholded control
    # carries thousands of near-indicator columns the real one cannot have.
    densities = (real.X > 0).sum(0)
    rand = random_dictionary_features(resid, densities, rng)
    ctrl_random = Design(rand, {"sae": slice(0, d_sae)}, {"sae": np.arange(d_sae)})
    log.info("%s: control density matched per feature - min %d, median %d, max %d "
             "active words; mean L0 %.1f", sub, int(densities.min()),
             int(np.median(densities)), int(densities.max()),
             float((rand > 0).sum(1).mean()))

    shuf = real.X.copy()
    for j in range(shuf.shape[1]):
        rng.shuffle(shuf[:, j])
    ctrl_shuffled = Design(shuf, {"sae": slice(0, d_sae)}, {"sae": np.arange(d_sae)})

    out: dict[str, np.ndarray] = {}
    for name, design in [("real", real), ("random_dict", ctrl_random),
                         ("shuffled", ctrl_shuffled)]:
        with stage(log, f"SCI {sub} {name}"):
            res = semantic_concentration(cfg, design, Yf, onsets, band="sae",
                                         valid=Vf)
        for k, v in res.items():
            out[f"{name}__{k}"] = back(v)
        finite = np.isfinite(res[f"sci_{cfg.mono['top_k'][0]}"])
        log.info("%s %s: r2_all median %.5f | SCI_%d median %.3f (n finite %d)",
                 sub, name, float(np.nanmedian(res["r2_all"])), cfg.mono["top_k"][0],
                 float(np.nanmedian(res[f"sci_{cfg.mono['top_k'][0]}"])), int(finite.sum()))

    np.savez_compressed(_out(cfg, f"sci_{sub}.npz"),
                        ch_names=np.asarray(ch, dtype=object),
                        lags_s=cfg.lags.centres_s, achieved_l0=achieved_l0,
                        n_features=d_sae, **out)


# --------------------------------------------- dense principal components at matched k

def analysis_dense_sci(cfg: Config, sub: str, des: Design, Y: np.ndarray,
                       ch: list[str], words: pd.DataFrame, V: np.ndarray | None = None,
                 peak: np.ndarray | None = None) -> None:
    """SCI over 300 principal components of the same residual stream.

    Decides whether the concentration H2 found is a property of the *learned* basis or of
    low-dimensional structure generally: principal components are the variance-optimal
    basis of exactly the stream the SAE was trained on, so they are the strongest available
    competitor.

    PCA is fitted once on all words rather than within folds. It is an unsupervised
    transform of the stimulus and sees no neural data, so it cannot leak the target, and it
    keeps the comparison fair - the SAE dictionary is likewise a fixed, globally-trained,
    stimulus-derived basis, and fold-wise PCA would give the dense arm a fold-adaptive
    basis the SAE is denied.
    """
    from sklearn.decomposition import PCA

    n_words, n_ch, n_lags = Y.shape
    Yf, Vf, back = _targets(Y, V, peak)
    onsets = words["start"].to_numpy()
    n_pcs = int(cfg.raw["comparison_models"]["dense_pca"]["n_components"])

    resid = dense_residual(cfg, words["word_idx"].to_numpy())
    pcs = PCA(n_components=n_pcs, random_state=cfg.seed).fit_transform(resid)
    log.info("%s: %d PCs of the layer-12 residual stream, %.1f%% of its variance",
             sub, n_pcs,
             100.0 * PCA(n_components=n_pcs, random_state=cfg.seed)
             .fit(resid).explained_variance_ratio_.sum())

    # band named "sae" so semantic_concentration's band argument applies unchanged, the
    # same convention analysis_sci uses for its two control dictionaries
    design = Design(pcs.astype(np.float32), {"sae": slice(0, n_pcs)},
                    {"sae": np.arange(n_pcs)})

    out: dict[str, np.ndarray] = {}
    with stage(log, f"dense-SCI {sub}"):
        res = semantic_concentration(cfg, design, Yf, onsets, band="sae",
                                         valid=Vf)
    for k, v in res.items():
        out[k] = back(v)
    log.info("%s dense-PCA: r2_all median %.5f | r2_top20 median %.5f",
             sub, float(np.nanmedian(res["r2_all"])),
             float(np.nanmedian(res["r2_top20"])))

    np.savez_compressed(_out(cfg, f"dense_sci_{sub}.npz"),
                        ch_names=np.asarray(ch, dtype=object),
                        lags_s=cfg.lags.centres_s, n_components=n_pcs, **out)


# --------------------------------------------- unsupervised 300-feature pool

def analysis_matched_sci(cfg: Config, sub: str, des: Design, Y: np.ndarray,
                         ch: list[str], words: pd.DataFrame, V: np.ndarray | None = None,
                 peak: np.ndarray | None = None) -> None:
    """SCI over the 300 highest-variance SAE features.

    Selecting 20 from 300 ordered principal components and 20 from 4,933 unordered sparse
    features differ in selection difficulty: a 16x larger unordered pool is noisier and
    overfits more. This arm selects from a pool of the same size.

    The pool is chosen by **variance across the analysed words** - unsupervised,
    stimulus-only, blind to neural data, and the analogue of what orders principal
    components (a variance optimum). Its feature ids are saved as ``pool_ids``.
    """
    n_words, n_ch, n_lags = Y.shape
    Yf, Vf, back = _targets(Y, V, peak)
    onsets = words["start"].to_numpy()

    real = des.only(["sae"])
    var = real.X.var(0)
    keep = np.argsort(-var)[:300]
    keep.sort()
    log.info("%s: matched pool = top 300 of %d SAE features by variance "
             "(%.1f%% of total feature variance, median density %d words)",
             sub, real.X.shape[1], 100.0 * var[keep].sum() / var.sum(),
             int(np.median((real.X[:, keep] > 0).sum(0))))

    design = Design(real.X[:, keep], {"sae": slice(0, 300)}, {"sae": np.arange(300)})

    out: dict[str, np.ndarray] = {}
    with stage(log, f"matched-SCI {sub}"):
        res = semantic_concentration(cfg, design, Yf, onsets, band="sae",
                                         valid=Vf)
    for k, v in res.items():
        out[k] = back(v)
    log.info("%s matched-pool: r2_all median %.5f | r2_top20 median %.5f",
             sub, float(np.nanmedian(res["r2_all"])),
             float(np.nanmedian(res["r2_top20"])))

    np.savez_compressed(_out(cfg, f"matched_sci_{sub}.npz"),
                        ch_names=np.asarray(ch, dtype=object),
                        lags_s=cfg.lags.centres_s,
                        pool_ids=real.feature_ids["sae"][keep], **out)


ANALYSES = {"dense": analysis_dense, "sci": analysis_sci,
            "dense_sci": analysis_dense_sci,
            "matched_sci": analysis_matched_sci}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--analyses", nargs="*", default=["dense", "sci", "dense_sci", "matched_sci"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--peak-only", action="store_true",
                    help="SCI analyses fit each electrode only at its delta_sae peak lag "
                         "from the main variance partition - the only lag any consumer "
                         "reads - and 'dense' reuses the main partition's sparse and "
                         "reduced fits. Requires the main encoding to be finished.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subs = args.subjects or cfg.subjects
    log_ = setup_logging(cfg.paths.results_root / "logs", "tier1", cfg.digest)
    log_.info("full config digest (incl. presentation): %s", cfg.full_digest)
    log_data_versions(log_, cfg.paths.bids_root)
    log_.info("analyses=%s subjects=%s", args.analyses, ",".join(subs))

    words = analysed_words(cfg)
    table = D.load_electrode_table(cfg)
    des = stage_design(cfg, words)

    # Subject-major, analysis-minor: the target tensor is the expensive thing to load,
    # so all analyses for a subject run while it is in memory.
    for sub in subs:
        Y, ch, V = stage_targets(cfg, sub, words, table)
        for name in args.analyses:
            if name not in ANALYSES:
                raise SystemExit(f"unknown analysis '{name}'. Available: {list(ANALYSES)}")
            dest = _out(cfg, {"dense": f"dense_vs_sae_{sub}.npz",
                              "sci": f"sci_{sub}.npz",
                              "dense_sci": f"dense_sci_{sub}.npz",
                              "matched_sci": f"matched_sci_{sub}.npz"}[name])
            if dest.exists():
                log_.info("%s %s: already done, skipping", sub, name)
                continue
            with stage(log_, f"{name} {sub}"):
                if name == "dense":
                    vp = (np.load(cfg.results("encoding",
                                              f"{sub}_variance_partition.npz"),
                                  allow_pickle=True) if args.peak_only else None)
                    ANALYSES[name](cfg, sub, des, Y, ch, words, V=V, vp=vp)
                elif name in ("sci", "dense_sci", "matched_sci"):
                    peak = None
                    if args.peak_only:
                        vp = np.load(cfg.results("encoding",
                                                 f"{sub}_variance_partition.npz"),
                                     allow_pickle=True)
                        if list(vp["ch_names"].astype(str)) != list(ch):
                            raise SystemExit(f"{sub}: channels differ from the partition")
                        d = vp["delta_sae"]
                        peak = np.nanargmax(np.where(np.isfinite(d), d, -np.inf), 1)
                    ANALYSES[name](cfg, sub, des, Y, ch, words, V=V, peak=peak)
                else:
                    ANALYSES[name](cfg, sub, des, Y, ch, words)
        del Y
    log_.info("tier-1 finished")


if __name__ == "__main__":
    main()
