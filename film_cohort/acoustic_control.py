"""Does the film cohort's coarser acoustic control explain its early auditory result?

Subcommands:

* ``podcast-band`` - a film-style acoustic band for the podcast. Per word, over
  [onset, onset + 500 ms]: the mean of the 80-bin log-mel spectrogram and the mean of its frame
  differences; RMS of the waveform; pitch as the median pYIN F0 of voiced frames (65-400 Hz),
  0 if none are voiced; and their differences from the previous word.
* ``podcast-fit`` - the film estimator (sparse band at 300 components, surprisal, film-style
  acoustic, lexical; lags 0-500 ms; covariance solver) on every podcast participant, compared
  with the same estimator with the original 17-delay acoustic band.
* ``film-fit`` - the film design plus a word-lagged acoustic band: the per-word log-mel and
  delta-mel vectors reduced to 32 components and, at each of the 17 delays, averaged over the
  same-session words whose midpoints fall within [onset + delay, onset + delay + 200 ms)
  (544 columns; windows with no word are zero).
* ``stats`` - the paired differences in early auditory cortex and the network coefficients.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from preprocessing import neural_targets as D
from capacity_matching import matched_capacity as M
from utils.config import load_config
from encoding.covariance_ridge import delta_r2_primal
from encoding.fit_encoding_model import analysed_words, stage_design, stage_targets

log = logging.getLogger("a43")
LAGS = slice(80, 101)
DELAYS = np.arange(-2.0, 2.0001, 0.25)
LANG, CTRL = ("IFG", "STG", "MTG", "TMP"), ("EAC", "PRC", "MFG")


def podcast_band(cfg, out: Path) -> Path:
    f = out / "a43a_podcast_film_style_band.npz"
    if f.exists():
        return f
    import librosa
    import scipy.io.wavfile as wavfile
    w = analysed_words(cfg).reset_index(drop=True)
    on = w["start"].to_numpy(float)
    with h5py.File(cfg.paths.spectrogram, "r") as h:
        S = h["vectors"][...].astype(np.float64)               # (80, frames) at 100 Hz
    fr = np.rint(on * 100).astype(int)
    mel = np.stack([S[:, max(a, 0):a + 50].mean(1) for a in fr])
    dmel = np.stack([np.diff(S[:, max(a, 0):a + 51], axis=1).mean(1) for a in fr])
    sr, x = wavfile.read(cfg.paths.bids_root / "stimuli" / "podcast.wav")
    x = x.astype(np.float64).mean(1) / 32768.0
    rms = np.array([np.sqrt(np.mean(x[int(a * sr):int((a + 0.5) * sr)] ** 2)) for a in on])
    x16 = librosa.resample(x, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(x16, fmin=65, fmax=400, sr=16000, frame_length=1024,
                                 hop_length=160)
    t0 = np.arange(len(f0)) * 0.01
    pitch = np.zeros(len(on))
    for i, a in enumerate(on):
        m = (t0 >= a) & (t0 < a + 0.5) & voiced & np.isfinite(f0)
        pitch[i] = float(np.median(f0[m])) if m.any() else 0.0
    d = lambda v: np.r_[0.0, np.diff(v)]                              # noqa: E731
    band = np.column_stack([mel, dmel, rms, pitch, d(rms), d(pitch)])
    np.savez(f, band=band, voiced_frac=np.array([(pitch > 0).mean()]))
    log.info("film-style podcast band: %s, %.0f%% of words voiced", band.shape,
             100 * (pitch > 0).mean())
    return f


def podcast_fit(cfg, out: Path) -> None:
    words = analysed_words(cfg)
    des = stage_design(cfg, words)
    table = D.load_electrode_table(cfg)
    onsets = words["start"].to_numpy()
    fb = np.load(out / "a43a_podcast_film_style_band.npz")["band"]
    bands = [("sae", M.zpca(des.band("sae"), 300)), ("surprisal", des.band("surprisal")),
             ("acoustic", fb), ("lexical", des.band("lexical"))]
    (out / "a43a").mkdir(exist_ok=True)
    for sub in cfg.subjects:
        f = out / "a43a" / f"{sub}.npz"
        if f.exists():
            continue
        Y, ch, V = stage_targets(cfg, sub, words, table)
        n, n_ch, _ = Y.shape
        Yl = np.ascontiguousarray(Y[:, :, LAGS]).reshape(n, -1)
        Vl = None if V is None else np.ascontiguousarray(V[:, :, LAGS]).reshape(n, -1)
        vp = delta_r2_primal(cfg, bands, Yl, onsets, "sae", valid=Vl, label=sub, form="cov")
        np.savez(f, delta_sae=vp["delta_sae"].reshape(n_ch, -1),
                 ch_names=np.asarray(ch, dtype=object))
        print("done", sub, flush=True)


def lagged_band(S: dict) -> np.ndarray:
    W = S["W"]
    A = S["parts"]["acoustic"][:, :256]       # load_subject's log10 mel and delta-mel
    A = A - A.mean(0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    P = A @ Vt[:32].T
    on = W["t_global"].to_numpy(float)
    mid = on + 0.5 * (W["end"].to_numpy(float) - W["start"].to_numpy(float))
    out = np.zeros((len(W), len(DELAYS) * 32))
    for sess, idx in W.groupby("sess", sort=False).indices.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(mid[idx])]
        m = mid[order]
        cs = np.vstack([np.zeros(32), np.cumsum(P[order], 0)])
        for k, dly in enumerate(DELAYS):
            lo = np.searchsorted(m, on[idx] + dly, "left")
            hi = np.searchsorted(m, on[idx] + dly + 0.2, "left")
            cnt = (hi - lo)[:, None]
            out[idx, k * 32:(k + 1) * 32] = np.where(cnt > 0, (cs[hi] - cs[lo]) /
                                                     np.maximum(cnt, 1), 0.0)
    return out


def film_fit(cfg, out: Path) -> None:
    from film_cohort.replication import load_subject
    base = cfg.results("film_replication")
    (out / "a43b").mkdir(exist_ok=True)
    subs = sorted((p.stem for p in (base / "targets").glob("sub_*.npz")),
                  key=lambda s: (base / "targets" / f"{s}.npz").stat().st_size)
    for sub in subs:
        f = out / "a43b" / f"{sub}.npz"
        if f.exists():
            continue
        S = load_subject(cfg, sub, base)
        z = np.load(base / "targets" / f"{sub}.npz", allow_pickle=True)
        Y, V = z["Y"].astype(np.float64), z["V"].astype(bool)
        n, n_ch, n_l = Y.shape
        P = S["parts"]
        bands = [("sae", M.zpca(S["sae"], 300))] + [
            (b, np.asarray(P[b], float)) for b in ("surprisal", "acoustic", "lexical",
                                                   "scene")] + [
            ("acoustic_lagged", lagged_band(S))]
        vp = delta_r2_primal(cfg, bands, Y.reshape(n, -1), S["W"]["t_global"].to_numpy(),
                             "sae", valid=V.reshape(n, -1), label=sub, form="cov")
        np.savez(f, delta_sae=vp["delta_sae"].reshape(n_ch, n_l),
                 roi=S["C"]["roi"].to_numpy().astype(str))
        print("done", sub, flush=True)


def stats(cfg, out: Path) -> None:
    from inference.statistics import mixed_model
    rng_seed = cfg.seed + 43
    rows = []
    # podcast: the original 17-delay acoustic band against the film-style band
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        a = np.load(out / "podcast_bridge" / f"{sub}_film_estimator.npz", allow_pickle=True)
        b = np.load(out / "a43a" / f"{sub}.npz", allow_pickle=True)
        assert np.array_equal(a["ch_names"].astype(str), b["ch_names"].astype(str))
        for i, roi in enumerate(v["roi"].astype(str)):
            rows.append({"part": "A43a", "subject": sub, "roi": roi,
                         "peak_ref": float(np.nanmax(a["delta_sae"][i])),
                         "peak_test": float(np.nanmax(b["delta_sae"][i]))})
    # film: the H1 fit against the refit with the word-lagged acoustic band
    for f in sorted((out / "a43b").glob("sub_*.npz")):
        a = np.load(out / "fits" / f.name, allow_pickle=True)
        b = np.load(f, allow_pickle=True)
        for i, roi in enumerate(b["roi"].astype(str)):
            rows.append({"part": "A43b", "subject": f.stem, "roi": roi,
                         "peak_ref": float(a["delta_sae"][i].max()),
                         "peak_test": float(b["delta_sae"][i].max())})
    T = pd.DataFrame(rows)
    T["diff"] = T.peak_test - T.peak_ref
    T.to_csv(out / "a43_peaks.csv", index=False)
    res = []
    for part, g in T.groupby("part"):
        for roi in ["EAC"] + [r for r in sorted(g.roi.unique()) if r != "EAC"]:
            h = g[g.roi == roi]
            if h.subject.nunique() < 3:
                continue
            m, lo, hi, p, _ = M.cluster_boot(h["diff"].to_numpy(), h.subject.to_numpy(),
                                             np.random.default_rng(rng_seed), 10000)
            res.append({"part": part, "roi": roi, "n": len(h),
                        "n_subjects": int(h.subject.nunique()), "median_diff": m, "lo": lo,
                        "hi": hi, "p": p, "primary": roi == "EAC"})
        for col, lab in (("peak_ref", "reference"), ("peak_test", "test")):
            d = g[g.roi.isin(LANG + CTRL)].copy()
            d["network"] = np.where(d.roi.isin(LANG), "language", "control")
            mm = mixed_model(d.rename(columns={col: "obs_peak"}), outcome="obs_peak",
                             fixed="network")
            res.append({"part": part, "roi": f"network beta ({lab})",
                        "median_diff": float(mm["params"]["C(network)[T.language]"]),
                        "p": float(mm["pvalues"]["C(network)[T.language]"])})
    R = pd.DataFrame(res)
    a = R[(R.part == "A43a") & (R.roi == "EAC")].iloc[0]
    b = R[(R.part == "A43b") & (R.roi == "EAC")].iloc[0]
    if a.lo > 0 and b.hi < 0:
        verdict = "SUPPORTED: the film H1 failure is attributed to acoustic control"
    elif not (a.lo > 0 or a.hi < 0) and not (b.lo > 0 or b.hi < 0):
        verdict = "NOT SUPPORTED: the EAC effect is a cohort difference"
    else:
        verdict = "UNRESOLVED"
    R["verdict"] = verdict
    R.to_csv(out / "a43_results.csv", index=False)
    print(R.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print("VERDICT:", verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("podcast-band", "podcast-fit", "film-fit", "stats"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("film_h1_h2")
    {"podcast-band": lambda: podcast_band(cfg, out), "podcast-fit": lambda: podcast_fit(cfg, out),
     "film-fit": lambda: film_fit(cfg, out), "stats": lambda: stats(cfg, out)}[args.step]()


if __name__ == "__main__":
    main()
