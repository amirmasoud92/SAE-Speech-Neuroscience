"""Brain Treebank high-gamma targets.

Stage 1, per session: resample to 512 Hz, despike (|x - median| > 4 IQR, 100-ms dilation,
PCHIP), re-reference within shaft (Laplacian, or bipolar at shaft ends), notch 60/120/180 Hz,
70-200 Hz fourth-order Butterworth zero-phase band-pass, Hilbert envelope. Kept: the 10-Hz log
grid for cleaning and the log 200-ms window means at lags 0-500 ms
(``neural_targets.build_targets`` of the primary cohort), plus the same cells at onset + 37 s
for an alignment check in STG.

Stage 2, per participant: the cross-validation blocks of the session-concatenated clock, then
``artifact_cleaning`` per session (blocks = CV block x session; common mode over the
participant's analysed contacts; session-wide noise scale), drift subtracted, and the
residual-offset guard over the participant's words.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from utils.paths import BTB_ROOT


BTB = BTB_ROOT
FS = 512.0
LAGS_S = np.arange(0, 21) * 0.025          # 0-500 ms, 25-ms steps
WIN_S = 0.2
SHIFT_S = 37.0                             # G1c misalignment control
RATES = (1024.0, 2048.0)
GAP_S = 1000.0                             # session separation on the concatenated clock


def fs_raw_of(sess: str) -> float:
    """The sampling rate measured from the session's trigger table."""
    T = pd.read_csv(BTB / "subject_timings" / "subject_timings" / f"{sess}_timings.csv")
    T = T.loc[:, ~T.columns.str.startswith("Unnamed")]
    q = (T[T["type"] == "trigger"] if "type" in T.columns else T).dropna(
        subset=["index", "movie_time"])
    dm = np.diff(q.movie_time.values.astype(float))
    di = np.diff(q["index"].values.astype(float))
    ok = (dm > 0.05) & (dm < 0.2)
    r = float(np.median(di[ok] / dm[ok]))
    return float(min(RATES, key=lambda f: abs(f - r)))


def despike(x: np.ndarray, sfreq: float, iqr_mult: float = 4.0, dil_len: float = 0.10):
    """``ecogprep.find_spike_events`` + ``interpolate_spikes`` for one channel."""
    from scipy import interpolate, signal, stats
    spikes = np.abs(x - np.median(x)) > stats.iqr(x) * iqr_mult
    if dil_len > 0:
        f = np.ones(int(dil_len * sfreq))
        spikes = signal.filtfilt(f, 1, spikes) > 0
    if spikes.any():
        good = np.flatnonzero(~spikes)
        x = x.copy()
        x[spikes] = interpolate.pchip_interpolate(good, x[good], np.flatnonzero(spikes))
    return x, float(spikes.mean())


def envelope(x: np.ndarray) -> np.ndarray:
    import mne
    from scipy.fft import next_fast_len
    from scipy.signal import hilbert
    x = mne.filter.notch_filter(x, FS, [60.0, 120.0, 180.0], notch_widths=2, verbose=False)
    x = mne.filter.filter_data(x, FS, 70.0, 200.0, method="iir",
                               iir_params=dict(order=4, ftype="butter"), verbose=False)
    n = x.shape[-1]
    return np.abs(hilbert(x, N=next_fast_len(n))[:n])


def log_cells(env: np.ndarray, onsets_s: np.ndarray) -> np.ndarray:
    """(n_words, n_ch, 21) log window means - ``data.build_targets`` on this lag grid."""
    from preprocessing.neural_targets import onset_samples, prefix_sums, window_average
    starts0 = onset_samples(onsets_s, FS)
    n_win = int(round(WIN_S * FS))
    cs = prefix_sums(env)
    out = np.empty((len(onsets_s), env.shape[0], len(LAGS_S)), dtype=np.float32)
    for i, off in enumerate(np.rint(LAGS_S * FS).astype(np.int64)):
        out[:, :, i] = window_average(env, starts0 + off, n_win, cs=cs)
    if (out <= 0).any():
        raise ValueError("non-positive window average before log")
    return np.log(out)


def session_stage(sess: str, contacts: list[dict], onsets_s: np.ndarray, out: Path) -> str:
    """Stage 1 for one session; writes ``<sess>.npz``."""
    import h5py
    import mne
    f_out = out / f"{sess}.npz"
    if f_out.exists():
        return f"{sess}: cached"
    fs = fs_raw_of(sess)
    sub = sess.split("_trial")[0]
    lab = json.load(open(BTB / "electrode_labels" / "electrode_labels" / sub /
                         "electrode_labels.json"))
    need = sorted({n for c in contacts for n in [c["electrode"]] + c["neighbours"]})
    raw, spike = {}, {}
    with h5py.File(BTB / f"{sess}.h5" / f"{sess}.h5", "r") as h:
        keys = {lab[int(k.split("_")[1])].rstrip("*"): k for k in h["data"].keys()
                if int(k.split("_")[1]) < len(lab)}
        n_raw = None
        for nm in need:
            x = h["data"][keys[nm]][:].astype(np.float64)
            n_raw = len(x) if n_raw is None else n_raw
            x = mne.filter.resample(x, down=fs / FS, npad="auto", verbose=False)
            raw[nm], spike[nm] = despike(x, FS)
    n = min(len(v) for v in raw.values())
    env = np.empty((len(contacts), n), dtype=np.float64)
    for i, c in enumerate(contacts):
        x = raw[c["electrode"]][:n]
        nb = [raw[m][:n] for m in c["neighbours"]]
        y = x - 0.5 * (nb[0] + nb[1]) if len(nb) == 2 else x - nb[0]
        env[i] = envelope(y)
    del raw
    from preprocessing.artifact_cleaning import to_grid
    grid = to_grid(env, FS)                                   # (n_bins, E)
    ok = (np.rint(onsets_s * FS) >= 0) & (
        np.rint(onsets_s * FS) + np.rint(LAGS_S[-1] * FS) + int(round(WIN_S * FS)) <= n)
    Y = np.full((len(onsets_s), len(contacts), len(LAGS_S)), np.nan, np.float32)
    Y[ok] = log_cells(env, onsets_s[ok])
    stg = np.array([c["roi"] == "STG" for c in contacts])
    sh = onsets_s + SHIFT_S
    ok_sh = ok & (np.rint(sh * FS) + np.rint(LAGS_S[-1] * FS) + int(round(WIN_S * FS)) <= n)
    Ysh = np.full((len(onsets_s), int(stg.sum()), len(LAGS_S)), np.nan, np.float32)
    if stg.any():
        Ysh[ok_sh] = log_cells(env[stg], sh[ok_sh])
    np.savez(f_out, grid=grid.astype(np.float32), Y=Y, Ysh=Ysh, ok=ok, ok_sh=ok_sh,
             n_times=n, fs_raw=fs, n_raw=n_raw,
             spike_frac=np.array([spike[c["electrode"]] for c in contacts]),
             names=np.array([c["electrode"] for c in contacts], dtype=object))
    return f"{sess}: fs {fs:.0f}, {len(contacts)} contacts, {n / FS / 60:.1f} min, " \
           f"{int(ok.sum())}/{len(ok)} words in range"


def contacts_of(sub: str, census: pd.DataFrame) -> list[dict]:
    """Analysed contacts of a subject (mapped region, not monopolar) with their references."""
    from film_cohort.census import shaft
    good = set(census.loc[census.subject == sub, "electrode"])
    C = census[(census.subject == sub) & (census.roi != "other") & (census.ref != "mono")]
    out = []
    for r in C.itertuples():
        s, c = shaft(r.electrode)
        nb = [f"{s}{c + d}" for d in (-1, 1) if f"{s}{c + d}" in good]
        out.append({"electrode": r.electrode, "roi": r.roi, "ref": r.ref, "neighbours": nb,
                    "region": r.region})
    return out


def subject_stage(sub: str, W: pd.DataFrame, contacts: list[dict], neural: Path,
                  out: Path, n_splits: int = 10, embargo_s: float = 10.0) -> dict:
    """Stage 2: clean per session on the concatenated clock; write ``<sub>.npz``."""
    from preprocessing import artifact_cleaning as CT
    sessions = list(dict.fromkeys(W.sess))
    Z = {s: np.load(neural / f"{s}.npz", allow_pickle=True) for s in sessions}
    keep = np.concatenate([Z[s]["ok"] for s in sessions])
    W = W[keep].reset_index(drop=True)
    dur = {s: float(Z[s]["n_times"]) / FS for s in sessions}
    off, t = {}, 0.0
    for s in sessions:
        off[s] = t
        t += dur[s] + GAP_S
    W["t_global"] = W.t_local + W.sess.map(off)
    edges = CT.block_time_edges(W.t_global.to_numpy(), n_splits, t)
    Ys, Vs, Yss, Vss, rep = [], [], [], [], {}
    for s in sessions:
        z = Z[s]
        grid = z["grid"].astype(np.float64)
        nb = grid.shape[0]
        inner = [e - off[s] for e in edges[1:-1] if off[s] < e < off[s] + dur[s]]
        bins = np.clip(np.rint(np.r_[0.0, inner, dur[s]] * CT.RATE).astype(int), 0, nb)
        bins[-1] = nb
        trend, valid, r = CT.clean_grid(grid, bins)
        m = (W.sess == s).to_numpy()
        tl = W.t_local.to_numpy()[m]
        drift, ok = CT.cells_from_grid(tl, LAGS_S, WIN_S, trend, valid)
        y = z["Y"][z["ok"]]
        Ys.append(y - drift)
        Vs.append(ok & np.isfinite(y))
        stg = np.array([c["roi"] == "STG" for c in contacts])
        if stg.any():
            d2, ok2 = CT.cells_from_grid(tl + SHIFT_S, LAGS_S, WIN_S, trend[:, stg],
                                         valid[:, stg])
            ysh = z["Ysh"][z["ok"]]
            Yss.append(np.nan_to_num(ysh - d2))
            Vss.append(ok2 & np.isfinite(ysh))
        rep[s] = {"masked_frac_grid": float((~valid).mean()),
                  "common_mode_s": int(r["common_mode_s"]),
                  "spike_frac_median": float(np.median(z["spike_frac"]))}
    Y, V = np.concatenate(Ys), np.concatenate(Vs)
    frac0 = float((~V).mean())
    V, n_off = CT.mask_residual_offsets(Y, V, W.t_global.to_numpy(), n_splits, embargo_s)
    extra = {}
    if Yss:
        Ysh, Vsh = np.concatenate(Yss), np.concatenate(Vss)
        Vsh, _ = CT.mask_residual_offsets(Ysh, Vsh, W.t_global.to_numpy(), n_splits,
                                          embargo_s)
        extra = {"Ysh": Ysh, "Vsh": Vsh}
    meta = pd.DataFrame(contacts).drop(columns="neighbours")
    np.savez(out / f"{sub}.npz", Y=Y, V=V, contacts=meta.to_records(index=False),
             words=W.drop(columns=["ud_rows", "mel", "delta_mel"]).to_records(index=False),
             mel=np.stack(W.mel.to_numpy()), delta_mel=np.stack(W.delta_mel.to_numpy()),
             **extra)
    rep["_subject"] = {"words": len(W), "contacts": len(contacts),
                       "masked_frac_procedure": frac0, "masked_frac_final":
                           float((~V).mean()), "n_offset_sets_masked": int(n_off)}
    (out / f"{sub}_cleaning.json").write_text(json.dumps(rep, indent=2))
    return rep["_subject"]


def main() -> None:
    from film_cohort.word_tables import all_films, subject_words
    from utils.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    base = cfg.results("film_replication")
    neural, targ = base / "neural", base / "targets"
    neural.mkdir(parents=True, exist_ok=True)
    targ.mkdir(parents=True, exist_ok=True)
    census = pd.read_csv(base / "btb_contact_census.csv")
    films = all_films(base / "words")
    subs = args.subjects or [s for s in sorted(census.subject.unique(),
                                               key=lambda s: int(s.split("_")[1]))
                             if s != "sub_5"]
    jobs, words, cons = [], {}, {}
    for sub in subs:
        W = subject_words(sub, films)
        fs = {s: fs_raw_of(s) for s in W.sess.unique()}
        W["t_local"] = W.neural_sample / W.sess.map(fs)
        words[sub], cons[sub] = W, contacts_of(sub, census)
        for s in W.sess.unique():
            jobs.append((s, cons[sub], W.loc[W.sess == s, "t_local"].to_numpy()))
    print(f"{len(jobs)} sessions, {sum(len(c) for c in cons.values())} contacts", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(session_stage, s, c, t, neural) for s, c, t in jobs]
        for fu in as_completed(futs):
            print(" ", fu.result(), flush=True)
    for sub in subs:
        r = subject_stage(sub, words[sub], cons[sub], neural, targ)
        print(sub, r, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
