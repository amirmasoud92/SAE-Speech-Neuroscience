"""Max-over-lags null from circular shifts of the stimulus.

The banded model selects band weights per electrode and uses a lag-specific acoustic band, so
it cannot be refitted 1,000 times. The null is computed exactly for a closed-form proxy of the
sparse band's per-lag unique variance:

    proxy dR2(e, l) = [sum y_res^2 - sum (y_res - p)^2] / sum (y - ybar)^2

over valid held-out cells pooled across the 10 folds. y_res is the target with the nuisance set
(surprisal, lexical, 100 acoustic components, intercept) removed on training rows, and p is the
ridge prediction from the sparse band at 300 fold-wise components. The penalty is chosen per
fold by inner cross-validation on the observed window targets and held fixed across shifts.

**Shifts.** The stimulus side (sparse design and nuisance set) is rolled by k words relative to
the responses, k ~ U[k_min, n - k_min] with k_min = 60 s of words. Each word's 161-lag response
epoch stays intact and the folds stay on the response time axis, so test-set denominators are
identical for every shift. k = 0 is the observed statistic, computed by the same code.

**Masks.** Responses are standardised per column over valid cells and masked cells are set to
the column mean in training; only valid cells are scored.

With ``--device cuda`` the per-shift products run on the GPU. Writes
``controls/fixed_window/shift_null.npz`` incrementally.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import argparse
import time
from pathlib import Path

import numpy as np

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config

HERE = Path(__file__).parent


def load_targets(cfg):
    lang, ctrl = set(cfg.electrodes["language_rois"]), set(cfg.electrodes["control_rois"])
    blocks, vblocks, subj, roi = [], [], [], []
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                    allow_pickle=True)
        r = v["roi"].astype(str)
        keep = np.flatnonzero(np.isin(r, list(lang | ctrl)))
        z = np.load(cfg.cache(cfg.digest, f"targets_{sub}.npz"), allow_pickle=True)
        blocks.append(z["Y"][:, keep, :].astype(np.float32))
        vblocks.append(z["V"][:, keep, :])
        subj += [sub] * len(keep)
        roi += list(r[keep])
        del z
    return (np.concatenate(blocks, 1), np.concatenate(vblocks, 1), np.array(subj),
            np.array(roi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shift", type=int, default=1000)
    ap.add_argument("--device", default="cpu",
                    help="cpu, or cuda for the per-shift products (about 3 GB of GPU memory)")
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    out = cfg.results("controls", "fixed_window")
    out.mkdir(parents=True, exist_ok=True)
    ctx = PS.context(cfg)
    Y3, V3, subj, roi = load_targets(cfg)
    n_w, n_e, n_l = Y3.shape
    lags = cfg.lags.centres_s
    win = (lags >= -1e-9) & (lags <= 0.5 + 1e-9)
    Vwin = V3[:, :, win].all(2)
    Ywin = np.where(Vwin, Y3[:, :, win].mean(2), 0.0).astype(np.float64)
    Ywin = (Ywin - Ywin.mean(0)) / (Ywin.std(0) + 1e-12)
    Y = np.empty((n_w, n_e * n_l), np.float32)
    V = V3.reshape(n_w, n_e * n_l)
    Yv = Y3.reshape(n_w, n_e * n_l)
    for c0 in range(0, Y.shape[1], 8192):
        sl = slice(c0, c0 + 8192)
        w = V[:, sl]
        y = np.where(w, Yv[:, sl], 0.0).astype(np.float64)
        n = np.maximum(w.sum(0), 1)
        mu = y.sum(0) / n
        sd = np.sqrt(np.where(w, (y - mu) ** 2, 0.0).sum(0) / n) + 1e-12
        Y[:, sl] = np.where(w, (y - mu) / sd, 0.0)
    del Y3, Yv, V3
    print(f"{n_e} language+control electrodes x {n_l} lags; {n_w} words", flush=True)

    Xraw, nuis, folds = M.sae_band(cfg), ctx["nuis"], ctx["folds"]
    import torch
    dev = torch.device(args.device)
    Yt = torch.from_numpy(Y).to(dev)
    fold = []
    for f, (tr, te) in enumerate(folds):
        Xp = PS.fold_pca(Xraw, tr, 300)
        lam, _ = PS.inner_lambda(Xp, nuis, Ywin, Vwin, tr, ctx["onsets"], cfg)
        fold.append(dict(tr=tr, te=te, lam=lam, Xp=Xp,
                         te_t=torch.from_numpy(te).to(dev),
                         ve=torch.from_numpy(np.ascontiguousarray(V[te])).to(dev),
                         ytr_mean=torch.from_numpy(Y[tr].mean(0, dtype=np.float64)).to(dev)))
        print(f"  fold {f + 1}: lambda {lam:g}", flush=True)
    # pooled SST over valid test cells; fixed, because the responses never move
    s1 = sum(Y[p["te"]].sum(0, dtype=np.float64) for p in fold)
    s2 = sum((Y[p["te"]].astype(np.float64) ** 2).sum(0) for p in fold)
    cnt = sum(V[p["te"]].sum(0) for p in fold)
    SST = s2 - s1 ** 2 / np.maximum(cnt, 1)

    def proxy(k):
        """Per-lag proxy dR2 (n_e, n_l) with the stimulus side rolled by k words. The gain
        sum y_res^2 - (y_res - p)^2 = sum 2 y_res p - p^2 is accumulated in float64."""
        gain = torch.zeros(Y.shape[1], dtype=torch.float64, device=dev)
        Nk = np.roll(nuis, k, 0)
        for p in fold:
            tr, te, lam = p["tr"], p["te"], p["lam"]
            Xk = np.roll(p["Xp"], k, 0)
            Ntr, Nte = Nk[tr], Nk[te]
            Ginv = np.linalg.inv(Ntr.T @ Ntr + 1e-8 * np.eye(Ntr.shape[1]))
            bx = Ginv @ (Ntr.T @ Xk[tr])
            Xtr, Xte = Xk[tr] - Ntr @ bx, Xk[te] - Nte @ bx
            mx = Xtr.mean(0)
            Xtr, Xte = Xtr - mx, Xte - mx
            U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
            A = (Xte @ Vt.T) * (s / (s ** 2 + lam))
            Mop = np.zeros((300 + Ntr.shape[1], n_w), np.float32)
            Mop[:300, tr] = U.T
            Mop[300:, tr] = Ginv @ Ntr.T
            W = torch.from_numpy(Mop).to(dev) @ Yt                      # (406, n_cols)
            by = W[300:].double()
            my = p["ytr_mean"] - torch.from_numpy(Ntr.mean(0)).to(dev) @ by
            yres = Yt[p["te_t"]].double() - torch.from_numpy(Nte).to(dev) @ by - my
            pred = torch.from_numpy(A).to(dev) @ W[:300].double()
            gain += torch.where(p["ve"], 2 * yres * pred - pred ** 2, 0.0).sum(0)
        return (gain.cpu().numpy() / np.maximum(SST, 1e-12)).reshape(n_e, n_l)

    t0 = time.time()
    d0 = proxy(0)
    pk0, wn0 = d0.max(1), d0[:, win].mean(1)
    print(f"observed proxy computed in {time.time() - t0:.1f} s", flush=True)
    f = out / "shift_null.npz"
    rng = np.random.default_rng(cfg.seed + 808)
    dur = ctx["onsets"].max() - ctx["onsets"].min()
    kmin = int(np.ceil(60.0 * n_w / dur))
    shifts = rng.integers(kmin, n_w - kmin, size=args.n_shift)
    PK = np.full((args.n_shift, n_e), np.nan, np.float32)
    WN = np.full((args.n_shift, n_e), np.nan, np.float32)
    done = 0
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if np.array_equal(z["shifts"][:args.n_shift], shifts) and "done" in z.files:
            done = min(int(z["done"]), args.n_shift)
            PK[:done], WN[:done] = z["null_peak"][:done], z["null_win"][:done]
            print(f"resuming at {done}", flush=True)

    def save(n_done):
        np.savez(f, shifts=shifts, done=n_done, null_peak=PK, null_win=WN, obs_peak=pk0,
                 obs_win=wn0, obs_profile=d0.astype(np.float32), subj=subj, roi=roi,
                 lags=lags, kmin=kmin, lambdas=np.array([p["lam"] for p in fold]))

    save(done)
    for i in range(done, args.n_shift):
        d = proxy(int(shifts[i]))
        PK[i], WN[i] = d.max(1), d[:, win].mean(1)
        if (i + 1) % 25 == 0 or i + 1 == args.n_shift:
            save(i + 1)
            print(f"  {i + 1}/{args.n_shift} shifts, {(time.time() - t0) / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
