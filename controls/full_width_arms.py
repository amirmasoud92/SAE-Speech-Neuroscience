"""Full-width arms: the dense stream and the sparse code without reduction.

Every arm uses the IFG 0-500 ms window target and the matched-capacity machinery, and is
fitted under both conventions (``LEG`` and ``TO``, see ``controls/window_arms.py``):

- ``Gemma dense raw-2304``: the Gemma-2-2B layer-12 residual, all 2,304 dimensions, z-scored;
- ``Gemma dense PCA-2304``: all 2,304 principal components of the same z-scored matrix;
- ``SAE raw-4933``: the 4,933 live sparse features, z-scored;
- ``SAE PCA-4933``: all 4,933 principal components.

Full-width arms use training-mean imputation of masked cells; ``Gemma dense raw-2304 exact``
is also fitted with exact masking to measure the imputation effect at full width.

Ridge with an isotropic penalty is invariant to an orthogonal rotation of its inputs, and
keeping every principal component is such a rotation, so each raw arm and its full-PCA arm
must give the same predictions; the analysis step checks this.

Usage: ``--part fit --conv LEG|TO --arms ...``, then ``--part analyse``. Writes
``controls/full_width/``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from capacity_matching import matched_capacity as M
from controls import training_only as PS
from utils.config import load_config
from controls.sign_flip_and_electrode_tests import signflip

HERE = Path(__file__).parent
ARMS = ["Gemma dense raw-2304", "Gemma dense PCA-2304", "Gemma dense raw-2304 exact",
        "SAE raw-4933", "SAE PCA-4933"]


def fname(conv: str, arm: str) -> str:
    return f"{conv}_{arm.replace(' ', '_')}"


def zscore_global(X: np.ndarray) -> np.ndarray:
    """Global per-column z-scoring, exactly as ``matched.zpca`` does before its PCA."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0)
    sd = X.std(0)
    return np.ascontiguousarray(X / np.where(sd > 1e-12, sd, 1.0))


def fit(cfg, conv: str, arms: list[str]) -> None:
    out = cfg.results("controls", "full_width")
    out.mkdir(parents=True, exist_ok=True)
    ctx = PS.context(cfg)
    Y, V, subj, meta = M.ifg_targets(cfg, "window")
    Y = (Y - Y.mean(0)) / (Y.std(0, keepdims=True) + 1e-12)
    for arm in arms:
        f = out / f"{fname(conv, arm)}.csv"
        if f.exists():
            print(f"  {conv} {arm}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        base, rest = arm.split(" raw-") if " raw-" in arm else arm.split(" PCA-")
        kind = "raw" if " raw-" in arm else "PCA"
        exact = arm.endswith(" exact")
        wide = arm.endswith(" wide-grid")             # sensitivity: grid to 1e8, TO only
        grid = np.r_[M.LAMBDAS, 1e7, 1e8] if wide else M.LAMBDAS
        X = M.gemma_dense(cfg, ctx["wi"]) if base == "Gemma dense" else M.sae_band(cfg)
        if conv == "LEG":
            if wide:
                raise ValueError("the wide-grid sensitivity is defined for TO only")
            Xr = zscore_global(X) if kind == "raw" else M.zpca(X, None)
            r, lam = PS.oof_legacy(Xr, ctx, Y, V, exact=exact)
            lam = [lam]
        else:
            k = "z" if kind == "raw" else X.shape[1]
            P, Yr, lams = PS.oof_to(X, k, ctx, Y, V, cfg, exact=exact,
                                    verbose=f"{conv} {arm}", lambdas=grid)
            r, lam = M.colcorr(P, Yr, V), lams.tolist()
        pd.DataFrame({"subject": subj, "ch": meta.ch, "r": r}).to_csv(f, index=False)
        json.dump({"arm": arm, "conv": conv, "n_raw_columns": int(X.shape[1]),
                   "transform": "z-score only" if kind == "raw" else "z-score + all PCs",
                   "masking": "exact" if exact else "training-mean imputation",
                   "lambdas": lam, "lambda_grid": grid.tolist(),
                   "mean_r": float(r.mean()), "median_r": float(np.median(r)),
                   "seconds": round(time.time() - t0, 1)},
                  open(f.with_suffix(".json"), "w"), indent=1)
        print(f"  {conv} {arm:28s} mean r {r.mean():+.4f}  median {np.median(r):+.4f}  "
              f"lambda {lam}  ({time.time() - t0:.0f} s)", flush=True)


def load_arm(cfg, conv, arm):
    d = cfg.results("controls", "full_width")
    if arm in ("Gemma dense PCA-300", "SAE PCA-300"):      # the stored exact window arms
        w = cfg.results("controls", "window_arms")
        name = "Gemma_dense-300" if arm.startswith("Gemma") else "SAE-300"
        R = pd.read_csv(w / f"{conv}_{name}.csv")
        J = json.load(open(w / f"{conv}_{name}.json"))
        return R, dict(arm=arm, conv=conv, masking="exact", lambdas=J["lambdas"],
                       source=f"window_arms/{conv}_{name}")
    R = pd.read_csv(d / f"{fname(conv, arm)}.csv")
    J = json.load(open(d / f"{fname(conv, arm)}.json"))
    return R, J


def analyse(cfg) -> None:
    out = cfg.results("controls", "full_width")
    grid = M.LAMBDAS
    fits, per, checks = [], {}, {}
    for conv in ("LEG", "TO"):
        for arm in ARMS + ["Gemma dense PCA-300"]:
            R, J = load_arm(cfg, conv, arm)
            per[(conv, arm)] = R
            lams = np.asarray(J["lambdas"], float)
            edge = bool(np.isin(lams, [grid.min(), grid.max()]).any())
            fits.append({"convention": "published" if conv == "LEG" else "training-only",
                         "arm": arm, "masking": J.get("masking", "exact"),
                         "n_electrodes": len(R), "mean_r": float(R.r.mean()),
                         "median_r": float(R.r.median()),
                         "lambda": (f"{lams[0]:.0e}" if len(lams) == 1 else
                                    "per fold: " + " ".join(f"{x:.0e}" for x in lams)),
                         "lambda_at_grid_edge": edge,
                         "source": J.get("source", f"full_width/{fname(conv, arm)}")})
    F = pd.DataFrame(fits)
    ref = per[("LEG", "Gemma dense raw-2304")][["subject", "ch"]]
    for k, R in per.items():                       # identical electrode order everywhere
        assert (R.subject.to_numpy() == ref.subject.to_numpy()).all(), k
        assert (R.ch.astype(str).to_numpy() == ref.ch.astype(str).to_numpy()).all(), k
    subj = ref.subject.to_numpy()

    # ---- checks: rotation invariance, reproduction of the stored values, imputation
    for conv in ("LEG", "TO"):
        for a, b in (("Gemma dense raw-2304", "Gemma dense PCA-2304"),
                     ("SAE raw-4933", "SAE PCA-4933")):
            d = per[(conv, a)].r.to_numpy() - per[(conv, b)].r.to_numpy()
            checks[f"{conv}: max |r({a}) - r({b})|"] = float(np.abs(d).max())
        d = per[(conv, "Gemma dense raw-2304 exact")].r - per[(conv, "Gemma dense raw-2304")].r
        checks[f"{conv}: imputation effect, dense 2304 (mean r exact - imputed)"] = float(d.mean())
    a30 = pd.read_csv(cfg.results("dimensionality_sweep", "a30_per_electrode.csv"))
    checks["LEG SAE PCA-4933 vs published sweep point: max |diff|"] = float(
        np.abs(per[("LEG", "SAE PCA-4933")].r.to_numpy() - a30["SAE PCA-4933"].to_numpy()).max())
    checks["LEG Gemma dense PCA-300 vs published arm: max |diff|"] = float(
        np.abs(per[("LEG", "Gemma dense PCA-300")].r.to_numpy()
               - a30["Gemma resid PCA-300"].to_numpy()).max())

    # ---- contrasts: participant-clustered bootstrap and exact sign flip, both conventions
    specs = [("sparse 4,933 PCs - dense 2,304 PCs", "SAE PCA-4933", "Gemma dense PCA-2304"),
             ("sparse raw - dense raw", "SAE raw-4933", "Gemma dense raw-2304"),
             ("sparse 4,933 PCs - dense 300 PCs", "SAE PCA-4933", "Gemma dense PCA-300")]
    rows = []
    for ci, conv in enumerate(("LEG", "TO")):
        for si, (lab, a, b) in enumerate(specs):
            x = per[(conv, a)].r.to_numpy() - per[(conv, b)].r.to_numpy()
            st, _ = PS.electrode_stat(x, subj, cfg.seed + 6100 + 10 * ci + si)
            sf = signflip(x, subj, "t")
            rows.append({"convention": "published" if conv == "LEG" else "training-only",
                         "contrast": lab, "arm_a": a, "arm_b": b,
                         "mean_diff": float(x.mean()), "median_diff": st["estimate"],
                         "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                         "p_bootstrap": st["p_two_sided"], "p_signflip_exact": sf["p_exact"],
                         "signflip_statistic": "t of participant means",
                         "n_assignments": sf["n_assignments"],
                         "participants_negative": sf["n_participants_negative"],
                         "participant_means": sf["participant_values"],
                         "n_electrodes": st["n_electrodes"],
                         "n_participants": st["n_participants"],
                         "bootstrap": "participant-clustered, 10,000 resamples, two-sided",
                         "seed": cfg.seed + 6100 + 10 * ci + si})
    # sensitivity (not requested): the TO sparse arm chose the grid's ceiling in most folds;
    # refitted with the grid extended to 1e8, same seed as the requested contrast
    wf = cfg.results("controls", "full_width") / f"{fname('TO', 'SAE raw-4933 wide-grid')}.csv"
    if wf.exists():
        R, J = load_arm(cfg, "TO", "SAE raw-4933 wide-grid")
        lams = np.asarray(J["lambdas"], float)
        fits.append({"convention": "training-only", "arm": "SAE raw-4933 wide-grid",
                     "masking": J["masking"], "n_electrodes": len(R),
                     "mean_r": float(R.r.mean()), "median_r": float(R.r.median()),
                     "lambda": "per fold: " + " ".join(f"{x:.0e}" for x in lams),
                     "lambda_at_grid_edge": bool((lams == max(J["lambda_grid"])).any()),
                     "source": f"full_width/{fname('TO', 'SAE raw-4933 wide-grid')}"})
        F = pd.DataFrame(fits)
        per[("TO", "SAE raw-4933 wide-grid")] = R
        x = R.r.to_numpy() - per[("TO", "Gemma dense raw-2304")].r.to_numpy()
        st, _ = PS.electrode_stat(x, subj, cfg.seed + 6111)
        sf = signflip(x, subj, "t")
        rows.append({"convention": "training-only",
                     "contrast": "sensitivity: sparse raw - dense raw, penalty grid to 1e8",
                     "arm_a": "SAE raw-4933 wide-grid", "arm_b": "Gemma dense raw-2304",
                     "mean_diff": float(x.mean()), "median_diff": st["estimate"],
                     "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                     "p_bootstrap": st["p_two_sided"], "p_signflip_exact": sf["p_exact"],
                     "signflip_statistic": "t of participant means",
                     "n_assignments": sf["n_assignments"],
                     "participants_negative": sf["n_participants_negative"],
                     "participant_means": sf["participant_values"],
                     "n_electrodes": st["n_electrodes"],
                     "n_participants": st["n_participants"],
                     "bootstrap": "participant-clustered, 10,000 resamples, two-sided",
                     "seed": cfg.seed + 6111})
    C = pd.DataFrame(rows)
    # the third contrast under the original convention, from the stored sweep values
    x = (a30["SAE PCA-4933"] - a30["Gemma resid PCA-300"]).to_numpy()
    checks["LEG sparse 4933 - dense 300 from stored sweep values: median diff"] = float(np.median(x))

    F.to_csv(out / "full_width_fits.csv", index=False)
    C.to_csv(out / "full_width_contrasts.csv", index=False)
    json.dump(checks, open(out / "full_width_checks.json", "w"), indent=1)
    pe = ref.copy()
    for (conv, arm), R in per.items():
        pe[f"{conv}|{arm}"] = R.r.to_numpy()
    pe.to_csv(out / "full_width_per_electrode.csv", index=False)


    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(F[["convention", "arm", "masking", "mean_r", "median_r", "lambda"]].to_string(index=False))
        print()
        print(C[["convention", "contrast", "mean_diff", "median_diff", "ci95_lo", "ci95_hi",
                 "p_bootstrap", "p_signflip_exact", "participants_negative"]].to_string(index=False))
        print()
        for k, v in checks.items():
            print(f"  {k}: {v:.6g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "analyse"), required=True)
    ap.add_argument("--conv", choices=("LEG", "TO"))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    args = ap.parse_args()
    cfg = load_config(HERE.parent / "config" / "config.yaml")
    if args.part == "fit":
        fit(cfg, args.conv, args.arms)
    else:
        analyse(cfg)


if __name__ == "__main__":
    main()
