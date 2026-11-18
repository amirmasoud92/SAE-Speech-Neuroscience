"""H1 at group level: mixed-effects models of peak unique variance.

* the mixed model on region, ``peak ~ C(roi) + (1 | subject)``;
* the same model collapsed to language against control networks;
* a participant-clustered bootstrap of the peak in every region.

Uses the observed variance partitions only. Writes ``group_models/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from inference.electrode_inference import load_observed
from inference.statistics import cluster_bootstrap, mixed_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--band", default="sae")
    args = ap.parse_args()
    cfg = load_config(args.config)

    obs, meta = load_observed(cfg, args.band)
    meta = meta.assign(obs_peak=np.nanmax(obs, axis=1))
    lang = set(cfg.electrodes["language_rois"])
    ctrl = set(cfg.electrodes["control_rois"])
    print(f"band {args.band}: {len(meta)} electrodes, {meta.subject.nunique()} subjects")
    print(f"language ROIs {sorted(lang)} | control ROIs {sorted(ctrl)}")

    out = {}
    out["mixed_model_roi"] = mixed_model(meta, outcome="obs_peak", fixed="roi")
    sub = meta[meta.roi.isin(lang | ctrl)].copy()
    sub["network"] = np.where(sub.roi.isin(lang), "language", "control")
    out["mixed_model_network"] = mixed_model(sub, outcome="obs_peak", fixed="network")

    print(f"\nlanguage {int((sub.network == 'language').sum())} electrodes, "
          f"control {int((sub.network == 'control').sum())}")

    print("\n--- mixed model, peak ~ C(roi) + (1 | subject) ---")
    print(json.dumps(out["mixed_model_roi"], indent=2, default=str)[:1200])
    print("\n--- mixed model, peak ~ C(network) + (1 | subject) ---")
    print(json.dumps(out["mixed_model_network"], indent=2, default=str)[:1200])

    print("\n--- subject-clustered bootstrap of the peak, per ROI ---")
    rows = []
    for roi, g in meta.groupby("roi"):
        if g.subject.nunique() < 3:
            continue
        point, lo, hi = cluster_bootstrap(
            g.obs_peak.to_numpy(), g.subject.to_numpy(),
            n_resamples=int(cfg.stats["bootstrap"]["n_resamples"]),
            ci=float(cfg.stats["bootstrap"]["ci"]), seed=cfg.seed)
        rows.append({"roi": roi, "network": "language" if roi in lang
                     else ("control" if roi in ctrl else "other"),
                     "n": len(g), "n_subjects": int(g.subject.nunique()),
                     "mean_peak": point, "ci_lo": lo, "ci_hi": hi})
    R = pd.DataFrame(rows).sort_values("mean_peak", ascending=False)
    for _, r in R.iterrows():
        star = "  <-- language" if r.network == "language" else (
            "  <-- control" if r.network == "control" else "")
        print(f"   {r.roi:<8s} n={r.n:>4d} ({r.n_subjects} subj)  "
              f"{r.mean_peak:.5f} [{r.ci_lo:.5f}, {r.ci_hi:.5f}]{star}")

    o = cfg.results("group_models")
    o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / f"h1_group_bootstrap_{args.band}.csv", index=False)
    (o / f"h1_group_models_{args.band}.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwritten:", o / f"h1_group_models_{args.band}.json")
    print("\nNOTE: this settles only H1's mixed-model arm. The proportion-of-significant-"
          "electrodes arm still depends on the phase-randomised null and remains open.")


if __name__ == "__main__":
    main()
