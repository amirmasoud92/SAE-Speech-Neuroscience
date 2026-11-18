"""Effect of single-precision rounding on the primary-cohort H1 statistic.

Refits the H1 design (full sparse band, surprisal, 17-delay acoustic band, lexical band) in
double precision, kernel form, at the 21 lags from 0 to 500 ms, and compares it with the
single-precision variance partition read at the same lags. The full partition is to be
re-run in double precision if the language-versus-control coefficient differs by more than
0.05 SD, if its P value crosses 0.05, or if the per-electrode peak correlation is below 0.99.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from inference.statistics import mixed_model

LAGS = slice(80, 101)


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    out = cfg.results("film_h1_h2")
    lang, ctrl = set(cfg.electrodes["language_rois"]), set(cfg.electrodes["control_rois"])
    rows = []
    for sub in cfg.subjects:
        v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"), allow_pickle=True)
        z = np.load(out / "podcast_bridge" / f"{sub}_original_float64.npz", allow_pickle=True)
        assert np.array_equal(z["ch_names"].astype(str), v["ch_names"].astype(str))
        for i, roi in enumerate(v["roi"].astype(str)):
            rows.append({"subject": sub, "roi": roi,
                         "float32": float(np.nanmax(v["delta_sae"][i, LAGS])),
                         "float64": float(np.nanmax(z["delta_sae"][i]))})
    T = pd.DataFrame(rows)
    r = float(np.corrcoef(T.float32, T.float64)[0, 1])
    res = {"n_electrodes": len(T), "peak_correlation": r,
           "median_abs_diff": float(np.median(np.abs(T.float32 - T.float64))),
           "max_abs_diff": float(np.max(np.abs(T.float32 - T.float64)))}
    for col in ("float32", "float64"):
        d = T[T.roi.isin(lang | ctrl)].copy()
        d["network"] = np.where(d.roi.isin(lang), "language", "control")
        m = mixed_model(d.rename(columns={col: "obs_peak"}), outcome="obs_peak", fixed="network")
        res[f"beta_{col}"] = float(m["params"]["C(network)[T.language]"])
        res[f"p_{col}"] = float(m["pvalues"]["C(network)[T.language]"])
    trig = (abs(res["beta_float64"] - res["beta_float32"]) > 0.05
            or (res["p_float32"] < 0.05) != (res["p_float64"] < 0.05) or r < 0.99)
    res["rerun_required"] = bool(trig)
    T.to_csv(out / "step0_float32_peaks.csv", index=False)
    (out / "step0_float32_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("STEP 0:", "RE-RUN REQUIRED" if trig else "no re-run; report in Methods")


if __name__ == "__main__":
    main()
