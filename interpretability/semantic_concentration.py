"""H2 inference: does cortical variance concentrate on few learned features?

* Admissible electrodes: one lag per electrode, the peak of the unique sparse-band variance in
  the variance partition; admissible only where R2(all) > 0 in both arms compared.
* Control: a random dictionary matched to the learned one feature by feature on activation
  density, and a column-shuffled control.
* Statistic: the fraction of admissible electrodes whose share R2(top-20) / R2(all) exceeds
  tau = 0.5, in the language network and in IFG.
* Two co-primary tests: a one-sided sign test across participants (one vote each) and the
  pooled difference over electrodes with a participant-clustered bootstrap interval.

Two participants on which the statistic and threshold were chosen are reported as
exploratory and are never pooled with the others. Reads saved arrays only. Writes
``concentration/``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import Config, load_config
from utils.run_logging import setup_logging
from inference.statistics import (admissible_sci, exceedance_fractions, paired_subject_test,
                                  pooled_exceedance_test)

log = logging.getLogger("sci_inference")

EXPLORATORY = ("sub-03", "sub-09")      # used to choose the statistic and threshold
ARMS = ("real", "random_dict", "shuffled")
CTRL = "share_random_dict"


def _shares(cfg: Config, sub: str, k: int) -> pd.DataFrame | None:
    """Per-electrode share for every arm, restricted to the admissible set."""
    f = cfg.results("basis_fits", f"sci_{sub}.npz")
    if not f.exists():
        return None
    s = np.load(f, allow_pickle=True)
    v = np.load(cfg.results("encoding", f"{sub}_variance_partition.npz"),
                allow_pickle=True)
    if not np.array_equal(s["ch_names"].astype(str), v["ch_names"].astype(str)):
        raise ValueError(f"{sub}: channel order differs between the SCI file and the "
                         f"variance partition; A1 pairs them electrode by electrode")

    d = v["delta_sae"]
    rows = np.arange(d.shape[0])
    li = np.nanargmax(np.where(np.isfinite(d), d, -np.inf), axis=1)
    adm = admissible_sci(s["real__r2_all"], s["random_dict__r2_all"],
                         s[f"real__sci_{k}"], s[f"random_dict__sci_{k}"], d)

    out = pd.DataFrame({"subject": sub, "ch": s["ch_names"].astype(str),
                        "roi": v["roi"].astype(str),
                        "admissible": adm["admissible"],
                        "peak_lag_s": s["lags_s"][li]})
    for arm in ARMS:
        num = s[f"{arm}__r2_top{k}"][rows, li]
        den = s[f"{arm}__r2_all"][rows, li]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"share_{arm}"] = np.where(den > 0, num / den, np.nan)
        out[f"r2_all_{arm}"] = den
        out[f"r2_top_{arm}"] = num
    return out[out.admissible].drop(columns="admissible")


def _curve(real: np.ndarray, ctrl: np.ndarray) -> str:
    """real/random exceedance percentages across tau, so the 0.5 cut is visible."""
    parts = []
    for t in np.arange(0.0, 1.0, 0.1):
        fr, fc, _ = exceedance_fractions(real, ctrl, t)
        parts.append(f"{100 * fr:.0f}/{100 * fc:.0f}")
    return " ".join(parts)


def _report(log_, df: pd.DataFrame, tau: float, seed: int) -> dict:
    """Per-participant fractions, then both co-primary tests over the confirmatory
    participants. Returns the test results so they can be written to a table.
    """
    if df.empty:
        log_.warning("  no admissible electrodes in this scope")
        return {}

    per = []
    for sub, g in df.groupby("subject", sort=False):
        fr, fc, _ = exceedance_fractions(g.share_real.to_numpy(), g[CTRL].to_numpy(), tau)
        fs, _, _ = exceedance_fractions(g.share_shuffled.to_numpy(),
                                        g.share_shuffled.to_numpy(), tau)
        per.append({"subject": sub, "n": len(g), "real": fr, "ctrl": fc, "shuf": fs,
                    "exploratory": sub in EXPLORATORY})
    per = pd.DataFrame(per)

    log_.info("  %-9s %5s %8s %8s %9s", "subject", "n", "real", "random", "shuffled")
    for r in per.itertuples():
        log_.info("  %-9s %5d %7.0f%% %7.0f%% %8.0f%%   %s", r.subject, r.n,
                  100 * r.real, 100 * r.ctrl, 100 * r.shuf,
                  "exploratory" if r.exploratory else "CONFIRMATORY")

    conf = per[~per.exploratory]
    cdf = df[~df.subject.isin(EXPLORATORY)]
    if conf.empty:
        log_.info("  no confirmatory subject has completed yet")
        return {}

    p = paired_subject_test(conf.real.to_numpy(), conf.ctrl.to_numpy())
    log_.info("  [A3 paired]   %d subjects, %d untied, %d favour real -> p_sign=%.4g",
              p["n_subjects"], p["n_untied"], p["n_real_higher"], p["p_sign"])
    if p["n_untied"] and 0.5 ** p["n_untied"] > 0.05:
        log_.warning("  [A3 paired]   floor is p=%.3f with %d untied subjects - cannot "
                     "reach 0.05 however large the effect", 0.5 ** p["n_untied"],
                     p["n_untied"])

    real, ctrl = cdf.share_real.to_numpy(), cdf[CTRL].to_numpy()
    pool = pooled_exceedance_test(real, ctrl, cdf.subject.to_numpy(), tau=tau, seed=seed)
    ci = (f"  95% CI [{pool['diff_ci_lo']:+.3f}, {pool['diff_ci_hi']:+.3f}]"
          if "diff_ci_lo" in pool else "  (CI needs >= 2 subjects)")
    log_.info("  [A3.2 pooled] %d electrodes, %d subjects: real %.0f%% vs random %.0f%%"
              " -> diff %+.3f%s", pool["n_electrodes"], pool["n_subjects"],
              100 * pool["frac_real"], 100 * pool["frac_ctrl"], pool["diff"], ci)
    log_.info("  [A3.2 curve]  real/random above tau=0.0..0.9: %s", _curve(real, ctrl))
    shuf = cdf.share_shuffled.to_numpy()
    return {"n_subjects_conf": int(p["n_subjects"]), "n_untied": int(p["n_untied"]),
            "n_real_higher": int(p["n_real_higher"]), "p_sign_one_sided": p["p_sign"],
            "p_wilcoxon_one_sided": p.get("p_wilcoxon", float("nan")),
            "n_electrodes_conf": int(pool["n_electrodes"]),
            "pct_real": 100 * float(pool["frac_real"]),
            "pct_random": 100 * float(pool["frac_ctrl"]),
            "pct_shuffled": 100 * float(np.mean(shuf > tau)) if len(shuf) else float("nan"),
            "diff": float(pool["diff"]), "diff_ci_lo": pool.get("diff_ci_lo", float("nan")),
            "diff_ci_hi": pool.get("diff_ci_hi", float("nan")),
            "real_leads_all_tau": bool(all(exceedance_fractions(real, ctrl, t)[0]
                                           > exceedance_fractions(real, ctrl, t)[1]
                                           for t in np.arange(0.0, 1.0, 0.1)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_ = setup_logging(cfg.paths.results_root / "logs", "sci_inference", cfg.digest)
    log_.info("full config digest (incl. presentation): %s", cfg.full_digest)
    log_.info("A3 statistic: fraction of admissible electrodes with "
              "R2(top%d)/R2(all) > %.2f", args.k, args.tau)

    frames, missing = [], []
    for sub in cfg.subjects:
        got = _shares(cfg, sub, args.k)
        if got is None:
            missing.append(sub)
        else:
            frames.append(got)
    if missing:
        log_.info("not yet computed: %s", ",".join(missing))
    if not frames:
        log_.warning("no subject has SCI output yet")
        return

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cfg.results("concentration", f"sci_shares_k{args.k}.csv"), index=False)

    lang = list(cfg.electrodes["language_rois"])
    rows = []
    for name, scope, sel in [("PRIMARY IFG", "IFG", df.roi == "IFG"),
                             ("SECONDARY language " + "+".join(lang), "language",
                              df.roi.isin(lang))]:
        log_.info("")
        log_.info("--- %s   (tau=%.2f, k=%d) ---", name, args.tau, args.k)
        r = _report(log_, df[sel], args.tau, cfg.seed)
        rows.append({"scope": scope, "tau": args.tau, "k": args.k, **r})
    pd.DataFrame(rows).to_csv(cfg.results("concentration", f"h2_summary_k{args.k}.csv"),
                              index=False)
    log_.info("")
    log_.info("sci inference finished")


if __name__ == "__main__":
    main()
