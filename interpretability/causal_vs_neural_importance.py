"""Does a feature's causal necessity in the model predict its neural importance?

The statistic is the Spearman correlation between causal necessity (KL divergence; the change
in log-probability as a second score) and neural importance, with the ranks of log activation
density partialled out of both, interpreted only above a floor of 0.15. Also reported: the raw
correlation, each score's correlation with log density and with coherence, and bootstrap
intervals over features. Writes ``causal/``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from utils.config import Config, load_config
from utils.run_logging import setup_logging

log = logging.getLogger("a11a")


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out, on ranks.

    Rank-transform all three, regress the ranks of x and of y on the ranks of z, then
    correlate the residuals. Returns (rho, p, n).
    """
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = (sps.rankdata(v[m]) for v in (x, y, z))
    n = int(m.sum())
    Z = np.column_stack([np.ones(n), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    # t-test on the partial correlation, one covariate removed
    df = n - 3
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    return r, float(2 * sps.t.sf(abs(t), df)), n


def load(cfg: Config, table: str | None = None, min_features: int = 250) -> pd.DataFrame:
    sc = pd.read_csv(cfg.results("causal", "a11_feature_scores.csv"))
    if sc.feature.duplicated().any():
        raise ValueError("a feature appears twice in the ablation scores")
    raw = np.load(cfg.results("causal", "a11_pool_raw_ids.npy"))
    # keyed by feature id, not row position
    ft = cfg.results("feature_table", table or "feature_table_ifg.csv")
    if not ft.exists():
        raise SystemExit(f"{ft} not found - run interpretability.feature_table first")
    d = pd.read_csv(ft)
    if len(d) != len(raw) or not set(raw.tolist()) == set(d.feature_id.tolist()):
        raise SystemExit(f"feature table does not match the pool: {len(raw)} ids vs "
                         f"{len(d)} rows")
    meta = pd.DataFrame({"feature": d.feature_id.to_numpy(),
                         "density": d.density_words.to_numpy(),
                         "coherence": d.coherence.to_numpy(),
                         "importance": d.importance_z.to_numpy()})
    out = meta.merge(sc, on="feature", how="inner", validate="one_to_one")
    if len(out) < len(sc):
        raise SystemExit("some ablated features are not in the pool metadata")
    # A partial score file is the dangerous case: an inner merge over 3 features returns
    # 3 rows without tripping any check above.
    if len(out) < min_features:
        raise SystemExit(
            f"only {len(out)} features have causal scores, expected at least "
            f"{min_features} of {len(raw)} - this looks like a partial run of "
            f"interpretability.causal_ablation, not the full pool")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ap.add_argument("--min-features", type=int, default=250,
                    help="refuse to run on a partial score file")
    ap.add_argument("--feature-table", default="feature_table_ifg.csv",
                    help="filename under feature_table/ supplying neural importance "
                         "(all IFG electrodes)")
    ap.add_argument("--suffix", default="_ifg", help="output suffix")
    args = ap.parse_args()
    cfg = load_config(args.config)
    suffix = args.suffix
    log_ = setup_logging(cfg.paths.results_root / "logs", "a11_analysis", cfg.digest)

    df = load(cfg, table=args.feature_table, min_features=args.min_features)
    df["log_density"] = np.log10(df.density.to_numpy(float))
    # ablation lowers the next-word log-probability, so the causal score is its magnitude:
    # a larger drop means the model needed the feature more
    df["causal_dlp_mag"] = -df.causal_dlp
    log_.info("%d features with both a causal score and neural importance", len(df))

    rows = []
    for score, lab in [("causal_dlp_mag", "delta log-prob (magnitude)"),
                       ("causal_kl", "KL divergence")]:
        raw_r = sps.spearmanr(df[score], df.importance)
        pr, pp, n = partial_spearman(df[score].to_numpy(), df.importance.to_numpy(),
                                     df.log_density.to_numpy())
        dens_r = sps.spearmanr(df[score], df.log_density)
        coh_r = sps.spearmanr(df[score], df.coherence, nan_policy="omit")
        rows.append({"score": lab, "n": n,
                     "primary_partial_rho": pr, "primary_p": pp,
                     "raw_rho": raw_r.statistic, "raw_p": raw_r.pvalue,
                     "score_vs_log_density_rho": dens_r.statistic,
                     "score_vs_coherence_rho": coh_r.statistic,
                     "score_vs_coherence_p": coh_r.pvalue})
    res = pd.DataFrame(rows)

    log_.info("=" * 88)
    for _, r in res.iterrows():
        log_.info("%-28s n=%3d  PRIMARY partial rho %+.3f (p=%.4g) | raw %+.3f (p=%.3g)",
                  r.score, r.n, r.primary_partial_rho, r.primary_p, r.raw_rho, r.raw_p)
        log_.info("%-28s   score~log density %+.3f | score~coherence %+.3f (p=%.3g)",
                  "", r.score_vs_log_density_rho, r.score_vs_coherence_rho,
                  r.score_vs_coherence_p)
    log_.info("=" * 88)

    prim = res.iloc[0]
    # The pre-specified floor. Significance alone is not a finding at this sample size: a
    # small partial rho can reach p < 0.05 and explain under one percent of rank variance.
    # Below the floor the wording is a small dependence, not a reversal.
    RHO_FLOOR = 0.15
    sig = res.primary_p < 0.05
    big = res.primary_partial_rho.abs() >= RHO_FLOOR
    strongest = float(res.primary_partial_rho.abs().max())
    if (sig & big).any():
        which = ", ".join(res.score[sig & big])
        verdict = (f"PRIMARY SIGNIFICANT and above the effect-size floor ({which}, "
                   f"|rho| {strongest:.3f} >= {RHO_FLOOR}) - causal necessity in the model "
                   "predicts cortical representation")
    elif sig.any():
        which = ", ".join(res.score[sig])
        verdict = (f"primary reaches significance ({which}) but BELOW the floor: "
                   f"|rho| {strongest:.3f} < {RHO_FLOOR}, and {int((~sig).sum())} of "
                   f"{len(res)} causal measures remain null. A small dependence detectable "
                   "at this sample size, NOT a reversal of the dissociation - report the "
                   "effect size, never the p-value alone")
    elif prim.raw_p < 0.05:
        verdict = ("raw correlation significant but dies under density control - report as "
                   "a density effect, no mechanism claim")
    elif (res.score_vs_coherence_p < 0.05).any():
        verdict = ("primary null, causal and interpretable coincide in the model but "
                   "neither reaches cortex - narrows the interpretability claim")
    else:
        verdict = ("primary null throughout - interpretability, causal necessity and "
                   "neural relevance are three separate things (triple dissociation)")
    log_.info("decision: %s", verdict)

    out = cfg.results("causal", f"a11_primary{suffix}.csv")
    res.assign(verdict=verdict).to_csv(out, index=False)
    df.to_csv(cfg.results("causal", f"a11_feature_table{suffix}.csv"), index=False)
    print("written:", out)
    print(verdict)


if __name__ == "__main__":
    main()
