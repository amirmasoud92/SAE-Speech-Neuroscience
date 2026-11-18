"""Statistics shared by the analyses.

* Threshold-free cluster enhancement over lags and max-statistic permutation p-values against
  an explicit surrogate null.
* Benjamini-Hochberg false discovery rate and a generalised-Pareto tail fit beyond the
  permutation floor.
* Linear mixed-effects models of peak unique variance with a per-participant random intercept,
  fitted by restricted maximum likelihood, with convergence and random-effect singularity
  reported.
* Bootstrap resampling clustered by participant, the unit of replication.
* Admissibility and summary helpers for the concentration statistic.

Group-level claims never pool electrodes without a participant random effect, because
electrode counts per region are dictated by clinical coverage.
"""
from __future__ import annotations

import logging

from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

log = logging.getLogger("stats")


# --------------------------------------------------------------------------- TFCE

def tfce_1d(x: np.ndarray, hmax: float | None = None, E: float = 0.5, H: float = 2.0,
            n_steps: int = 50, chunk: int = 4096) -> np.ndarray:
    """Threshold-free cluster enhancement along the last axis.

    Parameters
    ----------
    x : (..., n_lags) float array
        Statistic map. Only positive values are enhanced (one-tailed): the hypotheses
        here are directional and negative unique variance is not interpretable.
    hmax : float, optional
        Top of the integration range. **Pass the same value when enhancing an observed
        map and its null**, otherwise each gets a height grid scaled to its own maximum,
        the step sizes differ, and the two are not on a common scale. When omitted it
        defaults to this array's own maximum, which is only correct for a single map
        considered alone.
    E, H : float
        Standard TFCE exponents (Smith & Nichols 2009).
    n_steps : int
        Number of integration steps.
    chunk : int
        Rows processed at once; bounds peak memory on large null arrays.

    Returns
    -------
    (..., n_lags) float array of enhanced values.
    """
    x = np.asarray(x, dtype=float)
    flat = x.reshape(-1, x.shape[-1])
    out = np.zeros_like(flat)
    top = float(flat.max()) if hmax is None else float(hmax)
    if not np.isfinite(top) or top <= 0:
        return out.reshape(x.shape)

    heights = np.linspace(top / n_steps, top, n_steps)
    step = heights[1] - heights[0] if n_steps > 1 else top
    n_lags = flat.shape[-1]

    for lo in range(0, flat.shape[0], chunk):
        block = flat[lo:lo + chunk]
        n_rows = block.shape[0]
        # A False separator column is appended to every row and the whole block is
        # flattened, so a supra-threshold run can never span two rows. Run extents are
        # then obtained for the entire block with one cumulative sum.
        padded = np.zeros((n_rows, n_lags + 1), dtype=bool)
        acc = np.zeros(n_rows * (n_lags + 1), dtype=np.float64)
        for h in heights:
            padded[:, :n_lags] = block >= h
            fl = padded.reshape(-1)
            if not fl.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(
                ([np.int8(0)], fl.view(np.int8), [np.int8(0)]))))
            starts, ends = edges[::2], edges[1::2]
            lengths = (ends - starts).astype(np.float64)
            acc[:] = 0.0
            np.add.at(acc, starts, lengths)
            np.add.at(acc, ends, -lengths)
            np.cumsum(acc, out=acc)
            out[lo:lo + n_rows] += (acc.reshape(n_rows, n_lags + 1)[:, :n_lags] ** E
                                    * h ** H * step)
    return out.reshape(x.shape)


def _shared_hmax(observed: np.ndarray, null: np.ndarray) -> float:
    """Integration ceiling common to an observed map and its null."""
    return float(max(np.nanmax(observed), np.nanmax(null)))


def surrogate_pvalues(observed: np.ndarray, null: np.ndarray,
                      enhance: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """One-tailed p-values from an explicit surrogate null, with TFCE.

    Parameters
    ----------
    observed : (n_electrodes, n_lags)
    null : (n_permutations, n_electrodes, n_lags)
    enhance : bool
        Apply :func:`tfce_1d` to both before comparing.

    Returns
    -------
    p : (n_electrodes,) float
        ``(1 + #{null_max >= observed_max}) / (1 + n_permutations)``, using each
        electrode's own null. The +1 keeps p strictly positive, as it must be for a
        finite permutation test.
    obs_stat : (n_electrodes,) the maximum enhanced statistic per electrode.
    """
    if null.ndim != 3 or null.shape[1:] != observed.shape:
        raise ValueError(f"null {null.shape} incompatible with observed {observed.shape}")
    if enhance:
        hmax = _shared_hmax(observed, null)
        o = tfce_1d(observed, hmax=hmax)
        n = tfce_1d(null.reshape(-1, null.shape[-1]), hmax=hmax).reshape(null.shape)
    else:
        o, n = observed, null

    obs_stat = o.max(axis=-1)
    null_stat = n.max(axis=-1)                       # (n_perm, n_electrodes)
    n_perm = null_stat.shape[0]
    p = (1.0 + (null_stat >= obs_stat[None, :]).sum(0)) / (1.0 + n_perm)
    return p, obs_stat


def max_statistic_pvalues(observed: np.ndarray, null: np.ndarray,
                          enhance: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Family-wise-error p-values from a max-statistic permutation test.

    Each surrogate contributes a **single** value: the maximum enhanced statistic over
    every electrode and every lag. Comparing each electrode's observed maximum against
    that distribution controls FWER across the whole family with no further correction
    (Nichols & Holmes 2002; Maris & Oostenveld 2007).

    This is the primary inference because it is well resolved at a feasible permutation
    count: the null is built from ``n_perm`` draws of a maximum over ~200,000 statistics,
    so its upper tail is populated even when ``n_perm`` is small.

    Parameters
    ----------
    observed : (n_electrodes, n_lags)
    null : (n_permutations, n_electrodes, n_lags)

    Returns
    -------
    p_fwer : (n_electrodes,) family-wise corrected p-values
    null_max : (n_permutations,) the max-statistic null distribution
    """
    if null.ndim != 3 or null.shape[1:] != observed.shape:
        raise ValueError(f"null {null.shape} incompatible with observed {observed.shape}")
    if enhance:
        hmax = _shared_hmax(observed, null)
        o = tfce_1d(observed, hmax=hmax)
        n = tfce_1d(null.reshape(-1, null.shape[-1]), hmax=hmax).reshape(null.shape)
    else:
        o, n = observed, null

    obs_stat = o.max(axis=-1)                       # (n_electrodes,)
    null_max = n.max(axis=(1, 2))                   # (n_perm,) - max over the family
    p = (1.0 + (null_max[:, None] >= obs_stat[None, :]).sum(0)) / (1.0 + null_max.size)
    return p, null_max


def gpd_tail_pvalue(obs: float, null_stats: np.ndarray, n_exceedances: int = 50,
                    min_exceedances: int = 20,
                    gof_alpha: float = 0.05) -> tuple[float, bool]:
    """p-value for an observation beyond the permutation floor, via a Pareto tail fit.

    A permutation test with ``N`` surrogates cannot report a p-value below ``1/(N+1)``.
    With ~1,264 electrodes under BH-FDR that floor can make a genuinely sparse effect
    undetectable at any effect size. Fitting a generalised Pareto distribution to the
    null's upper tail extends resolution past the floor (Knijnenburg et al. 2009).

    Parameters
    ----------
    obs : observed statistic
    null_stats : (n_perm,) null distribution of the same statistic
    n_exceedances : how many of the largest null values to fit
    min_exceedances : lower bound before abandoning the fit
    gof_alpha : Anderson-Darling significance level for accepting the fit

    Returns
    -------
    p : float
        Tail-extrapolated p-value, or the empirical one if no adequate fit was found.
    used_gpd : bool
        Whether the returned p-value came from the Pareto fit. Always report this - an
        extrapolated p-value carries a modelling assumption an empirical one does not.
    """
    from scipy import stats as sps

    null_stats = np.asarray(null_stats, dtype=float)
    n = null_stats.size
    n_exceed = int((null_stats >= obs).sum())
    empirical = (1.0 + n_exceed) / (1.0 + n)

    # Only extrapolate when the observation is at or beyond the empirical floor.
    if n_exceed > 0:
        return empirical, False

    srt = np.sort(null_stats)
    for m in range(min(n_exceedances, n - 1), min_exceedances - 1, -5):
        thresh = srt[-(m + 1)]
        exc = srt[-m:] - thresh
        if exc.min() < 0 or np.ptp(exc) == 0:
            continue
        try:
            shape, loc, scale = sps.genpareto.fit(exc, floc=0.0)
            gof = sps.anderson_ksamp(
                [exc, sps.genpareto.rvs(shape, loc=loc, scale=scale, size=1000,
                                        random_state=0)])
            if gof.pvalue < gof_alpha:
                continue
            tail = sps.genpareto.sf(obs - thresh, shape, loc=loc, scale=scale)
            p = float(m / n * tail)
            if np.isfinite(p) and 0 < p <= empirical:
                return p, True
        except Exception:  # noqa: BLE001 - a failed fit falls through to the next m
            continue
    return empirical, False


def fdr(p: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns ``(reject, q_values)``."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    q = np.full(p.shape, np.nan)
    if ok.sum():
        r, qq, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
        reject[ok] = r
        q[ok] = qq
    return reject, q


# ------------------------------------------------------------------- effect sizes

def cluster_bootstrap(values: np.ndarray, subjects: np.ndarray, n_resamples: int = 10000,
                      ci: float = 0.95,
                      statistic: Callable[[np.ndarray], float] = np.mean,
                      seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap a statistic by resampling **subjects**, not electrodes.

    Electrodes within a subject are not independent and their counts are very unequal
    across subjects, so resampling electrodes would understate the
    uncertainty. Resampling whole subjects with all their electrodes respects the nesting.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    values = np.asarray(values, dtype=float)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    groups = [values[subjects == s] for s in uniq]
    rng = np.random.default_rng(seed)

    point = float(statistic(values))
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        boot[i] = statistic(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, float(lo), float(hi)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d = mean(a - b) / sd(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


# ------------------------------------------------------------------ group models

def mixed_model(df: pd.DataFrame, outcome: str, fixed: str = "roi",
                group: str = "subject") -> dict:
    """Linear mixed model ``outcome ~ C(fixed) + (1 | group)``.

    The outcome is z-scored before fitting and the coefficients are therefore in SD
    units. This is not cosmetic: the outcomes here are delta-R2 values of order 1e-3,
    and statsmodels' optimiser drives the random-effect variance to the boundary at that
    scale, producing a singular covariance and NaN standard errors. Standardising fixes
    the conditioning; ``outcome_sd`` is returned so any coefficient can be converted back.

    Diagnostics that ordinarily surface only as warnings - non-convergence and a singular
    random-effect covariance - are returned as fields, so a caller cannot miss them.
    """
    import statsmodels.formula.api as smf

    d = df[[outcome, fixed, group]].dropna().copy()
    if d[group].nunique() < 3:
        raise ValueError(f"only {d[group].nunique()} groups - a random intercept is "
                         f"not estimable")
    if d[fixed].nunique() < 2:
        raise ValueError(f"'{fixed}' has a single level - no contrast to estimate")

    mu, sd = d[outcome].mean(), d[outcome].std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"outcome '{outcome}' has zero variance")
    d["_z"] = (d[outcome] - mu) / sd

    model = smf.mixedlm(f"_z ~ C({fixed})", d, groups=d[group])

    # Optimiser order matters. On statsmodels 0.14.4, `method="lbfgs"` returns
    # group_var == 0 exactly while reporting converged=True, silently collapsing the
    # model to unclustered OLS - verified against planted data where the default,
    # "powell" and "cg" all recover the variance correctly. lbfgs is therefore not used.
    # See test_mixed_model_recovers_the_variance_component.
    fit, method_used = None, None
    for method in (None, "powell", "cg"):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                cand = model.fit(reml=True) if method is None \
                    else model.fit(reml=True, method=method)
            if bool(cand.converged):
                fit, method_used = cand, method or "default"
                break
            if fit is None:
                fit, method_used = cand, method or "default"
        except Exception as exc:  # noqa: BLE001 - record and try the next optimiser
            log.warning("MixedLM %s failed: %s", method or "default", exc)
    if fit is None:
        raise RuntimeError("no optimiser converged for the mixed model")

    group_var = float(np.asarray(fit.cov_re)[0, 0])
    singular = group_var < 1e-8
    if singular:
        log.warning("mixed model for '%s': random-effect variance is ~0 (%.3g). The "
                    "subject grouping explains no variance here; fixed effects are "
                    "still interpretable but are effectively unclustered.",
                    outcome, group_var)
    if not bool(fit.converged):
        log.warning("mixed model for '%s' did not converge (method=%s)", outcome,
                    method_used)

    return dict(params=fit.params.to_dict(), bse=fit.bse.to_dict(),
                pvalues=fit.pvalues.to_dict(),
                converged=bool(fit.converged), optimiser=method_used,
                group_var=group_var, singular_random_effect=bool(singular),
                outcome_mean=float(mu), outcome_sd=float(sd),
                units="outcome SD (coefficients are standardised)",
                n_obs=int(fit.nobs), n_groups=int(d[group].nunique()),
                summary=str(fit.summary()))


def temporal_consistency(stat: np.ndarray, lags_s: np.ndarray) -> dict[str, float]:
    """Do electrodes agree on *when* an effect occurs, or does each peak elsewhere?

    Distinguishes two quantities that are easy to conflate and that behave very
    differently under a max-over-lags statistic:

    ``peak_of_mean``
        average over electrodes at each lag, then take the maximum. Only large when
        electrodes share a latency.
    ``mean_of_peak``
        take each electrode's own maximum over lags, then average. Inflated by
        selection: for pure noise it grows with the number of lags searched.

    Their ratio is the diagnostic. Near 1, the effect has a consistent latency. Near 0,
    each electrode peaks somewhere different and the apparent per-electrode effect may
    be nothing more than the maximum of a noisy series - which is exactly what a
    surrogate null is needed to adjudicate.

    Parameters
    ----------
    stat : (n_electrodes, n_lags)
    lags_s : (n_lags,)

    Returns
    -------
    dict with ``peak_of_mean``, ``mean_of_peak``, ``ratio``, ``peak_lag_iqr_s`` and
    ``peak_of_mean_lag_s``.
    """
    stat = np.asarray(stat, dtype=float)
    if stat.ndim != 2 or stat.shape[1] != len(lags_s):
        raise ValueError("stat must be (n_electrodes, n_lags) matching lags_s")
    mean_curve = stat.mean(0)
    pom = float(mean_curve.max())
    mop = float(stat.max(1).mean())
    peaks = lags_s[stat.argmax(1)]
    return dict(peak_of_mean=pom, mean_of_peak=mop,
                ratio=float(pom / mop) if mop != 0 else float("nan"),
                peak_lag_iqr_s=float(np.percentile(peaks, 75)
                                     - np.percentile(peaks, 25)),
                peak_of_mean_lag_s=float(lags_s[int(np.argmax(mean_curve))]))


def peak_lag(stat: np.ndarray, lags_s: np.ndarray,
             restrict: tuple[float, float] | None = None) -> np.ndarray:
    """Lag of the maximum of ``stat`` along the last axis, in seconds.

    Parameters
    ----------
    stat : (..., n_lags)
    lags_s : (n_lags,)
    restrict : optional (lo, hi) window in seconds to search within.
    """
    stat = np.asarray(stat, float)
    if stat.shape[-1] != len(lags_s):
        raise ValueError("stat and lags_s disagree on the lag axis")
    sel = np.ones(len(lags_s), dtype=bool)
    if restrict is not None:
        sel = (lags_s >= restrict[0]) & (lags_s <= restrict[1])
    idx = np.argmax(np.where(sel, stat, -np.inf), axis=-1)
    return lags_s[idx]


# ------------------------------------------------------------------ SCI restriction

def admissible_sci(r2_all_real: np.ndarray, r2_all_ctrl: np.ndarray,
                   sci_real: np.ndarray, sci_ctrl: np.ndarray,
                   delta_sae: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce per-lag SCI to one admissible value per electrode.

    ``SCI_k = R2(top-k) / R2(all)`` is only interpretable where the denominator is
    meaningfully positive. Across all electrode x lag targets it usually is not - most
    lags sit far from any response - so the raw ratio has a negative median and excursions
    far outside [0, 1]. This applies the pre-specified restriction:

    1. one lag per electrode, ``argmax`` of ``delta_sae`` from the main variance
       partition - a different fit, blind to top-k selection, and biased against H2
       because a larger denominator lowers SCI for the real arm;
    2. an electrode is admissible only if ``R2(all) > 0`` in **both** arms at that lag,
       which is symmetric and so cannot favour either.

    Parameters
    ----------
    r2_all_real, r2_all_ctrl : (n_ch, n_lags) held-out R2 of the full band, per arm
    sci_real, sci_ctrl : (n_ch, n_lags) the SCI_k ratios, per arm
    delta_sae : (n_ch, n_lags) unique SAE variance from the main variance partition

    Returns
    -------
    dict with ``lag_idx`` (n_ch,), boolean ``admissible`` (n_ch,), and ``real``/``ctrl``
    (n_ch,) SCI values at the selected lag - ``nan`` where not admissible.
    """
    for name, a in [("r2_all_ctrl", r2_all_ctrl), ("sci_real", sci_real),
                    ("sci_ctrl", sci_ctrl), ("delta_sae", delta_sae)]:
        if a.shape != r2_all_real.shape:
            raise ValueError(f"{name} has shape {a.shape}, expected {r2_all_real.shape}")

    n_ch = r2_all_real.shape[0]
    rows = np.arange(n_ch)
    # nan-safe: an all-nan electrode would make argmax meaningless, so it is excluded
    # below by the positivity test rather than silently selecting lag 0.
    lag_idx = np.nanargmax(np.where(np.isfinite(delta_sae), delta_sae, -np.inf), axis=1)

    rr = r2_all_real[rows, lag_idx]
    rc = r2_all_ctrl[rows, lag_idx]
    admissible = (rr > 0) & (rc > 0) & np.isfinite(rr) & np.isfinite(rc)

    real = np.where(admissible, sci_real[rows, lag_idx], np.nan)
    ctrl = np.where(admissible, sci_ctrl[rows, lag_idx], np.nan)
    # A ratio can still be non-finite if the numerator is; drop those symmetrically so
    # the paired test never sees a pair with only one usable side.
    both = np.isfinite(real) & np.isfinite(ctrl)
    admissible &= both
    real = np.where(admissible, real, np.nan)
    ctrl = np.where(admissible, ctrl, np.nan)

    return {"lag_idx": lag_idx, "admissible": admissible, "real": real, "ctrl": ctrl}


def paired_sci_test(real: np.ndarray, ctrl: np.ndarray,
                    subjects: np.ndarray | None = None,
                    n_resamples: int = 10000, seed: int = 0) -> dict[str, float]:
    """Paired real-vs-control test over admissible electrodes.

    One-sided: H2 predicts ``SCI_real > SCI_ctrl``. Wilcoxon signed-rank is the primary
    test because SCI is a ratio and not remotely Gaussian. When ``subjects`` is given, a
    cluster bootstrap over subjects accompanies it, since electrodes within a subject are
    not independent and two subjects supply most of the IFG coverage.
    """
    from scipy import stats as sps

    m = np.isfinite(real) & np.isfinite(ctrl)
    real, ctrl = real[m], ctrl[m]
    out: dict[str, float] = {"n": int(m.sum())}
    if out["n"] < 3:
        return {**out, "p_wilcoxon": float("nan"), "median_real": float("nan"),
                "median_ctrl": float("nan"), "win_rate": float("nan")}

    out["median_real"] = float(np.median(real))
    out["median_ctrl"] = float(np.median(ctrl))
    out["win_rate"] = float((real > ctrl).mean())
    out["cohens_d"] = cohens_d_paired(real, ctrl)
    try:
        out["p_wilcoxon"] = float(sps.wilcoxon(real, ctrl, alternative="greater").pvalue)
    except ValueError:               # all differences zero
        out["p_wilcoxon"] = 1.0

    # A cluster bootstrap over a single subject resamples that subject every time and
    # returns a zero-width interval, which looks like an impossibly precise estimate
    # rather than the absence of one. Report nothing instead.
    if subjects is not None and len(np.unique(subjects[m])) >= 2:
        point, lo, hi = cluster_bootstrap(real - ctrl, subjects[m],
                                          n_resamples=n_resamples, seed=seed)
        out["diff_mean"] = float(point)
        out["diff_ci_lo"], out["diff_ci_hi"] = float(lo), float(hi)
    return out


def exceedance_fractions(share_real: np.ndarray, share_ctrl: np.ndarray,
                         tau: float = 0.5) -> tuple[float, float, np.ndarray]:
    """Fraction of electrodes whose share exceeds ``tau``, per arm, and the paired diff.

    A non-finite share counts as *not* exceeding: it arises where the arm's own
    ``R2(all)`` is non-positive, which is a failure to explain variance, not missing data.
    Treating it as missing would drop exactly the electrodes where a control legitimately
    fails and so would flatter the control.
    """
    a = np.where(np.isfinite(share_real), share_real, -np.inf) > tau
    b = np.where(np.isfinite(share_ctrl), share_ctrl, -np.inf) > tau
    return float(a.mean()), float(b.mean()), a.astype(float) - b.astype(float)


def pooled_exceedance_test(share_real: np.ndarray, share_ctrl: np.ndarray,
                           subjects: np.ndarray, tau: float = 0.5,
                           n_resamples: int = 10000, seed: int = 0) -> dict[str, float]:
    """Co-primary H2 test: pooled exceedance fractions with a subject-clustered CI.

    Electrodes are pooled across subjects, so a subject contributes in proportion to how
    many admissible electrodes it has rather than one vote each. The interval comes from
    resampling **subjects**, because electrodes within a subject are not independent and
    their counts differ by more than 3x across this dataset.
    """
    fr, fc, diff = exceedance_fractions(share_real, share_ctrl, tau)
    out = {"n_electrodes": int(len(diff)), "n_subjects": int(len(np.unique(subjects))),
           "frac_real": fr, "frac_ctrl": fc, "diff": fr - fc}
    if out["n_subjects"] >= 2:
        point, lo, hi = cluster_bootstrap(diff, subjects, n_resamples=n_resamples,
                                          seed=seed)
        out["diff_ci_lo"], out["diff_ci_hi"] = float(lo), float(hi)
    return out


def paired_subject_test(frac_real: np.ndarray, frac_ctrl: np.ndarray) -> dict[str, float]:
    """Co-primary H2 test: paired test over per-subject exceedance fractions.

    One vote per subject. Ties are dropped, which is correct for a paired test but means a
    subject with too few admissible electrodes to separate the arms contributes nothing -
    the reason the pooled test accompanies this one.
    """
    from scipy import stats as sps

    d = np.asarray(frac_real, float) - np.asarray(frac_ctrl, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    out = {"n_subjects": int(len(d)), "n_untied": int(len(nz)),
           "n_real_higher": int((nz > 0).sum())}
    if len(nz) == 0:
        return {**out, "p_sign": float("nan"), "p_wilcoxon": float("nan")}
    # Exact sign test: the probability of this many or more agreeing under p = 1/2.
    out["p_sign"] = float(sps.binomtest(int((nz > 0).sum()), len(nz), 0.5,
                                        alternative="greater").pvalue)
    try:
        out["p_wilcoxon"] = float(sps.wilcoxon(nz, alternative="greater").pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    return out
