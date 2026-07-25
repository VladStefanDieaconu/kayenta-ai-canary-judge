"""The statistical-ensemble judge: a pure, dependency-free decision module, in
the same spirit as hybrid_policy.py. No FastAPI/pydantic/Kayenta imports, so both
a host tool and (optionally) the service can import it, and it's easy to unit-test.

It never reimplements the Mann-Whitney judge. It calls the genuine NetflixACAJudge
(via judge_clients.run_default_judge, the same way the hybrid does) for the
baseline per-metric verdict, which already handles `clean_mean_shift` and
`subtle_regression` well, and adds one purpose-built, pure-stdlib test per
documented blind spot. A metric is escalated from Pass to High only when the
baseline missed it and a blind-spot test fires:

  - variance_increase  -> a rank test (Mann-Whitney U, normal approximation)
                          on |x - combined_median| (a Brown-Forsythe/Levene-type
                          test for equal spread), plus a variance-ratio gate.
  - tail_regression    -> a high-quantile ratio (default p90) plus a rank test
                          restricted to the upper half of the pooled sample.
  - gradual_drift      -> the Mann-Kendall trend test on the (experiment -
                          control) difference series (isolates the canary's
                          own drift from any shared trend).
  - cross_metric_marginal -> Fisher's method combining each metric's own
                          location-test p-value (Mann-Whitney U on the raw
                          series) into one joint p-value; fires only when every
                          individual metric was itself a clean Pass (the family's
                          defining property) but the joint signal is significant.

All p-values use closed-form / normal-approximation formulas (no scipy): the
normal CDF via math.erf, and the chi-square survival function for the always-even
degrees of freedom Fisher's method produces (2 x number of combined p-values),
which has an exact finite-sum closed form, no special functions needed.

Six tunable knobs total, reported explicitly so the configuration cost (how many
knobs it needs, how brittle it is) stays visible:
  variance_ratio_threshold, variance_alpha,
  tail_ratio_threshold, tail_alpha,
  trend_alpha,
  cross_metric_alpha
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from series_guards import detect_recovered_transient

PASS_LIKE = {"Pass", "Nodata"}


# Pure-stdlib statistical primitives.
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mannwhitney_u_p(a: List[float], b: List[float]) -> float:
    """Two-sided p-value for the Mann-Whitney U rank test, normal approximation
    with a tie correction. Same test family Kayenta's own judge uses, but we
    don't reimplement its decision; we just borrow the standard test for a
    different transformed view of the data, e.g. absolute deviations."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    combined = sorted((v, 0) for v in a) + sorted((v, 1) for v in b)
    combined.sort(key=lambda t: t[0])
    ranks: List[float] = [0.0] * len(combined)
    i = 0
    tie_term = 0.0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed average rank over the tie block
        t = j - i
        tie_term += t ** 3 - t
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(r for r, (v, grp) in zip(ranks, combined) if grp == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    if n <= 1:
        return 1.0
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return 1.0
    sigma = math.sqrt(sigma2)
    u = max(u1, n1 * n2 - u1)  # symmetric continuity correction toward the larger side
    z = (u - mu - 0.5) / sigma
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return max(0.0, min(1.0, p))


def mann_kendall_p(series: List[float]) -> Tuple[float, float]:
    """Mann-Kendall trend test. Returns (S statistic, two-sided p-value)."""
    n = len(series)
    if n < 3:
        return (0.0, 1.0)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            s += (diff > 0) - (diff < 0)
    # tie correction for the variance (groups of equal values)
    from collections import Counter
    counts = Counter(series)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return (float(s), 1.0)
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return (float(s), max(0.0, min(1.0, p)))


def chi2_sf_even_df(x: float, k: int) -> float:
    """Survival function 1-CDF of a chi-square distribution with even degrees
    of freedom df=2k. Exact closed form (no incomplete-gamma special function
    needed): P(X > x) = exp(-x/2) * sum_{i=0}^{k-1} (x/2)^i / i!"""
    if x <= 0 or k <= 0:
        return 1.0
    half_x = x / 2.0
    total = 0.0
    term = 1.0  # (half_x)^0 / 0!
    for i in range(k):
        if i > 0:
            term *= half_x / i
        total += term
    return max(0.0, min(1.0, math.exp(-half_x) * total))


def fisher_combine_p(pvalues: List[float]) -> float:
    """Fisher's method: combine independent p-values into one joint p-value."""
    eps = 1e-300
    clipped = [min(max(p, eps), 1.0) for p in pvalues]
    x = -2.0 * sum(math.log(p) for p in clipped)
    return chi2_sf_even_df(x, len(clipped))


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# The four blind-spot tests.
def variance_test(control: List[float], experiment: List[float],
                  ratio_threshold: float, alpha: float) -> Tuple[bool, str]:
    combined = sorted(control + experiment)
    med = _quantile(combined, 0.5)
    dev_c = [abs(v - med) for v in control]
    dev_e = [abs(v - med) for v in experiment]
    p = mannwhitney_u_p(dev_c, dev_e)
    sd_c = (sum((v - sum(control) / len(control)) ** 2 for v in control) / max(1, len(control) - 1)) ** 0.5
    sd_e = (sum((v - sum(experiment) / len(experiment)) ** 2 for v in experiment) / max(1, len(experiment) - 1)) ** 0.5
    ratio = (sd_e / sd_c) if sd_c > 0 else (float("inf") if sd_e > 0 else 1.0)
    fired = (p < alpha) and (ratio >= ratio_threshold)
    reason = f"variance test: stddev ratio {ratio:.2f}x (p={p:.4f})" if fired else ""
    return fired, reason


def tail_test(control: List[float], experiment: List[float],
             ratio_threshold: float, alpha: float, quantile: float = 0.90) -> Tuple[bool, str]:
    sc, se = sorted(control), sorted(experiment)
    qc, qe = _quantile(sc, quantile), _quantile(se, quantile)
    ratio = (qe / qc) if qc > 0 else (float("inf") if qe > 0 else 1.0)
    combined = sorted(control + experiment)
    upper_cut = _quantile(combined, 0.5)
    upper_c = [v for v in control if v >= upper_cut]
    upper_e = [v for v in experiment if v >= upper_cut]
    p = mannwhitney_u_p(upper_c, upper_e) if (upper_c and upper_e) else 1.0
    fired = (ratio >= ratio_threshold) and (p < alpha)
    reason = f"tail test: p{int(quantile*100)} ratio {ratio:.2f}x (upper-half p={p:.4f})" if fired else ""
    return fired, reason


def trend_test(control: List[float], experiment: List[float], alpha: float) -> Tuple[bool, str]:
    n = min(len(control), len(experiment))
    diff = [experiment[i] - control[i] for i in range(n)]
    s, p = mann_kendall_p(diff)
    fired = (p < alpha) and (s > 0)  # only an upward (worsening) trend counts
    reason = f"trend test: Mann-Kendall S={s:.0f} (p={p:.4f}, rising)" if fired else ""
    return fired, reason


# Config + top-level decision.
@dataclass
class EnsembleConfig:
    variance_ratio_threshold: float = 2.0
    variance_alpha: float = 0.05
    tail_ratio_threshold: float = 1.15
    tail_alpha: float = 0.05
    trend_alpha: float = 0.05
    cross_metric_alpha: float = 0.05
    # Structural ablation switch (not a tuned magnitude, so kept out of the
    # 6-knob count below). Without it, the variance/tail tests fire on
    # `healed_transient` 15/15 (see series_guards.py's docstring for why a plain
    # distributional test can't tell "elevated-then-recovered" apart from
    # "constantly elevated"). Defaults to True because the ensemble isn't
    # competitive without it; report it as an ablation (with it / without it)
    # rather than a tunable threshold.
    use_recovery_guard: bool = True

    def knob_count(self) -> int:
        return 6

    def as_dict(self) -> Dict[str, Any]:
        return {
            "variance_ratio_threshold": self.variance_ratio_threshold,
            "variance_alpha": self.variance_alpha,
            "tail_ratio_threshold": self.tail_ratio_threshold,
            "tail_alpha": self.tail_alpha,
            "trend_alpha": self.trend_alpha,
            "cross_metric_alpha": self.cross_metric_alpha,
            "use_recovery_guard": self.use_recovery_guard,
        }


@dataclass
class EnsembleMetricResult:
    name: str
    classification: str  # Pass | High | Nodata | (whatever the baseline returned, escalated)
    baseline_classification: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class EnsembleDecision:
    verdict: str          # Pass | Fail
    score: float
    per_metric: List[EnsembleMetricResult]
    cross_metric_fired: bool
    cross_metric_p: Optional[float]
    reason: str


def decide_ensemble(
    pairs: List[Dict[str, Any]],
    baseline_per_metric: List[Dict[str, Any]],
    baseline_score: float,
    cfg: EnsembleConfig,
    pass_t: float = 75.0,
    marginal_t: float = 50.0,
) -> EnsembleDecision:
    """Combine the genuine statistical judge's per-metric baseline (not
    reimplemented, passed in from judge_clients.run_default_judge's own
    per_metric output) with the four new blind-spot tests, run on the same raw
    (control, experiment) arrays already fetched for the baseline call."""
    baseline_by_name = {m["name"]: m for m in baseline_per_metric}
    per_metric: List[EnsembleMetricResult] = []
    metric_pvalues: List[float] = []
    any_fired = False

    for pair in pairs:
        name = str(pair.get("name") or pair.get("id") or "metric")
        values = pair.get("values") or {}
        control = [float(v) for v in (values.get("control") or []) if v is not None]
        experiment = [float(v) for v in (values.get("experiment") or []) if v is not None]
        base = baseline_by_name.get(name, {})
        base_class = str(base.get("classification", "Pass"))

        reasons: List[str] = []
        fired = False
        if control and experiment:
            metric_pvalues.append(mannwhitney_u_p(control, experiment))
            recovered = False
            if cfg.use_recovery_guard:
                recovered, early_r, late_r = detect_recovered_transient(control, experiment)
                if recovered:
                    reasons.append(f"recovery guard: early={early_r:.2f}x late={late_r:.2f}x "
                                   f"(elevated-then-recovered shape; variance/tail tests suppressed)")
            if base_class in PASS_LIKE and not recovered:
                for test_fn, *test_args in (
                    (variance_test, control, experiment, cfg.variance_ratio_threshold, cfg.variance_alpha),
                    (tail_test, control, experiment, cfg.tail_ratio_threshold, cfg.tail_alpha),
                    (trend_test, control, experiment, cfg.trend_alpha),
                ):
                    hit, reason = test_fn(*test_args)
                    if hit:
                        fired = True
                        reasons.append(reason)

        final_class = "High" if (base_class in PASS_LIKE and fired) else base_class
        if final_class not in PASS_LIKE:
            any_fired = True
        per_metric.append(EnsembleMetricResult(
            name=name, classification=final_class, baseline_classification=base_class, reasons=reasons))

    # Cross-metric joint signal: only meaningful with >=2 metrics, and only
    # fires when every metric was otherwise a clean baseline Pass (the
    # cross_metric_marginal family's defining property: each metric alone is
    # in-tolerance, but jointly the combined p-value is significant).
    cross_fired = False
    cross_p: Optional[float] = None
    all_clean = all(m.baseline_classification in PASS_LIKE for m in per_metric)
    if len(metric_pvalues) >= 2 and all_clean and not any_fired:
        cross_p = fisher_combine_p(metric_pvalues)
        if cross_p < cfg.cross_metric_alpha:
            cross_fired = True
            any_fired = True

    verdict = "Fail" if any_fired else "Pass"
    if any_fired:
        score = min(baseline_score, marginal_t - 0.01) if baseline_score >= marginal_t else baseline_score
    else:
        score = max(baseline_score, pass_t)

    reason_bits = [r for m in per_metric for r in m.reasons]
    if cross_fired:
        reason_bits.append(f"cross-metric Fisher combination: joint p={cross_p:.4f} "
                           f"across {len(metric_pvalues)} individually-clean metrics")
    reason = "; ".join(reason_bits) if reason_bits else "no blind-spot test fired; baseline stands"

    return EnsembleDecision(
        verdict=verdict, score=score, per_metric=per_metric,
        cross_metric_fired=cross_fired, cross_metric_p=cross_p, reason=reason,
    )
