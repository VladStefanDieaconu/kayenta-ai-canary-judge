"""Shared, pure-stdlib shape detectors on raw (control, experiment) series.

Used by the statistical-ensemble judge (ensemble_judge.py, to stop its
variance/tail tests firing on `healed_transient`) and by the hybrid policy's
false-positive guards (hybrid_policy.py, to suppress a gated-hybrid FAIL on
`healed_transient`/`noise_equivalent`). Kept in one place so there's a single
definition of each shape, not two copies that drift apart.

These aren't distributional significance tests (that's ensemble_judge.py's job).
They're simple structural checks on where in the window a difference sits, which
is exactly the information a location/rank test (and a plain variance/tail test)
discards.
"""

from __future__ import annotations

from typing import List, Tuple


def detect_recovered_transient(
    control: List[float],
    experiment: List[float],
    early_frac: float = 0.5,
    late_frac: float = 0.25,
    deg_ratio_threshold: float = 1.3,
    recovered_ratio_tolerance: float = 1.10,
) -> Tuple[bool, float, float]:
    """True if `experiment` looks elevated in the early part of the window and
    back near `control`'s baseline by the end, the `healed_transient` shape.
    Returns (recovered, early_ratio, late_ratio) so callers can log why.
    """
    n = min(len(control), len(experiment))
    if n == 0 or not control:
        return False, 1.0, 1.0
    early_n = max(1, int(n * early_frac))
    late_n = max(1, int(n * late_frac))
    c_mean = sum(control) / len(control)
    if c_mean == 0:
        return False, 1.0, 1.0
    early_ratio = (sum(experiment[:early_n]) / early_n) / c_mean
    late_ratio = (sum(experiment[-late_n:]) / late_n) / c_mean
    recovered = (early_ratio >= deg_ratio_threshold) and (late_ratio <= recovered_ratio_tolerance)
    return recovered, early_ratio, late_ratio


def detect_equal_variance_noise(
    control: List[float],
    experiment: List[float],
    mean_ratio_tolerance: float = 1.03,
    stddev_ratio_tolerance: float = 1.5,
) -> Tuple[bool, float]:
    """True if control and experiment have both (near-)equal means and
    (near-)equal spread, i.e. how `noise_equivalent` is actually built (both
    high-variance, same underlying distribution, no location shift).

    Checking the mean alone isn't enough, which was a real bug caught during
    validation: `variance_increase` also has an equal mean by design (it's a
    variance-only regression), and each individual metric of
    `cross_metric_marginal` looks unremarkable in isolation on both mean and
    spread (that family's defining property is that no single metric looks bad,
    only the joint pattern does). Requiring the spread to also be near-equal
    correctly excludes `variance_increase` (whose spread ratio is deliberately
    ~6-10x, not ~1x). `cross_metric_marginal` is excluded structurally instead,
    by scoping this guard to single-metric scenarios (see its caller), since
    that family is multi-metric by construction and this one is not.
    """
    if not control or not experiment:
        return False, 1.0
    c_mean = sum(control) / len(control)
    e_mean = sum(experiment) / len(experiment)
    if c_mean == 0:
        return (e_mean == 0), (1.0 if e_mean == 0 else float("inf"))
    mean_ratio = e_mean / c_mean
    equal_mean = abs(mean_ratio - 1.0) <= (mean_ratio_tolerance - 1.0)

    sd_c = (sum((v - c_mean) ** 2 for v in control) / max(1, len(control) - 1)) ** 0.5
    sd_e = (sum((v - e_mean) ** 2 for v in experiment) / max(1, len(experiment) - 1)) ** 0.5
    sd_ratio = max(sd_e, sd_c) / sd_c if sd_c > 0 else (1.0 if sd_e == 0 else float("inf"))
    equal_spread = sd_ratio <= stddev_ratio_tolerance

    return (equal_mean and equal_spread), mean_ratio
