"""Bounded confidence-interval helpers (pure stdlib).

The multi-seed CIs (results/agg/metrics_with_ci.csv,
family_recall_with_ci.csv, both written by tools/bounded_confidence_intervals.py's
mean_ci95) use a Wald t-interval on the 5 per-seed rates. On a rate near 0 or 1
with n=5, that interval can fall outside [0, 1], e.g. the published
tail_regression [-0.071, 0.151] and subtle_regression [0.849, 1.071]. This
module is imported by tools/bounded_confidence_intervals.py.

Two bounded methods, used together:
  - wilson_interval(k, n): Wilson score interval on pooled (successes, trials)
    counts. Stays inside [0, 1] by construction for any k, n (including k=0 or
    k=n).
  - percentile_bootstrap(values, statistic): nonparametric percentile
    bootstrap over a resampled-with-replacement population. When `values` are
    themselves already bounded (rates in [0, 1], or per-scenario rows used to
    recompute a metric), every resample statistic is too, so the interval is
    bounded automatically. Used as the seed-level (or row-level) cross-check,
    and as the only interval for metrics with no closed-form binomial reading
    (F1).

is_zero_variance() flags rates that are bit-identical across every seed (a
structural finding, e.g. the statistical judge's 0/5 on variance_increase in
all 5 seeds) so callers can report the flat rate plainly. An interval on such a
rate would imply sampling uncertainty the data does not contain.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence, Tuple

Z95 = 1.959963985


def wilson_interval(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for k successes out of n trials. Bounded to [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (center - half) / denom
    hi = (center + half) / denom
    return (max(0.0, lo), min(1.0, hi))


def percentile_bootstrap(
    values: Sequence,
    statistic: Callable[[Sequence], float],
    B: int = 10000,
    conf: float = 0.95,
    rng_seed: int = 20260710,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for `statistic` applied to resamples of `values`.

    A constant `values` list (zero variance) collapses to a zero-width
    interval at that constant.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(rng_seed)
    stats = []
    for _ in range(B):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(resample))
    stats.sort()
    lo_idx = int(round((1 - conf) / 2 * (B - 1)))
    hi_idx = int(round((1 + conf) / 2 * (B - 1)))
    hi_idx = min(hi_idx, B - 1)
    return (stats[lo_idx], stats[hi_idx])


def is_zero_variance(values: Sequence[float], tol: float = 1e-12) -> bool:
    if len(values) <= 1:
        return True
    m = values[0]
    return all(abs(v - m) < tol for v in values)


def confusion_counts(rows, true_key: str = "y_true", pred_key: str = "y_pred", pos: str = "FAIL"):
    """Same formula as tools/run_experiment.py::confusion(), generalised to
    whatever the truth/prediction column names are in the source CSV (that
    script's rows use truth/verdict; results/agg/long_results.csv uses
    y_true/y_pred)."""
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        t = r[true_key] == pos
        p = r[pred_key] == pos
        if t and p:
            c["TP"] += 1
        elif (not t) and p:
            c["FP"] += 1
        elif (not t) and (not p):
            c["TN"] += 1
        else:
            c["FN"] += 1
    return c


def metrics_from_conf(c):
    """Identical formulas to tools/run_experiment.py::metrics_from_conf()."""
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr}


def r3(x) -> float:
    return round(float(x), 3)


def holm_correction(pvalues: Sequence[float]) -> list:
    """Holm-Bonferroni step-down family-wise correction. Returns adjusted
    p-values in the same order as the input (not sorted)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def benjamini_hochberg(pvalues: Sequence[float]) -> list:
    """Benjamini-Hochberg step-up FDR correction (q-values), same order as input."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(1.0, pvalues[idx] * m / (rank + 1))
        running_min = min(running_min, val)
        q[idx] = running_min
    return q
