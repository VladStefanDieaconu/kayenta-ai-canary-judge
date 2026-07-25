"""Build the input representation the model sees, from the metricSetPairList.

Three representations, all derived from the same paired control/experiment
series the dummy judge already receives:
  - summary : compact per-metric descriptive statistics (cheapest, most stable)
  - raw     : the actual value arrays serialized as text (token-heavy)
  - plot    : a rendered matplotlib chart per metric (for the VLM path)

Bias controls: control is always presented before experiment (fixed position),
the ground-truth label is never included, and numeric precision is fixed.
"""

from __future__ import annotations

import base64
import io
import json
import math
from typing import Any, Dict, List, Optional

import numpy as np

PRECISION = 4


def clean(values: Optional[List[Optional[float]]]) -> List[float]:
    """Drop NaN/None placeholders Kayenta uses for missing samples."""
    out: List[float] = []
    for v in values or []:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        out.append(f)
    return out


def describe(values: List[float]) -> Dict[str, Any]:
    """Descriptive statistics for one series (fixed precision)."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    # simple linear trend (slope per sample) for direction signal
    if arr.size >= 2:
        slope = float(np.polyfit(np.arange(arr.size), arr, 1)[0])
    else:
        slope = 0.0
    r = PRECISION
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), r),
        "median": round(float(np.median(arr)), r),
        "p90": round(float(np.percentile(arr, 90)), r),
        "p95": round(float(np.percentile(arr, 95)), r),
        "p99": round(float(np.percentile(arr, 99)), r),
        "stddev": round(float(arr.std(ddof=0)), r),
        "min": round(float(arr.min()), r),
        "max": round(float(arr.max()), r),
        "slope_per_sample": round(slope, r),
    }


def _pair_values(pair: Dict[str, Any]) -> tuple[List[float], List[float]]:
    values = pair.get("values") or {}
    return clean(values.get("control")), clean(values.get("experiment"))


def _metric_name(pair: Dict[str, Any]) -> str:
    return str(pair.get("name") or pair.get("id") or "metric")


def summary_block(metric_set_pair_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-metric control/experiment stats + deltas/ratios."""
    out: List[Dict[str, Any]] = []
    for pair in metric_set_pair_list:
        control, experiment = _pair_values(pair)
        c, e = describe(control), describe(experiment)
        delta = ratio = None
        if c.get("n") and e.get("n"):
            delta = round(e["mean"] - c["mean"], PRECISION)
            ratio = round(e["mean"] / c["mean"], PRECISION) if c["mean"] else None
        out.append(
            {
                "name": _metric_name(pair),
                "control": c,
                "experiment": e,
                "experiment_minus_control_mean": delta,
                "experiment_over_control_mean": ratio,
            }
        )
    return out


def raw_block(metric_set_pair_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The actual value arrays as text (fixed precision)."""
    out: List[Dict[str, Any]] = []
    for pair in metric_set_pair_list:
        control, experiment = _pair_values(pair)
        out.append(
            {
                "name": _metric_name(pair),
                "control": [round(v, PRECISION) for v in control],
                "experiment": [round(v, PRECISION) for v in experiment],
            }
        )
    return out


def render_plot_png(metric_set_pair_list: List[Dict[str, Any]]) -> Optional[str]:
    """Render one chart per metric (control vs experiment overlaid) into a single
    figure and return a base64-encoded PNG. Returns None if nothing to plot.

    Kept legible (no aggressive downscaling); native-resolution VLMs lose
    chart-reading accuracy when the image is shrunk.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [p for p in metric_set_pair_list if (p.get("values") or {})]
    if not pairs:
        return None

    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False, dpi=110)
    for i, pair in enumerate(pairs):
        ax = axes[i][0]
        control, experiment = _pair_values(pair)
        if control:
            ax.plot(range(len(control)), control, label="Control", color="#1f77b4", marker="o", markersize=3)
        if experiment:
            ax.plot(range(len(experiment)), experiment, label="Experiment", color="#d62728", marker="o", markersize=3)
        ax.set_title(_metric_name(pair))
        ax.set_xlabel("sample index (time)")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def representation_text(mode: str, metric_set_pair_list: List[Dict[str, Any]]) -> str:
    """The text portion of the prompt for the given mode."""
    if mode == "raw":
        block = raw_block(metric_set_pair_list)
        header = (
            "Below are the RAW per-sample value arrays for each metric, for the "
            "baseline (control) and the canary (experiment). Control is listed first."
        )
    else:  # summary (also used as the text companion for plot)
        block = summary_block(metric_set_pair_list)
        header = (
            "Below are per-metric descriptive STATISTICS for the baseline (control) "
            "and the canary (experiment). Control is listed first."
        )
    return f"{header}\n\n```json\n{json.dumps(block, indent=2)}\n```"
