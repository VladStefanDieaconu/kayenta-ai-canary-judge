#!/usr/bin/env python3
"""Seed a scenario dataset that exposes blind spots in Kayenta's NetflixACAJudge
(Mann-Whitney): cases where the default judge returns PASS for a canary a human
would roll back. Based on spinnaker/spinnaker issue #6278 ("How to Overcome
MannWhitney Judge Shortcomings"), Case 1 (increase in metric variability), and
the rank test's documented insensitivity to temporal order and tail behaviour.

The Mann-Whitney U test compares the median (location) of two distributions. It
is blind to (a) a change in variance at the same median, and (b) an emerging
trend whose median hasn't moved yet. `direction` (increase/decrease/either) and
`effectSize` thresholds only gate the location shift, so they don't help here.

Two realistic Prometheus metrics (real names and units):

  http_request_duration_seconds   (p99 latency, seconds; the canonical Prometheus
      latency histogram; operationally `histogram_quantile(0.99, ...)`)
    control:    stable ~0.20s (tight).
    experiment: same median ~0.20s but ~20x the variance, tail latency flapping
      0.07s..0.34s. A bad, unstable canary; Mann-Whitney sees no median shift, so PASS.

  process_resident_memory_bytes   (RSS, bytes; standard process-collector gauge)
    control:    stable ~600 MiB.
    experiment: flat ~600 MiB for most of the window, then an emerging leak ramp
      in the last quarter up to ~760 MiB. Median unchanged (most of the window is
      baseline), so Mann-Whitney says PASS, but the tail/trend is a clear regression.

Both series carry only the `Canary` label (Control/Experiment), like the rest of
the harness, so `sum without(Canary) (...)` selects exactly one series per scope.
Deterministic (no RNG) and idempotent. Uses separate metric names from the dummy
dataset, so `make validate` is unaffected.

Usage: python tools/seed_scenario_data.py [--vm-url http://localhost:8428]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from statistics import mean, median, pstdev
from typing import Dict, List

import requests

DEFAULT_WINDOW_MINS = 30
DEFAULT_STEP_SECONDS = 60
MIB = 1024 * 1024


def _latency_control(i: int) -> float:
    # tight, stable ~0.20s
    return round(0.200 + 0.006 * math.sin(i / 2.0), 6)


def _latency_experiment(i: int) -> float:
    # Same median ~0.20s, ~20x variance: irregular flapping (two incommensurate
    # sines plus a small deterministic jitter), symmetric about 0.20 so the median
    # doesn't move, which is what Mann-Whitney is blind to.
    osc = 0.090 * math.sin(i * 1.7) + 0.050 * math.sin(i * 0.7) + 0.020 * math.sin(i * 3.1)
    return round(0.200 + osc, 6)


def _memory_control(i: int) -> float:
    return float(int(600 * MIB + 6 * MIB * math.sin(i / 2.5)))


def _memory_experiment(i: int, n: int) -> float:
    # flat ~600 MiB for the first 75% of the window, then an emerging leak ramp
    # in the last quarter to ~760 MiB. Most samples are still baseline, so the
    # median is unchanged (Mann-Whitney says PASS), but the end-of-window trend
    # and the p95/p99/max are clearly elevated.
    ramp_start = int(n * 0.75)
    base = 600 * MIB + 6 * MIB * math.sin(i / 2.5)
    if i >= ramp_start:
        frac = (i - ramp_start) / max(1, (n - 1 - ramp_start))
        base += frac * 160 * MIB
    return float(int(base))


METRICS = {
    "http_request_duration_seconds": (_latency_control, lambda i, n: _latency_experiment(i)),
    "process_resident_memory_bytes": (_memory_control, _memory_experiment),
}


def build_payload(window_mins: int, step: int) -> str:
    now = int(time.time())
    now -= now % step
    start = now - window_mins * 60
    timestamps = list(range(start, now + 1, step))
    n = len(timestamps)

    lines: List[str] = []
    for metric, (control_fn, experiment_fn) in METRICS.items():
        for idx, ts in enumerate(timestamps):
            ts_ms = ts * 1000
            c = control_fn(idx)
            e = experiment_fn(idx, n)
            lines.append(f'{metric}{{Canary="Control"}} {c} {ts_ms}')
            lines.append(f'{metric}{{Canary="Experiment"}} {e} {ts_ms}')
    return "\n".join(lines) + "\n"


def seed(vm_url: str, window_mins: int, step: int) -> int:
    payload = build_payload(window_mins, step)
    n_points = payload.count("\n")
    import_url = f"{vm_url.rstrip('/')}/api/v1/import/prometheus"
    resp = requests.post(import_url, data=payload.encode("utf-8"), timeout=30)
    resp.raise_for_status()
    try:
        requests.get(f"{vm_url.rstrip('/')}/internal/force_flush", timeout=10)
    except requests.RequestException:
        pass
    print(f"[scenario-seed] imported {n_points} samples into {import_url}")
    return n_points


def _series(fn_kind: str, window_mins: int, step: int):
    """Recompute the in-memory series (for the pre-flight summary)."""
    n = window_mins * 60 // step + 1
    out: Dict[str, Dict[str, List[float]]] = {}
    for metric, (cfn, efn) in METRICS.items():
        out[metric] = {
            "control": [cfn(i) for i in range(n)],
            "experiment": [efn(i, n) for i in range(n)],
        }
    return out


def preflight(window_mins: int, step: int) -> None:
    """Print the stats that show why Mann-Whitney is fooled: medians match while
    variance/tail differ."""
    data = _series("", window_mins, step)
    print("\n[scenario-seed] why the default (Mann-Whitney) judge is fooled:")
    for metric, s in data.items():
        for role in ("control", "experiment"):
            v = s[role]
            p99 = sorted(v)[max(0, int(len(v) * 0.99) - 1)]
            print(f"  {metric:34s} {role:11s} median={median(v):.4g} mean={mean(v):.4g} "
                  f"std={pstdev(v):.4g} max={max(v):.4g} p99={p99:.4g}")
    print("  -> medians match per metric (MW sees no shift) while variance/tail differ.\n")


def verify(vm_url: str, window_mins: int, step: int) -> bool:
    now = int(time.time())
    start = now - (window_mins + 2) * 60
    ok = True
    for metric in METRICS:
        for scope in ("Control", "Experiment"):
            r = requests.get(
                f"{vm_url.rstrip('/')}/api/v1/query_range",
                params={"query": f'{metric}{{Canary="{scope}"}}', "start": start, "end": now, "step": step},
                timeout=15,
            )
            r.raise_for_status()
            res = r.json().get("data", {}).get("result", [])
            n = len(res[0]["values"]) if res else 0
            print(f"[scenario-seed] verify {metric} Canary={scope}: {n} points")
            ok = ok and n > 0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the Mann-Whitney blind-spot scenario dataset")
    ap.add_argument("--vm-url", default="http://localhost:8428")
    ap.add_argument("--window-mins", type=int, default=DEFAULT_WINDOW_MINS)
    ap.add_argument("--step", type=int, default=DEFAULT_STEP_SECONDS)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    preflight(args.window_mins, args.step)
    seed(args.vm_url, args.window_mins, args.step)
    time.sleep(2)
    if args.no_verify:
        return 0
    return 0 if verify(args.vm_url, args.window_mins, args.step) else 1


if __name__ == "__main__":
    sys.exit(main())
