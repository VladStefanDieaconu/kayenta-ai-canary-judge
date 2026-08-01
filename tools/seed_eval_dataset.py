#!/usr/bin/env python3
"""Seed the labelled evaluation dataset into VictoriaMetrics and write the manifest.

Generates the nine scenario families (tools/eval_dataset.py), loads every
instance's control/experiment series into VictoriaMetrics with the framework's
`Canary="Control"`/`"Experiment"` scheme (only the Canary label, per-instance
namespaced metric names for isolation), writes data/eval_manifest.json, and emits
the programmatically-generated per-scenario statistical configs under
data/eval-configs/.

`--probe` additionally runs the real NetflixACAJudge (via Kayenta's /judges/judge
on the same pairs) over the first instance of every family and prints its verdict
plus per-metric classification, the fastest way to confirm the data embodies each
statistical blind spot before the full experiment.

Usage:
  python tools/seed_eval_dataset.py [--n 20] [--seed 20260621] [--probe]

`--n` and `--seed` default to $EVAL_N / $EVAL_SEED when set, matching
run_experiment.py, so a direct invocation seeds the same dataset the experiment
expects. Flags still win over the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import requests

# .env must be loaded before any argparse default reads os.environ; see the
# module docstring for the incident this prevents.
import repo_env  # noqa: E402,F401
import eval_dataset as ed
import judge_clients

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "eval_manifest.json"
CONFIG_DIR = REPO_ROOT / "data" / "eval-configs"
VM_URL = "http://localhost:8428"

DEFAULT_JUDGE = {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}}


def build_payload(scenarios: List[ed.Scenario], base_millis: int) -> str:
    lines: List[str] = []
    step_ms = ed.STEP_SECONDS * 1000
    for sc in scenarios:
        series = ed.series_for_scenario(sc)
        for m in sc.metrics:
            vals = series[m.name]
            for i in range(ed.N_SAMPLES):
                ts = base_millis + i * step_ms
                lines.append(f'{m.name}{{Canary="Control"}} {vals["control"][i]} {ts}')
                lines.append(f'{m.name}{{Canary="Experiment"}} {vals["experiment"][i]} {ts}')
    return "\n".join(lines) + "\n"


def seed_vm(payload: str, vm_url: str = VM_URL) -> int:
    n = payload.count("\n")
    url = f"{vm_url.rstrip('/')}/api/v1/import/prometheus"
    r = requests.post(url, data=payload.encode("utf-8"), timeout=120)
    r.raise_for_status()
    try:
        requests.get(f"{vm_url.rstrip('/')}/internal/force_flush", timeout=10)
    except requests.RequestException:
        pass
    print(f"[seed-eval] imported {n} samples into {url}")
    return n


def verify_pairing(scenarios: List[ed.Scenario], base_millis: int, vm_url: str = VM_URL) -> bool:
    """Confirm a sampling of metrics is queryable for both scopes (pairs, no Nodata)."""
    sample = [scenarios[0], scenarios[len(scenarios) // 2], scenarios[-1]]
    start = base_millis // 1000 - 120
    end = base_millis // 1000 + ed.N_SAMPLES * ed.STEP_SECONDS + 120
    ok = True
    for sc in sample:
        for m in sc.metrics:
            for scope in ("Control", "Experiment"):
                r = requests.get(f"{vm_url.rstrip('/')}/api/v1/query_range", params={
                    "query": f'sum without(Canary) ({m.name}{{Canary="{scope}"}})',
                    "start": start, "end": end, "step": ed.STEP_SECONDS}, timeout=15)
                r.raise_for_status()
                res = r.json().get("data", {}).get("result", [])
                npts = len(res[0]["values"]) if res else 0
                ok = ok and npts > 0
                print(f"[seed-eval] verify {sc.id:24s} {m.name:38s} {scope:10s} {npts} pts")
    return ok


def write_configs(scenarios: List[ed.Scenario]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for sc in scenarios:
        cfg = ed.build_canary_config(sc, DEFAULT_JUDGE)
        (CONFIG_DIR / f"{sc.id}.json").write_text(json.dumps(cfg, indent=2))
    print(f"[seed-eval] wrote {len(scenarios)} per-scenario configs to {CONFIG_DIR}")


def probe(scenarios: List[ed.Scenario], base_millis: int) -> None:
    """Run the real NetflixACAJudge on one instance per family; show its verdict
    vs ground truth, to validate the dataset design against the real judge."""
    print("\n" + "=" * 86)
    print("PROBE: genuine NetflixACAJudge (best-practice config) on instance 000 of each family")
    print("=" * 86)
    print("{:<22} {:<6} {:<8} {:>6}  {}".format("FAMILY", "TRUTH", "STAT", "SCORE", "PER-METRIC"))
    print("-" * 86)
    by_family: Dict[str, ed.Scenario] = {}
    for sc in scenarios:
        by_family.setdefault(sc.family, sc)
    for family in ed.FAMILIES:
        sc = by_family[family]
        series = ed.series_for_scenario(sc)
        pairs = ed.build_pairs(sc, series, base_millis)
        cfg = ed.build_canary_config(sc, DEFAULT_JUDGE)
        out = judge_clients.run_default_judge(pairs, cfg, ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"])
        if not out["ok"]:
            print(f"{family:<22} {sc.truth:<6} ERROR    {out['error']}")
            continue
        pm = ", ".join(f"{p['name'].split('_g')[0].split('_')[-1]}={p['classification']}" for p in out["per_metric"])
        agree = "OK" if (out["verdict"] == sc.truth) else ("blind" if sc.truth == "FAIL" else "FP")
        print("{:<22} {:<6} {:<8} {:>6}  {}  [{}]".format(
            family, sc.truth, out["verdict"], f"{out['score']:.0f}", pm, agree))
    print("-" * 86)
    print("Expected: PASS families -> STAT PASS; clean_mean_shift/subtle -> STAT FAIL;")
    print("variance/tail/gradual_drift/cross_metric -> STAT PASS (the structural blind spots).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the labelled evaluation dataset")
    ap.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", ed.DEFAULT_N_PER_FAMILY)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", ed.DEFAULT_MASTER_SEED)))
    ap.add_argument("--vm-url", default=VM_URL)
    ap.add_argument("--probe", action="store_true", help="also run the genuine NetflixACAJudge per family")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    scenarios = ed.build_scenarios(args.seed, args.n)
    base_millis = ed.aligned_base_millis()
    print(f"[seed-eval] {len(scenarios)} scenarios ({len(ed.FAMILIES)} families x {args.n}), "
          f"seed={args.seed}, window={ed.WINDOW_MINS}m step={ed.STEP_SECONDS}s")

    payload = build_payload(scenarios, base_millis)
    seed_vm(payload, args.vm_url)

    manifest = ed.build_manifest(args.seed, args.n, scenarios, base_millis)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[seed-eval] wrote manifest {MANIFEST} ({len(scenarios)} scenarios)")
    write_configs(scenarios)

    import time as _t
    _t.sleep(2)
    if not args.no_verify:
        if not verify_pairing(scenarios, base_millis, args.vm_url):
            print("[seed-eval] FAIL: some metrics did not pair (Nodata)", file=sys.stderr)
            return 1
    if args.probe:
        probe(scenarios, base_millis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
