#!/usr/bin/env python3
"""Reproducibility statement and a test-retest check.

Re-runs a handful of scenarios on the frontier model to report verdict
stability. Kept to ~5 scenarios since each run makes a real model call.

Temperature confirmation (done by reading the running code/config, not just
trusting prose): judge-service/app/judge_config.py resolves
`temperature=_env_float("JUDGE_TEMPERATURE", 0.0)`. 0.0 is both the coded
default and the value set in .env (`JUDGE_TEMPERATURE=0`). Every AI/hybrid call
in this experiment, including the frontier model, runs at temperature 0.

Test-retest: the 5 scenarios are chosen as representative cases
(no_change_000, variance_increase_000, tail_regression_000, gradual_drift_000,
cross_metric_marginal_000) from the same published master seed (20260621) and
N=20 dataset, so this re-probes already-discussed cases rather than drawing a
new sample. Each is sent to the frontier model's best representation (ai:raw)
twice back-to-back via judge_clients.judge_ai (the same call path
tools/run_frontier_experiment.py uses for the frontier run), and verdict/score
stability between the two calls is reported directly (in the spirit of the
existing single-scenario determinism preflight, but across a small sample
instead of just one case).

Output: results/agg/test_retest.csv (+ a one-line temperature note).

Usage:
  python tools/reproducibility_test_retest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402

AGG_DIR = REPO_ROOT / "results" / "agg"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]
MASTER_SEED = ed.DEFAULT_MASTER_SEED
N_PER_FAMILY = 20
TARGET_IDS = [
    "no_change_000",
    "variance_increase_000",
    "tail_regression_000",
    "gradual_drift_000",
    "cross_metric_marginal_000",
]
MODEL = "claude-opus-4-8-vlm"
MODE = "raw"

TEMPERATURE_NOTE = (
    "judge-service/app/judge_config.py resolves temperature via "
    "_env_float(\"JUDGE_TEMPERATURE\", 0.0); .env sets JUDGE_TEMPERATURE=0. "
    "Confirmed from the running code/config: "
    "every AI/hybrid call in this study, including the frontier model, runs at "
    "temperature 0, top_p=1.0, seed=42 (dropped by LiteLLM for backends without "
    "seed support)."
)


def main() -> int:
    import csv

    print("[test-retest] " + TEMPERATURE_NOTE)

    scenarios = {sc.id: sc for sc in ed.build_scenarios(MASTER_SEED, N_PER_FAMILY)}
    missing = [tid for tid in TARGET_IDS if tid not in scenarios]
    if missing:
        print(f"[test-retest] ERROR: scenario ids not found in the N=20 dataset: {missing}", file=sys.stderr)
        return 1

    base_millis = ed.aligned_base_millis()
    header = ["scenario", "family", "truth", "model", "mode", "run", "verdict", "score",
              "latency", "ok", "error", "rationale"]
    rows: List[List[Any]] = []
    stability: List[Dict[str, Any]] = []

    for tid in TARGET_IDS:
        sc = scenarios[tid]
        series = ed.series_for_scenario(sc)
        pairs = ed.build_pairs(sc, series, base_millis)
        cfg = ed.build_canary_config(sc, {})

        run_results = []
        for run_idx in (1, 2):
            t0 = time.time()
            ai = judge_clients.judge_ai(pairs, cfg, MODE, MODEL, PASS_T, MARGINAL_T)
            elapsed = time.time() - t0
            run_results.append(ai)
            rows.append([sc.id, sc.family, sc.truth, MODEL, MODE, run_idx,
                        ai["verdict"], round(float(ai["score"]), 2), round(float(ai["latency"]), 2),
                        int(bool(ai["ok"])), (ai.get("error") or "")[:140],
                        (ai.get("rationale") or "")[:200].replace("\n", " ")])
            print(f"  {sc.id:26s} run{run_idx}: verdict={ai['verdict']:5s} score={ai['score']:.1f} "
                  f"({elapsed:.1f}s)")

        v1, v2 = run_results[0]["verdict"], run_results[1]["verdict"]
        s1, s2 = float(run_results[0]["score"]), float(run_results[1]["score"])
        stability.append({
            "scenario": sc.id, "family": sc.family, "truth": sc.truth,
            "verdict_1": v1, "verdict_2": v2, "verdict_match": v1 == v2,
            "score_1": s1, "score_2": s2, "score_diff": round(abs(s1 - s2), 2),
        })

    with (AGG_DIR / "test_retest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        w.writerow([])
        w.writerow(["scenario", "family", "truth", "verdict_1", "verdict_2", "verdict_match",
                    "score_1", "score_2", "score_diff"])
        for s in stability:
            w.writerow([s["scenario"], s["family"], s["truth"], s["verdict_1"], s["verdict_2"],
                       int(s["verdict_match"]), s["score_1"], s["score_2"], s["score_diff"]])
        w.writerow([])
        w.writerow(["temperature_note"])
        w.writerow([TEMPERATURE_NOTE])

    n_match = sum(1 for s in stability if s["verdict_match"])
    print(f"\n[test-retest] verdict stability: {n_match}/{len(stability)} scenarios identical across the 2 runs")
    max_score_diff = max(s["score_diff"] for s in stability)
    print(f"[test-retest] max |score_1 - score_2| across the 5 scenarios: {max_score_diff}")
    print(f"[test-retest] wrote {AGG_DIR / 'test_retest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
