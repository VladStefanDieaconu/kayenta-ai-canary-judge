#!/usr/bin/env python3
"""The pipeline: runs all three judges against the dummy dataset and prints
their verdicts side by side. This is the functional test (`make validate` calls
it). Exits 0 only if all three judges produced a result.

All three judges run as real Kayenta standalone analyses, so each gets an
execution id you can view/compare in Referee. They differ only by canary config:
  A. Default  : NetflixACAJudge-v1.0 (statistical.json), in-process in Kayenta.
  B. Dummy AI : RemoteJudge-v1.0 (dummy-ai.json, judgeConfigurations.mode=dummy) -> judge-service.
  C. Hybrid   : RemoteJudge-v1.0 (hybrid.json, mode=hybrid) -> judge-service, which
                calls Kayenta back to run the real NetflixACAJudge on the same
                pairs, computes its AI verdict, and returns the merge.

Steps:
  1. Ensure dummy data is present in VictoriaMetrics (seed).
  2. Run analyses A, B, C over the same Control/Experiment scopes.
  3. Print all three (label, score, verdict, per-metric) and the Kayenta execution ids.
  4. Exit 0 only if all three produced a verdict.

The judge logic lives in the judge-service:
  - judge-service/app/dummy_judge.py  (the no-op default verdict)
  - judge-service/app/hybrid.py       (the hybrid merge policy)
A and B are authentic; the hybrid reuses the real NetflixACAJudge via Kayenta.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from kayenta_client import JudgeOutcome, KayentaClient, build_scope, default_window
import seed_dummy_data

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "kayenta" / "canary-configs"
THRESHOLDS = {"pass": 75.0, "marginal": 50.0}

KAYENTA_URL = os.environ.get("KAYENTA_URL", "http://localhost:8090")
VM_URL = os.environ.get("VM_URL", "http://localhost:8428")
REFEREE_URL = os.environ.get("REFEREE_URL", "http://localhost:3001")
# Referee SPA deep-link to the SCAPE Report Viewer (served under /dashboard,
# router basename /dashboard, route /reports/standalone_canary_analysis/:id).
REFEREE_REPORT_PATH = "/dashboard/reports/standalone_canary_analysis/"
LAST_RUN_FILE = REPO_ROOT / "data" / "last_run.json"

SEED_WINDOW_MINS = 30   # what the seeder writes
ANALYSIS_WINDOW_MINS = 25  # what we ask Kayenta to judge (strictly inside seeded range)


def load_config(name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def ensure_data() -> None:
    """Always (re)seed fresh dummy data ending ~now, so the analysis window
    (now-ANALYSIS_WINDOW_MINS .. now) is guaranteed to overlap the samples.
    Seeding is idempotent and cheap; stale data from a previous run would
    otherwise fall outside the window and yield Nodata."""
    import time

    print("[pipeline] (re)seeding dummy data into VictoriaMetrics ...")
    seed_dummy_data.seed(VM_URL, SEED_WINDOW_MINS, 60)
    time.sleep(2)
    if not seed_dummy_data.verify(VM_URL, SEED_WINDOW_MINS, 60):
        raise RuntimeError("dummy data not queryable after seeding")


def fmt_score(s: Optional[float]) -> str:
    return f"{s:.1f}" if s is not None else "n/a"


def print_outcome(o: JudgeOutcome) -> None:
    print(f"  Judge      : {o.judge_label}")
    # The judge's own reported name (AI modes encode the model here).
    if o.judge_name:
        print(f"  Reported   : {o.judge_name}" + (f"  [model={o.model}]" if o.model else ""))
    print(f"  Execution  : {o.execution_id}")
    print(f"  Score      : {fmt_score(o.score)}")
    print(f"  Verdict    : {o.verdict}")
    if o.score_message:
        print(f"  Message    : {o.score_message[:200]}")
    if o.per_metric:
        metrics = ", ".join(f"{m['name']}={m['classification']}" for m in o.per_metric)
        print(f"  Per-metric : {metrics}")
    else:
        print("  Per-metric : (detail not surfaced in result)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all three canary judges against dummy data")
    ap.add_argument("--kayenta-url", default=KAYENTA_URL)
    ap.add_argument("--no-wait", action="store_true", help="skip waiting for Kayenta health")
    ap.add_argument(
        "--only",
        choices=["default", "dummy", "hybrid"],
        help="run a single judge instead of all three",
    )
    args = ap.parse_args()

    client = KayentaClient(args.kayenta_url)

    print("=" * 72)
    print("canaryLLM baseline demo: default + DUMMY-AI + hybrid (NO model)")
    print("(this is the wiring/regression path; the real LLM/VLM judge is `make scenario`)")
    print("=" * 72)

    if not args.no_wait:
        print("[pipeline] waiting for Kayenta /health = UP ...")
        if not client.wait_for_health(timeout_s=240):
            print("[pipeline] ERROR: Kayenta not healthy", file=sys.stderr)
            return 2

    ensure_data()

    start, end = default_window(ANALYSIS_WINDOW_MINS)
    scopes: List[Dict[str, Any]] = [build_scope(start, end, step=60)]
    print(f"[pipeline] analysis window: {start.isoformat()} .. {end.isoformat()}")

    # All three baseline judges run as real Kayenta standalone analyses (each gets
    # an execution id viewable in Referee). They differ only by canary config, and
    # none needs a model (this is the wiring/regression path):
    #   A statistical.json -> NetflixACAJudge-v1.0
    #   B dummy-ai.json    -> RemoteJudge-v1.0, judgeConfigurations.mode=dummy (no-op)
    #   C hybrid.json      -> RemoteJudge-v1.0, judgeConfigurations.mode=hybrid
    #     (the judge-service calls Kayenta back for the real NetflixACAJudge
    #      verdict, computes its AI verdict, and returns the merge)
    all_runs = [
        ("A", "default", "Default statistical (NetflixACAJudge-v1.0)", "statistical.json"),
        ("B", "dummy", "Dummy AI (no-op, NO model, mode=dummy)", "dummy-ai.json"),
        ("C", "hybrid", "Hybrid (NetflixACAJudge + dummy AI, mode=hybrid)", "hybrid.json"),
    ]
    runs = [r for r in all_runs if (args.only is None or r[1] == args.only)]

    outcomes: Dict[str, JudgeOutcome] = {}
    for letter, key, label, cfg_file in runs:
        print(f"\n[pipeline] running {letter} - {label} ...")
        try:
            outcomes[key] = client.run_analysis(label, load_config(cfg_file), scopes, THRESHOLDS)
        except Exception as e:  # noqa: BLE001
            print(f"[pipeline] ERROR running {label}: {e}", file=sys.stderr)
            return 3

    labels = {
        "default": "A. DEFAULT statistical (NetflixACAJudge), real, no model",
        "dummy": "B. DUMMY AI (no-op stub, NO model, wiring only)",
        "hybrid": "C. HYBRID (NetflixACAJudge + dummy AI, merged in-Kayenta)",
    }

    # report
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    for _letter, key, _label, _cfg in runs:
        print(f"\n--- {labels[key]} ---")
        print_outcome(outcomes[key])

    # Record execution ids so `make referee` can deep-link. Merge into any
    # existing file so a single-judge (--only) run doesn't drop the others.
    def report_url(exec_id: str) -> str:
        return f"{REFEREE_URL.rstrip('/')}{REFEREE_REPORT_PATH}{exec_id}"

    record: Dict[str, Any] = {}
    if LAST_RUN_FILE.exists():
        try:
            record = json.loads(LAST_RUN_FILE.read_text())
        except (OSError, ValueError):
            record = {}
    for key, outcome in outcomes.items():
        record[key] = outcome.execution_id
    record["referee_base"] = REFEREE_URL.rstrip("/")
    record["report_path"] = REFEREE_REPORT_PATH
    try:
        LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN_FILE.write_text(json.dumps(record, indent=2))
    except OSError as e:  # non-fatal
        print(f"[pipeline] warning: could not write {LAST_RUN_FILE}: {e}", file=sys.stderr)

    print("\n" + "-" * 72)
    referee_keys = {"default": "make referee", "dummy": "make referee JUDGE=dummy", "hybrid": "make referee JUDGE=hybrid"}
    print("Open the rendered result(s) in Referee (Report Viewer):")
    for _letter, key, _label, _cfg in runs:
        print(f"  {key:8s}: {report_url(outcomes[key].execution_id)}   ({referee_keys[key]})")
    print("-" * 72)

    all_ok = all(o.has_result for o in outcomes.values())
    summary = "  ".join(
        f"{key}={outcomes[key].verdict}({fmt_score(outcomes[key].score)})" for _l, key, _lb, _c in runs
    )
    print(f"\nSummary: {summary}")
    if all_ok:
        n = len(outcomes)
        print(f"\n[OK] {'All ' + str(n) + ' judges' if n > 1 else 'Judge'} produced a verdict. PASS.")
        return 0
    print("\n[FAIL] Not all judges produced a verdict.", file=sys.stderr)
    for key, o in outcomes.items():
        if not o.has_result:
            print(f"  - {key} incomplete: {o.score_message[:300]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
