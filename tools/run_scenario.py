#!/usr/bin/env python3
"""Run the Mann-Whitney blind-spot scenario across all judges and contrast them.

Seeds the scenario dataset (unstable tail latency plus an emerging memory leak,
both with an unchanged median, see tools/seed_scenario_data.py and issue #6278),
then runs the Kayenta standalone analyses over it:

  scenario-statistical.json  -> NetflixACAJudge-v1.0           (expected: PASS, the miss)
  scenario-ai-summary.json   -> RemoteJudge, mode=summary LLM  (expected: catches it)
  scenario-ai-raw.json       -> RemoteJudge, mode=raw LLM
  scenario-ai-plot.json      -> RemoteJudge, mode=plot VLM

Prints a side-by-side table and a Referee deep-link for each. Every run is a real
Kayenta execution, unlike `make validate-ai`, which posts to the judge directly and
so has no execution to render. Records the execution ids to data/last_run.json so
`make referee JUDGE=scenario-default|scenario-summary|...` opens the report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from kayenta_client import KayentaClient, build_scope, default_window
import seed_scenario_data

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "kayenta" / "canary-configs"
LAST_RUN_FILE = REPO_ROOT / "data" / "last_run.json"
THRESHOLDS = {"pass": 75.0, "marginal": 50.0}

KAYENTA_URL = os.environ.get("KAYENTA_URL", "http://localhost:8090")
VM_URL = os.environ.get("VM_URL", "http://localhost:8428")
REFEREE_URL = os.environ.get("REFEREE_URL", "http://localhost:3001")
REPORT_PATH = "/dashboard/reports/standalone_canary_analysis/"
ANALYSIS_WINDOW_MINS = 25

RUNS = [
    ("scenario-default", "Default (NetflixACAJudge-v1.0)", "scenario-statistical.json"),
    ("scenario-summary", "AI summary LLM (qwen-llm)", "scenario-ai-summary.json"),
    ("scenario-raw", "AI raw LLM (qwen-llm)", "scenario-ai-raw.json"),
    ("scenario-plot", "AI plot VLM (moondream-vlm)", "scenario-ai-plot.json"),
    ("scenario-hybrid", "Hybrid: NetflixACAJudge + real summary AI", "scenario-ai-hybrid.json"),
]


def main() -> int:
    client = KayentaClient(KAYENTA_URL)
    print("=" * 78)
    print("Mann-Whitney blind-spot scenario: default judge vs the AI judges")
    print("=" * 78)
    if not client.wait_for_health(timeout_s=240):
        print("[scenario] Kayenta not healthy", file=sys.stderr)
        return 2

    print("[scenario] seeding scenario dataset ...")
    seed_scenario_data.seed(VM_URL, 30, 60)
    import time as _t
    _t.sleep(2)
    seed_scenario_data.verify(VM_URL, 30, 60)

    start, end = default_window(ANALYSIS_WINDOW_MINS)
    scopes: List[Dict[str, Any]] = [build_scope(start, end, step=60)]

    outcomes = {}
    for key, label, cfg_file in RUNS:
        print(f"\n[scenario] running {label} ...")
        cfg = json.loads((CONFIG_DIR / cfg_file).read_text())
        try:
            outcomes[key] = client.run_analysis(label, cfg, scopes, THRESHOLDS)
        except Exception as e:  # noqa: BLE001
            print(f"[scenario] ERROR running {label}: {e}", file=sys.stderr)
            return 3

    # report
    print("\n" + "=" * 78)
    print("RESULTS  (the canary is genuinely BAD: unstable tail latency + memory leak)")
    print("=" * 78)
    print("{:<30} {:<8} {:>6}  {}".format("JUDGE", "VERDICT", "SCORE", "PER-METRIC"))
    print("-" * 78)
    for key, label, _ in RUNS:
        o = outcomes[key]
        pm = ", ".join(f"{m['name'].split('_')[0]}={m['classification']}" for m in o.per_metric)
        sc = f"{o.score:.0f}" if o.score is not None else "-"
        print("{:<30} {:<8} {:>6}  {}".format(label[:30], o.verdict, sc, pm))

    # record and links
    record: Dict[str, Any] = {}
    if LAST_RUN_FILE.exists():
        try:
            record = json.loads(LAST_RUN_FILE.read_text())
        except (OSError, ValueError):
            record = {}
    for key, _, _ in RUNS:
        record[key] = outcomes[key].execution_id
    record["referee_base"] = REFEREE_URL.rstrip("/")
    record["report_path"] = REPORT_PATH
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(json.dumps(record, indent=2))

    print("\n" + "-" * 78)
    print("Compare in Referee (Report Viewer):")
    for key, label, _ in RUNS:
        print(f"  {key:18s}: {REFEREE_URL.rstrip('/')}{REPORT_PATH}{outcomes[key].execution_id}")
    print("  (or: make referee JUDGE=scenario-default | scenario-summary | scenario-raw | scenario-plot)")
    print("-" * 78)

    default = outcomes["scenario-default"]
    ai = [outcomes[k] for k in ("scenario-summary", "scenario-raw", "scenario-plot")]
    default_passed = default.passed is True
    ai_caught = sum(1 for o in ai if o.passed is False)
    print(f"\nDefault judge: {default.verdict} (score {default.score}).  "
          f"AI judges that caught the regression (FAIL): {ai_caught}/3.")
    if default_passed and ai_caught >= 1:
        print("[scenario] DEMONSTRATED: the default judge MISSED a bad canary that the AI judge(s) caught.")
    elif not default_passed:
        print("[scenario] NOTE: the default judge did not pass this run (data/window may need a reseed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
