#!/usr/bin/env python3
"""List the latest recorded Kayenta canary results in one view.

Reads data/last_run.json (written by run_pipeline / run_scenario / make judge),
fetches each execution's status from Kayenta, and prints a table with the verdict,
score, per-metric classifications, and a Referee deep-link.

Note: this shows the latest execution recorded per judge key (last_run.json keeps
one id per key). Kayenta itself has no "list all executions" REST endpoint; ids
are tracked here as they are run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
LAST_RUN_FILE = REPO_ROOT / "data" / "last_run.json"
KAYENTA_URL = os.environ.get("KAYENTA_URL", "http://localhost:8090")
REFEREE_URL = os.environ.get("REFEREE_URL", "http://localhost:3001")
REPORT_PATH = "/dashboard/reports/standalone_canary_analysis/"

# Friendly ordering / labels.
ORDER = ["default", "dummy", "hybrid",
         "scenario-default", "scenario-summary", "scenario-raw", "scenario-plot", "scenario-hybrid"]


def fetch(execution_id: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{KAYENTA_URL}/standalone_canary_analysis/{execution_id}", timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"_error": str(e)[:60]}


def find_judge_result(obj: Any):
    if isinstance(obj, dict):
        if "judgeResult" in obj:
            return obj["judgeResult"]
        for v in obj.values():
            f = find_judge_result(v)
            if f is not None:
                return f
    elif isinstance(obj, list):
        for i in obj:
            f = find_judge_result(i)
            if f is not None:
                return f
    return None


def main() -> int:
    if not LAST_RUN_FILE.exists():
        print("No results yet. Run `make demo-dummy`, `make judge`, or `make scenario` first.", file=sys.stderr)
        return 1
    rec = json.loads(LAST_RUN_FILE.read_text())
    keys = [k for k in ORDER if k in rec] + [k for k in rec if k not in ORDER and k not in ("referee_base", "report_path")]

    print("=" * 96)
    print("Latest recorded Kayenta canary results")
    print("=" * 96)
    print("{:<18} {:<8} {:>6}  {:<40} {}".format("JUDGE KEY", "VERDICT", "SCORE", "REPORTED JUDGE / MODEL", "PER-METRIC"))
    print("-" * 96)
    links = []
    for k in keys:
        exec_id = rec[k]
        status = fetch(exec_id)
        if "_error" in status:
            print("{:<18} {:<8} {:>6}  {:<40} {}".format(k, "ERR", "-", "", status["_error"]))
            continue
        complete = status.get("complete")
        result = status.get("canaryAnalysisExecutionResult") or {}
        scores = result.get("canaryScores") or []
        score = f"{scores[-1]:.0f}" if scores else "-"
        passed = result.get("didPassThresholds")
        verdict = "PASS" if passed is True else ("FAIL" if passed is False else ("…" if not complete else "?"))
        jr = find_judge_result(status) or {}
        judge_name = str(jr.get("judgeName", "") or "?")
        pm = ", ".join(f"{m.get('name','?').split('_')[0]}={m.get('classification','?')}"
                       for m in (jr.get("results") or []))
        print("{:<18} {:<8} {:>6}  {:<40} {}".format(k, verdict, score, judge_name[:40], pm[:30]))
        links.append((k, f"{REFEREE_URL.rstrip('/')}{REPORT_PATH}{exec_id}"))

    print("-" * 96)
    print("Open in Referee:")
    for k, url in links:
        print(f"  {k:18s}: {url}")
    print("  (or: make referee JUDGE=<key>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
