#!/usr/bin/env python3
"""A failed model call must not become a scored verdict.

Section 11 of the paper states that the released harness emits a failed call as
an explicit error carrying no verdict, excluded from every denominator. That is
a claim about this code, so it is checked here rather than asserted: the AI judge
is replaced by one that always fails in each of the ways that were actually
observed, one scenario is run, and the resulting rows are inspected.

The regression this guards against is the one the paper documents. The tolerant
path returns a well-formed object carrying verdict FAIL at score 0.0, which is
indistinguishable from a genuine failing judgement; because both families the
defect touched are FAIL-truth, twenty-one of the twenty-two fabricated rows
scored as *correct* and flattered the configurations they damaged.

No stack, no model, no network: the statistical judge and the AI judge are both
stubbed.

Usage:
  python tools/test_failed_call_is_not_scored.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import run_experiment as exp  # noqa: E402

# Every error_kind the harness can produce, plus the two seen in the study:
# a gateway 500 on the largest chart, and a completion-budget exhaustion.
ERROR_KINDS = ["api_error", "empty_completion", "parse_failure", "transport",
               "unknown_prompt"]

PASSING_STAT = {
    "verdict": "PASS", "classification": "Pass", "score": 100.0,
    "per_metric": [{"name": "m", "classification": "Pass", "reason": ""}],
    "rationale": "", "judge_name": "stub", "model": "-", "latency": 0.01,
    "ok": True, "error": "", "error_kind": "",
}


def failing_ai(kind: str):
    def _stub(*_args, **_kwargs) -> Dict[str, Any]:
        # The real shape returned by judge_clients._err(): a well-formed object
        # carrying a FAIL verdict. Scoring it is the fault under test.
        return judge_clients._err(f"stubbed {kind}", latency=0.02, error_kind=kind)
    return _stub


def check(kind: str) -> List[str]:
    failures: List[str] = []
    scenarios = ed.build_scenarios(ed.DEFAULT_MASTER_SEED, 1)
    sc = next(s for s in scenarios if s.truth == "FAIL")  # the damaging case
    present = [{"alias": "stub-llm", "modality": "text", "tag": "stub"}]

    orig_ai, orig_stat = judge_clients.judge_ai, judge_clients.run_default_judge
    judge_clients.judge_ai = failing_ai(kind)
    judge_clients.run_default_judge = lambda *a, **k: dict(PASSING_STAT)
    try:
        errors: List[Dict[str, Any]] = []
        rows = exp.run_scenario(sc, present, ed.aligned_base_millis(),
                                ["summary", "raw"], ["plot"], errors)
    finally:
        judge_clients.judge_ai, judge_clients.run_default_judge = orig_ai, orig_stat

    ai_rows = [r for r in rows if r["judge"].startswith("ai:")]
    hybrid_rows = [r for r in rows if r["judge"].startswith("hybrid:")]

    if ai_rows:
        failures.append(
            f"{kind}: {len(ai_rows)} AI row(s) written for a failed call; "
            f"first is verdict={ai_rows[0]['verdict']!r} "
            f"correct={ai_rows[0]['correct']!r} — a failed call was scored")
    if hybrid_rows:
        failures.append(
            f"{kind}: {len(hybrid_rows)} hybrid row(s) derived from a failed call")
    if len(errors) != 2:  # summary and raw both fail for a text model
        failures.append(f"{kind}: expected 2 recorded errors, got {len(errors)}")
    for e in errors:
        if e.get("error_kind") != kind:
            failures.append(f"{kind}: error row records error_kind="
                            f"{e.get('error_kind')!r}")
        if "verdict" in e or "correct" in e:
            failures.append(f"{kind}: error row carries a verdict or a correctness flag")
    return failures


def control() -> List[str]:
    """A succeeding call must still produce rows.

    Without this the test passes for the wrong reason: if run_scenario returned
    nothing at all, every assertion above would hold and the harness would be
    broken in the opposite direction.
    """
    failures: List[str] = []
    sc = next(s for s in ed.build_scenarios(ed.DEFAULT_MASTER_SEED, 1)
              if s.truth == "FAIL")
    present = [{"alias": "stub-llm", "modality": "text", "tag": "stub"}]
    good = {**PASSING_STAT, "verdict": "FAIL", "score": 10.0, "model": "stub-llm",
            "rationale": "variance is materially higher"}
    orig_ai, orig_stat = judge_clients.judge_ai, judge_clients.run_default_judge
    judge_clients.judge_ai = lambda *a, **k: dict(good)
    judge_clients.run_default_judge = lambda *a, **k: dict(PASSING_STAT)
    try:
        errors: List[Dict[str, Any]] = []
        rows = exp.run_scenario(sc, present, ed.aligned_base_millis(),
                                ["summary", "raw"], ["plot"], errors)
    finally:
        judge_clients.judge_ai, judge_clients.run_default_judge = orig_ai, orig_stat
    ai_rows = [r for r in rows if r["judge"].startswith("ai:")]
    hybrid_rows = [r for r in rows if r["judge"].startswith("hybrid:")]
    if len(ai_rows) != 2:
        failures.append(f"control: expected 2 AI rows for a succeeding text model, "
                        f"got {len(ai_rows)}")
    if len(hybrid_rows) != len(exp.HYBRID_POLICIES):
        failures.append(f"control: expected {len(exp.HYBRID_POLICIES)} hybrid rows, "
                        f"got {len(hybrid_rows)}")
    if errors:
        failures.append(f"control: {len(errors)} error(s) recorded for a clean call")
    return failures


def main() -> int:
    all_failures: List[str] = []
    fs = control()
    print(f"  {'control (no error)':18s} {'FAIL' if fs else 'ok'}")
    all_failures.extend(fs)
    for kind in ERROR_KINDS:
        fs = check(kind)
        print(f"  {kind:18s} {'FAIL' if fs else 'ok'}")
        all_failures.extend(fs)

    # results_schema, the frame the newer runners write through, must exclude
    # error rows from a default load. Same guarantee, different writer.
    import results_schema as rs
    row = rs.new_row(run_id="t", dataset_id="original-180", judge="ai",
                     representation="raw", scenario_id="x", truth_label="FAIL",
                     error_kind="api_error", error="stubbed")
    if not row["error_kind"]:
        all_failures.append("results_schema.new_row dropped error_kind")
    if row["verdict"] or row["correct"]:
        all_failures.append("results_schema error row carries a verdict")
    print(f"  {'results_schema':18s} "
          f"{'FAIL' if any('results_schema' in f for f in all_failures) else 'ok'}")

    if all_failures:
        print("\nFAILED:")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print("\nA failed call produces no verdict row, no hybrid row, and an "
          "explicit error record. Section 11's claim holds for this code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
