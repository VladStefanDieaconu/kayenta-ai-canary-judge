#!/usr/bin/env python3
"""Sensitivity: which scored AI verdicts JUDGE_STRICT_SCORE=1 could turn into errors.

With JUDGE_STRICT_SCORE=1, a completion whose `overallScore` is not a finite number
in [0, 100] becomes a `parse_failure` error row instead of a verdict. The published
runs had no such check, so this option can move archived verdicts. It is off by
default, and this script says which archived rows it could move. It never changes a
stored row and never calls a model.

What the frames can and cannot show. Up to 1.0.0 the judge turned the score into a
verdict in one of two ways, and neither left a trace except the stored score:
  - a numeric `overallScore` was clamped to [0, 100]. An out-of-range or infinite
    value is stored as 0 or 100, NaN as 100, and a boolean as 0 or 1;
  - a missing or non-numeric `overallScore` fell back to 100 - 30 x (metrics not
    classified Pass), floored at 0: stored as 100, 70, 40, 10 or 0.
So a scored row whose stored score is outside {0, 1, 10, 40, 70, 100} passed the
check for certain, and strict scoring cannot move it. A row at one of those six
values is undetermined: it may be a genuine score or a clamped or fallback one, and
only the logged completion can tell. For each configuration the script reports how
many rows are undetermined and the range the stored accuracy could take if any of
them became error rows.

The per-call payloads (`data/ai-logs/`) decide it. With `--ai-logs DIR` every logged
completion is re-parsed with the service's own parser and checked with
`llm_judge.score_check`, and each one that fails is listed with its model, mode,
rubric and the metric-name suffixes in its prompt. That mode needs the judge
service's dependencies. The payloads are not in the data deposit.

Frame layouts read: the long format (results_schema columns: `judge` = "ai",
`error_kind`) and the per-scenario `results_raw.csv` format (`kind` = "ai", `error`).

Usage:
  python tools/sensitivity_strict_score.py --frame <csv or directory> [--frame ...]
      [--ai-logs data/ai-logs] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Stored scores that a clamped, boolean or fallback score can produce.
UNDETERMINED_SCORES = (0.0, 1.0, 10.0, 40.0, 70.0, 100.0)
ROW_FIELDS = ("run_id", "dataset_id", "prompt_id", "model", "representation", "family", "scenario_id",
              "seed", "verdict", "score", "correct")


def frames(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        q = Path(p)
        out.extend(sorted(q.glob("*.csv")) if q.is_dir() else [q])
    return out


def scored_ai_rows(path: Path):
    """Yield (row as long-format fields, correct) for every scored AI row of a frame."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not {"score", "correct"} <= cols:
            return
        long_layout = "error_kind" in cols
        if not long_layout and "kind" not in cols:
            return
        for r in reader:
            if long_layout:
                if r.get("judge") != "ai" or r.get("error_kind"):
                    continue
                row = {k: r.get(k, "") for k in ROW_FIELDS}
            else:
                if r.get("kind") != "ai" or r.get("error"):
                    continue
                row = {k: "" for k in ROW_FIELDS}
                row.update(run_id=path.parent.name, model=r.get("model", ""),
                           representation=r.get("rep_or_policy", ""), family=r.get("family", ""),
                           scenario_id=r.get("scenario", ""), verdict=r.get("verdict", ""),
                           score=r.get("score", ""), correct=r.get("correct", ""))
            if row["correct"] in ("", None):
                continue
            yield row, int(float(row["correct"]))


def undetermined(score: str) -> bool:
    try:
        return float(score) in UNDETERMINED_SCORES
    except (TypeError, ValueError):
        return True  # a stored score that is not a number cannot be vouched for


def frame_report(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(
        lambda: {"scored": 0, "correct": 0, "undetermined": 0, "undetermined_correct": 0,
                 "undetermined_scores": Counter()})
    rows_out: List[Dict[str, Any]] = []
    for path in frames(paths):
        for row, correct in scored_ai_rows(path):
            # One configuration per file and run: rows of different runs are never merged.
            key = (path.name, row["run_id"], row["dataset_id"], row["prompt_id"], row["model"],
                   row["representation"])
            g = groups[key]
            g["scored"] += 1
            g["correct"] += correct
            if undetermined(row["score"]):
                g["undetermined"] += 1
                g["undetermined_correct"] += correct
                g["undetermined_scores"][row["score"]] += 1
                rows_out.append({**row, "file": path.name})
    table = []
    for key in sorted(groups):
        g = groups[key]
        n, c, u, uc = g["scored"], g["correct"], g["undetermined"], g["undetermined_correct"]
        ui = u - uc
        table.append({"file": key[0], "run_id": key[1], "dataset_id": key[2], "prompt_id": key[3],
                      "model": key[4], "representation": key[5], "scored": n, "correct": c,
                      "undetermined": u, "undetermined_correct": uc,
                      "undetermined_scores": dict(sorted(g["undetermined_scores"].items())),
                      "accuracy_stored": round(c / n, 4) if n else None,
                      # every undetermined correct row becomes an error / every incorrect one does
                      "accuracy_low_if_moved": round((c - uc) / (n - uc), 4) if n - uc else None,
                      "accuracy_high_if_moved": round(c / (n - ui), 4) if n - ui else None})
    return table, rows_out


def log_report(log_dir: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "judge-service"))
    from app import llm_judge, verdict_parser  # noqa: E402  (the service's own parser and check)

    failing: List[Dict[str, Any]] = []
    counts: Dict[str, Counter] = defaultdict(Counter)
    scanned = 0
    for path in sorted(log_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            counts["unreadable"]["files"] += 1
            continue
        scanned += 1
        key = f"{rec.get('model', '?')}|{rec.get('mode', '?')}|{rec.get('prompt_id') or rec.get('prompt_version', '')}"
        if rec.get("error_kind"):
            counts[key]["already_error"] += 1
            continue
        parsed: Optional[Dict[str, Any]] = verdict_parser.parse_json(rec.get("raw_response") or "")
        if parsed is None:
            counts[key]["unparsed"] += 1
            continue
        check = llm_judge.score_check(parsed)
        counts[key][check] += 1
        if check != "ok":
            prompt_text = json.dumps(rec.get("messages") or [])
            failing.append({"file": path.name, "model": rec.get("model"), "mode": rec.get("mode"),
                            "prompt_id": rec.get("prompt_id") or rec.get("prompt_version"),
                            "score_check": check, "overallScore": repr(parsed.get("overallScore")),
                            "metric_suffixes": sorted(set(re.findall(r"_g\d{4}\b", prompt_text)))})
    return {"payloads_scanned": scanned, "per_model_mode_rubric": {k: dict(v) for k, v in sorted(counts.items())},
            "would_become_errors": failing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", action="append", default=[])
    ap.add_argument("--ai-logs", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if not args.frame and not args.ai_logs:
        ap.error("give --frame, --ai-logs, or both")

    out: Dict[str, Any] = {"undetermined_scores": list(UNDETERMINED_SCORES)}
    if args.frame:
        table, rows = frame_report(args.frame)
        n = sum(t["scored"] for t in table)
        u = sum(t["undetermined"] for t in table)
        print(f"[strict-score] {n} scored AI rows in {len(table)} configurations. {n - u} cannot move "
              f"under JUDGE_STRICT_SCORE=1; {u} are undetermined from the frames (stored score in "
              f"{{0, 1, 10, 40, 70, 100}}) and need the logged completion to decide.")
        for t in table:
            if t["undetermined"]:
                print(f"  {t['file'][:60]:60s} {t['model']:18s} {t['representation']:8s} "
                      f"undetermined {t['undetermined']}/{t['scored']}  accuracy {t['accuracy_stored']} "
                      f"(range if moved {t['accuracy_low_if_moved']}-{t['accuracy_high_if_moved']})")
        out.update(configurations=table, undetermined_rows=rows)
    if args.ai_logs:
        logs = log_report(Path(args.ai_logs))
        print(f"[strict-score] {logs['payloads_scanned']} logged completions; "
              f"{len(logs['would_become_errors'])} fail the score check and would become errors")
        for f_ in logs["would_become_errors"]:
            print(f"    {f_['file']}: {f_['model']} {f_['mode']} {f_['prompt_id']} {f_['score_check']} "
                  f"overallScore={f_['overallScore']} suffixes={','.join(f_['metric_suffixes'][:3])}")
        out["ai_logs"] = logs
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
