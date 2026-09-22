#!/usr/bin/env python3
"""Sensitivity: which scored AI verdicts came from a completion that may be truncated.

A row is flagged when finish_reason is 'length' or tokens_out reached the alias's
completion budget (the `max_tokens` key in judge-service/models.yaml, else
--default-budget, 1024 by default). finish_reason alone is not enough: a reasoning
model has returned 'stop' at exactly the cap.

This never changes a stored row and never calls a model. It reads long-format
frames (the results_schema columns), lists every flagged row, and reports each
configuration's accuracy as stored and with the flagged rows excluded, so a reader
can see whether a reported number rests on a possibly truncated completion. Rows
with no tokens_out cannot be checked and are counted separately. The release
default is to report, not to exclude: excluding would change scored verdicts, so it
is done only here, in this script's own output.

Usage:
  python tools/sensitivity_truncation.py --frame <csv or directory> [--frame ...]
      [--budget alias=N ...] [--default-budget 1024] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "judge-service" / "models.yaml"


def registry_budgets(path: Path) -> Dict[str, int]:
    """alias -> max_tokens, read from the flow-style entries of models.yaml."""
    out: Dict[str, int] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9._-]+):\s*\{([^}]*)\}", line)
        if m and not line.lstrip().startswith("#"):
            b = re.search(r"max_tokens:\s*(\d+)", m.group(2))
            if b:
                out[m.group(1)] = int(b.group(1))
    return out


def frames(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        q = Path(p)
        out.extend(sorted(q.glob("*.csv")) if q.is_dir() else [q])
    return out


def as_int(v: Any):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame", action="append", required=True)
    ap.add_argument("--budget", action="append", default=[], help="alias=N override")
    ap.add_argument("--default-budget", type=int, default=1024)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    budgets = registry_budgets(REGISTRY)
    for item in args.budget:
        alias, _, n = item.partition("=")
        budgets[alias] = int(n)

    groups: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(
        lambda: {"scored": 0, "correct": 0, "flagged_scored": 0, "flagged_correct": 0,
                 "unknown_tokens": 0, "flagged_error_rows": 0})
    flagged_rows: List[Dict[str, Any]] = []
    for path in frames(args.frame):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if "tokens_out" not in (reader.fieldnames or []):
                continue
            for r in reader:
                if r.get("judge") != "ai":
                    continue
                # One configuration per run: rows of different runs are never merged.
                key = (r.get("run_id", ""), r.get("dataset_id", ""), r.get("prompt_id", ""), r.get("model", ""),
                       r.get("representation", ""))
                g = groups[key]
                budget = budgets.get(r.get("model", ""), args.default_budget)
                out = as_int(r.get("tokens_out"))
                flag = (str(r.get("finish_reason", "")).lower() == "length"
                        or (out is not None and out >= budget))
                is_error = bool(r.get("error_kind"))
                if is_error:
                    if flag:
                        g["flagged_error_rows"] += 1
                        flagged_rows.append({**{k: r.get(k, "") for k in ("run_id", "dataset_id", "prompt_id",
                                             "model", "representation", "family", "scenario_id", "seed",
                                             "verdict", "correct", "tokens_out", "finish_reason", "error_kind")},
                                             "budget": budget, "file": path.name})
                    continue
                correct = as_int(r.get("correct")) or 0
                g["scored"] += 1
                g["correct"] += correct
                if out is None:
                    g["unknown_tokens"] += 1
                if flag:
                    g["flagged_scored"] += 1
                    g["flagged_correct"] += correct
                    flagged_rows.append({**{k: r.get(k, "") for k in ("run_id", "dataset_id", "prompt_id",
                                         "model", "representation", "family", "scenario_id", "seed",
                                         "verdict", "correct", "tokens_out", "finish_reason", "error_kind")},
                                         "budget": budget, "file": path.name})

    table = []
    for key in sorted(groups):
        g = groups[key]
        kept = g["scored"] - g["flagged_scored"]
        table.append({"run_id": key[0], "dataset_id": key[1], "prompt_id": key[2], "model": key[3],
                      "representation": key[4],
                      **g,
                      "accuracy_stored": round(g["correct"] / g["scored"], 4) if g["scored"] else None,
                      "accuracy_excluding_flagged": (round((g["correct"] - g["flagged_correct"]) / kept, 4)
                                                     if kept else None)})

    moved = [t for t in table if t["flagged_scored"]]
    print(f"[truncation] {sum(t['scored'] for t in table)} scored AI rows in {len(table)} configurations; "
          f"{sum(t['flagged_scored'] for t in table)} scored and "
          f"{sum(t['flagged_error_rows'] for t in table)} error rows flagged")
    for t in moved:
        print(f"  {t['run_id'][:44]:44s} {t['prompt_id']:20s} {t['model']:18s} {t['representation']:8s} "
              f"flagged {t['flagged_scored']}/{t['scored']}  accuracy {t['accuracy_stored']} -> "
              f"{t['accuracy_excluding_flagged']}")
    for r in flagged_rows:
        print(f"    {r['file']}: {r['model']} {r['representation']} {r['scenario_id']} seed={r['seed']} "
              f"tokens_out={r['tokens_out']}/{r['budget']} finish={r['finish_reason']!r} "
              f"verdict={r['verdict'] or '-'} error_kind={r['error_kind'] or '-'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"budgets": budgets, "default_budget": args.default_budget,
                                               "configurations": table, "flagged_rows": flagged_rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
