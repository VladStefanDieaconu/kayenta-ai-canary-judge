#!/usr/bin/env python3
"""One long-format results schema, and the only loader that reads it.

Every experiment appends rows of the same shape to results/agg/experiments/, one
row per judged scenario, and every figure generator selects out of that frame
rather than parsing raw files itself. The reason is narrow and practical: the
study now compares three rubrics across two datasets and several judges, and the
previous arrangement -- one bespoke CSV per experiment, each with its own columns
and its own filename convention -- made every new comparison a new parser. It
also made provenance a matter of remembering which file came from which run.

So the frame carries its own provenance. `prompt_id` and `prompt_hash` say which
rubric produced a row; `dataset_id` says which scenario set it came from;
`run_id` ties it back to one invocation. The judges that make no model call --
statistical, hybrid, ensemble -- live in the same frame with an empty prompt_id,
so every judge is comparable without a join.

Rows are append-only. Nothing here rewrites or deletes an existing file.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", REPO_ROOT / "results" / "agg" / "experiments"))

# The order is the file order. Append new columns at the end so an older file
# stays readable by a newer loader.
COLUMNS: List[str] = [
    "run_id",          # one invocation of one runner
    "ts",              # UTC ISO8601, when the row was written
    "dataset_id",      # original-180 | generalisation-60
    "prompt_id",       # empty for judges that make no model call
    "prompt_hash",     # hash of the rubric text actually sent
    "model",           # registry alias, or '-' for model-independent judges
    "judge",           # statistical | ai | hybrid | ensemble
    "representation",  # summary|raw|plot for ai; policy name for hybrid; '' otherwise
    "family",
    "scenario_id",
    "seed",            # the scenario's own deterministic seed
    "truth_label",     # PASS | FAIL
    "verdict",         # PASS | FAIL
    "score",
    "correct",         # 1 | 0, or '' for an error row
    "latency_s",
    "tokens_in",
    "tokens_out",
    "finish_reason",
    "error_kind",      # '' | empty_completion | parse_failure | api_error | transport | unknown_prompt
    "error",
    "rationale",
]

# An error row is not a verdict. Analysis excludes these by default; counting
# them as FAIL is exactly the fault this schema exists to make impossible.
ERROR_KINDS = ("empty_completion", "parse_failure", "api_error", "transport", "unknown_prompt")


def new_row(**kwargs: Any) -> Dict[str, Any]:
    """A row with every column present, so writers cannot omit one silently."""
    unknown = set(kwargs) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result column(s): {sorted(unknown)}")
    row = {c: "" for c in COLUMNS}
    row.update(kwargs)
    return row


def append(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Append rows, writing the header only when creating the file."""
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})
    return len(rows)


def _coerce(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(row)
    for key in ("score", "latency_s"):
        try:
            out[key] = float(row[key]) if row.get(key) not in (None, "") else None
        except (TypeError, ValueError):
            out[key] = None
    for key in ("tokens_in", "tokens_out", "seed", "correct"):
        try:
            out[key] = int(float(row[key])) if row.get(key) not in (None, "") else None
        except (TypeError, ValueError):
            out[key] = None
    return out


def load(
    directory: Optional[Path] = None,
    *,
    prompt_id: Optional[str | Sequence[str]] = None,
    dataset_id: Optional[str | Sequence[str]] = None,
    model: Optional[str | Sequence[str]] = None,
    judge: Optional[str | Sequence[str]] = None,
    representation: Optional[str | Sequence[str]] = None,
    family: Optional[str | Sequence[str]] = None,
    include_errors: bool = False,
) -> List[Dict[str, Any]]:
    """Every row matching the given selection.

    Error rows are excluded unless asked for: a row with an error_kind carries no
    judgement, and letting one into an accuracy denominator is how a fabricated
    verdict becomes a published number.
    """
    directory = Path(directory) if directory is not None else EXPERIMENTS_DIR
    if not directory.is_dir():
        return []

    def wanted(value: Optional[str | Sequence[str]]) -> Optional[set]:
        if value is None:
            return None
        return {value} if isinstance(value, str) else set(value)

    filters = {
        "prompt_id": wanted(prompt_id), "dataset_id": wanted(dataset_id),
        "model": wanted(model), "judge": wanted(judge),
        "representation": wanted(representation), "family": wanted(family),
    }

    rows: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")):
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                if not include_errors and (raw.get("error_kind") or ""):
                    continue
                if any(sel is not None and raw.get(col, "") not in sel for col, sel in filters.items()):
                    continue
                rows.append(_coerce(raw))
    return rows


def judge_label(row: Dict[str, Any]) -> str:
    """`ai:raw`, `hybrid:gated`, `statistical` -- the name used in tables."""
    judge, rep = row.get("judge", ""), row.get("representation", "")
    return f"{judge}:{rep}" if rep else str(judge)


def confusion(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """TP/FP/TN/FN with FAIL as the positive class, matching run_experiment."""
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        truth, verdict = r.get("truth_label"), r.get("verdict")
        if truth == "FAIL" and verdict == "FAIL":
            c["TP"] += 1
        elif truth == "PASS" and verdict == "FAIL":
            c["FP"] += 1
        elif truth == "PASS" and verdict == "PASS":
            c["TN"] += 1
        elif truth == "FAIL" and verdict == "PASS":
            c["FN"] += 1
    return c


def available() -> Dict[str, List[str]]:
    """What is in the frame: the distinct values of each selectable column."""
    rows = load(include_errors=True)
    keys = ("run_id", "dataset_id", "prompt_id", "model", "judge", "representation")
    return {k: sorted({str(r.get(k, "")) for r in rows}) for k in keys}


if __name__ == "__main__":
    import json

    print(json.dumps(available(), indent=2))
