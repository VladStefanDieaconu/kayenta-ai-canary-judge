#!/usr/bin/env python3
"""Read an organisation's own labelled canary windows into the harness.

The evaluation dataset in this repository is generated, because a judge can only
be scored against telemetry whose correct verdict is already known. Your own
telemetry has correct verdicts too -- you know which deploys were rolled back --
and this module is how those windows reach the same judges, under the same
statistical configuration and the same scoring code, as the study's dataset.

The format and every rule enforced here are specified in `docs/own-data.md`,
which is the page to read first; this file is its implementation and the error
messages point back at its sections.

What it does:
  * reads a long-format CSV (one row per window/metric/side/sample) or the
    equivalent JSON, both normalised to the same internal rows;
  * validates, reporting *every* problem rather than the first, because a
    900-window export with three bad columns should cost one run, not three;
  * renames everything the model could see, so a window called
    `2026-03-14-rollback-FAIL` cannot leak its own answer into the prompt;
  * builds ordinary `eval_dataset.Scenario` objects carrying their series, which
    is why nothing downstream of `build_pairs()` needs to know where the numbers
    came from.

What it will not do, ever: resample, interpolate, align, pad or outlier-trim. A
window that cannot be judged honestly is rejected with a reason. Inventing the
missing half of a series produces a verdict that is not evidence about anything.

Stdlib only, like every host tool here.

Usage:
  python tools/own_data.py --check windows.csv      # validate + show what is seen
  python tools/own_data.py --describe windows.csv   # show what is seen
  python tools/own_data.py --export windows.csv     # a valid example, from the
                                                    # seeded dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import eval_dataset as ed

# Own-data metric names live at gid 2000 and up, disjoint from the seeded
# datasets' 0... and 1000... ranges, so an own-data metric can never be confused
# with a study metric in a metric store, an ai-log filename or a cache key.
OWN_DATA_GID_BASE = 2000

# Ten samples a side is a harness choice, not a Kayenta requirement, and the
# reasoning is in docs/own-data.md: at n=1 the genuine NetflixACAJudge fails
# classification and the harness records score=100/verdict=PASS -- a silent false
# pass -- and below n=21 the rubric's p95 is an interpolation between the two
# largest samples, so three of its named tail signals collapse onto the maximum.
MIN_SAMPLES_PER_SIDE = 10
MAX_MISSING_FRACTION = 0.20
MIN_SIDE_LENGTH_RATIO = 0.5
STEP_TOLERANCE = 0.05          # a gap is on-grid within 5% of a multiple of step

# A fixed window start for exported files, so `make own-data-template` produces
# the same bytes every time and the committed example does not churn.
EXPORT_BASE_MILLIS = int(datetime(2026, 6, 21, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)

REQUIRED_COLUMNS = ("scenario", "metric", "side", "timestamp", "value", "kind", "direction")
OPTIONAL_COLUMNS = ("label", "group", "unit", "family", "metric_label")
KNOWN_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# A closed vocabulary, because the renamed metric name is the only thing about
# your metric the model is told; see docs/own-data.md on leakage.
#
# `memory` is a member in its own right rather than a synonym folded into
# `resource`. It is the study's own kind (eval_dataset.METRIC_KINDS), so without
# it the exporter has to rewrite every memory metric of the seeded dataset as
# `resource` and export/re-import stops being the identity on
# `MetricSpec.kind` -- the property tools/test_own_data_contract.py checks. It is
# also the word a practitioner exporting RSS would reach for first.
KINDS = ("latency", "errors", "traffic", "saturation", "resource", "memory", "other")
DIRECTIONS = ("increase", "decrease", "either")
LABELS = ("PASS", "FAIL")
METRIC_LABELS = ("pass", "high")
SIDES = {"control": "control", "baseline": "control",
         "canary": "experiment", "experiment": "experiment"}
MISSING_VALUES = ("", "nan", "null", "none")
DEFAULT_FAMILY = "own"

# Only these may be overridden by --stat-config. `direction` is per-metric and
# comes from the CSV, and scoreThresholds are not a property of the statistical
# judge at all -- the same cut turns every judge's score into PASS or FAIL, so
# moving it makes a run incomparable with the article's on all of them.
OVERRIDABLE_STAT_KEYS = ("nanStrategy", "outliers", "effectSize", "critical", "mustHaveData")
EFFECT_SIZE_KEYS = ("measure", "allowedIncrease", "allowedDecrease",
                    "criticalIncrease", "criticalDecrease")

DOC = "docs/own-data.md"

_EPOCH_SECONDS = re.compile(r"^\d{10}(\.\d+)?$")
_EPOCH_MILLIS = re.compile(r"^\d{13}$")


class OwnDataError(Exception):
    """A user's file cannot be loaded. Carries every reason, not the first.

    Raised rather than printed so a caller (run_experiment) decides where the
    message goes; `messages` is the list validate() produced.
    """

    def __init__(self, messages: List[str]) -> None:
        self.messages = list(messages)
        super().__init__("\n".join(self.messages))


# Reading: CSV and JSON, normalised to the same rows.
def read_csv(path: str, ignored: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """One normalised row per (window, metric, side, sample).

    Reads with utf-8-sig, because a spreadsheet export carries a BOM and a BOM
    on the first header cell turns `scenario` into a column nobody recognises.
    Blank and `#`-commented lines *before* the header are skipped, which is what
    lets an exported file explain itself in its first lines; after the header
    every line is data, so a value may begin with `#`.

    Unrecognised columns are dropped and their names appended to `ignored`, so a
    raw Prometheus or Datadog export carrying `job`, `instance` and `__name__`
    loads unedited while a misspelled required column still fails below.
    """
    name = Path(path).name
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as e:
        raise OwnDataError([f"{name}: cannot be read ({e.strerror or e})"]) from e

    lines = text.splitlines(keepends=True)
    offset = 0
    while offset < len(lines) and (not lines[offset].strip() or lines[offset].lstrip().startswith("#")):
        offset += 1
    if offset >= len(lines):
        raise OwnDataError([f"{name}: no header row; expected a first line naming the columns "
                            f"{', '.join(REQUIRED_COLUMNS)} (see {DOC})"])

    reader = csv.reader(lines[offset:])
    header = [c.strip().lower() for c in next(reader)]
    _check_header(name, offset + 1, header, ignored)

    index = {c: i for i, c in enumerate(header) if c in KNOWN_COLUMNS}
    rows: List[Dict[str, str]] = []
    for raw in reader:
        line = offset + reader.line_num
        if not any(cell.strip() for cell in raw):
            continue
        row: Dict[str, str] = {}
        if len(raw) != len(header):
            row["_error"] = (f"has {len(raw)} field(s) but the header has {len(header)}; "
                             f"every row must carry one value per column")
        else:
            for col, i in index.items():
                row[col] = raw[i].strip()
        row["_line"] = f"{name}:{line}"
        rows.append(row)
    if not rows:
        raise OwnDataError([f"{name}: header only, no data rows (see {DOC})"])
    return rows


def _check_header(name: str, line: int, header: List[str],
                  ignored: Optional[List[str]]) -> None:
    problems = []
    seen = Counter(header)
    for col, count in seen.items():
        if count > 1 and col in KNOWN_COLUMNS:
            problems.append(f"{name}:{line}: column {col!r} appears {count} times; "
                            f"it must appear once")
    absent = [c for c in REQUIRED_COLUMNS if c not in header]
    if absent:
        unknown = [c for c in header if c not in KNOWN_COLUMNS]
        problems.append(
            f"{name}:{line}: required column(s) {', '.join(repr(c) for c in absent)} are missing. "
            f"The header must name {', '.join(REQUIRED_COLUMNS)}; "
            f"{', '.join(OPTIONAL_COLUMNS)} are optional (see {DOC})"
            # Named here because this is what a typo looks like from the inside:
            # the misspelling is an unrecognised column and the real one is absent.
            + (f". The header also has {', '.join(repr(c) for c in unknown)}, which "
               f"this loader does not recognise" if unknown else ""))
    if problems:
        raise OwnDataError(problems)
    if ignored is not None:
        ignored.extend(c for c in header if c not in KNOWN_COLUMNS)


def read_json(path: str, ignored: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """The same normalised rows, from the array-of-windows JSON form.

    One extra reader over the same validator, so the two forms accept and reject
    exactly the same data.
    """
    name = Path(path).name
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise OwnDataError([f"{name}: cannot be read ({e.strerror or e})"]) from e
    except json.JSONDecodeError as e:
        raise OwnDataError([f"{name}:{e.lineno}: not valid JSON ({e.msg} at column {e.colno})"]) from e

    if not isinstance(doc, list):
        raise OwnDataError([f"{name}: the top level must be an array of window objects, "
                            f"not {type(doc).__name__} (see {DOC})"])
    if not doc:
        raise OwnDataError([f"{name}: the array is empty; at least one window is required"])

    rows: List[Dict[str, str]] = []
    seen_keys: set = set()
    for si, window in enumerate(doc):
        loc = f"{name}: window[{si}]"
        if not isinstance(window, dict):
            rows.append({"_error": f"is {type(window).__name__}, not an object", "_line": loc})
            continue
        scenario = str(window.get("scenario", "")).strip()
        loc = f"{name}: window[{si}] {scenario!r}" if scenario else loc
        seen_keys.update(k for k in window if k not in ("scenario", "label", "family", "metrics"))
        metrics = window.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            rows.append({"_error": "has no 'metrics' array", "_line": loc})
            continue
        for mi, metric in enumerate(metrics):
            mloc = f"{loc} metric[{mi}]"
            if not isinstance(metric, dict):
                rows.append({"_error": f"is {type(metric).__name__}, not an object", "_line": mloc})
                continue
            mname = str(metric.get("metric", "")).strip()
            mloc = f"{loc} metric {mname!r}" if mname else mloc
            seen_keys.update(k for k in metric
                             if k not in ("metric", "kind", "direction", "unit", "group",
                                          "metric_label", "control", "canary",
                                          "baseline", "experiment"))
            for key in ("control", "baseline", "canary", "experiment"):
                side = metric.get(key)
                if side is None:
                    continue
                if not isinstance(side, dict):
                    rows.append({"_error": f"side {key!r} is {type(side).__name__}, not an object",
                                 "_line": mloc})
                    continue
                stamps = side.get("timestamps")
                values = side.get("values")
                if not isinstance(stamps, list) or not isinstance(values, list):
                    rows.append({"_error": f"side {key!r} needs a 'timestamps' array and a "
                                           f"'values' array", "_line": mloc})
                    continue
                if len(stamps) != len(values):
                    rows.append({"_error": f"side {key!r} has {len(stamps)} timestamp(s) and "
                                           f"{len(values)} value(s); they must be the same length",
                                 "_line": mloc})
                    continue
                for i, (ts, val) in enumerate(zip(stamps, values)):
                    rows.append({
                        "scenario": scenario, "metric": mname, "side": key,
                        "timestamp": "" if ts is None else str(ts).strip(),
                        "value": "" if val is None else str(val).strip(),
                        "kind": str(metric.get("kind", "")).strip(),
                        "direction": str(metric.get("direction", "")).strip(),
                        "label": str(window.get("label", "") or "").strip(),
                        "group": str(metric.get("group", "") or "").strip(),
                        "unit": str(metric.get("unit", "") or "").strip(),
                        "family": str(window.get("family", "") or "").strip(),
                        "metric_label": str(metric.get("metric_label", "") or "").strip(),
                        "_line": f"{mloc} {key}[{i}]",
                    })
    if not rows:
        raise OwnDataError([f"{name}: no samples found in any window (see {DOC})"])
    if ignored is not None:
        ignored.extend(sorted(seen_keys))
    return rows


def read(path: str, ignored: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Dispatch on the suffix: `.json` is JSON, anything else is the CSV form."""
    if Path(path).suffix.lower() == ".json":
        return read_json(path, ignored)
    return read_csv(path, ignored)


# Field parsing. Each returns (value, problem) so the caller can keep going.
def _parse_timestamp(text: str) -> Tuple[Optional[int], str, bool]:
    """-> (epoch millis, problem, was_naive).

    RFC 3339, epoch seconds (10 digits, optionally fractional) or epoch
    milliseconds (13 digits). A timestamp with no zone is read as UTC and
    counted, because a uniform offset cannot change the ordering, the derived
    step or the gaps -- which is all a timestamp is used for.
    """
    raw = text.strip()
    if not raw:
        return None, "is empty; every sample needs a timestamp", False
    if _EPOCH_MILLIS.match(raw):
        return int(raw), "", False
    if _EPOCH_SECONDS.match(raw):
        return int(round(float(raw) * 1000)), "", False
    iso = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None, (f"is {raw!r}; expected RFC 3339 (2026-03-14T09:00:00Z), epoch seconds "
                      f"(10 digits) or epoch milliseconds (13 digits)"), False
    naive = dt.tzinfo is None
    if naive:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000), "", naive


def _parse_value(text: str) -> Tuple[Optional[float], str]:
    """-> (value or None for a missing sample, problem).

    Non-finite values are rejected rather than passed through: a NaN or an
    infinity cannot be JSON-encoded and the request to Kayenta fails outright
    ("Out of range float values are not JSON compliant: nan"). A deliberately
    missing sample is spelt empty, NaN, nan, null or None and becomes JSON null.
    """
    raw = text.strip()
    if raw.lower() in MISSING_VALUES:
        return None, ""
    try:
        value = float(raw)
    except ValueError:
        return None, (f"is {raw!r}; expected a number, or one of "
                      f"(empty), NaN, null, None for a missing sample")
    if not math.isfinite(value):
        return None, (f"is {raw!r}; an infinite value cannot be sent to Kayenta. "
                      f"Leave the cell empty to mark the sample missing")
    return value, ""


def _parse_enum(text: str, allowed: Tuple[str, ...], column: str,
                lower: bool = True) -> Tuple[str, str]:
    raw = text.strip()
    candidate = raw.lower() if lower else raw.upper()
    if candidate in allowed:
        return candidate, ""
    return "", (f"column {column!r} is {raw!r}; expected one of "
                f"{'|'.join(allowed)}")


# Grouping: rows -> windows, with every per-row and per-field problem collected.
def _group(rows: List[Dict[str, str]]) -> Tuple["OrderedDict[str, Dict[str, Any]]", List[str], int]:
    """-> (windows, problems, naive_timestamp_count).

    Windows keep their fields as first seen and record the line, so a later
    conflicting value can be reported against both.
    """
    windows: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    problems: List[str] = []
    naive = 0

    for row in rows:
        line = row.get("_line", "?")
        if "_error" in row:
            problems.append(f"{line}: {row['_error']}")
            continue

        scenario = (row.get("scenario") or "").strip()
        metric = (row.get("metric") or "").strip()
        if not scenario:
            problems.append(f"{line}: column 'scenario' is empty; it names the window "
                            f"a verdict is reported at")
            continue
        if not metric:
            problems.append(f"{line}: column 'metric' is empty; it names the series")
            continue

        side, problem = _parse_enum(row.get("side", ""), tuple(SIDES), "side")
        if problem:
            problems.append(f"{line}: {problem} (baseline and experiment are accepted "
                            f"as synonyms for control and canary)")
            continue
        side = SIDES[side]

        millis, problem, was_naive = _parse_timestamp(row.get("timestamp", ""))
        if problem:
            problems.append(f"{line}: column 'timestamp' {problem}")
            continue
        naive += 1 if was_naive else 0

        value, problem = _parse_value(row.get("value", ""))
        if problem:
            problems.append(f"{line}: column 'value' {problem}")
            continue

        window = windows.get(scenario)
        if window is None:
            window = windows[scenario] = {
                "name": scenario, "line": line, "metrics": OrderedDict(),
                "label": "", "family": "",
            }
        _constant(problems, window, "label", row.get("label", ""), line,
                  LABELS, lower=False, scope=f"window {scenario!r}")
        _constant(problems, window, "family", row.get("family", ""), line,
                  None, scope=f"window {scenario!r}")

        entry = window["metrics"].get(metric)
        if entry is None:
            entry = window["metrics"][metric] = {
                "name": metric, "line": line,
                "kind": "", "direction": "", "group": "", "unit": "", "metric_label": "",
                "sides": {"control": {}, "experiment": {}},
                "side_lines": {},
                "sample_lines": {},
            }
        scope = f"window {scenario!r} metric {metric!r}"
        _constant(problems, entry, "kind", row.get("kind", ""), line, KINDS, scope=scope)
        _constant(problems, entry, "direction", row.get("direction", ""), line,
                  DIRECTIONS, scope=scope)
        _constant(problems, entry, "group", row.get("group", ""), line, None, scope=scope)
        _constant(problems, entry, "unit", row.get("unit", ""), line, None, scope=scope)
        _constant(problems, entry, "metric_label", row.get("metric_label", ""), line,
                  METRIC_LABELS, scope=scope)

        samples = entry["sides"][side]
        if millis in samples:
            first = entry["sample_lines"].get((side, millis), "?")
            problems.append(
                f"{line}: {scope} side {_side_name(side)!r} already has a sample at "
                f"{_iso(millis)} (first seen at {first}); duplicate timestamps are an "
                f"error and are never averaged")
            continue
        samples[millis] = value
        entry["sample_lines"][(side, millis)] = line
        entry["side_lines"].setdefault(side, line)

    if not windows and not problems:
        problems.append("no windows found; at least one is required")
    return windows, problems, naive


def _constant(problems: List[str], holder: Dict[str, Any], field: str, raw: str,
              line: str, allowed: Optional[Tuple[str, ...]],
              scope: str, lower: bool = True) -> None:
    """Record a field that must not vary within its scope, or report the clash.

    An empty cell is not a value: a file that names `group` once per window and
    leaves it blank on the other rows is a spreadsheet, not a conflict.
    """
    text = (raw or "").strip()
    if not text:
        return
    if allowed is not None:
        value, problem = _parse_enum(text, allowed, field, lower=lower)
        if problem:
            problems.append(f"{line}: {scope} {problem}")
            return
    else:
        value = text
    seen = holder.get(field) or ""
    if seen and seen != value:
        problems.append(
            f"{line}: {scope} gives {field}={value!r} but {holder.get(field + '_line', '?')} "
            f"gives {field}={seen!r}; {field} must be the same throughout")
        return
    holder[field] = value
    holder.setdefault(field + "_line", line)


# Window geometry: step, grid, gaps. Nothing here repairs anything.
def _derive_step(stamps: List[int]) -> Tuple[Optional[int], List[Tuple[int, int]]]:
    """-> (modal gap in millis, [(a, b) pairs that are off-grid]).

    The step is the most common gap between consecutive timestamps, ties broken
    towards the smaller, so it is the same for the same file every time. A gap
    is on-grid when it is within 5% of an integer multiple of that step, which
    is what lets a missing scrape through and stops an irregular export.
    """
    if len(stamps) < 2:
        return None, []
    diffs = [b - a for a, b in zip(stamps, stamps[1:])]
    counts = Counter(diffs)
    top = max(counts.values())
    step = min(d for d, c in counts.items() if c == top)
    off: List[Tuple[int, int]] = []
    for a, b in zip(stamps, stamps[1:]):
        gap = b - a
        k = round(gap / step)
        if k < 1 or abs(gap - k * step) > STEP_TOLERANCE * step:
            off.append((a, b))
    return step, off


def _shape(entry: Dict[str, Any], side: str, step: int) -> Dict[str, Any]:
    samples = entry["sides"][side]
    stamps = sorted(samples)
    present = sum(1 for ts in stamps if samples[ts] is not None)
    expected = int(round((stamps[-1] - stamps[0]) / step)) + 1 if len(stamps) > 1 else len(stamps)
    return {
        "first": stamps[0] if stamps else None,
        "last": stamps[-1] if stamps else None,
        "rows": len(stamps),
        "present": present,
        "expected": expected,
        "missing": max(0, expected - present),
        "missing_fraction": round(max(0, expected - present) / expected, 4) if expected else 0.0,
    }


def validate(rows: List[Dict[str, str]], *, min_samples: int = MIN_SAMPLES_PER_SIDE,
             allow_gaps: bool = False) -> List[str]:
    """Every problem with these rows, in the order they occur. Empty means good.

    Returning all of them rather than raising on the first is the point: a user's
    export will be wrong the first three times, and finding out one reason per
    run is three runs.
    """
    windows, problems, _ = _group(rows)
    if problems:
        # Shape checks are not run over rows that did not parse. "side 'control'
        # has 3 samples" is a misleading thing to tell someone whose other 28
        # rows were rejected a moment earlier for a bad timestamp.
        return problems
    return _shape_problems(windows, min_samples=min_samples, allow_gaps=allow_gaps)


def _by_line(problems: List[str]) -> List[str]:
    """Problems in file order, so a reader can work down the file fixing them.

    The shape checks run window by window and metric by metric, which is not the
    order anyone reads a CSV in. A locator that carries no line number (the JSON
    form) keeps its relative position at the end.
    """
    def key(problem: str) -> Tuple[int, int]:
        m = re.match(r"^[^\s:]+:(\d+):", problem)
        return (0, int(m.group(1))) if m else (1, 0)
    return sorted(problems, key=key)


def _shape_problems(windows: "OrderedDict[str, Dict[str, Any]]", *, min_samples: int,
                    allow_gaps: bool) -> List[str]:
    """Length, step, gap and grid checks over already-parsed windows."""
    problems: List[str] = []
    for scenario, window in windows.items():
        wloc = window["line"]
        if not window["metrics"]:
            problems.append(f"{wloc}: window {scenario!r} has no metrics")
            continue

        steps: Dict[str, int] = {}
        shapes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for metric, entry in window["metrics"].items():
            loc = entry["line"]
            scope = f"window {scenario!r} metric {metric!r}"
            if not entry["kind"]:
                problems.append(f"{loc}: {scope} has no 'kind'; expected one of "
                                f"{'|'.join(KINDS)}")
            if not entry["direction"]:
                problems.append(
                    f"{loc}: {scope} has no 'direction'; expected one of {'|'.join(DIRECTIONS)}. "
                    f"There is no default on purpose: guessing 'increase' for a success-rate "
                    f"metric inverts its verdict silently (see {DOC})")

            empty = [s for s in ("control", "experiment") if not entry["sides"][s]]
            if empty:
                problems.append(
                    f"{loc}: {scope} has no {' and no '.join(_side_name(e) for e in empty)} samples. Both sides are "
                    f"required: under mustHaveData=true Kayenta returns Nodata and the window "
                    f"would fail for an export bug rather than for a regression")
                continue

            metric_shapes: Dict[str, Dict[str, Any]] = {}
            metric_steps: Dict[str, int] = {}
            for side in ("control", "experiment"):
                sloc = entry["side_lines"].get(side, loc)
                stamps = sorted(entry["sides"][side])
                if len(stamps) < 2:
                    problems.append(
                        f"{sloc}: {scope} side {_side_name(side)!r} has {len(stamps)} sample(s); at least 2 "
                        f"are needed before a step can be derived at all")
                    continue
                step, off = _derive_step(stamps)
                for a, b in off[:3]:
                    problems.append(
                        f"{sloc}: {scope} side {_side_name(side)!r} jumps from {_iso(a)} to {_iso(b)} "
                        f"({(b - a) / 1000:.0f}s), which is not a multiple of the derived "
                        f"{step // 1000}s step within {int(STEP_TOLERANCE * 100)}%. Nothing is "
                        f"resampled for you; export at one resolution")
                if off:
                    if len(off) > 3:
                        problems.append(f"{sloc}: {scope} side {_side_name(side)!r} has "
                                        f"{len(off) - 3} further off-grid gap(s)")
                    continue
                metric_steps[side] = step
                metric_shapes[side] = _shape(entry, side, step)

            if len(metric_shapes) != 2:
                continue
            if metric_steps["control"] != metric_steps["experiment"]:
                problems.append(
                    f"{loc}: {scope} has a {metric_steps['control'] // 1000}s step on control and "
                    f"a {metric_steps['experiment'] // 1000}s step on canary; the two sides of a "
                    f"metric must share one step")
                continue
            steps[metric] = metric_steps["control"]
            shapes[metric] = metric_shapes

            for side, shape in metric_shapes.items():
                sloc = entry["side_lines"].get(side, loc)
                if shape["present"] < min_samples:
                    problems.append(
                        f"{sloc}: {scope} side {_side_name(side)!r} has {shape['present']} sample(s) with a "
                        f"value; at least {min_samples} are required. Below that the genuine "
                        f"judge's classification and the rubric's tail statistics stop carrying "
                        f"information (--min-samples lowers it; see {DOC})")
                if shape["missing"] and not allow_gaps and shape["missing_fraction"] > MAX_MISSING_FRACTION:
                    problems.append(
                        f"{sloc}: {scope} side {_side_name(side)!r} is missing {shape['missing']} of "
                        f"{shape['expected']} expected samples ({shape['missing_fraction']:.0%}); "
                        f"the limit is {MAX_MISSING_FRACTION:.0%}. A heavily holed series yields a "
                        f"trend line describing data that never existed (--allow-gaps overrides "
                        f"it and the manifest records that it was used)")

            lo, hi = sorted(s["present"] for s in metric_shapes.values())
            if hi and lo < MIN_SIDE_LENGTH_RATIO * hi:
                problems.append(
                    f"{loc}: {scope} has {metric_shapes['control']['present']} control sample(s) "
                    f"against {metric_shapes['experiment']['present']} canary; one side shorter "
                    f"than half the other is rejected, because every rubric variant speaks about "
                    f"'the end of the observation window' and with spans this different that "
                    f"phrase means two different amounts of time. Trim and re-export")

        distinct = sorted(set(steps.values()))
        if len(distinct) > 1:
            named = ", ".join(f"{m}={steps[m] // 1000}s" for m in sorted(steps))
            problems.append(
                f"{wloc}: window {scenario!r} mixes steps ({named}); a metricSetPair carries one "
                f"scope per window, so all its metrics must share one step. Split them into "
                f"separate windows or export at one resolution")

    return _by_line(problems)


# Scenario construction.
def to_scenarios(rows: List[Dict[str, str]], *, anonymise: bool = True,
                 stat_config: Optional[Dict[str, Any]] = None) -> List[ed.Scenario]:
    """Validated rows -> the Scenario objects the rest of the harness already takes.

    Windows are ordered by the user's own identifier, and the metrics within a
    window by the user's own metric name, so nothing the judges see depends on
    the order rows happen to appear in the file. Two exports of the same windows
    with the rows shuffled -- a second `ORDER BY`, a different scrape order, a
    concatenation of per-metric queries -- are the same run: the same opaque
    `own_0000`, the same `latency_2_g2000` suffix on the same metric, the same
    metricSetPairList order in the prompt, and the same group weights, including
    which group takes the 0.01 that `eval_dataset._group_weights` leaves over.
    `export()` writes each metric under the name this function gave it and sorts
    the rows by that same name, which is the key read back here, so a file this
    module produced reloads under the same `anonymise` setting to exactly what it
    was written from: the same series under the same names, and therefore the same
    `metricSetPairList` and the same canary config. What the file does not carry
    back is `MetricSpec.base` -- on a re-import of an export it is the exported
    name rather than your original one, since your original name was never
    written -- together with `Scenario.gid`, `.seed` and `.params`, which
    `tools/test_own_data_contract.py` asserts as the format's known limits.

    Renaming is on by default and is a protection, not a formatting preference.
    The only field of a pair that reaches the prompt is `name`
    (judge-service/app/representations.py), so a window called
    `2026-03-14-rollback-FAIL` whose metric is `checkout_latency_during_the_bad_deploy`
    hands the model its own answer. Renamed, the model sees `latency_g2000`.
    """
    windows, problems, _ = _group(rows)
    if problems:
        raise OwnDataError(problems)
    return _build(windows, anonymise=anonymise, stat_config=stat_config)


def _opaque_name(kind: str, index: int, count: int, gid: int) -> str:
    """The name the model sees for the `index`-th metric of `kind` in a window.

    The only metric of its kind is `latency_g2000`. Several of one kind are
    numbered from one and padded to the width of the count -- `latency_1_g2000`,
    `latency_2_g2000`, and `latency_01_g2000` ... `latency_12_g2000` for a window
    carrying twelve of them.

    The padding, and numbering the first one rather than leaving it bare, are the
    property this format rests on rather than a cosmetic choice. `_build` assigns
    these names in the sorted order of *your* metric names, and `load()` re-derives
    them by sorting whatever names the file it is given carries. For a file this
    module exported -- whose names are already these -- the two agree only if
    sorting the names gives back the order they were assigned in. The earlier
    scheme numbered from the second (`latency_g2000`, `latency_2_g2000`,
    `latency_3_g2000`), and those sort `_2`, `_3`, then the unnumbered one, because
    a digit sorts before a letter. A three-latency window exported and re-imported
    came back with every series under its neighbour's name, rotating once per round
    trip and never returning. `tools/test_own_data_roundtrip.py::anonymised` is that
    case.

    A window with one metric of a kind -- which is every window of the seeded
    dataset, and most real ones -- is named exactly as before.
    """
    if count < 2:
        return f"{kind}_g{gid:04d}"
    return f"{kind}_{index:0{len(str(count))}d}_g{gid:04d}"


def _build(windows: "OrderedDict[str, Dict[str, Any]]", *, anonymise: bool,
           stat_config: Optional[Dict[str, Any]]) -> List[ed.Scenario]:
    scenarios: List[ed.Scenario] = []
    for index, name in enumerate(sorted(windows)):
        window = windows[name]
        gid = OWN_DATA_GID_BASE + index
        sid = f"own_{index:04d}" if anonymise else name

        step, first, last = _window_grid(window)
        length = int(round((last - first) / step)) + 1

        built: List[Tuple[str, ed.MetricSpec, Dict[str, List[Optional[float]]],
                          Dict[str, Any]]] = []
        # How many metrics of each kind this window carries, counted before any
        # of them is renamed: _opaque_name() numbers a kind only when there is
        # more than one of it, and pads the number to the width of the count.
        per_kind: Counter = Counter(entry["kind"] for entry in window["metrics"].values())
        used: Counter = Counter()
        # Which metric gets which opaque name is decided by sorting YOUR names,
        # not by row order: see to_scenarios(). Two exports of one window with
        # the rows in a different order name the same series the same way.
        for metric in sorted(window["metrics"]):
            entry = window["metrics"][metric]
            kind = entry["kind"]
            used[kind] += 1
            renamed = (_opaque_name(kind, used[kind], per_kind[kind], gid)
                       if anonymise else metric)
            built.append((
                renamed,
                ed.MetricSpec(
                    name=renamed, base=metric, kind=kind,
                    group=entry["group"] or kind, unit=entry["unit"],
                    direction=entry["direction"],
                    truth=entry["metric_label"] or _metric_truth(window["label"]),
                ),
                {side: _series(entry["sides"][side], first, step, length)
                 for side in ("control", "experiment")},
                {
                    "metric": metric,
                    "unit": entry["unit"],
                    "sides": {side: _shape(entry, side, step)
                              for side in ("control", "experiment")},
                },
            ))

        # ...but the ORDER the window is then carried in is the order of the
        # names it carries, which is the order the file stores and the order the
        # model is shown. Ordering by your names instead would mean a window and
        # its own export were two different windows: `p99`, `rate_5xx`,
        # `rss_bytes` becomes `latency_g2000`, `errors_g2000`, `memory_g2000`,
        # the file stores those three sorted, and a re-import reads them back
        # `errors`, `latency`, `memory`. The metricSetPairList order reaches the
        # prompt, and `eval_dataset._group_weights` hands the leftover 0.01 to
        # whichever group comes first, so that is not a presentational
        # difference. Sorting on the name both ends share makes the window a
        # fixed point of its own export.
        built.sort(key=lambda item: item[0])
        metrics: List[ed.MetricSpec] = [spec for _, spec, _, _ in built]
        series: Dict[str, Dict[str, List[Optional[float]]]] = {
            name: values for name, _, values, _ in built}
        provenance: Dict[str, Any] = {name: prov for name, _, _, prov in built}

        scenarios.append(ed.Scenario(
            id=sid, family=window["family"] or DEFAULT_FAMILY, gid=gid, seed=0,
            truth=window["label"], metrics=metrics,
            params={"own_data": {
                "scenario": name,
                "window_start_iso": _iso(first),
                "window_end_iso": _iso(last),
                "step_seconds": step // 1000,
                "n_samples": length,
                "labelled": bool(window["label"]),
                "metrics": provenance,
            }},
            series=series, step_seconds=step // 1000, base_millis=first,
            canary_overrides=dict(stat_config) if stat_config else None,
        ))
    return scenarios


def _window_grid(window: Dict[str, Any]) -> Tuple[int, int, int]:
    """The one step and the one span every series of this window is laid on.

    A metricSetPair carries a single scope per window, so its arrays have to
    share a grid; validate() has already refused a window whose metrics disagree
    about the step. A canary that started four minutes late therefore keeps those
    four slots, as nulls, rather than being slid left to meet the baseline.
    """
    steps, firsts, lasts = [], [], []
    for entry in window["metrics"].values():
        for side in ("control", "experiment"):
            stamps = sorted(entry["sides"][side])
            step, _ = _derive_step(stamps)
            if step:
                steps.append(step)
            firsts.append(stamps[0])
            lasts.append(stamps[-1])
    # validate() has already refused a window whose sides have fewer than two
    # samples, so `steps` is non-empty on any path that went through it; the
    # fallback is for a caller that built scenarios without validating first.
    return (min(steps) if steps else ed.STEP_SECONDS * 1000), min(firsts), max(lasts)


def _series(samples: Dict[int, Optional[float]], first: int, step: int,
            length: int) -> List[Optional[float]]:
    """The side's values on the window grid, with None where there is no sample.

    None, not NaN: a NaN cannot be JSON-encoded and the request to Kayenta fails
    outright. Both the comparator (nanStrategy=remove) and every AI
    representation (representations.clean()) drop them, so a hole costs one
    sample instead of corrupting a series.
    """
    out: List[Optional[float]] = [None] * length
    for ts, value in samples.items():
        index = int(round((ts - first) / step))
        if 0 <= index < length:
            out[index] = value
    return out


def _side_name(side: str) -> str:
    """Sides, in the vocabulary the user typed. The internal name for the canary
    side is `experiment`, which is Kayenta's; telling someone their `experiment`
    side is short when their file says `canary` costs them a minute at 02:00."""
    return "canary" if side == "experiment" else "control"


def _metric_truth(label: str) -> str:
    return {"PASS": "pass", "FAIL": "high"}.get(label, "")


# The statistical override file.
def read_stat_config(path: str) -> Dict[str, Any]:
    """A per-service tolerance file, deep-merged onto STAT_CANARY_TEMPLATE later.

    A key outside the overridable set is an error rather than a no-op, so a typo
    cannot silently leave the tuned value in place while the user believes they
    changed it.
    """
    name = Path(path).name
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise OwnDataError([f"{name}: cannot be read ({e.strerror or e})"]) from e
    except json.JSONDecodeError as e:
        raise OwnDataError([f"{name}:{e.lineno}: not valid JSON ({e.msg} at column {e.colno})"]) from e
    if not isinstance(doc, dict):
        raise OwnDataError([f"{name}: must be a JSON object of statistical settings, "
                            f"not {type(doc).__name__} (see {DOC})"])

    problems: List[str] = []
    for key, value in doc.items():
        if key == "direction":
            problems.append(f"{name}: 'direction' is not overridable here; it is per-metric and "
                            f"comes from your 'direction' column, which a global override would "
                            f"quietly contradict")
        elif key == "scoreThresholds":
            problems.append(f"{name}: 'scoreThresholds' are not overridable. The same cut turns "
                            f"the AI judge's score and all three hybrid policies into PASS or "
                            f"FAIL, so moving it makes a run incomparable with the article's on "
                            f"every judge, not only the statistical one (see {DOC})")
        elif key not in OVERRIDABLE_STAT_KEYS:
            problems.append(f"{name}: {key!r} is not a statistical setting this harness applies; "
                            f"expected one of {', '.join(OVERRIDABLE_STAT_KEYS)}")
        elif key == "effectSize":
            if not isinstance(value, dict):
                problems.append(f"{name}: 'effectSize' must be an object, not "
                                f"{type(value).__name__}")
            else:
                for sub in value:
                    if sub not in EFFECT_SIZE_KEYS:
                        problems.append(f"{name}: 'effectSize.{sub}' is not a known gate; "
                                        f"expected one of {', '.join(EFFECT_SIZE_KEYS)}")
        elif key == "outliers" and not isinstance(value, dict):
            problems.append(f"{name}: 'outliers' must be an object, not {type(value).__name__}")
    if problems:
        raise OwnDataError(problems)
    return doc


# Loading and the manifest.
def load(path: str, *, anonymise: bool = True, min_samples: int = MIN_SAMPLES_PER_SIDE,
         allow_gaps: bool = False, stat_config: Optional[str] = None,
         limit: Optional[int] = None) -> Tuple[List[ed.Scenario], Dict[str, Any]]:
    """Read, validate and build. -> (scenarios, manifest).

    The one entry point a caller needs. Raises OwnDataError carrying every
    problem if the file cannot be loaded honestly.

    `limit` keeps only the first N windows, for a smoke run. The whole file is
    still validated first -- a smoke run that skipped validation would report
    success for a file the real run then refuses -- and the manifest describes
    the N that ran (`n_windows`, `scenarios`, `families`) as well as how many the
    file held (`n_windows_in_file`) and that it was cut (`limited_to`). The count
    of the file has to be taken here, before the truncation, because after it
    there is nothing left that remembers the file was longer.
    """
    ignored: List[str] = []
    rows = read(path, ignored)
    # Parsed once here, then shared: a 900-window export is a few hundred
    # thousand rows, and grouping it three times to satisfy three public
    # signatures would be three times the wait for the same answer.
    windows, problems, naive = _group(rows)
    problems += _shape_problems(windows, min_samples=min_samples, allow_gaps=allow_gaps) \
        if not problems else []
    if problems:
        raise OwnDataError(problems)

    overrides = read_stat_config(stat_config) if stat_config else None
    scenarios = _build(windows, anonymise=anonymise, stat_config=overrides)
    windows_in_file = len(scenarios)
    if limit is not None:
        scenarios = scenarios[:limit]

    warnings = _warnings(scenarios, min_samples, allow_gaps, naive, ignored)
    for warning in warnings:
        print(f"[own-data] warning: {warning}", file=sys.stderr)

    decisions = {
        "source_format": "json" if Path(path).suffix.lower() == ".json" else "csv",
        "anonymised": anonymise,
        "min_samples": min_samples,
        "allow_gaps": allow_gaps,
        "ignored_columns": sorted(set(ignored)),
        "naive_timestamps": naive,
        "statistical_overrides": overrides or {},
        "warnings": warnings,
        "limit": limit,
        "windows_in_file": windows_in_file,
    }
    return scenarios, build_manifest(scenarios, path, decisions)


def _warnings(scenarios: List[ed.Scenario], min_samples: int, allow_gaps: bool,
              naive: int, ignored: List[str]) -> List[str]:
    out: List[str] = []
    if ignored:
        out.append(f"ignored unrecognised column(s): {', '.join(sorted(set(ignored)))}")
    if naive:
        out.append(f"{naive} timestamp(s) carry no time zone and were read as UTC; a uniform "
                   f"offset cannot change the ordering, the step or the gaps, which is all a "
                   f"timestamp is used for")
    if min_samples < MIN_SAMPLES_PER_SIDE:
        out.append(f"--min-samples {min_samples} is below the {MIN_SAMPLES_PER_SIDE}-sample floor. "
                   f"Below it the genuine judge's classification and the rubric's p90/p95/p99 "
                   f"signals stop carrying information; see {DOC} on window shape")
    if allow_gaps:
        out.append("--allow-gaps was used: windows above the 20% missing-sample limit were "
                   "accepted, and their trend lines describe series that are partly absent")
    decreasing = [f"{s.id}/{m.name}" for s in scenarios for m in s.metrics
                  if m.direction in ("decrease", "either")]
    if decreasing:
        out.append(f"{len(decreasing)} metric(s) are not higher-is-worse "
                   f"({', '.join(decreasing[:5])}{', ...' if len(decreasing) > 5 else ''}). The "
                   f"statistical judge handles this correctly; the frozen rubric tells the model "
                   f"that 'clearly better/lower is low', so the AI and hybrid judges run outside "
                   f"the study's conditions on them and may read a harmful drop as an improvement")
    unlabelled = [s.id for s in scenarios if not s.truth]
    if unlabelled:
        out.append(f"{len(unlabelled)} of {len(scenarios)} window(s) carry no label. They are "
                   f"judged and written, and excluded from every accuracy denominator -- not "
                   f"counted as wrong, not counted at all")
    return out


def build_manifest(scenarios: List[ed.Scenario], source: str,
                   decisions: Dict[str, Any]) -> Dict[str, Any]:
    """The run's own manifest, carrying the keys the runner reads plus the record
    of every choice that shaped the load.

    Deliberately not `eval_dataset.build_manifest`: there is no master seed, no
    n-per-family and no single window geometry here, and a manifest that carried
    those fields with invented values would be a worse record than one that does
    not carry them.
    """
    families: List[str] = []
    for scenario in scenarios:
        if scenario.family not in families:
            families.append(scenario.family)
    overrides = decisions.get("statistical_overrides") or {}
    return {
        "own_data": True,
        "source": str(source),
        "source_format": decisions.get("source_format", "csv"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anonymised": bool(decisions.get("anonymised", True)),
        "families": families,
        "n_windows": len(scenarios),
        # How many windows the file held. Equal to n_windows for a full run, and
        # larger when only the first N of them were judged: the two together are
        # what stop a smoke run's scoreboard reading as the whole file's.
        "n_windows_in_file": decisions.get("windows_in_file", len(scenarios)),
        "n_labelled": sum(1 for s in scenarios if s.truth),
        "n_metrics": sum(len(s.metrics) for s in scenarios),
        "min_samples": decisions.get("min_samples", MIN_SAMPLES_PER_SIDE),
        "allow_gaps": bool(decisions.get("allow_gaps", False)),
        "ignored_columns": decisions.get("ignored_columns", []),
        "naive_timestamps": decisions.get("naive_timestamps", 0),
        # Set when only the first N windows of the file were run, so a smoke
        # run's numbers cannot be mistaken for the whole file's.
        "limited_to": decisions.get("limit"),
        "statistical_template": {"canary": ed.STAT_CANARY_TEMPLATE,
                                 "scoreThresholds": ed.SCORE_THRESHOLDS},
        "statistical_overrides": overrides,
        # A run under an overridden template is not comparable with the published
        # numbers, and says so here as well as in results.md.
        "comparable_with_published": not overrides,
        "warnings": decisions.get("warnings", []),
        "scenarios": [ed.scenario_to_manifest(s) for s in scenarios],
    }


# Export: the seeded dataset written out in this format.
def export(scenarios: List[ed.Scenario], path: str, *,
           base_millis: Optional[int] = None, step_seconds: Optional[int] = None,
           header: Optional[List[str]] = None) -> int:
    """Write scenarios as a long-format CSV in this format. -> rows written.

    A real entry point, not a test fixture. Two things need it: `make
    own-data-template`, which produces a file that is valid by construction
    rather than a hand-written example that drifts; and anyone who wants to see
    what the format looks like for data whose answers they already know.

    Values are written with Python's float repr, which round-trips exactly, so
    re-importing an exported file reproduces the same metricSetPairList --
    the property tools/test_own_data_roundtrip.py checks.

    A window's metrics are written sorted by the `metric` column written below,
    which is `MetricSpec.name` -- the same key `to_scenarios()` sorts on when it
    reads the file back. Writing them in the caller's order instead would make
    the file the only record of that order, and a file is the one thing a user is
    free to re-sort; a window exported here and re-imported would then be a
    different window from the same window exported here, shuffled and re-imported.
    Metric order within a window is not information about the data, so it is
    canonicalised on both sides rather than carried.

    The two ends agreeing on the key is necessary but was not sufficient: the
    loader re-derives an opaque name from the name it reads, so the names this
    writes have to sort into the order the loader assigns them in as well. That
    is `_opaque_name()`'s job, and the round trip under `anonymise=True` is what
    checks it.
    """
    columns = list(KNOWN_COLUMNS)
    written = 0
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        for line in header or []:
            fh.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(columns)
        for scenario in scenarios:
            series = ed.series_for_scenario(scenario)
            base = base_millis if base_millis is not None else (
                scenario.base_millis if scenario.base_millis is not None else EXPORT_BASE_MILLIS)
            step = (step_seconds if step_seconds is not None else scenario.step_seconds) * 1000
            # Sorted by the `metric` column this writes -- MetricSpec.name --
            # because that is the key to_scenarios() sorts on when it reads the
            # file back. Two exports of one run must be the same file, and a
            # file must reload to the run it was written from.
            for metric in sorted(scenario.metrics, key=lambda m: m.name):
                values = series[metric.name]
                for side, key in (("control", "control"), ("canary", "experiment")):
                    for i, value in enumerate(values[key]):
                        writer.writerow([
                            scenario.id, metric.name, side, _iso(base + i * step),
                            "" if value is None else repr(float(value)),
                            metric.kind if metric.kind in KINDS else "other",
                            metric.direction, scenario.truth, metric.group, metric.unit,
                            scenario.family, metric.truth,
                        ])
                        written += 1
    return written


def export_seeded(path: str, n_per_family: int = 1,
                  master_seed: int = ed.DEFAULT_MASTER_SEED) -> int:
    """`make own-data-template`: a valid example file, cut from the seeded dataset.

    Honest by construction -- these are the study's own windows, with the study's
    own labels, written through the same exporter a user would run over data of
    their own. Nothing here is a plausible-looking number typed by hand.
    """
    scenarios = ed.build_scenarios(master_seed, n_per_family)
    return export(scenarios, path, base_millis=EXPORT_BASE_MILLIS,
                  step_seconds=ed.STEP_SECONDS, header=[
                      "An example file in the own-data format, and a valid one: these are",
                      f"{len(scenarios)} windows of this repository's own seeded evaluation dataset"
                      f" (seed",
                      f"{master_seed}, {n_per_family} instance(s) of each of "
                      f"{len(ed.FAMILIES)} families), exported by",
                      "tools/own_data.py --export. The numbers are the study's, not invented for",
                      "the example, and the 'label' column carries the verdict each window is",
                      "known to deserve.",
                      "",
                      "Replace the rows with your own and keep the columns. The first seven are",
                      "required; label, group, unit, family and metric_label are optional.",
                      f"The format is specified in {DOC}.",
                      "",
                      "Lines beginning with '#' before the header are ignored; after the header",
                      "every line is data.",
                  ])


# Reporting.
def describe(scenarios: List[ed.Scenario], manifest: Dict[str, Any]) -> str:
    """Exactly what the harness will judge, before a single model call is paid for."""
    lines: List[str] = []
    lines.append(f"[own-data] {manifest['n_windows']} window(s), {manifest['n_metrics']} metric "
                 f"series, {manifest['n_labelled']} labelled, from {manifest['source']}")
    lines.append(f"[own-data] renaming {'on' if manifest['anonymised'] else 'OFF (--keep-ids)'}; "
                 f"families: {', '.join(manifest['families'])}")
    if manifest.get("limited_to"):
        lines.append(f"[own-data] SMOKE RUN: only the first {manifest['limited_to']} of the "
                     f"{manifest.get('n_windows_in_file', manifest['n_windows'])} window(s) in "
                     f"the file will be judged; the whole file was validated")
    if manifest["statistical_overrides"]:
        lines.append(f"[own-data] statistical template OVERRIDDEN "
                     f"{json.dumps(manifest['statistical_overrides'], sort_keys=True)} -- results "
                     f"are NOT comparable with the published ones")
    lines.append("")
    row = "{:<10} {:<30} {:>3} {:>13} {:>6} {:>6} {:>5}  {}"
    lines.append(row.format("id", "your window", "m", "samples c/e", "step", "span",
                            "label", "metrics the model sees"))
    lines.append("-" * 120)
    for scenario in scenarios:
        own = scenario.params["own_data"]
        control = [p["sides"]["control"]["present"] for p in own["metrics"].values()]
        experiment = [p["sides"]["experiment"]["present"] for p in own["metrics"].values()]
        span = (own["n_samples"] - 1) * own["step_seconds"]
        lines.append(row.format(
            scenario.id, _ellipsis(own["scenario"], 30), len(scenario.metrics),
            f"{_range(control)}/{_range(experiment)}", f"{own['step_seconds']}s",
            _span(span), scenario.truth or "-",
            ", ".join(m.name for m in scenario.metrics)))
        # Named per metric and per side rather than totalled, because "4 missing"
        # over a window says nothing about whether one series is unusable.
        for name, provenance in own["metrics"].items():
            for side, shape in provenance["sides"].items():
                if shape["missing"]:
                    lines.append(f"{'':<10} {name} ({provenance['metric']}) {_side_name(side)}: "
                                 f"{shape['missing']} of {shape['expected']} sample(s) missing "
                                 f"({shape['missing_fraction']:.0%})")
    lines.append("-" * 120)
    return "\n".join(lines)


def _range(values: List[int]) -> str:
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _ellipsis(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def _span(seconds: int) -> str:
    if seconds % 3600 == 0 and seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60}m"
    return f"{seconds // 60}m" if seconds % 60 == 0 else f"{seconds}s"


def _iso(millis: int) -> str:
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate, describe or produce a file of your own canary windows")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", metavar="FILE",
                      help="validate FILE, report every problem, then show what the harness sees")
    mode.add_argument("--describe", metavar="FILE",
                      help="show what the harness sees for an already-valid FILE")
    mode.add_argument("--export", metavar="OUT",
                      help="write a valid example file in this format, from the seeded dataset")
    ap.add_argument("--export-n", type=int, default=1,
                    help="instances per family to export (default 1)")
    ap.add_argument("--keep-ids", action="store_true",
                    help="do not rename windows and metrics (your names reach the model)")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_SIDE)
    ap.add_argument("--allow-gaps", action="store_true")
    ap.add_argument("--stat-config", default="",
                    help="JSON file of statistical settings to merge onto the template")
    args = ap.parse_args()

    if args.export:
        n = export_seeded(args.export, args.export_n)
        print(f"[own-data] wrote {n} rows to {args.export}")
        print(f"[own-data] validate it with: make own-data-check OWN_DATA={args.export}")
        return 0

    path = args.check or args.describe
    try:
        scenarios, manifest = load(
            path, anonymise=not args.keep_ids, min_samples=args.min_samples,
            allow_gaps=args.allow_gaps, stat_config=args.stat_config or None)
    except OwnDataError as e:
        print(f"[own-data] {len(e.messages)} problem(s) in {path}:\n", file=sys.stderr)
        for message in e.messages:
            print(f"  {message}", file=sys.stderr)
        print(f"\n[own-data] nothing was judged. The format is specified in {DOC}.",
              file=sys.stderr)
        return 1
    print(describe(scenarios, manifest))
    if args.check:
        print(f"\n[own-data] {path} is valid. Run it with: make own-data {_make_flags(args, path)}")
    return 0


def _make_flags(args: argparse.Namespace, path: str) -> str:
    """The `make own-data` arguments that reproduce this validation.

    Every switch in force here is echoed, because a file accepted under
    `--min-samples 8` or `--allow-gaps` is rejected by a run without them: a
    "Run it with:" line that dropped them would hand the user a command that
    refuses the very file it had just passed. The Makefile variables are named
    rather than the flags, because that is the documented way to reach them
    (docs/own-data.md, "Make targets").
    """
    flags = [f"OWN_DATA={path}"]
    if args.stat_config:
        flags.append(f"STAT_CONFIG={args.stat_config}")
    if args.min_samples != MIN_SAMPLES_PER_SIDE:
        flags.append(f"MIN_SAMPLES={args.min_samples}")
    if args.allow_gaps:
        flags.append("ALLOW_GAPS=1")
    if args.keep_ids:
        flags.append("KEEP_IDS=1")
    return " ".join(flags)


if __name__ == "__main__":
    sys.exit(main())
