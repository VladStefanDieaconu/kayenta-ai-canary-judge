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
import hashlib
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
#
# Which gid a window gets inside that range is a salted hash of the window's own
# identifier (`_assign_gids`), not its position in the sorted file. It used to be
# the position -- `OWN_DATA_GID_BASE + index` over `sorted(windows)` -- and that
# made the opaque suffix monotone in the alphabetical order of the practitioner's
# own window names. A practitioner whose window ids sort by outcome (`2026-03-14
# -ok-...` before `2026-03-14-rollback-...`, or anything else that puts the good
# deploys first) then got an opaque name that ranks the windows by their answer,
# which is the label leak this harness exists to measure, shipped as the feature
# that prevents it. Reviewer 2, finding R2-3.
OWN_DATA_GID_BASE = 2000
# Exclusive. Four digits keeps `_g{gid:04d}` the width every other name in the
# harness has; the span is what a file's windows are hashed into, so it is also
# the largest number of windows one file can carry.
OWN_DATA_GID_LIMIT = 10000
OWN_DATA_GID_SPAN = OWN_DATA_GID_LIMIT - OWN_DATA_GID_BASE

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
OPTIONAL_COLUMNS = ("label", "group", "unit", "family", "metric_label", "metric_order")
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
                                          "metric_label", "metric_order", "control", "canary",
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
                        # Stated, like the CSV column, rather than taken from this
                        # metric's place in the `metrics` array: the two forms have
                        # to accept and reject exactly the same data, and neither
                        # may depend on the sequence things happen to appear in.
                        "metric_order": ("" if metric.get("metric_order") is None
                                         else str(metric["metric_order"]).strip()),
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
        _constant(problems, entry, "metric_order", row.get("metric_order", ""), line,
                  None, scope=scope)

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

    problems += _order_problems(windows)

    if not windows and not problems:
        problems.append("no windows found; at least one is required")
    return windows, problems, naive


def _order_problems(windows: "OrderedDict[str, Dict[str, Any]]") -> List[str]:
    """Check every window's `metric_order` declaration, and parse the ones that hold.

    A window either declares the order of all its metrics or declares none of
    them. A partial declaration is refused rather than half-honoured: the
    alternative is to interpolate the undeclared metrics somewhere, and every
    place to put them is a guess about which metric the classifier hands its
    leftover group weight to.

    Declared numbers only have to be distinct integers, not 1..n: an export that
    numbers by tens, or one that carries a subset of a longer list, still states
    an unambiguous order. They are read as integers rather than compared as text
    so that 2 sorts before 10.
    """
    problems: List[str] = []
    for window in windows.values():
        entries = list(window["metrics"].values())
        declared = [e for e in entries if e.get("metric_order")]
        if not declared:
            continue
        scope = f"window {window['name']!r}"
        if len(declared) != len(entries):
            missing = sorted(e["name"] for e in entries if not e.get("metric_order"))
            problems.append(
                f"{window['line']}: {scope} declares metric_order for "
                f"{len(declared)} of its {len(entries)} metric(s) but not for "
                f"{', '.join(repr(m) for m in missing)}; a window declares the order of "
                f"all of its metrics or of none of them (see {DOC})")
            continue
        ranks: Dict[int, str] = {}
        bad = False
        for entry in entries:
            text = entry["metric_order"]
            try:
                rank = int(text)
            except ValueError:
                problems.append(
                    f"{entry.get('metric_order_line', entry['line'])}: {scope} metric "
                    f"{entry['name']!r} gives metric_order={text!r}, which is not an "
                    f"integer")
                bad = True
                continue
            if rank in ranks:
                problems.append(
                    f"{entry.get('metric_order_line', entry['line'])}: {scope} gives "
                    f"metric_order={rank} to both {ranks[rank]!r} and {entry['name']!r}; "
                    f"the declared order must be a strict order, so the numbers must "
                    f"be distinct")
                bad = True
                continue
            ranks[rank] = entry["name"]
            entry["order_rank"] = rank
        if bad:
            for entry in entries:
                entry.pop("order_rank", None)
    return problems


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
    window by what the file DECLARES in `metric_order` -- falling back to the
    user's own metric name when it declares nothing. Either way the key is read
    out of the file's content, never out of the sequence its rows arrive in, so
    two exports of the same windows with the rows shuffled -- a second `ORDER
    BY`, a different scrape order, a concatenation of per-metric queries -- are
    the same run: the same opaque `own_0000`, the same `latency_2_g4831` suffix
    on the same metric, the same metricSetPairList order in the prompt, and the
    same group weights, including which group takes the 0.01 that
    `eval_dataset._group_weights` leaves over.

    Sorting on the name was the right answer to row-order dependence and the
    wrong answer to fidelity: it is an order the loader invents, and for a window
    carrying several metrics it is usually not the one its owner had. The order
    reaches the prompt and moves that leftover 0.01, so a practitioner whose
    export happens to be alphabetical would otherwise be judged under a different
    classifier configuration from one whose export is by kind, with nothing in
    either file recording the difference. `metric_order` lets the file say it,
    and `export()` writes it, so the round trip preserves the order as well as
    the values.

    The `_gNNNN` in that suffix is a salted hash of the window's own identifier
    (`_assign_gids`), not its place in the sorted file, and the salt is a digest
    of the file's content. What that buys is in `_assign_gids`; what it costs is
    that the number means nothing outside the one file it was drawn in, which is
    the point.
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
    hands the model its own answer. Renamed, the model sees `latency_g4831`,
    whose number is drawn from the window's name and the file's content and
    orders the windows by nothing at all.
    """
    windows, problems, _ = _group(rows)
    if problems:
        raise OwnDataError(problems)
    return _build(windows, anonymise=anonymise, stat_config=stat_config)


def _opaque_name(kind: str, index: int, count: int, gid: int) -> str:
    """The name the model sees for the `index`-th metric of `kind` in a window.

    The only metric of its kind is `latency_g4831`. Several of one kind are
    numbered from one and padded to the width of the count -- `latency_1_g4831`,
    `latency_2_g4831`, and `latency_01_g4831` ... `latency_12_g4831` for a window
    carrying twelve of them. The gid is the window's, from `_assign_gids`; 4831
    is an example and means nothing outside the file it was drawn in.

    The padding, and numbering the first one rather than leaving it bare, are the
    property this format rests on rather than a cosmetic choice. `_build` assigns
    these names in the sorted order of *your* metric names, and `load()` re-derives
    them by sorting whatever names the file it is given carries. For a file this
    module exported -- whose names are already these -- the two agree only if
    sorting the names gives back the order they were assigned in. The earlier
    scheme numbered from the second (`latency_g4831`, `latency_2_g4831`,
    `latency_3_g4831`), and those sort `_2`, `_3`, then the unnumbered one, because
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


# The opaque gid: a salted hash of the window's own name, never its position.
#
# `_opaque_name` above decides what a metric of a given kind is called inside a
# window; these three decide the number that distinguishes one window's metrics
# from another's. Three properties are wanted at once and only the first is
# obvious:
#
#   * it must not encode the sort order of the window ids, or the suffix ranks
#     the windows by whatever their names happen to sort by -- which, for names
#     that carry the outcome, is the label (R2-3);
#   * it must be a deterministic function of the file, because the round trip
#     rests on two loads of one file being one run;
#   * and a file this module exported has to reload to itself, names included,
#     which is why `_carried_gid` exists.
_OPAQUE_WINDOW_RE = re.compile(r"^own_\d{4}$")
_OPAQUE_METRIC_RE = re.compile(r"^(?P<kind>[a-z]+)(?:_(?P<index>\d+))?_g(?P<gid>\d{4,})$")


def _content_salt(windows: "OrderedDict[str, Dict[str, Any]]") -> str:
    """The per-file salt the gids are drawn with. -> 64 hex characters.

    Derived from the file's *content*, for two reasons. It keeps two different
    datasets out of each other's namespace: the same window name in two files
    does not get the same opaque suffix, so a suffix seen in one run says nothing
    about a window in another. And, because a run is only reproducible if its
    names are, it is recorded in the manifest (`gid_salt`), so the names a run
    was judged under can be re-derived from the file it was judged from.

    `metric_order` is deliberately NOT one of the fields the digest covers, for
    the same reason the salt exists: declaring the order a window already had
    must not rename its series. Adding the column to a file changes which metric
    is last, and therefore the prompt and the group weights, and it changes
    neither the gids nor the opaque names -- which is what lets arm 1c carry
    arm 1's names string for string while stating the generator's order.

    Canonical rather than byte-wise: the digest is taken over the windows as
    grouped, with the windows sorted by name, a window's metrics sorted by name
    and a metric's samples sorted by timestamp. Row order, line endings, comment
    lines and the CSV/JSON choice therefore do not reach it. That is not a
    nicety -- `to_scenarios` promises that two exports of the same windows with
    the rows shuffled are the same run, and a salt taken over the raw bytes would
    have given the shuffled file a different salt and every window a different
    name.
    """
    h = hashlib.sha256()
    h.update(b"own-data/gid-salt/v1\n")
    for name in sorted(windows):
        window = windows[name]
        h.update(f"W\x00{name}\x00{window.get('label', '')}\x00"
                 f"{window.get('family', '')}\n".encode("utf-8"))
        metrics = window["metrics"]
        for metric in sorted(metrics):
            entry = metrics[metric]
            h.update(f"M\x00{metric}\x00{entry.get('kind', '')}\x00"
                     f"{entry.get('direction', '')}\x00{entry.get('group', '')}\x00"
                     f"{entry.get('unit', '')}\x00"
                     f"{entry.get('metric_label', '')}\n".encode("utf-8"))
            for side in ("control", "experiment"):
                samples = entry["sides"][side]
                for stamp in sorted(samples):
                    value = samples[stamp]
                    h.update(f"S\x00{side}\x00{stamp}\x00"
                             f"{'' if value is None else repr(float(value))}\n".encode("utf-8"))
    return h.hexdigest()


def _gid_digest(salt: str, name: str) -> str:
    """The window's own draw from the salted space. Nothing else feeds it."""
    return hashlib.sha256(f"{salt}\x00{name}".encode("utf-8")).hexdigest()


def _carried_gid(name: str, window: Dict[str, Any]) -> Optional[int]:
    """The gid a window already carries, if this module is what named it.

    A file this module exported writes the opaque names back out, so re-importing
    it must give those same names or the format is not a fixed point: the window
    id in such a file is the `own_NNNN` this module invented rather than the
    practitioner's own identifier, and hashing *that* would name the same series
    something else on every cycle. `tools/test_own_data_roundtrip.py::anonymised`
    is the check, and it exports and re-imports twice for the reason given there.

    Recognition is strict, not a prefix guess: the window id has to be exactly
    the shape this module assigns, every metric name has to parse as an opaque
    name in the own-data gid range, they all have to carry one gid, and rebuilding
    the whole window's names under that gid has to give back exactly the names the
    file carries -- which is `_opaque_name`'s numbering and padding as well as the
    kind. Anything else is somebody's own naming scheme that happens to rhyme, and
    is hashed like any other.

    Anonymising an already-anonymous window a second time buys nothing, so this
    costs no protection: these names were this module's own output, and they
    carry nothing about the window they came from beyond what it had already
    decided to carry.
    """
    if not _OPAQUE_WINDOW_RE.match(name):
        return None
    metrics = window["metrics"]
    if not metrics:
        return None
    gids = set()
    for metric, entry in metrics.items():
        match = _OPAQUE_METRIC_RE.match(metric)
        if not match or match.group("kind") != entry.get("kind"):
            return None
        gid = int(match.group("gid"))
        if not OWN_DATA_GID_BASE <= gid < OWN_DATA_GID_LIMIT:
            return None
        gids.add(gid)
    if len(gids) != 1:
        return None
    gid = gids.pop()
    per_kind: Counter = Counter(entry["kind"] for entry in metrics.values())
    used: Counter = Counter()
    for metric in sorted(metrics):
        kind = metrics[metric]["kind"]
        used[kind] += 1
        if _opaque_name(kind, used[kind], per_kind[kind], gid) != metric:
            return None
    return gid


def _assign_gids(windows: "OrderedDict[str, Dict[str, Any]]",
                 salt: str) -> "OrderedDict[str, int]":
    """One gid per window, drawn from the salted space. -> {window name: gid}.

    Collisions are found and resolved, never ignored: 180 windows in an 8000-slot
    space collide about twice, and two windows sharing a gid would put two
    different series under one name in the same run. The window that keeps a
    contested slot is decided by the digest, and the windows are walked in digest
    order rather than in name order, so the tie-break cannot put the alphabet back
    into the suffix the way an `index`-ordered probe would. The loser takes the
    next free slot above its own draw, wrapping; linear probing rather than a
    second hash because it is guaranteed to find a free slot whenever one exists,
    and the bias it introduces is in the space, not in the labels.

    Windows that already carry one of this module's gids (`_carried_gid`) are
    seated first and keep it, so re-importing an export is the identity.
    """
    names = sorted(windows)
    if len(names) > OWN_DATA_GID_SPAN:
        raise OwnDataError([
            f"{len(names)} windows: this format reserves {OWN_DATA_GID_SPAN} opaque metric "
            f"namespaces (gid {OWN_DATA_GID_BASE}..{OWN_DATA_GID_LIMIT - 1}), and two windows "
            f"under one gid would put two series under one name. Split the file, or widen "
            f"OWN_DATA_GID_LIMIT and re-export -- see {DOC} on renaming."])

    assigned: Dict[str, int] = {}
    taken: Dict[int, str] = {}
    for name in names:
        carried = _carried_gid(name, windows[name])
        if carried is not None and carried not in taken:
            assigned[name] = carried
            taken[carried] = name

    for _digest, name in sorted((_gid_digest(salt, n), n)
                                for n in names if n not in assigned):
        start = int(_digest[:16], 16) % OWN_DATA_GID_SPAN
        gid = OWN_DATA_GID_BASE + start
        for probe in range(OWN_DATA_GID_SPAN):
            gid = OWN_DATA_GID_BASE + (start + probe) % OWN_DATA_GID_SPAN
            if gid not in taken:
                break
        assigned[name] = gid
        taken[gid] = name
    return OrderedDict((name, assigned[name]) for name in names)


def _build(windows: "OrderedDict[str, Dict[str, Any]]", *, anonymise: bool,
           stat_config: Optional[Dict[str, Any]],
           salt: Optional[str] = None) -> List[ed.Scenario]:
    # The caller passes the salt when it also wants to record it (load() puts it
    # in the manifest); computing it here otherwise keeps to_scenarios() a
    # one-argument entry point.
    gids = _assign_gids(windows, salt if salt is not None else _content_salt(windows))
    scenarios: List[ed.Scenario] = []
    for index, name in enumerate(sorted(windows)):
        window = windows[name]
        gid = gids[name]
        sid = f"own_{index:04d}" if anonymise else name

        step, first, last = _window_grid(window)
        length = int(round((last - first) / step)) + 1

        built: List[Tuple[str, ed.MetricSpec, Dict[str, List[Optional[float]]],
                          Dict[str, Any], Optional[int]]] = []
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
                entry.get("order_rank"),
            ))

        # ...but the ORDER the window is then carried in is the file's to state.
        #
        # It is not a presentational choice. The metricSetPairList order reaches
        # the prompt verbatim (judge-service/app/representations.py), and
        # `eval_dataset._group_weights` hands the leftover 0.01 to the group of
        # the LAST metric, so which metric is last is part of the configuration
        # the run was judged under.
        #
        # So the file declares it, in `metric_order`, and a window that declares
        # one is carried in it whatever sequence its rows arrive in -- the
        # declaration is a column, not a row order, which is the one thing a user
        # is free to re-sort. `export()` writes the order the scenario was built
        # with, so an export/re-import round trip preserves it along with the
        # values.
        #
        # A file that declares nothing falls back to the old rule, sorting on
        # the name both ends share: it is the only order both ends can agree on
        # without being told, and it is what keeps a window a fixed point of its
        # own export. What it is not is the order the window's owner had, which
        # is why `metric_order` exists. Ordering by YOUR names would be worse
        # still under renaming: `p99`, `rate_5xx`, `rss_bytes` becomes
        # `latency_g4831`, `errors_g4831`, `memory_g4831`, and a re-import of
        # that export reads them back `errors`, `latency`, `memory`.
        #
        # Either way the key is a function of the window's content and never of
        # the row sequence, so two loads of one file, and of that file shuffled,
        # are one run.
        declared = bool(built) and all(item[4] is not None for item in built)
        # `_order_problems` has already refused a window that declares an order
        # for some of its metrics and not the others, and one whose declared
        # numbers are not distinct, so `declared` is all-or-nothing here and the
        # rank alone is a total order. The name is kept as a second key anyway,
        # so that a caller reaching _build() without validating first gets a
        # deterministic answer rather than Python's list order.
        built.sort(key=lambda item: (item[4], item[0]) if declared else (0, item[0]))
        metrics: List[ed.MetricSpec] = [spec for _, spec, _, _, _ in built]
        series: Dict[str, Dict[str, List[Optional[float]]]] = {
            name: values for name, _, values, _, _ in built}
        provenance: Dict[str, Any] = {name: prov for name, _, _, prov, _ in built}

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
    # The salt the opaque names were drawn with, taken before the truncation and
    # kept so the manifest can record it: the names a run was judged under have
    # to be re-derivable from the file it was judged from.
    salt = _content_salt(windows)
    declared_order = sum(
        1 for window in windows.values()
        if window["metrics"] and all(entry.get("order_rank") is not None
                                     for entry in window["metrics"].values()))
    scenarios = _build(windows, anonymise=anonymise, stat_config=overrides, salt=salt)
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
        "gid_salt": salt,
        "windows_with_declared_metric_order": declared_order,
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
        # The salt the opaque metric names were drawn with. A run's names are a
        # salted hash of the window ids (own_data._assign_gids), so without this
        # the mapping from `latency_g4831` back to your own metric is only in the
        # `scenarios` block below; with it the whole naming is reproducible from
        # the input file alone.
        "gid_salt": decisions.get("gid_salt", ""),
        # How many of the windows stated the order of their own metrics in the
        # `metric_order` column, and how many fell back to sorting on the name.
        # Which metric is last decides which group takes the leftover 0.01 of
        # `classifier.groupWeights`, so a run whose file declared nothing and a
        # run whose file declared its order are not necessarily the same run,
        # and the manifest says which this was.
        "n_windows_with_declared_metric_order":
            decisions.get("windows_with_declared_metric_order", 0),
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
    reads the file back. The ROW order is canonical because a file is the one
    thing a user is free to re-sort, and a window exported here and re-imported
    must not depend on the sequence its rows arrive in.

    The metric order is a different question, and it is carried rather than
    canonicalised. `metric_order` states each metric's place in
    `scenario.metrics`, one-based, so a window whose owner ordered it latency,
    memory, errors comes back in that order however the rows are sorted. That
    order is not decoration: it is the order the pairs reach the prompt in, and
    `eval_dataset._group_weights` hands its leftover 0.01 to the group of the
    last metric. Writing the order into a column rather than into the row
    sequence is what lets the file be both order-faithful and shuffle-proof at
    once.

    The two ends agreeing on the row key is necessary but was not sufficient: the
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
            # The declared order is the scenario's own, taken before the rows
            # are sorted, so sorting the rows cannot change it.
            order = {m.name: i + 1 for i, m in enumerate(scenario.metrics)}
            for metric in sorted(scenario.metrics, key=lambda m: m.name):
                values = series[metric.name]
                for side, key in (("control", "control"), ("canary", "experiment")):
                    for i, value in enumerate(values[key]):
                        writer.writerow([
                            scenario.id, metric.name, side, _iso(base + i * step),
                            "" if value is None else repr(float(value)),
                            metric.kind if metric.kind in KINDS else "other",
                            metric.direction, scenario.truth, metric.group, metric.unit,
                            scenario.family, metric.truth, order[metric.name],
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
                      "required; label, group, unit, family, metric_label and metric_order are",
                      "optional. metric_order states each metric's place within its window, and",
                      "is written here because the order reaches the prompt and decides which",
                      "group takes the leftover 0.01 of the classifier's group weights.",
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
