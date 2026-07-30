#!/usr/bin/env python3
"""Rebuild the N=20 result set with the two defective configurations re-measured.

`reference-results/n20/results_raw.csv` contains 22 rows that are failed calls
recorded as verdicts: 21 from `moondream-vlm / ai:plot` and 1 from
`deepseek-r1-llm / ai:summary`. Those two configurations were re-run in full
under the fixed harness, which records a failed call as an error row instead.

This script splices: every row from the published run except the two affected
configurations, plus the re-measured rows for those configurations, into
`results/corrected/results_raw.csv`. Downstream tables and figures are then
derived from that file exactly as they were from the original.

Two decisions worth stating, because both change what a reviewer sees:

  * The whole configuration is replaced, not just the 22 rows. Splicing 22 new
    rows into 158 old ones would mix two harnesses within a single accuracy
    figure. Re-running all 180 also tests whether the rows that were genuine
    still reproduce, which is reported.
  * The hybrid policies for those models are replaced too, because they consume
    the AI verdict and were computed from the fabricated one.

`reference-results/` is never written to. The published run stays exactly as it
was produced; this is a second, corrected artefact beside it.

Usage:
  python tools/build_corrected_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402

REFERENCE = REPO_ROOT / "reference-results" / "n20" / "results_raw.csv"
AGG = REPO_ROOT / "results" / "agg"
OUT_DIR = REPO_ROOT / "results" / "corrected"

# alias -> the re-run's long CSV, and which judges that run supersedes.
RERUNS: Dict[str, Dict[str, Any]] = {
    "moondream-vlm": {
        "path": AGG / "frontier_long_moondream-vlm__defect-rerun-moondream-vlm.csv",
        "judges": {"ai:plot", "hybrid:gated", "hybrid:or", "hybrid:and"},
    },
    "deepseek-r1-llm": {
        "path": AGG / "frontier_long_deepseek-r1-llm__defect-rerun-deepseek-r1-llm.csv",
        "judges": {"ai:summary", "hybrid:gated", "hybrid:or", "hybrid:and"},
    },
}

HEADER = ["scenario", "family", "truth", "judge", "kind", "rep_or_policy", "model",
          "verdict", "score", "correct", "latency", "error", "note", "rationale"]


def is_error_row(row: Dict[str, str]) -> bool:
    """The published signature of a failed call recorded as a verdict."""
    return row.get("rationale", "").startswith("AI judge error")


def main() -> int:
    ap = argparse.ArgumentParser(description="Splice the re-measured configurations in")
    ap.add_argument("--out", default=str(OUT_DIR / "results_raw.csv"))
    ap.add_argument("--skip-missing", action="store_true",
                    help="carry a model through uncorrected if its re-run has not finished")
    args = ap.parse_args()

    published = list(csv.DictReader(open(REFERENCE, newline="")))
    print(f"[corrected] published rows: {len(published)}")

    available = {}
    for model, spec in RERUNS.items():
        if spec["path"].is_file():
            available[model] = spec
        elif args.skip_missing:
            print(f"[corrected] SKIPPING {model}: no re-run at {spec['path']} "
                  f"(its published rows are carried through uncorrected)")
        else:
            print(f"[corrected] MISSING re-run for {model}: {spec['path']}", file=sys.stderr)
            return 2
    if not available:
        print("[corrected] nothing to correct", file=sys.stderr)
        return 2

    replaced: List[Tuple[str, str]] = []
    for model, spec in available.items():
        for j in spec["judges"]:
            replaced.append((model, j))
    replaced_set = set(replaced)

    kept = [r for r in published if (r["model"], r["judge"]) not in replaced_set]
    dropped = [r for r in published if (r["model"], r["judge"]) in replaced_set]
    print(f"[corrected] superseded rows dropped: {len(dropped)} "
          f"({len(replaced_set)} model/judge configurations)")

    fresh: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, int]] = {}
    for model, spec in available.items():
        rows = list(csv.DictReader(open(spec["path"], newline="")))
        take = [r for r in rows if (r["model"], r["judge"]) in replaced_set]
        # A failed call is dropped, not carried across. The wide-format row still
        # renders it as FAIL at score 0.0 -- the error column is populated now, so
        # it is at least visible -- but writing it into the corrected results would
        # put the original defect straight back: an accuracy computed over those
        # rows counts a failed call as a correct FAIL. Their absence is the point;
        # the affected cells have no measurement, and the denominators say so.
        errs = [r for r in take if r["error"]]
        judged = [r for r in take if not r["error"]]
        fresh.extend(judged)
        stats[model] = {"re_measured": len(take), "judged": len(judged),
                        "errors_dropped": len(errs)}
        print(f"[corrected] {model}: {len(take)} rows re-measured, "
              f"{len(judged)} judged, {len(errs)} failed calls dropped")
        if errs:
            per_fam: Dict[str, int] = defaultdict(int)
            for r in errs:
                per_fam[f"{r['judge']}/{r['family']}"] += 1
            for k, v in sorted(per_fam.items()):
                print(f"      dropped {v:3d}  {k}")

    merged = kept + fresh
    # Stable order: by family declaration order, then scenario, then judge.
    merged.sort(key=lambda r: (ed.family_order(r["family"]), r["scenario"], r["judge"], r["model"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in HEADER})
    print(f"[corrected] {len(merged)} rows -> {out}")

    # Which cells lost their denominator entirely. A family whose every scenario
    # failed has no measurement, and a table that prints a number there is
    # reporting the failure mode as a result -- which is what happened before.
    have: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in merged:
        have[(r["model"], r["judge"], r["family"])] += 1
    empty = sorted(k for model, spec in available.items()
                   for j in spec["judges"] for fam in ed.FAMILIES
                   for k in [(model, j, fam)] if have.get(k, 0) == 0)
    if empty:
        print("\n[corrected] cells with NO measurement (report as 'no data', never as a rate):")
        for model, j, fam in empty:
            print(f"    {model} / {j} / {fam}")
    thin = sorted((k, v) for k, v in have.items()
                  if v < 20 and k[0] in available and k[1] in available[k[0]]["judges"])
    if thin:
        print("\n[corrected] cells with a reduced denominator:")
        for (model, j, fam), v in thin:
            print(f"    {model} / {j} / {fam}: {v}/20")

    # Did the previously-genuine rows reproduce?
    pub_index = {(r["scenario"], r["judge"], r["model"]): r for r in dropped}
    agree = differ = was_error = missing = 0
    diffs: List[Dict[str, Any]] = []
    for r in fresh:
        key = (r["scenario"], r["judge"], r["model"])
        old = pub_index.get(key)
        if old is None:
            missing += 1
            continue
        if is_error_row(old):
            was_error += 1
            continue
        if old["verdict"] == r["verdict"] and old["score"] == r["score"]:
            agree += 1
        else:
            differ += 1
            if len(diffs) < 15:
                diffs.append({"scenario": r["scenario"], "judge": r["judge"],
                              "was": (old["verdict"], old["score"]),
                              "now": (r["verdict"], r["score"])})

    print("\n[corrected] reproduction of the rows that were genuine:")
    print(f"  identical verdict and score : {agree}")
    print(f"  differing                   : {differ}")
    print(f"  previously an error row     : {was_error}")
    print(f"  not in the published run    : {missing}")
    for d in diffs:
        print(f"    {d['scenario']} {d['judge']}: {d['was']} -> {d['now']}")

    report = {"published_rows": len(published), "dropped": len(dropped),
              "re_measured": len(fresh), "total": len(merged),
              "per_model": stats,
              "reproduction": {"identical": agree, "differing": differ,
                               "previously_error": was_error, "missing": missing,
                               "examples": diffs}}
    (out.parent / "correction_report.json").write_text(json.dumps(report, indent=2))
    print(f"[corrected] report -> {out.parent / 'correction_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
