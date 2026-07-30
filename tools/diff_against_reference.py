#!/usr/bin/env python3
"""Diff rows already produced by a re-run against reference-results/, offline.

`tools/verify_reproduction.py` judges each scenario afresh and compares as it goes,
which is the right tool when you have nothing yet. Once a re-run has happened its
rows are already on disk, and judging them a second time would double the cost to
answer a question the existing rows answer exactly. This does the comparison
directly: no model, no gateway, no spend.

Compares verdict, score and rationale for every (scenario, judge, model) present in
both sides, and reports what is missing from each. Error rows are listed separately
because they carry no verdict and cannot be compared -- they are the point of the
exercise, not an omission.

Usage:
  python tools/diff_against_reference.py
  python tools/diff_against_reference.py --models phi4-llm,qwen-llm --out audit/phase4/diff.json
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

import results_schema as rs  # noqa: E402

REFERENCE = REPO_ROOT / "reference-results" / "n20" / "results_raw.csv"


def load_reference(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in csv.DictReader(open(path, newline="")):
        out[(r["scenario"], r["judge"], r["model"])] = r
    return out


# reference-results/ stores a lossy rendering of each row: run_experiment._row()
# rounds the score to two decimals and truncates the rationale to 200 characters
# with newlines flattened. The long-format frame keeps the full values. Comparing
# the two directly reports differences that are entirely formatting -- 0.9887
# against 0.99, or a 243-character rationale against its 200-character prefix --
# so the observed side is rendered the same way before anything is compared.
REFERENCE_SCORE_DP = 2
REFERENCE_RATIONALE_CHARS = 200


def as_reference_score(value: Any) -> float:
    return round(float(value), REFERENCE_SCORE_DP)


def as_reference_rationale(text: str) -> str:
    return (text or "")[:REFERENCE_RATIONALE_CHARS].replace("\n", " ")


def same_score(a: str, b: Any) -> bool:
    try:
        return abs(float(a) - as_reference_score(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff re-run rows against reference-results/")
    ap.add_argument("--models", default="", help="comma-separated aliases; default = all in the frame")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--include-hybrid", action="store_true",
                    help="also compare hybrid rows; only meaningful for a run that used the "
                         "model's primary representation (see the note in the source)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ref = load_reference(Path(args.reference))
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}

    rows = rs.load(dataset_id="original-180", include_errors=True)
    # Only rows this session produced: the imported published run would trivially
    # match itself and would say nothing about reproducibility.
    rows = [r for r in rows if not r["run_id"].startswith("published__")]
    # Filtering the whole frame on prompt_id would silently drop every hybrid and
    # statistical row, because those judges make no model call and carry an empty
    # prompt_id by design. They are part of what has to reproduce, so the frozen-rubric
    # condition is applied only to the rows it can apply to.
    rows = [r for r in rows if r["judge"] != "ai" or r["prompt_id"] == "v1-frozen-2026-06"]

    # Hybrid rows are excluded unless asked for, and that needs justifying rather
    # than assuming. A hybrid verdict is a pure function of the statistical verdict
    # and one AI verdict, computed by the same shared policy code in both runners --
    # so if those two inputs reproduce, the hybrid does too.
    #
    # It cannot simply be compared here because the runners disagree about *which*
    # AI verdict feeds it. run_experiment always uses the model's primary
    # representation (summary for text, plot for vision); run_frontier_experiment
    # uses plot if present and otherwise modes[0]. A `--modes raw` run therefore
    # emits hybrids derived from raw, which are a different quantity from the
    # published summary-derived ones and differ from them legitimately.
    #
    # Where the derivation did match -- the two full re-runs, which used the primary
    # representation -- the hybrids were compared and reproduced exactly
    # (see tools/build_corrected_results.py: 1,352 of 1,352 rows identical).
    if not args.include_hybrid:
        rows = [r for r in rows if r["judge"] != "hybrid"]
    if wanted:
        rows = [r for r in rows if r["model"] in wanted]
    if not rows:
        print("[diff] no re-run rows in the frame for that selection", file=sys.stderr)
        return 2

    compared = identical = 0
    errors: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []
    missing = 0
    per_config: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"compared": 0, "verdict": 0, "score": 0, "rationale": 0, "error": 0})

    for r in rows:
        judge = rs.judge_label(r)
        key = (r["scenario_id"], judge, r["model"])
        stats = per_config[(r["model"], judge)]
        if r["error_kind"]:
            stats["error"] += 1
            errors.append({"scenario": r["scenario_id"], "judge": judge, "model": r["model"],
                           "error_kind": r["error_kind"]})
            continue
        old = ref.get(key)
        if old is None:
            missing += 1
            continue
        compared += 1
        stats["compared"] += 1
        row_ok = True
        if old["verdict"] != r["verdict"]:
            stats["verdict"] += 1
            row_ok = False
            diffs.append({"scenario": key[0], "judge": judge, "model": key[2], "field": "verdict",
                          "reference": old["verdict"], "observed": r["verdict"]})
        if not same_score(old["score"], r["score"]):
            stats["score"] += 1
            row_ok = False
            diffs.append({"scenario": key[0], "judge": judge, "model": key[2], "field": "score",
                          "reference": old["score"], "observed": r["score"]})
        # Rationale is only recorded for the AI judges; hybrid rows carry none.
        if r["judge"] == "ai" and (old.get("rationale") or "") != as_reference_rationale(r.get("rationale")):
            stats["rationale"] += 1
            row_ok = False
            diffs.append({"scenario": key[0], "judge": judge, "model": key[2], "field": "rationale",
                          "reference": (old.get("rationale") or "")[:160],
                          "observed": as_reference_rationale(r.get("rationale"))[:160]})
        identical += int(row_ok)

    print("=" * 78)
    print("reproduction diff: rows re-executed tonight vs reference-results/n20/results_raw.csv")
    print("=" * 78)
    print(f"rows compared        : {compared}")
    print(f"identical            : {identical}")
    print(f"DIFF COUNT           : {len(diffs)}")
    print(f"error rows (no verdict, excluded from the diff): {len(errors)}")
    print(f"not present in the reference                   : {missing}"
          f"  (hosted models have no rows in results_raw.csv, which covers the local set only)")
    print("-" * 78)
    for (model, judge), s in sorted(per_config.items()):
        total = s["verdict"] + s["score"] + s["rationale"]
        flag = "OK  " if total == 0 else "DIFF"
        print(f"  {flag} {model:20s} {judge:16s} compared={s['compared']:4d} "
              f"verdict={s['verdict']} score={s['score']} rationale={s['rationale']} "
              f"errors={s['error']}")
    for d in diffs[:20]:
        print(f"    - {d['scenario']} {d['model']}/{d['judge']} {d['field']}")
        print(f"        reference: {str(d['reference'])[:150]}")
        print(f"        observed : {str(d['observed'])[:150]}")
    if len(diffs) > 20:
        print(f"    ... and {len(diffs)-20} more")

    if errors:
        by_kind: Dict[str, int] = defaultdict(int)
        for e in errors:
            by_kind[f"{e['model']} {e['error_kind']}"] += 1
        print("\n  error rows by cause:")
        for k, v in sorted(by_kind.items()):
            print(f"    {k}: {v}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "compared": compared, "identical": identical, "diff_count": len(diffs),
            "error_rows": len(errors), "not_in_reference": missing,
            "per_config": {f"{m}/{j}": s for (m, j), s in per_config.items()},
            "diffs": diffs[:200], "errors": errors,
        }, indent=2))
        print(f"\n[diff] report -> {args.out}")

    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
