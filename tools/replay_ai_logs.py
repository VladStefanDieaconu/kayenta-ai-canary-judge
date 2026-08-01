#!/usr/bin/env python3
"""Replay archived ai-log payloads through the current parser and guard.

Two questions, both of which have to be answered before the empty-response guard
can be trusted:

  1. Is the change behaviour-preserving on everything that already worked? Every
     archived non-empty response must still parse to exactly the verdict and
     score recorded next to it at the time. A parser that quietly became stricter
     would change published numbers without changing a single verdict in a fresh
     run, because a fresh run would simply not produce the same text.

  2. Does the guard catch what it was written for? Every archived empty response
     must now be classified as an error rather than converted into a verdict.
     These are the payloads from the contaminated gpt-oss window, where an empty
     completion became FAIL/0.0 with error='' and every clean-run indicator read
     green.

No model is called. This reads only what is already on disk.

Usage:
  python tools/replay_ai_logs.py [--dir data/ai-logs] [--model gpt-oss-120b-llm]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import verdict_parser  # noqa: E402

VERDICT_MAP = {"pass": "Pass", "marginal": "Marginal", "fail": "Fail"}


def classify_payload(raw: str) -> str:
    """The guard's own three-way split, applied to an archived completion."""
    return verdict_parser.classify_completion(raw)


def derived_verdict(parsed: Optional[Dict[str, Any]]) -> tuple:
    """(verdict, score) exactly as _map_to_result derives them from the JSON.

    Only the branch that depends on the model's own overallScore/overallVerdict is
    reproduced here; the per-metric fallback needs the metric pairs, which the
    archive does not store, and a payload that reaches it is by definition one the
    guard should now reject anyway.
    """
    if not parsed or not isinstance(parsed.get("overallScore"), (int, float)):
        return (None, None)
    score = float(max(0.0, min(100.0, float(parsed["overallScore"]))))
    verdict = VERDICT_MAP.get(str(parsed.get("overallVerdict", "")).lower(), "")
    if not verdict:
        verdict = "Pass" if score >= 75 else ("Marginal" if score >= 50 else "Fail")
    return (verdict, score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay archived ai-logs through the current parser")
    ap.add_argument("--dir", default=str(REPO_ROOT / "data" / "ai-logs"))
    ap.add_argument("--model", default="", help="restrict to one alias")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "agg" / "replay_ai_logs.json"))
    args = ap.parse_args()

    paths = sorted(Path(args.dir).glob("*.json"))
    print(f"[replay] {len(paths)} archived payloads in {args.dir}")

    kinds = Counter()
    by_model_kind: Dict[str, Counter] = defaultdict(Counter)
    verdict_mismatch: List[Dict[str, Any]] = []
    score_mismatch: List[Dict[str, Any]] = []
    reparse_mismatch: List[Dict[str, Any]] = []
    empty_examples: List[Dict[str, Any]] = []
    unreadable = 0

    for path in paths:
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable += 1
            continue
        model = str(rec.get("model", "?"))
        if args.model and model != args.model:
            continue

        raw = rec.get("raw_response") or ""
        recorded = rec.get("parsed_verdict")
        kind = classify_payload(raw)
        kinds[kind] += 1
        by_model_kind[model][kind] += 1

        if kind == "empty_completion":
            if len(empty_examples) < 5:
                empty_examples.append({"file": path.name, "model": model,
                                       "mode": rec.get("mode"),
                                       "recorded_parsed_verdict": recorded})
            continue

        reparsed = verdict_parser.parse_json(raw)
        # 1. the object itself must come back identical
        if reparsed != recorded and recorded is not None:
            reparse_mismatch.append({"file": path.name, "model": model})
        # 2. and so must the verdict and score derived from it
        rv, rs = derived_verdict(recorded)
        nv, ns = derived_verdict(reparsed)
        if rv != nv:
            verdict_mismatch.append({"file": path.name, "model": model,
                                     "recorded": rv, "replayed": nv})
        if rs != ns:
            score_mismatch.append({"file": path.name, "model": model,
                                   "recorded": rs, "replayed": ns})

    total = sum(kinds.values())
    print("\n=== payload classification under the current guard ===")
    for kind in ("parsed", "empty_completion", "parse_failure"):
        print(f"  {kind:18s} {kinds.get(kind, 0)}")
    print(f"  {'total':18s} {total}   (unreadable files: {unreadable})")

    print("\n=== 1. behaviour preservation on non-empty payloads ===")
    non_empty = kinds.get("parsed", 0) + kinds.get("parse_failure", 0)
    print(f"  non-empty payloads replayed : {non_empty}")
    print(f"  parsed object differs       : {len(reparse_mismatch)}")
    print(f"  derived VERDICT differs     : {len(verdict_mismatch)}")
    print(f"  derived SCORE differs       : {len(score_mismatch)}")
    for m in (verdict_mismatch + score_mismatch)[:10]:
        print(f"    - {m}")

    print("\n=== 2. empty completions now yield error rows ===")
    print(f"  empty completions found     : {kinds.get('empty_completion', 0)}")
    print("  each is classified 'empty_completion', which the guard returns as an")
    print("  error row with ok=False -- never as a verdict.")
    for e in empty_examples:
        print(f"    - {e['file']} model={e['model']} mode={e['mode']} "
              f"recorded_parsed_verdict={e['recorded_parsed_verdict']}")

    print("\n=== per-model breakdown ===")
    for model in sorted(by_model_kind):
        c = by_model_kind[model]
        empty = c.get("empty_completion", 0)
        flag = "  <-- empty completions" if empty else ""
        print(f"  {model:26s} parsed={c.get('parsed',0):5d} empty={empty:4d} "
              f"parse_failure={c.get('parse_failure',0):4d}{flag}")

    report = {
        "payloads_scanned": total, "unreadable": unreadable,
        "classification": dict(kinds),
        "non_empty_replayed": non_empty,
        "parsed_object_mismatches": len(reparse_mismatch),
        "verdict_mismatches": len(verdict_mismatch),
        "score_mismatches": len(score_mismatch),
        "empty_completions": kinds.get("empty_completion", 0),
        "per_model": {m: dict(c) for m, c in by_model_kind.items()},
        "verdict_mismatch_examples": verdict_mismatch[:50],
        "score_mismatch_examples": score_mismatch[:50],
        "empty_examples": empty_examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[replay] report -> {out}")

    ok = (len(verdict_mismatch) == 0 and len(score_mismatch) == 0)
    print(f"[replay] behaviour preserved on non-empty payloads: {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
