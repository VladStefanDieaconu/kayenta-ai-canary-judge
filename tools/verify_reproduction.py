#!/usr/bin/env python3
"""Re-execute a slice of the benchmark and diff it against reference-results/.

The repository's central claim is that a reader can clone it, run the benchmark
under the default configuration, and obtain the numbers the paper reports. This
script is how that claim is tested, and it is deliberately blunt about it: it
compares verdicts, scores *and* rationale strings, and prints an exact count of
rows that differ.

Rationales are the sensitive detector. A verdict is one of two values and a
score one of a hundred, so both can coincide by chance across a whole run; a
sixty-word free-text rationale reproducing character for character cannot. If
verdicts match and rationales do not, something about the prompt or the decoding
path has moved even though the headline numbers have not.

Usage:
  python tools/verify_reproduction.py --models qwen-llm --modes summary --limit 10
  python tools/verify_reproduction.py --models qwen-llm,phi4-llm,deepseek-r1-llm,moondream-vlm
  python tools/verify_reproduction.py --all-local            # every model in the reference

Exit status is 0 only when the diff count is zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

# .env must be loaded before any argparse default reads os.environ; see the
# module docstring for the incident this prevents.
import repo_env  # noqa: E402,F401
import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import prompt_registry  # noqa: E402

REFERENCE = REPO_ROOT / "reference-results" / "n20" / "results_raw.csv"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]


def load_reference(path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Index the reference rows by (scenario, judge, model)."""
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            index[(row["scenario"], row["judge"], row["model"])] = row
    return index


def ai_configurations(index: Dict[Tuple[str, str, str], Dict[str, str]]) -> List[Tuple[str, str]]:
    """Every (model, mode) pair the reference actually contains, in a stable order."""
    seen: List[Tuple[str, str]] = []
    for (_, judge, model) in index:
        if not judge.startswith("ai:"):
            continue
        pair = (model, judge.split(":", 1)[1])
        if pair not in seen:
            seen.append(pair)
    return sorted(seen)


def _same_score(a: str, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff a re-run against reference-results/")
    ap.add_argument("--models", default="", help="comma-separated aliases; default = --all-local")
    ap.add_argument("--modes", default="", help="comma-separated representations; default = whatever the reference has")
    ap.add_argument("--all-local", action="store_true", help="every AI configuration present in the reference")
    ap.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", ed.DEFAULT_N_PER_FAMILY)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", ed.DEFAULT_MASTER_SEED)))
    ap.add_argument("--limit", type=int, default=0, help="first K scenarios only (smoke check)")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--out", default="", help="write a JSON report here")
    ap.add_argument("--prompt-id", default="", help="override the rubric (default: the service's own default)")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.is_file():
        print(f"[verify] no reference at {ref_path}", file=sys.stderr)
        return 2
    index = load_reference(ref_path)

    configs = ai_configurations(index)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        configs = [c for c in configs if c[0] in wanted]
    if args.modes:
        wanted_modes = {m.strip() for m in args.modes.split(",") if m.strip()}
        configs = [c for c in configs if c[1] in wanted_modes]
    if not configs:
        print("[verify] no configurations selected", file=sys.stderr)
        return 2

    prompt = prompt_registry.load(args.prompt_id or os.environ.get("JUDGE_PROMPT_ID")
                                  or prompt_registry.DEFAULT_PROMPT_ID)

    scenarios = ed.build_scenarios(args.seed, args.n)
    if args.limit:
        scenarios = scenarios[: args.limit]
    base_millis = ed.aligned_base_millis()

    print("=" * 78)
    print(f"reproduction check  prompt={prompt.id} ({prompt.hash})  n={args.n} seed={args.seed}")
    print(f"  {len(scenarios)} scenarios x {len(configs)} configurations = {len(scenarios)*len(configs)} calls")
    print(f"  configurations: {', '.join(f'{m}/{md}' for m, md in configs)}")
    print("=" * 78)

    t0 = time.time()
    compared = 0
    missing_reference = 0
    errors = 0
    diffs: List[Dict[str, Any]] = []
    per_config: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"compared": 0, "verdict": 0, "score": 0, "rationale": 0, "error": 0}
    )

    for i, sc in enumerate(scenarios, 1):
        series = ed.series_for_scenario(sc)
        pairs = ed.build_pairs(sc, series, base_millis)
        cfg = ed.build_canary_config(sc, {})
        for model, mode in configs:
            ref = index.get((sc.id, f"ai:{mode}", model))
            if ref is None:
                missing_reference += 1
                continue
            out = judge_clients.judge_ai(pairs, cfg, mode, model, PASS_T, MARGINAL_T,
                                         timeout=300, prompt_id=prompt.id)
            stats = per_config[(model, mode)]
            stats["compared"] += 1
            compared += 1
            if not out["ok"]:
                errors += 1
                stats["error"] += 1
                diffs.append({"scenario": sc.id, "model": model, "mode": mode,
                              "field": "error", "reference": ref["verdict"],
                              "observed": f"{out.get('error_kind')}: {out['error']}"})
                continue
            row_diffs = []
            if out["verdict"] != ref["verdict"]:
                stats["verdict"] += 1
                row_diffs.append(("verdict", ref["verdict"], out["verdict"]))
            if not _same_score(ref["score"], out["score"]):
                stats["score"] += 1
                row_diffs.append(("score", ref["score"], out["score"]))
            if (ref.get("rationale") or "") != (out.get("rationale") or ""):
                stats["rationale"] += 1
                row_diffs.append(("rationale", ref.get("rationale", ""), out.get("rationale", "")))
            for field, r, o in row_diffs:
                diffs.append({"scenario": sc.id, "model": model, "mode": mode,
                              "field": field, "reference": r, "observed": o})
        if i % 10 == 0 or i == len(scenarios):
            print(f"  [{i:3d}/{len(scenarios)}] compared={compared} diffs={len(diffs)} "
                  f"({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print("-" * 78)
    print(f"rows compared      : {compared}")
    print(f"reference missing  : {missing_reference}")
    print(f"judge errors       : {errors}")
    print(f"DIFF COUNT         : {len(diffs)}")
    print("-" * 78)
    for (model, mode), s in sorted(per_config.items()):
        total = s["verdict"] + s["score"] + s["rationale"] + s["error"]
        flag = "OK  " if total == 0 else "DIFF"
        print(f"  {flag} {model}/{mode}: compared={s['compared']} verdict={s['verdict']} "
              f"score={s['score']} rationale={s['rationale']} error={s['error']}")
    for d in diffs[:20]:
        print(f"    - {d['scenario']} {d['model']}/{d['mode']} {d['field']}:")
        print(f"        reference: {str(d['reference'])[:160]}")
        print(f"        observed : {str(d['observed'])[:160]}")
    if len(diffs) > 20:
        print(f"    ... and {len(diffs)-20} more")
    print(f"\n[verify] {elapsed/60:.1f} min")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "prompt_id": prompt.id, "prompt_hash": prompt.hash,
            "n": args.n, "seed": args.seed, "scenarios": len(scenarios),
            "configurations": [f"{m}/{md}" for m, md in configs],
            "compared": compared, "missing_reference": missing_reference,
            "errors": errors, "diff_count": len(diffs),
            "per_config": {f"{m}/{md}": s for (m, md), s in per_config.items()},
            "diffs": diffs[:200], "elapsed_s": round(elapsed, 1),
        }, indent=2))
        print(f"[verify] report -> {args.out}")

    return 0 if len(diffs) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
