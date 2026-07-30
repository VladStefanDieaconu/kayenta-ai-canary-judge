#!/usr/bin/env python3
"""Bring the pre-schema frontier runs into the long-format frame.

The three hosted frozen-prompt runs on the original 180 predate the schema in
results_schema.py, so they sit in per-model CSVs with their own columns. The
prompt ablation has to compare against them, and the alternative -- teaching the
figure generators to read two formats -- is exactly what the single schema exists
to avoid. So they are imported once, with the provenance they were missing.

The prompt is stamped as `v1-frozen-2026-06` because that is demonstrably the
rubric those runs used: it was the only rubric that existed, its text is
reproduced byte for byte in prompts/v1-frozen-2026-06.prompt (verified in
Checkpoint 16), and the local benchmark re-executed against it with zero
differing rows. Stamping it is a statement of fact about those runs, not a guess.

Token counts, finish_reason and error_kind are left empty: the runs did not
record them. An empty cell says "not recorded"; a zero would say "measured as
zero", and that distinction is the whole point of capturing usage now.

Usage:
  python tools/import_legacy_results.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import results_schema as rs  # noqa: E402

FROZEN_ID = "v1-frozen-2026-06"
AGG = REPO_ROOT / "results" / "agg"

# Each source is one published hosted run over the original 180 under the frozen
# rubric. gpt-oss's ai:raw here is the corrected 4096 re-run: verified byte-identical
# to audit/gptoss-raw-corrected/, and carrying none of the contaminated window's
# fabricated score-0.0 rows.
SOURCES = [
    ("claude-opus-4-8-vlm", AGG / "frontier_long_claude-opus-4-8-vlm.csv"),
    ("gpt-oss-120b-llm", AGG / "frontier_long_gpt-oss-120b-llm.csv"),
    ("qwen3-vl-235b-vlm", AGG / "frontier_long_qwen3-vl-235b-vlm.csv"),
]


def seeds_for(dataset_id: str, n: int, seed: int) -> Dict[str, int]:
    return {s.id: s.seed for s in ed.build_scenarios(seed, n, dataset_id)}


def convert(model: str, path: Path, seeds: Dict[str, int], run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for r in csv.DictReader(open(path, newline="")):
        judge_raw = r["judge"]
        if judge_raw.startswith("ai:"):
            judge, rep, prompt_id = "ai", judge_raw.split(":", 1)[1], FROZEN_ID
        elif judge_raw.startswith("hybrid:"):
            judge, rep, prompt_id = "hybrid", judge_raw.split(":", 1)[1], ""
        elif judge_raw == "statistical":
            judge, rep, prompt_id = "statistical", "", ""
        else:
            continue
        rows.append(rs.new_row(
            run_id=run_id, ts=ts, dataset_id="original-180",
            prompt_id=prompt_id,
            # The hash is left empty rather than filled in from the current file:
            # these runs did not record one, and writing today's hash onto them
            # would assert a check that was never performed.
            prompt_hash="",
            model=r["model"], judge=judge, representation=rep,
            family=r["family"], scenario_id=r["scenario"],
            seed=seeds.get(r["scenario"], ""), truth_label=r["truth"],
            verdict=r["verdict"], score=r["score"], correct=r["correct"],
            latency_s=r["latency"], rationale=r.get("rationale", ""),
        ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Import the pre-schema frontier runs")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = seeds_for("original-180", args.n, args.seed)
    total = 0
    for model, path in SOURCES:
        if not path.is_file():
            print(f"[import] MISSING {path}", file=sys.stderr)
            continue
        run_id = f"published__{model}__{FROZEN_ID}__original-180"
        out = rs.EXPERIMENTS_DIR / f"{run_id}.csv"
        if out.exists():
            print(f"[import] {out.name} already present; skipping (append-only)")
            continue
        rows = convert(model, path, seeds, run_id)
        ai = sum(1 for r in rows if r["judge"] == "ai")
        print(f"[import] {model}: {len(rows)} rows ({ai} ai) <- {path.name}")
        if not args.dry_run:
            rs.append(out, rows)
        total += len(rows)
    print(f"[import] {total} rows{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
