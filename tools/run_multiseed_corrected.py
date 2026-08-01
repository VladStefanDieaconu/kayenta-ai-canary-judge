#!/usr/bin/env python3
"""The five-seed sweep, written through the long-format schema.

The retired multi-seed runner produced reference-results/n20/agg/long_results.csv,
whose `error` column is empty on all 8,550 rows and which carries no rationale at
all. That artefact cannot be audited: a row that fabricated a verdict from a
failed call is indistinguishable from a row a model actually answered. This
runner exists to replace it with one that can be.

Three differences from the runner it replaces, and no others -- the sweep itself
is tools/run_experiment.py's run_scenario(), called unchanged, so the judges,
thresholds, representations and hybrid policies are the same code:

  1. Rows go through results_schema, so every row carries the rubric that
     produced it (prompt_id, prompt_hash), whether the completion was truncated
     (finish_reason), and what it cost (tokens_in, tokens_out).
  2. A failed call is written as an error row -- error_kind set, verdict and
     correct empty -- into the same frame. results_schema.load() excludes those
     from every denominator by default, so the failure is preserved as evidence
     without being able to reach a metric.
  3. The output directory is an argument. Nothing is written to results/agg/.

No result cache. The retired runner keyed a disk cache on
(mode, model, pairs); the 6,372 entries it left behind predate the provenance
fields, so every one of them would produce a row with an empty prompt_hash, and
the key does not include the rubric, so there is no way to establish after the
fact which rubric a cached judgement came from. Reusing it would reproduce in a
new column the same unauditability this run exists to remove. Transient failures
are handled by retry instead (below), which caching was also doing.

Usage:
  python tools/run_multiseed_corrected.py --out results/n5-corrected
  python tools/run_multiseed_corrected.py --n 1 --seeds 20260621 --models qwen-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import repo_env  # noqa: E402,F401  (.env before anything reads os.environ)
import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import results_schema as rs  # noqa: E402

DEFAULT_SEEDS = [20260621, 1, 2, 3, 4]
DEFAULT_N = 5
RETRIES = 3

_orig_judge_ai = judge_clients.judge_ai


def _retrying_judge_ai(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Re-attempt a failed call before recording it as a failure.

    A judgement is only absent if the model could not produce one; a gateway
    blip is not that. Retrying keeps the error rows meaning "this model, on this
    scenario, did not answer" rather than "the network hiccuped once".
    """
    out = _orig_judge_ai(*args, **kwargs)
    for attempt in range(1, RETRIES):
        if out["ok"]:
            return out
        time.sleep(min(2 ** attempt, 15))
        out = _orig_judge_ai(*args, **kwargs)
    return out


judge_clients.judge_ai = _retrying_judge_ai

import run_experiment as exp  # noqa: E402  (resolves the patched judge_ai at call time)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_schema_rows(rows: List[Dict[str, Any]], errors: List[Dict[str, Any]],
                   run_id: str, seed: int, dataset_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(rs.new_row(
            run_id=run_id, ts=_now(), dataset_id=dataset_id,
            prompt_id=r.get("prompt_id", ""), prompt_hash=r.get("prompt_hash", ""),
            model=r["model"], judge=r["kind"], representation=r["rep_or_policy"],
            family=r["family"], scenario_id=r["scenario"], seed=seed,
            truth_label=r["truth"], verdict=r["verdict"], score=r["score"],
            correct=r["correct"], latency_s=r["latency"],
            tokens_in=r.get("tokens_in", ""), tokens_out=r.get("tokens_out", ""),
            finish_reason=r.get("finish_reason", ""),
            error_kind="", error=r.get("error", ""), rationale=r.get("rationale", ""),
        ))
    for e in errors:
        judge, _, rep = e["judge"].partition(":")
        out.append(rs.new_row(
            run_id=run_id, ts=_now(), dataset_id=dataset_id,
            prompt_id=e.get("prompt_id", ""), prompt_hash=e.get("prompt_hash", ""),
            model=e["model"], judge=judge, representation=rep,
            family=e["family"], scenario_id=e["scenario"], seed=seed,
            truth_label=e["truth"],
            # verdict, score and correct stay empty. A failed call has no
            # judgement to record, and an empty correct keeps it out of every
            # accuracy denominator even if a reader loads the frame by hand.
            latency_s=e.get("latency", ""),
            tokens_in=e.get("tokens_in", ""), tokens_out=e.get("tokens_out", ""),
            finish_reason=e.get("finish_reason", ""),
            error_kind=e.get("error_kind", "") or "unspecified", error=e.get("error", ""),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="scenario instances per family")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--out", default="results/n5-corrected")
    ap.add_argument("--models", default=os.environ.get("EVAL_MODELS", ""),
                    help="comma-separated alias allow-list (default: every alias present)")
    ap.add_argument("--dataset-id", default="original-180")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    restrict = [m.strip() for m in args.models.split(",") if m.strip()] or None
    out_dir = (REPO_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    long_path = out_dir / "long_results.csv"
    if long_path.exists():
        print(f"[corrected] {long_path} exists; refusing to append to a previous run.",
              file=sys.stderr)
        return 2

    for name, url in (("Kayenta", "http://localhost:8090/health"),
                      ("judge-service", "http://localhost:5001/health")):
        try:
            if requests.get(url, timeout=10).json().get("status") != "UP":
                raise RuntimeError("not UP")
        except Exception as e:  # noqa: BLE001
            print(f"[corrected] {name} not healthy ({e}); run `make up` first.", file=sys.stderr)
            return 2

    present, skipped = exp.discover_models(restrict)
    if not present:
        print("[corrected] no models present locally.", file=sys.stderr)
        return 1

    run_id = f"n{args.n}-corrected-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    n_text = len([m for m in present if m["modality"] == "text"])
    n_vision = len([m for m in present if m["modality"] == "vision"])
    n_calls = n_text * len(exp.TEXT_MODES) + n_vision * len(exp.VISION_MODES)
    n_scen = len(ed.FAMILIES) * args.n * len(seeds)
    n_cfg = 1 + n_text * (len(exp.TEXT_MODES) + len(exp.HYBRID_POLICIES)) \
              + n_vision * (len(exp.VISION_MODES) + len(exp.HYBRID_POLICIES))

    out_dir.mkdir(parents=True, exist_ok=True)
    scope = {
        "run_id": run_id, "seeds": seeds, "n_per_family": args.n,
        "families": list(ed.FAMILIES), "scenarios_per_seed": len(ed.FAMILIES) * args.n,
        "scenarios_total": n_scen, "models": [m["alias"] for m in present],
        "skipped_models": skipped, "representations": {
            "text": exp.TEXT_MODES, "vision": exp.VISION_MODES,
            "hybrid_policies": exp.HYBRID_POLICIES},
        "model_calls_total": n_scen * n_calls, "configurations": n_cfg,
        "rows_per_configuration": n_scen, "cache": "disabled", "retries": RETRIES,
        "started": _now(),
    }
    (out_dir / "run_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(f"[corrected] run_id={run_id}")
    print(f"[corrected] {n_scen} scenarios x {n_calls} model calls = {n_scen * n_calls} calls")
    print(f"[corrected] {n_cfg} configurations x {n_scen} rows = {n_cfg * n_scen} rows")
    print(f"[corrected] models: {[m['alias'] for m in present]}")

    t_start = time.time()
    written = 0
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        scenarios = ed.build_scenarios(seed, args.n)
        base_millis = ed.aligned_base_millis()
        seed_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenarios, 1):
            t0 = time.time()
            before = len(errors)
            srows = exp.run_scenario(sc, present, base_millis, exp.TEXT_MODES,
                                     exp.VISION_MODES, errors)
            seed_rows.extend(srows)
            new_err = len(errors) - before
            print(f"  seed={seed} [{i:3d}/{len(scenarios)}] {sc.id:24s} truth={sc.truth} "
                  f"rows={len(srows)} errors={new_err} ({time.time()-t0:.0f}s)", flush=True)
        # Checkpoint per seed: the frame is append-only, so a crash costs the
        # seed in flight and nothing earlier.
        written += rs.append(long_path, to_schema_rows(seed_rows, errors, run_id, seed,
                                                       args.dataset_id))
        print(f"[corrected] checkpointed seed {seed}: {written} rows so far, "
              f"{len(errors)} error row(s) this seed", flush=True)

    scope["finished"] = _now()
    scope["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    scope["rows_written"] = written
    (out_dir / "run_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(f"\n[corrected] DONE in {scope['elapsed_min']} min -> {long_path} ({written} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
