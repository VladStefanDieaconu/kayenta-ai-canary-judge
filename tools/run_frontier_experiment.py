#!/usr/bin/env python3
"""Score one frontier hosted model on the same labelled dataset as the N=20 run.

Kept entirely separate from tools/run_experiment.py so the statistical/AI/hybrid
numbers for the local Ollama models are untouched; this script only adds new
output under results/agg/.

For every scenario in the identical 9-family x n-per-family dataset
(tools/eval_dataset.py, same master seed) it runs:
  - the genuine NetflixACAJudge (statistical; recomputed fresh, cheap and fast
    with no LLM call, used only to derive the hybrid policies, exactly as
    run_experiment.py does)
  - the configurable AI judge, for each requested representation (default:
    summary, plot), against the frontier model alias
  - hybrid:gated / hybrid:or / hybrid:and, via the same pure
    judge-service/app/hybrid_policy.decide_policy() run_experiment.py uses,
    combining the statistical result with this model's primary representation
    (plot, since the alias is registered vision-modality). This is free (no
    extra API call), so it comes along at no extra cost or time.

Retry: each judge_ai() call is retried (exponential backoff) on failure, on top
of judge-service's own one-shot JSON retry and LiteLLM's num_retries=2 scoped to
this model's litellm_params (see litellm/config.yaml).

Caching: a simple on-disk JSON cache keyed by (model, mode, scenario id,
seed) under data/ai-logs/_frontier_cache/, so re-running after an
interruption does not re-spend hosted-model calls for scenarios already
judged. Pass --no-cache to force fresh calls.

Usage:
  python tools/run_frontier_experiment.py --model claude-opus-4-8-vlm
      [--modes summary,plot] [--n 20] [--seed 20260621] [--limit-families f1,f2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

# .env must be loaded before any argparse default reads os.environ; see the
# module docstring for the incident this prevents.
import repo_env  # noqa: E402,F401
import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import hybrid_policy  # noqa: E402
import prompt_registry  # noqa: E402
import results_schema  # noqa: E402
# Reuse the published experiment's pure scoring/aggregation code rather than redefining it.
import run_experiment as exp  # noqa: E402

AGG_DIR = REPO_ROOT / "results" / "agg"
CACHE_DIR = REPO_ROOT / "data" / "ai-logs" / "_frontier_cache"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]
HYBRID_POLICIES = ["gated", "or", "and"]


def _cache_path(model: str, mode: str, scenario_id: str, sc_seed: int, prompt_id: str) -> Path:
    # Keyed on the scenario's own deterministic seed (which fully determines its
    # generated series via ed.series_for_scenario), not the master seed or n.
    # A scenario_id like "variance_increase_000" maps to a different sc_seed (and
    # therefore different underlying data) depending on n_per_family, because the
    # gid counter that feeds the per-scenario seed runs across all families. Keying
    # on sc_seed instead of (scenario_id, master_seed) means a stale/mismatched
    # cache entry can't be mistaken for the right one.
    #
    # The prompt id is part of the key for the same reason: two runs of the same
    # scenario under different rubrics are different measurements, and serving one
    # from the other's cache entry would silently merge the ablation's arms.
    safe = f"{model}__{mode}__{prompt_id}__{scenario_id}__scseed{sc_seed}.json"
    return CACHE_DIR / safe


def judge_ai_cached(pairs, cfg, mode: str, model: str, scenario_id: str, sc_seed: int,
                    use_cache: bool, prompt_id: str, retries: int = 4) -> Dict[str, Any]:
    path = _cache_path(model, mode, scenario_id, sc_seed, prompt_id)
    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text())
            cached["_cache_hit"] = True
            return cached
        except (OSError, ValueError):
            pass
    last: Optional[Dict[str, Any]] = None
    for attempt in range(1, retries + 1):
        out = judge_clients.judge_ai(pairs, cfg, mode, model, PASS_T, MARGINAL_T, timeout=180,
                                     prompt_id=prompt_id)
        if out["ok"]:
            out["_cache_hit"] = False
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out))
            return out
        last = out
        if attempt < retries:
            wait = min(2 ** attempt, 20)
            print(f"    retry {attempt}/{retries} ({scenario_id}/{mode}): {out['error'][:120]}", file=sys.stderr)
            time.sleep(wait)
    last["_cache_hit"] = False
    return last


def run_scenario(sc: ed.Scenario, model: str, modes: List[str], base_millis: int,
                 use_cache: bool, prompt_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (legacy rows, long-format rows) for one scenario."""
    series = ed.series_for_scenario(sc)
    pairs = ed.build_pairs(sc, series, base_millis)
    stat_cfg = ed.build_canary_config(sc, {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}})

    rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    stat = judge_clients.run_default_judge(pairs, stat_cfg, PASS_T, MARGINAL_T)
    d_score = stat["score"]
    d_verdict = hybrid_policy.classify(d_score, PASS_T, MARGINAL_T)
    d_nonpass = any(p["classification"] not in hybrid_policy.PASS_OK for p in stat["per_metric"])
    rows.append(exp._row(sc, "statistical", "", "-", stat["verdict"], d_score, stat["latency"], stat["error"]))
    long_rows.append(_long(sc, "statistical", "", "-", "", "", stat))

    ai_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        ai = judge_ai_cached(pairs, ed.build_canary_config(sc, {}), mode, model, sc.id, sc.seed,
                             use_cache, prompt_id)
        ai_by_mode[mode] = ai
        note = "cache" if ai.get("_cache_hit") else "live"
        rows.append(exp._row(sc, "ai", mode, model, ai["verdict"], ai["score"], ai["latency"],
                             ai["error"], rationale=ai["rationale"], note=note))
        long_rows.append(_long(sc, "ai", mode, model,
                               ai.get("prompt_id") or prompt_id, ai.get("prompt_hash", ""), ai))

    prim = "plot" if "plot" in ai_by_mode else (modes[0] if modes else None)
    ai = ai_by_mode.get(prim) if prim else None
    if ai is not None and ai["ok"]:
        a_score = ai["score"]
        a_verdict = hybrid_policy.classify(a_score, PASS_T, MARGINAL_T)
        blind_text = ai["rationale"] + " " + " ".join(p["reason"] for p in ai["per_metric"])
        a_blind = hybrid_policy.text_flags_blindspot(blind_text)
        for pol in HYBRID_POLICIES:
            dec = hybrid_policy.decide_policy(pol, d_score, d_verdict, d_nonpass,
                                              a_score, a_verdict, a_blind, PASS_T, MARGINAL_T)
            hv = "PASS" if dec.verdict == "Pass" else "FAIL"
            rows.append(exp._row(sc, "hybrid", pol, model, hv, dec.score, 0.0, "",
                                 note=f"ai={prim}/{a_verdict};{dec.followed}"))
            long_rows.append(_long(sc, "hybrid", pol, model, "", "",
                                   {"verdict": hv, "score": dec.score, "latency": 0.0,
                                    "ok": True, "error": "", "rationale": ""}))
    return rows, long_rows


RUN_ID = ""  # set in main(); stamped onto every long-format row of this invocation


def _long(sc: ed.Scenario, judge: str, rep: str, model: str, prompt_id: str,
          prompt_hash: str, out: Dict[str, Any]) -> Dict[str, Any]:
    """One long-format row. Errors carry no verdict and no correctness."""
    ok = bool(out.get("ok", True))
    verdict = out.get("verdict", "") if ok else ""
    correct = "" if not ok else (1 if verdict == sc.truth else 0)
    return results_schema.new_row(
        run_id=RUN_ID,
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset_id=ed.family_dataset(sc.family),
        prompt_id=prompt_id, prompt_hash=prompt_hash,
        model=model, judge=judge, representation=rep,
        family=sc.family, scenario_id=sc.id, seed=sc.seed, truth_label=sc.truth,
        verdict=verdict, score=("" if not ok else out.get("score", "")), correct=correct,
        latency_s=round(float(out.get("latency", 0.0) or 0.0), 3),
        tokens_in=out.get("tokens_in") if out.get("tokens_in") is not None else "",
        tokens_out=out.get("tokens_out") if out.get("tokens_out") is not None else "",
        finish_reason=out.get("finish_reason", "") or "",
        error_kind=out.get("error_kind", "") or "",
        error=(out.get("error", "") or "")[:300],
        rationale=out.get("rationale", "") or "",
    )


def _stack_healthy(attempts: int = 40, gap: float = 15.0) -> bool:
    """Wait for Kayenta and judge-service, tolerating a busy judge-service.

    judge-service is single-worker uvicorn with a blocking handler, so while it is
    serving a judge call it does not answer /health at all. A single short probe
    therefore reports a perfectly healthy service as down whenever another run is
    in flight, which is exactly when a second run is most likely to be started.
    Retry for up to ten minutes before giving up. Three minutes was not enough:
    with four runners sharing the one worker, a queued /health probe timed out
    twelve times in a row and an entire 60-scenario arm never started.
    """
    deadline_note = f"{attempts} attempts, {gap:.0f}s apart"
    last = ""
    for i in range(1, attempts + 1):
        try:
            if requests.get("http://localhost:8090/health", timeout=10).json().get("status") != "UP":
                raise RuntimeError("Kayenta not UP")
            if requests.get("http://localhost:5001/health", timeout=30).json().get("status") != "UP":
                raise RuntimeError("judge-service not UP")
            return True
        except Exception as e:  # noqa: BLE001
            last = str(e)
            if i < attempts:
                print(f"[frontier] stack not answering yet ({last[:90]}); "
                      f"retry {i}/{attempts}", file=sys.stderr, flush=True)
                time.sleep(gap)
    print(f"[frontier] stack not healthy after {deadline_note} ({last}); "
          f"run `make up` first.", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Score one frontier hosted model")
    ap.add_argument("--model", default="claude-opus-4-8-vlm")
    ap.add_argument("--modes", default="summary,plot")
    ap.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", ed.DEFAULT_N_PER_FAMILY)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", ed.DEFAULT_MASTER_SEED)))
    ap.add_argument("--limit-families", default="", help="comma-separated family allow-list (debug/smoke)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--prompt-id", default=os.environ.get("JUDGE_PROMPT_ID", prompt_registry.DEFAULT_PROMPT_ID),
                    help="rubric to judge under (a filename stem in prompts/)")
    ap.add_argument("--dataset", default=ed.DEFAULT_DATASET_ID, choices=sorted(ed.DATASETS),
                    help="which scenario set to score")
    ap.add_argument("--out-tag", default="",
                    help="suffix the summary filenames, so a re-run under the default "
                         "prompt and dataset cannot land on top of a published one")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    fam_restrict = {f.strip() for f in args.limit_families.split(",") if f.strip()}
    use_cache = not args.no_cache

    # Resolve the rubric host-side too, so an unknown id fails before any call is
    # spent rather than 180 identical errors later.
    prompt = prompt_registry.load(args.prompt_id)

    global RUN_ID
    RUN_ID = f"{args.model}__{prompt.id}__{args.dataset}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    if not _stack_healthy():
        return 2

    scenarios = ed.build_scenarios(args.seed, args.n, args.dataset)
    if fam_restrict:
        scenarios = [s for s in scenarios if s.family in fam_restrict]
    base_millis = ed.aligned_base_millis()

    long_path = results_schema.EXPERIMENTS_DIR / f"{RUN_ID}.csv"

    print("=" * 78)
    print(f"kayenta-ai-canary-judge frontier-model experiment  model={args.model} modes={modes} "
          f"n={args.n}/family seed={args.seed} scenarios={len(scenarios)} cache={use_cache}")
    print(f"  prompt={prompt.id} ({prompt.hash})  dataset={args.dataset}")
    print(f"  config: {repo_env.describe()}")
    print(f"  long-format -> {long_path}")
    print("=" * 78)

    t0 = time.time()
    all_rows: List[Dict[str, Any]] = []
    n_error = 0
    for i, sc in enumerate(scenarios, 1):
        ts = time.time()
        rows, long_rows = run_scenario(sc, args.model, modes, base_millis, use_cache, prompt.id)
        all_rows.extend(rows)
        # Flush per scenario: a run killed at scenario 140 keeps 139 scenarios of
        # usable rows rather than none.
        results_schema.append(long_path, long_rows)
        n_cache = sum(1 for r in rows if r.get("note") == "cache")
        errs = [r for r in long_rows if r["error_kind"]]
        n_error += len(errs)
        flag = f" ERRORS={[e['error_kind'] for e in errs]}" if errs else ""
        print(f"  [{i:3d}/{len(scenarios)}] {sc.id:24s} truth={sc.truth} "
              f"({time.time()-ts:.0f}s, {n_cache} cached){flag}", flush=True)

    elapsed = time.time() - t0
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    write_outputs(all_rows, args.model, modes, elapsed, scenarios, args, prompt)
    print(f"\n[frontier] DONE in {elapsed/60:.1f} min -> {AGG_DIR}")
    print(f"[frontier] error rows: {n_error} (an error row is not a verdict; see {long_path})")
    return 0


def write_outputs(rows, model, modes, elapsed, scenarios, args, prompt):
    # The published frontier runs wrote three fixed filenames. A run under a
    # different rubric or dataset must not land on top of them, so anything
    # non-default is suffixed; the default configuration writes exactly where it
    # always did. The long-format frame in results/agg/experiments/ is the one
    # analysis reads -- these remain for continuity with the earlier checkpoints.
    tag = ""
    if getattr(args, "out_tag", ""):
        tag = f"__{args.out_tag}"
    elif prompt.id != prompt_registry.DEFAULT_PROMPT_ID or args.dataset != ed.DEFAULT_DATASET_ID:
        tag = f"__{prompt.id}__{args.dataset}"

    header = ["scenario", "family", "truth", "judge", "kind", "rep_or_policy", "model",
              "verdict", "score", "correct", "latency", "error", "note", "rationale"]
    exp.write_csv(AGG_DIR / f"frontier_long_{model}{tag}.csv", header,
                 [[r[k] for k in header] for r in rows])

    agg = exp.aggregate(rows)

    # stable, readable order: statistical, ai:<mode>..., hybrid:...
    def rank(key):
        j = key[0]
        if j == "statistical":
            return (0, key[1])
        if j.startswith("ai:"):
            return (1, j, key[1])
        if j.startswith("hybrid:"):
            return (2, j, key[1])
        return (9, j, key[1])
    order = sorted(agg.keys(), key=rank)

    sum_header = ["judge", "model", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
                  "TP", "FP", "TN", "FN", "avg_latency"]
    sum_rows = []
    for key in order:
        a = agg[key]
        sum_rows.append([key[0], key[1], a["n"], exp.r3(a["accuracy"]), exp.r3(a["precision"]),
                         exp.r3(a["recall"]), exp.r3(a["f1"]), exp.r3(a["fpr"]), exp.r3(a["fnr"]),
                         a["TP"], a["FP"], a["TN"], a["FN"], a["avg_latency"]])
    exp.write_csv(AGG_DIR / f"frontier_representation{tag}.csv", sum_header, sum_rows)

    families = sorted({s.family for s in scenarios}, key=ed.family_order)
    fam_header = ["judge", "model"] + families
    fam_rows = []
    by_group_family = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_group_family[(r["judge"], r["model"])][r["family"]].append(r["correct"])
    for key in order:
        row = [key[0], key[1]]
        for fam in families:
            vals = by_group_family[key].get(fam, [])
            row.append(exp.r3(sum(vals) / len(vals)) if vals else "")
        fam_rows.append(row)
    exp.write_csv(AGG_DIR / f"frontier_family_matrix{tag}.csv", fam_header, fam_rows)

    # short markdown comparison
    ai_keys = {k[0]: k for k in order if k[0].startswith("ai:")}
    L: List[str] = []
    L.append("# Frontier model: representation comparison")
    L.append("")
    L.append(f"_model={model} · modes={modes} · {len(scenarios)} scenarios · "
             f"{elapsed/60:.1f} min · seed={args.seed} n={args.n}/family._")
    L.append("")
    L.append("| representation | acc | prec | recall | F1 | FPR | FNR | avg latency (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mode in modes:
        k = ai_keys.get(f"ai:{mode}")
        if not k:
            continue
        a = agg[k]
        L.append(f"| {mode} | {a['accuracy']:.2f} | {a['precision']:.2f} | {a['recall']:.2f} | "
                 f"{a['f1']:.2f} | {a['fpr']:.2f} | {a['fnr']:.2f} | {a['avg_latency']:.1f} |")
    for pol in HYBRID_POLICIES:
        k = ("hybrid:" + pol, model)
        if k in agg:
            a = agg[k]
            L.append(f"| hybrid:{pol} | {a['accuracy']:.2f} | {a['precision']:.2f} | {a['recall']:.2f} | "
                     f"{a['f1']:.2f} | {a['fpr']:.2f} | {a['fnr']:.2f} | - |")
    L.append(f"| statistical | {agg[('statistical','-')]['accuracy']:.2f} | "
             f"{agg[('statistical','-')]['precision']:.2f} | {agg[('statistical','-')]['recall']:.2f} | "
             f"{agg[('statistical','-')]['f1']:.2f} | {agg[('statistical','-')]['fpr']:.2f} | "
             f"{agg[('statistical','-')]['fnr']:.2f} | {agg[('statistical','-')]['avg_latency']:.2f} |")
    L.append("")
    if "summary" in modes and "plot" in modes and ("ai:summary", model) in agg and ("ai:plot", model) in agg:
        sa, pa = agg[("ai:summary", model)]["accuracy"], agg[("ai:plot", model)]["accuracy"]
        sf, pf = agg[("ai:summary", model)]["f1"], agg[("ai:plot", model)]["f1"]
        verdict = "summary >= plot" if (sa >= pa and sf >= pf) else ("plot > summary" if pa > sa else "mixed")
        L.append(f"**Summary vs. plot at frontier scale**: accuracy {sa:.2f} vs {pa:.2f}, "
                 f"F1 {sf:.2f} vs {pf:.2f} -> **{verdict}**.")
        L.append("")
    L.append("## Per-family accuracy")
    L.append("")
    L.append("| judge | " + " | ".join(families) + " |")
    L.append("|" + "|".join(["---"] * (len(families) + 1)) + "|")
    judge_keys_unique = []
    for k in order:
        if k[0] not in judge_keys_unique:
            judge_keys_unique.append(k[0])
    for j in judge_keys_unique:
        cells = [j]
        keys = [k for k in order if k[0] == j]
        for fam in families:
            accs = []
            for key in keys:
                vals = [r["correct"] for r in rows if (r["judge"], r["model"]) == key and r["family"] == fam]
                if vals:
                    accs.append(sum(vals) / len(vals))
            cells.append(f"{sum(accs)/len(accs):.2f}" if accs else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    (AGG_DIR / f"frontier_opus{tag}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
