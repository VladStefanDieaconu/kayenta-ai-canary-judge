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
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import hybrid_policy  # noqa: E402
# Reuse the published experiment's pure scoring/aggregation code rather than redefining it.
import run_experiment as exp  # noqa: E402

AGG_DIR = REPO_ROOT / "results" / "agg"
CACHE_DIR = REPO_ROOT / "data" / "ai-logs" / "_frontier_cache"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]
HYBRID_POLICIES = ["gated", "or", "and"]


def _cache_path(model: str, mode: str, scenario_id: str, sc_seed: int) -> Path:
    # Keyed on the scenario's own deterministic seed (which fully determines its
    # generated series via ed.series_for_scenario), not the master seed or n.
    # A scenario_id like "variance_increase_000" maps to a different sc_seed (and
    # therefore different underlying data) depending on n_per_family, because the
    # gid counter that feeds the per-scenario seed runs across all families. Keying
    # on sc_seed instead of (scenario_id, master_seed) means a stale/mismatched
    # cache entry can't be mistaken for the right one.
    safe = f"{model}__{mode}__{scenario_id}__scseed{sc_seed}.json"
    return CACHE_DIR / safe


def judge_ai_cached(pairs, cfg, mode: str, model: str, scenario_id: str, sc_seed: int,
                    use_cache: bool, retries: int = 4) -> Dict[str, Any]:
    path = _cache_path(model, mode, scenario_id, sc_seed)
    if use_cache and path.exists():
        try:
            cached = json.loads(path.read_text())
            cached["_cache_hit"] = True
            return cached
        except (OSError, ValueError):
            pass
    last: Optional[Dict[str, Any]] = None
    for attempt in range(1, retries + 1):
        out = judge_clients.judge_ai(pairs, cfg, mode, model, PASS_T, MARGINAL_T, timeout=180)
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
                 use_cache: bool) -> List[Dict[str, Any]]:
    series = ed.series_for_scenario(sc)
    pairs = ed.build_pairs(sc, series, base_millis)
    stat_cfg = ed.build_canary_config(sc, {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}})

    rows: List[Dict[str, Any]] = []

    stat = judge_clients.run_default_judge(pairs, stat_cfg, PASS_T, MARGINAL_T)
    d_score = stat["score"]
    d_verdict = hybrid_policy.classify(d_score, PASS_T, MARGINAL_T)
    d_nonpass = any(p["classification"] not in hybrid_policy.PASS_OK for p in stat["per_metric"])
    rows.append(exp._row(sc, "statistical", "", "-", stat["verdict"], d_score, stat["latency"], stat["error"]))

    ai_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        ai = judge_ai_cached(pairs, ed.build_canary_config(sc, {}), mode, model, sc.id, sc.seed, use_cache)
        ai_by_mode[mode] = ai
        note = "cache" if ai.get("_cache_hit") else "live"
        rows.append(exp._row(sc, "ai", mode, model, ai["verdict"], ai["score"], ai["latency"],
                             ai["error"], rationale=ai["rationale"], note=note))

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
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Score one frontier hosted model")
    ap.add_argument("--model", default="claude-opus-4-8-vlm")
    ap.add_argument("--modes", default="summary,plot")
    ap.add_argument("--n", type=int, default=ed.DEFAULT_N_PER_FAMILY)
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--limit-families", default="", help="comma-separated family allow-list (debug/smoke)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    fam_restrict = {f.strip() for f in args.limit_families.split(",") if f.strip()}
    use_cache = not args.no_cache

    try:
        if requests.get("http://localhost:8090/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("Kayenta not UP")
        if requests.get("http://localhost:5001/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("judge-service not UP")
    except Exception as e:  # noqa: BLE001
        print(f"[frontier] stack not healthy ({e}); run `make up` first.", file=sys.stderr)
        return 2

    scenarios = ed.build_scenarios(args.seed, args.n)
    if fam_restrict:
        scenarios = [s for s in scenarios if s.family in fam_restrict]
    base_millis = ed.aligned_base_millis()

    print("=" * 78)
    print(f"kayenta-ai-canary-judge frontier-model experiment  model={args.model} modes={modes} "
          f"n={args.n}/family seed={args.seed} scenarios={len(scenarios)} cache={use_cache}")
    print("=" * 78)

    t0 = time.time()
    all_rows: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenarios, 1):
        ts = time.time()
        rows = run_scenario(sc, args.model, modes, base_millis, use_cache)
        all_rows.extend(rows)
        n_cache = sum(1 for r in rows if r.get("note") == "cache")
        print(f"  [{i:3d}/{len(scenarios)}] {sc.id:24s} truth={sc.truth} "
              f"({time.time()-ts:.0f}s, {n_cache} cached)")

    elapsed = time.time() - t0
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    write_outputs(all_rows, args.model, modes, elapsed, scenarios, args)
    print(f"\n[frontier] DONE in {elapsed/60:.1f} min -> {AGG_DIR}")
    return 0


def write_outputs(rows, model, modes, elapsed, scenarios, args):
    header = ["scenario", "family", "truth", "judge", "kind", "rep_or_policy", "model",
              "verdict", "score", "correct", "latency", "error", "note", "rationale"]
    exp.write_csv(AGG_DIR / f"frontier_long_{model}.csv", header,
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
    exp.write_csv(AGG_DIR / "frontier_representation.csv", sum_header, sum_rows)

    families = sorted({s.family for s in scenarios}, key=lambda f: ed.FAMILIES.index(f))
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
    exp.write_csv(AGG_DIR / "frontier_family_matrix.csv", fam_header, fam_rows)

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
    (AGG_DIR / "frontier_opus.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
