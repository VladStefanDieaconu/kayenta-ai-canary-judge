#!/usr/bin/env python3
"""Multi-seed judge sweep, so metrics come with a confidence interval instead
of a single-run point estimate.

Runs the same judge sweep as tools/run_experiment.py, reusing its pure
functions (discover_models, run_scenario, confusion, metrics_from_conf,
aggregate, mcnemar_exact, cohen_kappa) across multiple master seeds, so
per-judge metrics can be reported as mean +/- 95% CI across seeds.

Superseded by tools/run_multiseed_corrected.py, which runs the same sweep but
writes through results_schema: this one's long_results.csv carries no rationale
column and discards failed calls instead of recording them, which is what made
the artefact it produced un-auditable. Kept because `make analysis-live` still
names it and because it is the runner the published multi-seed artefact came
from. Prefer the corrected runner for anything new.

Scope note: a full N=20 x 5-seed re-run of the 8-model sweep would take
~25 hours (the N=20/180-scenario single-seed run took 304.9 min on its own).
This runs N=5/family x 5 seeds (225 scenario-equivalents, ~6-7h). One of the 5
seeds is the master seed 20260621, so that slice is directly comparable to a
single-seed run at the same n.

Caching + retry: judge_clients.judge_ai and judge_clients.run_default_judge are
monkey-patched (a module-attribute swap, so tools/run_experiment.py's
run_scenario() picks up the wrapped versions transparently, with no duplicated
sweep logic) with a disk cache keyed on a hash of the actual pairs content sent.
The key is content-addressed rather than scenario-name/seed indirection, so it
can't serve a stale result for different underlying data, plus retry-with-backoff
on failure. This makes the many-hour run resumable: killing and restarting
re-uses every already-computed cell.

Outputs (results/agg/):
  - long_results.csv       one row per (scenario, family, seed, judge, model);
                           the single source for every metric/figure below.
  - metrics_with_ci.csv    per (judge, model): mean +/- 95% CI across seeds.
  - family_recall_with_ci.csv   per (judge, model, family): mean +/- 95% CI
                           across seeds (this equals recall for FAIL families
                           and specificity/TNR for PASS families, since every
                           scenario in a family shares one ground truth).
  - mcnemar_per_seed.csv   statistical vs each judge, computed independently
                           per seed (not pooled), so seed-to-seed significance
                           variability (the N=5 -> N=20 finding in the README)
                           is visible directly.

Usage:
  python tools/run_experiment_multiseed.py [--n 5] [--seeds 20260621,1,2,3,4]
                                           [--models a,b] [--no-cache]
"""

from __future__ import annotations

import argparse
import hashlib
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

AGG_DIR = REPO_ROOT / "results" / "agg"
CACHE_DIR = REPO_ROOT / "data" / "ai-logs" / "_multiseed_cache"
DEFAULT_SEEDS = [20260621, 1, 2, 3, 4]
DEFAULT_N = 5

# Caching + retry wrappers, installed onto judge_clients before importing
# run_experiment, so run_experiment.run_scenario's `judge_clients.judge_ai(...)`
# module-attribute lookups resolve to these at call time.
_orig_judge_ai = judge_clients.judge_ai
_orig_run_default_judge = judge_clients.run_default_judge


def _content_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _cached_judge_ai(pairs, cfg, mode, model, pass_t=75.0, marginal_t=50.0,
                     judge_url=judge_clients.JUDGE_URL, timeout=300):
    key = _content_key("ai", mode, model, pairs)
    path = CACHE_DIR / f"{key}.json"
    if _USE_CACHE and path.exists():
        try:
            out = json.loads(path.read_text())
            out["_cache_hit"] = True
            return out
        except (OSError, ValueError):
            pass
    last = None
    for attempt in range(1, 4):
        out = _orig_judge_ai(pairs, cfg, mode, model, pass_t, marginal_t, judge_url, timeout)
        if out["ok"]:
            out["_cache_hit"] = False
            if _USE_CACHE:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out))
            return out
        last = out
        if attempt < 3:
            time.sleep(min(2 ** attempt, 15))
    last["_cache_hit"] = False
    return last


def _cached_run_default_judge(pairs, config, pass_t=75.0, marginal_t=50.0,
                              base_url=judge_clients.KAYENTA_URL):
    key = _content_key("stat", pairs)
    path = CACHE_DIR / f"{key}.json"
    if _USE_CACHE and path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            pass
    out = _orig_run_default_judge(pairs, config, pass_t, marginal_t, base_url)
    if out["ok"] and _USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


_USE_CACHE = True
judge_clients.judge_ai = _cached_judge_ai
judge_clients.run_default_judge = _cached_run_default_judge

import run_experiment as exp  # noqa: E402  (picks up the patched judge_clients calls)


# Stats helpers (pure stdlib). n is always small (number of seeds), so a
# Student's-t critical value table covers every df we'll see; falls back to a
# normal-approximation z=1.96 outside the table.
_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci95(values: List[float]) -> Tuple[float, float, float, int]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    m = sum(values) / n
    if n == 1:
        return (m, m, m, n)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sd = var ** 0.5
    se = sd / (n ** 0.5)
    t = _T_TABLE.get(n - 1, 1.96)
    half = t * se
    return (m, m - half, m + half, n)


def run_one_seed(seed: int, n_per_family: int, present, modes_text, modes_vision) -> List[Dict[str, Any]]:
    scenarios = ed.build_scenarios(seed, n_per_family)
    base_millis = ed.aligned_base_millis()
    rows: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenarios, 1):
        t0 = time.time()
        srows = exp.run_scenario(sc, present, base_millis, modes_text, modes_vision)
        for r in srows:
            r["seed"] = seed
        rows.extend(srows)
        n_cache = sum(1 for r in srows if r.get("note") == "cache")
        print(f"  seed={seed} [{i:3d}/{len(scenarios)}] {sc.id:24s} truth={sc.truth} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-seed judge sweep with confidence intervals")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--models", default=os.environ.get("EVAL_MODELS", ""),
                    help="comma-separated alias allow-list (default: all present)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    global _USE_CACHE
    _USE_CACHE = not args.no_cache

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    restrict = [m.strip() for m in args.models.split(",") if m.strip()] or None

    try:
        if requests.get("http://localhost:8090/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("Kayenta not UP")
        if requests.get("http://localhost:5001/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("judge-service not UP")
    except Exception as e:  # noqa: BLE001
        print(f"[multiseed] stack not healthy ({e}); run `make up` first.", file=sys.stderr)
        return 2

    present, skipped = exp.discover_models(restrict)
    if not present:
        print("[multiseed] no models present locally.", file=sys.stderr)
        return 1
    print(f"[multiseed] present models: {[m['alias'] for m in present]}")
    print(f"[multiseed] seeds={seeds} n={args.n}/family cache={_USE_CACHE}")

    t_start = time.time()
    all_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        rows = run_one_seed(seed, args.n, present, exp.TEXT_MODES, exp.VISION_MODES)
        all_rows.extend(rows)
        # checkpoint after every seed, so a crash never loses more than one seed's work
        AGG_DIR.mkdir(parents=True, exist_ok=True)
        write_long_results(all_rows)
        print(f"[multiseed] checkpointed after seed {seed} ({len(all_rows)} rows so far)")

    write_outputs(all_rows, seeds, present)
    print(f"\n[multiseed] DONE in {(time.time()-t_start)/60:.1f} min -> {AGG_DIR}")
    return 0


def write_long_results(rows: List[Dict[str, Any]]) -> None:
    header = ["scenario", "family", "seed", "judge", "model", "y_true", "y_pred",
              "score", "correct", "latency", "error", "note"]
    out_rows = []
    for r in rows:
        out_rows.append([r["scenario"], r["family"], r["seed"], r["judge"], r["model"],
                         r["truth"], r["verdict"], r["score"], r["correct"], r["latency"],
                         r["error"], r["note"]])
    exp.write_csv(AGG_DIR / "long_results.csv", header, out_rows)


def write_outputs(rows: List[Dict[str, Any]], seeds: List[int], present) -> None:
    write_long_results(rows)

    # per-seed, per (judge, model) confusion -> accuracy/precision/recall/F1/FPR/FNR
    by_seed_group: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seed_group[(r["seed"], r["judge"], r["model"])].append(r)

    per_seed_metrics: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (seed, judge, model), rs in by_seed_group.items():
        c = exp.confusion(rs)
        m = exp.metrics_from_conf(c)
        for k, v in m.items():
            per_seed_metrics[(judge, model)][k].append(v)

    order = sorted(per_seed_metrics.keys(), key=lambda k: (
        0 if k[0] == "statistical" else (1 if k[0].startswith("ai:") else 2), k[0], k[1]))

    ci_header = ["judge", "model", "n_seeds", "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high",
                 "precision_mean", "precision_ci_low", "precision_ci_high",
                 "recall_mean", "recall_ci_low", "recall_ci_high",
                 "f1_mean", "f1_ci_low", "f1_ci_high",
                 "fpr_mean", "fnr_mean"]
    ci_rows = []
    for key in order:
        d = per_seed_metrics[key]
        acc = mean_ci95(d["accuracy"])
        prec = mean_ci95(d["precision"])
        rec = mean_ci95(d["recall"])
        f1 = mean_ci95(d["f1"])
        ci_rows.append([key[0], key[1], acc[3],
                        exp.r3(acc[0]), exp.r3(acc[1]), exp.r3(acc[2]),
                        exp.r3(prec[0]), exp.r3(prec[1]), exp.r3(prec[2]),
                        exp.r3(rec[0]), exp.r3(rec[1]), exp.r3(rec[2]),
                        exp.r3(f1[0]), exp.r3(f1[1]), exp.r3(f1[2]),
                        exp.r3(sum(d["fpr"]) / len(d["fpr"])), exp.r3(sum(d["fnr"]) / len(d["fnr"]))])
    exp.write_csv(AGG_DIR / "metrics_with_ci.csv", ci_header, ci_rows)

    # per-family accuracy (== recall for FAIL families) per seed, then mean+CI
    families = ed.FAMILIES
    by_seed_group_family: Dict[Tuple[int, str, str, str], List[int]] = defaultdict(list)
    for r in rows:
        by_seed_group_family[(r["seed"], r["judge"], r["model"], r["family"])].append(r["correct"])

    fam_ci_header = ["judge", "model", "family", "n_seeds", "accuracy_mean", "ci_low", "ci_high"]
    fam_ci_rows = []
    for key in order:
        judge, model = key
        for fam in families:
            per_seed_vals = []
            for seed in seeds:
                vals = by_seed_group_family.get((seed, judge, model, fam))
                if vals:
                    per_seed_vals.append(sum(vals) / len(vals))
            if per_seed_vals:
                m = mean_ci95(per_seed_vals)
                fam_ci_rows.append([judge, model, fam, m[3], exp.r3(m[0]), exp.r3(m[1]), exp.r3(m[2])])
    exp.write_csv(AGG_DIR / "family_recall_with_ci.csv", fam_ci_header, fam_ci_rows)

    # McNemar per seed: statistical vs each other judge, per model
    mcnemar_rows = []
    idx_correct: Dict[Tuple[int, str, str], Dict[str, bool]] = defaultdict(dict)
    for r in rows:
        idx_correct[(r["seed"], r["judge"], r["model"])][r["scenario"]] = bool(r["correct"])
    for seed in seeds:
        for mdl in present:
            alias = mdl["alias"]
            stat_key = (seed, "statistical", "-")
            for j in (f"ai:{exp.PRIMARY_MODE[mdl['modality']]}", "hybrid:gated"):
                other_key = (seed, j, alias)
                scen = sorted(set(idx_correct.get(stat_key, {})) & set(idx_correct.get(other_key, {})))
                if not scen:
                    continue
                ca = [idx_correct[stat_key][s] for s in scen]
                cb = [idx_correct[other_key][s] for s in scen]
                mc = exp.mcnemar_exact(ca, cb)
                mcnemar_rows.append([seed, alias, "statistical", j, mc["discordant"],
                                     mc["n01_a_wrong_b_right"], mc["n10_a_right_b_wrong"],
                                     mc["chi2_cc"], mc["p_value"]])
    exp.write_csv(AGG_DIR / "mcnemar_per_seed.csv",
                 ["seed", "model", "judge_a", "judge_b", "discordant", "a_wrong_b_right",
                  "a_right_b_wrong", "chi2_cc", "p_value"], mcnemar_rows)

    print(f"[multiseed] wrote long_results.csv ({len(rows)} rows), "
          f"metrics_with_ci.csv ({len(ci_rows)} rows), "
          f"family_recall_with_ci.csv ({len(fam_ci_rows)} rows), "
          f"mcnemar_per_seed.csv ({len(mcnemar_rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
