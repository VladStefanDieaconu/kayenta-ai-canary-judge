#!/usr/bin/env python3
"""Give the ensemble judge a multi-seed confidence interval.

Re-runs the ensemble (tools/run_ensemble_experiment.py's
materialise()/evaluate_config()) across the same 5 seeds used by
tools/run_experiment_multiseed.py (20260621/1/2/3/4, n=5/family), using the
already-tuned config from results/agg/ensemble_config.json (6 knobs:
variance_ratio_threshold=2.06, variance_alpha=0.181, tail_ratio_threshold=1.40,
tail_alpha=0.174, trend_alpha=0.181, cross_metric_alpha=0.050) rather than
re-tuning, so the tuning seeds and the reported seeds stay disjoint.

The ensemble's baseline per-metric verdict still comes from Kayenta's genuine
NetflixACAJudge (via judge_clients.run_default_judge, the same call
tools/run_ensemble_experiment.py and the hybrid make): a local, free,
in-process statistical judge, not an LLM. No AI judge is called here.

Bounded CI method: Wilson score interval on pooled counts across the 5 seeds
(primary), seed-level percentile bootstrap (cross-check), via
tools/ci_utils.py, matching tools/bounded_confidence_intervals.py.

Output: results/agg/ensemble_multiseed_ci.csv, holding
  (a) per-seed and pooled ensemble metrics with bounded CI,
  (b) a side-by-side of the ensemble CI, the best local LLM's CI (from
      results/agg/metrics_with_ci.csv), and the frontier point estimate
      (results/agg/frontier_representation.csv).

Usage:
  python tools/ensemble_multiseed_ci.py [--seeds 20260621,1,2,3,4] [--n 5]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import ci_utils as ci  # noqa: E402
import ensemble_judge as ej  # noqa: E402
import run_ensemble_experiment as ree  # noqa: E402  (reuse materialise/evaluate_config)
import eval_dataset as ed  # noqa: E402
import json

AGG_DIR = REPO_ROOT / "results" / "agg"
DEFAULT_SEEDS = [20260621, 1, 2, 3, 4]
DEFAULT_N = 5


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_tuned_config() -> ej.EnsembleConfig:
    cfg_json = json.loads((AGG_DIR / "ensemble_config.json").read_text())
    tc = cfg_json["tuned_config"]
    return ej.EnsembleConfig(**tc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-seed CI for the tuned ensemble")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cfg = load_tuned_config()
    print(f"[ensemble-multiseed] tuned config (from results/agg/ensemble_config.json): {cfg.as_dict()}")
    print(f"[ensemble-multiseed] seeds={seeds} n={args.n}/family")

    t0 = time.time()
    per_seed_rows: Dict[int, List[Dict[str, Any]]] = {}
    for seed in seeds:
        materialised = ree.materialise(seed, args.n)
        result = ree.evaluate_config(materialised, cfg)
        per_seed_rows[seed] = result["rows"]
        print(f"  seed={seed}: acc={result['metrics']['accuracy']:.3f} "
              f"rec={result['metrics']['recall']:.3f} prec={result['metrics']['precision']:.3f} "
              f"({time.time()-t0:.0f}s elapsed)")

    all_rows: List[Dict[str, Any]] = []
    for seed, rows in per_seed_rows.items():
        for r in rows:
            r2 = dict(r)
            r2["seed"] = seed
            all_rows.append(r2)

    # persisted so tools/pooled_mcnemar.py can pair ensemble correctness
    # against other judges' rows without re-calling Kayenta.
    write_csv(AGG_DIR / "ensemble_multiseed_rows.csv",
              ["scenario", "family", "seed", "truth", "verdict", "correct"],
              [[r["scenario"], r["family"], r["seed"], r["truth"], r["verdict"], r["correct"]]
               for r in all_rows])

    # pooled and per-seed metrics, bounded CI (same method as bounded_confidence_intervals.py)
    metrics = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    c_pooled = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in all_rows:
        truth_fail = r["truth"] == "FAIL"
        pred_fail = r["verdict"] == "FAIL"
        if truth_fail and pred_fail:
            c_pooled["TP"] += 1
        elif (not truth_fail) and pred_fail:
            c_pooled["FP"] += 1
        elif (not truth_fail) and (not pred_fail):
            c_pooled["TN"] += 1
        else:
            c_pooled["FN"] += 1
    pooled_metrics = ci.metrics_from_conf(c_pooled)

    per_seed_metric_vals: Dict[str, List[float]] = defaultdict(list)
    for seed in seeds:
        c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for r in per_seed_rows[seed]:
            truth_fail = r["truth"] == "FAIL"
            pred_fail = r["verdict"] == "FAIL"
            if truth_fail and pred_fail:
                c["TP"] += 1
            elif (not truth_fail) and pred_fail:
                c["FP"] += 1
            elif (not truth_fail) and (not pred_fail):
                c["TN"] += 1
            else:
                c["FN"] += 1
        m = ci.metrics_from_conf(c)
        for k, v in m.items():
            per_seed_metric_vals[k].append(v)

    wilson_by_metric = {
        "accuracy": ci.wilson_interval(c_pooled["TP"] + c_pooled["TN"], sum(c_pooled.values())),
        "precision": ci.wilson_interval(c_pooled["TP"], c_pooled["TP"] + c_pooled["FP"]),
        "recall": ci.wilson_interval(c_pooled["TP"], c_pooled["TP"] + c_pooled["FN"]),
        "fpr": ci.wilson_interval(c_pooled["FP"], c_pooled["FP"] + c_pooled["TN"]),
        "fnr": ci.wilson_interval(c_pooled["FN"], c_pooled["FN"] + c_pooled["TP"]),
    }

    header = ["judge", "n_seeds", "n_pooled"]
    for m in metrics:
        header += [f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high", f"{m}_boot_low", f"{m}_boot_high"]
    header += ["identical_across_seeds"]

    row = ["ensemble:tuned", len(seeds), len(all_rows)]
    identical = []
    for m in metrics:
        mean_v = pooled_metrics[m]
        vals = per_seed_metric_vals[m]
        zero_var = ci.is_zero_variance(vals)
        if m == "f1":
            wlo, whi = ("", "")
        else:
            wlo, whi = wilson_by_metric[m]
        if zero_var:
            blo, bhi = (vals[0], vals[0])
            identical.append(m)
        else:
            blo, bhi = ci.percentile_bootstrap(vals, lambda rs: sum(rs) / len(rs))
        row += [ci.r3(mean_v), ci.r3(wlo) if wlo != "" else "", ci.r3(whi) if whi != "" else "",
                ci.r3(blo), ci.r3(bhi)]
    row.append(";".join(identical))

    ensemble_rows = [row]

    # side-by-side: ensemble vs best local LLM CI vs frontier point estimate
    llm_ci_rows = read_csv(AGG_DIR / "metrics_with_ci.csv")
    best_llm = max(
        (r for r in llm_ci_rows if r["judge"] == "ai:summary"),
        key=lambda r: float(r["accuracy_mean"]),
    )
    frontier_rows = read_csv(AGG_DIR / "frontier_representation.csv") if (AGG_DIR / "frontier_representation.csv").exists() else []
    frontier_best = max(
        (r for r in frontier_rows if r["judge"].startswith("ai:")),
        key=lambda r: float(r["accuracy"]),
        default=None,
    )

    compare_header = ["config", "n", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "ci_method", "note"]
    compare_rows = [
        ["ensemble:tuned (5-seed, n=5/family)", len(all_rows), ci.r3(pooled_metrics["accuracy"]),
         ci.r3(wilson_by_metric["accuracy"][0]), ci.r3(wilson_by_metric["accuracy"][1]),
         "wilson(pooled)", "zero LLM calls; 6-knob statistical ensemble on top of the genuine judge"],
        [f"best local LLM: ai:summary/{best_llm['model']} (5-seed, n=5/family)",
         best_llm["n_pooled"], best_llm["accuracy_mean"], best_llm["accuracy_ci_low"], best_llm["accuracy_ci_high"],
         "wilson(pooled), from bounded_confidence_intervals.py", "same 5-seed multiseed scope as the ensemble row above"],
    ]
    if frontier_best is not None:
        compare_rows.append([
            f"frontier: {frontier_best['judge']}/{frontier_best['model']} (single seed, N=20, point estimate only)",
            frontier_best["n"], frontier_best["accuracy"], "", "",
            "none (single run, no seed axis)",
            "frontier run; not re-run here, point estimate only, no CI possible from 1 seed",
        ])
    # two clearly-separated blocks in the same file: per-seed/pooled ensemble
    # metrics, then the ensemble-vs-LLM-vs-frontier comparison.
    with (AGG_DIR / "ensemble_multiseed_ci.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(ensemble_rows)
        w.writerow([])
        w.writerow(compare_header)
        w.writerows(compare_rows)

    print(f"\n[ensemble-multiseed] DONE in {(time.time()-t0)/60:.1f} min -> "
          f"{AGG_DIR / 'ensemble_multiseed_ci.csv'}")
    print(f"[ensemble-multiseed] ensemble:tuned pooled acc={pooled_metrics['accuracy']:.3f} "
          f"CI=[{wilson_by_metric['accuracy'][0]:.3f}, {wilson_by_metric['accuracy'][1]:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
