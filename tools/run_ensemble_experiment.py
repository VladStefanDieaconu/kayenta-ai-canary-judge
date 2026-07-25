#!/usr/bin/env python3
"""Statistical-ensemble baseline judge: tuning and held-out evaluation.

Doesn't reimplement the Mann-Whitney judge: the genuine NetflixACAJudge is
called once per scenario (via judge_clients.run_default_judge, exactly as the
hybrid already does) and cached, since it doesn't depend on the ensemble config
at all. Everything the ensemble adds (ensemble_judge.decide_ensemble) is pure
in-memory computation, so tuning can afford a real random search instead of
hand-picked defaults.

Two disjoint seed sets, by design: tune on held-out seeds, not on the reported
set.
  - Tuning seeds: used only to pick the ensemble's 6 knobs (random search,
    fixed trial budget, reported in full so the configuration cost is visible).
  - Reported seed: the same master seed (20260621) and 180-scenario dataset
    used in results/, so the ensemble's numbers are directly comparable to the
    statistical/AI/hybrid rows there, with no tuning-set leakage into the
    reported figure.

Outputs (results/agg/):
  - ensemble_config.json       the tuned config plus the full trial log (so the
                               "how many knobs / how brittle" question is
                               answerable from the artifact, not just prose).
  - ensemble_sensitivity.csv   one-at-a-time perturbation sweep around the
                               tuned config, evaluated on the reported set.
  - ensemble_scoreboard.csv    statistical vs ensemble vs ensemble-no-guard,
                               on the reported set (Table-5-shaped).
  - ensemble_family_matrix.csv per-family accuracy, same shape as
                               results/family_matrix.csv.

Usage:
  python tools/run_ensemble_experiment.py [--tuning-seeds 111,222,333] [--tuning-n 10]
                                          [--reported-seed 20260621] [--reported-n 20]
                                          [--trials 300]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import eval_dataset as ed  # noqa: E402
import judge_clients  # noqa: E402
import ensemble_judge as ej  # noqa: E402
import run_experiment as exp  # noqa: E402  (reuse confusion/metrics_from_conf/write_csv/r3)

AGG_DIR = REPO_ROOT / "results" / "agg"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]
BLIND_SPOT_FAMILIES = ["variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]
PASS_FAMILIES = ["no_change", "noise_equivalent", "healed_transient"]


# Materialise a scenario set once: fetch the genuine statistical judge's
# per-scenario result (the only part that needs a network call) and keep the
# raw pairs alongside it. Every candidate ensemble config then runs only pure
# in-memory computation over this cached list.
def materialise(seed: int, n_per_family: int) -> List[Dict[str, Any]]:
    scenarios = ed.build_scenarios(seed, n_per_family)
    base_millis = ed.aligned_base_millis()
    out = []
    for sc in scenarios:
        series = ed.series_for_scenario(sc)
        pairs = ed.build_pairs(sc, series, base_millis)
        cfg = ed.build_canary_config(sc, {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}})
        stat = judge_clients.run_default_judge(pairs, cfg, PASS_T, MARGINAL_T)
        out.append({"scenario": sc.id, "family": sc.family, "truth": sc.truth,
                    "pairs": pairs, "stat_verdict": stat["verdict"], "stat_score": stat["score"],
                    "stat_per_metric": stat["per_metric"]})
    return out


def evaluate_config(materialised: List[Dict[str, Any]], cfg: ej.EnsembleConfig) -> Dict[str, Any]:
    rows = []
    for m in materialised:
        dec = ej.decide_ensemble(m["pairs"], m["stat_per_metric"], m["stat_score"], cfg, PASS_T, MARGINAL_T)
        verdict = "PASS" if dec.verdict == "Pass" else "FAIL"
        rows.append({"scenario": m["scenario"], "family": m["family"], "truth": m["truth"],
                     "verdict": verdict, "correct": int(verdict == m["truth"])})
    c = exp.confusion(rows)
    metrics = exp.metrics_from_conf(c)
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r["correct"])
    family_acc = {fam: (sum(v) / len(v) if v else None) for fam, v in by_family.items()}
    return {"metrics": metrics, "confusion": c, "family_acc": family_acc, "rows": rows}


def objective(result: Dict[str, Any]) -> float:
    """Reward recall on the 4 blind-spot families; penalise dropping specificity
    (i.e. adding false positives) on the 3 genuinely-PASS families below a floor."""
    fam_acc = result["family_acc"]
    blind_vals = [fam_acc[f] for f in BLIND_SPOT_FAMILIES if fam_acc.get(f) is not None]
    pass_vals = [fam_acc[f] for f in PASS_FAMILIES if fam_acc.get(f) is not None]
    blind_mean = sum(blind_vals) / len(blind_vals) if blind_vals else 0.0
    pass_mean = sum(pass_vals) / len(pass_vals) if pass_vals else 1.0
    floor = 0.85
    penalty = max(0.0, floor - pass_mean) * 2.0
    return blind_mean - penalty


# Random search over the 6 continuous knobs. use_recovery_guard is held True
# during tuning; it's an ablation switch, evaluated separately at the end.
SEARCH_SPACE = {
    "variance_ratio_threshold": (1.2, 4.0),
    "variance_alpha": (0.01, 0.20),
    "tail_ratio_threshold": (1.02, 1.5),
    "tail_alpha": (0.01, 0.20),
    "trend_alpha": (0.01, 0.20),
    "cross_metric_alpha": (0.01, 0.20),
}


def random_config(rng: random.Random) -> ej.EnsembleConfig:
    kwargs = {k: rng.uniform(lo, hi) for k, (lo, hi) in SEARCH_SPACE.items()}
    return ej.EnsembleConfig(**kwargs)


def tune(materialised_tuning: List[Dict[str, Any]], trials: int, seed: int) -> Tuple[ej.EnsembleConfig, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    log: List[Dict[str, Any]] = []
    best_cfg, best_obj = None, -1e9
    for i in range(trials):
        cfg = random_config(rng)
        result = evaluate_config(materialised_tuning, cfg)
        obj = objective(result)
        log.append({"trial": i, **cfg.as_dict(), "objective": round(obj, 4),
                    "accuracy": round(result["metrics"]["accuracy"], 4)})
        if obj > best_obj:
            best_obj, best_cfg = obj, cfg
    return best_cfg, log


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune and evaluate the statistical-ensemble judge")
    ap.add_argument("--tuning-seeds", default="111,222,333")
    ap.add_argument("--tuning-n", type=int, default=10)
    ap.add_argument("--reported-seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--reported-n", type=int, default=20)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    tuning_seeds = [int(s) for s in args.tuning_seeds.split(",") if s]

    print("=" * 78)
    print("Statistical-ensemble baseline: tuning + held-out evaluation")
    print("=" * 78)

    t0 = time.time()
    print(f"[ensemble] materialising TUNING scenarios (seeds={tuning_seeds}, n={args.tuning_n}/family) ...")
    materialised_tuning: List[Dict[str, Any]] = []
    for s in tuning_seeds:
        materialised_tuning.extend(materialise(s, args.tuning_n))
    print(f"[ensemble] {len(materialised_tuning)} tuning scenarios materialised in {time.time()-t0:.0f}s")

    print(f"[ensemble] random search: {args.trials} trials on the TUNING set only ...")
    t1 = time.time()
    best_cfg, log = tune(materialised_tuning, args.trials, args.random_seed)
    print(f"[ensemble] search done in {time.time()-t1:.0f}s. Best tuning objective: "
          f"{max(l['objective'] for l in log):.4f}")
    print(f"[ensemble] tuned config: {best_cfg.as_dict()}")

    print(f"[ensemble] materialising REPORTED scenarios (seed={args.reported_seed}, "
          f"n={args.reported_n}/family; the published N=20 dataset) ...")
    t2 = time.time()
    materialised_reported = materialise(args.reported_seed, args.reported_n)
    print(f"[ensemble] {len(materialised_reported)} reported scenarios materialised in {time.time()-t2:.0f}s")

    # headline: statistical alone vs tuned ensemble (with/without guard)
    stat_rows = [{"scenario": m["scenario"], "family": m["family"], "truth": m["truth"],
                 "verdict": m["stat_verdict"], "correct": int(m["stat_verdict"] == m["truth"])}
                for m in materialised_reported]
    stat_c = exp.confusion(stat_rows)
    stat_metrics = exp.metrics_from_conf(stat_c)

    ensemble_result = evaluate_config(materialised_reported, best_cfg)
    no_guard_cfg = replace(best_cfg, use_recovery_guard=False)
    no_guard_result = evaluate_config(materialised_reported, no_guard_cfg)

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    write_outputs(stat_metrics, stat_c, materialised_reported, ensemble_result, no_guard_result,
                 best_cfg, log, tuning_seeds, args)
    print(f"\n[ensemble] DONE in {(time.time()-t0)/60:.1f} min -> {AGG_DIR}")
    return 0


def _fam_row(family_acc: Dict[str, Any]) -> List[Any]:
    return [family_acc.get(f, "") for f in ed.FAMILIES]


def write_outputs(stat_metrics, stat_c, materialised_reported, ensemble_result, no_guard_result,
                  best_cfg, log, tuning_seeds, args):
    # scoreboard
    header = ["judge", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "TP", "FP", "TN", "FN"]
    rows = []
    n = len(materialised_reported)
    rows.append(["statistical", n, exp.r3(stat_metrics["accuracy"]), exp.r3(stat_metrics["precision"]),
                exp.r3(stat_metrics["recall"]), exp.r3(stat_metrics["f1"]), exp.r3(stat_metrics["fpr"]),
                exp.r3(stat_metrics["fnr"]), stat_c["TP"], stat_c["FP"], stat_c["TN"], stat_c["FN"]])
    for label, result in (("ensemble:tuned", ensemble_result), ("ensemble:no_recovery_guard", no_guard_result)):
        m, c = result["metrics"], result["confusion"]
        rows.append([label, n, exp.r3(m["accuracy"]), exp.r3(m["precision"]), exp.r3(m["recall"]),
                    exp.r3(m["f1"]), exp.r3(m["fpr"]), exp.r3(m["fnr"]), c["TP"], c["FP"], c["TN"], c["FN"]])
    exp.write_csv(AGG_DIR / "ensemble_scoreboard.csv", header, rows)

    # family matrix
    fam_header = ["judge"] + ed.FAMILIES
    stat_by_family = defaultdict(list)
    for m in materialised_reported:
        stat_by_family[m["family"]].append(int(m["stat_verdict"] == m["truth"]))
    stat_family_acc = {f: (sum(v) / len(v) if v else None) for f, v in stat_by_family.items()}
    fam_rows = [
        ["statistical"] + [exp.r3(v) if v is not None else "" for v in _fam_row(stat_family_acc)],
        ["ensemble:tuned"] + [exp.r3(v) if v is not None else "" for v in _fam_row(ensemble_result["family_acc"])],
        ["ensemble:no_recovery_guard"] + [exp.r3(v) if v is not None else "" for v in _fam_row(no_guard_result["family_acc"])],
    ]
    exp.write_csv(AGG_DIR / "ensemble_family_matrix.csv", fam_header, fam_rows)

    # sensitivity sweep (one-at-a-time perturbation around the tuned config)
    sens_header = ["knob", "perturbation", "value", "accuracy", "recall_blindspot", "specificity_pass"]
    sens_rows = []
    base_result = ensemble_result
    base_blind = sum(base_result["family_acc"].get(f, 0) or 0 for f in BLIND_SPOT_FAMILIES) / len(BLIND_SPOT_FAMILIES)
    base_pass = sum(base_result["family_acc"].get(f, 0) or 0 for f in PASS_FAMILIES) / len(PASS_FAMILIES)
    sens_rows.append(["(tuned baseline)", "0%", "-", exp.r3(base_result["metrics"]["accuracy"]),
                      exp.r3(base_blind), exp.r3(base_pass)])
    for knob in SEARCH_SPACE:
        base_val = getattr(best_cfg, knob)
        for pct in (-0.5, -0.2, 0.2, 0.5):
            new_val = max(1e-6, base_val * (1 + pct))
            cfg2 = replace(best_cfg, **{knob: new_val})
            result2 = evaluate_config(materialised_reported, cfg2)
            blind2 = sum(result2["family_acc"].get(f, 0) or 0 for f in BLIND_SPOT_FAMILIES) / len(BLIND_SPOT_FAMILIES)
            pass2 = sum(result2["family_acc"].get(f, 0) or 0 for f in PASS_FAMILIES) / len(PASS_FAMILIES)
            sens_rows.append([knob, f"{pct:+.0%}", exp.r3(new_val), exp.r3(result2["metrics"]["accuracy"]),
                              exp.r3(blind2), exp.r3(pass2)])
    exp.write_csv(AGG_DIR / "ensemble_sensitivity.csv", sens_header, sens_rows)

    # config + trial log
    objectives = [l["objective"] for l in log]
    config_out = {
        "tuned_config": best_cfg.as_dict(),
        "knob_count_tunable": best_cfg.knob_count(),
        "knob_count_structural_ablation": 1,
        "tuning_seeds": tuning_seeds,
        "tuning_n_per_family": args.tuning_n,
        "reported_seed": args.reported_seed,
        "reported_n_per_family": args.reported_n,
        "trials": args.trials,
        "objective_best": max(objectives),
        "objective_worst": min(objectives),
        "objective_mean": sum(objectives) / len(objectives),
        "objective_std": (sum((o - sum(objectives) / len(objectives)) ** 2 for o in objectives) / len(objectives)) ** 0.5,
        "trial_log": log,
    }
    (AGG_DIR / "ensemble_config.json").write_text(json.dumps(config_out, indent=2))

    print(f"[ensemble] statistical acc={stat_metrics['accuracy']:.3f} rec={stat_metrics['recall']:.3f}")
    print(f"[ensemble] ensemble:tuned acc={ensemble_result['metrics']['accuracy']:.3f} "
          f"rec={ensemble_result['metrics']['recall']:.3f} prec={ensemble_result['metrics']['precision']:.3f}")
    print(f"[ensemble] ensemble:no_recovery_guard acc={no_guard_result['metrics']['accuracy']:.3f} "
          f"rec={no_guard_result['metrics']['recall']:.3f} prec={no_guard_result['metrics']['precision']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
