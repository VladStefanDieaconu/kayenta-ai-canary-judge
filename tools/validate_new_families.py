#!/usr/bin/env python3
"""Validate the generalisation-60 families before any model is asked to judge them.

A new scenario family is a claim about ground truth, and a claim about ground
truth has to be checked against something other than the intent of whoever wrote
the generator. This script checks three things that can be checked mechanically:

  1. Construction. Does each family actually have the shape it is supposed to
     have -- the residual elevation, the late onset, the identical marginals?
  2. The statistical judge. What does the genuine NetflixACAJudge do on them?
     The design intent is that it is blind to partial_recovery and to
     sustained_excursion, but the measured result is reported either way.
  3. The tuned ensemble, including its recovery guard. partial_recovery is the
     case that guard should *not* suppress: it is a degradation that improves
     and then stops improving. If the guard fires there it is a real defect in
     the guard and belongs in the paper.

Nothing here calls a model. Everything is deterministic and free to re-run.

Usage:
  python tools/validate_new_families.py [--n 20] [--seed 20260621] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import ensemble_judge as ej  # noqa: E402
import eval_dataset as ed  # noqa: E402
import results_schema  # noqa: E402
import run_ensemble_experiment as ens  # noqa: E402
import series_guards  # noqa: E402

TUNED_ENSEMBLE = REPO_ROOT / "judge-service" / "ensemble_tuned.json"
DATASET = "generalisation-60"


def tuned_config() -> ej.EnsembleConfig:
    """The published tuned ensemble, read rather than re-tuned.

    Re-tuning would fit the ensemble to the new families, and the question here
    is what the *published* ensemble does on scenarios it has never seen. The
    parameters were read from the tuning run's own artefact until that artefact
    was archived with the rest of the result data; they now ship as
    configuration, which is what they are.
    """
    cfg = json.loads(TUNED_ENSEMBLE.read_text())["tuned_config"]
    return ej.EnsembleConfig(**cfg)


def construction_checks(materialised: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-family measurements of the property each family is supposed to have."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for m in materialised:
        for _, d in m["series"].items():
            c, e = d["control"], d["experiment"]
            n = len(e)
            late = max(1, n // 4)
            early = max(1, n // 2)
            c_mean = st.mean(c)
            acc[m["family"]]["end_ratio"].append(st.mean(e[-late:]) / c_mean)
            acc[m["family"]]["early_ratio"].append(st.mean(e[:early]) / c_mean)
            acc[m["family"]]["whole_mean_ratio"].append(st.mean(e) / c_mean)
            acc[m["family"]]["sd_ratio"].append(st.pstdev(e) / st.pstdev(c) if st.pstdev(c) else float("nan"))
            acc[m["family"]]["max_ratio"].append(max(e) / max(c))
            acc[m["family"]]["identical_multiset"].append(
                float(sorted(round(v, 9) for v in c) == sorted(round(v, 9) for v in e))
            )
            # Longest run of consecutive samples above a threshold 15% over the
            # control's own median. The median is used rather than a percentile
            # because in sustained_excursion the control holds the same elevated
            # values as the canary -- a p90 threshold sits inside the excursion
            # band and reports a run length of 1 for both sides, which measures
            # the threshold rather than the series. This is the statistic the
            # family exists to separate: equal mass, different arrangement.
            thresh = st.median(c) * 1.15
            for label, series in (("ctl_longest_run", c), ("exp_longest_run", e)):
                best = run = 0
                for v in series:
                    run = run + 1 if v > thresh else 0
                    best = max(best, run)
                acc[m["family"]][label].append(float(best))
            acc[m["family"]]["ctl_samples_over"].append(float(sum(1 for v in c if v > thresh)))
            acc[m["family"]]["exp_samples_over"].append(float(sum(1 for v in e if v > thresh)))
    return {
        fam: {k: {"min": round(min(v), 4), "mean": round(sum(v) / len(v), 4), "max": round(max(v), 4)}
              for k, v in cols.items()}
        for fam, cols in acc.items()
    }


def occupancy_strips(materialised: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """One instance per family as a '#'/'.' strip above the same threshold.

    Cheap, exact, and pastes into a checkpoint. For sustained_excursion it shows
    the whole point of the family in two lines: the same number of marks on both
    rows, isolated on the control and gathered into runs on the canary.
    """
    out: Dict[str, Dict[str, str]] = {}
    for m in materialised:
        if m["family"] in out:
            continue
        name, d = next(iter(m["series"].items()))
        thresh = st.median(d["control"]) * 1.15
        out[m["family"]] = {
            "scenario": m["scenario"],
            "control": "".join("#" if v > thresh else "." for v in d["control"]),
            "experiment": "".join("#" if v > thresh else "." for v in d["experiment"]),
        }
    return out


def guard_report(materialised: List[Dict[str, Any]], cfg: ej.EnsembleConfig) -> Dict[str, Any]:
    """Whether the recovery guard fires, per scenario, with the ratios that decided it."""
    fired: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in materialised:
        for name, d in m["series"].items():
            rec, early, late = series_guards.detect_recovered_transient(d["control"], d["experiment"])
            fired[m["family"]].append({"scenario": m["scenario"], "metric": name,
                                       "recovered": rec, "early_ratio": round(early, 4),
                                       "late_ratio": round(late, 4)})
    return {fam: {"fired": sum(1 for r in rows if r["recovered"]), "n": len(rows),
                  "late_ratio_min": round(min(r["late_ratio"] for r in rows), 4),
                  "late_ratio_max": round(max(r["late_ratio"] for r in rows), 4),
                  "examples": rows[:3]}
            for fam, rows in fired.items()}


def per_family(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r["correct"])
    return {f: round(sum(v) / len(v), 3) for f, v in sorted(by.items(), key=lambda kv: ed.family_order(kv[0]))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the generalisation-60 families")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=ed.DEFAULT_MASTER_SEED)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "agg" / "new_families_validation.json"))
    ap.add_argument("--emit-long", action="store_true",
                    help="append statistical and ensemble rows to the long-format frame")
    args = ap.parse_args()

    print(f"[validate] materialising {DATASET} n={args.n} seed={args.seed} "
          f"(statistical judge runs once per scenario)")
    mat = ens.materialise(args.seed, args.n, DATASET)
    print(f"[validate] {len(mat)} scenarios")

    construction = construction_checks(mat)
    strips = occupancy_strips(mat)
    print("\n=== 1. construction ===")
    for fam, cols in construction.items():
        print(f"\n  {fam}")
        for k, v in cols.items():
            print(f"    {k:20s} min={v['min']:<10} mean={v['mean']:<10} max={v['max']}")

    print("\n  one instance per family, '#' = sample above 1.15x the control median:")
    for fam, s in strips.items():
        print(f"    {fam} ({s['scenario']})")
        print(f"      control    {s['control']}")
        print(f"      experiment {s['experiment']}")

    print("\n=== 2. statistical judge (genuine NetflixACAJudge) ===")
    stat_rows = [{"scenario": m["scenario"], "family": m["family"], "truth": m["truth"],
                  "verdict": m["stat_verdict"], "correct": int(m["stat_verdict"] == m["truth"])}
                 for m in mat]
    stat_fam = per_family(stat_rows)
    for fam, acc in stat_fam.items():
        n_fail = sum(1 for r in stat_rows if r["family"] == fam and r["verdict"] == "FAIL")
        print(f"  {fam:22s} accuracy={acc:.3f}  (FAIL on {n_fail}/{sum(1 for r in stat_rows if r['family']==fam)})")

    print("\n=== 3. tuned ensemble (published config, not re-tuned) ===")
    cfg = tuned_config()
    print(f"  config: {json.dumps(asdict(cfg))}")
    ens_result = ens.evaluate_config(mat, cfg)
    noguard = ens.evaluate_config(mat, replace(cfg, use_recovery_guard=False))
    print(f"  ensemble:tuned            per-family: {json.dumps(per_family(ens_result['rows']))}")
    print(f"  ensemble:no_recovery_guard per-family: {json.dumps(per_family(noguard['rows']))}")

    print("\n=== 4. recovery guard ===")
    guards = guard_report(mat, cfg)
    for fam, g in guards.items():
        verdict = "GUARD FIRES" if g["fired"] else "guard silent"
        print(f"  {fam:22s} {verdict}: {g['fired']}/{g['n']} metrics, "
              f"late_ratio in [{g['late_ratio_min']}, {g['late_ratio_max']}] (fires at <= 1.10)")

    report = {
        "dataset": DATASET, "n_per_family": args.n, "seed": args.seed,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ensemble_config": asdict(cfg),
        "construction": construction,
        "occupancy_strips": strips,
        "statistical_family_accuracy": stat_fam,
        "ensemble_tuned_family_accuracy": per_family(ens_result["rows"]),
        "ensemble_no_guard_family_accuracy": per_family(noguard["rows"]),
        "recovery_guard": guards,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[validate] report -> {out}")

    if args.emit_long:
        run_id = f"validation__{DATASET}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        long_rows = []
        seed_by_scenario = {m["scenario"]: m["seed"] for m in mat}
        for label, rows in (("statistical", stat_rows),
                            ("ensemble:tuned", ens_result["rows"]),
                            ("ensemble:no_recovery_guard", noguard["rows"])):
            judge, _, rep = label.partition(":")
            for r in rows:
                long_rows.append(results_schema.new_row(
                    run_id=run_id, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    dataset_id=DATASET, prompt_id="", prompt_hash="", model="-",
                    judge=judge, representation=rep, family=r["family"],
                    scenario_id=r["scenario"], seed=seed_by_scenario.get(r["scenario"], ""),
                    truth_label=r["truth"], verdict=r["verdict"], score="",
                    correct=r["correct"], latency_s="",
                ))
        path = results_schema.EXPERIMENTS_DIR / f"{run_id}.csv"
        results_schema.append(path, long_rows)
        print(f"[validate] {len(long_rows)} long-format rows -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
