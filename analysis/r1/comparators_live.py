#!/usr/bin/env python3
"""Model-free comparators that need Kayenta: the tuned ensemble and the guard signals.

For each dataset (original-180 as the reproduction check, then generalisation-60, kind-controls-180 and
heldout-benign-100), at master seed 20260621 with 20 windows per family:

* the tuned ensemble with its published configuration: judge-service/ensemble_tuned.json (tuned_config),
  tools/run_ensemble_experiment.materialise (one call of Kayenta's NetflixACAJudge per window, no model call) and
  evaluate_config, unchanged;
* the guard signals of the model-free guard rule: tools/run_fp_guard_experiment.build_guard_signals (recovered
  transient on any metric; equal-variance noise on single-metric windows; judge-service/app/series_guards.py
  default constants), unchanged. The guard rule itself (FAIL unless the statistical judge passes and a guard
  fires) is applied in analyze_r1.py with the statistical verdicts of the run records;
* the statistical verdict and score Kayenta returned inside materialise, kept only to check them against the run
  records;
* a check that tools/eval_dataset.py regenerates the protocol's series files value for value.

No constant is changed. Published counts on original-180: ensemble 102/120 detected and 9/60 false alarms.

Run (from the repository root, with the repository's host environment and the stack up):
  .venv/bin/python -I -B analysis/r1/comparators_live.py --out <new-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "judge-service" / "app"))

import eval_dataset as ed  # noqa: E402
import ensemble_judge as ej  # noqa: E402
import run_ensemble_experiment as rex  # noqa: E402
import run_fp_guard_experiment as fpg  # noqa: E402

MASTER_SEED = 20260621
N = 20
DATASETS = ["original-180", "generalisation-60", "kind-controls-180", "heldout-benign-100"]
COLUMNS = ["dataset_id", "scenario_id", "family", "truth", "metric_kinds", "n_metrics",
           "kayenta_statistical_verdict", "kayenta_statistical_score", "ensemble_verdict",
           "guard_recovered_transient", "guard_equal_variance_noise"]


def series_check(ds: str) -> dict:
    """tools/eval_dataset.py against protocol/series_<ds>.jsonl, value for value."""
    path = REPO / "protocol" / f"series_{ds}.jsonl"
    want = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            o = json.loads(ln)
            want[o["id"]] = o
    same = 0
    diff = []
    scs = ed.build_scenarios(MASTER_SEED, N, ds)
    for sc in scs:
        ser = ed.series_for_scenario(sc)
        o = want.get(sc.id)
        ok = o is not None and o["family"] == sc.family and o["truth"] == sc.truth and len(o["metrics"]) == len(sc.metrics)
        if ok:
            for m, om in zip(sc.metrics, o["metrics"]):
                ok = ok and m.name == om["name"] and ser[m.name]["control"] == om["control"] \
                    and ser[m.name]["experiment"] == om["experiment"]
        same += int(ok)
        if not ok:
            diff.append(sc.id)
    return {"file": f"protocol/series_{ds}.jsonl", "windows_generated": len(scs), "windows_in_file": len(want),
            "identical": same, "different": diff}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="new output folder (files are never overwritten)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tuned = json.loads((REPO / "judge-service" / "ensemble_tuned.json").read_text(encoding="utf-8"))["tuned_config"]
    cfg = ej.EnsembleConfig(**tuned)
    summary = {"ensemble_config": tuned, "ensemble_config_file": "judge-service/ensemble_tuned.json",
               "master_seed": MASTER_SEED, "n_per_family": N, "datasets": {}}
    for ds in a.datasets:
        target = out / f"comparators_live_{ds}.csv"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target.name}")
        mat = rex.materialise(MASTER_SEED, N, ds)
        res = rex.evaluate_config(mat, cfg)
        ens = {r["scenario"]: r["verdict"] for r in res["rows"]}
        sig = fpg.build_guard_signals(MASTER_SEED, N, ds)
        kinds = {sc.id: [m.kind for m in sc.metrics] for sc in ed.build_scenarios(MASTER_SEED, N, ds)}
        rows = []
        for m in mat:
            s = m["scenario"]
            rows.append({"dataset_id": ds, "scenario_id": s, "family": m["family"], "truth": m["truth"],
                         "metric_kinds": "+".join(kinds[s]), "n_metrics": len(kinds[s]),
                         "kayenta_statistical_verdict": m["stat_verdict"], "kayenta_statistical_score": m["stat_score"],
                         "ensemble_verdict": ens[s],
                         "guard_recovered_transient": int(bool(sig[s]["recovered_transient"])),
                         "guard_equal_variance_noise": int(bool(sig[s]["equal_variance_noise"]))})
        import csv
        with open(target, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        det = sum(1 for r in rows if r["truth"] == "FAIL" and r["ensemble_verdict"] == "FAIL")
        fa = sum(1 for r in rows if r["truth"] == "PASS" and r["ensemble_verdict"] == "FAIL")
        n_reg = sum(1 for r in rows if r["truth"] == "FAIL")
        n_ben = sum(1 for r in rows if r["truth"] == "PASS")
        d = {"windows": len(rows), "ensemble_detected": f"{det}/{n_reg}", "ensemble_false_alarms": f"{fa}/{n_ben}",
             "kayenta_statistical_FAIL": sum(1 for r in rows if r["kayenta_statistical_verdict"] == "FAIL"),
             "guard_recovered_transient_fires": sum(r["guard_recovered_transient"] for r in rows),
             "guard_equal_variance_noise_fires": sum(r["guard_equal_variance_noise"] for r in rows)}
        if ds == "original-180":
            d["published"] = {"ensemble_detected": "102/120", "ensemble_false_alarms": "9/60"}
            d["REPRODUCED"] = d["ensemble_detected"] == "102/120" and d["ensemble_false_alarms"] == "9/60"
        if ds in ("kind-controls-180", "heldout-benign-100"):
            d["series_check"] = series_check(ds)
        summary["datasets"][ds] = d
        print(ds, json.dumps(d))
    target = out / "comparators_live_summary.json"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target.name}")
    target.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
