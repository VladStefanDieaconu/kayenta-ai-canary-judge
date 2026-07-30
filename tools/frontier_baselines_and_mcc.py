#!/usr/bin/env python3
"""Trivial baselines plus MCC / balanced accuracy for every FRONTIER configuration.

Companion to tools/baselines_and_mcc.py, which reads results/summary.csv and
therefore covers only the local models. The frontier models are scored into
results/agg/frontier_long_<alias>.csv instead, so this script computes the
identical metrics over those files and emits one table in the same frame as
Table 14.

The metric formulas are imported from baselines_and_mcc.py rather than
reimplemented, so the two tables cannot drift apart.

Output: results/agg/frontier_scoreboard_with_baselines_and_mcc.csv
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_and_mcc import metrics_and_extras, r3, N_POS, N_NEG, N_TOTAL  # noqa: E402

AGG = REPO_ROOT / "results" / "agg"


def confusion(rows):
    tp = sum(1 for r in rows if r["truth"] == "FAIL" and r["verdict"] == "FAIL")
    fp = sum(1 for r in rows if r["truth"] == "PASS" and r["verdict"] == "FAIL")
    tn = sum(1 for r in rows if r["truth"] == "PASS" and r["verdict"] == "PASS")
    fn = sum(1 for r in rows if r["truth"] == "FAIL" and r["verdict"] == "PASS")
    return tp, fp, tn, fn


def main() -> int:
    header = ["judge", "model", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
              "balanced_accuracy", "mcc", "TP", "FP", "TN", "FN"]
    out = []
    for path in sorted(glob.glob(str(AGG / "frontier_long_*.csv"))):
        alias = Path(path).stem.replace("frontier_long_", "")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for judge in sorted({r["judge"] for r in rows}):
            if judge == "statistical":
                continue
            sub = [r for r in rows if r["judge"] == judge]
            tp, fp, tn, fn = confusion(sub)
            m = metrics_and_extras(tp, fp, tn, fn)
            out.append([judge, alias, len(sub),
                        r3(m["accuracy"]), r3(m["precision"]), r3(m["recall"]), r3(m["f1"]),
                        r3(m["fpr"]), r3(m["fnr"]), r3(m["balanced_accuracy"]), r3(m["mcc"]),
                        tp, fp, tn, fn])

    baselines = {"always_fail": (N_POS, N_NEG, 0.0, 0.0),
                 "always_pass": (0.0, 0.0, N_NEG, N_POS)}
    p = N_POS / N_TOTAL
    baselines["stratified_random"] = (N_POS * p, N_NEG * p, N_NEG * (1 - p), N_POS * (1 - p))
    for label, (tp, fp, tn, fn) in baselines.items():
        m = metrics_and_extras(tp, fp, tn, fn)
        out.append(["baseline", label, N_TOTAL,
                    r3(m["accuracy"]), r3(m["precision"]), r3(m["recall"]), r3(m["f1"]),
                    r3(m["fpr"]), r3(m["fnr"]), r3(m["balanced_accuracy"]), r3(m["mcc"]),
                    r3(tp), r3(fp), r3(tn), r3(fn)])

    dest = AGG / "frontier_scoreboard_with_baselines_and_mcc.csv"
    with dest.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(out)
    print(f"[frontier-baselines] wrote {len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
