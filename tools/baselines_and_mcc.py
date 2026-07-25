#!/usr/bin/env python3
"""Trivial baselines plus an imbalance-robust metric for the headline scoreboard.

Reads results/summary.csv (the published headline scoreboard, N=180,
unmodified) and adds:
  - three trivial-baseline rows computed directly from the dataset's known
    class balance (120 FAIL / 60 PASS), with no model calls and nothing re-run:
      * always-FAIL: predicts FAIL for every scenario.
      * always-PASS: predicts PASS for every scenario.
      * stratified-random: guesses FAIL independently per scenario with
        probability equal to the observed FAIL prevalence (120/180), i.e. a
        label-independent guess that matches the class prior. Reported as the
        closed-form expectation over that random rule (not one noisy simulated
        draw), so the numbers are exact and reproducible:
        E[TP]=n_pos*p, E[FP]=n_neg*p, E[TN]=n_neg*(1-p), E[FN]=n_pos*(1-p).
  - MCC (Matthews correlation coefficient) and balanced accuracy
    ((recall + specificity) / 2) columns for every row, including the new
    baselines. Both behave sensibly under the dataset's 120/60 class imbalance,
    unlike raw accuracy.

As a correctness check, always-FAIL's computed accuracy should equal 0.667
(120/180), which is above the statistical judge's accuracy (0.544). Raw accuracy
alone can make a real judge look worse than a baseline that always guesses the
majority class, which is why MCC and balanced accuracy matter here.

Output: results/agg/scoreboard_with_baselines_and_mcc.csv.

Usage:
  python tools/baselines_and_mcc.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
AGG_DIR = RESULTS_DIR / "agg"

N_POS = 120  # FAIL scenarios (6 FAIL families x 20)
N_NEG = 60   # PASS scenarios (3 PASS families x 20)
N_TOTAL = N_POS + N_NEG


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def r3(x) -> float:
    return round(float(x), 3)


def metrics_and_extras(tp: float, fp: float, tn: float, fn: float) -> Dict[str, float]:
    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = (rec + specificity) / 2
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "fnr": fnr, "balanced_accuracy": balanced_acc, "mcc": mcc}


def main() -> int:
    summary = read_csv(RESULTS_DIR / "summary.csv")

    header = ["judge", "model", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
              "balanced_accuracy", "mcc", "TP", "FP", "TN", "FN", "avg_latency"]
    out_rows = []

    for row in summary:
        tp, fp, tn, fn = (float(row[k]) for k in ("TP", "FP", "TN", "FN"))
        m = metrics_and_extras(tp, fp, tn, fn)
        out_rows.append([row["judge"], row["model"], row["n"],
                         r3(m["accuracy"]), r3(m["precision"]), r3(m["recall"]), r3(m["f1"]),
                         r3(m["fpr"]), r3(m["fnr"]), r3(m["balanced_accuracy"]), r3(m["mcc"]),
                         row["TP"], row["FP"], row["TN"], row["FN"], row["avg_latency"]])

    # trivial baselines
    baselines = {
        "always_fail": (N_POS, N_NEG, 0.0, 0.0),
        "always_pass": (0.0, 0.0, N_NEG, N_POS),
    }
    p = N_POS / N_TOTAL
    baselines["stratified_random"] = (N_POS * p, N_NEG * p, N_NEG * (1 - p), N_POS * (1 - p))

    for label, (tp, fp, tn, fn) in baselines.items():
        m = metrics_and_extras(tp, fp, tn, fn)
        out_rows.append(["baseline", label, N_TOTAL,
                         r3(m["accuracy"]), r3(m["precision"]), r3(m["recall"]), r3(m["f1"]),
                         r3(m["fpr"]), r3(m["fnr"]), r3(m["balanced_accuracy"]), r3(m["mcc"]),
                         r3(tp), r3(fp), r3(tn), r3(fn), 0.0])

    always_fail_acc = r3(metrics_and_extras(*baselines["always_fail"])["accuracy"])
    assert always_fail_acc == 0.667, f"always-FAIL accuracy = {always_fail_acc}, expected 0.667"

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(AGG_DIR / "scoreboard_with_baselines_and_mcc.csv", header, out_rows)

    stat_acc = next(r for r in out_rows if r[0] == "statistical")[3]
    print(f"[baselines] always-FAIL accuracy = {always_fail_acc} (statistical judge: {stat_acc})")
    print(f"[baselines] wrote {len(out_rows)} rows -> "
          f"{AGG_DIR / 'scoreboard_with_baselines_and_mcc.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
