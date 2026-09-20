#!/usr/bin/env python3
"""Trivial baselines plus an imbalance-robust metric for the headline scoreboard.

Reads <results>/summary.csv (the headline scoreboard, unmodified) and
<results>/results_raw.csv (for the class balance) and adds:
  - three trivial-baseline rows computed directly from the class balance of the
    run being scored, with no model calls and nothing re-run:
      * always-FAIL: predicts FAIL for every scenario.
      * always-PASS: predicts PASS for every scenario.
      * stratified-random: guesses FAIL independently per scenario with
        probability equal to the observed FAIL prevalence, i.e. a
        label-independent guess that matches the class prior. Reported as the
        closed-form expectation over that random rule (not one noisy simulated
        draw), so the numbers are exact and reproducible:
        E[TP]=n_pos*p, E[FP]=n_neg*p, E[TN]=n_neg*(1-p), E[FN]=n_pos*(1-p).
  - MCC (Matthews correlation coefficient) and balanced accuracy
    ((recall + specificity) / 2) columns for every row, including the new
    baselines. Both behave sensibly under class imbalance, unlike raw accuracy.

The class balance is counted, per run, from the distinct scenarios in
<results>/results_raw.csv, ignoring rows with no label. It used to be the
constants 120 FAIL / 60 PASS, which are the article's dataset: correct for the
seeded run (`make experiment` produces exactly 120/60) and wrong for every other
run the `--results` flag exists to score. On an own-data run of 4 FAIL and 3
PASS windows that printed an always-FAIL accuracy of 0.667 where the truth is
4/7 = 0.571, in the user's own scoreboard, in the same columns as their real
judge rows. A baseline is a property of the labels of the run it sits beside.

As a correctness check, the labelled windows are counted a second time by a
different route -- out of the TP/FN and TN/FP of the scoreboard beside the raw
rows -- and a scoreboard that scored more labelled windows than the raw rows
hold is refused, because then the two files are not the same run.

always-FAIL is the number a real judge has to beat: raw accuracy alone can make a
real judge look worse than a rule that always guesses the majority class, which
is why MCC and balanced accuracy are here too. For the published run it is 0.667
(120/180) against the statistical judge's 0.544.

A run in which nothing is labelled has no class balance and no accuracy to
compare one against. It is reported and skipped, leaving exit 0 for the target
that runs this first: running without labels is a documented mode, not an error.

Output: <results>/agg/scoreboard_with_baselines_and_mcc.csv.

Usage:
  python tools/baselines_and_mcc.py [--results DIR]
"""

from __future__ import annotations

import argparse

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
AGG_DIR = RESULTS_DIR / "agg"


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


def class_balance(raw_path: Path) -> Tuple[int, int]:
    """(n_pos, n_neg): distinct FAIL and PASS scenarios in one run's raw rows.

    Counted over scenarios, not rows: results_raw.csv carries one row per
    (scenario, judge, model), so counting rows would multiply the balance by the
    number of configurations and, worse, weight it by which models happened to be
    present. A row whose truth is empty carries no label -- unlabelled windows
    are a supported mode (docs/own-data.md) -- and is not a negative, so it is
    counted in neither class and reported separately by the caller.
    """
    fail, passed, unlabelled = set(), set(), set()
    for row in read_csv(raw_path):
        truth = (row.get("truth") or "").strip()
        scenario = row.get("scenario", "")
        if truth == "FAIL":
            fail.add(scenario)
        elif truth == "PASS":
            passed.add(scenario)
        else:
            unlabelled.add(scenario)
    overlap = fail & passed
    if overlap:
        raise ValueError(f"{raw_path}: {len(overlap)} scenario(s) carry both labels, "
                         f"e.g. {sorted(overlap)[0]!r}; the class balance is not defined")
    if unlabelled - fail - passed:
        print(f"[baselines] {len(unlabelled - fail - passed)} unlabelled scenario(s) are in "
              f"neither class; the baselines describe the labelled windows only")
    return len(fail), len(passed)


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
    # --results lets the same table be built from a corrected or alternative run
    # without editing the module. The default is unchanged, so an existing
    # invocation behaves exactly as before.
    ap = argparse.ArgumentParser(description="Scoreboard with trivial baselines and MCC")
    ap.add_argument("--results", default=str(RESULTS_DIR),
                    help="directory holding summary.csv; output goes to <results>/agg/")
    args = ap.parse_args()
    results_dir = Path(args.results)
    agg_dir = results_dir / "agg"

    raw_path = results_dir / "results_raw.csv"
    if not raw_path.is_file():
        print(f"[baselines] {raw_path} not found. The three baseline rows are computed from "
              f"the class balance of the run being scored, and that balance is only in the "
              f"raw rows; summary.csv does not carry it. Point --results at a directory a "
              f"run wrote, or run `make experiment` first.", file=sys.stderr)
        return 1
    n_pos, n_neg = class_balance(raw_path)
    n_total = n_pos + n_neg
    if n_total == 0:
        # Skipped, not failed. Running without labels is a documented and invited
        # mode (docs/own-data.md, "Running without labels"), and this script is
        # first in the Makefile's ANALYSIS_SCRIPTS: returning 1 here stopped
        # `make analysis` on a `|| exit 1` before tools/roc_pr_curves.py -- which
        # handles the same case gracefully -- had run at all, so a documented way
        # to use the harness reported a failed target. A genuine error still
        # exits 1; an absent class is not one.
        print(f"[baselines] no labelled scenario in {raw_path}; there is no class balance to "
              f"compute a baseline from and no accuracy to compare one against. Skipped: "
              f"nothing written to {agg_dir}. Label the windows you remember and run this "
              f"again; see docs/own-data.md.", file=sys.stderr)
        return 0

    summary = read_csv(results_dir / "summary.csv")

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

    # trivial baselines, from this run's own class balance
    baselines = {
        "always_fail": (n_pos, n_neg, 0.0, 0.0),
        "always_pass": (0.0, 0.0, n_neg, n_pos),
    }
    p = n_pos / n_total
    baselines["stratified_random"] = (n_pos * p, n_neg * p, n_neg * (1 - p), n_pos * (1 - p))

    for label, (tp, fp, tn, fn) in baselines.items():
        m = metrics_and_extras(tp, fp, tn, fn)
        out_rows.append(["baseline", label, n_total,
                         r3(m["accuracy"]), r3(m["precision"]), r3(m["recall"]), r3(m["f1"]),
                         r3(m["fpr"]), r3(m["fnr"]), r3(m["balanced_accuracy"]), r3(m["mcc"]),
                         r3(tp), r3(fp), r3(tn), r3(fn), 0.0])

    # A real check, and it took two goes to make it one. The original was
    # `== 0.667`, the article's own class balance asserted against every run.
    # Its replacement compared always-FAIL's accuracy with the FAIL rate -- but
    # always-FAIL is right on exactly the FAIL windows, so its accuracy IS
    # n_pos/n_total by construction, computed from the same two numbers through
    # the same function. That comparison cannot fail, whatever is wrong.
    #
    # This counts the labelled windows a second time by a different route: from
    # the scoreboard sitting beside the raw rows. A (judge, model) row's TP + FN
    # is how many labelled FAIL windows that configuration scored, and TN + FP
    # how many PASS. Calls excluded as failures (errors.csv) can make a row's
    # counts lower than the run's; nothing can make them higher. A row that
    # scored more FAIL windows than results_raw.csv contains means summary.csv
    # and results_raw.csv are not the same run -- a scoreboard left behind by an
    # earlier, larger run beside a fresh one, which is exactly how the study's
    # 120/60 ends up over somebody else's seven windows.
    scored_pos = scored_neg = 0
    for row in summary:
        try:
            tp, fp, tn, fn = (float(row[k]) for k in ("TP", "FP", "TN", "FN"))
        except (KeyError, TypeError, ValueError):
            continue
        scored_pos = max(scored_pos, int(tp + fn))
        scored_neg = max(scored_neg, int(tn + fp))
    if scored_pos > n_pos or scored_neg > n_neg:
        print(f"[baselines] {results_dir / 'summary.csv'} scores up to {scored_pos} FAIL and "
              f"{scored_neg} PASS window(s), but {raw_path} holds {n_pos} FAIL and {n_neg} "
              f"PASS. The scoreboard and the raw rows are not the same run, so a baseline "
              f"computed from one would sit in a table built from the other.", file=sys.stderr)
        return 1
    shortfall = (n_pos - scored_pos) + (n_neg - scored_neg)
    if shortfall:
        print(f"[baselines] the best-covered configuration scored {scored_pos}/{n_pos} FAIL "
              f"and {scored_neg}/{n_neg} PASS window(s); {shortfall} labelled window(s) were "
              f"scored by no configuration (see errors.csv)")

    always_fail_acc = r3(metrics_and_extras(*baselines["always_fail"])["accuracy"])

    agg_dir.mkdir(parents=True, exist_ok=True)
    write_csv(agg_dir / "scoreboard_with_baselines_and_mcc.csv", header, out_rows)

    stat_acc = next(r for r in out_rows if r[0] == "statistical")[3]
    print(f"[baselines] class balance from {raw_path}: {n_pos} FAIL / {n_neg} PASS "
          f"(n={n_total})")
    print(f"[baselines] always-FAIL accuracy = {always_fail_acc} (statistical judge: {stat_acc})")
    print(f"[baselines] wrote {len(out_rows)} rows -> "
          f"{agg_dir / 'scoreboard_with_baselines_and_mcc.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
