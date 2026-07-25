#!/usr/bin/env python3
"""Bounded confidence intervals for the judge-comparison metrics.

Computes confidence intervals directly from results produced by
tools/run_experiment.py and tools/run_experiment_multiseed.py
(results/summary.csv, results/family_matrix.csv, results/results_raw.csv,
results/agg/long_results.csv); it doesn't call any model.

Rationale: a plain Wald t-interval over the 5 per-seed rates
(tools/run_experiment_multiseed.py::mean_ci95) can fall outside [0, 1] on a
small-n rate near 0 or 1 (e.g. tail_regression [-0.071, 0.151],
subtle_regression [0.849, 1.071]), which is meaningless for a rate. This
script instead uses a Wilson score interval on pooled counts (primary) plus a
seed-level percentile bootstrap (cross-check), both bounded to [0, 1] by
construction, and writes a third table, headline_metrics_with_ci.csv,
applying the same bounded methods to the single-seed N=20 (180-scenario) run.

As a correctness check, the statistical judge's headline accuracy is
recomputed from results/summary.csv's own TP/FP/TN/FN and compared against
the recorded value (0.544).

Outputs (results/agg/), all bounded to [0, 1] by construction:
  - metrics_with_ci.csv          per judge/model: Wilson CI on each metric,
                                  a bootstrap cross-check, fpr/fnr CIs, and an
                                  identical_across_seeds flag.
  - family_recall_with_ci.csv    per judge/model/family: same treatment.
  - headline_metrics_with_ci.csv long-format (one row per judge/model/
                                  family-or-overall/metric) CI table for the
                                  single-seed N=20 run, row-bootstrap cross-check.

Usage:
  python tools/bounded_confidence_intervals.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_utils as ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
AGG_DIR = RESULTS_DIR / "agg"
FAMILIES = [
    "no_change", "noise_equivalent", "healed_transient", "clean_mean_shift",
    "variance_increase", "tail_regression", "gradual_drift",
    "cross_metric_marginal", "subtle_regression",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# Part A: multi-seed re-aggregation from results/agg/long_results.csv
# (fixes metrics_with_ci.csv and family_recall_with_ci.csv)
def part_a_multiseed() -> Tuple[int, int]:
    rows = read_csv(AGG_DIR / "long_results.csv")
    for r in rows:
        r["seed"] = int(r["seed"])
        r["correct"] = int(r["correct"])

    seeds = sorted(set(r["seed"] for r in rows))
    n_seeds = len(seeds)

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[(r["judge"], r["model"])].append(r)

    order = sorted(by_group.keys(), key=lambda k: (
        0 if k[0] == "statistical" else (1 if k[0].startswith("ai:") else 2), k[0], k[1]))

    # metrics_with_ci.csv
    metrics = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
    header = ["judge", "model", "n_seeds", "n_pooled"]
    for m in metrics:
        header += [f"{m}_mean", f"{m}_ci_low", f"{m}_ci_high"]
    for m in metrics:
        header += [f"{m}_boot_low", f"{m}_boot_high"]
    header += ["identical_across_seeds"]

    ci_rows = []
    for key in order:
        grp = by_group[key]
        c_pooled = ci.confusion_counts(grp)
        pooled_metrics = ci.metrics_from_conf(c_pooled)
        n_pooled = len(grp)

        by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in grp:
            by_seed[r["seed"]].append(r)
        per_seed_metric_vals: Dict[str, List[float]] = defaultdict(list)
        for seed in seeds:
            srows = by_seed.get(seed)
            if not srows:
                continue
            c = ci.confusion_counts(srows)
            m = ci.metrics_from_conf(c)
            for k, v in m.items():
                per_seed_metric_vals[k].append(v)

        row = [key[0], key[1], n_seeds, n_pooled]
        identical: List[str] = []
        wilson_by_metric = {
            "accuracy": ci.wilson_interval(c_pooled["TP"] + c_pooled["TN"],
                                           sum(c_pooled.values())),
            "precision": ci.wilson_interval(c_pooled["TP"], c_pooled["TP"] + c_pooled["FP"]),
            "recall": ci.wilson_interval(c_pooled["TP"], c_pooled["TP"] + c_pooled["FN"]),
            "fpr": ci.wilson_interval(c_pooled["FP"], c_pooled["FP"] + c_pooled["TN"]),
            "fnr": ci.wilson_interval(c_pooled["FN"], c_pooled["FN"] + c_pooled["TP"]),
        }
        for m in metrics:
            mean_v = pooled_metrics[m]
            if m == "f1":
                row += [ci.r3(mean_v), "", ""]  # no closed-form binomial CI for F1; see boot columns
            else:
                lo, hi = wilson_by_metric[m]
                row += [ci.r3(mean_v), ci.r3(lo), ci.r3(hi)]
        for m in metrics:
            vals = per_seed_metric_vals[m]
            zero_var = ci.is_zero_variance(vals)
            if zero_var:
                lo, hi = (vals[0], vals[0])
                identical.append(m)
            else:
                lo, hi = ci.percentile_bootstrap(vals, lambda rs: sum(rs) / len(rs))
            row += [ci.r3(lo), ci.r3(hi)]
        row.append(";".join(identical))
        ci_rows.append(row)
    write_csv(AGG_DIR / "metrics_with_ci.csv", header, ci_rows)

    # family_recall_with_ci.csv
    fam_header = ["judge", "model", "family", "n_seeds", "n_pooled", "accuracy_mean",
                  "ci_low", "ci_high", "boot_low", "boot_high", "identical_across_seeds"]
    fam_rows = []
    for key in order:
        grp = by_group[key]
        by_fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in grp:
            by_fam[r["family"]].append(r)
        for fam in FAMILIES:
            frows = by_fam.get(fam)
            if not frows:
                continue
            n_pooled = len(frows)
            k_pooled = sum(r["correct"] for r in frows)
            rate_pooled = k_pooled / n_pooled
            wlo, whi = ci.wilson_interval(k_pooled, n_pooled)

            by_seed_rates: List[float] = []
            by_seed: Dict[int, List[int]] = defaultdict(list)
            for r in frows:
                by_seed[r["seed"]].append(r["correct"])
            for seed in seeds:
                vals = by_seed.get(seed)
                if vals:
                    by_seed_rates.append(sum(vals) / len(vals))

            zero_var = ci.is_zero_variance(by_seed_rates)
            if zero_var:
                blo, bhi = (by_seed_rates[0], by_seed_rates[0])
            else:
                blo, bhi = ci.percentile_bootstrap(by_seed_rates, lambda rs: sum(rs) / len(rs))

            fam_rows.append([key[0], key[1], fam, len(by_seed_rates), n_pooled,
                             ci.r3(rate_pooled), ci.r3(wlo), ci.r3(whi),
                             ci.r3(blo), ci.r3(bhi), "yes" if zero_var else ""])
    write_csv(AGG_DIR / "family_recall_with_ci.csv", fam_header, fam_rows)

    return len(ci_rows), len(fam_rows)


# Part B: headline N=20 (single-seed, 180-scenario) bounded CIs, from
# results/summary.csv (TP/FP/TN/FN), results/family_matrix.csv, and
# results/results_raw.csv (row-level bootstrap population).
def part_b_headline() -> int:
    summary = read_csv(RESULTS_DIR / "summary.csv")
    family_matrix = read_csv(RESULTS_DIR / "family_matrix.csv")
    raw = read_csv(RESULTS_DIR / "results_raw.csv")

    # Correctness check: accuracy recomputed from the raw counts should match
    # the recorded value exactly.
    stat_row = next(r for r in summary if r["judge"] == "statistical")
    tp, fp, tn, fn = (int(stat_row[k]) for k in ("TP", "FP", "TN", "FN"))
    recomputed_acc = ci.r3((tp + tn) / (tp + fp + tn + fn))
    recorded_acc = ci.r3(float(stat_row["accuracy"]))
    assert recomputed_acc == recorded_acc == 0.544, (
        f"accuracy mismatch: recomputed {recomputed_acc} vs recorded {recorded_acc}")

    raw_by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in raw:
        raw_by_key[(r["judge"], r["model"])].append(r)

    header = ["judge", "model", "family", "metric", "k", "n", "rate",
              "wilson_low", "wilson_high", "boot_low", "boot_high"]
    out_rows = []

    def _metric_of_resample(resample, metric_name: str) -> float:
        cc = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for t, p in resample:
            if t and p:
                cc["TP"] += 1
            elif (not t) and p:
                cc["FP"] += 1
            elif (not t) and (not p):
                cc["TN"] += 1
            else:
                cc["FN"] += 1
        return ci.metrics_from_conf(cc)[metric_name]

    def add_overall(judge: str, model: str, row: Dict[str, str]) -> None:
        tp_, fp_, tn_, fn_ = (int(row[k]) for k in ("TP", "FP", "TN", "FN"))
        c = {"TP": tp_, "FP": fp_, "TN": tn_, "FN": fn_}
        n_tot = tp_ + fp_ + tn_ + fn_
        srows = raw_by_key.get((judge, model), [])
        pos_rows = [1 if r["truth"] == "FAIL" else 0 for r in srows]
        pred_rows = [1 if r["verdict"] == "FAIL" else 0 for r in srows]
        paired = list(zip(pos_rows, pred_rows))

        m = ci.metrics_from_conf(c)
        specs = [
            ("accuracy", tp_ + tn_, n_tot),
            ("precision", tp_, tp_ + fp_),
            ("recall", tp_, tp_ + fn_),
            ("fpr", fp_, fp_ + tn_),
            ("fnr", fn_, fn_ + tp_),
        ]
        for name, k_val, n_val in specs:
            wlo, whi = ci.wilson_interval(k_val, n_val)
            if paired:
                blo, bhi = ci.percentile_bootstrap(
                    paired, lambda rs, _n=name: _metric_of_resample(rs, _n))
            else:
                blo, bhi = (wlo, whi)
            out_rows.append([judge, model, "", name, k_val, n_val, ci.r3(m[name]),
                             ci.r3(wlo), ci.r3(whi), ci.r3(blo), ci.r3(bhi)])

        if paired:
            blo, bhi = ci.percentile_bootstrap(
                paired, lambda rs: _metric_of_resample(rs, "f1"))
        else:
            blo, bhi = (0.0, 0.0)
        out_rows.append([judge, model, "", "f1", "", "", ci.r3(m["f1"]),
                         "", "", ci.r3(blo), ci.r3(bhi)])

    for row in summary:
        add_overall(row["judge"], row["model"], row)

    for row in family_matrix:
        judge, model = row["judge"], row["model"]
        for fam in FAMILIES:
            rate = float(row[fam])
            k_val = round(rate * 20)
            wlo, whi = ci.wilson_interval(k_val, 20)
            out_rows.append([judge, model, fam, "accuracy", k_val, 20, ci.r3(rate),
                             ci.r3(wlo), ci.r3(whi), "", ""])

    write_csv(AGG_DIR / "headline_metrics_with_ci.csv", header, out_rows)
    return len(out_rows)


def main() -> int:
    n_ci, n_fam = part_a_multiseed()
    n_head = part_b_headline()
    print(f"[bounded_confidence_intervals] metrics_with_ci.csv: {n_ci} rows")
    print(f"[bounded_confidence_intervals] family_recall_with_ci.csv: {n_fam} rows")
    print(f"[bounded_confidence_intervals] headline_metrics_with_ci.csv: {n_head} rows")
    print("[bounded_confidence_intervals] statistical headline accuracy "
          "check passed (0.544)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
