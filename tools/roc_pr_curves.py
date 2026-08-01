#!/usr/bin/env python3
"""Threshold-independent evaluation via ROC and PR curves.

Every judge here emits a 0-100 score (higher = healthier / more PASS-like),
recorded per scenario in results/results_raw.csv (N=20 run) and
results/agg/frontier_long_claude-opus-4-8-vlm.csv (frontier run). This
sweeps a threshold over that score (independent of the fixed pass=75/
marginal=50 operating point used everywhere else) to build ROC and
PR curves and their AUCs, for:
  - the statistical judge (model-independent),
  - the best local single config, ai:summary/phi4-llm,
  - hybrid:gated/phi4-llm (also emits a score),
  - the frontier model's best representation, ai:raw/claude-opus-4-8-vlm,
  - hybrid:gated/claude-opus-4-8-vlm.

Positive class = FAIL, so a scenario is scored on fail_score = 100 - score
(higher fail_score means the judge thinks it's more likely a FAIL). The
threshold is swept over every distinct fail_score value observed, plus the two
sentinel operating points "predict nothing" and "predict everything".

Two precision-recall summaries are reported, because they disagree:
  - average_precision, the stepwise integral sum (R_n - R_{n-1}) * P_n over the
    threshold sweep. This is the primary summary.
  - auc_pr, the trapezoidal area, which interpolates linearly between attained
    operating points and so credits a judge with points it has no threshold to
    reach. The two diverge most for a judge emitting few distinct scores, which
    is the incumbent statistical judge (2 distinct scores here), so reporting
    the trapezoid alone biases the comparison in its favour.

ROC area is trapezoidal in both cases (FPR vs TPR), where interpolation is
the standard convention and the two definitions coincide.

Any judge with no per-scenario score (verdict-only) is skipped with a note
in the output; every judge evaluated here does emit a score, so the check
exists but never trips.

Outputs (results/agg/):
  - roc_pr_auc.csv           one row per (judge, model): n, auc_roc, auc_pr,
                             average_precision, note.
  - roc_pr_curve_points.csv  long-format curve points (source, curve, threshold, x, y).

Usage:
  python tools/roc_pr_curves.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def sources_for(results_dir: Path) -> List[Dict[str, Any]]:
    """The five configurations, resolved against one results directory.

    Taking the directory as an argument rather than a module constant is what
    lets this run against a results set other than the repository's own, which
    is the whole point for anyone bringing their own models. The frontier file
    is named rather than globbed: a glob over agg/ matches the defect re-runs
    and the reproduction checks too, and those are not this configuration.
    """
    agg = results_dir / "agg"
    raw = results_dir / "results_raw.csv"
    frontier = agg / "frontier_long_claude-opus-4-8-vlm.csv"
    return [
        {"name": "statistical", "path": raw, "judge": "statistical", "model": "-"},
        {"name": "ai:summary/phi4-llm (best local)", "path": raw,
         "judge": "ai:summary", "model": "phi4-llm"},
        {"name": "hybrid:gated/phi4-llm", "path": raw,
         "judge": "hybrid:gated", "model": "phi4-llm"},
        {"name": "ai:raw/claude-opus-4-8-vlm (frontier best rep)",
         "path": frontier, "judge": "ai:raw", "model": "claude-opus-4-8-vlm"},
        {"name": "hybrid:gated/claude-opus-4-8-vlm",
         "path": frontier, "judge": "hybrid:gated", "model": "claude-opus-4-8-vlm"},
    ]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def r4(x) -> float:
    return round(float(x), 4)


def confusion_at(pairs: List[Tuple[bool, float]], threshold: float) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for truth_fail, fail_score in pairs:
        pred_fail = fail_score >= threshold
        if truth_fail and pred_fail:
            tp += 1
        elif (not truth_fail) and pred_fail:
            fp += 1
        elif (not truth_fail) and (not pred_fail):
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def roc_pr_points(pairs: List[Tuple[bool, float]]) -> List[Dict[str, float]]:
    fail_scores = sorted(set(fs for _, fs in pairs))
    thresholds = [fail_scores[-1] + 1.0] + list(reversed(fail_scores)) + [fail_scores[0] - 1.0]
    pts = []
    for t in thresholds:
        tp, fp, tn, fn = confusion_at(pairs, t)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0  # convention: no positives predicted -> precision undefined, reported as 1.0
        pts.append({"threshold": t, "fpr": fpr, "tpr": tpr, "precision": precision, "recall": tpr})
    return pts


def trapz_auc(xs: List[float], ys: List[float]) -> float:
    pts = sorted(zip(xs, ys))
    auc = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        auc += (x1 - x0) * (y0 + y1) / 2
    return auc


def average_precision(pts: List[Dict[str, float]]) -> float:
    """Stepwise average precision: sum over the sweep of (R_n - R_{n-1}) * P_n.

    Takes the points in the order roc_pr_points() emits them (threshold high to
    low, so recall is non-decreasing) rather than a copy re-sorted by recall.
    Precision is not monotone in recall, so re-sorting first pairs a recall
    increment with a different threshold's precision and shifts the result.
    Unlike trapz_auc this credits only the operating points a threshold can
    actually reach; no interpolation.
    """
    ap = 0.0
    prev_recall = 0.0
    for p in pts:
        if p["recall"] > prev_recall:
            ap += (p["recall"] - prev_recall) * p["precision"]
            prev_recall = p["recall"]
    return ap


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--results", default=str(REPO_ROOT / "results"),
                     help="directory holding results_raw.csv and agg/")
    ap_.add_argument("--out", default=None,
                     help="output directory (default: <results>/agg)")
    args = ap_.parse_args()
    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_header = ["judge", "model", "n", "auc_roc", "auc_pr", "average_precision", "note"]
    curve_header = ["source", "judge", "model", "curve", "threshold", "x", "y"]
    auc_rows = []
    curve_rows = []

    for src in sources_for(results_dir):
        if not src["path"].exists():
            auc_rows.append([src["judge"], src["model"], 0, "", "", "",
                             f"source file not found: {src['path'].name}"])
            continue
        rows = read_csv(src["path"])
        sub = [r for r in rows if r["judge"] == src["judge"] and r["model"] == src["model"]]
        if not sub:
            auc_rows.append([src["judge"], src["model"], 0, "", "", "", "no rows for this (judge, model)"])
            continue
        missing_score = [r for r in sub if r.get("score") in (None, "", "nan")]
        if missing_score:
            auc_rows.append([src["judge"], src["model"], len(sub), "", "", "",
                             f"skipped: {len(missing_score)}/{len(sub)} rows have no score (verdict-only)"])
            continue

        pairs = [(r["truth"] == "FAIL", 100.0 - float(r["score"])) for r in sub]
        pts = roc_pr_points(pairs)
        auc_roc = trapz_auc([p["fpr"] for p in pts], [p["tpr"] for p in pts])
        auc_pr = trapz_auc([p["recall"] for p in pts], [p["precision"] for p in pts])
        ap = average_precision(pts)

        auc_rows.append([src["judge"], src["model"], len(sub), r4(auc_roc), r4(auc_pr), r4(ap),
                         f"n_thresholds={len(pts)}"])
        for p in pts:
            curve_rows.append([src["name"], src["judge"], src["model"], "roc",
                               r4(p["threshold"]), r4(p["fpr"]), r4(p["tpr"])])
            curve_rows.append([src["name"], src["judge"], src["model"], "pr",
                               r4(p["threshold"]), r4(p["recall"]), r4(p["precision"])])

    write_csv(out_dir / "roc_pr_auc.csv", auc_header, auc_rows)
    write_csv(out_dir / "roc_pr_curve_points.csv", curve_header, curve_rows)

    print(f"[roc-pr-auc] wrote {len(auc_rows)} judge/model AUC rows, {len(curve_rows)} curve points")
    for row in auc_rows:
        print(f"  {row[0]:20s} {row[1]:28s} auc_roc={row[3]} ap={row[5]} auc_pr(trapz)={row[4]} {row[6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
