#!/usr/bin/env python3
"""Threshold-independent evaluation via ROC and PR curves.

Every judge here emits a 0-100 score (higher = healthier / more PASS-like),
recorded per scenario in <results>/results_raw.csv and, for a hosted model, in
<results>/agg/frontier_long_<alias>.csv. This sweeps a threshold over that score
(independent of the fixed pass=75/marginal=50 operating point used everywhere
else) to build ROC and PR curves and their AUCs.

Which configurations are swept depends on which run is in front of it, and the
two cases are separate on purpose (`sources_for`).

The seeded run sweeps the five configurations the article reports, and they are
written down here rather than inferred: the statistical judge, ai:summary/phi4-llm,
hybrid:gated/phi4-llm, ai:raw/claude-opus-4-8-vlm and hybrid:gated/claude-opus-4-8-vlm.
That list is a fact about the study, in exactly the way
tools/figure_approach_bars.py's FRONTIER_MODELS is, and a directory listing is not
where a fact belongs. Derived instead, it grew from five rows to nine the moment it
met the study's own archived agg/, because that directory holds three untagged
frontier files and not the one the fixture had; every published number was still
bit-identical, and fig13_threshold_summaries.png -- laid out for five groups -- came
out with overprinted x-tick labels and colliding value labels.

An own-data run derives the list from what the run actually contains: the
statistical judge, the strongest local `ai:*` configuration in <results>/summary.csv,
the gated hybrid on that same model, and for each hosted model the run holds a file
for, its strongest representation and its gated hybrid. Nothing is emitted for a
configuration the run does not have. "(best local)" is then a fact about the run in
front of it, and says "tied N ways" when the lead is shared.

Positive class = FAIL, so a scenario is scored on fail_score = 100 - score
(higher fail_score means the judge thinks it's more likely a FAIL). The
threshold is swept over every distinct fail_score value observed, plus the two
sentinel operating points "predict nothing" and "predict everything".

A row whose `truth` is empty carries no label and is dropped before any pair is
built. results_raw.csv holds every judgement, labelled or not -- an own-data run
with unlabelled windows (docs/own-data.md) legitimately writes them -- and
`truth == "FAIL"` would read each of those as a genuine PASS-truth window, i.e.
as a true negative the judge was never tested on. The count dropped is reported
beside n rather than left implicit, because an ROC over 7 of 8 windows and an
ROC over 8 are different measurements. The console reports it for every run; the
CSV carries it as a trailing `n_unlabelled_dropped` column for an own-data run
only -- trailing so that no existing reader of auc_roc, auc_pr or
average_precision finds them one column to the right, and own-data only because
the seeded dataset labels every window, so on the seeded run the column would be
a column of zeroes in a file whose bytes are cited.

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

Outputs (<results>/agg/):
  - roc_pr_auc.csv           one row per (judge, model): n, auc_roc, auc_pr,
                             average_precision, note -- and, for an own-data run,
                             a trailing n_unlabelled_dropped.
  - roc_pr_curve_points.csv  long-format curve points (source, curve, threshold, x, y).

Usage:
  python tools/roc_pr_curves.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_if(path: Path) -> List[Dict[str, str]]:
    return read_csv(path) if path.is_file() else []


def _strongest(rows: List[Dict[str, str]], predicate) -> Tuple[Optional[Dict[str, str]], int]:
    """The row satisfying `predicate` with the highest accuracy, and how many tie.

    `max` over the rows in file order, which is the run's own judge order. That
    is deliberately the identical rule tools/figure_approach_bars.py applies to
    this identical file to pick its "best local" column, so the two artefacts of
    one run cannot name different models -- which they did, in the same output
    directory, on the same run.
    """
    candidates = [r for r in rows if predicate(r)]
    if not candidates:
        return None, 0
    best = max(candidates, key=lambda r: float(r["accuracy"]))
    tied = sum(1 for r in candidates
               if float(r["accuracy"]) == float(best["accuracy"]))
    return best, tied


def _accuracy_by_config(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], float]:
    """Per-(judge, model) accuracy over the labelled rows of a long-format file.

    The frontier files carry per-scenario rows rather than a scoreboard, so the
    rate has to be counted here. Unlabelled rows carry an empty `correct` and are
    not counted as wrong; they are not counted.
    """
    hits: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        if not (r.get("truth") or "").strip():
            continue
        correct = (r.get("correct") or "").strip()
        if correct not in ("0", "1"):
            continue
        hits.setdefault((r["judge"], r["model"]), []).append(int(correct))
    return {k: sum(v) / len(v) for k, v in hits.items() if v}


def is_own_data_run(results_dir: Path) -> bool:
    """Whether this results directory is one `--own-data` produced.

    The run records what it is: tools/run_experiment.py writes
    own_data_manifest.json beside the results of an own-data run and of no other
    run. Asking the run rather than asking where the directory sits is what makes
    the answer travel with it -- the study's deposited run scores as the seeded
    run wherever it is unpacked, which is the point of depositing it, and an
    own-data run scores as one wherever `--out` put it.

    The manifest has to describe the run beside it, not merely sit next to it.
    Read as a bare marker it made stale state decide the published table: one
    `--own-data --out results` left a manifest in the study's own directory, and
    every later seeded run into results/ inherited it, so roc_pr_auc.csv came out
    with a derived nine-row configuration list and an eighth
    n_unlabelled_dropped column in place of the published five rows in their
    seven columns, and fig13_threshold_summaries.png -- laid out for five groups
    -- was redrawn from it. run_experiment.py unlinks the marker on its seeded
    branch so it is not left behind; this is the other half, so that one left
    behind by any other route still cannot decide the question.

    The manifest names the windows the run judged, and results_raw.csv holds a
    row per judgement of each of them, so the two agree for the run the manifest
    is about and disagree for any other: a seeded results_raw.csv is
    clean_mean_shift_000 .. variance_increase_019 and an own-data manifest names
    own_0000 .. own_NNNN. Containment rather than equality, because a window
    whose every judge call failed is in the manifest and, correctly, in
    errors.csv rather than in results_raw.csv.
    """
    manifest_path = results_dir / "own_data_manifest.json"
    raw_path = results_dir / "results_raw.csv"
    if not manifest_path.is_file() or not raw_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        named = {str(s["id"]) for s in manifest["scenarios"]}
    except (OSError, ValueError, TypeError, KeyError):
        # Unreadable, not JSON, or not a manifest this tool wrote. It says
        # nothing about the run, so it decides nothing about the run.
        return False
    judged = {(r.get("scenario") or "").strip() for r in read_csv(raw_path)}
    judged.discard("")
    return bool(judged) and judged <= named


def published_sources(results_dir: Path, raw_path=None) -> List[Dict[str, Any]]:
    """The five configurations the article reports, resolved against one directory.

    Written down, not inferred. The frontier file is named rather than globbed:
    the study's archived agg/ holds three untagged frontier_long_*.csv files, one
    per hosted model, and the article's threshold-independent comparison is
    against one of them -- claude-opus-4-8-vlm, its frontier representative. A
    glob additionally matches the defect re-runs and the reproduction checks, and
    those are not this configuration either.

    Taking the directory as an argument rather than a module constant is what
    lets this run against a results set other than the repository's own, which
    matters for anyone re-deriving the published table from the deposit.
    """
    agg = results_dir / "agg"
    # raw_path overrides the per-scenario file (v1.1.0), so the local rows can be
    # read from a corrected results_raw.csv while agg/ stays where it is.
    raw = Path(raw_path) if raw_path else results_dir / "results_raw.csv"
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


def derived_sources(results_dir: Path) -> List[Dict[str, Any]]:
    """The configurations to sweep, read off an own-data run.

    The five literals above are a finding about the article's dataset, not a
    computation, and on a real own-data run of eight local models they reported
    three of them, two rows about `claude-opus-4-8-vlm` ("source file not found")
    that the user never ran, and called their phi4 row "(best local)" when phi4
    had only tied. For that run the list is derived:
      - the statistical judge, when the run has one;
      - the strongest `ai:*` configuration in <results>/summary.csv, by the same
        rule and out of the same file that tools/figure_approach_bars.py uses for
        its "best local" column, and labelled as tied when it is;
      - the gated hybrid on that same model, which is the pairing that makes the
        two readable as "what the gate adds";
      - for each hosted model this run holds a long-format file for, its
        strongest representation and its gated hybrid.

    Nothing is emitted for a configuration the run does not contain. The frontier
    files are matched on the name run_frontier_experiment.py writes for a default
    run and the `__`-tagged re-runs it writes for everything else are excluded by
    that same rule, so a rubric or dataset variant beside the real run is not
    read as the real run.
    """
    agg = results_dir / "agg"
    raw = results_dir / "results_raw.csv"
    summary = _read_if(results_dir / "summary.csv")
    sources: List[Dict[str, Any]] = []

    if any(r.get("judge") == "statistical" for r in summary):
        sources.append({"name": "statistical", "path": raw,
                        "judge": "statistical", "model": "-"})

    best_local, tied = _strongest(summary, lambda r: (r.get("judge") or "").startswith("ai:"))
    if best_local is not None:
        note = "best local" if tied == 1 else f"best local -- tied {tied} ways"
        sources.append({"name": f"{best_local['judge']}/{best_local['model']} ({note})",
                        "path": raw, "judge": best_local["judge"],
                        "model": best_local["model"]})
        if any(r.get("judge") == "hybrid:gated" and r.get("model") == best_local["model"]
               for r in summary):
            sources.append({"name": f"hybrid:gated/{best_local['model']}", "path": raw,
                            "judge": "hybrid:gated", "model": best_local["model"]})

    for path in sorted(agg.glob("frontier_long_*.csv")):
        alias = path.stem[len("frontier_long_"):]
        if "__" in alias:            # a rubric or dataset re-run, not the run
            continue
        accuracy = _accuracy_by_config(read_csv(path))
        rep = max((k for k in accuracy if k[0].startswith("ai:") and k[1] == alias),
                  key=lambda k: accuracy[k], default=None)
        if rep is not None:
            sources.append({"name": f"{rep[0]}/{alias} (frontier best rep)",
                            "path": path, "judge": rep[0], "model": alias})
        if ("hybrid:gated", alias) in accuracy:
            sources.append({"name": f"hybrid:gated/{alias}", "path": path,
                            "judge": "hybrid:gated", "model": alias})

    return sources


def sources_for(results_dir: Path, raw_path=None) -> List[Dict[str, Any]]:
    """The configurations to sweep: the published five, or this run's own.

    `raw_path` overrides the per-scenario file for the published five (v1.1.0),
    so the local rows can be read from a corrected results_raw.csv while agg/
    stays where it is. A derived list is read off the run's own summary.csv and
    results_raw.csv, which is the run the override is not about, so it does not
    reach it.
    """
    if is_own_data_run(results_dir):
        return derived_sources(results_dir)
    return published_sources(results_dir, raw_path)


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
    ap_.add_argument("--raw", default=None,
                     help="results_raw.csv for the local configurations "
                          "(default: <results>/results_raw.csv)")
    ap_.add_argument("--out", default=None,
                     help="output directory (default: <results>/agg)")
    args = ap_.parse_args()
    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir / "agg"

    # The eighth column exists for a run that can have something to put in it.
    # See the module header: appended rather than inserted, and left off the
    # seeded run entirely, so that run's roc_pr_auc.csv is the file it has
    # always been.
    own_data_run = is_own_data_run(results_dir)
    auc_header = ["judge", "model", "n", "auc_roc", "auc_pr", "average_precision", "note"]
    if own_data_run:
        auc_header = auc_header + ["n_unlabelled_dropped"]
    curve_header = ["source", "judge", "model", "curve", "threshold", "x", "y"]
    auc_rows = []      # the seven published columns, one list per configuration
    dropped_rows = []  # the eighth, in step with auc_rows; reported either way
    curve_rows = []

    sources = sources_for(results_dir, args.raw)
    if not sources:
        print(f"[roc-pr-auc] no configuration to sweep in {results_dir}: "
              f"summary.csv carries no statistical or ai:* row and agg/ holds no "
              f"frontier_long_*.csv. Nothing written.", file=sys.stderr)
        return 0

    def emit(judge: str, model: str, n: int, auc_roc, auc_pr, ap, note: str,
             dropped: int = 0) -> None:
        auc_rows.append([judge, model, n, auc_roc, auc_pr, ap, note])
        dropped_rows.append(dropped)

    for src in sources:
        if not src["path"].exists():
            emit(src["judge"], src["model"], 0, "", "", "",
                 f"source file not found: {src['path'].name}")
            continue
        rows = read_csv(src["path"])
        found = [r for r in rows if r["judge"] == src["judge"] and r["model"] == src["model"]]
        if not found:
            emit(src["judge"], src["model"], 0, "", "", "",
                 "no rows for this (judge, model)")
            continue
        # An unlabelled window is not a negative. Dropped here, before the pairs,
        # and counted so the n below is the number of windows this curve was
        # actually measured on.
        sub = [r for r in found if (r.get("truth") or "").strip()]
        dropped = len(found) - len(sub)
        if not sub:
            emit(src["judge"], src["model"], 0, "", "", "",
                 f"skipped: none of the {dropped} row(s) carry a label", dropped)
            continue
        missing_score = [r for r in sub if r.get("score") in (None, "", "nan")]
        if missing_score:
            emit(src["judge"], src["model"], len(sub), "", "", "",
                 f"skipped: {len(missing_score)}/{len(sub)} rows have no score (verdict-only)",
                 dropped)
            continue

        pairs = [(r["truth"] == "FAIL", 100.0 - float(r["score"])) for r in sub]
        pts = roc_pr_points(pairs)
        auc_roc = trapz_auc([p["fpr"] for p in pts], [p["tpr"] for p in pts])
        auc_pr = trapz_auc([p["recall"] for p in pts], [p["precision"] for p in pts])
        ap = average_precision(pts)

        emit(src["judge"], src["model"], len(sub), r4(auc_roc), r4(auc_pr), r4(ap),
             f"n_thresholds={len(pts)}", dropped)
        for p in pts:
            curve_rows.append([src["name"], src["judge"], src["model"], "roc",
                               r4(p["threshold"]), r4(p["fpr"]), r4(p["tpr"])])
            curve_rows.append([src["name"], src["judge"], src["model"], "pr",
                               r4(p["threshold"]), r4(p["recall"]), r4(p["precision"])])

    written = ([row + [dropped] for row, dropped in zip(auc_rows, dropped_rows)]
               if own_data_run else auc_rows)
    # Made here rather than at the top, so the run that sweeps nothing leaves
    # nothing: an unlabelled run got an empty agg/ and a "Nothing written", and
    # then `make analysis` signed off with "OK -> <results>/agg/" pointing at it.
    # tools/baselines_and_mcc.py skips the same run the same way.
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "roc_pr_auc.csv", auc_header, written)
    write_csv(out_dir / "roc_pr_curve_points.csv", curve_header, curve_rows)

    print(f"[roc-pr-auc] wrote {len(auc_rows)} judge/model AUC rows, {len(curve_rows)} curve points")
    for row, dropped in zip(auc_rows, dropped_rows):
        unlabelled = f" (+{dropped} unlabelled dropped)" if dropped else ""
        print(f"  {row[0]:20s} {row[1]:28s} n={row[2]}{unlabelled} auc_roc={row[3]} "
              f"ap={row[5]} auc_pr(trapz)={row[4]} {row[6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
