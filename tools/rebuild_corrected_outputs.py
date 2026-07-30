#!/usr/bin/env python3
"""Re-derive the scoreboard, family matrix and figures from the corrected results.

Runs the same aggregation and the same figure code the published run used --
imported from run_experiment.py and render_figures.py rather than reimplemented --
over `results/corrected/results_raw.csv`. Anything that differs from the published
output therefore differs because the *data* changed, not because the arithmetic
did, which is the only way the comparison is worth anything.

Writes to results/corrected/. `reference-results/` is not touched.

Usage:
  python tools/rebuild_corrected_outputs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import run_experiment as exp  # noqa: E402
from container_plot import png_size  # noqa: E402

CORRECTED = REPO_ROOT / "results" / "corrected"
AILOG = REPO_ROOT / "data" / "ai-logs"
PUBLISHED_SUMMARY = REPO_ROOT / "reference-results" / "n20" / "summary.csv"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        r["correct"] = int(r["correct"]) if r["correct"] not in ("", None) else 0
        r["score"] = float(r["score"]) if r["score"] not in ("", None) else 0.0
        r["latency"] = float(r["latency"]) if r["latency"] not in ("", None) else 0.0
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-derive tables and figures from corrected results")
    ap.add_argument("--raw", default=str(CORRECTED / "results_raw.csv"))
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_file():
        print(f"[rebuild] no corrected results at {raw}; run build_corrected_results.py first",
              file=sys.stderr)
        return 2

    rows = load_rows(raw)
    print(f"[rebuild] {len(rows)} rows from {raw}")

    agg = exp.aggregate(rows)
    order = exp._judge_order(agg)
    families = ed.FAMILIES

    CORRECTED.mkdir(parents=True, exist_ok=True)

    sum_header = ["judge", "model", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
                  "TP", "FP", "TN", "FN", "avg_latency"]
    sum_rows = []
    for key in order:
        a = agg[key]
        sum_rows.append([key[0], key[1], a["n"], exp.r3(a["accuracy"]), exp.r3(a["precision"]),
                         exp.r3(a["recall"]), exp.r3(a["f1"]), exp.r3(a["fpr"]), exp.r3(a["fnr"]),
                         a["TP"], a["FP"], a["TN"], a["FN"], a["avg_latency"]])
    exp.write_csv(CORRECTED / "summary.csv", sum_header, sum_rows)
    print(f"[rebuild] summary.csv ({len(sum_rows)} rows)")

    fam_header = ["judge", "model"] + families
    fam_rows = []
    by_group_family: Dict[Any, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_group_family[(r["judge"], r["model"])][r["family"]].append(r["correct"])
    for key in order:
        row = [key[0], key[1]]
        for fam in families:
            vals = by_group_family[key].get(fam, [])
            row.append(exp.r3(sum(vals) / len(vals)) if vals else "")
        fam_rows.append(row)
    exp.write_csv(CORRECTED / "family_matrix.csv", fam_header, fam_rows)
    print(f"[rebuild] family_matrix.csv ({len(fam_rows)} rows)")

    # The figure spec, written by the same function the published run used.
    exp._write_figdata(rows, agg, order, families, [])
    spec_src = AILOG / "_exp_figdata.json"
    shutil.copy(spec_src, CORRECTED / "figdata.json")
    print(f"[rebuild] figure spec -> {CORRECTED / 'figdata.json'}")

    # The heatmap's ensemble and frontier rows are read from a <results>/agg
    # directory *inside the container*, and only data/ai-logs is mounted there.
    # Stage the four tables the published figure used into the mount so the
    # corrected heatmap has the same rows as the published one; without this it
    # silently drops them and the two figures are not comparable.
    stage = AILOG / "_corrected_results" / "agg"
    stage.mkdir(parents=True, exist_ok=True)
    ref_agg = REPO_ROOT / "reference-results" / "n20" / "agg"
    staged = []
    for name in ("ensemble_family_matrix.csv",
                 "frontier_family_matrix_claude-opus-4-8-vlm.csv",
                 "frontier_family_matrix_gpt-oss-120b-llm.csv",
                 "frontier_family_matrix_qwen3-vl-235b-vlm.csv"):
        src = ref_agg / name
        if not src.is_file():
            src = REPO_ROOT / "results" / "agg" / name
        if src.is_file():
            shutil.copy(src, stage / name)
            staged.append(name)
    print(f"[rebuild] staged {len(staged)} extra heatmap tables for the container")

    script = (REPO_ROOT / "tools" / "render_figures.py").read_text()
    proc = subprocess.run(
        ["docker", "compose", "exec", "-T", "judge-service", "python", "-",
         "--results", "/app/data/ai-logs/_corrected_results"],
        input=script, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300)
    if proc.returncode != 0:
        print(f"[rebuild] figure render FAILED:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}",
              file=sys.stderr)
        return 1

    figdir = CORRECTED / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    for png in sorted((AILOG / "_exp_figs").glob("*.png")):
        if png.name in ("accuracy_bars.png", "family_heatmap.png", "confusion_matrices.png"):
            shutil.copy(png, figdir / png.name)
            written.append(png_size(figdir / png.name))
    print(f"[rebuild] {len(written)} figures -> {figdir}")
    for w in written:
        print(f"    {w['file']:26s} {w['width_px']} x {w['height_px']} px @ {w['dpi']} dpi "
              f"({w['bytes']/1024:.0f} kB)")

    # What moved against the published scoreboard.
    if PUBLISHED_SUMMARY.is_file():
        pub = {(r["judge"], r["model"]): r for r in csv.DictReader(open(PUBLISHED_SUMMARY, newline=""))}
        print("\n[rebuild] rows whose metrics changed against reference-results/n20/summary.csv:")
        changed = 0
        for row in sum_rows:
            key = (row[0], row[1])
            old = pub.get(key)
            if not old:
                continue
            if any(str(old[c]) != str(v) for c, v in
                   zip(sum_header, row) if c in ("accuracy", "precision", "recall", "f1")):
                changed += 1
                print(f"    {key[0]:16s} {key[1]:20s} "
                      f"acc {old['accuracy']} -> {row[3]}   "
                      f"prec {old['precision']} -> {row[4]}   "
                      f"rec {old['recall']} -> {row[5]}   F1 {old['f1']} -> {row[6]}")
        if not changed:
            print("    none")
        print(f"    ({changed} of {len(sum_rows)} configurations moved)")

    (CORRECTED / "figure_dimensions.json").write_text(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
