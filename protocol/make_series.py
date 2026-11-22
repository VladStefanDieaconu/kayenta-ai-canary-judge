#!/usr/bin/env python3
"""Write the series of the two 1.2.0 control datasets and their realised checks.

Run once, before any judge sees a window of either dataset:

    python protocol/make_series.py --deposit <data deposit 1.1.1>

For kind-controls-180 and heldout-benign-100, at master seed 20260621 and 20
windows per family, this writes

  protocol/series_<dataset>.jsonl            one window per line: id, family, gid,
                                             seed, truth, token, generator, params
                                             and, per metric, both series as generated
                                             by tools/eval_dataset.py
  protocol/realised_checks_<dataset>.csv     one row per window and metric
  protocol/isolated_spikes_both_repeats.csv  placement repeats per window
  protocol/realised_checks_provenance.json   which functions computed the checks

The realised checks are descriptions of the windows, not judges: mean ratio,
sample SD ratio, maximum ratio, the ratio of the mean of the last five canary
samples to the control mean, and the two one-sided tests (pooled variance) at an
equivalence margin of 5% of the control mean, alpha 0.05 per test. They are
computed with the functions that produced the "label basis or realised check"
column of Table 1 of the article: mean, sd and tost_p (with t_cdf, betainc_reg
and _betacf) of analysis/area-F-instrument-audit/analyze_F.py in the data
deposit (Table S42 names that script for Table 1). The functions are read out of
that file and executed as written; the file itself is not imported, because its
module-level code prepares a working copy of the deposit.

Every float is written with Python's repr, so the files round-trip exactly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "tools"))
import eval_dataset as ed  # noqa: E402

MASTER_SEED = 20260621
N_PER_FAMILY = 20
DATASETS = ("kind-controls-180", "heldout-benign-100")
ANALYZE_F = "analysis/area-F-instrument-audit/analyze_F.py"
FUNCTIONS = ("mean", "sd", "_betacf", "betainc_reg", "t_cdf", "tost_p")
TOST_MARGIN = 0.05
TOST_ALPHA = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_functions(deposit: Path):
    src_path = deposit / ANALYZE_F
    src = src_path.read_text()
    tree = ast.parse(src)
    ns = {"math": math}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            seg = ast.get_source_segment(src, node)
            found[node.name] = hashlib.sha256(seg.encode("utf-8")).hexdigest()
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ANALYZE_F), "exec"), ns)
    missing = [f for f in FUNCTIONS if f not in found]
    if missing:
        raise SystemExit(f"{ANALYZE_F}: function(s) not found: {missing}")
    return ns, {"file": ANALYZE_F, "file_sha256": sha256(src_path),
                "functions_sha256": found}


def window_record(dsid: str, sc: ed.Scenario, series) -> dict:
    return {
        "dataset_id": dsid, "id": sc.id, "family": sc.family, "gid": sc.gid, "seed": sc.seed,
        "truth": sc.truth, "token": sc.token, "generator": ed.generator_of(sc.family),
        "params": sc.params,
        "metrics": [{"name": m.name, "kind": m.kind, "base": m.base, "group": m.group,
                     "unit": m.unit, "direction": m.direction, "truth": m.truth,
                     "control": series[m.name]["control"],
                     "experiment": series[m.name]["experiment"]} for m in sc.metrics],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", required=True, help="the data deposit 1.1.1 (read only)")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    fn, prov = load_functions(Path(args.deposit))
    df = 2 * ed.N_SAMPLES - 2

    checks_header = ["dataset_id", "id", "family", "truth", "metric", "kind",
                     "mean_ratio", "sd_ratio", "max_ratio", "last5_mean_over_control_mean",
                     "tost_p_margin_5pct", "tost_equivalent_5pct"]
    repeats = []
    hashes = {}
    for dsid in DATASETS:
        scenarios = ed.build_scenarios(MASTER_SEED, N_PER_FAMILY, dsid)
        series_path = out / f"series_{dsid}.jsonl"
        checks_path = out / f"realised_checks_{dsid}.csv"
        with open(series_path, "w") as fs, open(checks_path, "w", newline="") as fc:
            w = csv.writer(fc, lineterminator="\n")
            w.writerow(checks_header)
            for sc in scenarios:
                series = ed.series_for_scenario(sc)
                fs.write(json.dumps(window_record(dsid, sc, series), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")
                for m in sc.metrics:
                    c, e = series[m.name]["control"], series[m.name]["experiment"]
                    p = fn["tost_p"](c, e, TOST_MARGIN, df)
                    w.writerow([dsid, sc.id, sc.family, sc.truth, m.name, m.kind,
                                repr(fn["mean"](e) / fn["mean"](c)), repr(fn["sd"](e) / fn["sd"](c)),
                                repr(max(e) / max(c)), repr(fn["mean"](e[-5:]) / fn["mean"](c)),
                                repr(p), int(p < TOST_ALPHA)])
                if ed.generator_of(sc.family) == "isolated_spikes_both":
                    pp = sc.params[sc.metrics[0].name]
                    repeats.append([sc.id, pp["n_elevated"], pp["placement_repeats_control"],
                                    pp["placement_repeats_experiment"]])
        hashes[series_path.name] = sha256(series_path)
        hashes[checks_path.name] = sha256(checks_path)
        print(f"{dsid}: {len(scenarios)} windows -> {series_path.name} {hashes[series_path.name]}")
        print(f"{dsid}: realised checks -> {checks_path.name} {hashes[checks_path.name]}")

    rep_path = out / "isolated_spikes_both_repeats.csv"
    with open(rep_path, "w", newline="") as fr:
        w = csv.writer(fr, lineterminator="\n")
        w.writerow(["id", "n_elevated", "placement_repeats_control", "placement_repeats_experiment"])
        w.writerows(repeats)
    hashes[rep_path.name] = sha256(rep_path)

    prov.update({"master_seed": MASTER_SEED, "n_per_family": N_PER_FAMILY,
                 "tost": {"margin_fraction_of_control_mean": TOST_MARGIN, "alpha_per_test": TOST_ALPHA,
                          "df": df, "variance": "pooled", "equivalent_iff": "tost_p < alpha"},
                 "max_ratio": "max(canary) / max(control)",
                 "last5": "mean(canary[-5:]) / mean(control)",
                 "outputs_sha256": hashes})
    (out / "realised_checks_provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
