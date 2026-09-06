#!/usr/bin/env python3
"""A properly pooled paired test across seeds.

Per-seed McNemar (results/agg/mcnemar_per_seed.csv) is underpowered at
n=5/family per seed: whether a single seed reaches p<0.05 is close to a coin
flip. This pools the 5-seed rows (results/agg/long_results.csv, no new model
calls) into one paired test per comparison. Each (seed, scenario) pair is
matched across the two judges being compared (scenario names repeat across
seeds since they're independently regenerated per seed, so the seed is part of
the pairing key, not the scenario name alone), and one exact McNemar test
(tools/run_experiment.py::mcnemar_exact, imported directly so the formula
matches every other McNemar number here) is run on the full pooled set of
discordant pairs.

Comparisons (25 total):
  - per present model (8): statistical vs ai:<primary>, statistical vs
    hybrid:gated, ai:<primary> vs hybrid:gated. These are the same three pairs
    results/mcnemar.csv and mcnemar_per_seed.csv report per single seed, now
    pooled across all 5 seeds instead of one seed / one model at a time.
  - ensemble:tuned vs the best local LLM (ai:summary/phi4-llm), reusing
    results/agg/ensemble_multiseed_rows.csv, written by
    tools/ensemble_multiseed_ci.py on the same 5 seeds, so the two sides of
    this comparison share scenario/seed pairing like every other row.

All 25 raw p-values are corrected together across the whole family with Holm
(family-wise, primary) and Benjamini-Hochberg (FDR, secondary), via
tools/ci_utils.py.

Output: results/agg/mcnemar_pooled.csv.

Usage:
  python tools/pooled_mcnemar.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import ci_utils as ci  # noqa: E402
import run_experiment as exp  # noqa: E402  (reuse mcnemar_exact)

AGG_DIR = REPO_ROOT / "results" / "agg"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    rows = read_csv(AGG_DIR / "long_results.csv")
    correct_by: Dict[Tuple[str, str], Dict[Tuple[int, str], bool]] = defaultdict(dict)
    for r in rows:
        key = (r["judge"], r["model"])
        correct_by[key][(int(r["seed"]), r["scenario"])] = bool(int(r["correct"]))

    models = sorted(set(m for (j, m) in correct_by if j.startswith("ai:")))
    # primary AI mode per model: ai:summary if present (text model), else ai:plot (vision)
    primary_judge: Dict[str, str] = {}
    for m in models:
        if ("ai:summary", m) in correct_by:
            primary_judge[m] = "ai:summary"
        elif ("ai:plot", m) in correct_by:
            primary_judge[m] = "ai:plot"

    comparisons: List[Tuple[str, str, str, str]] = []  # (comparison_id, model, judge_a, judge_b)
    for m in models:
        aj = primary_judge.get(m)
        if aj is None:
            continue
        comparisons.append((f"{m}:statistical_vs_{aj}", m, "statistical", aj))
        comparisons.append((f"{m}:statistical_vs_hybrid_gated", m, "statistical", "hybrid:gated"))
        comparisons.append((f"{m}:{aj}_vs_hybrid_gated", m, aj, "hybrid:gated"))

    # ensemble vs best local LLM (ai:summary/phi4-llm), from ensemble_multiseed_ci.py's persisted rows
    ens_path = AGG_DIR / "ensemble_multiseed_rows.csv"
    if ens_path.exists():
        ens_rows = read_csv(ens_path)
        ens_correct: Dict[Tuple[int, str], bool] = {
            (int(r["seed"]), r["scenario"]): bool(int(r["correct"])) for r in ens_rows
        }
        correct_by[("ensemble:tuned", "-")] = ens_correct
        best_llm_model = "phi4-llm"  # best standalone config is ai:summary/phi4-llm
        comparisons.append(("ensemble_tuned_vs_best_llm", "-", "ensemble:tuned", "ai:summary"))
        # override model on this synthetic row so the pairing below reads phi4-llm's ai:summary correctness
        _best_llm_key = ("ai:summary", best_llm_model)
    else:
        _best_llm_key = None

    header = ["comparison_id", "model", "judge_a", "judge_b", "n_pairs", "discordant",
              "a_wrong_b_right", "a_right_b_wrong", "chi2_cc", "p_raw", "p_holm", "q_bh"]
    raw_rows = []
    p_raws = []
    def _key(judge: str, model: str) -> Tuple[str, str]:
        return (judge, "-") if judge == "statistical" else (judge, model)

    for comp_id, model, judge_a, judge_b in comparisons:
        if comp_id == "ensemble_tuned_vs_best_llm":
            key_a, key_b = ("ensemble:tuned", "-"), _best_llm_key
        else:
            key_a, key_b = _key(judge_a, model), _key(judge_b, model)
        ca, cb = correct_by.get(key_a, {}), correct_by.get(key_b, {})
        common = sorted(set(ca) & set(cb))
        if not common:
            continue
        list_a = [ca[k] for k in common]
        list_b = [cb[k] for k in common]
        mc = exp.mcnemar_exact(list_a, list_b)
        raw_rows.append([comp_id, model if model != "-" else "-", judge_a, judge_b,
                         len(common), mc["discordant"], mc["n01_a_wrong_b_right"],
                         mc["n10_a_right_b_wrong"], mc["chi2_cc"], mc["p_value"]])
        p_raws.append(mc["p_value"])

    p_holm = ci.holm_correction(p_raws)
    q_bh = ci.benjamini_hochberg(p_raws)
    out_rows = []
    for row, ph, qb in zip(raw_rows, p_holm, q_bh):
        out_rows.append(row + [round(ph, 5), round(qb, 5)])

    write_csv(AGG_DIR / "mcnemar_pooled.csv", header, out_rows)

    n_sig_raw = sum(1 for r in out_rows if r[9] < 0.05)
    n_sig_holm = sum(1 for r in out_rows if r[10] < 0.05)
    print(f"[pooled-mcnemar] wrote {len(out_rows)} comparisons -> {AGG_DIR / 'mcnemar_pooled.csv'}")
    print(f"[pooled-mcnemar] raw p<0.05: {n_sig_raw}/{len(out_rows)}; "
          f"Holm-corrected p<0.05: {n_sig_holm}/{len(out_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
