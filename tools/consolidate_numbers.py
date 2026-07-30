#!/usr/bin/env python3
"""Regenerate every table the manuscript cites, from artefacts, with sources.

Three earlier consolidations were wrong, each caught by going back to the data,
so nothing here is copied from a checkpoint: every cell is recomputed from a file
on disk and every table names the file and columns it came from. Re-running this
after any new run is how the tables stay true.

Rates carry Wilson intervals. On a 20-scenario family cell, 0.000 and 1.000 are
common, and a Wald interval on either leaves the unit range.

Usage:
  python tools/consolidate_numbers.py [--out audit/CONSOLIDATED-NUMBERS.md]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import results_schema as rs  # noqa: E402
from analyse_ablation import metrics, paired_diff, wilson  # noqa: E402

REFERENCE_RAW = REPO_ROOT / "reference-results" / "n20" / "results_raw.csv"
FROZEN = "v1-frozen-2026-06"
PROMPTS = [FROZEN, "v2-operational-2026-07", "v3-temporal-2026-07"]
SHORT = {FROZEN: "v1 frozen", "v2-operational-2026-07": "v2 operational",
         "v3-temporal-2026-07": "v3 temporal"}

# The three always-something baselines on the 120 FAIL / 60 PASS split. Stated
# because an accuracy near 0.667 on this split is the always-FAIL baseline and
# not a result; every headline number is printed beside MCC and balanced accuracy
# so it cannot be read as one.
BASELINES = [
    ("always-FAIL", 0.667, 0.500, 0.000),
    ("always-PASS", 0.333, 0.500, 0.000),
    ("stratified random", 0.556, 0.500, 0.000),
]


def f3(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.3f}"


def ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"[{lo:.3f}, {hi:.3f}]" if n else "—"


def fam_cell(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "—"
    k = sum(1 for r in rows if r["correct"] == 1)
    return f"{k/len(rows):.3f}"


def section_representation(L: List[str]) -> None:
    L += ["## A — Frozen-prompt representation accuracy, three hosted models", "",
          "Source: `results/agg/experiments/published__*__v1-frozen-2026-06__original-180.csv`, "
          "imported verbatim from the published `results/agg/frontier_long_<model>.csv` "
          "(columns `verdict`, `truth`, `correct`). n = 180 per cell.", "",
          "| model | representation | acc | 95% CI (Wilson) | bal.acc | MCC | prec | rec | F1 | FPR | TP/FP/TN/FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in rs.load(dataset_id="original-180", judge="ai", prompt_id=FROZEN)}):
        for rep in ("summary", "raw", "plot"):
            rows = rs.load(dataset_id="original-180", judge="ai", prompt_id=FROZEN,
                           model=model, representation=rep)
            if not rows:
                continue
            m = metrics(rows)
            k = m["TP"] + m["TN"]
            L.append(f"| {model} | ai:{rep} | **{f3(m['accuracy'])}** | {ci(k, m['n'])} | "
                     f"{f3(m['balanced_accuracy'])} | {f3(m['mcc'])} | {f3(m['precision'])} | "
                     f"{f3(m['recall'])} | {f3(m['f1'])} | {f3(m['fpr'])} | "
                     f"{m['TP']}/{m['FP']}/{m['TN']}/{m['FN']} |")
    L += ["", "Baselines on this 120 FAIL / 60 PASS split, for reading the column above:", "",
          "| baseline | acc | bal.acc | MCC |", "|---|---|---|---|"]
    for name, a, b, c in BASELINES:
        L.append(f"| {name} | {a:.3f} | {b:.3f} | {c:.3f} |")
    L.append("")


def section_family(L: List[str]) -> None:
    fams = ed.FAMILIES
    L += ["## B — Frozen-prompt per-family accuracy (the nine original families)", "",
          "Source: same rows as §A, grouped on `family`. Each cell is k/20.", "",
          "| model / representation | " + " | ".join(f.replace("_", " ") for f in fams) + " |",
          "|" + "---|" * (len(fams) + 1)]
    for model in sorted({r["model"] for r in rs.load(dataset_id="original-180", judge="ai", prompt_id=FROZEN)}):
        for rep in ("summary", "raw", "plot"):
            rows = rs.load(dataset_id="original-180", judge="ai", prompt_id=FROZEN,
                           model=model, representation=rep)
            if not rows:
                continue
            cells = [fam_cell([r for r in rows if r["family"] == f]) for f in fams]
            L.append(f"| {model} / {rep} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["**Qwen3-VL-235B `no_change` and `noise_equivalent`**, which §I of MANUSCRIPT_DATA "
          "records as missing and the manuscript needs:", "",
          "| representation | no_change | noise_equivalent |", "|---|---|---|"]
    for rep in ("summary", "raw", "plot"):
        rows = rs.load(dataset_id="original-180", judge="ai", prompt_id=FROZEN,
                       model="qwen3-vl-235b-vlm", representation=rep)
        if not rows:
            continue
        nc = [r for r in rows if r["family"] == "no_change"]
        ne = [r for r in rows if r["family"] == "noise_equivalent"]
        L.append(f"| ai:{rep} | {fam_cell(nc)} ({sum(1 for r in nc if r['correct']==1)}/20) | "
                 f"{fam_cell(ne)} ({sum(1 for r in ne if r['correct']==1)}/20) |")
    L.append("")


def section_ablation(L: List[str], dataset: str, title: str) -> None:
    models = sorted({r["model"] for r in rs.load(dataset_id=dataset, judge="ai")})
    present = [p for p in PROMPTS if rs.load(dataset_id=dataset, judge="ai",
                                             prompt_id=p, representation="raw")]
    if not present or not models:
        L += [f"## {title}", "", "_No rows in the frame for this dataset._", ""]
        return

    L += [f"## {title}", "",
          "Source: `results/agg/experiments/*.csv` via `tools/results_schema.py`, "
          "selecting `dataset_id`, `prompt_id`, `representation='raw'`, `judge='ai'`. "
          "Error rows are excluded by the loader and counted separately in §F.", "",
          "| model | prompt | n | acc | 95% CI (Wilson) | bal.acc | MCC | prec | rec | F1 | FPR | TP/FP/TN/FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        for p in present:
            rows = rs.load(dataset_id=dataset, judge="ai", representation="raw",
                           model=model, prompt_id=p)
            if not rows:
                continue
            m = metrics(rows)
            k = m["TP"] + m["TN"]
            L.append(f"| {model} | {SHORT[p]} | {m['n']} | **{f3(m['accuracy'])}** | {ci(k, m['n'])} | "
                     f"{f3(m['balanced_accuracy'])} | {f3(m['mcc'])} | {f3(m['precision'])} | "
                     f"{f3(m['recall'])} | {f3(m['f1'])} | {f3(m['fpr'])} | "
                     f"{m['TP']}/{m['FP']}/{m['TN']}/{m['FN']} |")
    L.append("")

    fams = sorted({r["family"] for r in rs.load(dataset_id=dataset, judge="ai")},
                  key=ed.family_order)
    L += ["### Per-family accuracy (k/20)", "",
          "| model | prompt | " + " | ".join(f.replace("_", " ") for f in fams) + " |",
          "|" + "---|" * (len(fams) + 2)]
    for model in models:
        for p in present:
            rows = rs.load(dataset_id=dataset, judge="ai", representation="raw",
                           model=model, prompt_id=p)
            if not rows:
                continue
            cells = [fam_cell([r for r in rows if r["family"] == f]) for f in fams]
            L.append(f"| {model} | {SHORT[p]} | " + " | ".join(cells) + " |")
    L.append("")

    if FROZEN in present and len(present) > 1:
        L += ["### Paired change against the frozen rubric", "",
              "Both arms judge the identical scenario set, so this is a paired "
              "comparison and McNemar's exact test applies. `fixed` and `broke` count "
              "scenarios whose correctness changed, not whose verdict changed.", "",
              "| model | prompt | verdict changes | fixed | broke | net | McNemar exact p |",
              "|---|---|---|---|---|---|---|"]
        for model in models:
            base = rs.load(dataset_id=dataset, judge="ai", representation="raw",
                           model=model, prompt_id=FROZEN)
            for p in present:
                if p == FROZEN:
                    continue
                rows = rs.load(dataset_id=dataset, judge="ai", representation="raw",
                               model=model, prompt_id=p)
                if not rows or not base:
                    continue
                d = paired_diff(base, rows)
                L.append(f"| {model} | {SHORT[p]} | {d['verdict_changes']} | {d['b_fixed']} | "
                         f"{d['b_broke']} | {d['net']:+d} | {d['mcnemar_exact_p']} |")
        L.append("")


def section_new_family_baselines(L: List[str]) -> None:
    rows = rs.load(dataset_id="generalisation-60")
    if not rows:
        return
    fams = sorted({r["family"] for r in rows}, key=ed.family_order)
    L += ["## E — The generalisation-60 families under the model-free judges", "",
          "Source: `results/agg/experiments/validation__generalisation-60__*.csv` "
          "(`tools/validate_new_families.py`). The tuned ensemble uses the published "
          "config from `reference-results/n20/agg/ensemble_config.json`, not a re-tune.", "",
          "| judge | " + " | ".join(fams) + " |", "|" + "---|" * (len(fams) + 1)]
    for judge, rep, label in (("statistical", "", "statistical"),
                              ("ensemble", "tuned", "ensemble:tuned"),
                              ("ensemble", "no_recovery_guard", "ensemble:no_recovery_guard")):
        sel = rs.load(dataset_id="generalisation-60", judge=judge, representation=rep)
        if not sel:
            continue
        cells = [fam_cell([r for r in sel if r["family"] == f]) for f in fams]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")


def section_cost(L: List[str]) -> None:
    rows = rs.load(judge="ai", include_errors=True)
    with_usage = [r for r in rows if r["tokens_in"] is not None]
    L += ["## F — Tokens, latency and error rows (measured, not estimated)", "",
          "Source: the `tokens_in` / `tokens_out` / `latency_s` / `error_kind` columns of the "
          "long-format frame. These come from the gateway's own `usage` block, captured "
          "for the first time tonight.", ""]
    if not with_usage:
        L += ["_No rows carry usage yet._", ""]
    else:
        L += ["| model | prompt | calls | tokens in | tokens out | mean latency s |",
              "|---|---|---|---|---|---|"]
        agg = defaultdict(lambda: {"n": 0, "ti": 0, "to": 0, "lat": 0.0})
        for r in with_usage:
            a = agg[(r["model"], r["prompt_id"])]
            a["n"] += 1
            a["ti"] += r["tokens_in"] or 0
            a["to"] += r["tokens_out"] or 0
            a["lat"] += r["latency_s"] or 0.0
        tot_in = tot_out = tot_n = 0
        for (model, pid), a in sorted(agg.items()):
            L.append(f"| {model} | {SHORT.get(pid, pid or '—')} | {a['n']} | {a['ti']:,} | "
                     f"{a['to']:,} | {a['lat']/a['n']:.1f} |")
            tot_in += a["ti"]
            tot_out += a["to"]
            tot_n += a["n"]
        L += [f"| **total** | | **{tot_n}** | **{tot_in:,}** | **{tot_out:,}** | |", "",
              "The three earlier hosted runs remain **estimates**: they were made before "
              "usage was captured and the gateway's counts were discarded at the time. "
              "Only rows written tonight carry measured tokens.", ""]

    errs = [r for r in rows if r["error_kind"]]
    L += [f"**Error rows: {len(errs)}** of {len(rows)} AI rows. An error row carries no "
          "verdict and is excluded from every metric above.", ""]
    if errs:
        by = defaultdict(int)
        for r in errs:
            by[(r["model"], r["error_kind"])] += 1
        L += ["| model | error_kind | count |", "|---|---|---|"]
        for (m, k), v in sorted(by.items()):
            L.append(f"| {m} | `{k}` | {v} |")
        L.append("")


def section_reference_defect(L: List[str]) -> None:
    """The 22 fabricated rows, recomputed here rather than quoted from Checkpoint 16."""
    if not REFERENCE_RAW.is_file():
        return
    ref = list(csv.DictReader(open(REFERENCE_RAW, newline="")))
    ai = [r for r in ref if r["judge"].startswith("ai:")]
    bad = [r for r in ai if r["rationale"].startswith("AI judge error")]
    L += ["## G — Error results recorded as verdicts in `reference-results/`", "",
          f"Source: `reference-results/n20/results_raw.csv`, rows whose `rationale` begins "
          f"`AI judge error (` — the string `_error_result` writes. **{len(bad)} of {len(ai)} "
          "AI rows.** Every one has an empty `error` column and a verdict of FAIL at score 0.0.", ""]
    if not bad:
        L.append("")
        return
    by = defaultdict(list)
    for r in bad:
        by[(r["model"], r["judge"])].append(r)
    L += ["| model / judge | fabricated rows | scored correct | families |", "|---|---|---|---|"]
    for (m, j), v in sorted(by.items()):
        fams = ", ".join(f"{f}×{c}" for f, c in
                         sorted(defaultdict(int, {x["family"]: sum(1 for y in v if y["family"] == x["family"]) for x in v}).items()))
        L.append(f"| {m} / {j} | {len(v)} | {sum(1 for r in v if r['correct']=='1')} | {fams} |")
    L.append("")
    for (m, j), v in sorted(by.items()):
        rows_all = [r for r in ai if r["model"] == m and r["judge"] == j]
        good = [r for r in rows_all if not r["rationale"].startswith("AI judge error")]
        pub = metrics([{"truth_label": r["truth"], "verdict": r["verdict"]} for r in rows_all])
        exc = metrics([{"truth_label": r["truth"], "verdict": r["verdict"]} for r in good])
        L += [f"**{m} / {j}**", "",
              "| | n | acc | bal.acc | MCC |", "|---|---|---|---|---|",
              f"| as published | {pub['n']} | {f3(pub['accuracy'])} | {f3(pub['balanced_accuracy'])} | {f3(pub['mcc'])} |",
              f"| error rows excluded | {exc['n']} | {f3(exc['accuracy'])} | {f3(exc['balanced_accuracy'])} | {f3(exc['mcc'])} |",
              ""]
        fam_pub = defaultdict(list)
        fam_exc = defaultdict(list)
        for r in rows_all:
            fam_pub[r["family"]].append(int(r["correct"]))
        for r in good:
            fam_exc[r["family"]].append(int(r["correct"]))
        moved = [f for f in fam_pub
                 if (sum(fam_pub[f]) / len(fam_pub[f])) !=
                 ((sum(fam_exc[f]) / len(fam_exc[f])) if fam_exc[f] else None)]
        if moved:
            L += ["| family | published | errors excluded | n remaining |", "|---|---|---|---|"]
            for f in sorted(moved, key=ed.family_order):
                p = sum(fam_pub[f]) / len(fam_pub[f])
                e = (sum(fam_exc[f]) / len(fam_exc[f])) if fam_exc[f] else None
                L.append(f"| `{f}` | {p:.3f} | {f3(e) if e is not None else '**no data**'} | {len(fam_exc[f])} |")
            L.append("")

    # The hybrid policies consume the model's AI verdict, so a fabricated AI row
    # contaminates the hybrid row derived from the same scenario. Those hybrid
    # rows are not themselves fabricated, which is why they are reported
    # separately: the comparison drops the whole affected scenario rather than a
    # single row, because that is the unit the contamination travels in.
    for model in sorted({r["model"] for r in bad}):
        affected = {r["scenario"] for r in bad if r["model"] == model}
        judges = sorted({r["judge"] for r in ref
                         if r["model"] == model and r["judge"].startswith("hybrid")})
        if not judges:
            continue
        src = sorted({r["judge"] for r in bad if r["model"] == model})
        n_aff = len(affected)
        L += [f"### Downstream: {model}'s hybrid policies", "",
              f"{n_aff} scenario{'s' if n_aff != 1 else ''} "
              f"{'have' if n_aff != 1 else 'has'} a fabricated "
              f"{', '.join('`'+s+'`' for s in src)} input. Dropping "
              f"{'those scenarios' if n_aff != 1 else 'that scenario'} entirely:", "",
              "| judge | | n | acc | prec | rec | MCC |", "|---|---|---|---|---|---|---|"]
        for j in judges:
            rows_all = [r for r in ref if r["model"] == model and r["judge"] == j]
            kept = [r for r in rows_all if r["scenario"] not in affected]
            for label, sel in (("as published", rows_all), ("affected scenarios dropped", kept)):
                m = metrics([{"truth_label": r["truth"], "verdict": r["verdict"]} for r in sel])
                L.append(f"| {j} | {label} | {m['n']} | {f3(m['accuracy'])} | "
                         f"{f3(m['precision'])} | {f3(m['recall'])} | {f3(m['mcc'])} |")
        L.append("")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate the manuscript's tables from artefacts")
    ap.add_argument("--out", default=str(REPO_ROOT / "audit" / "CONSOLIDATED-NUMBERS.md"))
    args = ap.parse_args()

    L: List[str] = [
        "# CONSOLIDATED NUMBERS — regenerated from artefacts",
        "",
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by "
        "`tools/consolidate_numbers.py`. Every table below is recomputed from a file on "
        "disk; none is copied from a checkpoint. Re-run the script to refresh it.",
        "",
        "Rates carry Wilson 95% intervals. Accuracy is printed beside balanced accuracy and "
        "MCC throughout, because on this 120 FAIL / 60 PASS split the always-FAIL baseline is "
        "0.667 and a bare accuracy near it is not a result.",
        "",
    ]
    section_representation(L)
    section_family(L)
    section_ablation(L, "original-180", "C — The prompt ablation on the original 180 (ai:raw)")
    section_ablation(L, "generalisation-60", "D — Generalisation: the three held-out families (ai:raw)")
    section_new_family_baselines(L)
    section_cost(L)
    section_reference_defect(L)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"[consolidate] {len(L)} lines -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
