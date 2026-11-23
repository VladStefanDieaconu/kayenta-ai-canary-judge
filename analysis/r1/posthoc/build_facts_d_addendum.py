#!/usr/bin/env python3
"""Addendum to the list of D facts: facts read off the D2 and D3 output tables.

facts-D.json holds the facts the three scripts recorded while they ran. This
script adds, without recomputing anything, the figures of their output tables
that the write-up also needs: the D3 results by claim type, the sign-only trend
result, negation and placeholder counts, the secondary unit (rationale plus
per-metric reasons) and the realised-feature ranges per family behind the label
changes of P6 and P7. Ids continue after the last id of facts-D.json. Existing
files are never overwritten.

Run: python -I -B build_facts_d_addendum.py <output-dir> <time the D3 configuration was fixed>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT = "analysis/r1/posthoc/build_facts_d_addendum.py"
FAMILIES = ["no_change", "noise_equivalent", "healed_transient", "clean_mean_shift", "subtle_regression",
            "variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]


def rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out, fixed_at = Path(sys.argv[1]), sys.argv[2]
    base = json.loads((out / "facts-D.json").read_text(encoding="utf-8"))
    n0 = int(base[-1]["id"].split("-")[1])
    facts = []

    def add(key, value, description, source, computation):
        facts.append({"key": key, "value": str(value), "description": description, "source_file": source,
                      "filter_or_computation": computation, "script": SCRIPT})

    summ = rows(out / "d3_summary.csv")
    allp = [r for r in summ if r["unit"] == "primary" and r["level"] == "all configurations"][0]
    for t in ("level", "spread", "trend", "tail", "number"):
        c, n = int(allp[f"{t}_consistent"]), int(allp[f"{t}_checkable"])
        add(f"d3x.type.{t}", f"{c}/{n} ({c / n:.3f})",
            f"D3, all configurations pooled, primary unit: {t} claims consistent with the window, of checkable {t} claims",
            "posthoc/d3_summary.csv", "row unit=primary, level=all configurations")
    tr_s, tr_n = int(allp["trend_consistent_sign_only"]), int(allp["trend_checkable"])
    add("d3x.trend_sign_only", f"{tr_s}/{tr_n} ({tr_s / tr_n:.3f})",
        "D3, all configurations pooled: trend claims consistent when up and down claims are judged on the sign of the "
        "slope of canary minus control alone (secondary rule of the configuration)", "posthoc/d3_summary.csv",
        "row unit=primary, level=all configurations; column trend_consistent_sign_only")
    add("d3x.share_sign_only", allp["share_consistent_trend_sign_only"],
        "D3, all configurations pooled: share of checkable claims consistent with the window when trend claims are "
        "judged on the sign alone", "posthoc/d3_summary.csv", "column share_consistent_trend_sign_only")
    neg = int(allp["checkable_claims_with_negation"])
    add("d3x.negation", f"{neg}/{allp['checkable_claims']}",
        "D3, all configurations pooled: checkable claims in which a negation cue was detected (and the claim inverted)",
        "posthoc/d3_summary.csv", "column checkable_claims_with_negation")
    add("d3x.placeholders", f"{allp['n_placeholder']}/{allp['n_rationales']}",
        "D3: scored judgements whose rationale is empty, the harness default or a placeholder of the rubric schema "
        "(no claim extracted), all configurations", "posthoc/d3_summary.csv", "column n_placeholder")
    for r in summ:
        if r["unit"] == "primary" and r["level"] == "by configuration" and int(r["n_placeholder"]) > 0:
            add(f"d3x.placeholders.{r['model']}", f"{r['n_placeholder']}/{r['n_rationales']}",
                f"D3: placeholder rationales of {r['model']} ({r['representation']})", "posthoc/d3_summary.csv",
                "column n_placeholder")
    sec = [r for r in summ if r["unit"] == "secondary" and r["level"] == "all configurations"][0]
    add("d3x.secondary", f"with a checkable claim {sec['with_checkable_claim']}/{sec['n_rationales']} "
        f"({sec['share_with_checkable_claim']}); consistent {sec['consistent_claims']}/{sec['checkable_claims']} "
        f"({sec['share_consistent']}); FAIL reason absent {sec['fail_reason_absent']}/{sec['fail_with_checkable_stated_reason']} "
        f"({sec['share_fail_reason_absent_among_checkable']})",
        "D3 secondary unit (rationale plus per-metric reasons), all configurations pooled", "posthoc/d3_summary.csv",
        "row unit=secondary, level=all configurations")
    for r in summ:
        if r["unit"] == "primary" and r["level"] == "by configuration" and r["checkable_claims"] not in ("", "0"):
            add(f"d3x.trend.{r['model']}", f"{r['trend_consistent']}/{r['trend_checkable']} (sign only "
                f"{r['trend_consistent_sign_only']}/{r['trend_checkable']}); share consistent with the sign-only trend rule "
                f"{r['share_consistent_trend_sign_only']}",
                f"D3, {r['model']} ({r['representation']}): trend claims consistent (primary rule and sign only)",
                "posthoc/d3_summary.csv", "row unit=primary, level=by configuration")
    add("d3x.config_fixed_at", fixed_at, "time the D3 configuration file was fixed (Europe/Bucharest), before any "
        "rationale was processed by the check", "record kept by the authors when the configuration was fixed (not in this archive)", "clock reading recorded when fixed")

    win = rows(out / "d2_window_labels.csv")
    for f in FAMILIES:
        w = [r for r in win if r["family"] == f]
        mr = [float(r["max_mean_ratio"]) for r in w]
        l5 = [float(r["max_last5_ratio"]) for r in w]
        add(f"d2x.realised.{f}", f"maximum over metrics of the mean ratio {min(mr):.4f} to {max(mr):.4f}; of the last-five "
            f"ratio {min(l5):.4f} to {max(l5):.4f}",
            f"{f}: range over its 20 windows of the realised quantities P6 and P7 read", "posthoc/d2_window_labels.csv",
            "columns max_mean_ratio and max_last5_ratio")
    met = rows(out / "d2_policy_metrics.csv")
    bf = [r for r in met if r["policy"] == "P0" and r["config"] == "bfks"][0]
    add("d2x.bfks_ba_exact", f"{bf['BA_exact']} = 0.7875; published 0.788; the scripts print 0.787 (Python rounding of the "
        "binary value)", "BF+KS balanced accuracy under P0: exact value and rounding; use the published 0.788",
        "posthoc/d2_policy_metrics.csv", "row policy=P0, config=bfks, column BA_exact")
    facts_all = [{"id": f"D-{n0 + i:03d}", **f} for i, f in enumerate(facts, 1)]
    jf, mf = out / "facts-D-addendum.json", out / "FACTS-D-addendum.md"
    for p in (jf, mf):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p.name}")
    jf.write_text(json.dumps(facts_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["# Facts of the post-hoc analyses, addendum (read off the D2 and D3 output tables)", "",
             f"Ids continue after {base[-1]['id']} of FACTS-D.md. Nothing here is recomputed; each value is read from the "
             "file named in its row.", "",
             "| id | value | description | source file | filter or computation | script |", "|---|---|---|---|---|---|"]
    for f in facts_all:
        lines.append("| " + " | ".join(cell(f[k]) for k in ("id", "value", "description", "source_file",
                                                            "filter_or_computation", "script")) + " |")
    mf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {jf.name} and {mf.name}: {len(facts_all)} facts, {facts_all[0]['id']} to {facts_all[-1]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
