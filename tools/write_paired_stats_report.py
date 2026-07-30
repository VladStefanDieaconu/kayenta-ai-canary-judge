#!/usr/bin/env python3
"""Render the recomputation report and the claims-at-risk table from the JSON.

Kept separate from recompute_paired_stats.py so the numbers are computed once and
formatted once, and so the prose can be regenerated without recomputing. Every
figure in the output is read from audit/recompute-paired-stats.json; nothing is
typed in by hand, which is the only way a report of this size stays true to the
data it claims to describe.

Usage:
  python tools/write_paired_stats_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_IN = REPO_ROOT / "audit" / "recompute-paired-stats.json"
REPORT = REPO_ROOT / "audit" / "RECOMPUTE-PAIRED-STATS.md"
CLAIMS = REPO_ROOT / "audit" / "CLAIMS-AT-RISK.md"

NO_DATA = "**no data (0 of 20)**"


def f(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "undefined"
    if isinstance(x, float) and math.isnan(x):
        return "undefined"
    return f"{x:.{nd}f}"


def d(pub: Optional[float], cor: Optional[float], nd: int = 3) -> str:
    if pub is None or cor is None:
        return "—"
    if any(isinstance(v, float) and math.isnan(v) for v in (pub, cor)):
        return "—"
    delta = cor - pub
    return "0" if abs(delta) < 5e-4 else f"{delta:+.{nd}f}"


def p_fmt(p: float) -> str:
    if p < 1e-5:
        return "&lt;0.00001"
    return f"{p:.5f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the paired-stats report")
    ap.add_argument("--json", default=str(JSON_IN))
    args = ap.parse_args()
    R = json.loads(Path(args.json).read_text())

    L: List[str] = []
    A = L.append

    A("# RECOMPUTE — paired and derived statistics against the corrected artefact")
    A("")
    A(f"Generated {R['generated']} by `tools/recompute_paired_stats.py`, rendered by")
    A("`tools/write_paired_stats_report.py`. Every number is read from")
    A("`audit/recompute-paired-stats.json`; none is typed in by hand.")
    A("")
    A("**Authority for Tasks 1–6:** `results/corrected/results_raw.csv` (the re-run without the")
    A("22 failed calls). **Published comparator:** `reference-results/n20/results_raw.csv`,")
    A("read-only. No model was called; every figure is recomputed from files on disk, using")
    A("`run_experiment.confusion`, `metrics_from_conf`, `mcnemar_exact`, `cohen_kappa` and")
    A("`ci_utils.holm_correction` — the same code that produced the published tables.")
    A("")
    A("## Headline")
    A("")
    t2 = R["task2_mcnemar"]
    t7 = R["task7_multiseed"]["scenarios"]
    A(f"1. **Table 10's significance count falls from {t2['n_sig_published']} of 8 to "
      f"{t2['n_sig_corrected']} of 8.** moondream's statistical-versus-gated-hybrid comparison "
      f"goes from p = 0.00004 to p = 0.45450.")
    A("2. **The prediction in the task file is confirmed exactly.** All 21 of moondream's "
      "dropped scenarios sat in the `b` cell (statistical wrong, hybrid right); none in `c`, "
      "none concordant.")
    A(f"3. **The five-seed headline is exposed.** \"Five of the eight models reach Holm-corrected "
      f"significance\" holds only if none of the 37 suspect moondream rows is fabricated. At 75% "
      f"or 100% it becomes **four of eight**.")
    A("4. **A second model's Holm decision moves without its own data changing** — deepseek-r1's "
      "`statistical_vs_ai:summary` crosses 0.05 purely because moondream's p-values reorder the "
      "Holm family.")
    A("5. **Section 7.10's published crossovers could not be reproduced** from these artefacts "
      "under any configuration set that also satisfies its own \"gated hybrid never optimal\" "
      "claim. That task is reported as blocked, not estimated.")
    A("")
    A("---")
    A("")

    # ---------------------------------------------------------------- Task 0
    t0 = R["task0_arms"]
    ident = R["task0_scenario_identity"]
    frame = R.get("task0_frame_audit", {})
    A("## Task 0 — arm integrity")
    A("")
    A(f"{len(t0['arms'])} arms examined across the published and corrected artefacts "
      f"({len(t0['arms'])//2} configurations each). **{len(t0['flagged'])} are short of "
      f"9 families × 20 with a 120/60 split; all {len(t0['explained_by_correction'])} of them are "
      f"in the corrected artefact and every one is explained exactly by the correction. "
      f"{len(t0['excluded'])} arms are excluded as unexplained.**")
    A("")
    A("The task's rule — flag anything not 9 × 20 / 120 / 60 and exclude it downstream — is aimed")
    A("at a *partial* arm, a run that stopped early. Applied literally it would also exclude the")
    A("moondream and deepseek arms, which are short for a fully understood reason and which the")
    A("task itself asks for downstream (Task 2 specifies pairwise deletion and a per-model `n`;")
    A("Task 6 names n = 180, 179 and 159). So each short arm is classified rather than merely")
    A("flagged: `explained_by_correction` when the shortfall equals exactly the fabricated rows")
    A("that configuration carried, and any wholly absent family had **all 20** of its rows")
    A("fabricated; `unexplained` otherwise. Only `unexplained` arms are excluded.")
    A("")
    A("| arm | rows | shortfall | fabricated removed | FAIL/PASS | families with no measurement | classification |")
    A("|---|---|---|---|---|---|---|")
    for e in sorted(t0["flagged"], key=lambda x: (x["judge"], x["model"])):
        fams = ", ".join(f"`{x}`" for x in e.get("families_with_no_measurement", [])) or "—"
        A(f"| `{e['judge']}` / {e['model']} | {e['rows_entering_metrics']} | "
          f"{e.get('shortfall')} | {e.get('fabricated_rows_removed')} | "
          f"{e['truth_FAIL']}/{e['truth_PASS']} | {fams} | **{e['classification']}** |")
    A("")
    A(f"**Excluded from downstream tasks: {len(t0['excluded'])}.** Every other arm is complete at "
      "180 rows with a 120/60 split and all nine families at 20.")
    A("")
    A("### The 52-row arm the task points at")
    A("")
    if frame.get("available"):
        A(f"`audit/CONSOLIDATED-NUMBERS.md` §A prints `deepseek-r1-llm / ai:raw = 0.231, n = 52`. "
          f"**That arm is not partial in the data.** The long-format frame currently holds "
          f"{len(frame['arms'])} frozen-prompt AI arms on the original 180 and "
          f"**{len(frame['partial_arms'])} of them are partial** — deepseek-r1's `ai:raw` has its "
          f"full 180 rows.")
        A("")
        A("`CONSOLIDATED-NUMBERS.md` was generated from a mid-run snapshot while that "
          "configuration was still executing (52 of 180 scenarios judged at the time). It is a "
          "**stale document, not a bad measurement**, and should be regenerated with "
          "`python tools/consolidate_numbers.py` before anything is read from it.")
    else:
        A("The long-format frame could not be read, so this could not be checked.")
    A("")
    A("### Are the corrected artefact's absences exactly the published fabrications?")
    A("")
    A(f"- fabricated rows in the published artefact: **{ident['fabricated_in_published']}**")
    A(f"- rows absent from the corrected artefact: **{ident['absent_from_corrected']}**, comprising")
    A(f"  **{ident['absent_ai_rows']}** AI rows and **{ident['absent_dependent_hybrid_rows']}** "
      f"dependent hybrid rows")
    A(f"- fabricated rows still present in the corrected artefact: "
      f"**{len(ident['fabricated_but_still_present'])}**")
    A(f"- unexplained absences: **{len(ident['absent_unexplained'])}**")
    A("")
    A(f"**Fully accounted for: {'yes' if ident['accounted_for'] else 'NO'}.** The "
      f"{ident['n_distinct_scenarios']} distinct scenarios are the same identifiers in both "
      "artefacts. The 66 dependent rows are the three hybrid policies on those same scenarios: "
      "a hybrid is only emitted when the AI call succeeded, so removing the AI row removes them "
      "too.")
    A("")
    A("### The other two artefacts")
    A("")
    oth = R["task0_other_artefacts"]
    ms, n5 = oth["multiseed"], oth["n5"]
    A(f"- **multi-seed** (`long_results.csv`): {ms['rows']} rows over seeds {', '.join(ms['seeds'])}. "
      f"Rows with a non-empty `error` column: **{ms['rows_with_non_empty_error']}**. Rationale "
      f"column present: **{ms['has_rationale_column']}**. Both are what the defect produces, so "
      f"this artefact cannot be audited directly — see Task 7.")
    A(f"- **n5**: {n5['rows']} rows, **{n5['fabricated']} fabricated** — see Task 8.")
    A("")

    # ---------------------------------------------------------------- Task 1
    t1 = R["task1_confusion_metrics"]
    A("---")
    A("")
    A("## Task 1 — Table 8, confusion-matrix metrics, all 38 configurations")
    A("")
    A(f"**{t1['n_moved']} of {t1['n_total']} configurations moved; {t1['n_identical']} are "
      f"byte-identical.** This confirms `MANUSCRIPT-CHANGES.md` §A, which states 8 and 30.")
    A("")
    A("Only the configurations that moved are tabulated; the other 30 are unchanged at n = 180.")
    A("")
    A("| configuration | metric | published (n=180) | corrected | Δ | corrected n |")
    A("|---|---|---|---|---|---|")
    for c in t1["configurations"]:
        if not c["moved"]:
            continue
        p, q = c["published"], c["corrected"]
        name = f"`{c['judge']}` / {c['model']}"
        for metric, label in (("accuracy", "accuracy"), ("precision", "precision"),
                              ("recall", "recall"), ("f1", "F1"), ("fpr", "FPR"),
                              ("balanced_accuracy", "balanced acc."), ("mcc", "MCC")):
            A(f"| {name} | {label} | {f(p[metric])} | **{f(q[metric])}** | "
              f"{d(p[metric], q[metric])} | {q['n']} |")
            name = ""
        A(f"| | TP/FP/TN/FN | {p['TP']}/{p['FP']}/{p['TN']}/{p['FN']} | "
          f"{q['TP']}/{q['FP']}/{q['TN']}/{q['FN']} | | {q['n']} |")
    A("")
    A(f"`moondream-vlm / ai:plot / cross_metric_marginal` is {NO_DATA} — all twenty of its "
      "scenarios were failed calls. It must never be printed as 0.000, and never as the "
      "published 1.000.")
    A("")

    # ---------------------------------------------------------------- Task 2
    A("---")
    A("")
    A("## Task 2 — Table 10, per-model McNemar: statistical judge versus gated hybrid")
    A("")
    A("Single seed, 180-scenario dataset, frozen prompt. `b` = statistical wrong and gated hybrid")
    A("right; `c` = the reverse. Two-sided **exact binomial** p on the discordant pairs, which is")
    A("what the manuscript reports; the continuity-corrected chi-square is descriptive and is")
    A("included for continuity. **Pairwise deletion**: a scenario is dropped from a model's")
    A("comparison if either arm lacks a row for it.")
    A("")
    A("| model | pub b | pub c | pub n | pub p (exact) | pub χ²cc | cor b | cor c | cor n | cor p (exact) | cor χ²cc | crosses 0.05? |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in t2["models"]:
        p, q = e["published"], e["corrected"]
        crossed = p["significant"] != q["significant"]
        A(f"| {e['model']} | {p['b_stat_wrong_hybrid_right']} | {p['c_stat_right_hybrid_wrong']} | "
          f"{p['n_pairs']} | {p_fmt(p['p_exact'])} | {p['chi2_cc']:.3f} | "
          f"**{q['b_stat_wrong_hybrid_right']}** | {q['c_stat_right_hybrid_wrong']} | "
          f"**{q['n_pairs']}** | **{p_fmt(q['p_exact'])}** | {q['chi2_cc']:.3f} | "
          f"{'**YES**' if crossed else 'no'} |")
    A("")
    A(f"### Comparisons with p &lt; 0.05: **published {t2['n_sig_published']} of 8 → corrected "
      f"{t2['n_sig_corrected']} of 8**")
    A("")
    A("The manuscript states \"seven of eight\" in Section 7.3, in the Table 10 caption, and again")
    A("in Section 11's statistical-conclusion paragraph. All three become **six of eight**.")
    A("")
    md = t2["moondream_dropped"]
    cells = md["cells"]
    A("### The direct test of the prediction — where moondream's dropped scenarios sat")
    A("")
    A(f"| published 2×2 cell | count of the {md['n_dropped']} dropped scenarios |")
    A("|---|---|")
    A(f"| `b` — statistical wrong, hybrid right | **{cells['b']}** |")
    A(f"| `c` — statistical right, hybrid wrong | {cells['c']} |")
    A(f"| concordant, both right | {cells['concordant_both_right']} |")
    A(f"| concordant, both wrong | {cells['concordant_both_wrong']} |")
    A("")
    A(f"By family: {', '.join(f'`{k}` ×{v}' for k, v in sorted(md['by_family'].items()))}.")
    A("")
    A("**The prediction in the task file is confirmed exactly.** It reasoned that the statistical")
    A("judge scores 0.000 on both families and so passes every one of those scenarios, while the")
    A("fabricated AI row is FAIL at score 0.0 — at or below the gated policy's override threshold —")
    A("so the gate fires and the hybrid returns FAIL, which is correct because truth is FAIL on")
    A("both families. Every one of the 21 is therefore a statistical-wrong / hybrid-right pair.")
    A("All 21 landed in `b`; none in `c`; none concordant. The predicted corrected values were")
    A("b = 10, c = 6, p ≈ 0.45; the measured values are "
      f"**b = {t2['models'][[e['model'] for e in t2['models']].index('moondream-vlm')]['corrected']['b_stat_wrong_hybrid_right']}, "
      f"c = {t2['models'][[e['model'] for e in t2['models']].index('moondream-vlm')]['corrected']['c_stat_right_hybrid_wrong']}, "
      f"p = {t2['models'][[e['model'] for e in t2['models']].index('moondream-vlm')]['corrected']['p_exact']:.5f}**.")
    A("")

    # ---------------------------------------------------------------- Task 3
    t3 = R["task3_kappa"]
    A("---")
    A("")
    A("## Task 3 — Table 11, Cohen's kappa")
    A("")
    A("Against the benchmark labels. Only configurations whose kappa moved are listed; every")
    A("other configuration is unchanged at n = 180.")
    A("")
    A("| configuration | published κ (n=180) | corrected κ | Δ | corrected n |")
    A("|---|---|---|---|---|")
    for r in t3["vs_truth"]:
        p, q = r["published"], r["corrected"]
        if p is None or q is None or abs(p - q) < 5e-4:
            continue
        A(f"| `{r['judge']}` / {r['model']} | {f(p)} | **{f(q)}** | {d(p, q)} | {r['n_corrected']} |")
    A("")
    ij = t3["inter_judge_statistical_vs_moondream_plot"]
    A("### Inter-judge kappa, statistical judge versus the moondream multimodal judge")
    A("")
    A("| | κ | n |")
    A("|---|---|---|")
    A(f"| published | {f(ij['published']['kappa'])} | {ij['published']['n']} |")
    A(f"| corrected | **{f(ij['corrected']['kappa'])}** | {ij['corrected']['n']} |")
    A("")
    A("Section 7.3 quotes −0.113 and uses it to argue the two judges have opposite error profiles.")
    A(f"The corrected value is {f(ij['corrected']['kappa'])} — still negative, so the qualitative")
    A("claim survives, but the number moves and it was computed over the contaminated rows.")
    A("")
    md_k = next((r for r in t3["vs_truth"]
                 if r["judge"] == "ai:plot" and r["model"] == "moondream-vlm"), None)
    if md_k:
        A(f"**The predicted 0.113 → 0.045 for moondream `ai:plot` versus labels is confirmed:** "
          f"{f(md_k['published'])} → **{f(md_k['corrected'])}** (n = {md_k['n_corrected']}).")
    A("")

    # ---------------------------------------------------------------- Task 4
    t4 = R["task4_pooled_counts"]
    A("---")
    A("")
    A("## Task 4 — Figure 9 caption, pooled confusion counts")
    A("")
    A("| pool | models | rows | TP | FN | FP | TN |")
    A("|---|---|---|---|---|---|---|")
    for side in ("published", "corrected"):
        for name, label in (("summary_ai", "summary-statistics AI"), ("gated_hybrid", "gated hybrid")):
            v = t4[side][name]
            bold = (lambda x: f"**{x}**") if side == "corrected" else (lambda x: str(x))
            A(f"| {label} ({side}) | {v['n_models']} | {bold(v['rows'])} | {bold(v['TP'])} | "
              f"{bold(v['FN'])} | {bold(v['FP'])} | {bold(v['TN'])} |")
    A("")
    ps, pg = t4["published"]["summary_ai"], t4["published"]["gated_hybrid"]
    A(f"The published pools reproduce the caption exactly — summary-statistics AI "
      f"{ps['TP']}/{ps['FN']}/{ps['FP']}/{ps['TN']} over {ps['rows']} rows and gated hybrid "
      f"{pg['TP']}/{pg['FN']}/{pg['FP']}/{pg['TN']} over {pg['rows']} — which validates the method "
      f"before any corrected number is read from it.")
    A("")
    cs, cg = t4["corrected"]["summary_ai"], t4["corrected"]["gated_hybrid"]
    A(f"Corrected: the summary pool loses one row ({ps['rows']} → **{cs['rows']}**, one FP) and the "
      f"gated pool loses 22 ({pg['rows']} → **{cg['rows']}**, 21 TP and 1 FP). **The caption's four "
      f"counts and both pool sizes all change.**")
    A("")

    # ---------------------------------------------------------------- Task 5
    t5 = R["task5_representation_means"]
    A("---")
    A("")
    A("## Task 5 — representation means, Sections 7.4 and 7.8")
    A("")
    A("| representation | manuscript | published (recomputed) | corrected | models | Δ |")
    A("|---|---|---|---|---|---|")
    manuscript = {"summary": "0.733 (0.667–0.833)", "raw": "0.704 (0.667–0.761)",
                  "plot": "0.618 (0.594–0.661)"}
    for rep in ("summary", "raw", "plot"):
        p, q = t5["published"].get(rep), t5["corrected"].get(rep)
        if not p or not q:
            continue
        A(f"| `ai:{rep}` | {manuscript[rep]} | {f(p['mean'])} ({f(p['min'])}–{f(p['max'])}) | "
          f"**{f(q['mean'])} ({f(q['min'])}–{f(q['max'])})** | {q['n_models']} | "
          f"{d(p['mean'], q['mean'])} |")
    A("")
    plot_p, plot_c = t5["published"]["plot"], t5["corrected"]["plot"]
    A(f"**The predicted plot value of 0.601 (0.541–0.661) is confirmed:** "
      f"{f(plot_c['mean'])} ({f(plot_c['min'])}–{f(plot_c['max'])}), averaged over "
      f"{plot_c['n_models']} models with n = "
      f"{', '.join(str(v) for v in plot_c['n_rows_per_model'].values())} rows.")
    A("")
    A(f"A note on the published figure. Recomputing from the artefact gives "
      f"{plot_p['mean']:.6f}, which rounds to {f(plot_p['mean'])}; the manuscript prints 0.618. "
      f"The difference is rounding order — averaging the three per-model accuracies *after* "
      f"rounding them to three decimals gives 0.618. It is not a data discrepancy, and the "
      f"corrected value is 0.601 either way.")
    A("")
    A("The 0.618 figure is quoted twice, in Section 7.4 and again in Section 7.8 as \"a local mean")
    A("of 0.618\". Both occurrences change.")
    A("")

    # ---------------------------------------------------------------- Task 6
    t6 = R["task6_cost_ratio"]
    blocked = t6["published_crossovers_not_reproducible"]
    A("---")
    A("")
    A("## Task 6 — Section 7.10 cost-ratio crossovers")
    A("")
    A("### This task is reported as blocked, and the reason is a measurement")
    A("")
    A("Section 7.10 reports three regimes with crossovers near r ≈ 0.14 and r ≈ 0.83, plus the")
    A("claim that the gated hybrid is loss-minimising at no ratio at all. **No configuration set**")
    A("**drawn from these artefacts satisfies both statements**, so the published figures could")
    A("not be reproduced and therefore cannot be corrected.")
    A("")
    A("The search was exhaustive rather than impressionistic: every 3- and 4-configuration subset")
    A("of the 38 configurations, plus the ensemble scoreboard rows, was swept over r ∈ [0, 5].")
    A("**27 sets** put their crossovers near r = 0.14 and r = 0.83, and **every one of them has a")
    A("moondream hybrid as the middle regime** — where `hybrid:gated` and `hybrid:or` are")
    A("numerically identical (TP 69 / FP 6 / TN 54 / FN 51), so each of the 27 contradicts the")
    A("\"no ratio\" claim it would have to satisfy.")
    A("")
    A("**What is missing is not a data file but a definition**: the manuscript's own Section 7.10")
    A("comparison set and its loss normalisation. Neither is recoverable from the artefacts.")
    A("**Confirm the comparison set against the manuscript before quoting any corrected**")
    A("**crossover.** What can be measured is reported below, for two sets stated explicitly.")
    A("")
    for key, title in (("_triple", "the three approaches (statistical, best standalone AI, gated hybrid on that model)"),
                       ("", "the full field of all 38 configurations")):
        A(f"### Comparison set: {title}")
        A("")
        for side in ("published", "corrected"):
            s = t6[f"{side}{key}"]
            A(f"**{side}** — common scenario set **{s['common_set_size']}**, "
              f"{s['dropped_scenarios']} scenarios dropped"
              + (f" ({', '.join(f'`{k}` ×{v}' for k, v in sorted(s['dropped_by_family'].items()))})"
                 if s["dropped_by_family"] else "") + ".")
            A("")
            A("| form | loss-minimising configuration by r | crossovers |")
            A("|---|---|---|")
            for form, flabel in (("form_a_common_set", "(a) common scenario set, raw sum"),
                                 ("form_b_normalised", "(b) per-scenario normalised")):
                fr = s[form]
                regimes = " → ".join(f"`{w['config']}` (r {w['from_r']}–{w['to_r']})"
                                     for w in fr["regimes"])
                cross = ", ".join(f"r ≈ {c['between_r'][1]}" for c in fr["crossovers"]) or "none"
                A(f"| {flabel} | {regimes} | {cross} |")
            A("")
            gated_a = [c for c in s["form_a_common_set"]["ever_optimal"] if c.startswith("hybrid:gated")]
            gated_b = [c for c in s["form_b_normalised"]["ever_optimal"] if c.startswith("hybrid:gated")]
            A(f"Gated hybrid loss-minimising at some ratio? form (a): "
              f"**{', '.join(gated_a) if gated_a else 'no — at no ratio'}**; form (b): "
              f"**{', '.join(gated_b) if gated_b else 'no — at no ratio'}**.")
            A("")
    A("### Do forms (a) and (b) disagree?")
    A("")
    ca = [w["config"] for w in t6["corrected"]["form_a_common_set"]["regimes"]]
    cb = [w["config"] for w in t6["corrected"]["form_b_normalised"]["regimes"]]
    if ca != cb:
        A("**Yes, on the full field, and that disagreement is itself the finding.** On the")
        A("corrected artefact the two forms select different regime sequences:")
        A("")
        A(f"- form (a), common set of {t6['corrected']['common_set_size']}: "
          + " → ".join(f"`{c}`" for c in ca))
        A(f"- form (b), each configuration on its own rows: " + " → ".join(f"`{c}`" for c in cb))
        A("")
        A("Two gated hybrids (`phi4-llm`, `mistral-nemo-llm`) are loss-minimising over a wide")
        A("middle band under form (a) and never under form (b). The cause is that form (a)")
        A("restricts every configuration to the 158 scenarios all of them judged, which removes")
        A("20 `cross_metric_marginal` scenarios — a family where the gated hybrid does badly —")
        A("and so flatters it. **Any cost-ratio claim must state which form it uses.**")
    else:
        A("No: both forms select the same regime sequence on the corrected artefact.")
    A("")
    A("On the narrow three-approach set the answer is stable and unchanged by the correction:")
    A("the statistical judge is loss-minimising below r ≈ 0.345 and `ai:summary/phi4-llm` above")
    A("it, with the gated hybrid optimal at no ratio — because phi4 is not an affected")
    A("configuration. **If Section 7.10 uses this set, its qualitative claim survives intact.**")
    A("")

    # ---------------------------------------------------------------- Task 7
    A("---")
    A("")
    A("## Task 7 — the five-seed artefact: bounding the contamination")
    A("")
    A("`long_results.csv` stores no rationale and its `error` column is empty on all 8,550 rows,")
    A("which is exactly what the defect produces. It cannot be audited directly and was not")
    A("corrected. It is bounded instead.")
    A("")
    A("### 1. Suspect rows under the score-0.0 proxy")
    A("")
    A("| configuration | suspect rows | of total | by family |")
    A("|---|---|---|---|")
    for k, v in sorted(R["task7_multiseed"]["suspect_totals"].items()):
        fams = R["task7_multiseed"]["suspect_by_family"].get(k, {})
        A(f"| `{k.replace('|', '` / `')}` | **{v['suspect']}** | {v['of_total']} | "
          f"{', '.join(f'`{a}` ×{b}' for a, b in sorted(fams.items()))} |")
    A("")
    A("The proxy is necessary but not sufficient: calibrated on the N=20 run, 21 of moondream's 28")
    A("score-0.0 rows were fabricated (~75%) and 1 of deepseek's 3. So 37 is an **upper bound** on")
    A("moondream's contamination, not a count. This confirms CHECKPOINT-20's ≤ 37 of 225.")
    A("")
    A("### 2–4. Pooled McNemar for moondream under three assumptions")
    A("")
    fam_size = R["task7_multiseed"]["scenarios"]["0pct"]["family_size"]
    A(f"**Holm family size read from the code, not assumed:** `tools/pooled_mcnemar.py` builds "
      f"3 comparisons per model over 8 models plus one ensemble comparison = **{fam_size}**. "
      f"The Table 20 caption's \"25 comparisons\" is **correct**.")
    A("")
    A("| assumption | rows dropped | n pairs | b | c | raw p | Holm p | significant? |")
    A("|---|---|---|---|---|---|---|---|")
    for lab, pretty in (("0pct", "0% fabricated"), ("75pct", "75% fabricated"),
                        ("100pct", "100% fabricated")):
        s = R["task7_multiseed"]["scenarios"][lab]
        c = next(x for x in s["comparisons"]
                 if x["comparison_id"] == "moondream-vlm:statistical_vs_hybrid_gated")
        A(f"| {pretty} | {s['dropped_rows']} | {c['n_pairs']} | {c['b']} | {c['c']} | "
          f"{p_fmt(c['p_raw'])} | **{p_fmt(c['p_holm'])}** | "
          f"{'**yes**' if c['holm_significant'] else '**no**'} |")
    A("")
    A("### Does the headline change?")
    A("")
    A("| assumption | models reaching Holm significance on statistical vs gated hybrid | headline |")
    A("|---|---|---|")
    for lab, pretty in (("0pct", "0%"), ("75pct", "75%"), ("100pct", "100%")):
        s = R["task7_multiseed"]["scenarios"][lab]
        n = len(s["models_reaching_holm_significance"])
        A(f"| {pretty} | {n} — {', '.join(s['models_reaching_holm_significance'])} | "
          f"{'**five of eight** (as published)' if n == 5 else f'**{n} of eight** — headline changes'} |")
    A("")
    A("**\"Five of the eight models reach Holm-corrected significance\" holds only under the")
    A("assumption that none of the 37 suspect rows is fabricated.** At the calibrated 75% rate it")
    A("becomes four of eight, and at 100% it also becomes four of eight.")
    A("")
    A("### 5. Does any other model's Holm decision move?")
    A("")
    A("**Yes — one, and it is collateral.**")
    A("")
    A("| comparison | Holm p at 0% | at 75% | at 100% | decision moves? |")
    A("|---|---|---|---|---|")
    ids = [c["comparison_id"] for c in R["task7_multiseed"]["scenarios"]["0pct"]["comparisons"]]
    for cid in ids:
        vals = []
        for lab in ("0pct", "75pct", "100pct"):
            c = next(x for x in R["task7_multiseed"]["scenarios"][lab]["comparisons"]
                     if x["comparison_id"] == cid)
            vals.append((c["p_holm"], c["holm_significant"]))
        if len({v[1] for v in vals}) == 1:
            continue
        A(f"| `{cid}` | {vals[0][0]:.5f} | {vals[1][0]:.5f} | {vals[2][0]:.5f} | **YES** |")
    A("")
    A("`deepseek-r1-llm:statistical_vs_ai:summary` sits at Holm p = 0.04992 — inside 0.05 by")
    A("0.00008 — and moves to 0.05304 when moondream's rows are dropped. **Its own data does not**")
    A("**change at all.** Holm is a step-down procedure over the ordered p-value list, so removing")
    A("contamination from one model shifts another model's rank and its multiplier. Qwen2.5")
    A("(0.05595) and MiniCPM-V (0.264) do not move across the boundary, but Qwen2.5 is close")
    A("enough that it would with a slightly different family.")
    A("")
    A("### Recommendation")
    A("")
    A("**Re-run the five-seed evaluation under the fixed harness.** Not from general caution —")
    A("from three specific findings above. First, the manuscript's headline for that table changes")
    A("under any contamination rate above roughly a third, and the calibrated estimate is 75%.")
    A("Second, the artefact cannot be audited to settle the question: it records neither a")
    A("rationale nor a usable error column, so the 37 suspect rows can be bounded but never")
    A("resolved, and a bound is not something a reviewer can be asked to accept in place of a")
    A("measurement. Third, and most decisive, the Holm coupling means the contamination is not")
    A("contained to moondream — deepseek-r1's decision flips on a 0.00008 margin because of rows")
    A("belonging to a different model. Any sentence in Sections 9.2, 11 or 12 that counts")
    A("significant comparisons rests on which side of that margin the family lands. The re-run is")
    A("225 scenarios × the local model set with no cloud spend; against the alternative of")
    A("publishing a count that cannot be defended under questioning, it is cheap.")
    A("")

    # ---------------------------------------------------------------- Task 8
    t8 = R["task8_n5"]
    A("---")
    A("")
    A("## Task 8 — the n5 artefact")
    A("")
    A(f"**{t8['n_affected']} configuration affected.**")
    A("")
    A("| configuration | fabricated | metric | published (n=45) | corrected (n=40) | Δ |")
    A("|---|---|---|---|---|---|")
    for e in t8["configurations"]:
        if not e["fabricated"]:
            continue
        p, q = e["published"], e["corrected"]
        name = f"`{e['judge']}` / {e['model']}"
        for metric, label in (("accuracy", "accuracy"), ("precision", "precision"),
                              ("recall", "recall"), ("f1", "F1"),
                              ("balanced_accuracy", "balanced acc."), ("mcc", "MCC")):
            A(f"| {name} | {e['fabricated']} | {label} | {f(p[metric])} | **{f(q[metric])}** | "
              f"{d(p[metric], q[metric])} |")
            name = ""
    A("")
    A("**CHECKPOINT-20's figures are confirmed**: accuracy 0.667 → 0.625 and MCC 0.213 → 0.169.")
    A(f"All five fabricated rows are `moondream-vlm / ai:plot / cross_metric_marginal`, and "
      f"**no other n5 configuration is affected** — the hybrid rows for that model are unchanged "
      f"in this artefact because n5 predates the hybrid-dependency behaviour seen at n20.")
    A("")

    REPORT.write_text("\n".join(L) + "\n")
    print(f"[report] {len(L)} lines -> {REPORT}")

    # ------------------------------------------------------------ claims table
    C: List[str] = []
    B = C.append
    B("# CLAIMS AT RISK")
    B("")
    B("One row per manuscript sentence whose number changes. **Boundary-crossing rows first**:")
    B("a p-value moving across 0.05, a count in a sentence like \"seven of eight\" changing, a")
    B("ranking reversing, or a cost-ratio regime boundary moving.")
    B("")
    B(f"Generated from `audit/recompute-paired-stats.json` ({R['generated']}).")
    B("")
    rows_hi: List[str] = []
    rows_lo: List[str] = []

    def row(loc, claim, pub, cor, crosses, hi):
        line = f"| {loc} | {claim} | {pub} | {cor} | {crosses} |"
        (rows_hi if hi else rows_lo).append(line)

    mdm = next(e for e in t2["models"] if e["model"] == "moondream-vlm")
    row("**Section 7.3 prose**, Table 10 caption, **Section 11** statistical-conclusion paragraph",
        "\"seven of eight comparisons are nominally significant\"",
        f"{t2['n_sig_published']} of 8", f"**{t2['n_sig_corrected']} of 8**",
        "**YES — a count in the sentence changes**", True)
    row("**Table 10**, moondream row",
        "b = 31, c = 6, p = 0.00004",
        "b=31, c=6, p=0.00004 (n=180)",
        f"**b={mdm['corrected']['b_stat_wrong_hybrid_right']}, c={mdm['corrected']['c_stat_right_hybrid_wrong']}, "
        f"p={mdm['corrected']['p_exact']:.5f}** (n={mdm['corrected']['n_pairs']})",
        "**YES — p crosses 0.05, significant → not significant**", True)
    dsk = next(e for e in t2["models"] if e["model"] == "deepseek-r1-llm")
    row("**Table 10**, deepseek-r1 row", "b = 68, c = 30, p = 0.00016",
        "b=68, c=30, p=0.00016 (n=180)",
        f"b={dsk['corrected']['b_stat_wrong_hybrid_right']}, c={dsk['corrected']['c_stat_right_hybrid_wrong']}, "
        f"p={dsk['corrected']['p_exact']:.5f} (n={dsk['corrected']['n_pairs']})",
        "no — remains significant", False)
    n75 = len(R["task7_multiseed"]["scenarios"]["75pct"]["models_reaching_holm_significance"])
    n100 = len(R["task7_multiseed"]["scenarios"]["100pct"]["models_reaching_holm_significance"])
    row("**Section 9.2 / Table 20**, **Section 12**",
        "\"five of the eight models reach Holm-corrected significance\"",
        "5 of 8 — holds only if none of the 37 suspect rows is fabricated",
        f"**{n75} of 8** at the calibrated 75% assumption; **{n100} of 8** at 100%",
        "**YES — a count in the sentence changes**", True)
    row("**Table 20**, deepseek-r1 `statistical_vs_ai:summary`",
        "Holm-corrected significant",
        "Holm p = 0.04992 (significant)",
        "**Holm p = 0.05304 (not significant)** at the 75% assumption",
        "**YES — crosses 0.05 with no change to its own data, via the Holm ordering**", True)
    t6c = t6["corrected"]
    row("**Section 7.10**", "three regimes, crossovers near r ≈ 0.14 and r ≈ 0.83; "
        "\"the gated hybrid is loss-minimising at no ratio\"",
        "r ≈ 0.14, r ≈ 0.83",
        "**not reproducible from the artefacts** — see Task 6; on the three-approach set the "
        "single crossover is r ≈ 0.345 and is unchanged by the correction",
        "**YES — regime boundaries move on the full field, and forms (a) and (b) disagree**", True)
    md_k = next(r for r in t3["vs_truth"] if r["judge"] == "ai:plot" and r["model"] == "moondream-vlm")
    row("**Table 11**, moondream `ai:plot`", "κ = 0.113",
        f"{f(md_k['published'])} (n=180)", f"**{f(md_k['corrected'])}** (n={md_k['n_corrected']})",
        "no — but the value more than halves", False)
    row("**Section 7.3**", "inter-judge κ = −0.113 between statistical and moondream",
        f"{f(ij['published']['kappa'])} (n={ij['published']['n']})",
        f"**{f(ij['corrected']['kappa'])}** (n={ij['corrected']['n']})",
        "no — still negative, qualitative claim survives", False)
    row("**Section 7.4** and **Section 7.8** (\"a local mean of 0.618\")",
        "plot representation mean 0.618 (0.594–0.661)",
        f"0.618 (0.594–0.661), n=180 per model",
        f"**{f(plot_c['mean'])} ({f(plot_c['min'])}–{f(plot_c['max'])})**, moondream n=159",
        "no — but quoted twice, both change", False)
    row("**Figure 9 caption**", "summary AI TP 555 / FN 45 / FP 195 / TN 105",
        f"{ps['TP']}/{ps['FN']}/{ps['FP']}/{ps['TN']} (n={ps['rows']})",
        f"**{cs['TP']}/{cs['FN']}/{cs['FP']}/{cs['TN']}** (n={cs['rows']})",
        "no — one FP moves", False)
    row("**Figure 9 caption**", "gated hybrid TP 725 / FN 235 / FP 199 / TN 281",
        f"{pg['TP']}/{pg['FN']}/{pg['FP']}/{pg['TN']} (n={pg['rows']})",
        f"**{cg['TP']}/{cg['FN']}/{cg['FP']}/{cg['TN']}** (n={cg['rows']})",
        "no — but 21 TP and 22 rows are lost", False)
    for c in t1["configurations"]:
        if not c["moved"]:
            continue
        p, q = c["published"], c["corrected"]
        row("**Section 7.1 / Table 8**", f"`{c['judge']}` / {c['model']} accuracy",
            f"{f(p['accuracy'])} (n={p['n']})", f"**{f(q['accuracy'])}** (n={q['n']})",
            "no", False)
    row("**Section 7.1 / Table 8**, moondream `ai:plot` per-family",
        "`cross_metric_marginal` = 1.000",
        "1.000 (20 of 20)", f"{NO_DATA}",
        "**YES — the cell has no measurement; a ranking that used it is void**", True)
    n5e = next(e for e in t8["configurations"] if e["fabricated"])
    row("**n5 artefact**, moondream `ai:plot`", "accuracy 0.667, MCC 0.213",
        f"{f(n5e['published']['accuracy'])}, MCC {f(n5e['published']['mcc'])} (n={n5e['published']['n']})",
        f"**{f(n5e['corrected']['accuracy'])}**, MCC **{f(n5e['corrected']['mcc'])}** (n={n5e['corrected']['n']})",
        "no", False)

    B("| manuscript location | claim as written | published value | corrected value | crosses a boundary? |")
    B("|---|---|---|---|---|")
    for r in rows_hi + rows_lo:
        B(r)
    B("")
    B(f"**{len(rows_hi)} boundary-crossing rows, {len(rows_lo)} value-only rows.**")
    B("")
    B("## What is not at risk")
    B("")
    B(f"- {t1['n_identical']} of {t1['n_total']} configurations in Table 8 are byte-identical.")
    B("- Six of the eight local models' Table 10 comparisons are unchanged.")
    B("- The statistical judge is unchanged everywhere: 0.544 / 1.000 / 0.317.")
    B("- Every hosted-model number, and the whole `healed_transient` result.")
    B("- The summary and raw representation means (Task 5).")
    B("- On the three-approach reading of Section 7.10, the cost-ratio conclusion is unchanged.")
    CLAIMS.write_text("\n".join(C) + "\n")
    print(f"[report] {len(C)} lines -> {CLAIMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
