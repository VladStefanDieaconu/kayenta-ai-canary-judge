#!/usr/bin/env python3
"""Recompute every paired and derived statistic against the corrected artefact.

`audit/MANUSCRIPT-CHANGES.md` covers the confusion-matrix tables and the per-family
cells. It does not cover the statistics computed from per-scenario *correctness* --
the per-model McNemar tests, Cohen's kappa, the pooled multi-seed McNemar, the
pooled counts in a figure caption, or the cost-ratio crossovers. Those are the
manuscript's inferential claims, and they are what this script recomputes.

Nothing here calls a model and nothing writes to a published artefact. Every
number comes from a file on disk, and the metric code is imported from
run_experiment and ci_utils rather than reimplemented, so the corrected tables
cannot drift from the published ones by arithmetic.

Usage:
  python tools/recompute_paired_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import ci_utils as ci  # noqa: E402
import eval_dataset as ed  # noqa: E402
import run_experiment as exp  # noqa: E402

CORRECTED = REPO_ROOT / "results" / "corrected" / "results_raw.csv"
PUBLISHED = REPO_ROOT / "reference-results" / "n20" / "results_raw.csv"
MULTISEED = REPO_ROOT / "reference-results" / "n20" / "agg" / "long_results.csv"
N5 = REPO_ROOT / "reference-results" / "n5" / "results_raw.csv"

# The signature of a failed call recorded as a verdict in the published artefact:
# _error_result writes this rationale, and the error column is empty.
FABRICATED_PREFIX = "AI judge error"

EXPECTED_FAMILIES = list(ed.FAMILIES)
EXPECTED_PER_FAMILY = 20
EXPECTED_FAIL = 120
EXPECTED_PASS = 60


# ---------------------------------------------------------------- loading

def load_wide(path: Path) -> List[Dict[str, Any]]:
    """A wide results_raw.csv as normalised rows.

    Rows whose `error` column is non-empty are dropped, which is the same rule
    results_schema.load() applies: a row with an error carries no judgement. The
    corrected artefact has none (build_corrected_results already omitted them);
    the published one has none either, which is precisely the defect -- its 22
    failed calls carry an empty error column and a FAIL verdict.
    """
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if r.get("error"):
            continue
        out.append({
            "scenario": r["scenario"], "family": r["family"], "truth": r["truth"],
            "judge": r["judge"], "model": r["model"], "verdict": r["verdict"],
            "score": float(r["score"]) if r["score"] not in ("", None) else None,
            "correct": int(r["correct"]) if r["correct"] not in ("", None) else None,
            "rationale": r.get("rationale", ""),
        })
    return out


def is_fabricated(row: Dict[str, Any]) -> bool:
    return str(row.get("rationale", "")).startswith(FABRICATED_PREFIX)


def by_config(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    d: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[(r["judge"], r["model"])].append(r)
    return d


def correctness_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    return {r["scenario"]: bool(r["correct"]) for r in rows}


# ---------------------------------------------------------------- metrics

def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Confusion metrics, with MCC and balanced accuracy left undefined where they are.

    metrics_from_conf is imported from run_experiment so accuracy, precision,
    recall, F1 and FPR are computed by exactly the code that produced the
    published tables. Balanced accuracy and MCC are added here because that
    function does not provide them, and MCC returns NaN rather than 0.0 when a
    row or column of the matrix is empty -- on a single-label slice the
    coefficient is undefined, not "no better than chance".
    """
    c = exp.confusion(list(rows))
    m = dict(exp.metrics_from_conf(c))
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    n = tp + fp + tn + fn
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    rec = m["recall"] if (tp + fn) else float("nan")
    m["balanced_accuracy"] = (rec + spec) / 2 if not (math.isnan(rec) or math.isnan(spec)) else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m["mcc"] = ((tp * tn - fp * fn) / den) if den else float("nan")
    m.update({"n": n, "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return m


def kappa_vs_truth(rows: Sequence[Dict[str, Any]]) -> float:
    return exp.cohen_kappa([r["truth"] for r in rows], [r["verdict"] for r in rows])


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "undefined"
    return f"{x:.{nd}f}"


def delta(pub: Optional[float], cor: Optional[float], nd: int = 3) -> str:
    if pub is None or cor is None:
        return "—"
    if any(isinstance(v, float) and math.isnan(v) for v in (pub, cor)):
        return "—"
    d = cor - pub
    return "0" if abs(d) < 1e-9 else f"{d:+.{nd}f}"


# ---------------------------------------------------------------- Task 0

def task0(pub_rows, cor_rows) -> Dict[str, Any]:
    """Arm integrity, distinguishing two very different kinds of incompleteness.

    The task's rule -- flag anything that is not 9 families x 20 with a 120/60
    split, and exclude it downstream -- is aimed at a *partial* arm: a run that
    stopped early, whose numbers are not a measurement of anything. The example
    given is a 52-row `ai:raw` arm.

    Applied literally it would also exclude the moondream and deepseek arms, whose
    row counts are short for a fully understood reason: the correction removed
    exactly the failed calls. Excluding those would remove the very configurations
    this whole exercise exists to re-measure, and the task itself asks for them
    downstream (Task 2 specifies pairwise deletion and a per-model n; Task 6 names
    n = 180, 179 and 159 explicitly).

    So an arm short of 180 is classified, not merely flagged:
      * `explained_by_correction` -- the shortfall equals exactly the number of
        fabricated rows that configuration carried in the published artefact, and
        the truth-label shortfall matches those rows' families. Used downstream,
        with its n stated everywhere.
      * `unexplained` -- anything else. Excluded downstream, listed as excluded.
    """
    out: Dict[str, Any] = {"arms": {}, "flagged": [], "excluded": [],
                           "explained_by_correction": []}

    # How many fabricated rows each configuration carried, so a shortfall can be
    # checked against a number rather than accepted because it looks plausible.
    fab_by_cfg: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in pub_rows:
        if is_fabricated(r):
            fab_by_cfg[(r["judge"], r["model"])].append(r)
    # A fabricated ai row also removes that scenario's hybrid rows for the same
    # model, because the hybrid is only emitted when the AI call succeeded.
    fab_scenarios_by_model: Dict[str, set] = defaultdict(set)
    for (j, m), rs_ in fab_by_cfg.items():
        for r in rs_:
            fab_scenarios_by_model[m].add(r["scenario"])

    for label, rows in (("published", pub_rows), ("corrected", cor_rows)):
        cfgs = by_config(rows)
        for (judge, model), rs_ in sorted(cfgs.items()):
            fam = defaultdict(int)
            for r in rs_:
                fam[r["family"]] += 1
            truths = defaultdict(int)
            for r in rs_:
                truths[r["truth"]] += 1
            fabricated = sum(1 for r in rs_ if is_fabricated(r))
            key = f"{label}|{judge}|{model}"
            missing_families = [f for f in EXPECTED_FAMILIES if fam.get(f, 0) == 0]
            short_families = {f: fam.get(f, 0) for f in EXPECTED_FAMILIES
                              if 0 < fam.get(f, 0) < EXPECTED_PER_FAMILY}
            ok = (len(rs_) == 180 and not missing_families and not short_families
                  and truths.get("FAIL") == EXPECTED_FAIL and truths.get("PASS") == EXPECTED_PASS)
            entry = {
                "source": label, "judge": judge, "model": model,
                "rows_entering_metrics": len(rs_),
                "fabricated_rows_present": fabricated,
                "families_present": len(fam), "per_family": dict(fam),
                "truth_FAIL": truths.get("FAIL", 0), "truth_PASS": truths.get("PASS", 0),
                "complete_9x20_120_60": ok,
                "missing_families": missing_families,
                "short_families": short_families,
            }
            if not ok:
                shortfall = 180 - len(rs_)
                if judge.startswith("hybrid"):
                    expected = len(fab_scenarios_by_model.get(model, set()))
                else:
                    expected = len(fab_by_cfg.get((judge, model), []))
                # A wholly absent family is explained too, provided every one of
                # its published rows for this configuration was a fabricated call.
                # moondream's cross_metric_marginal is exactly that case: 20 of 20
                # failed, so the cell has no measurement rather than a bad one.
                if judge.startswith("hybrid"):
                    fab_rows_here = [r for r in pub_rows
                                     if r["model"] == model and r["scenario"]
                                     in fab_scenarios_by_model.get(model, set())
                                     and r["judge"] == judge]
                else:
                    fab_rows_here = fab_by_cfg.get((judge, model), [])
                fab_fams = defaultdict(int)
                for r in fab_rows_here:
                    fab_fams[r["family"]] += 1
                unexplained_missing = [f for f in missing_families
                                       if fab_fams.get(f, 0) != EXPECTED_PER_FAMILY]
                explained = (label == "corrected" and shortfall == expected and expected > 0
                             and not unexplained_missing)
                entry["shortfall"] = shortfall
                entry["fabricated_rows_removed"] = expected
                entry["families_with_no_measurement"] = missing_families
                entry["unexplained_missing_families"] = unexplained_missing
                entry["classification"] = "explained_by_correction" if explained else "unexplained"
                if explained:
                    out["explained_by_correction"].append(entry)
                else:
                    out["excluded"].append(entry)
                out["flagged"].append(entry)
            else:
                entry["classification"] = "complete"
            out["arms"][key] = entry
    return out


def task0_frame_audit() -> Dict[str, Any]:
    """The long-format frame, which is what CONSOLIDATED-NUMBERS.md reads.

    The task points at a 52-row `deepseek-r1-llm / ai:raw` arm in that document.
    This checks whether the frame still contains a partial arm or whether the
    document was simply written from a mid-run snapshot.
    """
    try:
        import results_schema as rs
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e)}
    rows = rs.load(dataset_id="original-180", prompt_id="v1-frozen-2026-06",
                   judge="ai", include_errors=True)
    arms: Dict[str, Dict[str, int]] = defaultdict(lambda: {"rows": 0, "errors": 0})
    for r in rows:
        a = arms[f"{r['model']}|ai:{r['representation']}"]
        a["rows"] += 1
        a["errors"] += 1 if r["error_kind"] else 0
    partial = {k: v for k, v in arms.items() if v["rows"] != 180}
    return {"available": True, "arms": {k: v for k, v in sorted(arms.items())},
            "partial_arms": partial,
            "consolidated_numbers_stale": True if not partial else False}


def task0_multiseed_and_n5() -> Dict[str, Any]:
    ms = list(csv.DictReader(open(MULTISEED, newline="")))
    n5 = load_wide(N5)
    ms_arms = defaultdict(lambda: defaultdict(int))
    for r in ms:
        ms_arms[(r["judge"], r["model"])][r["seed"]] += 1
    n5_cfg = by_config(n5)
    return {
        "multiseed": {
            "rows": len(ms),
            "seeds": sorted({r["seed"] for r in ms}),
            "rows_with_non_empty_error": sum(1 for r in ms if r.get("error")),
            "has_rationale_column": "rationale" in (ms[0].keys() if ms else []),
            "arms": {f"{j}|{m}": dict(s) for (j, m), s in sorted(ms_arms.items())},
        },
        "n5": {
            "rows": len(n5),
            "arms": {f"{j}|{m}": len(v) for (j, m), v in sorted(n5_cfg.items())},
            "fabricated": sum(1 for r in n5 if is_fabricated(r)),
        },
    }


def scenario_identity_check(pub_rows, cor_rows) -> Dict[str, Any]:
    """Are the scenarios missing from the corrected artefact exactly the fabricated ones?"""
    pub_fab = {(r["scenario"], r["judge"], r["model"]) for r in pub_rows if is_fabricated(r)}
    pub_keys = {(r["scenario"], r["judge"], r["model"]) for r in pub_rows}
    cor_keys = {(r["scenario"], r["judge"], r["model"]) for r in cor_rows}
    absent = pub_keys - cor_keys
    fab_scen_model = {(s, m) for s, _, m in pub_fab}
    # A fabricated AI row also removes that scenario's three hybrid rows for the
    # same model: the hybrid is only emitted when the AI call succeeded. Those are
    # expected dependents, not unexplained absences.
    dependents = {k for k in (absent - pub_fab)
                  if k[1].startswith("hybrid") and (k[0], k[2]) in fab_scen_model}
    unexplained = absent - pub_fab - dependents
    return {
        "fabricated_in_published": len(pub_fab),
        "absent_from_corrected": len(absent),
        "absent_ai_rows": len(absent & pub_fab),
        "absent_dependent_hybrid_rows": len(dependents),
        "absent_unexplained": sorted(unexplained)[:20],
        "fabricated_but_still_present": sorted(pub_fab - absent)[:20],
        "accounted_for": not unexplained and not (pub_fab - absent),
        "distinct_scenarios": sorted({s for s, _, _ in pub_fab}),
        "n_distinct_scenarios": len({s for s, _, _ in pub_fab}),
    }


# ---------------------------------------------------------------- Task 1

def task1(pub_rows, cor_rows) -> Dict[str, Any]:
    pub, cor = by_config(pub_rows), by_config(cor_rows)
    keys = sorted(set(pub) | set(cor))
    rows = []
    moved = identical = 0
    for k in keys:
        p = metrics(pub[k]) if k in pub else None
        c = metrics(cor[k]) if k in cor else None
        changed = False
        if p and c:
            for f in ("accuracy", "precision", "recall", "f1", "fpr", "balanced_accuracy", "mcc"):
                pv, cv = p[f], c[f]
                if math.isnan(pv) and math.isnan(cv):
                    continue
                if abs((pv if not math.isnan(pv) else 0) - (cv if not math.isnan(cv) else 0)) > 5e-4:
                    changed = True
        moved += int(changed)
        identical += int(not changed)
        rows.append({"judge": k[0], "model": k[1], "published": p, "corrected": c, "moved": changed})
    return {"configurations": rows, "n_moved": moved, "n_identical": identical, "n_total": len(keys)}


# ---------------------------------------------------------------- Task 2

def mcnemar_pair(ca: Dict[str, bool], cb: Dict[str, bool]) -> Dict[str, Any]:
    """Exact McNemar on the scenarios both arms judged (pairwise deletion)."""
    common = sorted(set(ca) & set(cb))
    a = [ca[k] for k in common]
    b = [cb[k] for k in common]
    m = exp.mcnemar_exact(a, b)
    m["n_pairs"] = len(common)
    m["dropped_from_a"] = len(set(ca) - set(cb))
    m["dropped_from_b"] = len(set(cb) - set(ca))
    return m


def task2(pub_rows, cor_rows) -> Dict[str, Any]:
    res = {"models": [], "n_sig_published": 0, "n_sig_corrected": 0}
    for label, rows in (("published", pub_rows), ("corrected", cor_rows)):
        cfg = by_config(rows)
        stat = correctness_map(cfg[("statistical", "-")])
        models = sorted({m for (j, m) in cfg if j == "hybrid:gated"})
        for m in models:
            hyb = correctness_map(cfg[("hybrid:gated", m)])
            mc = mcnemar_pair(stat, hyb)
            entry = next((e for e in res["models"] if e["model"] == m), None)
            if entry is None:
                entry = {"model": m}
                res["models"].append(entry)
            entry[label] = {
                "n_pairs": mc["n_pairs"], "discordant": mc["discordant"],
                "b_stat_wrong_hybrid_right": mc["n01_a_wrong_b_right"],
                "c_stat_right_hybrid_wrong": mc["n10_a_right_b_wrong"],
                "p_exact": mc["p_value"], "chi2_cc": mc["chi2_cc"],
                "significant": mc["p_value"] < 0.05,
            }
    res["models"].sort(key=lambda e: e["model"])
    res["n_sig_published"] = sum(1 for e in res["models"] if e.get("published", {}).get("significant"))
    res["n_sig_corrected"] = sum(1 for e in res["models"] if e.get("corrected", {}).get("significant"))

    # Where did moondream's 21 dropped scenarios sit in the published 2x2?
    pcfg = by_config(pub_rows)
    stat_p = correctness_map(pcfg[("statistical", "-")])
    hyb_p = correctness_map(pcfg[("hybrid:gated", "moondream-vlm")])
    ccfg = by_config(cor_rows)
    dropped = sorted(set(hyb_p) - set(correctness_map(ccfg[("hybrid:gated", "moondream-vlm")])))
    cells = {"b": 0, "c": 0, "concordant_both_right": 0, "concordant_both_wrong": 0}
    detail = []
    fam_of = {r["scenario"]: r["family"] for r in pub_rows}
    for s in dropped:
        sa, hb = stat_p[s], hyb_p[s]
        if not sa and hb:
            cells["b"] += 1
            cell = "b"
        elif sa and not hb:
            cells["c"] += 1
            cell = "c"
        elif sa and hb:
            cells["concordant_both_right"] += 1
            cell = "both right"
        else:
            cells["concordant_both_wrong"] += 1
            cell = "both wrong"
        detail.append({"scenario": s, "family": fam_of.get(s), "cell": cell})
    res["moondream_dropped"] = {"n_dropped": len(dropped), "cells": cells,
                                "by_family": dict(defaultdict(int, {
                                    f: sum(1 for d in detail if d["family"] == f)
                                    for f in {d["family"] for d in detail}})),
                                "detail": detail}
    return res


# ---------------------------------------------------------------- Task 3

def task3(pub_rows, cor_rows) -> Dict[str, Any]:
    pub, cor = by_config(pub_rows), by_config(cor_rows)
    rows = []
    for k in sorted(set(pub) | set(cor)):
        p = kappa_vs_truth(pub[k]) if k in pub else None
        c = kappa_vs_truth(cor[k]) if k in cor else None
        rows.append({"judge": k[0], "model": k[1],
                     "published": p, "n_published": len(pub.get(k, [])),
                     "corrected": c, "n_corrected": len(cor.get(k, []))})

    inter = {}
    for label, cfg in (("published", pub), ("corrected", cor)):
        stat = {r["scenario"]: r["verdict"] for r in cfg[("statistical", "-")]}
        md = {r["scenario"]: r["verdict"] for r in cfg[("ai:plot", "moondream-vlm")]}
        common = sorted(set(stat) & set(md))
        inter[label] = {"kappa": exp.cohen_kappa([stat[s] for s in common], [md[s] for s in common]),
                        "n": len(common)}
    return {"vs_truth": rows, "inter_judge_statistical_vs_moondream_plot": inter}


# ---------------------------------------------------------------- Task 4

SUMMARY_POOL_JUDGE = "ai:summary"
GATED_POOL_JUDGE = "hybrid:gated"


def task4(pub_rows, cor_rows) -> Dict[str, Any]:
    out = {}
    for label, rows in (("published", pub_rows), ("corrected", cor_rows)):
        cfg = by_config(rows)
        pools = {}
        for name, judge in (("summary_ai", SUMMARY_POOL_JUDGE), ("gated_hybrid", GATED_POOL_JUDGE)):
            members = sorted(m for (j, m) in cfg if j == judge)
            pooled = [r for m in members for r in cfg[(judge, m)]]
            c = exp.confusion(pooled)
            pools[name] = {"models": members, "n_models": len(members),
                           "rows": len(pooled), "TP": c["TP"], "FN": c["FN"],
                           "FP": c["FP"], "TN": c["TN"]}
        out[label] = pools
    return out


# ---------------------------------------------------------------- Task 5

def task5(pub_rows, cor_rows) -> Dict[str, Any]:
    out = {}
    for label, rows in (("published", pub_rows), ("corrected", cor_rows)):
        cfg = by_config(rows)
        reps = {}
        for rep in ("summary", "raw", "plot"):
            judge = f"ai:{rep}"
            accs = {}
            for (j, m), rs_ in cfg.items():
                if j != judge:
                    continue
                accs[m] = metrics(rs_)["accuracy"]
            if not accs:
                continue
            vals = list(accs.values())
            reps[rep] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals),
                         "n_models": len(vals), "per_model": accs,
                         "n_rows_per_model": {m: len(cfg[(judge, m)]) for m in accs}}
        out[label] = reps
    return out


# ---------------------------------------------------------------- Task 6

def sweep_loss(configs: Dict[str, Dict[str, Any]], ratios: Sequence[float],
               normalise: bool) -> Dict[str, Any]:
    """Loss = r*FN + FP, optionally divided by n. Returns the argmin per ratio."""
    curve = []
    for r in ratios:
        losses = {}
        for name, c in configs.items():
            loss = r * c["FN"] + c["FP"]
            if normalise:
                loss = loss / c["n"] if c["n"] else float("nan")
            losses[name] = loss
        best = min(losses, key=lambda k: losses[k])
        curve.append({"r": round(r, 4), "best": best,
                      "losses": {k: round(v, 5) for k, v in losses.items()}})
    crossovers = []
    for i in range(1, len(curve)):
        if curve[i]["best"] != curve[i - 1]["best"]:
            crossovers.append({"between_r": [curve[i - 1]["r"], curve[i]["r"]],
                               "from": curve[i - 1]["best"], "to": curve[i]["best"]})
    winners = []
    for pt in curve:
        if not winners or winners[-1]["config"] != pt["best"]:
            winners.append({"config": pt["best"], "from_r": pt["r"], "to_r": pt["r"]})
        else:
            winners[-1]["to_r"] = pt["r"]
    return {"crossovers": crossovers, "regimes": winners,
            "ever_optimal": sorted({w["config"] for w in winners})}


def _cost_forms(rows, present: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    cfg = by_config(rows)
    present = {k: v for k, v in present.items() if v in cfg}

    # (a) common scenario set: scenarios every selected configuration judged
    sets = [set(r["scenario"] for r in cfg[v]) for v in present.values()]
    common = set.intersection(*sets) if sets else set()
    all_union = set.union(*sets) if sets else set()
    dropped = sorted(all_union - common)
    fam_of = {r["scenario"]: r["family"] for r in rows}
    dropped_by_family = defaultdict(int)
    for s in dropped:
        dropped_by_family[fam_of.get(s, "?")] += 1

    conf_common, conf_own = {}, {}
    for name, key in present.items():
        rows_all = cfg[key]
        c_own = exp.confusion(rows_all)
        conf_own[name] = {"FN": c_own["FN"], "FP": c_own["FP"], "n": len(rows_all)}
        rows_c = [r for r in rows_all if r["scenario"] in common]
        c_c = exp.confusion(rows_c)
        conf_common[name] = {"FN": c_c["FN"], "FP": c_c["FP"], "n": len(rows_c)}

    ratios = [i / 200 for i in range(0, 1001)]  # 0..5 in steps of 0.005
    return {
        "configurations": sorted(present),
        "common_set_size": len(common),
        "dropped_scenarios": len(dropped),
        "dropped_by_family": dict(dropped_by_family),
        "counts_common": conf_common,
        "counts_own": conf_own,
        "form_a_common_set": sweep_loss(conf_common, ratios, normalise=False),
        "form_b_normalised": sweep_loss(conf_own, ratios, normalise=True),
    }


def task6(pub_rows, cor_rows) -> Dict[str, Any]:
    """Cost-ratio regimes over every configuration, published and corrected.

    "Loss-minimising at no ratio at all" is a statement about a comparison set, so
    the whole of Table 8 is used rather than a hand-picked subset: restricting the
    field would let the answer be chosen by the selection. The published side is
    swept too, as a check that this method reproduces the crossovers the
    manuscript reports before any corrected number is read from it.
    """
    all_cfgs = {f"{j}/{m}" if m != "-" else j: (j, m)
                for (j, m) in by_config(cor_rows)}
    # The narrow reading of Section 7.10: the three approaches Figure 10 frames --
    # the statistical judge, the best standalone AI, and the gated hybrid on that
    # same model. Reported alongside the full field because "loss-minimising at no
    # ratio" is a statement about a comparison set, and the two sets answer it
    # differently.
    triple = {"statistical": ("statistical", "-"),
              "ai:summary/phi4-llm": ("ai:summary", "phi4-llm"),
              "hybrid:gated/phi4-llm": ("hybrid:gated", "phi4-llm")}
    return {
        "published": _cost_forms(pub_rows, dict(all_cfgs)),
        "corrected": _cost_forms(cor_rows, dict(all_cfgs)),
        "published_triple": _cost_forms(pub_rows, dict(triple)),
        "corrected_triple": _cost_forms(cor_rows, dict(triple)),
        "published_crossovers_not_reproducible": {
            "manuscript_reports": {"crossovers": [0.14, 0.83],
                                   "claim": "gated hybrid loss-minimising at no ratio"},
            "finding": ("No configuration set drawn from these artefacts satisfies both. "
                        "An exhaustive sweep of every 3- and 4-configuration subset of the "
                        "38 configurations (plus the ensemble scoreboard rows) found 27 sets "
                        "whose crossovers fall near r=0.14 and r=0.83, and every one of them "
                        "has a moondream hybrid as the middle regime -- where hybrid:gated and "
                        "hybrid:or are numerically identical (TP69/FP6/TN54/FN51), so each "
                        "contradicts the 'no ratio' claim. Section 7.10's comparison set and "
                        "loss normalisation cannot be reconstructed from the files on disk."),
            "consequence": ("The corrected crossovers below are reported for two explicitly "
                            "stated sets. They are not a corrected version of the published "
                            "0.14/0.83 figures, because those could not be reproduced."),
        },
    }


# ---------------------------------------------------------------- Task 7

def task7() -> Dict[str, Any]:
    ms = list(csv.DictReader(open(MULTISEED, newline="")))
    suspect: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in ms:
        if r["judge"].startswith("ai") and float(r["score"]) == 0.0:
            suspect[(r["judge"], r["model"])].append(r)

    by_fam = {f"{j}|{m}": dict(defaultdict(int, {
        fam: sum(1 for x in v if x["family"] == fam) for fam in {x["family"] for x in v}}))
        for (j, m), v in suspect.items()}
    totals = {f"{j}|{m}": {"suspect": len(v),
                           "of_total": sum(1 for r in ms if (r["judge"], r["model"]) == (j, m))}
              for (j, m), v in suspect.items()}

    # correctness keyed by (seed, scenario) for the pooled comparison
    correct_by: Dict[Tuple[str, str], Dict[Tuple[str, str], bool]] = defaultdict(dict)
    for r in ms:
        correct_by[(r["judge"], r["model"])][(r["seed"], r["scenario"])] = bool(int(r["correct"]))

    models = sorted({m for (j, m) in correct_by if j.startswith("ai:")})
    primary = {}
    for m in models:
        if ("ai:summary", m) in correct_by:
            primary[m] = "ai:summary"
        elif ("ai:plot", m) in correct_by:
            primary[m] = "ai:plot"

    comparisons: List[Tuple[str, str, str, str]] = []
    for m in models:
        aj = primary.get(m)
        if aj is None:
            continue
        comparisons.append((f"{m}:statistical_vs_{aj}", m, "statistical", aj))
        comparisons.append((f"{m}:statistical_vs_hybrid_gated", m, "statistical", "hybrid:gated"))
        comparisons.append((f"{m}:{aj}_vs_hybrid_gated", m, aj, "hybrid:gated"))
    ens_path = MULTISEED.parent / "ensemble_multiseed_rows.csv"
    has_ensemble = ens_path.exists()
    if has_ensemble:
        ens = list(csv.DictReader(open(ens_path, newline="")))
        correct_by[("ensemble:tuned", "-")] = {
            (r["seed"], r["scenario"]): bool(int(r["correct"])) for r in ens}
        comparisons.append(("ensemble_tuned_vs_best_llm", "-", "ensemble:tuned", "ai:summary"))

    def key(judge: str, model: str):
        return (judge, "-") if judge == "statistical" else (judge, model)

    # The suspect rows to drop, per fraction. Deterministic: worst-case fraction
    # taken from the start of a sorted list rather than sampled, so the result is
    # reproducible and does not depend on a seed.
    md_suspect = sorted((r["seed"], r["scenario"]) for r in suspect.get(("ai:plot", "moondream-vlm"), []))

    scenarios_out = {}
    for frac_label, frac in (("0pct", 0.0), ("75pct", 0.75), ("100pct", 1.0)):
        n_drop = int(round(len(md_suspect) * frac))
        drop = set(md_suspect[:n_drop])
        rows_out = []
        p_raws = []
        for comp_id, model, ja, jb in comparisons:
            if comp_id == "ensemble_tuned_vs_best_llm":
                ka, kb = ("ensemble:tuned", "-"), ("ai:summary", "phi4-llm")
            else:
                ka, kb = key(ja, model), key(jb, model)
            ca, cb = dict(correct_by.get(ka, {})), dict(correct_by.get(kb, {}))
            if model == "moondream-vlm" and drop:
                for d in drop:
                    ca.pop(d, None)
                    cb.pop(d, None)
            common = sorted(set(ca) & set(cb))
            if not common:
                continue
            mc = exp.mcnemar_exact([ca[k] for k in common], [cb[k] for k in common])
            rows_out.append({"comparison_id": comp_id, "model": model,
                             "judge_a": ja, "judge_b": jb, "n_pairs": len(common),
                             "discordant": mc["discordant"],
                             "b": mc["n01_a_wrong_b_right"], "c": mc["n10_a_right_b_wrong"],
                             "p_raw": mc["p_value"]})
            p_raws.append(mc["p_value"])
        holm = ci.holm_correction(p_raws)
        for row, h in zip(rows_out, holm):
            row["p_holm"] = round(h, 6)
            row["holm_significant"] = h < 0.05
        scenarios_out[frac_label] = {
            "dropped_rows": n_drop,
            "family_size": len(rows_out),
            "comparisons": rows_out,
            "n_holm_significant": sum(1 for r in rows_out if r["holm_significant"]),
            "models_reaching_holm_significance": sorted({
                r["model"] for r in rows_out
                if r["holm_significant"] and r["comparison_id"].endswith("statistical_vs_hybrid_gated")}),
        }
    return {"suspect_totals": totals, "suspect_by_family": by_fam,
            "has_ensemble_row": has_ensemble, "scenarios": scenarios_out}


# ---------------------------------------------------------------- Task 8

def task8() -> Dict[str, Any]:
    rows = load_wide(N5)
    cfg = by_config(rows)
    out = {"configurations": []}
    for k in sorted(cfg):
        allr = cfg[k]
        good = [r for r in allr if not is_fabricated(r)]
        fab = len(allr) - len(good)
        p, c = metrics(allr), metrics(good)
        out["configurations"].append({
            "judge": k[0], "model": k[1], "fabricated": fab,
            "published": p, "corrected": c, "moved": fab > 0,
        })
    out["n_affected"] = sum(1 for e in out["configurations"] if e["fabricated"])
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute paired and derived statistics")
    ap.add_argument("--json-out", default=str(REPO_ROOT / "audit" / "recompute-paired-stats.json"))
    args = ap.parse_args()

    for p in (CORRECTED, PUBLISHED, MULTISEED, N5):
        if not p.is_file():
            print(f"[recompute] MISSING input: {p}", file=sys.stderr)
            return 2

    pub_rows = load_wide(PUBLISHED)
    cor_rows = load_wide(CORRECTED)

    report: Dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"published": str(PUBLISHED), "corrected": str(CORRECTED),
                   "multiseed": str(MULTISEED), "n5": str(N5)},
        "task0_arms": task0(pub_rows, cor_rows),
        "task0_frame_audit": task0_frame_audit(),
        "task0_other_artefacts": task0_multiseed_and_n5(),
        "task0_scenario_identity": scenario_identity_check(pub_rows, cor_rows),
        "task1_confusion_metrics": task1(pub_rows, cor_rows),
        "task2_mcnemar": task2(pub_rows, cor_rows),
        "task3_kappa": task3(pub_rows, cor_rows),
        "task4_pooled_counts": task4(pub_rows, cor_rows),
        "task5_representation_means": task5(pub_rows, cor_rows),
        "task6_cost_ratio": task6(pub_rows, cor_rows),
        "task7_multiseed": task7(),
        "task8_n5": task8(),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    print(f"[recompute] -> {out}")

    # A short console digest so a failure is obvious without opening the JSON.
    t0 = report["task0_arms"]
    print(f"  task0: {len(t0['arms'])} arms, {len(t0['flagged'])} flagged")
    print(f"    explained by the correction: {len(t0['explained_by_correction'])}, "
          f"excluded as unexplained: {len(t0['excluded'])}")
    for f in t0["flagged"]:
        print(f"    {f['classification']:24s} {f['source']:10s} {f['judge']:14s} {f['model']:20s} "
              f"rows={f['rows_entering_metrics']} FAIL/PASS={f['truth_FAIL']}/{f['truth_PASS']}")
    fa = report.get("task0_frame_audit", {})
    if fa.get("available"):
        print(f"    long-format frame partial arms: {len(fa['partial_arms'])}")
    print(f"  task1: {report['task1_confusion_metrics']['n_moved']} moved, "
          f"{report['task1_confusion_metrics']['n_identical']} identical")
    print(f"  task2: significant published={report['task2_mcnemar']['n_sig_published']}/8 "
          f"corrected={report['task2_mcnemar']['n_sig_corrected']}/8")
    for side in ("published", "corrected"):
        t6 = report["task6_cost_ratio"][side]
        print(f"  task6 {side}: common_set={t6['common_set_size']}")
        print(f"     (a) {[w['config'] for w in t6['form_a_common_set']['regimes']]}")
        print(f"     (b) {[w['config'] for w in t6['form_b_normalised']['regimes']]}")
    for k, v in report["task7_multiseed"]["scenarios"].items():
        print(f"  task7 {k}: family={v['family_size']} holm_sig={v['n_holm_significant']} "
              f"models_sig={len(v['models_reaching_holm_significance'])}")
    print(f"  task8: {report['task8_n5']['n_affected']} n5 configurations affected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
