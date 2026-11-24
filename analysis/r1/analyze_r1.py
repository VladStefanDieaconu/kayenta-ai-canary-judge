#!/usr/bin/env python3
"""Analysis of the first-revision runs as fixed in protocol/r1-controls-protocol.json.

Reads only the long-format records of the runs (runs/records), the protocol's series files, the
generator in tools/eval_dataset.py (original-180 and generalisation-60 at master seed 20260621), the outputs of
comparators_live.py and comparators_bfks.py, and, for the session anchors and the pooled figures, the published
records of the result data deposit 1.1.1 and Archive S1. No model call, no service call.

Sections (outputs in --out):
  0  statistical verdicts in every invocation, against each other, Kayenta's own call and the published records
  1  kind controls (runs H2, L2): per family and configuration, false alarms by kind, detection, BA and MCC with
     the percentile bootstrap (2,000 replicates, fixed families, seed 20261122); kc_tri_marginal separately
  2  the hybrid as scored on the control windows, checked against the published rule
  3  the metric-kind rules on the controls and on the pooled 360 windows (cross-session)
  4  paired comparisons on the controls: 21 model configurations against the metric-kind + statistical rule and
     against the statistical judge, exact McNemar, Holm over the 42 tests
  5  reading rules (kind-driven; not explained by kind on these controls; otherwise no label)
  6  held-out sets (runs H3, H4): false alarms and detection with Wilson intervals, score distributions, rubric
     effects (Newcombe method 10, exact McNemar, Holm over 18 tests), policies P6 and P7 on the 40
     recovered-transient windows; the model-free comparators on the held-out sets
  7  session anchors (runs H1, L1 against the published runs; H4 against the published generalisation-60 runs)
  8  tables for the writers (Tables 9 and 10 and the supplementary tables)

Run (from the repository root, analysis interpreter with numpy):
  python -I -B analysis/r1/analyze_r1.py --runs <runs> --comparators <dir> --bfks <dir> \
      --deposit <data-archive> --archive-s1 <unzipped Archive S1> --out <new-dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r1_common as rc  # noqa: E402
import d2_label_policies as d2  # noqa: E402  (the post-hoc label-policy code, unchanged)

SCRIPT = "analysis/r1/analyze_r1.py"


class Facts:
    def __init__(self):
        self.items: List[Dict[str, str]] = []
        self.keys = set()

    def add(self, key: str, value: Any, description: str, source: str, computation: str) -> None:
        if key in self.keys:
            raise SystemExit(f"duplicate fact key {key}")
        self.keys.add(key)
        self.items.append({"key": key, "value": str(value), "description": description, "source_file": source,
                           "filter_or_computation": computation, "script": SCRIPT})


class Config:
    def __init__(self, cid: str, label: str, label_s: str, group: str, verdicts: Dict[str, str],
                 scores: Optional[Dict[str, float]] = None, source: str = "", filt: str = ""):
        self.cid, self.label, self.label_s, self.group = cid, label, label_s, group
        self.verdicts, self.scores = verdicts, scores or {}
        self.source, self.filt = source, filt


def fails(cfg: Config, ids) -> Tuple[int, int]:
    ids = [s for s in ids if s in cfg.verdicts]
    return sum(1 for s in ids if cfg.verdicts[s] == "FAIL"), len(ids)


# ============================================================================= main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--comparators", required=True, help="output folder of comparators_live.py")
    ap.add_argument("--bfks", required=True, help="output folder of comparators_bfks.py")
    ap.add_argument("--deposit", required=True, help="result data deposit 1.1.1 (read only)")
    ap.add_argument("--archive-s1", required=True, help="unzipped Archive S1 (read only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = Path(a.runs).resolve()
    base = runs.parent
    dep = Path(a.deposit)
    s1 = Path(a.archive_s1)
    out = Path(a.out)
    cmp_dir = Path(a.comparators)
    bf_dir = Path(a.bfks)
    F = Facts()
    protocol = rc.load_protocol()
    S = rc.archived_stats(dep)
    recs = rc.Records(runs / "records", protocol, base)
    series = rc.load_all_series()
    truth = {ds: {s: w["truth"] for s, w in series[ds].items()} for ds in series}
    fam = {ds: {s: w["family"] for s, w in series[ds].items()} for ds in series}
    results: Dict[str, Any] = {}
    REC = "runs/records/"
    CMP_REL = "analysis/comparators/"

    # records' labels and families must equal the series'
    lab_mis = [r["_file"] + ":" + r["scenario_id"] for r in recs.rows
               if r["truth_label"] != truth[r["dataset_id"]].get(r["scenario_id"]) or r["family"] != fam[r["dataset_id"]].get(r["scenario_id"])]
    results["labels_records_vs_series_mismatches"] = len(lab_mis)
    if lab_mis:
        raise SystemExit(f"labels or families of the records differ from the series: {lab_mis[:5]}")

    # ------------------------------------------------------------------ 0. statistical verdicts
    sc_rows = recs.stat_consistency()
    rc.write_csv(out / "s0_statistical_verdicts_by_window.csv", sc_rows)
    differing = [r for r in sc_rows if r["distinct_verdict_score"] != 1 or r["error_rows"]]
    stat_by_inv = defaultdict(lambda: [0, 0])
    for (ds, s), rows in recs.stat.items():
        for r in rows:
            stat_by_inv[r["_inv"]][0] += 1
    s0 = {"windows": len(sc_rows), "statistical_rows": sum(r["invocations"] for r in sc_rows),
          "windows_with_differing_verdict_or_score": len(differing), "statistical_error_rows": sum(r["error_rows"] for r in sc_rows)}
    # Kayenta's own call inside comparators_live (ensemble materialisation)
    live = {}
    for ds in (rc.O180, rc.G60, rc.KC, rc.HB):
        live[ds] = {r["scenario_id"]: r for r in rc.read_csv(cmp_dir / f"comparators_live_{ds}.csv")}
    kay_cmp = {}
    for ds in (rc.G60, rc.KC, rc.HB):
        rv = recs.stat_verdicts(ds)
        rs_ = recs.stat_scores(ds)
        same_v = sum(1 for s in rv if live[ds][s]["kayenta_statistical_verdict"] == rv[s])
        same_s = sum(1 for s in rv if float(live[ds][s]["kayenta_statistical_score"]) == rs_[s])
        kay_cmp[ds] = {"windows": len(rv), "same_verdict": same_v, "same_score": same_s}
    # original-180 anchors: records against the published statistical rows and Kayenta's call
    corr = rc.read_csv(dep / "results" / "corrected" / "results_raw.csv")
    pub_stat = {r["scenario"]: (r["verdict"], float(r["score"])) for r in corr if r["judge"] == "statistical"}
    rv = recs.stat_verdicts(rc.O180)
    rs_ = recs.stat_scores(rc.O180)
    kay_cmp[rc.O180 + " (18 anchor windows)"] = {
        "windows": len(rv), "same_verdict": sum(1 for s in rv if live[rc.O180][s]["kayenta_statistical_verdict"] == rv[s]),
        "same_score": sum(1 for s in rv if float(live[rc.O180][s]["kayenta_statistical_score"]) == rs_[s]),
        "same_verdict_as_published": sum(1 for s in rv if pub_stat[s][0] == rv[s]),
        "same_score_as_published": sum(1 for s in rv if pub_stat[s][1] == rs_[s])}
    kay_full = sum(1 for s, (v, sc) in pub_stat.items() if live[rc.O180][s]["kayenta_statistical_verdict"] == v
                   and float(live[rc.O180][s]["kayenta_statistical_score"]) == sc)
    kay_cmp[rc.O180 + " (all 180, Kayenta call of comparators_live against the published records)"] = {"same_verdict_and_score": kay_full}
    s0["kayenta_and_published"] = kay_cmp
    results["s0_statistical"] = s0
    stat_fail = {ds: sum(1 for v in recs.stat_verdicts(ds).values() if v == "FAIL") for ds in (rc.KC, rc.HB, rc.G60, rc.O180)}
    F.add("s0.stat.consistency", f"{s0['windows_with_differing_verdict_or_score']} of {s0['windows']} windows",
          "windows whose statistical verdict or score differs between the invocations that judged them (every window "
          f"in every invocation: {s0['statistical_rows']} statistical rows)", REC + "*/*.csv",
          "judge=statistical rows grouped by (dataset_id, scenario_id); distinct (verdict, score) pairs counted")
    F.add("s0.stat.error_rows", s0["statistical_error_rows"], "statistical (Kayenta) rows with an error in the records",
          REC + "*/*.csv", "judge=statistical and error_kind non-empty")
    F.add("s0.stat.anchor_published", f"{kay_cmp[rc.O180 + ' (18 anchor windows)']['same_verdict_as_published']}/18 verdicts, "
          f"{kay_cmp[rc.O180 + ' (18 anchor windows)']['same_score_as_published']}/18 scores",
          "statistical verdicts and scores of the November runs on the 18 anchor windows equal to the published ones",
          REC + "H1-*/*.csv; " + REC + "L1-*/*.csv; data-archive/results/corrected/results_raw.csv",
          "anchor windows _000 and _001 of each family; verdict and score compared")
    for ds in (rc.KC, rc.HB, rc.G60):
        n = len(recs.stat_verdicts(ds))
        F.add(f"s0.stat.fail.{ds}", f"{stat_fail[ds]}/{n}", f"windows of {ds} failed by the statistical judge (November runs)",
              REC + "*/*.csv", f"judge=statistical; dataset_id={ds}; verdict=FAIL; one verdict per window (identical in every invocation)")
    for ds in (rc.KC, rc.HB, rc.G60):
        d = kay_cmp[ds]
        F.add(f"s0.stat.kayenta_check.{ds}", f"{d['same_verdict']}/{d['windows']} verdicts, {d['same_score']}/{d['windows']} scores",
              f"statistical verdicts of the run records equal to Kayenta's own call made for the tuned ensemble on {ds}",
              REC + "*/*.csv; " + CMP_REL + f"comparators_live_{ds}.csv", "verdict and score compared per window")

    # ------------------------------------------------------------------ 1. configurations on the kind controls
    def kind_rule(ds):
        return {s: ("FAIL" if any(m["kind"] in ("memory", "errors") for m in w["metrics"]) else "PASS")
                for s, w in series[ds].items()}

    def guard_rule(ds, stat):
        lv = live[ds]
        return {s: ("PASS" if stat[s] == "PASS" and (lv[s]["guard_recovered_transient"] == "1" or lv[s]["guard_equal_variance_noise"] == "1")
                    else "FAIL") for s in stat}

    def bfks(ds):
        return {r["scenario_id"]: r["bfks_verdict"] for r in rc.read_csv(bf_dir / f"bfks_verdicts_{ds}.csv")}

    def comparators(ds, stat, stat_scores, stat_src):
        kr = kind_rule(ds)
        return [
            Config("statistical", "Statistical judge", "Statistical judge", "statistical", stat, stat_scores, stat_src,
                   "judge=statistical (identical in every invocation)"),
            Config("rule:kind", "Metric-kind rule", "Metric-kind rule", "rule", kr, None, f"protocol/series_{ds}.jsonl" if ds in (rc.KC, rc.HB) else "tools/eval_dataset.py",
                   "FAIL iff the window has a metric of kind memory or errors"),
            Config("rule:kind+stat", "Metric-kind + statistical rule", "Metric-kind + statistical rule", "rule",
                   {s: ("FAIL" if kr[s] == "FAIL" or stat[s] == "FAIL" else "PASS") for s in stat}, None,
                   stat_src + "; series", "FAIL iff the metric-kind rule or the statistical judge fails the window"),
            Config("ensemble", "Tuned ensemble", "Tuned ensemble", "comparator",
                   {s: r["ensemble_verdict"] for s, r in live[ds].items()}, None, CMP_REL + f"comparators_live_{ds}.csv",
                   "judge-service/ensemble_tuned.json tuned_config; run_ensemble_experiment.materialise + evaluate_config"),
            Config("guard", "Guard rule", "Guard rule", "comparator", guard_rule(ds, stat), None,
                   stat_src + "; " + CMP_REL + f"comparators_live_{ds}.csv",
                   "FAIL unless the statistical judge passes and a guard fires (recovered transient on any metric; equal-variance noise on single-metric windows)"),
            Config("bfks", "Post-hoc BF+KS (fixed alpha)", "Post-hoc BF+KS (fixed alpha)", "comparator", bfks(ds), None,
                   f"analysis/comparators/bfks_verdicts_{ds}.csv", "FAIL iff any per-metric p < 0.05/(2m)"),
        ]

    def model_configs(ds, rubric=rc.V1):
        out_ = []
        for m, r in rc.MODEL_CONFIGS:
            v = recs.ai_verdicts(ds, m, r, rubric)
            if not v and not recs.ai_failed_ids(ds, m, r, rubric):
                continue
            out_.append(Config(f"ai|{m}|{r}", rc.config_label(m, r), rc.config_label_s(m, r),
                               "hosted" if m in rc.HOSTED else "local", v, recs.ai_scores(ds, m, r, rubric),
                               recs.source_files(ds, m, rubric), f"judge=ai; model={m}; representation={r}; prompt_id={rubric}; error_kind empty"))
        return out_

    def hybrid_configs(ds, rubric=rc.V1):
        out_ = []
        for m, p in rc.HYBRID_CONFIGS:
            v = recs.hyb_verdicts(ds, m, p, rubric)
            if not v:
                continue
            out_.append(Config(f"hybrid|{m}|{p}", rc.config_label(m, p, hybrid=True),
                               f"{rc.POLICY_NAME[p].capitalize()}, {rc.PAPER_MODEL[m]} ({rc.REP_PLURAL[rc.PRIMARY[m]]})",
                               "hybrid", v, recs.hyb_scores(ds, m, p, rubric), recs.source_files(ds, m, rubric),
                               f"judge=hybrid; model={m}; representation={p}"))
        return out_

    kc_stat = recs.stat_verdicts(rc.KC)
    kc_src = "; ".join(sorted({r["_file"] for r in recs.rows if r["dataset_id"] == rc.KC and r["judge"] == "statistical"}))
    KC_SRC_SHORT = REC + "H2-*/ and L2-*/ (11 record files)"
    kc_cfgs = comparators(rc.KC, kc_stat, recs.stat_scores(rc.KC), KC_SRC_SHORT) + model_configs(rc.KC) + hybrid_configs(rc.KC)
    if len([c for c in kc_cfgs if c.cid.startswith("ai|")]) != 21 or len([c for c in kc_cfgs if c.cid.startswith("hybrid|")]) != 33:
        raise SystemExit("expected 21 model configurations and 33 hybrids on the kind controls")
    T = truth[rc.KC]
    FAM = fam[rc.KC]
    by_fam = defaultdict(list)
    for s, f_ in FAM.items():
        by_fam[f_].append(s)
    benign = [s for f_ in rc.KC_BENIGN for s in by_fam[f_]]
    regress = [s for f_ in rc.KC_REGRESSION for s in by_fam[f_]]
    drift = [s for f_ in rc.KC_DRIFT for s in by_fam[f_]]
    fam_rows, sum_rows = [], []
    kc_summary: Dict[str, Dict[str, Any]] = {}
    for c in kc_cfgs:
        row = {"configuration": c.cid, "label": c.label}
        cells = {}
        for f_ in rc.KC_FAMILIES:
            ids = [s for s in by_fam[f_] if s in c.verdicts]
            k = sum(1 for s in ids if c.verdicts[s] == T[s])
            lo, hi = rc.wilson(k, len(ids))
            row[f"{f_}_correct"] = k
            row[f"{f_}_n"] = len(ids)
            row[f"{f_}_lo"] = rc.f3(lo)
            row[f"{f_}_hi"] = rc.f3(hi)
            row[f"{f_}_FAIL"] = sum(1 for s in ids if c.verdicts[s] == "FAIL")
            cells[f_] = (k, len(ids))
        fam_rows.append(row)
        fa, nb = fails(c, benign)
        de, nr = fails(c, regress)
        dd, nd = fails(c, drift)
        tm, ntm = fails(c, by_fam["kc_tri_marginal"])
        kinds = {kk: fails(c, [s for f_ in fl for s in by_fam[f_]]) for kk, fl in rc.KC_KIND_GROUPS.items()}
        bm = rc.ba_mcc_with_ci(S, c.verdicts, T, FAM, c.cid)
        srow = {"configuration": c.cid, "label": c.label, "group": c.group, "scored": len(c.verdicts),
                "false_alarms": fa, "benign_scored": nb, "fa_lo": rc.f3(rc.wilson(fa, nb)[0]), "fa_hi": rc.f3(rc.wilson(fa, nb)[1]),
                **{f"fa_{kk}": f"{v[0]}/{v[1]}" for kk, v in kinds.items()},
                "detected": de, "regressions_scored": nr, "det_lo": rc.f3(rc.wilson(de, nr)[0]), "det_hi": rc.f3(rc.wilson(de, nr)[1]),
                "drift_detected": dd, "drift_scored": nd, "drift_lo": rc.f3(rc.wilson(dd, nd)[0]), "drift_hi": rc.f3(rc.wilson(dd, nd)[1]),
                "tri_marginal_detected": tm, "tri_marginal_scored": ntm,
                "TP": bm["TP"], "FP": bm["FP"], "TN": bm["TN"], "FN": bm["FN"],
                "BA": bm["BA"], "BA_lo": bm["BA_lo"], "BA_hi": bm["BA_hi"], "BA_ci": bm["BA_ci"],
                "MCC": bm["MCC"], "MCC_lo": bm["MCC_lo"], "MCC_hi": bm["MCC_hi"], "MCC_ci": bm["MCC_ci"],
                "MCC_value": repr(bm["MCC_value"]), "constant_within_families": bm["constant_within_families"],
                "families_resampled": bm["families_resampled"], "source": c.source, "filter": c.filt}
        sum_rows.append(srow)
        kc_summary[c.cid] = {**srow, "cells": cells}
    rc.write_csv(out / "s1_controls_per_family.csv", fam_rows)
    rc.write_csv(out / "s1_controls_summary.csv", sum_rows)
    results["s1_controls"] = {c: {k: v for k, v in d.items() if k != "cells"} for c, d in kc_summary.items()}
    for c in kc_cfgs:
        d = kc_summary[c.cid]
        cells = "; ".join(f"{rc.KC_CODE[f_]} {d['cells'][f_][0]}/{d['cells'][f_][1]}" for f_ in rc.KC_FAMILIES)
        F.add(f"kc.{c.cid}.families", cells,
              f"{c.label} on kind-controls-180: correct verdicts per family (LN kc_latency_nochange, MN kc_memory_nochange, "
              "EN kc_errors_nochange, TN kc_tri_nochange, MR kc_memory_shared_ramp, LD kc_latency_drift, MD kc_memory_drift, "
              "ED kc_errors_drift, TM kc_tri_marginal); denominators are scored windows",
              c.source, c.filt + "; correct = verdict equals the label")
        F.add(f"kc.{c.cid}.fa", f"{d['false_alarms']}/{d['benign_scored']} [{d['fa_lo']}, {d['fa_hi']}]",
              f"{c.label} on kind-controls-180: false alarms over the benign windows, Wilson 95% interval", c.source,
              c.filt + "; FAIL verdicts on the five PASS families")
        F.add(f"kc.{c.cid}.fa_by_kind", "; ".join(f"{kk} {d['fa_' + kk]}" for kk in rc.KC_KIND_GROUPS),
              f"{c.label} on kind-controls-180: false alarms by kind (latency kc_latency_nochange; memory kc_memory_nochange "
              "and kc_memory_shared_ramp; errors kc_errors_nochange; three-metric kc_tri_nochange)", c.source, c.filt)
        F.add(f"kc.{c.cid}.det", f"{d['detected']}/{d['regressions_scored']} [{d['det_lo']}, {d['det_hi']}]",
              f"{c.label} on kind-controls-180: detection over the regression windows (three drift families and kc_tri_marginal), Wilson 95% interval",
              c.source, c.filt + "; FAIL verdicts on the four FAIL families")
        F.add(f"kc.{c.cid}.det_drift", f"{d['drift_detected']}/{d['drift_scored']} [{d['drift_lo']}, {d['drift_hi']}]",
              f"{c.label} on kind-controls-180: detection over the drift windows without kc_tri_marginal, Wilson 95% interval",
              c.source, c.filt + "; FAIL verdicts on kc_latency_drift, kc_memory_drift, kc_errors_drift")
        F.add(f"kc.{c.cid}.tm", f"{d['tri_marginal_detected']}/{d['tri_marginal_scored']}",
              f"{c.label} on kind-controls-180: kc_tri_marginal (contested label) detected", c.source, c.filt)
        F.add(f"kc.{c.cid}.ba", f"{d['BA']} {d['BA_ci']}",
              f"{c.label} on kind-controls-180: balanced accuracy with the percentile bootstrap interval (2,000 replicates within fixed families, seed 20261122)",
              c.source, c.filt + "; BA and interval as analyze_A prints them (f'{x:.3f}'); interval analyze_A.boot_config")
        F.add(f"kc.{c.cid}.mcc", f"{rc.typo(d['MCC'])} {rc.typo(d['MCC_ci'])}",
              f"{c.label} on kind-controls-180: MCC with the percentile bootstrap interval (2,000 replicates within fixed families, seed 20261122)",
              c.source, c.filt + "; analyze_A.boot_config")

    # ------------------------------------------------------------------ 2. hybrid as scored: the published rule
    hyb_rows = []
    for ds in (rc.KC, rc.O180):
        stat = recs.stat_verdicts(ds)
        for m in rc.TABLE_MODEL_ORDER:
            prim = rc.PRIMARY[m]
            sc = recs.ai_scores(ds, m, prim)
            gated = recs.hyb_verdicts(ds, m, "gated")
            n = agree = 0
            for s, v in gated.items():
                n += 1
                rule = "FAIL" if stat[s] == "FAIL" or sc[s] < 50 else "PASS"
                agree += int(rule == v)
            hyb_rows.append({"dataset": ds, "model": m, "primary_representation": prim, "windows_with_hybrid": n,
                             "equal_to_statistical_FAIL_or_score_below_50": agree})
    rc.write_csv(out / "s2_hybrid_rule_check.csv", hyb_rows)
    kc_h = [r for r in hyb_rows if r["dataset"] == rc.KC]
    F.add("hyb.rule_check.kc", f"{sum(r['equal_to_statistical_FAIL_or_score_below_50'] for r in kc_h):,}/{sum(r['windows_with_hybrid'] for r in kc_h):,}",
          "hybrid-as-scored verdicts on kind-controls-180 equal to 'statistical FAIL, or model score below 50' (11 models, primary representations)",
          KC_SRC_SHORT, "judge=hybrid, representation=gated, against the statistical verdict and the primary representation's score")
    results["s2_hybrid_rule"] = hyb_rows

    # ------------------------------------------------------------------ 3. kind rules on the pooled 360 windows (cross-session)
    pub_cfg_o180: Dict[str, Dict[str, str]] = {}
    pub_src: Dict[str, str] = {}
    pub_stat_v = {s: v for s, (v, _) in pub_stat.items()}
    pub_cfg_o180["statistical"] = pub_stat_v
    pub_src["statistical"] = "data-archive/results/corrected/results_raw.csv"
    pub_scores: Dict[str, Dict[str, float]] = {}
    for r in corr:
        if r["judge"].startswith("ai:") and not r["error"].strip():
            cid = f"ai|{r['model']}|{r['judge'][3:]}"
            pub_cfg_o180.setdefault(cid, {})[r["scenario"]] = r["verdict"]
            pub_scores.setdefault(cid, {})[r["scenario"]] = float(r["score"])
            pub_src[cid] = "data-archive/results/corrected/results_raw.csv"
        elif r["judge"].startswith("hybrid:"):
            cid = f"hybrid|{r['model']}|{r['judge'][7:]}"
            pub_cfg_o180.setdefault(cid, {})[r["scenario"]] = r["verdict"]
            pub_scores.setdefault(cid, {})[r["scenario"]] = float(r["score"])
            pub_src[cid] = "data-archive/results/corrected/results_raw.csv"
    for m in rc.HOSTED:
        rel = f"results/agg/frontier_long_{m}.csv"
        for r in rc.read_csv(dep / rel):
            if r["judge"].startswith("ai:"):
                cid = f"ai|{m}|{r['judge'][3:]}"
            elif r["judge"].startswith("hybrid:"):
                cid = f"hybrid|{m}|{r['judge'][7:]}"
            else:
                continue
            if r["error"].strip():
                continue
            pub_cfg_o180.setdefault(cid, {})[r["scenario"]] = r["verdict"]
            pub_scores.setdefault(cid, {})[r["scenario"]] = float(r["score"])
            pub_src[cid] = f"data-archive/{rel}"
    ens_pub = {r["scenario_id"]: r["verdict"] for r in rc.read_csv(dep / "analysis" / "area-D-ensemble-guards" / "ensemble_verdicts_180.csv")}
    ens_live = {s: r["ensemble_verdict"] for s, r in live[rc.O180].items()}
    results["s3_ensemble_original180_live_vs_published"] = sum(1 for s in ens_pub if ens_pub[s] == ens_live[s])
    pub_cfg_o180["ensemble"] = ens_pub
    pub_src["ensemble"] = "data-archive/analysis/area-D-ensemble-guards/ensemble_verdicts_180.csv"
    pub_cfg_o180["guard"] = guard_rule(rc.O180, pub_stat_v)
    pub_src["guard"] = "data-archive/results/corrected/results_raw.csv; " + CMP_REL + "comparators_live_original-180.csv"
    pub_cfg_o180["bfks"] = {r["scenario"]: r["verdict"] for r in rc.read_csv(s1 / "classical_verdicts.csv")}
    pub_src["bfks"] = "archive-s1/classical_verdicts.csv"
    kr_o = kind_rule(rc.O180)
    pub_cfg_o180["rule:kind"] = kr_o
    pub_src["rule:kind"] = "tools/eval_dataset.py (metric kinds)"
    pub_cfg_o180["rule:kind+stat"] = {s: ("FAIL" if kr_o[s] == "FAIL" or pub_stat_v[s] == "FAIL" else "PASS") for s in pub_stat_v}
    pub_src["rule:kind+stat"] = "data-archive/results/corrected/results_raw.csv; tools/eval_dataset.py"
    pooled_rows = []
    T360 = {**{"o:" + s: truth[rc.O180][s] for s in truth[rc.O180]}, **{"k:" + s: T[s] for s in T}}
    F360 = {**{"o:" + s: "o:" + fam[rc.O180][s] for s in fam[rc.O180]}, **{"k:" + s: "k:" + FAM[s] for s in FAM}}
    for c in kc_cfgs:
        po = pub_cfg_o180.get(c.cid)
        if po is None:
            continue
        o_bm = rc.ba_mcc_with_ci(S, po, truth[rc.O180], fam[rc.O180], c.cid)
        v360 = {**{"o:" + s: v for s, v in po.items()}, **{"k:" + s: v for s, v in c.verdicts.items()}}
        bm = rc.ba_mcc_with_ci(S, v360, T360, F360, c.cid)
        pooled_rows.append({"configuration": c.cid, "label": c.label, "cross_session": "yes" if c.cid not in ("rule:kind",) else "no (no judge involved)",
                            "original180_scored": len(po), "original180_TP": o_bm["TP"], "original180_FP": o_bm["FP"],
                            "original180_TN": o_bm["TN"], "original180_FN": o_bm["FN"],
                            "original180_BA": o_bm["BA"], "original180_MCC": o_bm["MCC"],
                            "controls_scored": len(c.verdicts), "controls_BA": kc_summary[c.cid]["BA"], "controls_MCC": kc_summary[c.cid]["MCC"],
                            "pooled_scored": len(v360), "TP": bm["TP"], "FP": bm["FP"], "TN": bm["TN"], "FN": bm["FN"],
                            "pooled_BA": bm["BA"], "pooled_BA_ci": bm["BA_ci"], "pooled_MCC": bm["MCC"], "pooled_MCC_ci": bm["MCC_ci"],
                            "original180_source": pub_src[c.cid], "controls_source": c.source})
    md_pub = pub_cfg_o180["ai|moondream-vlm|plot"]
    F.add("o180.moondream.three_metric", f"cross_metric_marginal {sum(1 for s in md_pub if fam[rc.O180][s] == 'cross_metric_marginal')}/20 scored; "
          f"gradual_drift {sum(1 for s in md_pub if fam[rc.O180][s] == 'gradual_drift')}/20 scored",
          "published Moondream plot records on original-180: windows with a scored verdict in the three-metric family and in gradual_drift",
          "data-archive/results/corrected/results_raw.csv", "judge=ai:plot; model=moondream-vlm; error empty")
    rc.write_csv(out / "s3_pooled_360.csv", pooled_rows)
    results["s3_pooled_360"] = pooled_rows
    for r in pooled_rows:
        F.add(f"pooled.{r['configuration']}", f"BA {r['pooled_BA']} {r['pooled_BA_ci']}; MCC {rc.typo(r['pooled_MCC'])} {rc.typo(r['pooled_MCC_ci'])}; "
              f"TP/FP/TN/FN {r['TP']}/{r['FP']}/{r['TN']}/{r['FN']} over {r['pooled_scored']}",
              f"{r['label']} on the pooled 360 windows (original-180 plus kind-controls-180); cross-session: original-180 verdicts are "
              "those of the published runs, control verdicts those of November 2026",
              r["original180_source"] + "; " + r["controls_source"],
              "verdicts of both sets pooled; bootstrap within the 18 fixed families, seed 20261122")
        F.add(f"o180.{r['configuration']}", f"detection {r['original180_TP']}/{r['original180_TP'] + r['original180_FN']}; false alarms "
              f"{r['original180_FP']}/{r['original180_FP'] + r['original180_TN']}; BA {r['original180_BA']}; MCC {rc.typo(r['original180_MCC'])}",
              f"{r['label']} on original-180 (published records), recomputed", r["original180_source"],
              "published verdicts against the labels; BA printed as analyze_A prints it (f'{x:.3f}' of the float)")

    # ------------------------------------------------------------------ 4. paired comparisons on the controls (42 tests)
    cfg_by = {c.cid: c for c in kc_cfgs}
    tests = []
    for m, r in rc.MODEL_CONFIGS:
        mc = cfg_by[f"ai|{m}|{r}"]
        for comp in ("rule:kind+stat", "statistical"):
            cc = cfg_by[comp]
            common = sorted(set(mc.verdicts) & set(cc.verdicts))
            a_ = b_ = c_ = d_ = 0
            for s in common:
                x = mc.verdicts[s] == T[s]
                y = cc.verdicts[s] == T[s]
                if x and y:
                    a_ += 1
                elif x:
                    b_ += 1
                elif y:
                    c_ += 1
                else:
                    d_ += 1
            th, lo, hi, _ = S["newcombe10"](a_, b_, c_, d_)
            p = S["p_mcnemar_exact"](b_, c_)
            tests.append({"model_configuration": mc.cid, "label": mc.label, "comparator": comp, "comparator_label": cc.label,
                          "n": len(common), "both_correct": a_, "model_only_correct_b": b_, "comparator_only_correct_c": c_,
                          "both_wrong": d_, "accuracy_difference": th, "diff_lo": lo, "diff_hi": hi, "p_exact": p})
    adj = S["holm_exact"]([t["p_exact"] for t in tests])
    for t, h in zip(tests, adj):
        t["p_holm42"] = h
    rows = []
    for t in tests:
        rows.append({**{k: v for k, v in t.items() if k not in ("p_exact", "p_holm42", "accuracy_difference", "diff_lo", "diff_hi")},
                     "accuracy_difference": rc.signed3(t["accuracy_difference"]), "diff_ci": f"[{rc.f3(t['diff_lo'])}, {rc.f3(t['diff_hi'])}]",
                     "p_exact": repr(float(t["p_exact"])), "p_exact_fraction": f"{t['p_exact'].numerator}/{t['p_exact'].denominator}",
                     "p_holm42": repr(float(t["p_holm42"])), "p_exact_print": rc.pfmt(t["p_exact"]), "p_holm42_print": rc.pfmt(t["p_holm42"]),
                     "holm_below_0.05": "yes" if t["p_holm42"] < Fraction(1, 20) else "no"})
    rc.write_csv(out / "s4_paired_tests_controls.csv", rows)
    results["s4_paired"] = rows
    for r in rows:
        F.add(f"paired.{r['model_configuration']}.{r['comparator']}",
              f"n {r['n']}; b/c {r['model_only_correct_b']}/{r['comparator_only_correct_c']}; accuracy difference {rc.typo(r['accuracy_difference'])} "
              f"{rc.typo(r['diff_ci'])}; exact McNemar p {r['p_exact_print']}; Holm p over 42 {r['p_holm42_print']}",
              f"{r['label']} against the {r['comparator_label'].lower()} on kind-controls-180: paired correctness",
              cfg_by[r["model_configuration"]].source, "b = model correct and comparator wrong; c = the reverse; difference model minus comparator, "
              "Newcombe method 10 (analyze_A.newcombe10); analyze_A.p_mcnemar_exact; analyze_A.holm_exact over the 42 tests")
    n_sig = sum(1 for r in rows if r["holm_below_0.05"] == "yes")
    F.add("paired.summary", f"{n_sig} of 42", "paired comparisons on the controls with Holm-adjusted p below 0.05",
          "analysis/s4_paired_tests_controls.csv", "count of holm_below_0.05 = yes")
    for comp in ("rule:kind+stat", "statistical"):
        sub = [r for r in rows if r["comparator"] == comp]
        better = sum(1 for r in sub if r["holm_below_0.05"] == "yes" and r["accuracy_difference"].startswith("+"))
        worse = sum(1 for r in sub if r["holm_below_0.05"] == "yes" and r["accuracy_difference"].startswith("-"))
        F.add(f"paired.summary.{comp}", f"model more accurate in {better}, less accurate in {worse}, of 21 (Holm p < 0.05)",
              f"direction of the significant paired differences against the {cfg_by[comp].label.lower()} on kind-controls-180",
              "analysis/s4_paired_tests_controls.csv", "sign of accuracy_difference where holm_below_0.05 = yes")

    # ------------------------------------------------------------------ 5. reading rules
    rr = protocol["reading_rules"]
    read_rows = []
    for c in kc_cfgs:
        d = kc_summary[c.cid]
        fails_f = {f_: (sum(1 for s in by_fam[f_] if c.verdicts.get(s) == "FAIL"), sum(1 for s in by_fam[f_] if s in c.verdicts))
                   for f_ in rc.KC_FAMILIES}
        rowr = {"configuration": c.cid, "label": c.label,
                "LN_fails": f"{fails_f['kc_latency_nochange'][0]}/{fails_f['kc_latency_nochange'][1]}"}
        for K in ("memory", "errors"):
            bfs = rr["benign_families_of_kind"][K]
            dfam = rr["drift_family_of_kind"][K]
            kd_fams = bfs + ["kc_latency_nochange"]
            ne_fams = bfs + [dfam]
            kd_complete = all(fails_f[f_][1] == 20 for f_ in kd_fams)
            ne_complete = all(fails_f[f_][1] == 20 for f_ in ne_fams)
            kd = any(fails_f[f_][0] >= 15 for f_ in bfs) and fails_f["kc_latency_nochange"][0] <= 5
            ne = all(fails_f[f_][0] <= 2 for f_ in bfs) and fails_f[dfam][0] >= 18
            if kd and kd_complete:
                lab = "kind-driven"
            elif ne and ne_complete:
                lab = "not explained by kind on these controls"
            else:
                lab = "no label"
            rowr[f"{K}_benign_fails"] = "; ".join(f"{rc.KC_CODE[f_]} {fails_f[f_][0]}/{fails_f[f_][1]}" for f_ in bfs)
            rowr[f"{K}_drift_detected"] = f"{fails_f[dfam][0]}/{fails_f[dfam][1]}"
            rowr[f"{K}_label"] = lab
            rowr[f"{K}_families_complete"] = "yes" if (kd_complete and ne_complete) else "no"
        read_rows.append(rowr)
    rc.write_csv(out / "s5_reading_rules.csv", read_rows)
    results["s5_reading_rules"] = read_rows
    lab_count = Counter()
    for r in read_rows:
        lab_count[(r["memory_label"], r["errors_label"])] += 1
    for r in read_rows:
        F.add(f"read.{r['configuration']}", f"memory: {r['memory_label']}; errors: {r['errors_label']}",
              f"{r['label']}: reading rules of the protocol on kind-controls-180 (benign fails: {r['memory_benign_fails']}; "
              f"{r['errors_benign_fails']}; LN {r['LN_fails']}; drift detected: memory {r['memory_drift_detected']}, errors {r['errors_drift_detected']})",
              cfg_by[r["configuration"]].source, "protocol reading_rules: kind-driven if >= 15/20 FAIL on a benign family of the kind and "
              "<= 5/20 on kc_latency_nochange; not explained by kind if <= 2/20 on every benign family of the kind and >= 18/20 detected "
              "on its drift family; a label needs 20 scored windows in every family the rule reads")
    for grp in ("model", "hybrid", "other"):
        sel = [r for r in read_rows if (r["configuration"].startswith("ai|") if grp == "model" else
                                        r["configuration"].startswith("hybrid|") if grp == "hybrid" else
                                        not r["configuration"].startswith(("ai|", "hybrid|")))]
        for K in ("memory", "errors"):
            cnt = Counter(r[f"{K}_label"] for r in sel)
            F.add(f"read.count.{grp}.{K}", "; ".join(f"{k} {v}" for k, v in sorted(cnt.items())),
                  f"reading-rule labels on kind {K} across the {len(sel)} {'model configurations' if grp == 'model' else 'hybrids' if grp == 'hybrid' else 'statistical judge and model-free comparators'}",
                  "analysis/s5_reading_rules.csv", f"count of {K}_label")

    # ------------------------------------------------------------------ 6. held-out sets
    hb_by = defaultdict(list)
    for s, f_ in fam[rc.HB].items():
        hb_by[f_].append(s)
    g_by = defaultdict(list)
    for s, f_ in fam[rc.G60].items():
        g_by[f_].append(s)
    exch = [s for f_ in rc.HB_EXCH for s in hb_by[f_]]
    rt = [s for f_ in rc.HB_RT for s in hb_by[f_]]
    g_all = sorted(fam[rc.G60])
    ho_rows, dist_rows = [], []
    ho = {}
    for m in rc.HOSTED:
        for rub in rc.RUBRICS:
            vb = recs.ai_verdicts(rc.HB, m, "raw", rub)
            sb = recs.ai_scores(rc.HB, m, "raw", rub)
            vg = recs.ai_verdicts(rc.G60, m, "raw", rub)
            sg = recs.ai_scores(rc.G60, m, "raw", rub)
            row = {"model": m, "judge": rc.PAPER_MODEL[m], "rubric": rc.RUBRIC_SHORT[rub], "prompt_id": rub,
                   "benign_scored": len(vb), "regressions_scored": len(vg)}
            for f_ in rc.HB_FAMILIES:
                k = sum(1 for s in hb_by[f_] if vb.get(s) == "FAIL")
                n = sum(1 for s in hb_by[f_] if s in vb)
                row[f"{rc.HB_CODE[f_]}_false_alarms"] = f"{k}/{n}"
                row[f"{rc.HB_CODE[f_]}_ci"] = "[{}, {}]".format(*map(rc.f3, rc.wilson(k, n)))
            ke = sum(1 for s in exch if vb.get(s) == "FAIL")
            kr = sum(1 for s in rt if vb.get(s) == "FAIL")
            row["exch_false_alarms"] = f"{ke}/{len(exch)}"
            row["exch_ci"] = "[{}, {}]".format(*map(rc.f3, rc.wilson(ke, len(exch))))
            row["rt_false_alarms"] = f"{kr}/{len(rt)}"
            row["rt_ci"] = "[{}, {}]".format(*map(rc.f3, rc.wilson(kr, len(rt))))
            for f_ in rc.G60_FAMILIES:
                k = sum(1 for s in g_by[f_] if vg.get(s) == "FAIL")
                n = sum(1 for s in g_by[f_] if s in vg)
                row[f"{rc.G60_CODE[f_]}_detected"] = f"{k}/{n}"
                row[f"{rc.G60_CODE[f_]}_ci"] = "[{}, {}]".format(*map(rc.f3, rc.wilson(k, n)))
            kg = sum(1 for s in g_all if vg.get(s) == "FAIL")
            row["det_60"] = f"{kg}/{len(g_all)}"
            row["det_ci"] = "[{}, {}]".format(*map(rc.f3, rc.wilson(kg, len(g_all))))
            for f_ in rc.HB_FAMILIES + rc.G60_FAMILIES:
                ids = hb_by[f_] if f_ in hb_by else g_by[f_]
                scs = [(sb if f_ in hb_by else sg)[s] for s in ids if s in (sb if f_ in hb_by else sg)]
                bins = Counter(min(int(x // 5), 19) for x in scs)
                dist_rows.append({"model": m, "rubric": rc.RUBRIC_SHORT[rub], "family": f_, "n": len(scs),
                                  "median": statistics.median(scs) if scs else "", "min": min(scs) if scs else "",
                                  "max": max(scs) if scs else "", "below_75": sum(1 for x in scs if x < 75),
                                  "in_75_80": sum(1 for x in scs if 75 <= x < 80), "in_80_85": sum(1 for x in scs if 80 <= x < 85),
                                  **{f"bin_{5 * i:02d}": bins.get(i, 0) for i in range(20)}})
            ho[(m, rub)] = {"vb": vb, "vg": vg, "sb": sb, "sg": sg, "row": row}
            ho_rows.append(row)
    rc.write_csv(out / "s6_heldout_counts.csv", ho_rows)
    rc.write_csv(out / "s6_heldout_score_distributions.csv", dist_rows)
    results["s6_heldout"] = ho_rows
    for row in ho_rows:
        m, rub = row["model"], row["prompt_id"]
        srcf = recs.source_files(rc.HB, m, rub) + "; " + recs.source_files(rc.G60, m, rub)
        F.add(f"ho.{m}.{row['rubric']}.benign", "; ".join(f"{rc.HB_CODE[f_]} {row[rc.HB_CODE[f_] + '_false_alarms']} {row[rc.HB_CODE[f_] + '_ci']}" for f_ in rc.HB_FAMILIES),
              f"{row['judge']} on raw series, rubric {row['rubric']}: false alarms per held-out benign family (SP hb_shared_partial_recovery, "
              "SL hb_shared_late_step, IS hb_isolated_spikes_both, RM hb_recovered_mid_transient, BB hb_brief_early_blip), Wilson 95%",
              srcf, f"judge=ai; representation=raw; prompt_id={rub}; dataset heldout-benign-100; FAIL verdicts")
        F.add(f"ho.{m}.{row['rubric']}.exch", f"{row['exch_false_alarms']} {row['exch_ci']}",
              f"{row['judge']} on raw series, rubric {row['rubric']}: false alarms over the 60 exchangeable held-out benign windows",
              srcf, "FAIL verdicts on hb_shared_partial_recovery, hb_shared_late_step, hb_isolated_spikes_both; Wilson 95%")
        F.add(f"ho.{m}.{row['rubric']}.rt", f"{row['rt_false_alarms']} {row['rt_ci']}",
              f"{row['judge']} on raw series, rubric {row['rubric']}: false alarms over the 40 recovered-transient windows (contested label)",
              srcf, "FAIL verdicts on hb_recovered_mid_transient and hb_brief_early_blip; Wilson 95%")
        F.add(f"ho.{m}.{row['rubric']}.det", f"{row['det_60']} {row['det_ci']}; " + "; ".join(f"{rc.G60_CODE[f_]} {row[rc.G60_CODE[f_] + '_detected']}" for f_ in rc.G60_FAMILIES),
              f"{row['judge']} on raw series, rubric {row['rubric']}: detection of the 60 held-out regressions in the November session "
              "(PR partial_recovery, LT late_transient, SE sustained_excursion)", srcf,
              f"judge=ai; representation=raw; prompt_id={rub}; dataset generalisation-60; FAIL verdicts; Wilson 95%")
    for dr in dist_rows:
        F.add(f"dist.{dr['model']}.{dr['rubric']}.{dr['family']}", f"median {dr['median']}; min {dr['min']}; max {dr['max']}; below 75: {dr['below_75']}/{dr['n']}",
              f"{rc.PAPER_MODEL[dr['model']]} raw series, rubric {dr['rubric']}, {dr['family']}: score distribution",
              REC + ("H3-" if dr["family"] in hb_by else "H4-") + f"{dr['model']}-{dr['rubric']}/", "scores of the scored ai rows")

    # rubric effects: 3 models x (v2, v3 against v1) x (RT false alarms, exchangeable false alarms, detection)
    eff = []
    for m in rc.HOSTED:
        for var in (rc.RUBRICS[1], rc.RUBRICS[2]):
            for meas, ids, ds in (("false alarms, 40 recovered-transient windows", rt, rc.HB),
                                  ("false alarms, 60 exchangeable windows", exch, rc.HB),
                                  ("detection, 60 held-out regressions", g_all, rc.G60)):
                vv = ho[(m, var)]["vb" if ds == rc.HB else "vg"]
                v1 = ho[(m, rc.V1)]["vb" if ds == rc.HB else "vg"]
                common = [s for s in ids if s in vv and s in v1]
                a_ = sum(1 for s in common if vv[s] == "FAIL" and v1[s] == "FAIL")
                b_ = sum(1 for s in common if vv[s] == "FAIL" and v1[s] != "FAIL")
                c_ = sum(1 for s in common if vv[s] != "FAIL" and v1[s] == "FAIL")
                d_ = len(common) - a_ - b_ - c_
                th, lo, hi, _ = S["newcombe10"](a_, b_, c_, d_)
                p = S["p_mcnemar_exact"](b_, c_)
                eff.append({"model": m, "judge": rc.PAPER_MODEL[m], "contrast": f"{rc.RUBRIC_SHORT[var]} vs v1", "measure": meas, "n": len(common),
                            "v1_FAIL": a_ + c_, "variant_FAIL": a_ + b_, "both_FAIL": a_, "variant_only_FAIL_b": b_, "v1_only_FAIL_c": c_,
                            "neither": d_, "difference": th, "lo": lo, "hi": hi, "p": p})
    adj = S["holm_exact"]([e["p"] for e in eff])
    eff_rows = []
    for e, h in zip(eff, adj):
        eff_rows.append({**{k: v for k, v in e.items() if k not in ("difference", "lo", "hi", "p")},
                         "difference": rc.signed3(e["difference"]), "ci": f"[{rc.f3(e['lo'])}, {rc.f3(e['hi'])}]",
                         "p_exact": repr(float(e["p"])), "p_print": rc.pfmt(e["p"]), "p_holm18": repr(float(h)),
                         "p_holm18_print": rc.pfmt(h), "holm_below_0.05": "yes" if h < Fraction(1, 20) else "no"})
    rc.write_csv(out / "s6_rubric_effects.csv", eff_rows)
    results["s6_rubric_effects"] = eff_rows
    for e in eff_rows:
        F.add(f"rubric.{e['model']}.{e['contrast'].split()[0]}.{'rt' if 'recovered' in e['measure'] else 'exch' if 'exchangeable' in e['measure'] else 'det'}",
              f"{e['variant_FAIL']}/{e['n']} against {e['v1_FAIL']}/{e['n']}; difference {rc.typo(e['difference'])} {rc.typo(e['ci'])}; "
              f"b/c {e['variant_only_FAIL_b']}/{e['v1_only_FAIL_c']}; exact McNemar p {e['p_print']}; Holm p over 18 {e['p_holm18_print']}",
              f"{e['judge']} raw series, rubric {e['contrast']}: {e['measure']} (FAIL verdicts, variant against v1, paired on the same windows)",
              REC + f"H3-{e['model']}-v*/ and H4-{e['model']}-v*/", "Newcombe method 10 (analyze_A.newcombe10), variant minus v1; "
              "analyze_A.p_mcnemar_exact on the discordant pairs; analyze_A.holm_exact over the 18 tests")
    F.add("rubric.summary", f"{sum(1 for e in eff_rows if e['holm_below_0.05'] == 'yes')} of 18",
          "rubric-effect tests with Holm-adjusted p below 0.05", "analysis/s6_rubric_effects.csv", "count")

    # P6 and P7 on the 40 recovered-transient windows (the post-hoc label-policy code, unchanged)
    d2_series = {s: {"family": fam[rc.HB][s], "label": truth[rc.HB][s],
                     "metrics": [{"name": m["name"], "kind": m["kind"], "control": m["control"], "canary": m["canary"]}
                                 for m in series[rc.HB][s]["metrics"]]} for s in rt}
    feats = rc.pc.realised(d2_series)
    labels = {s: truth[rc.HB][s] for s in rt}
    fam_rt = {s: fam[rc.HB][s] for s in rt}
    pol = {pid: d2.policy_labels(pid, labels, fam_rt, feats) for pid in ("P6", "P7")}
    chk = {r["id"]: r for r in rc.read_csv(rc.REPO / "protocol" / f"realised_checks_{rc.HB}.csv")}
    agree6 = sum(1 for s in rt if (float(chk[s]["mean_ratio"]) > 1.05) == (pol["P6"][s] == "FAIL"))
    agree7 = sum(1 for s in rt if (float(chk[s]["last5_mean_over_control_mean"]) > 1.05) == (pol["P7"][s] == "FAIL"))
    pol_rows = []
    for pid in ("P6", "P7"):
        changed = {f_: sum(1 for s in hb_by[f_] if pol[pid][s] != labels[s]) for f_ in rc.HB_RT}
        for m in rc.HOSTED:
            for rub in rc.RUBRICS:
                vb = ho[(m, rub)]["vb"]
                still = [s for s in rt if pol[pid][s] == "PASS"]
                now_fail = [s for s in rt if pol[pid][s] == "FAIL"]
                fa = sum(1 for s in still if vb.get(s) == "FAIL")
                de = sum(1 for s in now_fail if vb.get(s) == "FAIL")
                pol_rows.append({"policy": pid, "model": m, "rubric": rc.RUBRIC_SHORT[rub],
                                 "labels_changed": sum(changed.values()),
                                 **{f"changed_{rc.HB_CODE[f_]}": changed[f_] for f_ in rc.HB_RT},
                                 "windows_PASS": len(still), "false_alarms": f"{fa}/{len(still)}",
                                 "windows_FAIL": len(now_fail), "detected": f"{de}/{len(now_fail)}"})
    rc.write_csv(out / "s6_policies_p6_p7.csv", pol_rows)
    results["s6_policies"] = {"rows": pol_rows, "P6_agrees_with_protocol_realised_checks": f"{agree6}/40",
                              "P7_agrees_with_protocol_realised_checks": f"{agree7}/40"}
    for pid in ("P6", "P7"):
        r0 = [r for r in pol_rows if r["policy"] == pid][0]
        F.add(f"policy.{pid}.changed", f"{r0['labels_changed']} of 40 (RM {r0['changed_RM']}/20, BB {r0['changed_BB']}/20)",
              f"recovered-transient windows of heldout-benign-100 whose label changes from PASS to FAIL under {pid} "
              f"({'realised mean ratio of a metric above 1.05' if pid == 'P6' else 'mean of the last five canary samples above 1.05 times the control mean'})",
              f"protocol/series_{rc.HB}.jsonl; protocol/realised_checks_{rc.HB}.csv",
              "analysis/r1/posthoc/d2_label_policies.policy_labels with posthoc_common.realised, unchanged")
        for r in [x for x in pol_rows if x["policy"] == pid]:
            F.add(f"policy.{pid}.{r['model']}.{r['rubric']}", f"false alarms {r['false_alarms']}; detected {r['detected']}",
                  f"{rc.PAPER_MODEL[r['model']]} raw series, rubric {r['rubric']}, the 40 recovered-transient windows labelled under {pid}",
                  REC + f"H3-{r['model']}-{r['rubric']}/", "windows still PASS: FAIL verdicts are false alarms; windows now FAIL: FAIL verdicts are detections")

    # model-free comparators on heldout-benign-100 and generalisation-60
    cmp_rows = []
    for ds in (rc.HB, rc.G60):
        st_v = recs.stat_verdicts(ds)
        for c in comparators(ds, st_v, recs.stat_scores(ds), REC + ("H3-*/" if ds == rc.HB else "H4-*/")):
            row = {"dataset": ds, "configuration": c.cid, "label": c.label, "scored": len(c.verdicts)}
            fams_ = rc.HB_FAMILIES if ds == rc.HB else rc.G60_FAMILIES
            byf = hb_by if ds == rc.HB else g_by
            for f_ in fams_:
                k = sum(1 for s in byf[f_] if c.verdicts.get(s) == "FAIL")
                row[f_] = f"{k}/{len(byf[f_])}"
            if ds == rc.HB:
                ke = sum(1 for s in exch if c.verdicts[s] == "FAIL")
                kr = sum(1 for s in rt if c.verdicts[s] == "FAIL")
                row["exchangeable_false_alarms"] = f"{ke}/60 [{rc.f3(rc.wilson(ke, 60)[0])}, {rc.f3(rc.wilson(ke, 60)[1])}]"
                row["recovered_transient_false_alarms"] = f"{kr}/40 [{rc.f3(rc.wilson(kr, 40)[0])}, {rc.f3(rc.wilson(kr, 40)[1])}]"
                F.add(f"cmp.{ds}.{c.cid}", f"exchangeable {row['exchangeable_false_alarms']}; recovered transient {row['recovered_transient_false_alarms']}; "
                      + "; ".join(f"{rc.HB_CODE[f_]} {row[f_]}" for f_ in fams_),
                      f"{c.label} on heldout-benign-100: false alarms", c.source, c.filt)
            else:
                kg = sum(1 for s in g_all if c.verdicts[s] == "FAIL")
                row["detected_60"] = f"{kg}/60 [{rc.f3(rc.wilson(kg, 60)[0])}, {rc.f3(rc.wilson(kg, 60)[1])}]"
                F.add(f"cmp.{ds}.{c.cid}", f"detected {row['detected_60']}; " + "; ".join(f"{rc.G60_CODE[f_]} {row[f_]}" for f_ in fams_),
                      f"{c.label} on generalisation-60: detection", c.source, c.filt)
            cmp_rows.append(row)
    rc.write_csv(out / "s6_comparators_heldout.csv", cmp_rows, columns=["dataset", "configuration", "label", "scored"] + rc.HB_FAMILIES
                 + ["exchangeable_false_alarms", "recovered_transient_false_alarms"] + rc.G60_FAMILIES + ["detected_60"])
    results["s6_comparators_heldout"] = cmp_rows

    # ------------------------------------------------------------------ 7. session anchors
    anchors = sorted(s for s in truth[rc.O180] if s.endswith(("_000", "_001")))
    anc_rows, anc_win = [], []
    exp_dir = dep / "results" / "agg" / "experiments"
    # dates of the published runs, read from their records
    import glob as _glob
    n20_manifest = json.loads((dep / "reference-results" / "n20" / "eval_manifest.json").read_text(encoding="utf-8"))
    def ts_range(path):
        ts = sorted(r["ts"] for r in rc.read_csv(path))
        return ts[0], ts[-1]
    rerun_md = ts_range(next(Path(p) for p in sorted(_glob.glob(str(exp_dir / "moondream-vlm__v1-frozen-2026-06__original-180__*.csv")))))
    rerun_ds = ts_range(exp_dir / "deepseek-r1-llm__v1-frozen-2026-06__original-180__20260729T225610Z.csv")
    imp = ts_range(exp_dir / "published__claude-opus-4-8-vlm__v1-frozen-2026-06__original-180.csv")
    pub_dates = {}
    for m in rc.LOCAL:
        pub_dates[m] = (f"2026-06-21 (reference run, generated_at {n20_manifest['generated_at']} in "
                        "data-archive/reference-results/n20/eval_manifest.json)")
    pub_dates["moondream-vlm"] += f"; plots re-measured {rerun_md[0]} to {rerun_md[1]}"
    pub_dates["deepseek-r1-llm"] += f"; summaries re-measured {rerun_ds[0]} to {rerun_ds[1]}"
    for m in rc.HOSTED:
        pub_dates[m] = (f"not recorded: data-archive/results/agg/frontier_long_{m}.csv has no time column; its long-format copy "
                        f"carries only the import stamp {imp[0]} (an upper bound on the run date)")
    # moondream and deepseek re-measured rows equal the corrected records on the anchors (and everywhere)
    def rows_of(path, judge, model):
        return {(r["scenario_id"]): (r["verdict"], float(r["score"])) for r in rc.read_csv(path)
                if r["judge"] == judge.split(":")[0] and r["model"] == model and r["representation"] == judge.split(":")[1]
                and not r["error_kind"].strip()}
    remeasured_check = {}
    for m, rep, path in (("moondream-vlm", "plot", next(Path(p) for p in sorted(_glob.glob(str(exp_dir / "moondream-vlm__v1-frozen-2026-06__original-180__*.csv"))))),
                         ("deepseek-r1-llm", "summary", exp_dir / "deepseek-r1-llm__v1-frozen-2026-06__original-180__20260729T225610Z.csv")):
        rr_ = rows_of(path, f"ai:{rep}", m)
        pub = {s: (pub_cfg_o180[f"ai|{m}|{rep}"][s], pub_scores[f"ai|{m}|{rep}"][s]) for s in pub_cfg_o180[f"ai|{m}|{rep}"]}
        # the corrected records print scores to two decimals; the re-measured run file keeps them unrounded
        remeasured_check[f"{m}|{rep}"] = {"published_rows": len(pub), "re-measured_rows": len(rr_),
                                          "identical_verdicts": sum(1 for s in pub if s in rr_ and rr_[s][0] == pub[s][0]),
                                          "identical_scores_at_two_decimals": sum(1 for s in pub if s in rr_ and round(rr_[s][1], 2) == round(pub[s][1], 2)),
                                          "re-measured_file": "data-archive/results/agg/experiments/" + Path(path).name}
    results["s7_remeasured_check"] = remeasured_check
    all_cfgs = [(f"ai|{m}|{r}", m) for m, r in rc.MODEL_CONFIGS] + [(f"hybrid|{m}|{p}", m) for m, p in rc.HYBRID_CONFIGS]
    for cid, m in all_cfgs:
        kind_, _, rp = cid.split("|")
        if kind_ == "ai":
            nv, ns = recs.ai_verdicts(rc.O180, m, rp), recs.ai_scores(rc.O180, m, rp)
            failed_ids = recs.ai_failed_ids(rc.O180, m, rp)
        else:
            nv, ns = recs.hyb_verdicts(rc.O180, m, rp), recs.hyb_scores(rc.O180, m, rp)
            failed_ids = []
        pv, ps = pub_cfg_o180.get(cid, {}), pub_scores.get(cid, {})
        common = [s for s in anchors if s in nv and s in pv]
        same = sum(1 for s in common if nv[s] == pv[s])
        mad = sum(abs(ns[s] - ps[s]) for s in common) / len(common) if common else None
        same_score = sum(1 for s in common if round(ns[s], 2) == round(ps[s], 2))  # published local scores carry two decimals
        for s in anchors:
            anc_win.append({"configuration": cid, "scenario_id": s, "november_verdict": nv.get(s, "failed call" if s in failed_ids else ""),
                            "november_score": ns.get(s, ""), "published_verdict": pv.get(s, "no published record"),
                            "published_score": ps.get(s, ""), "same_verdict": ("yes" if nv.get(s) == pv.get(s) else "no") if s in common else ""})
        anc_rows.append({"configuration": cid, "label": rc.config_label(m, rp, hybrid=(kind_ == "hybrid")), "kind": kind_,
                         "anchor_windows": len(anchors), "november_scored": sum(1 for s in anchors if s in nv),
                         "published_scored": sum(1 for s in anchors if s in pv), "compared": len(common),
                         "identical_verdicts": same, "identical": f"{same}/{len(common)}",
                         "identical_scores": same_score, "mean_abs_score_difference": f"{mad:.2f}" if mad is not None else "",
                         "differing_windows": ";".join(s for s in common if nv[s] != pv[s]),
                         "november_run": REC + ("H1-" if m in rc.HOSTED else "L1-") + m + "/",
                         "published_records": pub_src.get(cid, ""), "published_run_date": pub_dates[m]})
    rc.write_csv(out / "s7_anchors_by_configuration.csv", anc_rows)
    rc.write_csv(out / "s7_anchors_by_window.csv", anc_win)
    # H4 against the published generalisation-60 runs
    h4_rows = []
    for m in rc.HOSTED:
        for rub in rc.RUBRICS:
            pf = next(Path(p) for p in sorted(_glob.glob(str(exp_dir / f"{m}__{rub}__generalisation-60__*.csv"))))
            prow = [r for r in rc.read_csv(pf)]
            pv = {r["scenario_id"]: r["verdict"] for r in prow if r["judge"] == "ai" and r["representation"] == "raw" and not r["error_kind"].strip()}
            ps = {r["scenario_id"]: float(r["score"]) for r in prow if r["judge"] == "ai" and r["representation"] == "raw" and not r["error_kind"].strip()}
            pst = {r["scenario_id"]: (r["verdict"], float(r["score"])) for r in prow if r["judge"] == "statistical"}
            nv = ho[(m, rub)]["vg"]
            ns = ho[(m, rub)]["sg"]
            common = sorted(set(pv) & set(nv))
            same = sum(1 for s in common if nv[s] == pv[s])
            mad = sum(abs(ns[s] - ps[s]) for s in common) / len(common)
            nst = recs.stat_verdicts(rc.G60)
            nss = recs.stat_scores(rc.G60)
            stat_same = sum(1 for s in pst if pst[s] == (nst[s], nss[s]))
            ts = sorted(r["ts"] for r in prow)
            nts = sorted(r["ts"] for r in recs.rows if r["_inv"] == f"H4-{m}-{rc.RUBRIC_SHORT[rub]}")
            pub_det = sum(1 for v in pv.values() if v == "FAIL")
            h4_rows.append({"model": m, "judge": rc.PAPER_MODEL[m], "rubric": rc.RUBRIC_SHORT[rub], "compared": len(common),
                            "identical": f"{same}/{len(common)}", "identical_verdicts": same,
                            "mean_abs_score_difference": f"{mad:.2f}", "published_detected": f"{pub_det}/{len(pv)}",
                            "november_detected": f"{sum(1 for v in nv.values() if v == 'FAIL')}/{len(nv)}",
                            "statistical_identical_to_published": f"{stat_same}/{len(pst)}",
                            "published_file": "data-archive/results/agg/experiments/" + pf.name,
                            "published_first_row": ts[0], "published_last_row": ts[-1],
                            "november_first_row": nts[0], "november_last_row": nts[-1],
                            "differing_windows": ";".join(s for s in common if nv[s] != pv[s])})
    rc.write_csv(out / "s7_anchors_h4.csv", h4_rows)
    results["s7_anchors"] = {"by_configuration": anc_rows, "h4": h4_rows, "published_dates": pub_dates}
    for r in anc_rows:
        F.add(f"anchor.{r['configuration']}", f"{r['identical']} identical verdicts; mean absolute score difference {r['mean_abs_score_difference']}",
              f"{r['label']}: November verdicts against the published ones on the anchor windows (_000 and _001 of each of the nine "
              f"families; {r['compared']} with a record in both); published run: {r['published_run_date']}",
              r["november_run"] + "; " + r["published_records"], "verdicts compared per window; |November score - published score| averaged")
    for r in h4_rows:
        F.add(f"anchor.h4.{r['model']}.{r['rubric']}", f"{r['identical']} identical verdicts; mean absolute score difference {r['mean_abs_score_difference']}; "
              f"published detection {r['published_detected']}, November {r['november_detected']}",
              f"{r['judge']} raw series, rubric {r['rubric']}, the 60 held-out regressions: November (H4, {r['november_first_row']} to "
              f"{r['november_last_row']}) against the published run ({r['published_first_row']} to {r['published_last_row']})",
              REC + f"H4-{r['model']}-{r['rubric']}/; {r['published_file']}", "judge=ai, representation=raw; verdicts per window")
    ident = [r for r in anc_rows if r["kind"] == "ai"]
    F.add("anchor.summary.models", f"{sum(r['identical_verdicts'] for r in ident)}/{sum(r['compared'] for r in ident)}",
          "identical verdicts over the 21 model configurations on the anchor windows (November against published)",
          "analysis/s7_anchors_by_configuration.csv", "sum over kind=ai rows")
    F.add("anchor.summary.h4", f"{sum(r['identical_verdicts'] for r in h4_rows)}/{sum(r['compared'] for r in h4_rows)}",
          "identical verdicts over the nine hosted raw-series runs on the 60 held-out regressions (November against published)",
          "analysis/s7_anchors_h4.csv", "sum over rows")

    # ------------------------------------------------------------------ 8. tables for the writers
    tables = out / "tables"
    order9 = ["statistical", "rule:kind", "rule:kind+stat", "ensemble", "guard", "bfks"] + [f"ai|{m}|{r}" for m, r in rc.MODEL_CONFIGS]

    def cnt_cell(k, n, full):
        if n == 0:
            return "no data"
        return str(k) if n == full else f"{k}/{n}"

    t9, t9ci = [], []
    for cid in order9:
        d = kc_summary[cid]
        cells = [cnt_cell(d["cells"][f_][0], d["cells"][f_][1], 20) for f_ in rc.KC_FAMILIES]
        base = {"Judge (representation)": cfg_by[cid].label, **dict(zip([rc.KC_CODE[f_] for f_ in rc.KC_FAMILIES], cells)),
                "FA": cnt_cell(d["false_alarms"], d["benign_scored"], 100), "Det.": cnt_cell(d["detected"], d["regressions_scored"], 80)}
        t9.append({**base, "BA": d["BA"], "MCC": rc.typo(d["MCC"])})
        ba_ci = d["BA_ci"].replace(", ", ", ")
        mcc_ci = rc.typo(d["MCC_ci"]).replace(", ", ", ")
        t9ci.append({"Judge": cfg_by[cid].label, **{k: v for k, v in base.items() if k != "Judge (representation)"},
                     "BA [95% CI]": f"{d['BA']} {ba_ci}", "MCC [95% CI]": f"{rc.typo(d['MCC'])} {mcc_ci}"})
    rc.write_csv(tables / "table9.csv", t9)
    rc.write_csv(tables / "table9_ci.csv", t9ci)

    t10 = []
    for m in rc.HOSTED:
        for rub in rc.RUBRICS:
            row = ho[(m, rub)]["row"]
            h4 = [r for r in h4_rows if r["model"] == m and r["rubric"] == rc.RUBRIC_SHORT[rub]][0]
            t10.append({"Judge (raw series)": rc.PAPER_MODEL[m], "Rubric": rc.RUBRIC_SHORT[rub],
                        **{rc.HB_CODE[f_]: row[rc.HB_CODE[f_] + "_false_alarms"].split("/")[0] for f_ in rc.HB_FAMILIES},
                        "Exch.": row["exch_false_alarms"].split("/")[0], "Rec.": row["rt_false_alarms"].split("/")[0],
                        **{rc.G60_CODE[f_]: row[rc.G60_CODE[f_] + "_detected"].split("/")[0] for f_ in rc.G60_FAMILIES},
                        "Det.": row["det_60"].split("/")[0], "Publ.": h4["published_detected"].split("/")[0]})
    rc.write_csv(tables / "table10.csv", t10)
    rc.write_csv(tables / "table10_note_dates.csv", [{"judge": r["judge"], "rubric": r["rubric"],
                                                      "published_run": f"{r['published_first_row']} to {r['published_last_row']}",
                                                      "november_run": f"{r['november_first_row']} to {r['november_last_row']}",
                                                      "published_file": r["published_file"]} for r in h4_rows])

    # supplementary: parameters and realised checks of the fourteen new families
    chk_rows = rc.read_csv(rc.REPO / "protocol" / f"realised_checks_{rc.KC}.csv") + rc.read_csv(rc.REPO / "protocol" / f"realised_checks_{rc.HB}.csv")
    fam_def = {f_["family"]: (ds, f_) for ds in (rc.KC, rc.HB) for f_ in protocol["datasets"][ds]["families_in_order"]}
    gen_txt = protocol["generators"]
    srows = []
    for f_ in rc.KC_FAMILIES + rc.HB_FAMILIES:
        ds, fd = fam_def[f_]
        crs = [r for r in chk_rows if r["family"] == f_]
        parts = []
        for kind in fd["metric_kinds"]:
            kr_ = [r for r in crs if r["kind"] == kind]
            mr = [float(r["mean_ratio"]) for r in kr_]
            sd = [float(r["sd_ratio"]) for r in kr_]
            mx = [float(r["max_ratio"]) for r in kr_]
            l5 = [float(r["last5_mean_over_control_mean"]) for r in kr_]
            eq = sum(int(r["tost_equivalent_5pct"]) for r in kr_)
            parts.append(f"{kind}: mean ratio {min(mr):.4f}–{max(mr):.4f}; SD ratio {min(sd):.3f}–{max(sd):.3f}; "
                         f"maximum ratio {min(mx):.3f}–{max(mx):.3f}; last-5 ratio {min(l5):.3f}–{max(l5):.3f}; "
                         f"TOST equivalent in {eq}/{len(kr_)}")
        gname = fd["generator"]
        srows.append({"Family": f_, "Dataset": ds, "Label": fd["truth"], "Metric kinds": ", ".join(fd["metric_kinds"]),
                      "Generator": gname, "Parameters": gen_txt.get(gname, ""), "Realised checks (20 windows)": "; ".join(parts)})
        F.add(f"family.{f_}.checks", "; ".join(parts), f"realised checks of {f_} over its 20 windows (ranges; TOST at 5% of the control mean)",
              f"protocol/realised_checks_{ds}.csv", f"family={f_}; min and max per metric kind; count of tost_equivalent_5pct = 1")
    rc.write_csv(tables / "tableS_new_families.csv", srows)

    # supplementary: full per-family results with intervals (kind controls)
    order_s = order9 + [f"hybrid|{m}|{p}" for m, p in rc.HYBRID_CONFIGS]
    fullf, fulls = [], []
    for cid in order_s:
        d = kc_summary[cid]
        c = cfg_by[cid]
        fullf.append({"Configuration": c.label_s, **{rc.KC_CODE[f_]: rc.kn_ci(*d["cells"][f_]) for f_ in rc.KC_FAMILIES}})
        fulls.append({"Configuration": c.label_s, "Scored": d["scored"],
                      "False alarms [95% CI]": rc.kn_ci(d["false_alarms"], d["benign_scored"]),
                      "By kind: latency; memory; errors; three-metric": "; ".join(d["fa_" + kk] for kk in rc.KC_KIND_GROUPS),
                      "Detection, 80 [95% CI]": rc.kn_ci(d["detected"], d["regressions_scored"]),
                      "Drift, 60 [95% CI]": rc.kn_ci(d["drift_detected"], d["drift_scored"]),
                      "TM": f"{d['tri_marginal_detected']}/{d['tri_marginal_scored']}" if d["tri_marginal_scored"] else "no data",
                      "BA [95% CI]": f"{d['BA']} {d['BA_ci']}", "MCC [95% CI]": rc.typo(f"{d['MCC']} {d['MCC_ci']}")})
    rc.write_csv(tables / "tableS_controls_per_family.csv", fullf)
    rc.write_csv(tables / "tableS_controls_summary.csv", fulls)
    rc.write_csv(tables / "tableS_reading_rules.csv", [{"Configuration": cfg_by[r["configuration"]].label_s,
                                                        "kc_latency_nochange FAIL": r["LN_fails"],
                                                        "Memory benign FAIL": r["memory_benign_fails"], "Memory drift detected": r["memory_drift_detected"],
                                                        "Memory reading": r["memory_label"],
                                                        "Errors benign FAIL": r["errors_benign_fails"], "Errors drift detected": r["errors_drift_detected"],
                                                        "Errors reading": r["errors_label"]} for r in sorted(read_rows, key=lambda x: order_s.index(x["configuration"]))])
    rc.write_csv(tables / "tableS_paired_tests.csv", [{"Model configuration": cfg_by[r["model_configuration"]].label_s,
                                                       "Against": r["comparator_label"], "n": r["n"],
                                                       "b / c": f"{r['model_only_correct_b']}/{r['comparator_only_correct_c']}",
                                                       "Accuracy difference [95% CI]": rc.typo(f"{r['accuracy_difference']} {r['diff_ci']}"),
                                                       "Exact McNemar p": r["p_exact_print"], "Holm p over 42": r["p_holm42_print"]} for r in rows])
    rc.write_csv(tables / "tableS_session_anchors.csv",
                 [{"Configuration": (cfg_by[r["configuration"]].label_s if r["configuration"] in cfg_by else r["label"]),
                   "Windows compared": r["compared"], "Identical verdicts": r["identical"],
                   "Mean absolute score difference": r["mean_abs_score_difference"], "Published run": r["published_run_date"]}
                  for r in anc_rows]
                 + [{"Configuration": f"{r['judge']}, raw series, rubric {r['rubric']}, 60 held-out regressions", "Windows compared": r["compared"],
                     "Identical verdicts": r["identical"], "Mean absolute score difference": r["mean_abs_score_difference"],
                     "Published run": f"{r['published_first_row']} to {r['published_last_row']}"} for r in h4_rows])
    inv_conf = []
    for ds in (rc.O180, rc.KC, rc.HB, rc.G60):
        for m in rc.TABLE_MODEL_ORDER:
            for rep in rc.REPS[m]:
                for rub in rc.RUBRICS:
                    okn = len(recs.ai_verdicts(ds, m, rep, rub))
                    fl = recs.ai_failed_ids(ds, m, rep, rub)
                    if okn + len(fl) == 0:
                        continue
                    inv_conf.append({"Dataset": ds, "Configuration": rc.config_label_s(m, rep), "Rubric": rc.RUBRIC_SHORT[rub],
                                     "Judgements": okn + len(fl), "Scored": okn, "Failed calls": len(fl),
                                     "Failed windows": (", ".join(sorted({fam[ds][s] for s in fl})) + f" ({len(fl)})") if fl else ""})
    rc.write_csv(tables / "tableS_failed_calls.csv", inv_conf)
    hfull = []
    for row in ho_rows:
        h4 = [r for r in h4_rows if r["model"] == row["model"] and r["rubric"] == row["rubric"]][0]
        hfull.append({"Judge": f"{row['judge']}, raw series", "Rubric": row["rubric"],
                      **{rc.HB_CODE[f_]: f"{row[rc.HB_CODE[f_] + '_false_alarms']} {row[rc.HB_CODE[f_] + '_ci']}" for f_ in rc.HB_FAMILIES},
                      "Exchangeable, 60": f"{row['exch_false_alarms']} {row['exch_ci']}",
                      "Recovered transient, 40": f"{row['rt_false_alarms']} {row['rt_ci']}",
                      **{rc.G60_CODE[f_]: f"{row[rc.G60_CODE[f_] + '_detected']} {row[rc.G60_CODE[f_] + '_ci']}" for f_ in rc.G60_FAMILIES},
                      "Detection, 60": f"{row['det_60']} {row['det_ci']}",
                      "Published detection, 60": h4["published_detected"]})
    rc.write_csv(tables / "tableS_heldout_full.csv", hfull)
    rc.write_csv(tables / "tableS_rubric_effects.csv", [{"Judge": f"{e['judge']}, raw series", "Contrast": e["contrast"],
                                                         "Measure": e["measure"], "n": e["n"],
                                                         "FAIL, variant / v1": f"{e['variant_FAIL']}/{e['v1_FAIL']}",
                                                         "b / c": f"{e['variant_only_FAIL_b']}/{e['v1_only_FAIL_c']}",
                                                         "Difference [95% CI]": rc.typo(f"{e['difference']} {e['ci']}"),
                                                         "Exact McNemar p": e["p_print"], "Holm p over 18": e["p_holm18_print"]} for e in eff_rows])
    rc.write_csv(tables / "tableS_policies_p6_p7.csv", [{"Policy": r["policy"], "Judge": f"{rc.PAPER_MODEL[r['model']]}, raw series",
                                                         "Rubric": r["rubric"], "Labels changed (RM; BB)": f"{r['labels_changed']} ({r['changed_RM']}; {r['changed_BB']})",
                                                         "False alarms": r["false_alarms"], "Detected": r["detected"]} for r in pol_rows])
    rc.write_csv(tables / "tableS_comparators_heldout.csv", [{"Dataset": r["dataset"], "Configuration": r["label"],
                                                              **({rc.HB_CODE[f_]: r[f_] for f_ in rc.HB_FAMILIES} if r["dataset"] == rc.HB else {}),
                                                              **({"Exchangeable, 60": r["exchangeable_false_alarms"], "Recovered transient, 40": r["recovered_transient_false_alarms"]} if r["dataset"] == rc.HB else {}),
                                                              **({rc.G60_CODE[f_]: r[f_] for f_ in rc.G60_FAMILIES} if r["dataset"] == rc.G60 else {}),
                                                              **({"Detection, 60": r["detected_60"]} if r["dataset"] == rc.G60 else {})}
                                                             for r in cmp_rows],
                 columns=["Dataset", "Configuration"] + [rc.HB_CODE[f_] for f_ in rc.HB_FAMILIES] + ["Exchangeable, 60", "Recovered transient, 40"]
                 + [rc.G60_CODE[f_] for f_ in rc.G60_FAMILIES] + ["Detection, 60"])
    rc.write_csv(tables / "tableS_pooled_360.csv", [{"Configuration": cfg_by[r["configuration"]].label_s,
                                                     "original-180 BA / MCC": rc.typo(f"{r['original180_BA']} / {r['original180_MCC']}"),
                                                     "kind controls BA / MCC": rc.typo(f"{r['controls_BA']} / {r['controls_MCC']}"),
                                                     "Pooled n": r["pooled_scored"], "Pooled BA [95% CI]": f"{r['pooled_BA']} {r['pooled_BA_ci']}",
                                                     "Pooled MCC [95% CI]": rc.typo(f"{r['pooled_MCC']} {r['pooled_MCC_ci']}"),
                                                     "Cross-session": r["cross_session"]} for r in pooled_rows])

    # facts on the headline quantities of the tables
    for r in t10:
        F.add(f"t10.{r['Judge (raw series)']}.{r['Rubric']}", ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("Judge (raw series)", "Rubric")),
              f"Table 10 row: {r['Judge (raw series)']} raw series, rubric {r['Rubric']}: false alarms of 20 per benign family "
              "(SP, SL, IS, RM, BB), of 60 (Exch.) and 40 (Rec.); detection of 20 per regression family (PR, LT, SE), of 60 (Det.); "
              "published detection of 60 (Publ.)", "analysis/tables/table10.csv", "from s6_heldout_counts.csv and s7_anchors_h4.csv")
    results["facts"] = len(F.items)
    rc.write_json(out / "facts_analysis.json", F.items)
    rc.write_json(out / "results.json", results)
    print(json.dumps({"s0": s0, "facts": len(F.items)}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
