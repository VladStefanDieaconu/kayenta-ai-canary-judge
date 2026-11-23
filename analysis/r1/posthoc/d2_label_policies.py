#!/usr/bin/env python3
"""D2: label-policy sensitivity of the published results (Reviewer 2, comment 2).

Recomputes detection, false-alarm rate, balanced accuracy (BA) and MCC for every
configuration of Tables 6 and 8 of the submitted manuscript under eight labelling
policies. The verdicts never change; only the labels and the window set do.

  P0  the published labels
  P1  healed_transient excluded
  P2  healed_transient relabelled FAIL
  P3  cross_metric_marginal excluded
  P4  cross_metric_marginal relabelled PASS
  P5  noise_equivalent excluded
  P6  mean-level objective: FAIL iff the realised mean ratio (canary mean over
      control mean) of any metric exceeds the configured allowedIncrease of 1.05
  P7  end-of-window objective: FAIL iff, for any metric, the mean of the last five
      canary samples exceeds 1.05 times the control mean

P1 and P2 must reproduce Table S13 of the Supplementary Materials (every cell is
compared). P0 must reproduce the point values of Tables 6 and 8. For P6 and P7 the
number of windows of each family that change label is reported, from the realised
series deposited with the data (the own-data export of the 180 windows).

Intervals are computed as in the submitted version: Wilson 95% for rates; for BA
and MCC the family-stratified percentile bootstrap with 2,000 replicates of the
archived area-A code, with the published resampling streams where they exist
(Table 6 and Table S13 rows) and new, named streams otherwise; Table S13's hosted
rows use the archived area-C procedure and its ensemble row the area-D procedure,
as in the submitted Table S13. The family-mixture ranges of BF+KS minus Claude
Opus 4.8 use the procedure of Archive S1 (posthoc_reanalysis.py).

For each policy the script also gives the order of the configurations and
evaluates the statements of Sections 4.2, 4.6 and 5.1 (read from the published
manuscript text) as survives, partly or does not survive, with the numbers.

Run (from the repository root):
  python -I -B analysis/r1/posthoc/d2_label_policies.py <data-archive> <archive-s1-dir> \
      <published_tables.json> <new-output-dir>
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import posthoc_common as pc  # noqa: E402

SCRIPT = "analysis/r1/posthoc/d2_label_policies.py"
POLICIES = [
    ("P0", "published labels"),
    ("P1", "healed_transient excluded"),
    ("P2", "healed_transient relabelled FAIL"),
    ("P3", "cross_metric_marginal excluded"),
    ("P4", "cross_metric_marginal relabelled PASS"),
    ("P5", "noise_equivalent excluded"),
    ("P6", "mean-level objective: FAIL iff the realised mean ratio of any metric exceeds 1.05"),
    ("P7", "end-of-window objective: FAIL iff the mean of the last five canary samples of any metric exceeds 1.05 "
           "times the control mean"),
]
PIDS = [p for p, _ in POLICIES]
D2_SET = ["statistical", "ai:raw:claude-opus-4-8-vlm", "ai:raw:gpt-oss-120b-llm", "ai:raw:qwen3-vl-235b-vlm",
          "ai:summary:phi4-llm", "ai:summary:mistral-nemo-llm", "ai:summary:qwen-llm", "ai:summary:deepseek-r1-llm",
          "ai:summary:olmo2-llm", "ai:plot:granite-vision-vlm", "ai:plot:minicpm-v-vlm", "ai:plot:moondream-vlm",
          "ensemble:tuned", "rule:kind", "rule:kind+stat", "rule:suffix", "bfks", "statistical:gate1.01", "rule:guard"]
TABLE12 = ["statistical", "bfks", "ensemble:tuned", "rule:guard", "ai:raw:claude-opus-4-8-vlm",
           "ai:raw:gpt-oss-120b-llm", "ai:raw:qwen3-vl-235b-vlm", "ai:summary:phi4-llm"]
HOSTED_RAW = ["ai:raw:claude-opus-4-8-vlm", "ai:raw:gpt-oss-120b-llm", "ai:raw:qwen3-vl-235b-vlm"]
LOCAL_PRIMARY = ["ai:summary:phi4-llm", "ai:summary:mistral-nemo-llm", "ai:summary:qwen-llm",
                 "ai:summary:deepseek-r1-llm", "ai:summary:olmo2-llm", "ai:plot:granite-vision-vlm",
                 "ai:plot:minicpm-v-vlm", "ai:plot:moondream-vlm"]
# Rows of Table S13, in its order, as configuration ids.
S13_ROWS = (["ai:summary:deepseek-r1-llm", "ai:summary:mistral-nemo-llm", "ai:summary:olmo2-llm", "ai:summary:phi4-llm",
             "ai:summary:qwen-llm", "ai:raw:deepseek-r1-llm", "ai:raw:mistral-nemo-llm", "ai:raw:olmo2-llm",
             "ai:raw:phi4-llm", "ai:raw:qwen-llm", "ai:plot:granite-vision-vlm", "ai:plot:minicpm-v-vlm",
             "ai:plot:moondream-vlm", "ai:summary:claude-opus-4-8-vlm", "ai:raw:claude-opus-4-8-vlm",
             "ai:plot:claude-opus-4-8-vlm", "ai:summary:gpt-oss-120b-llm", "ai:raw:gpt-oss-120b-llm",
             "ai:summary:qwen3-vl-235b-vlm", "ai:raw:qwen3-vl-235b-vlm", "ai:plot:qwen3-vl-235b-vlm"]
            + [f"hybrid:gated:{m}" for m in ["deepseek-r1-llm", "granite-vision-vlm", "minicpm-v-vlm", "mistral-nemo-llm",
                                             "moondream-vlm", "olmo2-llm", "phi4-llm", "qwen-llm",
                                             "claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm"]]
            + ["statistical", "ensemble:tuned"])
HOSTED_MODELS = pc.HOSTED


def slug_a(cid: str) -> str:
    """The configuration slug of the archived area-A script ('statistical' or '<judge>:<model>')."""
    return cid if cid == "statistical" else cid


# ----------------------------------------------------------------------------- labels
def policy_labels(pid, labels, fam, feats):
    if pid == "P0":
        return dict(labels)
    if pid == "P1":
        return {s: t for s, t in labels.items() if fam[s] != "healed_transient"}
    if pid == "P2":
        return {s: ("FAIL" if fam[s] == "healed_transient" else t) for s, t in labels.items()}
    if pid == "P3":
        return {s: t for s, t in labels.items() if fam[s] != "cross_metric_marginal"}
    if pid == "P4":
        return {s: ("PASS" if fam[s] == "cross_metric_marginal" else t) for s, t in labels.items()}
    if pid == "P5":
        return {s: t for s, t in labels.items() if fam[s] != "noise_equivalent"}
    if pid == "P6":
        return {s: ("FAIL" if max(x["mean_ratio"] for x in feats[s]) > 1.05 else "PASS") for s in labels}
    if pid == "P7":
        return {s: ("FAIL" if max(x["last5_ratio"] for x in feats[s]) > 1.05 else "PASS") for s in labels}
    raise KeyError(pid)


# ----------------------------------------------------------------------------- Archive S1 bootstrap draws
def s1_draws(fam_arr, families, mode_groups, B=2000, seed=20261014):
    """Index draws of Archive S1's posthoc_reanalysis.py: for each mode, B replicates; within fixed families
    each family is resampled to 20 with replacement; in the family-mixture mode the benign and regression families
    are first resampled with replacement (3 of the first group, 6 of the second). Draws do not depend on labels.
    `families` and `mode_groups` reproduce the published order when all nine families are present."""
    rng = np.random.default_rng(seed)
    index = {f: np.where(fam_arr == f)[0] for f in families}
    out = {}
    for mode in ("within_fixed_families", "resampled_family_mixture"):
        reps = []
        for _ in range(B):
            if mode == "within_fixed_families":
                chosen = families
            else:
                g1, g2 = mode_groups
                chosen = list(rng.choice(g1, len(g1))) + list(rng.choice(g2, len(g2)))
            reps.append(np.concatenate([rng.choice(index[f], len(index[f])) for f in chosen]))
        out[mode] = reps
    return out


def s1_ba_mcc(v, y, reps):
    ba, mcc, und = [], [], 0
    for ix in reps:
        yy, pp = y[ix], v[ix]
        tp = int((pp & yy).sum()); fp = int((pp & ~yy).sum()); tn = int((~pp & ~yy).sum()); fn = int((~pp & yy).sum())
        if tp + fn == 0 or tn + fp == 0:
            und += 1
            ba.append(np.nan); mcc.append(np.nan)
            continue
        ba.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
        mcc.append(pc.mcc_of(tp, fp, tn, fn))
    return np.array(ba), np.array(mcc), und


def same3(pub: str, val: float) -> str:
    """'yes' when the published three-decimal text equals the reproduced value printed by Python; 'yes (exact half)'
    when the reproduced value lies exactly on a half and the published text is its other rounding; else ''."""
    pub = pub.replace("−", "-").strip()
    if f"{val:.3f}" == pub or f"{val:.3f}" == pub.replace("-0.000", "0.000"):
        return "yes"
    x = val * 1000
    if abs(abs(x - math.floor(x)) - 0.5) < 1e-6 and abs(float(pub) - val) <= 0.0005 + 1e-9:
        return "yes (exact half)"
    return ""


def half_up3(x: float) -> str:
    """Three decimals of a float's shortest decimal representation, rounded half up (how a value such as
    0.7625, printed in Archive S1 as 0.7625, appears in the published tables)."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def match_note(itxt: str, pubv: str, ptxt: str, ivt, point_value: float = None) -> str:
    if itxt == pubv:
        return "yes"
    mm = re.match(r"\s*([\-\d.]+)\s*\[([\-\d.]+),\s*([\-\d.]+)\]", pubv)
    if mm and point_value is not None:
        r = [same3(mm.group(1), point_value), same3(mm.group(2), ivt[0]), same3(mm.group(3), ivt[1])]
        if all(r):
            return "yes (exact half)" if any(x == "yes (exact half)" for x in r) else "yes"
        if r[0]:
            return "point yes, interval differs"
    return "point yes, interval differs" if pubv.startswith(ptxt) else "NO"


# ----------------------------------------------------------------------------- main
def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 2
    arch, s1, pubf, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)  # existing files are never overwritten (write_* refuse)
    F = pc.Facts(SCRIPT)
    pub = json.loads(pubf.read_text(encoding="utf-8"))
    series = pc.load_series(arch)
    feats = pc.realised(series)
    labels = {s: d["label"] for s, d in series.items()}
    fam = {s: d["family"] for s, d in series.items()}
    gen_order = pc.generator_family_order(series)
    cfg = pc.load_configs(arch, s1, series)
    extra = pc.add_reimplemented(cfg, arch, series)
    A = pc.area_a(arch)
    B = pc.area_b(arch)
    C = pc.area_c(arch, gen_order)
    D = pc.area_d(arch)
    ci = pc.repo_modules()["ci_utils"]
    wilson = ci.wilson_interval

    lab = {pid: policy_labels(pid, labels, fam, feats) for pid in PIDS}

    # ---- realised features and label changes
    win_rows, chg_rows = [], []
    for s in sorted(series, key=lambda x: (pc.FAMILIES.index(fam[x]), x)):
        win_rows.append({"scenario": s, "family": fam[s], "P0_label": labels[s],
                         "max_mean_ratio": f"{max(x['mean_ratio'] for x in feats[s]):.6f}",
                         "max_last5_ratio": f"{max(x['last5_ratio'] for x in feats[s]):.6f}",
                         "P6_label": lab["P6"][s], "P7_label": lab["P7"][s],
                         "P6_changed": "yes" if lab["P6"][s] != labels[s] else "no",
                         "P7_changed": "yes" if lab["P7"][s] != labels[s] else "no"})
    for f in pc.FAMILIES:
        ids = [s for s in series if fam[s] == f]
        row = {"family": f, "n": len(ids), "P0_label": labels[ids[0]]}
        for pid in ("P6", "P7"):
            nf = sum(1 for s in ids if lab[pid][s] == "FAIL")
            ch = sum(1 for s in ids if lab[pid][s] != labels[s])
            row[f"{pid}_FAIL"] = nf
            row[f"{pid}_PASS"] = len(ids) - nf
            row[f"{pid}_changed"] = ch
            F.add(f"d2.change.{pid}.{f}", f"{ch}/{len(ids)}",
                  f"{f}: windows whose label changes under {pid} ({dict(POLICIES)[pid]}); published label "
                  f"{labels[ids[0]]}, {nf} of {len(ids)} FAIL under {pid}", f"data-archive/{pc.SERIES_REL}",
                  "per window and metric: mean ratio = mean(canary)/mean(control); last-five ratio = "
                  "mean(last 5 canary samples)/mean(control); FAIL iff the maximum over metrics exceeds 1.05")
        chg_rows.append(row)
    for pid in ("P6", "P7"):
        tot = sum(1 for s in series if lab[pid][s] != labels[s])
        nfail = sum(1 for s in series if lab[pid][s] == "FAIL")
        F.add(f"d2.change.{pid}.total", f"{tot}/180",
              f"windows whose label changes under {pid}; {nfail} FAIL and {180 - nfail} PASS under {pid}",
              f"data-archive/{pc.SERIES_REL}", "sum over the nine families")
    # Table 1 realised checks, as a check on the series and the feature code
    t1 = []
    def rng_of(f, key, kind=None):
        vals = [x[key] for s in series if fam[s] == f for x in feats[s] if kind is None or x["kind"] == kind]
        return min(vals), max(vals)
    lo, hi = rng_of("variance_increase", "mean_ratio"); t1.append(("variance_increase mean ratio", "0.9374–1.1045", f"{lo:.4f}–{hi:.4f}"))
    lo, hi = rng_of("variance_increase", "sd_ratio"); t1.append(("variance_increase sample SD ratio", "5.933–12.083", f"{lo:.3f}–{hi:.3f}"))
    lo, hi = rng_of("tail_regression", "mean_ratio"); t1.append(("tail_regression mean ratio", "1.165–1.377", f"{lo:.3f}–{hi:.3f}"))
    lo, hi = rng_of("gradual_drift", "mean_ratio", "memory"); t1.append(("gradual_drift memory mean ratio", "1.0270–1.0500", f"{lo:.4f}–{hi:.4f}"))
    n_cmm = sum(1 for s in series if fam[s] == "cross_metric_marginal" and all(x["mean_ratio"] <= 1.05 for x in feats[s]))
    t1.append(("cross_metric_marginal: every mean ratio <= 1.05", "20/20", f"{n_cmm}/20"))
    n_ht = sum(1 for s in series if fam[s] == "healed_transient" and max(x["mean_ratio"] for x in feats[s]) >= 1.25)
    t1.append(("healed_transient mean ratio >= criticalIncrease 1.25", "20/20", f"{n_ht}/20"))
    n29 = sum(1 for s in series if labels[s] == "PASS" for x in feats[s] if x["mean_ratio"] >= 1.01)
    t1.append(("benign metric sides reaching mean ratio 1.01 (Section 5.1)", "29", str(n29)))
    repro = [{"check": c, "published": p, "reproduced": r, "match": "yes" if p == r else "NO"} for c, p, r in t1]

    # ---- per-family counts and metrics per policy and configuration
    def fam_counts(cid, pid, f):
        v, L = cfg[cid].verdicts, lab[pid]
        ids = [s for s in v if fam[s] == f and s in L]
        nF = sum(1 for s in ids if L[s] == "FAIL"); dF = sum(1 for s in ids if L[s] == "FAIL" and v[s] == "FAIL")
        nP = sum(1 for s in ids if L[s] == "PASS"); aP = sum(1 for s in ids if L[s] == "PASS" and v[s] == "FAIL")
        return nF, dF, nP, aP

    def conf(cid, pid, keep=None):
        v, L = cfg[cid].verdicts, lab[pid]
        LL = {s: t for s, t in L.items() if s in v and (keep is None or keep(s))}
        return pc.confusion(v, LL), LL

    def point(cid, pid, keep=None):
        (tp, fp, tn, fn), LL = conf(cid, pid, keep)
        m = A["metrics"](tp, fp, tn, fn)
        ex = pc.exact_rates(tp, fp, tn, fn)
        m["ba_exact"] = pc.frac_str(ex["ba"]) if ex["ba"] is not None else ""
        return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "n": tp + fp + tn + fn, **m}

    # stream and method of each interval
    def a_stream(pid, cid):
        if pid == "P0":
            if cid == "statistical" or cid.startswith(("ai:summary:", "ai:raw:", "ai:plot:", "hybrid:gated:")):
                model = cid.split(":")[-1]
                if model in HOSTED_MODELS and cid.startswith("ai:"):
                    return f"hosted|{cid}"
                return f"local|{cid}"
            return {"ensemble:tuned": "tab|ensemble:tuned", "rule:kind": "tab|rule:metric-kind",
                    "rule:kind+stat": "tab|rule:name-reading"}.get(cid, f"r1-d2|P0|{cid}")
        if pid in ("P1", "P2") and (cid == "statistical" or cid.startswith(("ai:", "hybrid:gated:"))):
            return f"ht-{'excl' if pid == 'P1' else 'relab'}|{cid}"
        return f"r1-d2|{pid}|{cid}"

    def interval_a(cid, pid, keep=None, stream=None):
        _, LL = conf(cid, pid, keep)
        v = cfg[cid].verdicts
        codes = {s: A["code"](LL[s], v[s]) for s in sorted(LL)}
        st = stream or a_stream(pid, cid)
        bt = A["boot_config"](codes, fam, st)
        return {"ba": bt["ba"], "mcc": bt["mcc"], "f1": bt["f1"], "method": f"area-A family-stratified bootstrap, "
                f"B=2000, seed 20260912 XOR CRC32('config|{st}')"}

    def interval_c(cid, pid):
        scope = {"P0": "as labelled", "P1": "excluded", "P2": "relabelled"}[pid]
        v = cfg[cid].verdicts
        d = {s: {"family": fam[s], "truth_label": labels[s], "verdict": v[s]} for s in v}
        struct, strata = C["strata_v"](d, scope)
        reps = C["fsb_reps"](struct, strata)
        res = {}
        for k, fn in (("ba", C["v_ba"]), ("mcc", C["v_mcc"]), ("f1", C["v_f1"])):
            c_, und = C["pct_ci"]([fn(*r) for r in reps])
            res[k] = c_
        res["method"] = "area-C family-stratified bootstrap (Table S13 hosted rows), B=2000, seed 20260913, nearest rank"
        return res

    def interval_d(cid, pid):
        mode = {"P1": "excluded", "P2": "relabelled FAIL"}[pid]
        v = cfg[cid].verdicts
        rows = {}
        for s in sorted(v):
            if fam[s] == "healed_transient" and mode == "excluded":
                continue
            rows[s] = {"family": fam[s], "truth": "FAIL" if fam[s] == "healed_transient" else labels[s], "verdict": v[s]}
        by = D["strata"](sorted(rows), lambda k: k.rsplit("_", 1)[0])
        sec = "§14 healed_transient excluded or relabelled FAIL"
        L_ = f"180 set, healed_transient {mode}: ensemble:tuned"
        res = {}
        for k, mname, fn in (("ba", "balanced accuracy", D["ba_c"]), ("mcc", "MCC", D["mcc_c"])):
            lo, hi, u = D["strat_boot_u"](by, lambda smp, fn=fn: fn(D["rconf"](rows, smp)), f"{sec}:{L_}:{mname}")
            res[k] = (lo, hi)
        res["method"] = "area-D family-stratified bootstrap (Table S13 ensemble row), B=2000, CRC32 seed of the label"
        return res

    # Archive S1 bootstrap draws (label-independent), used for BF+KS under P0 as in Table 6
    ids_s1 = sorted(series, key=lambda x: (pc.FAMILIES.index(fam[x]), x))
    fam_arr = np.array([fam[s] for s in ids_s1])
    draws = s1_draws(fam_arr, pc.FAMILIES, (pc.FAMILIES[:3], pc.FAMILIES[3:]))

    def interval_s1(cid, pid):
        L = lab[pid]
        y = np.array([L[s] == "FAIL" for s in ids_s1])
        v = np.array([cfg[cid].verdicts.get(s) == "FAIL" for s in ids_s1])
        ba, mcc, und = s1_ba_mcc(v, y, draws["within_fixed_families"])
        return {"ba": tuple(float(x) for x in np.percentile(ba, [2.5, 97.5])),
                "mcc": tuple(float(x) for x in np.percentile(mcc, [2.5, 97.5])), "f1": None,
                "method": "Archive S1 within-family bootstrap (Table 6 BF+KS row), B=2000, numpy default_rng(20261014), "
                          "linear percentiles"}

    metric_rows, fam_rows = [], []
    P = {}
    for pid in PIDS:
        for cid in D2_SET:
            m = point(cid, pid)
            if pid == "P0" and cid == "bfks":
                iv = interval_s1(cid, pid)
            elif pid in ("P1", "P2") and cid in HOSTED_RAW:
                iv = interval_c(cid, pid)
            elif pid in ("P1", "P2") and cid == "ensemble:tuned":
                iv = interval_d(cid, pid)
            else:
                iv = interval_a(cid, pid)
            nF, nP = m["TP"] + m["FN"], m["TN"] + m["FP"]
            rw, fw = wilson(m["TP"], nF), wilson(m["FP"], nP)
            P[(pid, cid)] = {**m, "iv": iv}
            row = {"policy": pid, "policy_definition": dict(POLICIES)[pid], "config": cid, "label": cfg[cid].label,
                   "in_table12": "yes" if cid in TABLE12 else "no", "n": m["n"], "n_FAIL": nF, "n_PASS": nP,
                   "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"],
                   "detection": f"{m['TP']}/{nF}", "detection_rate": pc.f3(m["rec"]),
                   "detection_wilson": pc.ci3(*rw) if nF else "",
                   "false_alarms": f"{m['FP']}/{nP}", "FPR": pc.f3(m["fpr"]), "FPR_wilson": pc.ci3(*fw) if nP else "",
                   "BA": pc.f3(m["ba"]), "BA_exact": m["ba_exact"], "BA_CI": pc.ci3(*iv["ba"]) if iv["ba"] else "",
                   "MCC": pc.f3(m["mcc"]), "MCC_CI": pc.ci3(*iv["mcc"]) if iv["mcc"] else "",
                   "F1": pc.f3(m["f1"]), "accuracy": pc.f3(m["acc"]), "interval_method": iv["method"],
                   "note": ("scored on fewer windows: failed calls excluded (Text S3)" if m["n"] < len(lab[pid]) else "")}
            metric_rows.append(row)
            for f in pc.FAMILIES:
                nF_, dF, nP_, aP = fam_counts(cid, pid, f)
                fam_rows.append({"policy": pid, "config": cid, "label": cfg[cid].label, "family": f,
                                 "n_FAIL_labelled": nF_, "detected": dF, "n_PASS_labelled": nP_, "false_alarms": aP})
            if cid in TABLE12 or pid == "P0":
                F.add(f"d2.metrics.{pid}.{cid}",
                      f"detection {m['TP']}/{nF}; false alarms {m['FP']}/{nP}; BA {pc.f3(m['ba'])} "
                      f"{pc.ci3(*iv['ba']) if iv['ba'] else ''}; MCC {pc.f3(m['mcc'])} {pc.ci3(*iv['mcc']) if iv['mcc'] else ''}",
                      f"{cfg[cid].label} under {pid} ({dict(POLICIES)[pid]})", cfg[cid].source,
                      f"{cfg[cid].filt}; labels of {pid}; {iv['method']}")
    # orders
    order_rows = []
    for pid in PIDS:
        for key, nm in (("mcc", "MCC"), ("ba", "BA")):
            for subset, sname in ((D2_SET, "Tables 6 and 8"), (TABLE12, "Table 12 set")):
                ranked = sorted(subset, key=lambda c: (-(P[(pid, c)][key] if P[(pid, c)][key] is not None else -9),
                                                      cfg[c].label))
                txt = " > ".join(f"{cfg[c].label} {pc.f3(P[(pid, c)][key])}" for c in ranked)
                order_rows.append({"policy": pid, "metric": nm, "set": sname, "order": txt})
                if sname == "Table 12 set":
                    F.add(f"d2.order.{pid}.{key}", txt, f"order of the Table 12 configurations by {nm} under {pid}",
                          "posthoc/d2_policy_metrics.csv", f"descending {nm}; ties by label")

    # ---- reproduction of Tables 6 and 8 (P0) and Table S13 (P1, P2)
    def parse_ratio_cell(cell):
        m = re.match(r"\s*([\d.]+)\s*\((\d+)/(\d+)\)\s*/\s*\[([−\-\d.]+),\s*([−\-\d.]+)\]", cell)
        return m.groups() if m else None

    def parse_pt_ci(cell):
        m = re.match(r"\s*([−\-\d.]+)\s*/\s*\[([−\-\d.]+),\s*([−\-\d.]+)\]", cell)
        return tuple(x.replace("−", "-") for x in m.groups()) if m else None

    s13 = pub["tables"]["Table S13"]["rows"]
    s13_rows = []
    if len(s13) - 1 != len(S13_ROWS):
        raise SystemExit(f"Table S13 has {len(s13) - 1} rows, expected {len(S13_ROWS)}")
    for r_i, cid in enumerate(S13_ROWS, start=1):
        cells = s13[r_i]
        label = cells[0]
        model = cid.split(":")[-1]
        hosted = model in HOSTED_MODELS
        for pid, offs in (("P1", 1), ("P2", 5)):
            m = point(cid, pid)
            if cid == "ensemble:tuned":
                iv = interval_d(cid, pid)
            elif hosted:
                iv = interval_c(cid, pid)
            else:
                iv = interval_a(cid, pid)
            nF, nP = m["TP"] + m["FN"], m["TN"] + m["FP"]
            mine = {}
            if pid == "P1":
                fw = wilson(m["FP"], nP); rw = wilson(m["TP"], nF)
                mine["FPR"] = (f"{m['fpr']:.3f}", str(m["FP"]), str(nP), f"{fw[0]:.3f}", f"{fw[1]:.3f}")
                mine["recall"] = (f"{m['rec']:.3f}", str(m["TP"]), str(nF), f"{rw[0]:.3f}", f"{rw[1]:.3f}")
                cols = [("FPR", 1), ("recall", 2), ("BA", 3), ("MCC", 4)]
            else:
                rw = wilson(m["TP"], nF)
                mine["recall"] = (f"{m['rec']:.3f}", str(m["TP"]), str(nF), f"{rw[0]:.3f}", f"{rw[1]:.3f}")
                cols = [("recall", 5), ("BA", 6), ("MCC", 7)]
            mine["BA"] = (f"{m['ba']:.3f}", f"{iv['ba'][0]:.3f}", f"{iv['ba'][1]:.3f}")
            mine["MCC"] = (f"{m['mcc']:.3f}", f"{iv['mcc'][0]:.3f}", f"{iv['mcc'][1]:.3f}")
            rawv = {"BA": (m["ba"], iv["ba"][0], iv["ba"][1]), "MCC": (m["mcc"], iv["mcc"][0], iv["mcc"][1])}
            if pid == "P1":
                rawv["FPR"] = (m["fpr"], fw[0], fw[1])
            rawv["recall"] = (m["rec"], rw[0], rw[1])
            for name, col in cols:
                cell = cells[col]
                if name in ("FPR", "recall"):
                    g = parse_ratio_cell(cell)
                    got = mine[name]
                    vals = rawv[name]
                    pt_ok = bool(g) and g[1:3] == got[1:3] and bool(same3(g[0], vals[0]))
                    iv_ok = bool(g) and bool(same3(g[3], vals[1])) and bool(same3(g[4], vals[2]))
                    rep_txt = f"{got[0]} ({got[1]}/{got[2]}) [{got[3]}, {got[4]}]"
                else:
                    g = parse_pt_ci(cell)
                    got = mine[name]
                    vals = rawv[name]
                    pt_ok = bool(g) and bool(same3(g[0], vals[0]))
                    iv_ok = bool(g) and bool(same3(g[1], vals[1])) and bool(same3(g[2], vals[2]))
                    rep_txt = f"{got[0]} [{got[1]}, {got[2]}]"
                s13_rows.append({"row": label, "config": cid, "policy": pid, "column": name,
                                 "published": cell.replace(" / ", " "), "reproduced": rep_txt,
                                 "point_and_count_match": "yes" if pt_ok else "NO",
                                 "interval_match": "yes" if iv_ok else "NO",
                                 "interval_method": iv["method"] if name in ("BA", "MCC") else "Wilson 95% (tools/ci_utils.py)"})
    pt_ok = sum(1 for r in s13_rows if r["point_and_count_match"] == "yes")
    iv_ok = sum(1 for r in s13_rows if r["interval_match"] == "yes")
    F.add("d2.repro.s13.points", f"{pt_ok}/{len(s13_rows)}",
          "cells of Table S13 (34 rows; FPR, recall, BA and MCC excluded; recall, BA and MCC relabelled) whose point "
          "value and count are reproduced exactly", "posthoc/d2_table_s13_reproduction.csv",
          "P1 and P2 recomputed from per-window verdicts; compared with the published cell text")
    F.add("d2.repro.s13.intervals", f"{iv_ok}/{len(s13_rows)}",
          "cells of Table S13 whose 95% interval is reproduced exactly with the archived interval code and streams",
          "posthoc/d2_table_s13_reproduction.csv", "Wilson (rates); area A, C and D bootstrap code (BA, MCC)")

    # Table 6 (P0), points and intervals
    t6 = pub["tables"]["Table 6"]["rows"]
    t6_map = {"Statistical judge": "statistical", "Claude Opus 4.8 (raw)": "ai:raw:claude-opus-4-8-vlm",
              "gpt-oss-120b (raw)": "ai:raw:gpt-oss-120b-llm", "Qwen3-VL 235B (raw)": "ai:raw:qwen3-vl-235b-vlm",
              "Phi-4 (summary)": "ai:summary:phi4-llm", "Mistral-Nemo (summary)": "ai:summary:mistral-nemo-llm",
              "Qwen2.5 (summary)": "ai:summary:qwen-llm", "DeepSeek-R1* (summary)": "ai:summary:deepseek-r1-llm",
              "OLMo 2 (summary)": "ai:summary:olmo2-llm", "Granite Vision (plot)": "ai:plot:granite-vision-vlm",
              "MiniCPM-V (plot)": "ai:plot:minicpm-v-vlm", "Moondream† (plot)": "ai:plot:moondream-vlm",
              "Tuned ensemble": "ensemble:tuned", "Metric-kind rule": "rule:kind",
              "Metric-kind + statistical rule": "rule:kind+stat", "Suffix rule": "rule:suffix",
              "Post-hoc BF+KS (fixed alpha)": "bfks"}
    t6_rows = []
    fam_cols = t6[0][1:10]
    for cells in t6[1:]:
        cid = t6_map[cells[0]]
        v = cfg[cid].verdicts
        for j, ab in enumerate(fam_cols):
            f = [k for k, a in pc.ABBR.items() if a == ab][0]
            ids = [s for s in v if fam[s] == f]
            k = sum(1 for s in ids if v[s] == labels[s])
            got = str(k) if len(ids) == 20 else (f"{k}/{len(ids)}" if ids else "no data")
            t6_rows.append({"table": "Table 6", "row": cells[0], "column": ab, "published": cells[1 + j],
                            "reproduced": got, "match": "yes" if cells[1 + j] == got else "NO"})
        m = P[("P0", cid)]
        for name, col, key in (("BA [95% CI]", 10, "ba"), ("MCC [95% CI]", 11, "mcc")):
            pubv = cells[col].replace("−", "-").replace("\xa0", " ")
            ptxt = f"{m[key]:.3f}"
            ivt = m["iv"][key]
            itxt = f"{ptxt} [{ivt[0]:.3f}, {ivt[1]:.3f}]".replace("-0.000", "0.000")
            degenerate = "N/A" in pubv
            if degenerate:
                ok_pt = pubv.startswith(ptxt)
                t6_rows.append({"table": "Table 6", "row": cells[0], "column": name, "published": cells[col],
                                "reproduced": f"{ptxt} [N/A]", "match": "yes" if ok_pt else "NO"})
            else:
                s1_iv = None
                t6_rows.append({"table": "Table 6", "row": cells[0], "column": name, "published": cells[col],
                                "reproduced": itxt, "match": match_note(itxt, pubv, ptxt, ivt, m[key]),
                                "interval_method": m["iv"]["method"]})
    # Table 8 (P0)
    t8 = pub["tables"]["Table 8"]["rows"]
    t8_map = {"Statistical judge": "statistical",
              "Statistical judge, allowedIncrease 1.01 (re-implementation)": "statistical:gate1.01",
              "Tuned ensemble": "ensemble:tuned", "Guard rule": "rule:guard",
              "Phi-4, summary (highest-F1 local model)": "ai:summary:phi4-llm", "Claude Opus 4.8, raw": "ai:raw:claude-opus-4-8-vlm",
              "Suffix rule": "rule:suffix", "Post-hoc BF+KS (fixed alpha)": "bfks"}
    for cells in t8[1:]:
        cid = t8_map[cells[0]]
        m = P[("P0", cid)]
        nc = point(cid, "P1")
        rw, fw = wilson(m["TP"], m["TP"] + m["FN"]), wilson(m["FP"], m["FP"] + m["TN"])
        got = {1: f"{m['rec']:.3f} [{rw[0]:.3f}, {rw[1]:.3f}]", 2: f"{m['fpr']:.3f} [{fw[0]:.3f}, {fw[1]:.3f}]",
               3: f"{nc['fpr']:.3f} ({nc['FP']}/{nc['FP'] + nc['TN']})", 6: f"{m['f1']:.3f}"}
        for col, name in ((1, "Recall [95% CI]"), (2, "FPR, 180 set [95% CI]"), (3, "NC/NE FPR (n = 40)"), (6, "F1")):
            pubv = cells[col].replace("\xa0", " ")
            t6_rows.append({"table": "Table 8", "row": cells[0], "column": name, "published": cells[col],
                            "reproduced": got[col], "match": "yes" if pubv == got[col] else "NO"})
        for col, name, key in ((4, "BA [95% CI]", "ba"), (5, "MCC [95% CI]", "mcc")):
            pubv = cells[col].replace("−", "-").replace("\xa0", " ")
            ptxt = f"{m[key]:.3f}"
            if "N/A" in pubv:
                t6_rows.append({"table": "Table 8", "row": cells[0], "column": name, "published": cells[col],
                                "reproduced": f"{ptxt} [N/A]", "match": "yes" if pubv.startswith(ptxt) else "NO"})
            else:
                ivt = m["iv"][key]
                itxt = f"{ptxt} [{ivt[0]:.3f}, {ivt[1]:.3f}]"
                t6_rows.append({"table": "Table 8", "row": cells[0], "column": name, "published": cells[col],
                                "reproduced": itxt, "match": match_note(itxt, pubv, ptxt, ivt, m[key]),
                                "interval_method": m["iv"]["method"]})

    # Archive S1 bootstrap: within-family and family-mixture ranges (Table 6 BF+KS row; Table S9 paired range)
    vec = lambda cid: np.array([cfg[cid].verdicts.get(s) == "FAIL" for s in ids_s1])  # noqa: E731
    y0 = np.array([labels[s] == "FAIL" for s in ids_s1])
    s1_checks = []
    for cid, key in (("bfks", "BF+KS"), ("ai:raw:claude-opus-4-8-vlm", "Opus"), ("statistical", "Kayenta"),
                     ("ai:summary:phi4-llm", "Phi-4"), ("ensemble:tuned", "Tuned ensemble")):
        ba, mcc, _ = s1_ba_mcc(vec(cid), y0, draws["within_fixed_families"])
        s1_checks.append((cid, "BA within", [float(x) for x in np.percentile(ba, [2.5, 97.5])]))
        s1_checks.append((cid, "MCC within", [float(x) for x in np.percentile(mcc, [2.5, 97.5])]))
    s1_pub = json.loads((s1 / "analysis.json").read_text(encoding="utf-8"))
    s1_name = {"bfks": "BF+KS", "ai:raw:claude-opus-4-8-vlm": "Opus", "statistical": "Kayenta", "ai:summary:phi4-llm": "Phi-4",
               "ensemble:tuned": "Tuned ensemble"}
    s1_ok = 0
    for cid, what, got in s1_checks:
        metric = what.split()[0]
        exp = s1_pub["summary"][s1_name[cid]]["within_fixed_families"][metric]
        ok = all(abs(a - b) < 1e-12 for a, b in zip(exp, got))
        s1_ok += int(ok)
        repro.append({"check": f"Archive S1 within-family bootstrap, {s1_name[cid]} {metric}", "published": str(exp),
                      "reproduced": str(got), "match": "yes" if ok else "NO"})
    bb, _, _ = s1_ba_mcc(vec("bfks"), y0, draws["resampled_family_mixture"])
    oo, _, _ = s1_ba_mcc(vec("ai:raw:claude-opus-4-8-vlm"), y0, draws["resampled_family_mixture"])
    got_mix = [float(x) for x in np.percentile(bb - oo, [2.5, 97.5])]
    exp_mix = s1_pub["paired_BA_ranges"]["BF+KS minus Opus"]["resampled_family_mixture"]
    ok_mix = all(abs(a - b) < 1e-12 for a, b in zip(exp_mix, got_mix))
    repro.append({"check": "Archive S1 family-mixture range, BF+KS minus Opus BA (Table S9)", "published": str(exp_mix),
                  "reproduced": str(got_mix), "match": "yes" if ok_mix else "NO"})

    def s1_range(pid, a_cid, b_cid):
        L = lab[pid]
        keep = [i for i, s in enumerate(ids_s1) if s in L]
        idsk = [ids_s1[i] for i in keep]
        famk = np.array([fam[s] for s in idsk])
        present = [f for f in pc.FAMILIES if f in set(famk)]
        groups = ([f for f in present if f in pc.BENIGN_FAMILIES], [f for f in present if f in pc.REGRESSION_FAMILIES])
        if pid == "P0":
            dr = draws
        else:
            dr = s1_draws(famk, present, groups)
        y = np.array([L[s] == "FAIL" for s in idsk])
        va = np.array([cfg[a_cid].verdicts.get(s) == "FAIL" for s in idsk])
        vb = np.array([cfg[b_cid].verdicts.get(s) == "FAIL" for s in idsk])
        res = {}
        for mode in ("within_fixed_families", "resampled_family_mixture"):
            ba_a, _, ua = s1_ba_mcc(va, y, dr[mode])
            ba_b, _, ub = s1_ba_mcc(vb, y, dr[mode])
            d = ba_a - ba_b
            ok = ~np.isnan(d)
            res[mode] = ([float(x) for x in np.percentile(d[ok], [2.5, 97.5])], int((~ok).sum()))
        return res

    # ---- threshold-free summaries (AUC-ROC and average precision), Section 4.2
    def auc_ap(cid, pid):
        L = lab[pid]
        sc = cfg[cid].scores
        pairs = [(L[s] == "FAIL", 100.0 - sc[s]) for s in sorted(sc) if s in L]
        r = B["roc_pr"](pairs)
        return r["auc"], r["ap"]
    auc_rows = []
    AUC = {}
    for pid in PIDS:
        for cid in ("ai:raw:claude-opus-4-8-vlm", "ai:summary:phi4-llm", "statistical"):
            a, p_ = auc_ap(cid, pid)
            AUC[(pid, cid)] = (a, p_)
            auc_rows.append({"policy": pid, "config": cid, "label": cfg[cid].label, "AUC_ROC": f"{a:.3f}",
                             "average_precision": f"{p_:.3f}"})
    for cid, exp_auc, exp_ap in (("ai:raw:claude-opus-4-8-vlm", "0.913", "0.947"), ("ai:summary:phi4-llm", "0.740", "0.785"),
                                 ("statistical", "0.658", "0.772")):
        a, p_ = AUC[("P0", cid)]
        repro.append({"check": f"AUC-ROC and average precision, {cfg[cid].label} (Section 4.2; Figure 3)",
                      "published": f"{exp_auc}, {exp_ap}", "reproduced": f"{a:.3f}, {p_:.3f}",
                      "match": "yes" if (f"{a:.3f}", f"{p_:.3f}") == (exp_auc, exp_ap) else "NO"})

    # ---- representation contrasts (Section 4.2, paragraph on representations), accuracy and BA
    rep_rows = []
    REP = {}
    for pid in PIDS:
        for model in pc.LOCAL_TEXT + pc.HOSTED:
            for rep in ("summary", "raw", "plot"):
                cid = f"ai:{rep}:{model}"
                if cid not in cfg:
                    continue
                m = point(cid, pid)
                REP[(pid, cid)] = m
                rep_rows.append({"policy": pid, "model": pc.PAPER_MODEL[model], "representation": rep, "n": m["n"],
                                 "accuracy": pc.f3(m["acc"]), "BA": pc.f3(m["ba"]), "MCC": pc.f3(m["mcc"])})

    # ---- statement evaluation
    stm_rows = []
    sec_text = pub["sections"]

    def quote(section, start, n_sent=1):
        for para in sec_text[section]:
            i = para.find(start)
            if i >= 0:
                rest = para[i:]
                end = 0
                for _ in range(n_sent):
                    # sentence end: '.' followed by space and a capital or '(', or the end of the paragraph
                    mm = re.search(r"\.(\s+(?=[A-Z(])|$)", rest[end:])
                    if not mm:
                        end = len(rest)
                        break
                    end += mm.start() + 1
                return rest[:end].strip()
        raise SystemExit(f"statement not found in Section {section}: {start!r}")

    def add_stmt(sid, section, start, fn, kind="label-dependent", n_sent=1):
        q = quote(section, start, n_sent)
        for pid in PIDS:
            status, numbers = fn(pid)
            stm_rows.append({"id": sid, "section": section, "statement": q, "policy": pid, "status": status,
                             "numbers": numbers, "kind": kind})

    def dets(cids, pid, fams):
        out = []
        for c in cids:
            parts = []
            for f in fams:
                nF, dF, nP, aP = fam_counts(c, pid, f)
                parts.append(f"{pc.ABBR[f]} {dF}/{nF}" + (f" (+{aP}/{nP} relabelled PASS failed)" if nP else ""))
            out.append(f"{cfg[c].label}: " + ", ".join(parts))
        return "; ".join(out)

    def full_det(c, pid, fams):
        return all(fam_counts(c, pid, f)[1] == fam_counts(c, pid, f)[0] and fam_counts(c, pid, f)[0] > 0 for f in fams)

    def ncne_fa(c, pid):
        fa = n = 0
        for f in ("no_change", "noise_equivalent"):
            nF, dF, nP, aP = fam_counts(c, pid, f)
            fa += aP; n += nP
        return fa, n

    def s01(pid):
        fams = ["variance_increase", "tail_regression"]
        hosted_full = all(full_det(c, pid, fams) for c in HOSTED_RAW)
        stat_none = all(fam_counts("statistical", pid, f)[1] == 0 for f in fams)
        n20 = all(fam_counts("statistical", pid, f)[0] == 20 for f in fams)
        nums = dets(HOSTED_RAW + ["statistical"], pid, fams)
        if hosted_full and stat_none:
            return ("survives" if n20 else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.2-1", "4.2", "All three hosted raw-series judges detect variance_increase", s01)

    def s02(pid):
        fams = ["variance_increase", "tail_regression"]
        full = [c for c in LOCAL_PRIMARY if full_det(c, pid, fams)]
        expected = [c for c in LOCAL_PRIMARY if c not in ("ai:summary:deepseek-r1-llm", "ai:plot:moondream-vlm")]
        nums = dets(LOCAL_PRIMARY, pid, fams)
        if set(full) == set(expected):
            same = all(fam_counts(c, pid, f)[0] == 20 for c in LOCAL_PRIMARY for f in fams)
            return ("survives" if same else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.2-2", "4.2", "At primary representations, local models also detect both fully", s02)

    def s03(pid):
        h = [ncne_fa(c, pid) for c in HOSTED_RAW]
        l_ = [ncne_fa(c, pid) for c in LOCAL_PRIMARY]
        hmin, hmax = min(a for a, _ in h), max(a for a, _ in h)
        lmin, lmax = min(a for a, _ in l_), max(a for a, _ in l_)
        nums = ("hosted NC/NE false alarms " + ", ".join(f"{cfg[c].label} {a}/{n}" for c, (a, n) in zip(HOSTED_RAW, h))
                + "; local primary " + ", ".join(f"{cfg[c].label} {a}/{n}" for c, (a, n) in zip(LOCAL_PRIMARY, l_)))
        same = (hmin, hmax, lmin, lmax) == (4, 14, 5, 40) and all(n == 40 for _, n in h)
        return ("survives" if same else "partly"), nums
    add_stmt("S4.2-3", "4.2", "No-change/noise-equivalent (NC/NE) false alarms are", s03)

    def s04(pid):
        ok = True
        parts = []
        for c in ("rule:kind", "rule:kind+stat"):
            fa, n = ncne_fa(c, pid)
            dv = sum(fam_counts(c, pid, f)[1] for f in ("variance_increase", "tail_regression"))
            nv = sum(fam_counts(c, pid, f)[0] for f in ("variance_increase", "tail_regression"))
            ok &= fa == 0 and dv == 0
            parts.append(f"{cfg[c].label}: NC/NE false alarms {fa}/{n}, VI+TR detected {dv}/{nv}")
        return ("survives" if ok else "does not survive"), "; ".join(parts)
    add_stmt("S4.2-4", "4.2", "The metric-kind rules pass these latency-only benign windows", s04)

    def s05(pid):
        st, nums = s04(pid)
        m = P[(pid, "ai:summary:olmo2-llm")]
        return ("survives" if st == "survives" else "does not survive"), f"{nums}; OLMo 2 BA {pc.f3(m['ba'])}, MCC {pc.f3(m['mcc'])}"
    add_stmt("S4.2-5", "4.2", "Thus kind alone cannot explain variance/tail detection", s05)

    def s06(pid):
        cs = HOSTED_RAW + [c for c in LOCAL_PRIMARY if c.startswith("ai:summary:")]
        ok = all(sum(1 for s, v in cfg[c].verdicts.items() if fam[s] == "healed_transient" and v == "FAIL") == 20 for c in cs)
        role = {"P1": "HT excluded: the verdicts stand but are not scored", "P2": "the rejections count as detections",
                "P6": "the rejections count as detections (all 20 HT windows FAIL under P6)",
                "P7": "the rejections are false alarms (all 20 HT windows PASS under P7)"}.get(pid, "the rejections are false alarms")
        return ("survives" if ok else "does not survive"), f"all 20 HT windows FAIL for all {len(cs)} configurations: {'yes' if ok else 'no'}; {role}"
    add_stmt("S4.2-6", "4.2", "Every hosted raw-series judge and local text judge at its primary representation rejects", s06,
             "verdict fact")

    def s07(pid):
        return "survives", f"healed_transient mean ratio >= 1.25 in {n_ht}/20 (realised series; label-independent)"
    add_stmt("S4.2-7", "4.2", "These canaries exceed the template", s07, "label-independent")

    def s08(pid):
        h = [ncne_fa(c, pid) for c in HOSTED_RAW]
        same = [a for a, _ in h] == [4, 14, 14] and all(n == 40 for _, n in h)
        return ("survives" if same else "partly"), "hosted FPR on NC/NE windows labelled PASS: " + ", ".join(
            f"{cfg[c].label} {a}/{n}" for c, (a, n) in zip(HOSTED_RAW, h))
    add_stmt("S4.2-8", "4.2", "Excluding HT leaves hosted FPRs", s08)

    def s09(pid):
        parts, ok_all, same = [], True, True
        pubacc = {"claude-opus-4-8-vlm": {"raw": "0.867", "plot": "0.817", "summary": "0.789"},
                  "qwen3-vl-235b-vlm": {"summary": "0.817", "raw": "0.811", "plot": "0.811"}}
        for model in ("claude-opus-4-8-vlm", "qwen3-vl-235b-vlm"):
            acc = {rep: REP[(pid, f"ai:{rep}:{model}")]["acc"] for rep in ("summary", "raw", "plot")}
            ok_all &= acc["plot"] <= max(acc["summary"], acc["raw"]) + 1e-12
            same &= all(bool(same3(pubacc[model][r], acc[r])) for r in acc)
            parts.append(f"{pc.PAPER_MODEL[model]} accuracy raw {acc['raw']:.3f}, plot {acc['plot']:.3f}, summary {acc['summary']:.3f}")
        if not ok_all:
            return "does not survive", "; ".join(parts)
        return ("survives" if same else "partly"), "; ".join(parts)
    add_stmt("S4.2-9", "4.2", "Every hosted number reported above is a raw-series configuration", s09, n_sent=2)

    def s10(pid):
        o, p4, k = AUC[(pid, "ai:raw:claude-opus-4-8-vlm")], AUC[(pid, "ai:summary:phi4-llm")], AUC[(pid, "statistical")]
        nums = (f"AUC-ROC {o[0]:.3f}, {p4[0]:.3f}, {k[0]:.3f}; average precision {o[1]:.3f}, {p4[1]:.3f}, {k[1]:.3f}; "
                f"differences over the statistical judge {o[0] - k[0]:+.3f} (Opus), {p4[0] - k[0]:+.3f} (Phi-4)")
        same = all(bool(same3(t, v)) for t, v in zip(("0.913", "0.740", "0.658", "0.947", "0.785", "0.772"),
                                                    (o[0], p4[0], k[0], o[1], p4[1], k[1])))
        if o[0] > p4[0] > k[0]:
            return ("survives" if same else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.2-10", "4.2", "Claude Opus 4.8 raw, Phi-4 summaries and Kayenta yield AUC-ROC", s10, n_sent=2)

    def s11(pid):
        models = pc.LOCAL_TEXT + pc.HOSTED
        ba = {(m, r): REP[(pid, f"ai:{r}:{m}")]["ba"] for m in models for r in ("summary", "raw")}
        lo_s = min(models, key=lambda m: (ba[(m, 'summary')], m)); hi_s = max(models, key=lambda m: (ba[(m, 'summary')], m))
        lo_r = min(models, key=lambda m: (ba[(m, 'raw')], m)); hi_r = max(models, key=lambda m: (ba[(m, 'raw')], m))
        d = {m: ba[(m, 'raw')] - ba[(m, 'summary')] for m in models}
        dmin = min(models, key=lambda m: d[m]); dmax = max(models, key=lambda m: d[m])
        nums = (f"summaries {ba[(lo_s, 'summary')]:.3f} ({pc.PAPER_MODEL[lo_s]}) to {ba[(hi_s, 'summary')]:.3f} "
                f"({pc.PAPER_MODEL[hi_s]}); raw series {ba[(lo_r, 'raw')]:.3f} ({pc.PAPER_MODEL[lo_r]}) to "
                f"{ba[(hi_r, 'raw')]:.3f} ({pc.PAPER_MODEL[hi_r]}); raw minus summary {d[dmin]:+.3f} ({pc.PAPER_MODEL[dmin]}) "
                f"to {d[dmax]:+.3f} ({pc.PAPER_MODEL[dmax]})")
        same_models = (hi_s, hi_r, dmin, dmax) == ("phi4-llm", "claude-opus-4-8-vlm", "phi4-llm", "claude-opus-4-8-vlm")
        same_nums = all(bool(same3(t, v)) for t, v in zip(("0.762", "0.800", "-0.221", "0.062"),
                                                         (ba[(hi_s, 'summary')], ba[(hi_r, 'raw')], d[dmin], d[dmax])))
        if same_models:
            return ("survives" if same_nums else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.2-11", "4.2", "Across the eight models run on each representation", s11)

    def s12(pid):
        m = P[(pid, "ensemble:tuned")]
        same = (m["FP"], m["FP"] + m["TN"]) == (9, 60)
        return ("survives" if same else "partly"), (f"main-benchmark false alarms {m['FP']}/{m['FP'] + m['TN']}; held-out "
                                                    "seeds 1-4, tuning and five-seed counts are on other windows and are not relabelled here")
    add_stmt("S4.6-1", "4.6", "The tuned ensemble has FPR 9/60", s12)

    def s13(pid):
        L = lab[pid]
        a, b = cfg["ensemble:tuned"].verdicts, cfg["ai:summary:phi4-llm"].verdicts
        common = sorted(s for s in L if s in a and s in b)
        ca = {s: A["code"](L[s], a[s]) for s in common}
        cb = {s: A["code"](L[s], b[s]) for s in common}
        f1 = lambda c: A["metrics"](*[sum(1 for x in c.values() if x == k) for k in range(4)])["f1"]  # noqa: E731
        diff = f1(ca) - f1(cb)
        # paired family-stratified bootstrap of the F1 difference (area-A resampling scheme, own stream)
        import random as _r
        st = A["strata"](common, fam)
        rng = _r.Random(A["seed_for"]("paired", f"r1-d2|{pid}|f1|ensemble:tuned-minus-ai:summary:phi4-llm"))
        ds = []
        for _ in range(2000):
            xa, xb = [0, 0, 0, 0], [0, 0, 0, 0]
            for g in st:
                Lg = len(g)
                for _ in range(Lg):
                    s = g[rng.randrange(Lg)]
                    xa[ca[s]] += 1; xb[cb[s]] += 1
            ds.append(A["metrics"](*xa)["f1"] - A["metrics"](*xb)["f1"])
        lo_, hi_ = A["pct"](ds, 0.025), A["pct"](ds, 0.975)
        nums = (f"ensemble minus Phi-4 F1 {diff:+.3f} [{lo_:+.3f}, {hi_:+.3f}] (paired family-stratified bootstrap, B=2000, "
                "own stream; the published interval comes from the area-D analysis)")
        if lo_ <= 0 <= hi_:
            return ("survives" if f"{diff:+.3f}" == "-0.003" else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.6-2", "4.6", "Its paired F1 difference from Phi-4", s13)

    def s14(pid):
        L = lab[pid]
        tr = [s for s in series if fam[s] == "tail_regression"]
        unchanged = all(L.get(s) == "FAIL" for s in tr)
        if unchanged:
            return "survives", "all 20 tail_regression windows remain FAIL-labelled, so the counts 8/20 and 5/20 stand"
        n = sum(1 for s in tr if L.get(s) == "FAIL")
        return "not evaluated", (f"{n}/20 tail_regression windows FAIL-labelled; per-window verdicts of the ensemble "
                                 "without its recovery guard are not in the deposit")
    add_stmt("S4.6-3", "4.6", "Its recovery guard reduces tail detection", s14)

    def s15(pid):
        m = P[(pid, "rule:guard")]
        nums = (f"detection {m['TP']}/{m['TP'] + m['FN']}, false alarms {m['FP']}/{m['FP'] + m['TN']}, BA {pc.f3(m['ba'])}, "
                f"MCC {pc.f3(m['mcc'])}")
        same = (m["TP"], m["TP"] + m["FN"], m["FP"], m["FP"] + m["TN"]) == (114, 120, 12, 60)
        return ("survives" if same else "partly"), nums
    add_stmt("S4.6-4", "4.6", "The model-free guard rule detects 114/120", s15)

    def s16(pid):
        L = lab[pid]
        k = 0
        for mdl in pc.LOCAL_TEXT + pc.LOCAL_VISION:
            g, h = cfg[f"hybrid:gated:{mdl}"].verdicts, cfg[f"hybrid:guarded:{mdl}"].verdicts
            k += sum(1 for s in g if fam[s] == "tail_regression" and L.get(s) == "FAIL" and g[s] == "FAIL" and h[s] == "PASS")
        nums = (f"true tail_regression detections of the eight local hybrids suppressed by the guards: {k}; the seeds 1-4 "
                "figures are on other windows and are not relabelled here")
        return ("survives" if k == 43 else ("does not survive" if k == 0 else "partly")), nums
    add_stmt("S4.6-5", "4.6", "Guards suppress 43 true local-hybrid detections", s16)

    def s17(pid):
        return "survives", "held-out sustained_excursion windows; not relabelled by these policies"
    add_stmt("S4.6-6", "4.6", "The equal-variance guard also fires on all 20 sustained_excursion", s17, "label-independent")

    def s18a(pid):
        m = P[(pid, "bfks")]
        fa_fams = {f: fam_counts("bfks", pid, f)[3] for f in pc.FAMILIES}
        fa_txt = ", ".join(f"{pc.ABBR[f]} {v}" for f, v in fa_fams.items() if v)
        nums = f"detection {m['TP']}/{m['TP'] + m['FN']}; false alarms {m['FP']}/{m['FP'] + m['TN']} ({fa_txt or 'none'})"
        same = (m["TP"], m["TP"] + m["FN"], m["FP"], m["FP"] + m["TN"]) == (109, 120, 20, 60)
        return ("survives" if same else "partly"), nums
    add_stmt("S4.6-7", "4.6", "The BF+KS battery detects 109/120 regressions", s18a)

    def s18b(pid):
        keep = lambda s: fam[s] != "healed_transient"  # noqa: E731
        if pid in ("P1",):
            keep = None
        b, o = point("bfks", pid, keep), point("ai:raw:claude-opus-4-8-vlm", pid, keep)
        fa, n = ncne_fa("bfks", pid)
        nums = f"BF+KS NC/NE false alarms {fa}/{n}; BA without healed_transient {pc.f3(b['ba'])} (BF+KS) and {pc.f3(o['ba'])} (Opus raw)"
        if b["ba"] is None or o["ba"] is None:
            return "not evaluated", nums
        if b["ba"] >= o["ba"] - 1e-12 and fa == 0:
            same = bool(same3("0.954", b["ba"])) and bool(same3("0.950", o["ba"])) and n == 40
            return ("survives" if same else "partly"), nums
        return "does not survive", nums
    add_stmt("S4.6-8", "4.6", "Its NC/NE FPR is 0/40 and BA without healed_transient", s18b)

    def s18c(pid):
        b, o = P[(pid, "bfks")], P[(pid, "ai:raw:claude-opus-4-8-vlm")]
        rr = s1_range(pid, "bfks", "ai:raw:claude-opus-4-8-vlm")
        mix, und = rr["resampled_family_mixture"]
        nums = (f"BA {pc.f3(b['ba'])} (BF+KS) versus {pc.f3(o['ba'])} (Opus raw); difference {b['ba'] - o['ba']:+.3f}; "
                f"family-mixture range [{mix[0]:+.3f}, {mix[1]:+.3f}]" + (f" ({und} undefined replicates dropped)" if und else ""))
        spans = mix[0] <= 0 <= mix[1]
        lower = b["ba"] < o["ba"]
        same = bool(same3("0.788", b["ba"])) and bool(same3("0.800", o["ba"])) and bool(same3("-0.013", b["ba"] - o["ba"]))
        if spans and lower:
            return ("survives" if same else "partly"), nums
        if spans:
            return "partly", nums + " (BF+KS now has the higher point estimate)"
        return "does not survive", nums
    add_stmt("S4.6-9", "4.6", "Its full-set BA is 0.788 versus 0.800 for Opus", s18c, n_sent=2)

    def s19(pid):
        st1, n1 = s01(pid)
        h = [ncne_fa(c, pid) for c in HOSTED_RAW]
        fa_txt = ", ".join(f"{a}/{n}" for a, n in h)
        same = [a for a, _ in h] == [4, 14, 14] and all(n == 40 for _, n in h)
        if st1 == "does not survive":
            return st1, n1 + f"; NC/NE false alarms {fa_txt}"
        return ("survives" if (st1 == "survives" and same) else "partly"), n1 + f"; NC/NE false alarms {fa_txt}"
    add_stmt("S5.1-1", "5.1", "Model judges add useful spread/tail discrimination", s19)

    def s20(pid):
        fams = ["variance_increase", "tail_regression"]
        cs = ["ai:summary:mistral-nemo-llm", "ai:summary:phi4-llm", "ai:plot:minicpm-v-vlm"]
        full = all(full_det(c, pid, fams) for c in cs)
        fa = [ncne_fa(c, pid) for c in cs]
        nums = "; ".join(f"{cfg[c].label}: " + ", ".join(f"{pc.ABBR[f]} {fam_counts(c, pid, f)[1]}/{fam_counts(c, pid, f)[0]}"
                                                         for f in fams) + f", NC/NE false alarms {a}/{n}"
                         for c, (a, n) in zip(cs, fa))
        if not full:
            return "does not survive", nums
        same = [a for a, _ in fa] == [5, 7, 10] and all(n == 40 for _, n in fa) and all(
            fam_counts(c, pid, f)[0] == 20 for c in cs for f in fams)
        return ("survives" if same else "partly"), nums
    add_stmt("S5.1-2", "5.1", "Mistral-Nemo, Phi-4 and MiniCPM-V also detect them", s20)

    def s21(pid):
        m = P[(pid, "ai:summary:olmo2-llm")]
        g = P[(pid, "ai:plot:granite-vision-vlm")]
        return "survives", f"OLMo 2 BA {pc.f3(m['ba'])}, MCC {pc.f3(m['mcc'])}; Granite Vision BA {pc.f3(g['ba'])}, MCC {pc.f3(g['mcc'])}"
    add_stmt("S5.1-3", "5.1", "Constant or near-constant FAIL behaviour", s21)

    def s22(pid):
        return "survives", "held-out sustained_excursion windows; not relabelled by these policies"
    add_stmt("S5.1-4", "5.1", "The held-out sustained_excursion misses show", s22, "label-independent")

    def s23(pid):
        st6, n6 = s06(pid)
        st8, n8 = s08(pid)
        if st6 != "survives":
            return "does not survive", n6
        return st8, f"{n6}; {n8}"
    add_stmt("S5.1-5", "5.1", "The three hosted raw-series judges reject all 20 healed_transient windows", s23)

    def s24(pid):
        return "survives", "describes P1 and P2 (Table S13)"
    add_stmt("S5.1-6", "5.1", "The healed_transient PASS label is contested", s24, "label-independent")

    def s25(pid):
        ok = 0
        tot = 0
        st = cfg["statistical"].verdicts
        for mdl in pc.LOCAL_TEXT + pc.LOCAL_VISION:
            h = cfg[f"hybrid:gated:{mdl}"].verdicts
            sc = cfg[f"ai:{pc.PRIMARY[mdl]}:{mdl}"].scores
            for s, v in h.items():
                tot += 1
                ok += int(v == ("FAIL" if st[s] == "FAIL" or sc[s] < 50 else "PASS"))
        return "survives", f"hybrid as scored equals statistical FAIL OR model score below 50 on {ok}/{tot} local hybrid rows (verdict fact)"
    add_stmt("S5.1-7", "5.1", "The scored hybrid is the union of statistical FAIL", s25, "verdict fact")

    def s26(pid):
        e, g, o = P[(pid, "ensemble:tuned")], P[(pid, "rule:guard")], P[(pid, "ai:raw:claude-opus-4-8-vlm")]
        nums = f"BA ensemble {pc.f3(e['ba'])}, guard rule {pc.f3(g['ba'])}, Opus raw {pc.f3(o['ba'])}"
        if e["ba"] > o["ba"] and g["ba"] > o["ba"]:
            same = bool(same3("0.850", e["ba"])) and bool(same3("0.875", g["ba"])) and bool(same3("0.800", o["ba"]))
            return ("survives" if same else "partly"), nums
        if e["ba"] > o["ba"] or g["ba"] > o["ba"]:
            return "partly", nums + " (only one of the two rules is above Opus)"
        return "does not survive", nums
    add_stmt("S5.1-8", "5.1", "Tuned ensemble and guard-rule BA are 0.850 and 0.875", s26)

    def s27(pid):
        b = P[(pid, "bfks")]
        fa, n = ncne_fa("bfks", pid)
        nums = f"BF+KS BA {pc.f3(b['ba'])}; NC/NE false alarms {fa}/{n}"
        if fa != 0:
            return "does not survive", nums
        return ("survives" if same3("0.788", b["ba"]) and n == 40 else "partly"), nums
    add_stmt("S5.1-9", "5.1", "The fixed-alpha BF+KS battery reaches BA 0.788", s27)

    def s28(pid):
        rr = s1_range(pid, "bfks", "ai:raw:claude-opus-4-8-vlm")
        mix, und = rr["resampled_family_mixture"]
        nums = f"BF+KS minus Opus raw BA, family-mixture range [{mix[0]:+.3f}, {mix[1]:+.3f}]" + (
            f" ({und} undefined replicates dropped)" if und else "")
        if mix[0] <= 0 <= mix[1]:
            same = (f"{mix[0]:.4f}", f"{mix[1]:.4f}") == ("-0.0875", "0.0583")
            return ("survives" if same or pid == "P0" else "partly"), nums
        return "does not survive", nums
    add_stmt("S5.1-10", "5.1", "Paired family-mixture ranges include zero for BF+KS minus Opus", s28)

    def s29(pid):
        L = lab[pid]
        a, b = cfg["statistical"].verdicts, cfg["statistical:gate1.01"].verdicts
        pr = [s for s in L if fam[s] in pc.PROBE_FAMILIES and L[s] == "FAIL"]
        fl = [s for s in L if L[s] == "FAIL"]
        ben = [s for s in L if L[s] == "PASS"]
        p105 = sum(a[s] == "FAIL" for s in pr); p101 = sum(b[s] == "FAIL" for s in pr)
        r105 = sum(a[s] == "FAIL" for s in fl); r101 = sum(b[s] == "FAIL" for s in fl)
        fa = sum(b[s] == "FAIL" for s in ben)
        nums = (f"probe detection {p105}/{len(pr)} -> {p101}/{len(pr)}; total recall {r105}/{len(fl)} -> {r101}/{len(fl)}; "
                f"false alarms at gate 1.01 {fa}/{len(ben)}")
        same = (p105, p101, len(pr), r105, r101, len(fl), fa, len(ben)) == (0, 21, 80, 38, 61, 120, 0, 60)
        if fa > 0 or r101 == r105:
            return "does not survive", nums
        return ("survives" if same else "partly"), nums
    add_stmt("S5.1-11", "5.1", "Lowering Kayenta", s29)

    def s30(pid):
        L = lab[pid]
        a, b = cfg["statistical"].verdicts, cfg["statistical:gate1.01"].verdicts
        parts = []
        gains = {}
        for f in ("cross_metric_marginal", "gradual_drift", "subtle_regression", "variance_increase", "tail_regression"):
            ids = [s for s in L if fam[s] == f and L[s] == "FAIL"]
            g_ = sum(1 for s in ids if b[s] == "FAIL" and a[s] == "PASS")
            tot = sum(1 for s in ids if b[s] == "FAIL")
            gains[f] = (g_, tot, len(ids))
            parts.append(f"{pc.ABBR[f]} +{g_} (detected {tot}/{len(ids)})")
        newfa = sum(1 for s in L if L[s] == "PASS" and b[s] == "FAIL" and a[s] == "PASS")
        nums = "; ".join(parts) + f"; new false alarms at gate 1.01: {newfa}"
        same = (gains["cross_metric_marginal"][1:] == (20, 20) and gains["gradual_drift"][0] == 1 and
                gains["subtle_regression"][0] == 2 and gains["variance_increase"][1] == 0 and gains["tail_regression"][1] == 0)
        qual = gains["variance_increase"][1] == 0 and gains["tail_regression"][1] == 0
        tot_gain = sum(g[0] for g in gains.values())
        if newfa > 0 or tot_gain == 0 or not qual:
            return "does not survive", nums
        return ("survives" if same and newfa == 0 else "partly"), nums
    add_stmt("S5.1-12", "5.1", "It recovers all CMM windows", s30)

    def s31(pid):
        L = lab[pid]
        b = cfg["statistical:gate1.01"].verdicts
        sides = [(s, x) for s in L if L[s] == "PASS" for x in feats[s] if x["mean_ratio"] >= 1.01]
        flagged = sum(1 for s in {s for s, _ in sides} if b[s] == "FAIL")
        nums = f"metric sides of PASS-labelled windows with mean ratio >= 1.01: {len(sides)}; windows among them failed at gate 1.01: {flagged}"
        if flagged == 0:
            return ("survives" if len(sides) == 29 else "partly"), nums
        return "does not survive", nums
    add_stmt("S5.1-13", "5.1", "None of the 29 benign metric sides", s31)

    def s32(pid):
        fams = ["variance_increase", "tail_regression"]
        nF = sum(fam_counts("bfks", pid, f)[0] for f in fams)
        dB = sum(fam_counts("bfks", pid, f)[1] for f in fams)
        dS = sum(fam_counts("statistical", pid, f)[1] for f in fams)
        h = [sum(fam_counts(c, pid, f)[1] for f in fams) for c in HOSTED_RAW]
        nums = (f"FAIL-labelled VI+TR windows: {nF}; detected by BF+KS {dB}, statistical judge {dS}, hosted raw "
                + ", ".join(map(str, h)))
        ok = nF > 0 and dB >= nF / 2 and min(h) > dS
        return ("survives" if ok else "does not survive"), nums
    add_stmt("S5.1-14", "5.1", "Model judges’ gains over Kayenta concern spread/tail shapes", s32)

    def s33(pid):
        base = [r for r in order_rows if r["policy"] == "P0" and r["metric"] == "MCC" and r["set"] == "Table 12 set"][0]["order"]
        cur = [r for r in order_rows if r["policy"] == pid and r["metric"] == "MCC" and r["set"] == "Table 12 set"][0]["order"]
        strip = lambda t: [re.sub(r" [−\-\d.]+$", "", x) for x in t.split(" > ")]  # noqa: E731
        changed = strip(base) != strip(cur)
        return "survives", ("MCC order of the Table 12 configurations differs from P0" if changed else
                            "MCC order of the Table 12 configurations as under P0")
    add_stmt("S5.1-15", "5.1", "The evidence supports configuration-specific mechanisms", s33)

    # statement summary facts
    for pid in PIDS:
        rows = [r for r in stm_rows if r["policy"] == pid]
        cnt = defaultdict(int)
        for r in rows:
            cnt[r["status"]] += 1
        F.add(f"d2.statements.{pid}", ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())),
              f"statements of Sections 4.2, 4.6 and 5.1 evaluated under {pid}: number per status",
              "posthoc/d2_statements.csv", "rules coded per statement in d2_label_policies.py")
    for r in stm_rows:
        if r["status"] != "survives" or r["policy"] == "P0":
            F.add(f"d2.stmt.{r['id']}.{r['policy']}", f"{r['status']}: {r['numbers']}",
                  f"Section {r['section']} statement under {r['policy']}: \"{r['statement']}\"",
                  "posthoc/d2_statements.csv", "evaluation rule of the statement in d2_label_policies.py")
    # BF+KS minus Opus ranges per policy (facts)
    for pid in PIDS:
        rr = s1_range(pid, "bfks", "ai:raw:claude-opus-4-8-vlm")
        F.add(f"d2.bfks_minus_opus.{pid}",
              f"within fixed families [{rr['within_fixed_families'][0][0]:+.3f}, {rr['within_fixed_families'][0][1]:+.3f}]; "
              f"family mixture [{rr['resampled_family_mixture'][0][0]:+.3f}, {rr['resampled_family_mixture'][0][1]:+.3f}]",
              f"BF+KS minus Claude Opus 4.8 raw, balanced accuracy, 95% percentile ranges under {pid}",
              f"archive-s1/{pc.BFKS_FILE}; data-archive/results/agg/frontier_long_claude-opus-4-8-vlm.csv",
              "procedure of Archive S1 posthoc_reanalysis.py (numpy default_rng(20261014), B=2000); for P1, P3 and P5 the "
              "remaining families of each design group are resampled; replicates with an empty class are dropped")

    # ---- write
    pc.write_csv(out / "d2_window_labels.csv", win_rows)
    pc.write_csv(out / "d2_label_changes.csv", chg_rows)
    pc.write_csv(out / "d2_policy_metrics.csv", metric_rows)
    pc.write_csv(out / "d2_policy_family_counts.csv", fam_rows)
    pc.write_csv(out / "d2_orders.csv", order_rows)
    pc.write_csv(out / "d2_statements.csv", stm_rows)
    pc.write_csv(out / "d2_table_s13_reproduction.csv", s13_rows)
    pc.write_csv(out / "d2_tables_6_8_reproduction.csv", t6_rows,
                 ["table", "row", "column", "published", "reproduced", "match", "interval_method"])
    pc.write_csv(out / "d2_auc.csv", auc_rows)
    pc.write_csv(out / "d2_representations.csv", rep_rows)
    pc.write_csv(out / "d2_reproduction_checks.csv", repro)
    t68_ok = sum(1 for r in t6_rows if r["match"].startswith("yes"))
    t68_pt = sum(1 for r in t6_rows if r["match"].startswith("yes") or r["match"] == "point yes, interval differs")
    F.add("d2.repro.t6_t8", f"{t68_ok}/{len(t6_rows)} cells exact; {t68_pt}/{len(t6_rows)} with the point value exact",
          "cells of Tables 6 and 8 (P0) reproduced from per-window verdicts", "posthoc/d2_tables_6_8_reproduction.csv",
          "family counts, rates, Wilson intervals, BA and MCC with the published bootstrap streams where they are known")
    F.add("d2.repro.checks", f"{sum(1 for r in repro if r['match'] == 'yes')}/{len(repro)}",
          "other reproduction checks (Table 1 realised checks, the 29 benign metric sides of Section 5.1, Archive S1 "
          "bootstrap intervals, Table S9 range, AUC-ROC and average precision of Figure 3)",
          "posthoc/d2_reproduction_checks.csv", "see the file")
    pc.write_json(out / "d2_facts.json", F.items)
    pc.write_json(out / "d2_run.json", {"script": SCRIPT, "code_sha256": pc.code_hashes(arch),
                                        "published_tables": pub["inputs"], "python": sys.version.split()[0],
                                        "numpy": np.__version__, "generator_family_order": gen_order,
                                        "reimplementation_agreement_at_1.05": extra["reimpl_agree_1.05"]})
    bad = [r for r in repro if r["match"] != "yes"]
    s13_bad = [r for r in s13_rows if r["point_and_count_match"] != "yes" or r["interval_match"] != "yes"]
    t_bad = [r for r in t6_rows if not r["match"].startswith("yes")]
    print(f"D2: Table S13 points {pt_ok}/{len(s13_rows)}, intervals {iv_ok}/{len(s13_rows)}; Tables 6/8 exact "
          f"{t68_ok}/{len(t6_rows)}, points {t68_pt}/{len(t6_rows)}; other checks {len(repro) - len(bad)}/{len(repro)}")
    for r in bad + s13_bad[:40] + t_bad[:40]:
        print("  ", {k: r[k] for k in r if k in ("check", "row", "policy", "column", "published", "reproduced", "match",
                                                   "point_and_count_match", "interval_match")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
