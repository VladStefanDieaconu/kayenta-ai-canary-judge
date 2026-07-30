#!/usr/bin/env python3
"""Score the prompt ablation: every metric, per prompt, per model, per dataset.

Reads only through results_schema.load(), so it has no knowledge of which run
wrote which file. Everything it prints is derived from the long-format frame and
sourced to a column.

Every rate is reported with a Wilson interval rather than a Wald one. On this
design a per-family cell is 20 scenarios and rates of exactly 0.000 and 1.000 are
common; a Wald interval on either leaves the unit range, and a bare 0.000 with no
interval invites a reader to treat 0/20 as stronger evidence than it is.

Usage:
  python tools/analyse_ablation.py                       # everything in the frame
  python tools/analyse_ablation.py --dataset original-180
  python tools/analyse_ablation.py --out audit/ablation-analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import results_schema as rs  # noqa: E402

FROZEN = "v1-frozen-2026-06"
PROMPT_ORDER = [FROZEN, "v2-operational-2026-07", "v3-temporal-2026-07"]


def wilson(k: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval. Stays inside [0, 1] at k=0 and k=n, which is the
    entire reason it is used here rather than the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    c = rs.confusion(rows)
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec) else float("nan")
    bal = (rec + spec) / 2 if not (math.isnan(rec) or math.isnan(spec)) else float("nan")
    # MCC is undefined when a whole row or column of the confusion matrix is zero,
    # which is the case on any single-label dataset: generalisation-60 is entirely
    # FAIL, so TN + FP = 0. Returning 0.0 there would print a number that reads as
    # "no better than chance" when the correct statement is that the coefficient
    # cannot be computed. Accuracy equals recall on such a set, and neither
    # balanced accuracy nor FPR means anything.
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else float("nan")
    lo, hi = wilson(tp + tn, n)
    return {"n": n, "accuracy": acc, "accuracy_ci95": [lo, hi], "precision": prec,
            "recall": rec, "specificity": spec, "f1": f1, "balanced_accuracy": bal,
            "mcc": mcc, "fpr": (1 - spec) if not math.isnan(spec) else float("nan"),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def family_accuracy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r)
    out = {}
    for fam in sorted(by, key=ed.family_order):
        rs_ = by[fam]
        k = sum(1 for r in rs_ if r["correct"] == 1)
        n = len(rs_)
        lo, hi = wilson(k, n)
        out[fam] = {"k": k, "n": n, "accuracy": k / n if n else float("nan"),
                    "ci95": [lo, hi]}
    return out


def paired_diff(a_rows, b_rows) -> Dict[str, Any]:
    """Per-scenario paired comparison, which is what a prompt ablation actually is.

    Two point estimates from two configurations are not a paired difference, and
    the distinction has to be stated rather than implied: here both arms judge the
    identical scenario set, so the comparison is paired and McNemar applies.
    """
    a = {r["scenario_id"]: r for r in a_rows}
    b = {r["scenario_id"]: r for r in b_rows}
    shared = sorted(set(a) & set(b))
    changed, b01, b10 = [], 0, 0
    for s in shared:
        if a[s]["verdict"] != b[s]["verdict"]:
            changed.append({"scenario": s, "family": a[s]["family"],
                            "from": a[s]["verdict"], "to": b[s]["verdict"],
                            "truth": a[s]["truth_label"]})
        ca, cb = a[s]["correct"] == 1, b[s]["correct"] == 1
        if not ca and cb:
            b01 += 1
        elif ca and not cb:
            b10 += 1
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, 2.0 * sum(math.comb(n_disc, i) * 0.5 ** n_disc for i in range(k + 1)))
    return {"n_paired": len(shared), "verdict_changes": len(changed),
            "b_fixed": b01, "b_broke": b10, "net": b01 - b10,
            "mcnemar_exact_p": round(p, 5), "changes": changed}


def statistical_invariant() -> Dict[str, Any]:
    """The statistical judge must be identical in every run of a given scenario.

    It is deterministic and prompt-independent, so a disagreement between two runs
    cannot be caused by the rubric under test -- it means the two runs scored
    different data, and any comparison between them is void. Checking it costs
    nothing and catches a whole class of silently-wrong run.
    """
    published: Dict[str, set] = defaultdict(set)
    observed: Dict[str, set] = defaultdict(set)
    for r in rs.load(dataset_id="original-180", judge="statistical"):
        target = published if r["run_id"].startswith("published__") else observed
        target[r["scenario_id"]].add((r["verdict"], r["score"]))

    shared = sorted(set(published) & set(observed))
    mismatched = [s for s in shared if published[s] != observed[s]]
    internally_split = [s for s in observed if len(observed[s]) > 1]
    return {"scenarios_compared": len(shared),
            "mismatched_vs_published": mismatched,
            "runs_disagreeing_with_each_other": internally_split,
            "ok": not mismatched and not internally_split}


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  -  "
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the prompt ablation")
    ap.add_argument("--dataset", default="", help="restrict to one dataset_id")
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "agg" / "ablation_analysis.json"))
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["original-180", "generalisation-60"]
    report: Dict[str, Any] = {"representation": args.representation, "datasets": {}}

    all_rows = rs.load(include_errors=True)
    err_rows = [r for r in all_rows if r["error_kind"]]
    print(f"frame: {len(all_rows)} rows, {len(err_rows)} error rows "
          f"(excluded from every metric below)")
    if err_rows:
        by = defaultdict(int)
        for r in err_rows:
            by[(r["model"], r["prompt_id"], r["error_kind"])] += 1
        for k, v in sorted(by.items()):
            print(f"   error: {k[0]} {k[1]} {k[2]} x{v}")

    inv = statistical_invariant()
    report["statistical_invariant"] = inv
    status = "OK" if inv["ok"] else "VOID"
    print(f"statistical-judge invariant: {status} "
          f"({inv['scenarios_compared']} scenarios compared against the published run, "
          f"{len(inv['mismatched_vs_published'])} mismatched, "
          f"{len(inv['runs_disagreeing_with_each_other'])} internally inconsistent)")
    if not inv["ok"]:
        print("  the runs below scored different data; treat every comparison as void:")
        for s in (inv["mismatched_vs_published"] + inv["runs_disagreeing_with_each_other"])[:10]:
            print(f"    {s}")

    for dsid in datasets:
        ds_report: Dict[str, Any] = {"models": {}}
        models = sorted({r["model"] for r in rs.load(dataset_id=dsid, judge="ai")})
        if not models:
            print(f"\n### {dsid}: no AI rows in the frame yet")
            continue
        prompts = [p for p in PROMPT_ORDER
                   if rs.load(dataset_id=dsid, judge="ai", prompt_id=p)]

        print(f"\n{'='*100}\n### dataset {dsid}   representation ai:{args.representation}\n{'='*100}")
        truths = {r["truth_label"] for r in rs.load(dataset_id=dsid, judge="ai")}
        if len(truths) == 1:
            only = truths.pop()
            print(f"NOTE: every scenario in {dsid} is labelled {only}. Accuracy therefore "
                  f"equals {'recall' if only == 'FAIL' else 'specificity'}, and balanced "
                  f"accuracy, FPR and MCC are undefined (shown as '-').")

        for model in models:
            ds_report["models"][model] = {}
            print(f"\n-- {model} " + "-" * (96 - len(model)))
            header = (f"{'prompt':26s} {'n':>4s} {'acc':>7s} {'95% CI':>15s} {'prec':>7s} "
                      f"{'rec':>7s} {'F1':>7s} {'bal.acc':>8s} {'MCC':>7s} {'FPR':>7s}  TP/FP/TN/FN")
            print(header)
            base_rows = None
            for pid in prompts:
                rows = rs.load(dataset_id=dsid, judge="ai", representation=args.representation,
                               model=model, prompt_id=pid)
                if not rows:
                    continue
                m = metrics(rows)
                ci = f"[{m['accuracy_ci95'][0]:.3f},{m['accuracy_ci95'][1]:.3f}]"
                print(f"{pid:26s} {m['n']:4d} {fmt(m['accuracy']):>7s} {ci:>15s} "
                      f"{fmt(m['precision']):>7s} {fmt(m['recall']):>7s} {fmt(m['f1']):>7s} "
                      f"{fmt(m['balanced_accuracy']):>8s} {fmt(m['mcc']):>7s} {fmt(m['fpr']):>7s}  "
                      f"{m['TP']}/{m['FP']}/{m['TN']}/{m['FN']}")
                ds_report["models"][model][pid] = {
                    "metrics": m, "family_accuracy": family_accuracy(rows)}
                if pid == FROZEN:
                    base_rows = rows
                elif base_rows:
                    ds_report["models"][model][pid]["vs_frozen"] = paired_diff(base_rows, rows)

            fams = sorted({r["family"] for r in rs.load(dataset_id=dsid, judge="ai", model=model)},
                          key=ed.family_order)
            if fams:
                print(f"\n  per-family accuracy (k/20, Wilson 95%)")
                print("  " + f"{'prompt':26s}" + "".join(f"{f[:19]:>21s}" for f in fams))
                for pid in prompts:
                    entry = ds_report["models"][model].get(pid)
                    if not entry:
                        continue
                    cells = []
                    for f in fams:
                        fa = entry["family_accuracy"].get(f)
                        cells.append(f"{fa['accuracy']:.2f} [{fa['ci95'][0]:.2f},{fa['ci95'][1]:.2f}]".rjust(21)
                                     if fa else " " * 21)
                    print("  " + f"{pid:26s}" + "".join(cells))

            for pid in prompts:
                vs = (ds_report["models"][model].get(pid) or {}).get("vs_frozen")
                if not vs:
                    continue
                print(f"\n  {pid} vs frozen (paired, n={vs['n_paired']}): "
                      f"{vs['verdict_changes']} verdict changes, "
                      f"fixed {vs['b_fixed']} / broke {vs['b_broke']} "
                      f"(net {vs['net']:+d}), McNemar exact p={vs['mcnemar_exact_p']}")
                by_fam = defaultdict(lambda: [0, 0])
                for ch in vs["changes"]:
                    correct_now = ch["to"] == ch["truth"]
                    by_fam[ch["family"]][0 if correct_now else 1] += 1
                for f in sorted(by_fam, key=ed.family_order):
                    good, bad = by_fam[f]
                    print(f"      {f:24s} {good:+d} now correct, {bad} now wrong")

        report["datasets"][dsid] = ds_report

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[ablation] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
