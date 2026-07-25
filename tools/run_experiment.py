#!/usr/bin/env python3
"""The scored experiment.

Iterates every scenario instance in the labelled dataset (tools/eval_dataset.py)
and scores every judge across every model present locally:

  * statistical  : the real NetflixACAJudge (Kayenta /judges/judge on the
                   in-memory pairs; model-independent).
  * ai:summary / ai:raw   : the configurable AI judge, text representations, one
                   run per present text model.
  * ai:plot      : the AI judge, chart representation, one run per present vision
                   model.
  * hybrid:gated / hybrid:or / hybrid:and  : the three hybrid policies
                   (hybrid_policy.decide_policy, the same code the service uses),
                   combining the statistical verdict with each model's AI verdict
                   (text models use their summary verdict, vision models plot).

Every judge sees the same pairs per scenario (built once from the seed), so the
comparison is fair. Absent models are skipped, not failed. Runs are deterministic
(temperature 0, fixed seeds); a preflight runs one scenario twice and checks the
verdict matches.

Outputs (results/): results_raw.csv, summary.csv, summary.md, family_matrix.csv,
mcnemar.csv, kappa.csv, figures/*.png, results.md.

Usage:
  python tools/run_experiment.py [--n 6] [--seed 20260621] [--models a,b]
                                 [--quick] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import eval_dataset as ed
import judge_clients

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))
import hybrid_policy  # noqa: E402  (shared pure policy, imported from the service)

RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REGISTRY = REPO_ROOT / "judge-service" / "models.yaml"
LITELLM_CONFIG = REPO_ROOT / "litellm" / "config.yaml"
AILOG_DIR = REPO_ROOT / "data" / "ai-logs"
OLLAMA_URL = "http://localhost:11434"
PASS_T, MARGINAL_T = ed.SCORE_THRESHOLDS["pass"], ed.SCORE_THRESHOLDS["marginal"]

TEXT_MODES = ["summary", "raw"]
VISION_MODES = ["plot"]
PRIMARY_MODE = {"text": "summary", "vision": "plot"}
HYBRID_POLICIES = ["gated", "or", "and"]
QUICK_MODELS = ["qwen-llm", "moondream-vlm"]


# Model discovery: which registered aliases are actually pulled in Ollama.
def registered_aliases() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in REGISTRY.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(r"\s+([A-Za-z0-9_-]+):\s*\{\s*modality:\s*(text|vision)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def alias_to_tag() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    current: Optional[str] = None
    for line in LITELLM_CONFIG.read_text().splitlines():
        if line.strip().startswith("#"):
            continue
        m = re.search(r"model_name:\s*([A-Za-z0-9_-]+)", line)
        if m:
            current = m.group(1)
        t = re.search(r"model:\s*ollama(?:_chat)?/(\S+)", line)
        if t and current:
            mapping[current] = t.group(1)
            current = None
    return mapping


def ollama_tags() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def discover_models(restrict: Optional[List[str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Return (present, skipped) model dicts {alias, modality, tag}."""
    tags = set(ollama_tags())
    tagmap = alias_to_tag()
    present, skipped = [], []
    for alias, modality in registered_aliases():
        if restrict and alias not in restrict:
            continue
        tag = tagmap.get(alias)
        info = {"alias": alias, "modality": modality, "tag": tag or "?"}
        if tag and tag in tags:
            present.append(info)
        else:
            skipped.append({**info, "reason": "not pulled" if tag else "non-local alias"})
    return present, skipped


# Metrics (stdlib only). FAIL is the positive class.
def confusion(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        truth_fail = r["truth"] == "FAIL"
        pred_fail = r["verdict"] == "FAIL"
        if truth_fail and pred_fail:
            c["TP"] += 1
        elif (not truth_fail) and pred_fail:
            c["FP"] += 1
        elif (not truth_fail) and (not pred_fail):
            c["TN"] += 1
        else:
            c["FN"] += 1
    return c


def metrics_from_conf(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot if tot else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr}


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def mcnemar_exact(correct_a: List[bool], correct_b: List[bool]) -> Dict[str, Any]:
    """Exact (binomial) McNemar on paired per-scenario correctness.
    b = A wrong and B right; c = A right and B wrong. Two-sided exact p."""
    b = sum(1 for x, y in zip(correct_a, correct_b) if (not x) and y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if x and (not y))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2.0 * _binom_cdf(k, n, 0.5))
    stat = (abs(b - c) - 1) ** 2 / n if n else 0.0  # continuity-corrected chi-square
    return {"n01_a_wrong_b_right": b, "n10_a_right_b_wrong": c, "discordant": n,
            "chi2_cc": round(stat, 4), "p_value": round(p, 5)}


def cohen_kappa(a: List[str], b: List[str]) -> float:
    """Cohen's kappa for two raters over the {PASS, FAIL} label set."""
    labels = ["PASS", "FAIL"]
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


# Judge keys.
def judge_key(kind: str, sub: str) -> str:
    return f"{kind}:{sub}" if sub else kind


# Run one scenario across all judges/models.
def run_scenario(sc: ed.Scenario, present: List[Dict[str, str]], base_millis: int,
                 modes_text: List[str], modes_vision: List[str]) -> List[Dict[str, Any]]:
    series = ed.series_for_scenario(sc)
    pairs = ed.build_pairs(sc, series, base_millis)
    stat_cfg = ed.build_canary_config(sc, {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}})

    rows: List[Dict[str, Any]] = []

    # statistical (the real NetflixACAJudge)
    stat = judge_clients.run_default_judge(pairs, stat_cfg, PASS_T, MARGINAL_T)
    d_score = stat["score"]
    d_verdict = hybrid_policy.classify(d_score, PASS_T, MARGINAL_T)
    d_nonpass = any(p["classification"] not in hybrid_policy.PASS_OK for p in stat["per_metric"])
    rows.append(_row(sc, "statistical", "", "-", stat["verdict"], d_score, stat["latency"], stat["error"]))

    # AI judges plus hybrids, per model
    for mdl in present:
        alias, modality = mdl["alias"], mdl["modality"]
        modes = modes_text if modality == "text" else modes_vision
        ai_by_mode: Dict[str, Dict[str, Any]] = {}
        for mode in modes:
            ai = judge_clients.judge_ai(pairs, ed.build_canary_config(sc, {}), mode, alias, PASS_T, MARGINAL_T)
            ai_by_mode[mode] = ai
            rows.append(_row(sc, "ai", mode, alias, ai["verdict"], ai["score"], ai["latency"],
                             ai["error"], rationale=ai["rationale"]))

        # hybrid uses this model's primary representation as the AI half
        prim = PRIMARY_MODE[modality]
        ai = ai_by_mode.get(prim)
        if ai is not None and ai["ok"]:
            a_score = ai["score"]
            a_verdict = hybrid_policy.classify(a_score, PASS_T, MARGINAL_T)
            blind_text = ai["rationale"] + " " + " ".join(p["reason"] for p in ai["per_metric"])
            a_blind = hybrid_policy.text_flags_blindspot(blind_text)
            for pol in HYBRID_POLICIES:
                dec = hybrid_policy.decide_policy(pol, d_score, d_verdict, d_nonpass,
                                                  a_score, a_verdict, a_blind, PASS_T, MARGINAL_T)
                hv = "PASS" if dec.verdict == "Pass" else "FAIL"
                rows.append(_row(sc, "hybrid", pol, alias, hv, dec.score, 0.0, "",
                                 note=f"ai={prim}/{a_verdict};{dec.followed}"))
    return rows


def _row(sc: ed.Scenario, kind: str, sub: str, model: str, verdict: str, score: float,
         latency: float, error: str, rationale: str = "", note: str = "") -> Dict[str, Any]:
    jk = judge_key(kind, sub)
    return {
        "scenario": sc.id, "family": sc.family, "truth": sc.truth,
        "judge": jk, "kind": kind, "rep_or_policy": sub, "model": model,
        "verdict": verdict, "score": round(float(score), 2),
        "correct": int(verdict == sc.truth),
        "latency": round(float(latency), 2), "error": error[:140],
        "rationale": rationale[:200].replace("\n", " "), "note": note,
    }


# Determinism preflight.
def determinism_check(sc: ed.Scenario, model: str, base_millis: int) -> Dict[str, Any]:
    series = ed.series_for_scenario(sc)
    pairs = ed.build_pairs(sc, series, base_millis)
    cfg = ed.build_canary_config(sc, {})
    r1 = judge_clients.judge_ai(pairs, cfg, "summary", model, PASS_T, MARGINAL_T)
    r2 = judge_clients.judge_ai(pairs, cfg, "summary", model, PASS_T, MARGINAL_T)
    identical = (r1["ok"] and r2["ok"] and r1["verdict"] == r2["verdict"]
                 and abs(r1["score"] - r2["score"]) < 1e-6)
    return {"scenario": sc.id, "model": model, "run1": f"{r1['verdict']}({r1['score']})",
            "run2": f"{r2['verdict']}({r2['score']})", "identical": identical}


# Aggregation and output.
def group_key(r: Dict[str, Any]) -> Tuple[str, str]:
    return (r["judge"], r["model"])


def aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, rs in groups.items():
        c = confusion(rs)
        m = metrics_from_conf(c)
        lat = [r["latency"] for r in rs if r["latency"] > 0]
        out[key] = {**c, **m, "n": len(rs), "avg_latency": round(sum(lat) / len(lat), 2) if lat else 0.0}
    return out


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def render_figures() -> bool:
    script = (REPO_ROOT / "tools" / "render_figures.py").read_text()
    try:
        p = subprocess.run(["docker", "compose", "exec", "-T", "judge-service", "python", "-"],
                           input=script, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180)
        if p.returncode != 0:
            print(f"[experiment] figure render failed: {p.stderr[-300:]}", file=sys.stderr)
            return False
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[experiment] figure render error: {e}", file=sys.stderr)
        return False
    src = AILOG_DIR / "_exp_figs"
    if not src.exists():
        return False
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    n = 0
    for png in src.glob("*.png"):
        shutil.copy(png, FIG_DIR / png.name)
        n += 1
    return n > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the scored canary-judge experiment")
    ap.add_argument("--n", type=int, default=int(os.environ.get("EVAL_N", ed.DEFAULT_N_PER_FAMILY)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", ed.DEFAULT_MASTER_SEED)))
    ap.add_argument("--models", default=os.environ.get("EVAL_MODELS", ""),
                    help="comma-separated alias allow-list (default: all present)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (n=1, two default models)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    n = 1 if args.quick else args.n
    restrict = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.quick and not restrict:
        restrict = QUICK_MODELS

    t_start = time.time()
    print("=" * 80)
    print(f"kayenta-ai-canary-judge scored experiment  (n={n}/family, seed={args.seed}, quick={args.quick})")
    print("=" * 80)

    # Health.
    try:
        if requests.get("http://localhost:8090/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("Kayenta not UP")
        if requests.get("http://localhost:5001/health", timeout=10).json().get("status") != "UP":
            raise RuntimeError("judge-service not UP")
    except Exception as e:  # noqa: BLE001
        print(f"[experiment] stack not healthy ({e}); run `make up` first.", file=sys.stderr)
        return 2

    present, skipped = discover_models(restrict or None)
    if not present:
        print("[experiment] no models present locally to evaluate.", file=sys.stderr)
        return 1
    print(f"[experiment] present models: {[m['alias'] for m in present]}")
    if skipped:
        print(f"[experiment] skipped (absent): {[(m['alias'], m['reason']) for m in skipped]}")

    # Build and seed the dataset (load into VM with the Canary scheme, plus manifest).
    scenarios = ed.build_scenarios(args.seed, n)
    base_millis = ed.aligned_base_millis()
    print(f"[experiment] seeding {len(scenarios)} scenarios into VictoriaMetrics ...")
    import seed_eval_dataset as seeder
    seeder.seed_vm(seeder.build_payload(scenarios, base_millis))
    manifest = ed.build_manifest(args.seed, n, scenarios, base_millis)
    (REPO_ROOT / "data" / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Determinism preflight.
    det = determinism_check(scenarios[0], present[0]["alias"], base_millis)
    print(f"[experiment] determinism: {det['model']} on {det['scenario']} -> "
          f"{det['run1']} vs {det['run2']} -> {'IDENTICAL' if det['identical'] else 'DIFFERENT'}")

    # Main sweep.
    all_rows: List[Dict[str, Any]] = []
    text_models = [m for m in present if m["modality"] == "text"]
    vision_models = [m for m in present if m["modality"] == "vision"]
    n_ai = len(text_models) * len(TEXT_MODES) + len(vision_models) * len(VISION_MODES)
    print(f"[experiment] {len(scenarios)} scenarios x ({n_ai} AI calls + statistical + hybrids) ...")
    for i, sc in enumerate(scenarios, 1):
        t0 = time.time()
        rows = run_scenario(sc, present, base_millis, TEXT_MODES, VISION_MODES)
        all_rows.extend(rows)
        print(f"  [{i:3d}/{len(scenarios)}] {sc.id:24s} truth={sc.truth} "
              f"({time.time()-t0:.0f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_outputs(all_rows, present, skipped, det, manifest, args, n,
                  no_figures=args.no_figures, elapsed=time.time() - t_start)
    print(f"\n[experiment] DONE in {(time.time()-t_start)/60:.1f} min -> {RESULTS_DIR}")
    return 0


def write_outputs(rows, present, skipped, det, manifest, args, n, no_figures, elapsed):
    # results_raw.csv
    raw_header = ["scenario", "family", "truth", "judge", "kind", "rep_or_policy", "model",
                  "verdict", "score", "correct", "latency", "error", "note", "rationale"]
    write_csv(RESULTS_DIR / "results_raw.csv", raw_header,
              [[r[k] for k in raw_header] for r in rows])

    # aggregate scoreboard (judge x model)
    agg = aggregate(rows)
    order = _judge_order(agg)
    sum_header = ["judge", "model", "n", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
                  "TP", "FP", "TN", "FN", "avg_latency"]
    sum_rows = []
    for key in order:
        a = agg[key]
        sum_rows.append([key[0], key[1], a["n"], r3(a["accuracy"]), r3(a["precision"]), r3(a["recall"]),
                         r3(a["f1"]), r3(a["fpr"]), r3(a["fnr"]), a["TP"], a["FP"], a["TN"], a["FN"],
                         a["avg_latency"]])
    write_csv(RESULTS_DIR / "summary.csv", sum_header, sum_rows)

    # family matrix (judge x model x family accuracy)
    families = manifest["families"]
    fam_header = ["judge", "model"] + families
    fam_rows = []
    by_group_family = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_group_family[(r["judge"], r["model"])][r["family"]].append(r["correct"])
    for key in order:
        row = [key[0], key[1]]
        for fam in families:
            vals = by_group_family[key].get(fam, [])
            row.append(r3(sum(vals) / len(vals)) if vals else "")
        fam_rows.append(row)
    write_csv(RESULTS_DIR / "family_matrix.csv", fam_header, fam_rows)

    # McNemar and kappa for key judge pairs (per model)
    mcnemar_rows, kappa_rows = _paired_stats(rows, present)
    write_csv(RESULTS_DIR / "mcnemar.csv",
              ["model", "judge_a", "judge_b", "discordant", "n01_a_wrong_b_right",
               "n10_a_right_b_wrong", "chi2_cc", "p_value"], mcnemar_rows)
    # kappa vs ground truth (per judge x model), plus judge-to-judge
    kappa_truth = []
    for key in order:
        labs = [r["verdict"] for r in rows if (r["judge"], r["model"]) == key]
        truths = [r["truth"] for r in rows if (r["judge"], r["model"]) == key]
        kappa_truth.append([key[0], key[1], "ground_truth", r3(cohen_kappa(labs, truths))])
    write_csv(RESULTS_DIR / "kappa.csv", ["judge_a", "model", "judge_b", "kappa"],
              kappa_truth + kappa_rows)

    # figures
    figs_ok = False
    if not no_figures:
        _write_figdata(rows, agg, order, families, present)
        figs_ok = render_figures()

    # summary.md and results.md
    _write_summary_md(sum_header, sum_rows, order, agg)
    _write_results_md(rows, agg, order, families, present, skipped, det, manifest, args, n,
                      mcnemar_rows, figs_ok, elapsed)


def r3(x):
    return round(float(x), 3)


def _judge_order(agg) -> List[Tuple[str, str]]:
    # statistical first, then ai:summary/raw/plot, then hybrids; models sorted.
    def rank(key):
        j = key[0]
        base = {"statistical": 0}.get(j, 5)
        if j.startswith("ai:"):
            base = 1 + {"ai:summary": 0, "ai:raw": 1, "ai:plot": 2}.get(j, 3)
        if j.startswith("hybrid:"):
            base = 10 + {"hybrid:gated": 0, "hybrid:or": 1, "hybrid:and": 2}.get(j, 3)
        return (base, key[1])
    return sorted(agg.keys(), key=rank)


def _paired_stats(rows, present):
    """Per model: statistical vs ai:summary, statistical vs hybrid:gated,
    ai:summary vs hybrid:gated (paired by scenario)."""
    mcnemar_rows, kappa_rows = [], []
    # index correctness/verdict by (judge, model, scenario)
    idx_correct: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    idx_verdict: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for r in rows:
        idx_correct[(r["judge"], r["model"])][r["scenario"]] = bool(r["correct"])
        idx_verdict[(r["judge"], r["model"])][r["scenario"]] = r["verdict"]
    stat_key = ("statistical", "-")
    for mdl in present:
        alias, modality = mdl["alias"], mdl["modality"]
        ai_key = (f"ai:{PRIMARY_MODE[modality]}", alias)
        hyb_key = ("hybrid:gated", alias)
        pairs = [("statistical", stat_key, f"ai:{PRIMARY_MODE[modality]}", ai_key),
                 ("statistical", stat_key, "hybrid:gated", hyb_key),
                 (f"ai:{PRIMARY_MODE[modality]}", ai_key, "hybrid:gated", hyb_key)]
        for na, ka, nb, kb in pairs:
            scen = sorted(set(idx_correct[ka]) & set(idx_correct[kb]))
            if not scen:
                continue
            ca = [idx_correct[ka][s] for s in scen]
            cb = [idx_correct[kb][s] for s in scen]
            mc = mcnemar_exact(ca, cb)
            mcnemar_rows.append([alias, na, nb, mc["discordant"], mc["n01_a_wrong_b_right"],
                                 mc["n10_a_right_b_wrong"], mc["chi2_cc"], mc["p_value"]])
            va = [idx_verdict[ka][s] for s in scen]
            vb = [idx_verdict[kb][s] for s in scen]
            kappa_rows.append([na, alias, nb, r3(cohen_kappa(va, vb))])
    return mcnemar_rows, kappa_rows


def _write_figdata(rows, agg, order, families, present):
    # bar: per judge_key averaged over models
    bar: Dict[str, Dict[str, float]] = {}
    by_judge = defaultdict(list)
    for key in order:
        by_judge[key[0]].append(key)
    for j, keys in by_judge.items():
        for mname in ("accuracy", "precision", "recall", "f1"):
            bar.setdefault(j, {})[mname] = round(sum(agg[k][mname] for k in keys) / len(keys), 3)
    # heatmap: judge_key x family accuracy (averaged over models)
    judges = list(by_judge.keys())
    z = []
    for j in judges:
        keys = by_judge[j]
        rowz = []
        for fam in families:
            accs = []
            for key in keys:
                vals = [r["correct"] for r in rows if (r["judge"], r["model"]) == key and r["family"] == fam]
                if vals:
                    accs.append(sum(vals) / len(vals))
            rowz.append(round(sum(accs) / len(accs), 3) if accs else 0.0)
        z.append(rowz)
    # confusion for headline judges (aggregated over models)
    conf = {}
    for j in ("statistical", "ai:summary", "hybrid:gated"):
        keys = [k for k in order if k[0] == j]
        if keys:
            rs = [r for r in rows if r["judge"] == j]
            conf[j] = confusion(rs)
    figdata = {"bar": bar, "heatmap": {"rows": judges, "cols": families, "z": z}, "confusion": conf}
    AILOG_DIR.mkdir(parents=True, exist_ok=True)
    (AILOG_DIR / "_exp_figdata.json").write_text(json.dumps(figdata))


def _write_summary_md(sum_header, sum_rows, order, agg):
    lines = ["# Experiment scoreboard (Table 5)", "",
             "FAIL is the positive class. One row per (judge, model).", ""]
    lines.append("| " + " | ".join(sum_header) + " |")
    lines.append("|" + "|".join(["---"] * len(sum_header)) + "|")
    for row in sum_rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def _write_results_md(rows, agg, order, families, present, skipped, det, manifest, args, n,
                      mcnemar_rows, figs_ok, elapsed):
    total_scen = len({r["scenario"] for r in rows})
    # best per judge family + overall picks
    by_judge_overall = defaultdict(list)
    for key in order:
        by_judge_overall[key[0]].append((key[1], agg[key]["accuracy"], agg[key]["f1"]))

    def avg_acc(j):
        ks = [k for k in order if k[0] == j]
        return sum(agg[k]["accuracy"] for k in ks) / len(ks) if ks else 0.0

    judges = sorted({k[0] for k in order}, key=lambda j: -avg_acc(j))
    best_judge = judges[0] if judges else "-"

    # per-family best judge
    fam_best = {}
    for fam in families:
        scores = {}
        for j in {k[0] for k in order}:
            ks = [k for k in order if k[0] == j]
            accs = []
            for key in ks:
                vals = [r["correct"] for r in rows if (r["judge"], r["model"]) == key and r["family"] == fam]
                if vals:
                    accs.append(sum(vals) / len(vals))
            if accs:
                scores[j] = sum(accs) / len(accs)
        if scores:
            fam_best[fam] = max(scores.items(), key=lambda kv: kv[1])

    L: List[str] = []
    L.append("# kayenta-ai-canary-judge: scored judge experiment (results)")
    L.append("")
    L.append(f"_Generated {manifest['generated_at']} · {total_scen} scenarios "
             f"({len(families)} families × n={n}) · {elapsed/60:.1f} min._")
    L.append("")
    L.append("## Setup")
    L.append(f"- **Dataset**: {len(families)} families × {n} instances = {total_scen} labelled scenarios, "
             f"master seed {manifest['master_seed']}, deterministic.")
    L.append(f"- **Models present**: {', '.join(m['alias'] for m in present)}.")
    if skipped:
        L.append(f"- **Models skipped (absent)**: {', '.join(m['alias'] for m in skipped)}.")
    L.append(f"- **Statistical config (uniform, best-practice)**: direction=increase, "
             f"effectSize=meanRatio (allowedIncrease 1.05 / criticalIncrease 1.25), nanStrategy=remove, "
             f"outliers=keep, critical=true, mustHaveData=true; scoreThresholds pass={int(PASS_T)}/marginal={int(MARGINAL_T)}.")
    L.append(f"- **AI prompt**: frozen ({_prompt_version()}); temperature 0; structured JSON; "
             f"determinism preflight on {det['scenario']}/{det['model']}: "
             f"{'IDENTICAL ✓' if det['identical'] else 'DIFFERENT ✗'} ({det['run1']} vs {det['run2']}).")
    L.append("")

    L.append("## Scoreboard (Table 5): accuracy / precision / recall / F1, FAIL = positive")
    L.append("")
    L.append("| judge | model | acc | prec | recall | F1 | FPR | FNR |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in order:
        a = agg[key]
        L.append(f"| {key[0]} | {key[1]} | {a['accuracy']:.2f} | {a['precision']:.2f} | "
                 f"{a['recall']:.2f} | {a['f1']:.2f} | {a['fpr']:.2f} | {a['fnr']:.2f} |")
    L.append("")

    L.append("## Per-family accuracy (Table 6), averaged over models")
    L.append("")
    judge_keys_unique = []
    for k in order:
        if k[0] not in judge_keys_unique:
            judge_keys_unique.append(k[0])
    L.append("| judge | " + " | ".join(families) + " |")
    L.append("|" + "|".join(["---"] * (len(families) + 1)) + "|")
    for j in judge_keys_unique:
        cells = [j]
        for fam in families:
            accs = []
            for key in [k for k in order if k[0] == j]:
                vals = [r["correct"] for r in rows if (r["judge"], r["model"]) == key and r["family"] == fam]
                if vals:
                    accs.append(sum(vals) / len(vals))
            cells.append(f"{sum(accs)/len(accs):.2f}" if accs else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L.append("## McNemar (Table 7): paired correctness, exact two-sided p")
    L.append("")
    L.append("| model | judge A | judge B | discordant | A✗B✓ | A✓B✗ | χ²(cc) | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mr in mcnemar_rows:
        L.append("| " + " | ".join(str(x) for x in mr) + " |")
    L.append("")

    if figs_ok:
        L.append("## Figures")
        L.append("- ![accuracy](figures/accuracy_bars.png)")
        L.append("- ![family heatmap](figures/family_heatmap.png)")
        L.append("- ![confusion](figures/confusion_matrices.png)")
        L.append("")

    # auto-interpretation
    L.append("## Interpretation (auto-generated)")
    L.append("")
    stat_acc = agg.get(("statistical", "-"), {}).get("accuracy")
    L.append(f"- **Statistical (NetflixACAJudge)** overall accuracy "
             f"{stat_acc:.2f}." if stat_acc is not None else "- Statistical row missing.")
    # blind-spot families
    blind = ["variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]
    blind_present = [f for f in blind if f in families]
    if stat_acc is not None and blind_present:
        sb = []
        for fam in blind_present:
            vals = [r["correct"] for r in rows if r["judge"] == "statistical" and r["family"] == fam]
            if vals:
                sb.append(f"{fam} {sum(vals)/len(vals):.0%}")
        L.append(f"- On the structural blind-spot families it is weak: {', '.join(sb)} "
                 f"(median/rank test is blind to variance, tails, temporal order and cross-metric dilution).")
    # AI best on blind spots
    for j in ("ai:summary", "ai:raw", "ai:plot"):
        ks = [k for k in order if k[0] == j]
        if not ks:
            continue
        bacc = []
        for fam in blind_present:
            vals = [r["correct"] for r in rows for key in ks
                    if (r["judge"], r["model"]) == key and r["family"] == fam]
            if vals:
                bacc.append(sum(vals) / len(vals))
        if bacc:
            L.append(f"- **{j}** blind-spot accuracy (avg over models & those families): {sum(bacc)/len(bacc):.2f}.")
    # hybrid overall
    for j in ("hybrid:gated", "hybrid:or", "hybrid:and"):
        ks = [k for k in order if k[0] == j]
        if ks:
            L.append(f"- **{j}** overall accuracy (avg over models): {avg_acc(j):.2f}.")
    L.append(f"- **Best judge family overall (by mean accuracy over models): `{best_judge}`** "
             f"({avg_acc(best_judge):.2f}).")
    L.append("- Per-family winner: " + ", ".join(f"{fam}→{fam_best[fam][0]}({fam_best[fam][1]:.0%})"
                                                 for fam in families if fam in fam_best) + ".")
    L.append("")
    L.append("> FAIL families with genuine PASS counterparts make these numbers meaningful: a judge that "
             "always says FAIL is penalised on `no_change`/`noise_equivalent`/`healed_transient`.")
    (RESULTS_DIR / "results.md").write_text("\n".join(L) + "\n")


def _prompt_version() -> str:
    try:
        txt = (REPO_ROOT / "judge-service" / "app" / "llm_judge.py").read_text()
        m = re.search(r'PROMPT_VERSION\s*=\s*"([^"]+)"', txt)
        return m.group(1) if m else "frozen"
    except OSError:
        return "frozen"


if __name__ == "__main__":
    sys.exit(main())
