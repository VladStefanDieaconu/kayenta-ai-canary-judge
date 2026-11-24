#!/usr/bin/env python3
"""D3: model rationales against measurable window statistics (Reviewer 2, comment 8).

For each scored model judgement whose full rationale is on record, extract the
checkable claims of the rationale (level, spread, trend, tail and quoted numbers)
with the patterns of d3_rationale_config.json, and check each claim against the
statistics of the window that was judged: the window's own series and the summary
packet the model saw. Report per configuration:

  * the share of rationales with at least one checkable claim;
  * among checkable claims, the share consistent with the window;
  * among FAIL verdicts, the share whose stated reason is absent from the data
    (every level-up, spread-up, trend-up and tail-up claim inconsistent).

Automatic extraction is lexical. It also matches negated, hedged and malformed
text, as Text S8 of the Supplementary Materials warns for the lexical audit; a
random sample of extracted claims is written for an author's hand check.

Inputs (one or both):
  --archive DIR   the published data deposit: the name-attribution arms
                  results/name-attribution-r1 and results/name-attribution-r2-hosted,
                  whose judge-call payload records keep the full rationale and the
                  prompt (headline rationales are cut to 200 characters and are not used);
  --records CSV.. long-format records (tools/results_schema.py columns) of new
                  runs, with the window series from --series-jsonl FILE.. or
                  regenerated from tools/eval_dataset.py with --generate DATASET_ID..
                  (master seed --master-seed, --n-per-family windows per family).

Run (from the repository root):
  python -I -B analysis/r1/posthoc/d3_rationale_check.py --archive <data-archive> --out <new-dir>
  python -I -B analysis/r1/posthoc/d3_rationale_check.py --records <run>.csv ... --generate <dataset-id> ... --out <new-dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import posthoc_common as pc  # noqa: E402

SCRIPT = "analysis/r1/posthoc/d3_rationale_check.py"
TYPES = ["level", "spread", "trend", "tail", "number"]
PROBLEM = {"level": "up", "spread": "up", "trend": "up", "tail": "up"}
KIND_OF_NAME = [("latency", re.compile(r"latenc|duration|http_request", re.I)),
                ("memory", re.compile(r"memory|resident|bytes", re.I)),
                ("errors", re.compile(r"error", re.I))]


def kind_of_metric(name: str) -> str:
    for k, rx in KIND_OF_NAME:
        if rx.search(name):
            return k
    return "unknown"


# ----------------------------------------------------------------------------- extraction
class Extractor:
    def __init__(self, cfg: Dict[str, Any]):
        I = re.I
        self.cfg = cfg
        self.split = re.compile(cfg["clause_split"], I)
        self.neg = re.compile(cfg["negation"]["cues"], I)
        self.neg_n = int(cfg["negation"]["max_words_before"])
        self.kinds = {k: re.compile(v, I) for k, v in cfg["metric_kinds"].items()}
        dw = cfg["direction_words"]
        self.dir = {d: re.compile(dw[d], I) for d in ("up", "down", "same")}
        self.win = int(dw["window_words"])
        self.types = {}
        lv = cfg["claims"]["level"]
        self.level_fallback = re.compile(lv["fallback_triggers"], I) if lv.get("fallback_triggers") else None
        for t in ("level", "spread", "trend", "tail"):
            c = cfg["claims"][t]
            self.types[t] = {"trig": re.compile(c["triggers"], I),
                             "excl": re.compile(c["exclude_contexts"], I) if c.get("exclude_contexts") else None,
                             "up": re.compile(c["phrases_up"], I), "down": re.compile(c["phrases_down"], I),
                             "same": re.compile(c["phrases_same"], I)}
        nc = cfg["claims"]["number"]
        self.num = re.compile(nc["pattern"], I)
        self.num_before = re.compile(nc["exclude_contexts_before"], I)
        self.num_after = re.compile(nc["exclude_contexts_after"], I)
        self.num_excl_values = set(float(x) for x in nc.get("exclude_values", []))
        self.placeholders = [re.compile(p, I) for p in cfg["unit"]["placeholder_patterns"]]

    def is_placeholder(self, text: str) -> bool:
        return any(p.search(text.strip()) for p in self.placeholders)

    def clauses(self, text: str) -> List[str]:
        parts = [p.strip() for p in self.split.split(text or "") if p and p.strip()]
        return parts

    @staticmethod
    def _word_index(clause: str, pos: int) -> int:
        return len(re.findall(r"\S+", clause[:pos]))

    def _negated(self, clause: str, positions: List[int]) -> bool:
        words = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", clause)]
        for pos in positions:
            wi = self._word_index(clause, pos)
            lo = max(0, wi - self.neg_n)
            window = " ".join(w for _, w in words[lo:wi])
            if self.neg.search(window):
                return True
        return False

    def _named_kinds(self, clause: str) -> List[str]:
        return [k for k, rx in self.kinds.items() if rx.search(clause)]

    def extract(self, text: str) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        if not text or self.is_placeholder(text):
            return claims
        for ci, clause in enumerate(self.clauses(text)):
            kinds = self._named_kinds(clause)
            for t, R in self.types.items():
                excl_spans = [m.span() for m in R["excl"].finditer(clause)] if R["excl"] else []
                trig = [m for m in R["trig"].finditer(clause)
                        if not any(a <= m.start() < b for a, b in excl_spans)]
                if not trig:
                    continue
                direction, dpos, how = None, None, ""
                # 1. phrases of the type, nearest to the first trigger; a phrase lying inside a longer phrase
                #    match (e.g. 'stable' inside 'less stable') is not a match of its own
                ph = []
                for d in ("up", "down", "same"):
                    for m in R[d].finditer(clause):
                        if any(a <= m.start() < b for a, b in excl_spans):
                            continue
                        ph.append((d, m))
                ph = [(d, m) for d, m in ph if not any(
                    (m2.start() <= m.start() and m.end() <= m2.end() and (m2.end() - m2.start()) > (m.end() - m.start()))
                    for _, m2 in ph)]
                best = None
                for d, m in ph:
                    dist = min(abs(m.start() - tg.start()) for tg in trig)
                    key = (dist, ("up", "down", "same").index(d))
                    if best is None or key < best[0]:
                        best = (key, d, m)
                if best is not None:
                    direction, dpos, how = best[1], best[2].start(), f"phrase '{best[2].group(0)}'"
                else:
                    # 2. direction words within the word window of a trigger: up/down first, then same
                    words = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", clause)]
                    tw = [self._word_index(clause, tg.start()) for tg in trig]
                    for group in (("up", "down"), ("same",)):
                        cand = []
                        for d in group:
                            for m in self.dir[d].finditer(clause):
                                wi = self._word_index(clause, m.start())
                                dist = min(abs(wi - x) for x in tw)
                                if dist <= self.win and not any(a <= m.start() < b for a, b in excl_spans):
                                    # a direction word that is itself a trigger of the type does not count
                                    if any(tg.start() == m.start() for tg in trig):
                                        continue
                                    cand.append((dist, m.start(), d, m))
                        if cand:
                            cand.sort(key=lambda x: (x[0], x[1]))
                            _, _, direction, mm = cand[0]
                            dpos, how = mm.start(), f"word '{mm.group(0)}'"
                            break
                negated = self._negated(clause, [trig[0].start()] + ([dpos] if dpos is not None else []))
                if direction is None:
                    if t == "trend" and negated:
                        # "no trend", "without any drift": the negated trend term itself says none
                        direction, how = "same", "negated trend term"
                    else:
                        claims.append({"type": t, "direction": "undetermined", "negated": negated,
                                       "trigger": trig[0].group(0), "how": "no direction word", "clause": clause,
                                       "clause_index": ci, "kinds": kinds})
                        continue
                elif negated:
                    direction = {"up": "not_up", "down": "not_down", "same": "differs"}[direction]
                claims.append({"type": t, "direction": direction, "negated": negated, "trigger": trig[0].group(0),
                               "how": how, "clause": clause, "clause_index": ci, "kinds": kinds,
                               "named": self._named_stat(t, clause)})
            # level fallback: a metric word with a direction word, when the clause made no other claim
            if self.level_fallback is not None and not any(c["clause_index"] == ci and c["clause"] == clause for c in claims):
                fb = list(self.level_fallback.finditer(clause))
                if fb:
                    tw = [self._word_index(clause, m.start()) for m in fb]
                    best = None
                    for group in (("up", "down"), ("same",)):
                        for d in group:
                            for m in self.dir[d].finditer(clause):
                                dist = min(abs(self._word_index(clause, m.start()) - x) for x in tw)
                                if dist <= self.win and (best is None or (dist, m.start()) < best[0]):
                                    best = ((dist, m.start()), d, m)
                        if best is not None:
                            break
                    if best is not None:
                        direction = best[1]
                        negated = self._negated(clause, [fb[0].start(), best[2].start()])
                        if negated:
                            direction = {"up": "not_up", "down": "not_down", "same": "differs"}[direction]
                        claims.append({"type": "level", "direction": direction, "negated": negated,
                                       "trigger": fb[0].group(0), "how": f"fallback: word '{best[2].group(0)}'",
                                       "clause": clause, "clause_index": ci, "kinds": kinds, "named": "either"})
            # quoted numbers
            for m in self.num.finditer(clause):
                before = clause[max(0, m.start() - 14):m.start()]
                after = clause[m.end():m.end() + 12]
                if self.num_before.search(before) or self.num_after.search(after):
                    continue
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                if val in self.num_excl_values:
                    continue
                unit = (m.group(2) or "").strip().lower()
                unit = "%" if unit == "%" else ("x" if unit in ("x", "×") else "")
                # a unit sign the pattern's look-ahead left out (e.g. '36%.' at the end of a sentence)
                if not unit and re.match(r"^\s?%", after):
                    unit = "%"
                elif not unit and re.match("^(?:x\\b|\u00d7)", after):
                    unit = "x"
                if not unit and re.match(r"^\s*percent\b", after, re.I):
                    unit = "%"
                dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
                claims.append({"type": "number", "direction": "value", "value": abs(val), "unit": unit, "decimals": dec,
                               "negated": False, "trigger": m.group(0).strip(), "how": "number", "clause": clause,
                               "clause_index": ci, "kinds": kinds,
                               "pct_context": bool(re.search(r"^\s*%?\s*(?:higher|lower|increase|decrease|more|less|above|below|up|down|greater|smaller)",
                                                             after, re.I))})
        return claims

    @staticmethod
    def _named_stat(t: str, clause: str) -> str:
        c = clause.lower()
        if t == "level":
            has_med = bool(re.search(r"\bmedians?\b", c))
            has_mean = bool(re.search(r"\b(?:mean|means|average|averages|avg)\b", c))
            return "median" if has_med and not has_mean else ("mean" if has_mean and not has_med else "either")
        if t == "tail":
            for k in ("p99", "p95", "p90"):
                if k in c:
                    return k
            if re.search(r"\b(?:max|maximum|maxima|peak|peaks)\b", c):
                return "max"
            return "tail"
        return ""


# ----------------------------------------------------------------------------- window statistics
def describe_np(x: np.ndarray) -> Dict[str, float]:
    return {"n": float(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "stddev": float(x.std(ddof=0)), "stddev_sample": float(x.std(ddof=1)), "min": float(x.min()),
            "max": float(x.max()), "slope_per_sample": float(np.polyfit(np.arange(x.size), x, 1)[0])}


def ratio(a: float, b: float) -> Optional[float]:
    return a / b if b else None


class Window:
    def __init__(self, wid: str, family: str, truth: str, metrics: List[Dict[str, Any]], mk_fn):
        self.id, self.family, self.truth = wid, family, truth
        self.metrics = []
        for m in metrics:
            b, c = np.asarray(m["control"], float), np.asarray(m["canary"], float)
            db, dc = describe_np(b), describe_np(c)
            diff = c - b
            s, p = mk_fn([float(x) for x in diff])
            st = {"kind": m["kind"], "name": m["name"], "control": db, "canary": dc,
                  "mean_ratio": ratio(dc["mean"], db["mean"]), "median_ratio": ratio(dc["median"], db["median"]),
                  "sd_ratio": ratio(dc["stddev_sample"], db["stddev_sample"]),
                  "p90_ratio": ratio(dc["p90"], db["p90"]), "p95_ratio": ratio(dc["p95"], db["p95"]),
                  "p99_ratio": ratio(dc["p99"], db["p99"]), "max_ratio": ratio(dc["max"], db["max"]),
                  "min_ratio": ratio(dc["min"], db["min"]),
                  "slope_diff": float(np.polyfit(np.arange(diff.size), diff, 1)[0]), "mk_s": s, "mk_p": p,
                  "mean_diff": dc["mean"] - db["mean"]}
            self.metrics.append(st)

    def summary_text(self) -> str:
        out = []
        for st in self.metrics:
            f = lambda x: "NA" if x is None else f"{x:.3f}"  # noqa: E731
            out.append(f"{st['kind']}: mean ratio {f(st['mean_ratio'])}, median ratio {f(st['median_ratio'])}, "
                       f"SD ratio {f(st['sd_ratio'])}, p95 ratio {f(st['p95_ratio'])}, max ratio {f(st['max_ratio'])}, "
                       f"slope(canary-control) {st['slope_diff']:.3g}/sample, Mann-Kendall S {st['mk_s']:.0f} p {st['mk_p']:.3g}")
        return " | ".join(out)


# ----------------------------------------------------------------------------- checking
def band(cfg, t):
    return tuple(cfg["claims"][t].get("same_band", [0.95, 1.05]))


def judge_ratio(direction: str, r: Optional[float], lo: float, hi: float) -> Optional[bool]:
    if r is None:
        return None
    return {"up": r > 1.0, "down": r < 1.0, "same": lo <= r <= hi, "not_up": r <= hi, "not_down": r >= lo,
            "differs": r < lo or r > hi}[direction]


ANY_DIRS = {"up", "down", "differs"}


def combine(direction: str, vals: List[Optional[bool]]) -> Optional[bool]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return any(vals) if direction in ANY_DIRS else all(vals)


def check(claim: Dict[str, Any], w: Window, packet: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    """(status, statistic used, values) with status consistent | inconsistent | undetermined."""
    t, d = claim["type"], claim["direction"]
    if d == "undetermined":
        return "undetermined", "", ""
    mets = w.metrics
    named = [k for k in claim.get("kinds", []) if any(m["kind"] == k for m in w.metrics)]
    if len(named) == 1:
        mets = [m for m in w.metrics if m["kind"] == named[0]]
    if t == "number":
        return check_number(claim, w, packet, cfg)
    if t == "level":
        lo, hi = band(cfg, "level")
        which = claim.get("named", "either")
        keys = {"mean": ["mean_ratio"], "median": ["median_ratio"], "either": ["mean_ratio", "median_ratio"]}[which]
        per_metric = []
        for m in mets:
            r = [judge_ratio(d, m[k], lo, hi) for k in keys]
            r = [x for x in r if x is not None]
            per_metric.append((any(r) if r else None) if which == "either" else (r[0] if r else None))
        res = combine(d, per_metric)
        vals = "; ".join(f"{m['kind']} " + ", ".join(f"{k} {m[k]:.4f}" for k in keys if m[k] is not None) for m in mets)
        return ("undetermined" if res is None else ("consistent" if res else "inconsistent")), "+".join(keys), vals
    if t == "spread":
        lo, hi = band(cfg, "spread")
        res = combine(d, [judge_ratio(d, m["sd_ratio"], lo, hi) for m in mets])
        vals = "; ".join(f"{m['kind']} sd_ratio {m['sd_ratio']:.4f}" for m in mets if m["sd_ratio"] is not None)
        return ("undetermined" if res is None else ("consistent" if res else "inconsistent")), "sd_ratio", vals
    if t == "trend":
        a = float(cfg["claims"]["trend"]["alpha"])
        per = []
        for m in mets:
            up = m["slope_diff"] > 0 and m["mk_p"] < a and m["mk_s"] > 0
            dn = m["slope_diff"] < 0 and m["mk_p"] < a and m["mk_s"] < 0
            per.append({"up": up, "down": dn, "same": m["mk_p"] >= a, "not_up": not up, "not_down": not dn,
                        "differs": m["mk_p"] < a}[d])
        res = combine(d, per)
        vals = "; ".join(f"{m['kind']} slope {m['slope_diff']:.4g}, MK S {m['mk_s']:.0f}, p {m['mk_p']:.4g}" for m in mets)
        return ("consistent" if res else "inconsistent"), "slope(canary-control)+Mann-Kendall", vals
    if t == "tail":
        lo, hi = band(cfg, "tail")
        nm = claim.get("named", "tail")
        per = []
        for m in mets:
            if nm in ("p90", "p95", "p99", "max"):
                per.append(judge_ratio(d, m[f"{nm}_ratio"], lo, hi))
            else:
                rs = [m["max_ratio"], m["p95_ratio"]]
                if any(x is None for x in rs):
                    per.append(None)
                    continue
                per.append({"up": max(rs) > 1.0, "down": min(rs) < 1.0, "same": all(lo <= x <= hi for x in rs),
                            "not_up": all(x <= hi for x in rs), "not_down": all(x >= lo for x in rs),
                            "differs": any(x < lo or x > hi for x in rs)}[d])
        res = combine(d, per)
        keys = [f"{nm}_ratio"] if nm in ("p90", "p95", "p99", "max") else ["max_ratio", "p95_ratio"]
        vals = "; ".join(f"{m['kind']} " + ", ".join(f"{k} {m[k]:.4f}" for k in keys if m[k] is not None) for m in mets)
        return ("undetermined" if res is None else ("consistent" if res else "inconsistent")), "+".join(keys), vals
    return "undetermined", "", ""


def trend_sign_only(claim: Dict[str, Any], w: Window) -> str:
    """Secondary trend result of the configuration: up and down judged on the sign of the slope alone."""
    if claim["type"] != "trend" or claim["direction"] not in ("up", "down"):
        return ""
    mets = w.metrics
    named = [k for k in claim.get("kinds", []) if any(m["kind"] == k for m in w.metrics)]
    if len(named) == 1:
        mets = [m for m in w.metrics if m["kind"] == named[0]]
    ok = any((m["slope_diff"] > 0) if claim["direction"] == "up" else (m["slope_diff"] < 0) for m in mets)
    return "consistent" if ok else "inconsistent"


def number_candidates(w: Window, packet: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
    plain, pct, mult = [], [], []
    for m in w.metrics:
        for side in ("control", "canary"):
            for k, v in m[side].items():
                plain.append((f"{m['kind']} {side} {k}", v))
        plain.append((f"{m['kind']} mean difference", m["mean_diff"]))
        for k in ("mean_ratio", "median_ratio", "sd_ratio", "p90_ratio", "p95_ratio", "p99_ratio", "max_ratio", "min_ratio"):
            r = m[k]
            if r is None:
                continue
            plain.append((f"{m['kind']} {k}", r))
            mult.append((f"{m['kind']} {k}", r))
            if r:
                mult.append((f"{m['kind']} 1/{k}", 1 / r))
            pct += [(f"{m['kind']} 100*({k}-1)", 100 * (r - 1)), (f"{m['kind']} 100*(1-{k})", 100 * (1 - r)),
                    (f"{m['kind']} 100*{k}", 100 * r)]
        plain.append((f"{m['kind']} slope(canary-control)", m["slope_diff"]))
    for blk in packet or []:
        nm = blk.get("name", "")
        for side in ("control", "experiment"):
            for k, v in (blk.get(side) or {}).items():
                if isinstance(v, (int, float)):
                    plain.append((f"packet {nm} {side} {k}", float(v)))
        for k in ("experiment_minus_control_mean", "experiment_over_control_mean"):
            v = blk.get(k)
            if isinstance(v, (int, float)):
                plain.append((f"packet {nm} {k}", float(v)))
                if k.endswith("over_control_mean"):
                    mult.append((f"packet {nm} {k}", float(v)))
                    pct += [(f"packet 100*({k}-1)", 100 * (float(v) - 1)), (f"packet 100*(1-{k})", 100 * (1 - float(v)))]
    return {"": plain, "%": pct, "x": mult}


def check_number(claim, w, packet, cfg):
    tol = float(cfg["claims"]["number"]["relative_tolerance"])
    x = claim["value"]
    half = 0.5 * 10 ** (-claim["decimals"])
    cands = number_candidates(w, packet)
    pool = cands[claim["unit"]] if claim["unit"] else (cands[""] + (cands["%"] if claim.get("pct_context") else []))
    best = None
    for name, c in pool:
        if c is None or not math.isfinite(c):
            continue
        err = abs(abs(x) - abs(c))
        ok = (abs(c) > 0 and err <= tol * abs(c)) or err <= half
        if ok and (best is None or err < best[2]):
            best = (name, c, err)
    if best:
        return "consistent", "number", f"{x:g}{claim['unit']} matches {best[0]} = {best[1]:.6g}"
    return "inconsistent", "number", f"{x:g}{claim['unit']}: no statistic within 5%"


# ----------------------------------------------------------------------------- prompt parsing
BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.S)


def prompt_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def packet_from_prompt(messages) -> Tuple[List[Dict[str, Any]], str]:
    t = prompt_text(messages)
    m = BLOCK.search(t)
    if not m:
        return [], ""
    try:
        blk = json.loads(m.group(1))
    except json.JSONDecodeError:
        return [], ""
    if blk and isinstance(blk[0].get("control"), list):
        return blk, "raw"
    return blk, "summary"


# ----------------------------------------------------------------------------- inputs: published arms
R1 = "results/name-attribution-r1"
R2 = "results/name-attribution-r2-hosted"
ARMS = [  # (source label, arm, payload glob relative to the archive, pooled rows, mapping, names column, key column)
    ("local", "arm1", f"{R1}/arm1-permuted/ai-logs/*/*/*.json", f"{R1}/arm1-permuted/pooled/arm1_results_raw_pooled.csv",
     f"{R1}/datasets/arm1_scenario_mapping.csv", "arm1_metric_names", "scenario"),
    ("local", "arm1c", f"{R1}/arm1c-permuted-order-preserved/ai-logs/*/*/*.json",
     f"{R1}/arm1c-permuted-order-preserved/pooled/arm1c_results_raw_pooled.csv",
     f"{R1}/datasets/arm1c_scenario_mapping.csv", "arm1c_metric_names", "scenario"),
    ("local", "arm2", f"{R1}/arm2-anonymised/ai-logs/*/*/*.json", f"{R1}/arm2-anonymised/pooled/arm2_results_raw_pooled.csv",
     f"{R1}/datasets/arm2_scenario_mapping.csv", "anonymised_metric_names", "own_id"),
    ("hosted", "arm1", f"{R2}/arm-arm1/*/ai-logs/*.json", f"{R2}/pooled/hosted_arm1_results_raw_pooled.csv",
     f"{R1}/datasets/arm1_scenario_mapping.csv", "arm1_metric_names", "scenario"),
    ("hosted", "arm1c", f"{R2}/arm-arm1c/*/ai-logs/*.json", f"{R2}/pooled/hosted_arm1c_results_raw_pooled.csv",
     f"{R1}/datasets/arm1c_scenario_mapping.csv", "arm1c_metric_names", "scenario"),
    ("hosted", "arm2", f"{R2}/arm-arm2/*/ai-logs/*.json", f"{R2}/pooled/hosted_arm2_results_raw_pooled.csv",
     f"{R1}/datasets/arm2_scenario_mapping.csv", "anonymised_metric_names", "own_id"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def units_from_archive(arch: Path, mk_fn, diag: Dict[str, Any]) -> List[Dict[str, Any]]:
    series = pc.load_series(arch)
    windows: Dict[str, Window] = {}

    def window(study_id: str) -> Window:
        if study_id not in windows:
            d = series[study_id]
            windows[study_id] = Window(study_id, d["family"], d["label"], d["metrics"], mk_fn)
        return windows[study_id]

    units = []
    for src, arm, pglob, pooled_rel, map_rel, names_col, key_col in ARMS:
        mp = pc.read_csv(arch / map_rel)
        by_names = {}
        for r in mp:
            names = frozenset(r[names_col].split("|"))
            study = r.get("source_scenario") or r["scenario"]
            by_names[names] = (r[key_col], study)
        rows = pc.read_csv(arch / pooled_rel)
        scored = {}
        for r in rows:
            if r["kind"] != "ai" or (r.get("error") or "").strip():
                continue
            scored[(r["model"], r["rep_or_policy"], r["scenario"])] = r
        cand = defaultdict(list)
        n_pay = n_err = n_nomap = 0
        for path in sorted(glob.glob(str(arch / pglob))):
            if "superseded" in path or "00_smoke" in path or "00_documented" in path:
                continue
            rec = json.loads(Path(path).read_text(encoding="utf-8"))
            n_pay += 1
            if (rec.get("error_kind") or "").strip() or not rec.get("parsed_verdict"):
                n_err += 1
                continue
            packet, ptype = packet_from_prompt(rec.get("messages") or [])
            names = frozenset(b.get("name", "") for b in packet)
            if names not in by_names:
                n_nomap += 1
                continue
            key, study = by_names[names]
            rel = Path(path).resolve().relative_to(arch.resolve()).as_posix()
            cand[(rec.get("model"), rec.get("mode"), key)].append((rel, rec, packet, study))
        n_match = n_nopayload = n_multi = 0
        def score_of(pv):
            v = pv.get("overallScore")
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        for (model, mode, key), r in sorted(scored.items()):
            cs = cand.get((model, mode, key), [])
            row_rat = norm(r.get("rationale", ""))
            default = row_rat == f"AI judge {mode}/{model}"  # the harness's default when the completion has no rationale

            def same_rat(c):
                pr = norm(str(c[1]["parsed_verdict"].get("rationale", "") or ""))
                return pr[:len(row_rat)] == row_rat or (default and pr == "")

            def same_score(c):
                sc = score_of(c[1]["parsed_verdict"])
                # a completion without a usable score is scored by the harness at a fallback value (Text S3)
                return sc is None or sc == float(r["score"])
            good = [c for c in cs if same_rat(c) and same_score(c)]
            if not good:
                n_nopayload += 1
                continue
            if len(good) > 1:
                n_multi += 1
            fname, rec, packet, study = sorted(good, key=lambda c: c[0])[-1]
            pv = rec["parsed_verdict"]
            reasons = " ".join(str(m.get("reason", "")) for m in (pv.get("metrics") or []) if isinstance(m, dict))
            units.append({"source": f"{src} {arm}", "arm": arm, "model": model, "representation": mode,
                          "prompt_id": rec.get("prompt_id", ""), "dataset_id": "original-180", "window": study,
                          "row_key": key, "family": series[study]["family"], "truth": series[study]["label"],
                          "verdict": r["verdict"], "score": float(r["score"]), "rationale": str(pv.get("rationale", "")),
                          "metric_reasons": reasons, "packet": packet, "window_obj": window(study),
                          "payload": f"data-archive/{fname}",
                          "repair_turn": len(rec.get("messages") or []) > 2})
            n_match += 1
        diag[f"{src} {arm}"] = {"payload_files": n_pay, "payloads_with_error_or_no_verdict": n_err,
                                "payloads_whose_metric_names_map_to_no_window": n_nomap,
                                "scored_rows": len(scored), "scored_rows_joined_to_a_payload": n_match,
                                "scored_rows_without_payload": n_nopayload,
                                "scored_rows_with_several_matching_payloads_latest_kept": n_multi}
    return units


# ----------------------------------------------------------------------------- inputs: long-format records
def load_series_jsonl(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Tolerant reader: one JSON object per line with an id (id | scenario_id | scenario), an optional dataset id,
    family, truth (truth | truth_label | label) and metrics as a list of {name, kind?, control, experiment|canary} or a
    mapping name -> {control, experiment|canary}."""
    out = {}
    for p in paths:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            wid = o.get("id") or o.get("scenario_id") or o.get("scenario")
            ds = o.get("dataset_id") or o.get("dataset") or ""
            mets = o.get("metrics") or o.get("series")
            ml = []
            if isinstance(mets, dict):
                mets = [{"name": k, **v} for k, v in mets.items()]
            for m in mets or []:
                ml.append({"name": m["name"], "kind": m.get("kind") or kind_of_metric(m["name"]),
                           "control": m["control"], "canary": m.get("experiment", m.get("canary"))})
            out[(ds, wid)] = {"family": o.get("family", ""), "label": o.get("truth") or o.get("truth_label") or o.get("label", ""),
                              "metrics": ml}
    return out


def generate_series(dataset_ids: List[str], master_seed: int, n: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    sys.path.insert(0, str(pc.REPO / "tools"))
    import eval_dataset as ed  # noqa: E402
    out = {}
    for ds in dataset_ids:
        for sc in ed.build_scenarios(master_seed, n, ds):
            ser = ed.series_for_scenario(sc)
            out[(ds, sc.id)] = {"family": sc.family, "label": sc.truth,
                                "metrics": [{"name": m.name, "kind": m.kind, "control": ser[m.name]["control"],
                                             "canary": ser[m.name]["experiment"]} for m in sc.metrics]}
    return out


def units_from_records(paths: List[Path], series: Dict[Tuple[str, str], Dict[str, Any]], mk_fn,
                       diag: Dict[str, Any]) -> List[Dict[str, Any]]:
    rep = pc.representations_module()
    windows = {}
    units = []
    n_rows = n_missing = 0
    for p in paths:
        for r in pc.read_csv(p):
            if r.get("judge") != "ai" or (r.get("error_kind") or "").strip() or r.get("verdict") not in ("PASS", "FAIL"):
                continue
            n_rows += 1
            key = (r.get("dataset_id", ""), r["scenario_id"])
            d = series.get(key) or series.get(("", r["scenario_id"]))
            if d is None:
                n_missing += 1
                continue
            if key not in windows:
                windows[key] = Window(r["scenario_id"], d["family"], d["label"], d["metrics"], mk_fn)
            if r.get("representation") == "raw":
                packet = [{"name": m["name"], "control": [round(v, rep.PRECISION) for v in m["control"]],
                           "experiment": [round(v, rep.PRECISION) for v in m["canary"]]} for m in d["metrics"]]
                packet = [{"name": b["name"], "control": rep.describe(b["control"]), "experiment": rep.describe(b["experiment"])}
                          for b in packet]
            else:
                packet = []
                for m in d["metrics"]:
                    c, e = rep.describe(list(m["control"])), rep.describe(list(m["canary"]))
                    packet.append({"name": m["name"], "control": c, "experiment": e,
                                   "experiment_minus_control_mean": round(e["mean"] - c["mean"], rep.PRECISION),
                                   "experiment_over_control_mean": round(e["mean"] / c["mean"], rep.PRECISION) if c["mean"] else None})
            units.append({"source": f"records {p.name}", "arm": r.get("run_id", ""), "model": r["model"],
                          "representation": r.get("representation", ""), "prompt_id": r.get("prompt_id", ""),
                          "dataset_id": r.get("dataset_id", ""), "window": r["scenario_id"], "row_key": r["scenario_id"],
                          "family": r.get("family", d["family"]), "truth": r.get("truth_label", d["label"]),
                          "verdict": r["verdict"], "score": float(r["score"]) if r.get("score") else None,
                          "rationale": r.get("rationale", ""), "metric_reasons": "", "packet": packet,
                          "window_obj": windows[key], "payload": "", "repair_turn": False})
    diag["records"] = {"scored_ai_rows": n_rows, "rows_without_series": n_missing,
                       "rationales_of_exactly_500_characters": sum(1 for u in units if len(u["rationale"]) == 500)}
    return units


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "d3_rationale_config.json"))
    ap.add_argument("--out", required=True, help="output directory; an existing file is never overwritten")
    ap.add_argument("--archive", help="published data deposit (name-attribution arms)")
    ap.add_argument("--records", nargs="*", default=[], help="long-format record CSV files of new runs")
    ap.add_argument("--series-jsonl", nargs="*", default=[], help="window series files (JSON lines)")
    ap.add_argument("--generate", nargs="*", default=[], help="dataset ids to regenerate with tools/eval_dataset.py")
    ap.add_argument("--master-seed", type=int, default=20260621)
    ap.add_argument("--n-per-family", type=int, default=20)
    ap.add_argument("--spotcheck-name", default="rationale-spotcheck.csv")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)  # existing files are never overwritten (write_* refuse)
    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ex = Extractor(cfg)
    mk = pc.repo_modules()["ensemble_judge"].mann_kendall_p
    diag: Dict[str, Any] = {}
    units: List[Dict[str, Any]] = []
    if a.archive:
        units += units_from_archive(Path(a.archive), mk, diag)
    if a.records:
        ser = {}
        if a.series_jsonl:
            ser.update(load_series_jsonl([Path(p) for p in a.series_jsonl]))
        if a.generate:
            ser.update(generate_series(a.generate, a.master_seed, a.n_per_family))
        units += units_from_records([Path(p) for p in a.records], ser, mk, diag)
    if not units:
        raise SystemExit("no scored judgement with a rationale was found")

    F = pc.Facts(SCRIPT)
    claim_rows, rat_rows = [], []
    cid = 0
    for u in units:
        u["is_placeholder"] = ex.is_placeholder(u["rationale"])
        for unit_name, fields in (("primary", cfg["unit"]["primary_text_fields"]),
                                  ("secondary", cfg["unit"]["secondary_text_fields"])):
            text = " ".join(u[f] for f in fields if u.get(f)) if not (unit_name == "primary" and u["is_placeholder"]) else ""
            if unit_name == "secondary" and u["is_placeholder"]:
                text = u.get("metric_reasons", "")
            claims = ex.extract(text)
            res = []
            for c in claims:
                st, stat, vals = check(c, u["window_obj"], u["packet"], cfg)
                cid += 1
                res.append((c, st))
                claim_rows.append({"claim_id": f"C{cid:06d}", "unit": unit_name, "source": u["source"], "arm": u["arm"],
                                   "model": u["model"], "representation": u["representation"],
                                   "prompt_id": u["prompt_id"], "dataset_id": u["dataset_id"], "window": u["window"],
                                   "family": u["family"], "label": u["truth"], "verdict": u["verdict"],
                                   "score": u["score"], "claim_type": c["type"],
                                   "direction_or_value": (f"{c['value']:g}{c['unit']}" if c["type"] == "number" else c["direction"]),
                                   "negation_detected": "yes" if c.get("negated") else "no",
                                   "metric_named": ",".join(c.get("kinds", [])), "trigger": c["trigger"],
                                   "matched_by": c["how"], "clause": c["clause"], "statistic": stat, "values": vals,
                                   "result": st, "result_trend_sign_only": trend_sign_only(c, u["window_obj"]),
                                   "problem_claim": "yes" if PROBLEM.get(c["type"]) == c["direction"] else "no"})
            checkable = [x for x in res if x[1] in ("consistent", "inconsistent")]
            prob = [x for x in checkable if PROBLEM.get(x[0]["type"]) == x[0]["direction"]]
            status = ""
            if u["verdict"] == "FAIL":
                if not prob:
                    status = "no checkable stated reason"
                elif all(x[1] == "inconsistent" for x in prob):
                    status = "stated reason absent from the data"
                else:
                    status = "stated reason present"
            rat_rows.append({"unit": unit_name, "source": u["source"], "arm": u["arm"], "model": u["model"],
                             "representation": u["representation"], "prompt_id": u["prompt_id"],
                             "dataset_id": u["dataset_id"], "window": u["window"], "family": u["family"],
                             "label": u["truth"], "verdict": u["verdict"], "score": u["score"],
                             "placeholder": "yes" if u["is_placeholder"] else "no",
                             "repair_turn": "yes" if u["repair_turn"] else "no",
                             "n_claims": len(res), "n_checkable": len(checkable),
                             "n_consistent": sum(1 for x in checkable if x[1] == "consistent"),
                             "n_problem_claims": len(prob), "fail_reason_status": status,
                             "window_statistics": u["window_obj"].summary_text(),
                             "rationale": u["rationale"]})

    # ---- summaries
    def summarise(rows_r, rows_c):
        n = len(rows_r)
        wc = sum(1 for r in rows_r if r["n_checkable"] > 0)
        chk = [c for c in rows_c if c["result"] in ("consistent", "inconsistent")]
        cons = sum(1 for c in chk if c["result"] == "consistent")
        fails = [r for r in rows_r if r["verdict"] == "FAIL"]
        fp = [r for r in fails if r["fail_reason_status"] in ("stated reason absent from the data", "stated reason present")]
        absent = sum(1 for r in fails if r["fail_reason_status"] == "stated reason absent from the data")
        out = {"n_rationales": n, "n_placeholder": sum(1 for r in rows_r if r["placeholder"] == "yes"),
               "with_checkable_claim": wc, "share_with_checkable_claim": f"{wc / n:.3f}" if n else "",
               "checkable_claims": len(chk), "consistent_claims": cons,
               "share_consistent": f"{cons / len(chk):.3f}" if chk else "",
               "fail_verdicts": len(fails), "fail_with_checkable_stated_reason": len(fp),
               "fail_reason_absent": absent,
               "share_fail_reason_absent_among_checkable": f"{absent / len(fp):.3f}" if fp else "",
               "share_fail_reason_absent_among_all_fail": f"{absent / len(fails):.3f}" if fails else ""}
        for t in TYPES:
            ct = [c for c in chk if c["claim_type"] == t]
            out[f"{t}_checkable"] = len(ct)
            out[f"{t}_consistent"] = sum(1 for c in ct if c["result"] == "consistent")
        tr = [c for c in chk if c["claim_type"] == "trend"]
        out["trend_consistent_sign_only"] = sum(1 for c in tr if (c["result_trend_sign_only"] or c["result"]) == "consistent")
        out["share_consistent_trend_sign_only"] = (
            f"{(cons - out['trend_consistent'] + out['trend_consistent_sign_only']) / len(chk):.3f}" if chk else "")
        neg = [c for c in chk if c["negation_detected"] == "yes"]
        out["checkable_claims_with_negation"] = len(neg)
        return out

    summary_rows = []
    for unit_name in ("primary", "secondary"):
        R = [r for r in rat_rows if r["unit"] == unit_name]
        Cc = [c for c in claim_rows if c["unit"] == unit_name]
        groups = defaultdict(lambda: ([], []))
        for r in R:
            for key in (("by configuration", r["model"], r["representation"], r["prompt_id"], "all arms"),
                        ("by configuration and arm", r["model"], r["representation"], r["prompt_id"], r["source"]),
                        ("all configurations", "all", "", "", "all arms")):
                groups[key][0].append(r)
        for c in Cc:
            for key in (("by configuration", c["model"], c["representation"], c["prompt_id"], "all arms"),
                        ("by configuration and arm", c["model"], c["representation"], c["prompt_id"], c["source"]),
                        ("all configurations", "all", "", "", "all arms")):
                groups[key][1].append(c)
        for key in sorted(groups, key=lambda k: (["all configurations", "by configuration", "by configuration and arm"].index(k[0]), k[1:])):
            s = summarise(*groups[key])
            summary_rows.append({"unit": unit_name, "level": key[0], "model": key[1], "representation": key[2],
                                 "prompt_id": key[3], "arms": key[4], **s})
            if unit_name == "primary" and key[0] in ("by configuration", "all configurations"):
                lab = "all configurations pooled" if key[0] == "all configurations" else f"{pc.PAPER_MODEL.get(key[1], key[1])} ({key[2]})"
                tag = "all" if key[0] == "all configurations" else f"{key[1]}.{key[2]}.{key[3]}"
                F.add(f"d3.{tag}",
                      f"rationales {s['n_rationales']}; with a checkable claim {s['with_checkable_claim']} "
                      f"({s['share_with_checkable_claim']}); checkable claims {s['checkable_claims']}, consistent "
                      f"{s['consistent_claims']} ({s['share_consistent']}); FAIL verdicts {s['fail_verdicts']}, with a "
                      f"checkable stated reason {s['fail_with_checkable_stated_reason']}, reason absent from the data "
                      f"{s['fail_reason_absent']} ({s['share_fail_reason_absent_among_checkable']})",
                      f"D3 rationale check, {lab}, rubric {key[3] or 'n/a'}, primary unit (rationale field); full payload "
                      f"records of the name-attribution arms" if a.archive else f"D3 rationale check, {lab}, records",
                      "data-archive/results/name-attribution-r1; data-archive/results/name-attribution-r2-hosted" if a.archive else "records",
                      "d3_rationale_config.json patterns and tolerances; claims checked against the window series and the prompt's summary packet")
    # ---- spot check
    sc = cfg["spot_check"]
    pool = [c for c in claim_rows if c["unit"] == "primary" and c["result"] in ("consistent", "inconsistent")]
    rng = random.Random(int(sc["seed"]))
    pick = sorted(rng.sample(range(len(pool)), min(int(sc["n"]), len(pool))))
    rat_by = {(r["source"], r["model"], r["representation"], r["window"]): r for r in rat_rows if r["unit"] == "primary"}
    spot = []
    for i, k in enumerate(pick, 1):
        c = pool[k]
        r = rat_by[(c["source"], c["model"], c["representation"], c["window"])]
        spot.append({"sample_no": i, "claim_id": c["claim_id"], "source": c["source"], "model": c["model"],
                     "representation": c["representation"], "window": c["window"], "family": c["family"],
                     "label": c["label"], "verdict": c["verdict"], "score": c["score"], "claim_type": c["claim_type"],
                     "extracted_direction_or_value": c["direction_or_value"], "negation_detected": c["negation_detected"],
                     "metric_named": c["metric_named"], "trigger": c["trigger"], "matched_by": c["matched_by"],
                     "sentence": c["clause"], "full_rationale": r["rationale"], "statistic_checked": c["statistic"],
                     "statistic_values": c["values"], "window_statistics": r["window_statistics"],
                     "automatic_result": c["result"],
                     "author_extraction_correct (yes/no)": "", "author_consistent_with_window (yes/no)": "",
                     "author_note": ""})
    pc.write_csv(out / "d3_claims.csv", claim_rows)
    pc.write_csv(out / "d3_rationales.csv", rat_rows)
    pc.write_csv(out / "d3_summary.csv", summary_rows)
    pc.write_csv(out / a.spotcheck_name, spot)
    F.add("d3.config", pc.sha256_file(cfg_path), "SHA-256 of the D3 configuration file (claim patterns and tolerances)",
          "analysis/r1/posthoc/d3_rationale_config.json", "sha256 of the file bytes")
    F.add("d3.spotcheck", f"{len(spot)} claims, seed {sc['seed']}, from {len(pool)} checkable claims of the primary unit",
          "random sample of extracted claims written for an author's hand check", f"{a.spotcheck_name}",
          "random.Random(seed).sample over the checkable claims in file order")
    pc.write_json(out / "d3_facts.json", F.items)
    pc.write_json(out / "d3_run.json", {"script": SCRIPT, "config": {"file": cfg_path.name, "sha256": pc.sha256_file(cfg_path),
                                                                     "id": cfg["id"]},
                                        "code_sha256": {**({} if not a.archive else pc.code_hashes(Path(a.archive)))},
                                        "inputs": {"archive": bool(a.archive), "records": [Path(p).name for p in a.records],
                                                   "series_jsonl": [Path(p).name for p in a.series_jsonl],
                                                   "generate": a.generate},
                                        "join_diagnostics": diag, "n_units": len(units), "n_claims": len(claim_rows),
                                        "python": sys.version.split()[0], "numpy": np.__version__})
    print(json.dumps({"units": len(units), "claims": len(claim_rows), "join": diag}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
