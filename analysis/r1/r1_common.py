"""Shared code of the analysis of the first-revision runs (protocol/r1-controls-protocol.json).

Everything here reads files; nothing calls a model, Kayenta or the judge service.

Statistics are computed with the code that produced the numbers of the submitted
version, not with a rewrite of it:

* Wilson 95% intervals: tools/ci_utils.py wilson_interval (z = 1.959963985), as
  the protocol's analysis plan fixes;
* the family-stratified percentile bootstrap, the confusion coding, Newcombe's
  method 10, the exact McNemar test and the exact Holm adjustment: the functions
  code, metrics, pct, strata, boot_config, wilson, newcombe10, p_mcnemar_exact and
  holm_exact of analysis/area-A-headline/analyze_A.py of the result data deposit
  (version 1.1.1), the script Table S42 of the Supplementary Materials names.
  They are loaded from that file function by function (the loader of
  analysis/r1/posthoc/posthoc_common.py), so the archived script's start-up code,
  which creates directories inside the deposit, never runs. The bootstrap stream
  is the one the protocol fixes: random.Random(20261122) for every configuration,
  2,000 replicates (the archived boot_config seeds from a hash of the label; the
  protocol replaces that seed and nothing else).

Numbers are printed as the archived area-A code prints them (f"{x:.3f}" of the floating-point value), so that a
value that sits exactly on a half rounds as in Tables 6 and 8 (Phi-4 on summaries, BA 0.7625, printed 0.762;
gpt-oss-120b on raw series, BA 0.7125, printed 0.713). The exact rational BA is kept beside it (BA_exact).

Paths written into outputs are relative to the authors' working folder (its prefix is explained in the README of
Archive S2), to the data deposit ("data-archive/...") or to the repository root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import zlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent            # analysis/r1
REPO = HERE.parents[1]                             # repository root
POSTHOC = HERE / "posthoc"
PROTOCOL_REL = "protocol/r1-controls-protocol.json"
MASTER_SEED = 20260621
N_PER_FAMILY = 20
BOOT_SEED = 20261122
BOOT_B = 2000

if str(POSTHOC) not in sys.path:
    sys.path.insert(0, str(POSTHOC))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import posthoc_common as pc  # noqa: E402  (the post-hoc loader, used read-only)
import ci_utils  # noqa: E402

HOSTED = ["claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm"]
LOCAL_TEXT = ["qwen-llm", "phi4-llm", "olmo2-llm", "mistral-nemo-llm", "deepseek-r1-llm"]
LOCAL_VISION = ["moondream-vlm", "granite-vision-vlm", "minicpm-v-vlm"]
LOCAL = LOCAL_TEXT + LOCAL_VISION
MODELS = HOSTED + LOCAL
REPS = {"claude-opus-4-8-vlm": ["summary", "raw", "plot"], "gpt-oss-120b-llm": ["summary", "raw"],
        "qwen3-vl-235b-vlm": ["summary", "raw", "plot"],
        **{m: ["summary", "raw"] for m in LOCAL_TEXT}, **{m: ["plot"] for m in LOCAL_VISION}}
PRIMARY = {"claude-opus-4-8-vlm": "plot", "gpt-oss-120b-llm": "summary", "qwen3-vl-235b-vlm": "plot",
           **{m: "summary" for m in LOCAL_TEXT}, **{m: "plot" for m in LOCAL_VISION}}
PAPER_MODEL = {"claude-opus-4-8-vlm": "Claude Opus 4.8", "gpt-oss-120b-llm": "gpt-oss-120b",
               "qwen3-vl-235b-vlm": "Qwen3-VL 235B", "qwen-llm": "Qwen2.5", "phi4-llm": "Phi-4",
               "olmo2-llm": "OLMo 2", "mistral-nemo-llm": "Mistral-Nemo", "deepseek-r1-llm": "DeepSeek-R1",
               "moondream-vlm": "Moondream", "granite-vision-vlm": "Granite Vision", "minicpm-v-vlm": "MiniCPM-V"}
REP_PLURAL = {"summary": "summaries", "raw": "raw series", "plot": "plots"}
POLICIES = ["gated", "or", "and"]
POLICY_NAME = {"gated": "hybrid as scored", "or": "fail-if-either hybrid", "and": "both-must-fail hybrid"}
RUBRICS = ["v1-frozen-2026-06", "v2-operational-2026-07", "v3-temporal-2026-07"]
RUBRIC_SHORT = {"v1-frozen-2026-06": "v1", "v2-operational-2026-07": "v2", "v3-temporal-2026-07": "v3"}
V1 = RUBRICS[0]

# Row order of Table 9 (the table template), model configurations by model then representation.
TABLE_MODEL_ORDER = (["claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm", "phi4-llm", "mistral-nemo-llm",
                      "qwen-llm", "deepseek-r1-llm", "olmo2-llm", "granite-vision-vlm", "minicpm-v-vlm",
                      "moondream-vlm"])
MODEL_CONFIGS = [(m, r) for m in TABLE_MODEL_ORDER for r in REPS[m]]   # 21
HYBRID_CONFIGS = [(m, p) for m in TABLE_MODEL_ORDER for p in POLICIES]  # 33

KC = "kind-controls-180"
HB = "heldout-benign-100"
O180 = "original-180"
G60 = "generalisation-60"

KC_FAMILIES = ["kc_latency_nochange", "kc_memory_nochange", "kc_errors_nochange", "kc_tri_nochange",
               "kc_memory_shared_ramp", "kc_latency_drift", "kc_memory_drift", "kc_errors_drift", "kc_tri_marginal"]
KC_CODE = dict(zip(KC_FAMILIES, ["LN", "MN", "EN", "TN", "MR", "LD", "MD", "ED", "TM"]))
KC_BENIGN = KC_FAMILIES[:5]
KC_REGRESSION = KC_FAMILIES[5:]
KC_DRIFT = ["kc_latency_drift", "kc_memory_drift", "kc_errors_drift"]
KC_KIND_GROUPS = {"latency": ["kc_latency_nochange"], "memory": ["kc_memory_nochange", "kc_memory_shared_ramp"],
                  "errors": ["kc_errors_nochange"], "three-metric": ["kc_tri_nochange"]}
HB_FAMILIES = ["hb_shared_partial_recovery", "hb_shared_late_step", "hb_isolated_spikes_both",
               "hb_recovered_mid_transient", "hb_brief_early_blip"]
HB_CODE = dict(zip(HB_FAMILIES, ["SP", "SL", "IS", "RM", "BB"]))
HB_EXCH = HB_FAMILIES[:3]
HB_RT = HB_FAMILIES[3:]
G60_FAMILIES = ["partial_recovery", "late_transient", "sustained_excursion"]
G60_CODE = dict(zip(G60_FAMILIES, ["PR", "LT", "SE"]))
O180_FAMILIES = ["no_change", "noise_equivalent", "healed_transient", "clean_mean_shift", "subtle_regression",
                 "variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]

RECORD_COLUMNS = ["run_id", "ts", "dataset_id", "prompt_id", "prompt_hash", "model", "judge", "representation",
                  "family", "scenario_id", "seed", "truth_label", "verdict", "score", "correct", "latency_s",
                  "tokens_in", "tokens_out", "finish_reason", "error_kind", "error", "rationale"]


# ----------------------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    return pc.sha256_file(path)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    """Write a CSV; refuses to overwrite an existing file (results are never overwritten)."""
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}: write to a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}: write to a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def rel_to(path: Path, base: Path, prefix: str) -> str:
    """A relative path for outputs: prefix + path relative to base."""
    try:
        return prefix + str(Path(path).resolve().relative_to(Path(base).resolve())).replace("\\", "/")
    except ValueError:
        raise SystemExit(f"{path} is not under the base folder; refusing to write an absolute path")


# ----------------------------------------------------------------------------- number formatting
MINUS = "−"


def ba_float(tp: int, fp: int, tn: int, fn: int) -> Optional[float]:
    """BA as analyze_A.metrics computes it: (recall + specificity) / 2 in floating point."""
    if tp + fn == 0 or tn + fp == 0:
        return None
    return (tp / (tp + fn) + tn / (tn + fp)) / 2


def f3(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.3f}"


def typo(s: str) -> str:
    """Typographic minus for tables the writers paste (Table 6 prints U+2212)."""
    return s.replace("-", MINUS)


def signed3(x: float) -> str:
    s = f"{x:+.3f}"
    return "+0.000" if s in ("-0.000", "+0.000") else s


def pfmt(p: Fraction) -> str:
    """p-values as the Supplementary Materials print them (Table S31): 4 decimals, or 3 significant digits
    in scientific notation below 0.001."""
    v = float(p)
    if v >= 0.001:
        return f"{v:.4f}"
    m, e = f"{v:.2e}".split("e")
    return f"{m} × 10{MINUS}{int(e[1:])}" if e.startswith("-") else f"{m} × 10{int(e)}"


def wilson(k: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return (None, None)
    return ci_utils.wilson_interval(k, n)


def kn_ci(k: int, n: int) -> str:
    if n <= 0:
        return "no data"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.3f}, {hi:.3f}]"


# ----------------------------------------------------------------------------- archived statistics code
def archived_stats(deposit: Path) -> Dict[str, Any]:
    """Functions of analysis/area-A-headline/analyze_A.py (data deposit 1.1.1), loaded read only.

    seed_for is replaced by the protocol's fixed stream: every boot_config call draws from
    random.Random(20261122)."""
    env = {"random": random, "zlib": zlib, "math": math, "defaultdict": defaultdict, "Fraction": Fraction,
           "B": BOOT_B, "SEED": 20260912, "Z": 1.959963984540054,
           "seed_for": lambda *parts: BOOT_SEED}
    pc.load_functions(deposit / "analysis" / "area-A-headline" / "analyze_A.py",
                      ["code", "metrics", "wilson", "pct", "strata", "boot_config", "newcombe10",
                       "p_mcnemar_exact", "holm_exact"], env)
    return env


# ----------------------------------------------------------------------------- confusion and summaries
def confusion(verdicts: Dict[str, str], truth: Dict[str, str], ids: Optional[Iterable[str]] = None):
    tp = fp = tn = fn = 0
    for s in (ids if ids is not None else verdicts.keys()):
        if s not in verdicts:
            continue
        t, p = truth[s] == "FAIL", verdicts[s] == "FAIL"
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif not t:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def ba_exact(tp: int, fp: int, tn: int, fn: int) -> Optional[Fraction]:
    if tp + fn == 0 or tn + fp == 0:
        return None
    return (Fraction(tp, tp + fn) + Fraction(tn, tn + fp)) / 2


def mcc_value(tp: int, fp: int, tn: int, fn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def constant_within_families(verdicts: Dict[str, str], fam_of: Dict[str, str]) -> bool:
    seen: Dict[str, set] = defaultdict(set)
    for s, v in verdicts.items():
        seen[fam_of[s]].add(v)
    return all(len(v) == 1 for v in seen.values())


def ba_mcc_with_ci(stats: Dict[str, Any], verdicts: Dict[str, str], truth: Dict[str, str],
                   fam_of: Dict[str, str], label: str) -> Dict[str, Any]:
    """Point BA and MCC and the protocol's percentile bootstrap (analyze_A.boot_config, seed 20261122)."""
    tp, fp, tn, fn = confusion(verdicts, truth)
    pos, neg = tp + fn, tn + fp
    ba = ba_exact(tp, fp, tn, fn)
    mcc = mcc_value(tp, fp, tn, fn)
    codes = {s: stats["code"](truth[s], v) for s, v in verdicts.items()}
    const = constant_within_families(verdicts, fam_of)
    out = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "n": tp + fp + tn + fn,
           "BA_exact": f"{ba.numerator}/{ba.denominator}" if ba is not None else "",
           "BA": f3(ba_float(tp, fp, tn, fn)), "MCC_value": mcc, "MCC": f3(mcc),
           "constant_within_families": const,
           "families_resampled": len({fam_of[s] for s in verdicts})}
    if codes and pos and neg:
        bc = stats["boot_config"](codes, fam_of, label)
        blo, bhi = bc["ba"]
        mlo, mhi = bc["mcc"]
        out.update({"BA_lo": f3(blo), "BA_hi": f3(bhi),
                    "MCC_lo": f3(mlo), "MCC_hi": f3(mhi), "MCC_lo_value": mlo, "MCC_hi_value": mhi})
    else:
        out.update({"BA_lo": "", "BA_hi": "", "MCC_lo": "", "MCC_hi": ""})
    if const:
        out["BA_ci"] = "[N/A]"
        out["MCC_ci"] = "[N/A]"
    else:
        out["BA_ci"] = f"[{out['BA_lo']}, {out['BA_hi']}]"
        out["MCC_ci"] = f"[{out['MCC_lo']}, {out['MCC_hi']}]"
    return out


# ----------------------------------------------------------------------------- protocol, series, labels
def load_protocol() -> Dict[str, Any]:
    return json.loads((REPO / PROTOCOL_REL).read_text(encoding="utf-8"))


def invocation_table(protocol: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["invocation_id"]: r for r in protocol["run_table"]}


def load_series_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    """Protocol series file -> scenario id -> {family, truth, metrics: [{name, kind, unit, control, canary}], params}."""
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        o = json.loads(ln)
        out[o["id"]] = {"family": o["family"], "truth": o["truth"], "gid": o["gid"], "seed": o["seed"],
                        "params": o.get("params", {}),
                        "metrics": [{"name": m["name"], "kind": m["kind"], "unit": m["unit"],
                                     "control": list(m["control"]), "canary": list(m["experiment"])}
                                    for m in o["metrics"]]}
    return out


def generate_series(dataset_id: str) -> Dict[str, Dict[str, Any]]:
    """original-180 and generalisation-60 from tools/eval_dataset.py at master seed 20260621 (20 per family)."""
    import eval_dataset as ed  # noqa: E402
    out = {}
    for sc in ed.build_scenarios(MASTER_SEED, N_PER_FAMILY, dataset_id):
        ser = ed.series_for_scenario(sc)
        out[sc.id] = {"family": sc.family, "truth": sc.truth, "gid": getattr(sc, "gid", None), "seed": sc.seed,
                      "params": sc.params,
                      "metrics": [{"name": m.name, "kind": m.kind, "unit": m.unit,
                                   "control": list(ser[m.name]["control"]),
                                   "canary": list(ser[m.name]["experiment"])} for m in sc.metrics]}
    return out


def load_all_series() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {KC: load_series_jsonl(REPO / "protocol" / f"series_{KC}.jsonl"),
            HB: load_series_jsonl(REPO / "protocol" / f"series_{HB}.jsonl"),
            O180: generate_series(O180), G60: generate_series(G60)}


# ----------------------------------------------------------------------------- run records
class Records:
    """The long-format records of the runs (runs/records/<invocation>/*.csv).

    One record per judgement key is the successful call; rows of later attempts that repeat a key are cache hits
    and must carry the same verdict and score (checked). Hybrid and statistical rows carry no prompt id, so the
    rubric of a row is the rubric of its invocation in the protocol's run table."""

    def __init__(self, records_dir: Path, protocol: Dict[str, Any], base: Path):
        self.dir = records_dir
        self.inv = invocation_table(protocol)
        self.files: List[Tuple[str, Path]] = []
        self.rows: List[Dict[str, str]] = []
        for d in sorted(p for p in records_dir.iterdir() if p.is_dir()):
            if d.name not in self.inv:
                raise SystemExit(f"records folder {d.name} is not an invocation of the protocol's run table")
            for f in sorted(d.glob("*.csv")):
                self.files.append((d.name, f))
                with open(f, newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh)
                    if rd.fieldnames != RECORD_COLUMNS:
                        raise SystemExit(f"{f.name}: unexpected columns {rd.fieldnames}")
                    for r in rd:
                        r = dict(r)
                        r["_inv"] = d.name
                        r["_file"] = rel_to(f, base, "")
                        r["_rubric"] = self.inv[d.name]["rubric"]
                        r["_run"] = self.inv[d.name]["run"]
                        self.rows.append(r)
        self.stat: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
        self.ai: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
        self.hyb: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
        for r in self.rows:
            if r["judge"] == "statistical":
                self.stat[(r["dataset_id"], r["scenario_id"])].append(r)
            elif r["judge"] == "ai":
                if r["prompt_id"] != r["_rubric"]:
                    raise SystemExit(f"{r['_file']}: prompt id {r['prompt_id']} differs from the invocation's rubric")
                self.ai[(r["dataset_id"], r["scenario_id"], r["model"], r["representation"], r["_rubric"])].append(r)
            elif r["judge"] == "hybrid":
                self.hyb[(r["dataset_id"], r["scenario_id"], r["model"], r["representation"], r["_rubric"])].append(r)
            else:
                raise SystemExit(f"{r['_file']}: unknown judge {r['judge']}")
        self.cache_differences: List[Dict[str, str]] = []
        self.ai_ok: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
        self.ai_failed: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = {}
        for k, rows in self.ai.items():
            ok = sorted([r for r in rows if not r["error_kind"].strip()], key=lambda r: r["ts"])
            if ok:
                first = ok[0]
                for r in ok[1:]:
                    if (r["verdict"], r["score"]) != (first["verdict"], first["score"]):
                        self.cache_differences.append({"key": "|".join(k), "first": f"{first['verdict']} {first['score']}",
                                                       "later": f"{r['verdict']} {r['score']}", "file": r["_file"]})
                self.ai_ok[k] = first
            else:
                self.ai_failed[k] = rows
        self.hyb_ok: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
        for k, rows in self.hyb.items():
            ok = sorted([r for r in rows if not r["error_kind"].strip()], key=lambda r: r["ts"])
            if ok:
                first = ok[0]
                for r in ok[1:]:
                    if (r["verdict"], r["score"]) != (first["verdict"], first["score"]):
                        self.cache_differences.append({"key": "hybrid|" + "|".join(k), "first": f"{first['verdict']} {first['score']}",
                                                       "later": f"{r['verdict']} {r['score']}", "file": r["_file"]})
                self.hyb_ok[k] = first

    # --- statistical verdicts
    def stat_consistency(self) -> List[Dict[str, Any]]:
        """One row per window: number of invocations, distinct (verdict, score) pairs, error rows."""
        out = []
        for (ds, s), rows in sorted(self.stat.items()):
            pairs = Counter((r["verdict"], r["score"]) for r in rows if not r["error_kind"].strip())
            errs = sum(1 for r in rows if r["error_kind"].strip())
            out.append({"dataset_id": ds, "scenario_id": s, "invocations": len(rows), "error_rows": errs,
                        "distinct_verdict_score": len(pairs),
                        "verdicts": "; ".join(f"{v} {sc} x{n}" for (v, sc), n in sorted(pairs.items()))})
        return out

    def stat_verdicts(self, ds: str) -> Dict[str, str]:
        out = {}
        for (d, s), rows in self.stat.items():
            if d != ds:
                continue
            vs = {r["verdict"] for r in rows if not r["error_kind"].strip()}
            if len(vs) != 1:
                raise SystemExit(f"statistical verdict of {ds} {s} differs between invocations: {vs}")
            out[s] = vs.pop()
        return out

    def stat_scores(self, ds: str) -> Dict[str, float]:
        out = {}
        for (d, s), rows in self.stat.items():
            if d == ds:
                out[s] = float(rows[0]["score"])
        return out

    # --- model and hybrid configurations
    def ai_verdicts(self, ds: str, model: str, rep: str, rubric: str = V1) -> Dict[str, str]:
        return {k[1]: r["verdict"] for k, r in self.ai_ok.items()
                if k[0] == ds and k[2] == model and k[3] == rep and k[4] == rubric}

    def ai_scores(self, ds: str, model: str, rep: str, rubric: str = V1) -> Dict[str, float]:
        return {k[1]: float(r["score"]) for k, r in self.ai_ok.items()
                if k[0] == ds and k[2] == model and k[3] == rep and k[4] == rubric}

    def ai_failed_ids(self, ds: str, model: str, rep: str, rubric: str = V1) -> List[str]:
        return sorted(k[1] for k in self.ai_failed if k[0] == ds and k[2] == model and k[3] == rep and k[4] == rubric)

    def hyb_verdicts(self, ds: str, model: str, policy: str, rubric: str = V1) -> Dict[str, str]:
        return {k[1]: r["verdict"] for k, r in self.hyb_ok.items()
                if k[0] == ds and k[2] == model and k[3] == policy and k[4] == rubric}

    def hyb_scores(self, ds: str, model: str, policy: str, rubric: str = V1) -> Dict[str, float]:
        return {k[1]: float(r["score"]) for k, r in self.hyb_ok.items()
                if k[0] == ds and k[2] == model and k[3] == policy and k[4] == rubric}

    def source_files(self, ds: str, model: Optional[str] = None, rubric: Optional[str] = None) -> str:
        fs = sorted({r["_file"] for r in self.rows if r["dataset_id"] == ds
                     and (model is None or r["model"] == model) and (rubric is None or r["_rubric"] == rubric)})
        return "; ".join(fs)


def config_label(model: str, rep_or_policy: str, hybrid: bool = False) -> str:
    if hybrid:
        return f"{POLICY_NAME[rep_or_policy].capitalize()}, {PAPER_MODEL[model]} ({PRIMARY[model]})"
    return f"{PAPER_MODEL[model]} ({rep_or_policy})"


def config_label_s(model: str, rep: str) -> str:
    """Supplementary style: 'Claude Opus 4.8, raw series'."""
    return f"{PAPER_MODEL[model]}, {REP_PLURAL[rep]}"
