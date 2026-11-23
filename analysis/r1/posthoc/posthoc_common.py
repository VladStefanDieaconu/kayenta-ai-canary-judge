"""Shared inputs of the post-hoc analyses of the first revision (D1, D2 and D3).

Everything here reads the published result data deposit (version 1.1.1) and
Archive S1 of the submission read only. Nothing calls a model, a network service
or the testbed stack, and nothing is written outside the output directory the
calling script is given.

The statistics are computed with the code that produced the submitted numbers,
not with a rewrite of it:

* the bootstrap, metric, Wilson, ROC and average-precision functions of the
  archived analysis scripts named in Table S42 (analysis/area-A-headline,
  area-B-representation-cost, area-C-frontier-ablation-heldout and
  area-D-ensemble-guards of the data deposit) are loaded from their source
  files function by function, without running the scripts, so that the
  scripts' own start-up code (which creates directories) never executes;
* the re-implementation of Kayenta's Mann-Whitney classifier is the archived
  analysis/area-F-instrument-audit/inputs/kmw_lib.py, loaded the way the
  archived gate_sweep.py loads it;
* the series guards, the Mann-Kendall test and the summary-packet statistics
  are those of this repository's judge service (judge-service/app), and the
  Wilson interval and Holm correction those of tools/ci_utils.py.

Paths written into any output are relative: "data-archive/<path>" for the
deposit, "archive-s1/<file>" for Archive S1 and repository-relative paths for
this repository.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # analysis/r1/posthoc -> repository root
JUDGE_APP = REPO / "judge-service" / "app"

# Family order of Tables 1 and 6 of the manuscript.
FAMILIES = ["no_change", "noise_equivalent", "healed_transient", "clean_mean_shift", "subtle_regression",
            "variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]
ABBR = {"no_change": "NC", "noise_equivalent": "NE", "healed_transient": "HT", "clean_mean_shift": "CMS",
        "subtle_regression": "SR", "variance_increase": "VI", "tail_regression": "TR", "gradual_drift": "GD",
        "cross_metric_marginal": "CMM"}
BENIGN_FAMILIES = ["no_change", "noise_equivalent", "healed_transient"]
REGRESSION_FAMILIES = [f for f in FAMILIES if f not in BENIGN_FAMILIES]
PROBE_FAMILIES = ["variance_increase", "tail_regression", "gradual_drift", "cross_metric_marginal"]

SERIES_REL = "results/name-attribution-r1/datasets/arm2_own_data_anonymised_180.csv"
CORRECTED_REL = "results/corrected/results_raw.csv"
ENSEMBLE_REL = "analysis/area-D-ensemble-guards/ensemble_verdicts_180.csv"
GUARD_FAMILY_REL = "results/corrected/agg/fp_guard_family_matrix.csv"
GUARD_MODELS_REL = "analysis/area-D-ensemble-guards/out/guard_models_180.csv"
KMW_REL = "analysis/area-F-instrument-audit/inputs/kmw_lib.py"
AREA_A_REL = "analysis/area-A-headline/analyze_A.py"
AREA_B_REL = "analysis/area-B-representation-cost/analyze_B.py"
AREA_C_REL = "analysis/area-C-frontier-ablation-heldout/analyze_C.py"
AREA_D_REL = "analysis/area-D-ensemble-guards/analyze_D.py"
BFKS_FILE = "classical_verdicts.csv"

HOSTED = ["claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm"]
LOCAL_TEXT = ["phi4-llm", "mistral-nemo-llm", "qwen-llm", "deepseek-r1-llm", "olmo2-llm"]
LOCAL_VISION = ["granite-vision-vlm", "minicpm-v-vlm", "moondream-vlm"]
PAPER_MODEL = {"deepseek-r1-llm": "DeepSeek-R1", "mistral-nemo-llm": "Mistral-Nemo", "olmo2-llm": "OLMo 2",
               "phi4-llm": "Phi-4", "qwen-llm": "Qwen2.5", "granite-vision-vlm": "Granite Vision",
               "minicpm-v-vlm": "MiniCPM-V", "moondream-vlm": "Moondream",
               "claude-opus-4-8-vlm": "Claude Opus 4.8", "gpt-oss-120b-llm": "gpt-oss-120b",
               "qwen3-vl-235b-vlm": "Qwen3-VL 235B"}
PRIMARY = {**{m: "summary" for m in LOCAL_TEXT}, **{m: "plot" for m in LOCAL_VISION},
           "claude-opus-4-8-vlm": "plot", "gpt-oss-120b-llm": "summary", "qwen3-vl-235b-vlm": "plot"}


# ----------------------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    """Write a CSV; refuses to overwrite (results are never overwritten)."""
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.name}: write to a new output directory")
    cols = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_text(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.name}: write to a new output directory")
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n")


def frac_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def f3(x: Optional[float]) -> str:
    return "undefined" if x is None else f"{x:.3f}"


def r3_exact(x: Optional[Fraction]) -> str:
    """Three decimals of an exact rational, rounded half to even (the rounding of every published rate, BA and
    accuracy that sits exactly on a half, e.g. 0.7625 -> 0.762 and 0.7875 -> 0.788)."""
    if x is None:
        return "undefined"
    from decimal import ROUND_HALF_EVEN, Decimal
    d = Decimal(x.numerator) / Decimal(x.denominator)
    return str(d.quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


class XFloat(float):
    """A float that remembers its exact rational value and prints three decimals from it (half to even).
    Arithmetic returns plain floats."""

    def __new__(cls, frac: Fraction):
        obj = float.__new__(cls, float(frac))
        obj.frac = frac
        return obj

    def __format__(self, spec: str) -> str:
        if spec == ".3f":
            return r3_exact(self.frac)
        return float.__format__(float(self), spec)


def exact_rates(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Optional[Fraction]]:
    """Recall, FPR, specificity, BA, accuracy and F1 as exact fractions (None when undefined)."""
    rec = Fraction(tp, tp + fn) if tp + fn else None
    spec = Fraction(tn, tn + fp) if tn + fp else None
    fpr = Fraction(fp, fp + tn) if fp + tn else None
    ba = (rec + spec) / 2 if rec is not None and spec is not None else None
    n = tp + fp + tn + fn
    acc = Fraction(tp + tn, n) if n else None
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"rec": rec, "spec": spec, "fpr": fpr, "ba": ba, "acc": acc, "f1": f1}


def ci3(lo: Optional[float], hi: Optional[float]) -> str:
    return "[undefined]" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"


class Facts:
    """Facts of one script: key, value, description, source, filter or computation, script."""

    def __init__(self, script: str):
        self.script = script
        self.items: List[Dict[str, str]] = []
        self._keys = set()

    def add(self, key: str, value: Any, description: str, source_file: str, computation: str) -> None:
        if key in self._keys:
            raise SystemExit(f"duplicate fact key {key}")
        self._keys.add(key)
        self.items.append({"key": key, "value": str(value), "description": description,
                           "source_file": source_file, "filter_or_computation": computation,
                           "script": self.script})


# ----------------------------------------------------------------------------- archived code, read only
def load_functions(path: Path, names: Sequence[str], env: Dict[str, Any]) -> Dict[str, Any]:
    """Execute only the named top-level function definitions of a source file in `env`.

    The archived analysis scripts create directories and read environment
    variables at import time; this loader compiles the selected `def` nodes and
    nothing else, so none of that runs and the archive stays untouched."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    picked = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = [n for n in names if n not in picked]
    if missing:
        raise SystemExit(f"{path.name}: function(s) not found: {missing}")
    mod = ast.Module(body=[picked[n] for n in names], type_ignores=[])
    exec(compile(mod, filename=str(path), mode="exec"), env)
    return env


def area_a(arch: Path) -> Dict[str, Any]:
    env = {"random": random, "zlib": zlib, "math": math, "defaultdict": defaultdict,
           "B": 2000, "SEED": 20260912, "Z": 1.959963984540054}
    return load_functions(arch / AREA_A_REL, ["code", "metrics", "wilson", "seed_for", "pct", "strata",
                                             "boot_config", "boot_paired"], env)


def area_b(arch: Path) -> Dict[str, Any]:
    env = {"Counter": Counter, "math": math}
    return load_functions(arch / AREA_B_REL, ["dg_segment", "roc_pr", "envelope"], {**env, "Fraction": Fraction})


def area_c(arch: Path, fams: List[str]) -> Dict[str, Any]:
    env = {"math": math, "defaultdict": defaultdict, "_rnd": random, "BOOT_B": 2000, "SEED_V": 20260913,
           "FAMS": list(fams), "GFAMS": [], "_FSB_DESIGN": {}, "_FSB_K": {}}
    return load_functions(arch / AREA_C_REL, ["nearest_rank", "mcc_val", "fsb_design", "strata_v", "point_counts",
                                             "fsb_reps", "v_f1", "v_ba", "v_mcc", "pct_ci"], env)


def area_d(arch: Path) -> Dict[str, Any]:
    env = {"random": random, "zlib": zlib, "math": math, "defaultdict": defaultdict, "B_BOOT": 2000}
    return load_functions(arch / AREA_D_REL, ["conf", "rconf", "strata", "boot_seed", "strat_boot", "strat_boot_u",
                                             "mcc_c", "ba_c"], env)


def kmw(arch: Path) -> Dict[str, Any]:
    """The archived re-implementation of Kayenta's MannWhitneyClassifier, loaded as gate_sweep.py loads it
    (the source up to its demonstration cases)."""
    src = (arch / KMW_REL).read_text(encoding="utf-8").split("c1e=[")[0]
    ns: Dict[str, Any] = {}
    exec(compile(src, str(arch / KMW_REL), "exec"), ns)
    return ns


def _import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def repo_modules() -> Dict[str, Any]:
    """series_guards, ensemble_judge (Mann-Kendall), representations (summary packet) and ci_utils."""
    if str(JUDGE_APP) not in sys.path:
        sys.path.insert(0, str(JUDGE_APP))
    out = {
        "series_guards": _import_path("series_guards", JUDGE_APP / "series_guards.py"),
        "ensemble_judge": _import_path("ensemble_judge", JUDGE_APP / "ensemble_judge.py"),
        "ci_utils": _import_path("ci_utils", REPO / "tools" / "ci_utils.py"),
    }
    return out


def representations_module():
    if str(JUDGE_APP) not in sys.path:
        sys.path.insert(0, str(JUDGE_APP))
    return _import_path("representations", JUDGE_APP / "representations.py")


def code_hashes(arch: Path) -> Dict[str, str]:
    """SHA-256 of every source file whose functions are reused, for the run record."""
    out = {}
    for rel in (AREA_A_REL, AREA_B_REL, AREA_C_REL, AREA_D_REL, KMW_REL):
        out[f"data-archive/{rel}"] = sha256_file(arch / rel)
    for rel in ("judge-service/app/series_guards.py", "judge-service/app/ensemble_judge.py",
                "judge-service/app/representations.py", "tools/ci_utils.py"):
        out[rel] = sha256_file(REPO / rel)
    return out


# ----------------------------------------------------------------------------- series and realised features
def load_series(arch: Path) -> Dict[str, Dict[str, Any]]:
    """scenario -> {family, label, metrics: [{name, kind, control, canary}]} from the deposited own-data
    export of the 180 benchmark windows (the series Archive S1 also reads)."""
    rows = read_csv(arch / SERIES_REL)
    by: Dict[str, Dict[str, Any]] = {}
    acc: Dict[Tuple[str, str], Dict[str, List[Tuple[str, float]]]] = defaultdict(lambda: {"control": [], "canary": []})
    kinds: Dict[Tuple[str, str], str] = {}
    for r in rows:
        s, m = r["scenario"], r["metric"]
        if s not in by:
            by[s] = {"family": r["family"], "label": r["label"], "metric_names": []}
        if m not in by[s]["metric_names"]:
            by[s]["metric_names"].append(m)
        acc[(s, m)][r["side"]].append((r["timestamp"], float(r["value"])))
        kinds[(s, m)] = r["kind"]
    for s, d in by.items():
        d["metrics"] = []
        for m in d["metric_names"]:
            c = [v for _, v in sorted(acc[(s, m)]["canary"])]
            b = [v for _, v in sorted(acc[(s, m)]["control"])]
            if len(c) != 31 or len(b) != 31:
                raise SystemExit(f"{s} {m}: expected 31 samples per side, got {len(b)} and {len(c)}")
            d["metrics"].append({"name": m, "kind": kinds[(s, m)], "control": b, "canary": c})
    if len(by) != 180:
        raise SystemExit(f"expected 180 windows, found {len(by)}")
    return by


def mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def sd1(x: Sequence[float]) -> float:
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def realised(series: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Per window and metric: the realised quantities the label policies P6 and P7 read, and the
    realised checks of Table 1 used to verify the series."""
    out = {}
    for s, d in series.items():
        rows = []
        for m in d["metrics"]:
            b, c = m["control"], m["canary"]
            mb = mean(b)
            rows.append({"metric": m["name"], "kind": m["kind"],
                         "mean_ratio": mean(c) / mb,
                         "last5_ratio": mean(c[-5:]) / mb,
                         "sd_ratio": sd1(c) / sd1(b)})
        out[s] = rows
    return out


def generator_family_order(series: Dict[str, Dict[str, Any]]) -> List[str]:
    """Family order of the generator, read off the reported metric-name suffixes (gid order)."""
    first = {}
    for s, d in series.items():
        gid = int(re.search(r"_g(\d{4})$", d["metric_names"][0]).group(1))
        f = d["family"]
        first[f] = min(first.get(f, 10 ** 9), gid)
    return sorted(first, key=lambda f: first[f])


# ----------------------------------------------------------------------------- configurations and verdicts
class Config:
    def __init__(self, cid: str, label: str, source: str, filt: str, verdicts: Dict[str, str],
                 scores: Optional[Dict[str, float]] = None, group: str = ""):
        self.cid, self.label, self.source, self.filt = cid, label, source, filt
        self.verdicts, self.scores, self.group = verdicts, scores or {}, group


def _rows_by(rows, judge, model):
    out = {}
    for r in rows:
        if r["judge"] == judge and r["model"] == model and not (r.get("error") or "").strip():
            out[r["scenario"]] = r
    return out


def load_configs(arch: Path, s1: Path, series: Dict[str, Dict[str, Any]]) -> Dict[str, Config]:
    """Every configuration of Tables 6, 8 and S13 of the submitted version, as per-window verdicts."""
    cfg: Dict[str, Config] = {}
    corr = read_csv(arch / CORRECTED_REL)
    stat_rows = _rows_by(corr, "statistical", "-")
    stat = {s: r["verdict"] for s, r in stat_rows.items()}
    cfg["statistical"] = Config("statistical", "Statistical judge", f"data-archive/{CORRECTED_REL}",
                                "judge=statistical; model=-", stat, {s: float(r["score"]) for s, r in stat_rows.items()},
                                "statistical")
    rep_word = {"summary": "summary", "raw": "raw", "plot": "plot"}
    for m in LOCAL_TEXT + LOCAL_VISION:
        reps = ["summary", "raw"] if m in LOCAL_TEXT else ["plot"]
        for rep in reps:
            rr = _rows_by(corr, f"ai:{rep}", m)
            cid = f"ai:{rep}:{m}"
            cfg[cid] = Config(cid, f"{PAPER_MODEL[m]} ({rep_word[rep]})", f"data-archive/{CORRECTED_REL}",
                              f"judge=ai:{rep}; model={m}; error empty", {s: r["verdict"] for s, r in rr.items()},
                              {s: float(r["score"]) for s, r in rr.items()}, "local")
        rr = _rows_by(corr, "hybrid:gated", m)
        cid = f"hybrid:gated:{m}"
        cfg[cid] = Config(cid, f"Hybrid as scored, {PAPER_MODEL[m]} ({PRIMARY[m]})", f"data-archive/{CORRECTED_REL}",
                          f"judge=hybrid:gated; model={m}", {s: r["verdict"] for s, r in rr.items()}, None, "hybrid")
    for m in HOSTED:
        rel = f"results/agg/frontier_long_{m}.csv"
        rows = read_csv(arch / rel)
        reps = ["summary", "raw"] + (["plot"] if m != "gpt-oss-120b-llm" else [])
        for rep in reps:
            rr = _rows_by(rows, f"ai:{rep}", m)
            cid = f"ai:{rep}:{m}"
            cfg[cid] = Config(cid, f"{PAPER_MODEL[m]} ({rep_word[rep]})", f"data-archive/{rel}",
                              f"judge=ai:{rep}; model={m}", {s: r["verdict"] for s, r in rr.items()},
                              {s: float(r["score"]) for s, r in rr.items()}, "hosted")
        rr = _rows_by(rows, "hybrid:gated", m)
        cid = f"hybrid:gated:{m}"
        cfg[cid] = Config(cid, f"Hybrid as scored, {PAPER_MODEL[m]} ({PRIMARY[m]})", f"data-archive/{rel}",
                          f"judge=hybrid:gated; model={m}", {s: r["verdict"] for s, r in rr.items()}, None, "hybrid")
    ens = read_csv(arch / ENSEMBLE_REL)
    cfg["ensemble:tuned"] = Config("ensemble:tuned", "Tuned ensemble", f"data-archive/{ENSEMBLE_REL}", "all rows",
                                   {r["scenario_id"]: r["verdict"] for r in ens},
                                   {r["scenario_id"]: float(r["score"]) for r in ens}, "model-free")
    kind = {s: ("FAIL" if any(m["kind"] != "latency" for m in d["metrics"]) else "PASS") for s, d in series.items()}
    cfg["rule:kind"] = Config("rule:kind", "Metric-kind rule", f"data-archive/{SERIES_REL}",
                              "FAIL iff a metric of kind other than latency is present", kind, None, "rule")
    cfg["rule:kind+stat"] = Config("rule:kind+stat", "Metric-kind + statistical rule",
                                   f"data-archive/{SERIES_REL}; data-archive/{CORRECTED_REL}",
                                   "FAIL iff the metric-kind rule fails or the statistical judge fails",
                                   {s: ("FAIL" if kind[s] == "FAIL" or stat[s] == "FAIL" else "PASS") for s in series},
                                   None, "rule")
    suffix = {}
    for s, d in series.items():
        gids = {int(re.search(r"_g(\d{4})$", n).group(1)) for n in d["metric_names"]}
        if len(gids) != 1:
            raise SystemExit(f"{s}: metric names carry {len(gids)} suffixes")
        suffix[s] = "FAIL" if gids.pop() >= 60 else "PASS"
    cfg["rule:suffix"] = Config("rule:suffix", "Suffix rule", f"data-archive/{SERIES_REL}",
                                "FAIL iff the reported metric-name suffix _gNNNN is 0060 or greater", suffix, None, "rule")
    bf = read_csv(s1 / BFKS_FILE)
    cfg["bfks"] = Config("bfks", "Post-hoc BF+KS (fixed alpha)", f"archive-s1/{BFKS_FILE}", "all rows",
                         {r["scenario"]: r["verdict"] for r in bf}, None, "model-free")
    return cfg


def add_reimplemented(cfg: Dict[str, Config], arch: Path, series: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The statistical judge re-implemented at allowedIncrease 1.05 (check) and 1.01 (Table 8), the
    model-free guard rule, and the guarded gated hybrids (Algorithm S2)."""
    k = kmw(arch)
    mods = repo_modules()
    sg = mods["series_guards"]
    rep = {}
    for gate in (1.05, 1.01):
        v = {}
        for s, d in series.items():
            fail = any(k["classify"](m["control"], m["canary"], 0.25, 0.98, "increase", (1.0, gate)).startswith("High")
                       for m in d["metrics"])
            v[s] = "FAIL" if fail else "PASS"
        rep[gate] = v
    cfg["statistical:gate1.01"] = Config("statistical:gate1.01",
                                         "Statistical judge, allowedIncrease 1.01 (re-implementation)",
                                         f"data-archive/{SERIES_REL}; data-archive/{KMW_REL}",
                                         "re-implemented MannWhitneyClassifier, tolerance 0.25, confidence 0.98, "
                                         "direction increase, effect-size gate 1.01; FAIL iff any metric is High",
                                         rep[1.01], None, "statistical")
    guard = {}
    for s, d in series.items():
        rec = any(sg.detect_recovered_transient(m["control"], m["canary"])[0] for m in d["metrics"])
        ev = len(d["metrics"]) == 1 and sg.detect_equal_variance_noise(d["metrics"][0]["control"],
                                                                        d["metrics"][0]["canary"])[0]
        guard[s] = bool(rec or ev)
    stat = cfg["statistical"].verdicts
    cfg["rule:guard"] = Config("rule:guard", "Guard rule",
                               f"data-archive/{SERIES_REL}; data-archive/{CORRECTED_REL}; judge-service/app/series_guards.py",
                               "PASS iff the statistical judge passes and a guard of Algorithm S2 fires "
                               "(recovered transient on any metric, or equal-variance noise on a single-metric window)",
                               {s: ("PASS" if stat[s] == "PASS" and guard[s] else "FAIL") for s in series}, None, "model-free")
    for m in LOCAL_TEXT + LOCAL_VISION:
        h = cfg[f"hybrid:gated:{m}"].verdicts
        cfg[f"hybrid:guarded:{m}"] = Config(f"hybrid:guarded:{m}", f"Guarded hybrid, {PAPER_MODEL[m]} ({PRIMARY[m]})",
                                            f"data-archive/{CORRECTED_REL}; judge-service/app/series_guards.py",
                                            "hybrid:gated verdict, set to PASS where the statistical judge passes and a guard fires",
                                            {s: ("PASS" if stat[s] == "PASS" and guard[s] else v) for s, v in h.items()},
                                            None, "hybrid")
    agree105 = sum(1 for s in series if rep[1.05][s] == stat[s])
    return {"reimpl_agree_1.05": agree105, "guard_fires": guard}


# ----------------------------------------------------------------------------- confusion and metrics
def confusion(verdicts: Dict[str, str], labels: Dict[str, str]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for s, v in verdicts.items():
        if s not in labels:
            continue
        t = labels[s] == "FAIL"
        p = v == "FAIL"
        if t and p:
            tp += 1
        elif p:
            fp += 1
        elif not t:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def mcc_of(tp: int, fp: int, tn: int, fn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0
