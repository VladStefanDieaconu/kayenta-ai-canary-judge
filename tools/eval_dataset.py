#!/usr/bin/env python3
"""Parametric, seeded, labelled evaluation dataset for the canary-judge study.

This is the shared library behind `tools/seed_eval_dataset.py` (loads the data
into VictoriaMetrics) and `tools/run_experiment.py` (scores every judge over it).
Keeping generation in one deterministic place means the statistical judge (which
reads the data back from VM, or via Kayenta's /judges/judge on the same pairs) and
the AI judge (fed the same in-memory pairs) see identical inputs.

Design:
  * Nine scenario families, each mapping to a cell of the comparison
    (statistical-is-better, statistical-is-blind, statistical-errs, AI-errs).
  * N instances per family, with parameters drawn from a per-instance seeded RNG,
    so every instance is reproducible from the master seed alone.
  * Each instance is a control + experiment series for 1-3 metrics over a fixed
    recent window at a fixed step, carrying only the `Canary` label (plus the
    metric name), as the framework requires.
  * Isolation: every instance namespaces its metric names with an opaque global
    id suffix (`..._g0007`), never the family name, so the ground truth never
    leaks into the prompt the model sees.
  * A manifest records, per scenario: id, family, seed, parameters, the
    ground-truth verdict (PASS/FAIL) and per-metric truth.
  * Per-scenario canary configs are built programmatically from one tuned,
    best-practice statistical template (see STAT_CANARY_TEMPLATE); no hand-written
    files, and the same statistical config across every scenario.

Stdlib only (no numpy) so it runs in the gated host venv. Determinism comes from
`random.Random(seed)`.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

MIB = 1024 * 1024

DEFAULT_MASTER_SEED = 20260621
DEFAULT_N_PER_FAMILY = 12
WINDOW_MINS = 30
STEP_SECONDS = 60
N_SAMPLES = WINDOW_MINS * 60 // STEP_SECONDS + 1  # 31

# Best-practice statistical config (needed for a fair comparison).
# Confirmed against Kayenta source (NetflixACAJudge.scala, MannWhitneyClassifier
# .scala, EffectSizes.scala, WeightedSumScorer.scala):
#   - direction=increase: every metric here is "higher is worse" (latency, memory,
#     errors), so only regressions are penalised (correct per-metric direction).
#   - effectSize.measure=meanRatio (Kayenta default): allowedIncrease=1.05 flags a
#     metric High only when the canary mean is >5% above control and the rank test
#     also finds a significant location shift (the two gates are ANDed). A sensitive
#     5% gate gives the statistical judge its best recall on clean mean shifts
#     without false-flagging benign noise (whose mean ratio is ~1.0).
#   - criticalIncrease=1.25: a >25% regression is a critical (hard) failure.
#   - nanStrategy=remove: drop missing samples cleanly (best practice).
#   - outliers.strategy=keep (not remove): deliberately no IQR-trim. Trimming
#     outliers would discard exactly the tail spikes in the tail/variance families
#     and manufacture the blind spot. Keeping them means the statistical judge
#     still sees the spikes yet structurally misses them, because the rank test is
#     median-based. That is what makes the shortfall real, not a config artifact.
#   - critical=true + mustHaveData=true: best-practice promotion gating (any
#     flagged SLO metric blocks promotion; an SLO metric with no data fails).
# Applied uniformly to every metric in every scenario (no per-scenario tuning).
STAT_CANARY_TEMPLATE: Dict[str, Any] = {
    "direction": "increase",
    "nanStrategy": "remove",
    "outliers": {"strategy": "keep"},
    "effectSize": {
        "measure": "meanRatio",
        "allowedIncrease": 1.05,
        "allowedDecrease": 0.95,
        "criticalIncrease": 1.25,
        "criticalDecrease": 0.75,
    },
    "critical": True,
    "mustHaveData": True,
}
SCORE_THRESHOLDS = {"pass": 75.0, "marginal": 50.0}

# Metric kinds: realistic Prometheus names, baseline level, baseline noise, group.
METRIC_KINDS: Dict[str, Dict[str, Any]] = {
    "latency": {
        "metric": "http_request_duration_seconds",  # p99 latency, seconds
        "group": "latency", "unit": "s", "base": 0.200, "noise": 0.008,
    },
    "memory": {
        "metric": "process_resident_memory_bytes",   # RSS, bytes
        "group": "resources", "unit": "bytes", "base": 600.0 * MIB, "noise": 6.0 * MIB,
    },
    "errors": {
        "metric": "http_server_errors_per_second",    # error rate, errors/s
        "group": "errors", "unit": "err/s", "base": 0.50, "noise": 0.02,
    },
}

# Family -> (truth verdict, metric kinds it uses).
FAMILY_SPEC: Dict[str, Dict[str, Any]] = {
    "no_change":             {"truth": "PASS", "kinds": ["latency"]},
    "noise_equivalent":      {"truth": "PASS", "kinds": ["latency"]},
    "healed_transient":      {"truth": "PASS", "kinds": ["latency"]},
    "clean_mean_shift":      {"truth": "FAIL", "kinds": ["latency"]},
    "variance_increase":     {"truth": "FAIL", "kinds": ["latency"]},
    "tail_regression":       {"truth": "FAIL", "kinds": ["latency"]},
    "gradual_drift":         {"truth": "FAIL", "kinds": ["memory"]},
    "cross_metric_marginal": {"truth": "FAIL", "kinds": ["latency", "memory", "errors"]},
    "subtle_regression":     {"truth": "FAIL", "kinds": ["latency"]},
}
FAMILIES: List[str] = list(FAMILY_SPEC.keys())

# A second, separate dataset, added for the prompt ablation.
#
# These three families are NOT appended to FAMILY_SPEC, and that is deliberate.
# `gid` counts across every family in order and feeds each scenario's seed, so
# adding a tenth family would re-seed all 180 existing scenarios, rename every
# metric, and invalidate every result ever produced from them. They therefore live in
# their own dataset with its own gid range, and build_scenarios() still returns
# exactly the original 180 unless asked for otherwise.
#
# What they are for: the prompt variants under test speak about the end of the
# observation window. Scoring them only on healed_transient would measure
# whether an instruction works on the one case it was written for. Two of these
# families are adversarial to that instruction in opposite directions, and the
# third is held out from every rubric entirely.
GENERALISATION_FAMILY_SPEC: Dict[str, Dict[str, Any]] = {
    "partial_recovery":    {"truth": "FAIL", "kinds": ["latency"]},
    "late_transient":      {"truth": "FAIL", "kinds": ["latency"]},
    "sustained_excursion": {"truth": "FAIL", "kinds": ["latency"]},
}
GENERALISATION_FAMILIES: List[str] = list(GENERALISATION_FAMILY_SPEC.keys())

# gid_base keeps the two datasets' metric-name namespaces disjoint, so a scenario
# from one can never be confused with a scenario from the other -- in a metric
# store, in an ai-log filename, or in a cache key.
DATASETS: Dict[str, Dict[str, Any]] = {
    "original-180":      {"spec": FAMILY_SPEC, "families": FAMILIES, "gid_base": 0},
    "generalisation-60": {"spec": GENERALISATION_FAMILY_SPEC,
                          "families": GENERALISATION_FAMILIES, "gid_base": 1000},
}
DEFAULT_DATASET_ID = "original-180"


def family_order(family: str) -> Tuple[int, int]:
    """Sort key placing families in dataset order, then in declaration order.

    Reporting code used to sort on FAMILIES.index(), which raises the moment a
    run contains a family from the second dataset.
    """
    for di, (_, ds) in enumerate(DATASETS.items()):
        if family in ds["families"]:
            return (di, ds["families"].index(family))
    return (99, 0)


def family_dataset(family: str) -> str:
    """Which dataset a family belongs to. Used when reporting mixed runs."""
    for dsid, ds in DATASETS.items():
        if family in ds["families"]:
            return dsid
    return ""


@dataclass
class MetricSpec:
    name: str            # namespaced Prometheus metric name (e.g. ..._g0007)
    base: str            # base metric name (kind)
    kind: str            # latency | memory | errors
    group: str           # canary group
    unit: str
    direction: str       # always "increase" here
    truth: str           # per-metric truth: pass | high


@dataclass
class Scenario:
    id: str
    family: str
    gid: int
    seed: int
    truth: str           # PASS | FAIL
    metrics: List[MetricSpec]
    params: Dict[str, Any] = field(default_factory=dict)


# Series generation: one function per family, parameterised by a seeded RNG.
# Each returns (control, experiment, per_metric_truth, params).
def _noisy(rng: random.Random, level: float, noise: float, n: int) -> List[float]:
    return [level + rng.gauss(0.0, noise) for _ in range(n)]


def _gen_kind(family: str, kind: str, rng: random.Random, n: int) -> Tuple[List[float], List[float], str, Dict[str, Any]]:
    spec = METRIC_KINDS[kind]
    base, noise = spec["base"], spec["noise"]
    p: Dict[str, Any] = {}

    if family == "no_change":
        # baseline ~ experiment, mild noise; mean ratio within +-1.5% (< the 1.05 gate).
        u = rng.uniform(-0.015, 0.015)
        p["mean_ratio_target"] = round(1 + u, 4)
        return _noisy(rng, base, noise, n), _noisy(rng, base * (1 + u), noise, n), "pass", p

    if family == "noise_equivalent":
        # both high-variance, same distribution (no location shift).
        hi = noise * rng.uniform(5.0, 8.0)
        p["high_noise_factor"] = round(hi / noise, 2)
        return _noisy(rng, base, hi, n), _noisy(rng, base, hi, n), "pass", p

    if family == "healed_transient":
        # experiment degraded early (elevated), then recovers to control by mid/late
        # window. Overall location is elevated (statistical may false-FAIL); a
        # holistic reader sees the downward recovery trend and PASSes.
        deg = rng.uniform(1.5, 1.9)
        dur = rng.uniform(0.40, 0.50)
        k = max(2, int(n * dur))
        rec = max(3, int(n * 0.20))
        p["degradation_factor"] = round(deg, 3)
        p["degraded_fraction"] = round(dur, 3)
        control = _noisy(rng, base, noise, n)
        exp: List[float] = []
        for i in range(n):
            if i < k:
                level = base * deg
            elif i < k + rec:
                level = base * deg - (base * deg - base) * ((i - k) / rec)
            else:
                level = base
            exp.append(level + rng.gauss(0.0, noise))
        return control, exp, "pass", p

    if family == "clean_mean_shift":
        # experiment median clearly worse (its home turf for the rank test).
        shift = rng.uniform(1.25, 1.45)
        p["shift_factor"] = round(shift, 3)
        return _noisy(rng, base, noise, n), _noisy(rng, base * shift, noise, n), "high", p

    if family == "variance_increase":
        # equal median, much higher variance (rank test is blind to spread).
        hi = noise * rng.uniform(6.0, 10.0)
        p["variance_factor"] = round(hi / noise, 2)
        return _noisy(rng, base, noise, n), _noisy(rng, base, hi, n), "high", p

    if family == "tail_regression":
        # median/mean ~flat, p95/p99/max clearly worse (a few large spikes). The
        # rank test sees no median shift -> Pass; the tail is a real regression.
        control = _noisy(rng, base, noise, n)
        exp = _noisy(rng, base, noise, n)
        num_spikes = rng.randint(3, 5)
        positions = rng.sample(range(n), num_spikes)
        for pos in positions:
            exp[pos] = base * rng.uniform(2.5, 4.0)
        p["num_spikes"] = num_spikes
        return control, exp, "high", p

    if family == "gradual_drift":
        # flat for the first ~80% of the window, then a progressive ramp worse in
        # the last fifth. The median (and >half the samples) sit in the flat
        # region so the order-blind rank test sees no location shift and the mean
        # ratio stays under the effect-size gate -> statistical PASSes; the steep
        # end-of-window trend/slope and tail are clearly bad -> AI catches it.
        start = rng.uniform(0.80, 0.85)
        end_mag = rng.uniform(1.30, 1.50)
        s = max(1, int(n * start))
        p["ramp_start_fraction"] = round(start, 3)
        p["end_magnitude"] = round(end_mag, 3)
        control = _noisy(rng, base, noise, n)
        exp = []
        for i in range(n):
            if i < s:
                level = base
            else:
                frac = (i - s) / max(1, (n - 1 - s))
                level = base * (1 + (end_mag - 1) * frac)
            exp.append(level + rng.gauss(0.0, noise))
        return control, exp, "high", p

    if family == "cross_metric_marginal":
        # each metric individually within per-metric tolerance (mean ratio < 1.05,
        # with margin for sample-mean noise) so the statistical judge passes each,
        # but all of them jointly degraded -> a real systemic regression the AI can
        # reason across.
        r = rng.uniform(1.018, 1.032)
        p["per_metric_ratio"] = round(r, 4)
        return _noisy(rng, base, noise, n), _noisy(rng, base * r, noise, n), "high", p

    if family == "subtle_regression":
        # small but real shift near the tolerance boundary; hard for all, AI (esp.
        # small models) at risk of a false negative.
        r = rng.uniform(1.060, 1.085)
        p["shift_ratio"] = round(r, 4)
        return _noisy(rng, base, noise, n), _noisy(rng, base * r, noise, n), "high", p

    # --- generalisation-60 families -------------------------------------------
    if family == "partial_recovery":
        # healed_transient with a floor. Same degradation magnitude, same degraded
        # fraction, same recovery length -- the single difference is that recovery
        # stops at `residual` above control instead of reaching it. Holding
        # everything else identical is the point: any verdict difference between
        # this family and healed_transient is attributable to the residual alone,
        # which is the contrast the prompt ablation is trying to measure.
        #
        # Truth is FAIL. The canary ends the window ~20% worse than control on a
        # latency SLI and stays there; that is the state promotion would make
        # permanent. Adversarial to any rubric that says "returned to control
        # levels", because a model reading that loosely sees recovery and passes.
        deg = rng.uniform(1.5, 1.9)
        dur = rng.uniform(0.40, 0.50)
        residual = rng.uniform(1.18, 1.25)
        k = max(2, int(n * dur))
        rec = max(3, int(n * 0.20))
        p["degradation_factor"] = round(deg, 3)
        p["degraded_fraction"] = round(dur, 3)
        p["residual_ratio"] = round(residual, 4)
        control = _noisy(rng, base, noise, n)
        floor = base * residual
        exp: List[float] = []
        for i in range(n):
            if i < k:
                level = base * deg
            elif i < k + rec:
                level = base * deg - (base * deg - floor) * ((i - k) / rec)
            else:
                level = floor
            exp.append(level + rng.gauss(0.0, noise))
        return control, exp, "high", p

    if family == "late_transient":
        # Healthy for the first three quarters, then a step change that is still
        # present at the last sample. A step rather than gradual_drift's ramp, so
        # the two are not the same waveform under different names.
        #
        # Truth is FAIL, and it is adversarial in the opposite direction to
        # partial_recovery: it punishes a judge that has learned to discount early
        # evidence but reads recency naively, and it checks that a variant which
        # repairs the healed transient has not simply inverted the original error.
        onset = rng.uniform(0.72, 0.78)
        mag = rng.uniform(1.35, 1.55)
        s = max(1, int(n * onset))
        p["onset_fraction"] = round(onset, 3)
        p["step_magnitude"] = round(mag, 3)
        control = _noisy(rng, base, noise, n)
        exp = [(base if i < s else base * mag) + rng.gauss(0.0, noise) for i in range(n)]
        return control, exp, "high", p

    if family == "sustained_excursion":
        # The held-out family. Control and experiment are the SAME multiset of
        # values -- experiment is a permutation of control -- so mean, stddev,
        # min, max and every percentile are identical by construction, not merely
        # close. The only difference is arrangement in time: control's elevated
        # samples are isolated single-sample spikes, the canary's are gathered
        # into a few contiguous multi-sample excursions.
        #
        # Truth is FAIL. A service that sits above threshold for several
        # consecutive minutes burns an error budget and trips multi-window
        # alerting; the same number of isolated one-sample spikes does not. This
        # is the standard sustained-vs-transient distinction in SLO practice.
        #
        # It is held out because no version of the rubric names this pattern:
        # spread, tail and trend are all named, and none of them separates these
        # two series. The summary representation provably cannot -- its statistics
        # are equal on both sides apart from slope, which the excursion placement
        # below keeps near zero on purpose, so the family tests judgment rather
        # than instruction-following.
        n_runs = rng.randint(2, 3)
        run_len = rng.randint(3, 4)
        n_elev = n_runs * run_len
        elev_ratio = rng.uniform(1.25, 1.45)

        baseline_vals = _noisy(rng, base, noise, n - n_elev)
        elevated_vals = [base * elev_ratio + rng.gauss(0.0, noise) for _ in range(n_elev)]

        # Control: elevated samples scattered as isolated spikes, never adjacent,
        # so no run of two exists anywhere in the baseline series.
        positions: List[int] = []
        candidates = list(range(n))
        rng.shuffle(candidates)
        for c in candidates:
            if len(positions) == n_elev:
                break
            if all(abs(c - q) > 1 for q in positions):
                positions.append(c)
        control_pool = list(baseline_vals)
        rng.shuffle(control_pool)
        control = []
        ci = 0
        elev_iter = list(elevated_vals)
        rng.shuffle(elev_iter)
        for i in range(n):
            if i in positions:
                control.append(elev_iter[positions.index(i)])
            else:
                control.append(control_pool[ci])
                ci += 1

        # Experiment: the same values, with the elevated ones gathered into
        # contiguous runs. Run starts are spread across the window so the fitted
        # slope stays near zero and the excursions are not merely a late ramp.
        slot = n // n_runs
        starts = []
        for r_i in range(n_runs):
            lo = r_i * slot
            hi = min(n - run_len, lo + slot - run_len)
            starts.append(rng.randint(lo, max(lo, hi)))
        run_positions: List[int] = []
        for st in starts:
            run_positions.extend(range(st, min(n, st + run_len)))
        run_positions = sorted(set(run_positions))[:n_elev]

        exp_pool = list(baseline_vals)
        rng.shuffle(exp_pool)
        exp_elev = list(elevated_vals)
        rng.shuffle(exp_elev)
        exp = []
        ei = 0
        pi = 0
        for i in range(n):
            if i in run_positions and pi < len(exp_elev):
                exp.append(exp_elev[pi])
                pi += 1
            else:
                exp.append(exp_pool[ei])
                ei += 1

        p["n_runs"] = n_runs
        p["run_length"] = run_len
        p["n_elevated"] = n_elev
        p["elevated_ratio"] = round(elev_ratio, 4)
        p["excursion_starts"] = starts
        return control, exp, "high", p

    raise ValueError(f"unknown family {family}")


# Scenario construction (deterministic from the master seed).
def build_scenarios(master_seed: int = DEFAULT_MASTER_SEED,
                    n_per_family: int = DEFAULT_N_PER_FAMILY,
                    dataset_id: str = DEFAULT_DATASET_ID) -> List[Scenario]:
    """The scenarios of one dataset, deterministic from the master seed.

    The default is the original nine families with gid starting at 0, which is
    what every published run used; the signature gained a parameter but not a
    behaviour.
    """
    if dataset_id not in DATASETS:
        raise ValueError(f"unknown dataset_id {dataset_id!r}; known: {sorted(DATASETS)}")
    ds = DATASETS[dataset_id]
    spec_map, family_list = ds["spec"], ds["families"]

    scenarios: List[Scenario] = []
    gid = ds["gid_base"]
    for family in family_list:
        spec = spec_map[family]
        for idx in range(n_per_family):
            seed = (master_seed * 1_000_003 + gid * 97 + idx) & 0x7FFFFFFF
            metrics: List[MetricSpec] = []
            for kind in spec["kinds"]:
                mk = METRIC_KINDS[kind]
                metrics.append(MetricSpec(
                    name=f"{mk['metric']}_g{gid:04d}",
                    base=mk["metric"], kind=kind, group=mk["group"], unit=mk["unit"],
                    direction="increase", truth="pass" if spec["truth"] == "PASS" else "high",
                ))
            scenarios.append(Scenario(
                id=f"{family}_{idx:03d}", family=family, gid=gid, seed=seed,
                truth=spec["truth"], metrics=metrics,
            ))
            gid += 1
    return scenarios


def series_for_scenario(scenario: Scenario) -> Dict[str, Dict[str, List[float]]]:
    """Regenerate the (control, experiment) arrays for every metric of a scenario,
    deterministically from its seed. Both the seeder and the runner call this, so
    the data fed to VM and to the AI judge are byte-identical."""
    out: Dict[str, Dict[str, List[float]]] = {}
    params: Dict[str, Any] = {}
    # A separate RNG stream per metric (seed offset by metric index) keeps each
    # metric independent yet reproducible.
    for mi, m in enumerate(scenario.metrics):
        rng = random.Random(scenario.seed + 1000 * mi)
        control, experiment, truth, p = _gen_kind(scenario.family, m.kind, rng, N_SAMPLES)
        out[m.name] = {"control": control, "experiment": experiment}
        m.truth = truth
        params[m.name] = p
    scenario.params = params
    return out


# Programmatic canary config + metricSetPairList (from the tuned template).
def _group_weights(groups: List[str]) -> Dict[str, float]:
    """Distribute 100 evenly across the groups actually present. Weights must sum
    to 100 (a structural requirement, not a per-scenario tuning knob)."""
    uniq = list(dict.fromkeys(groups))
    n = len(uniq)
    base = round(100.0 / n, 2)
    weights = {g: base for g in uniq}
    weights[uniq[-1]] = round(100.0 - base * (n - 1), 2)  # fix rounding so sum==100
    return weights


def build_canary_config(scenario: Scenario, judge: Dict[str, Any]) -> Dict[str, Any]:
    """Build a canary config for one scenario from the tuned template. `judge` is
    the judge block, e.g. {"name":"NetflixACAJudge-v1.0","judgeConfigurations":{}}
    or {"name":"RemoteJudge-v1.0","judgeConfigurations":{"mode":"summary",...}}."""
    metrics = []
    groups = []
    for m in scenario.metrics:
        groups.append(m.group)
        metrics.append({
            "name": m.name,
            "query": {
                "type": "prometheus", "serviceType": "prometheus",
                "metricName": m.name,
                "customInlineTemplate": f'PromQL:sum without(Canary) ({m.name}{{Canary="${{scope}}"}})',
                "labelBindings": [],
            },
            "groups": [m.group],
            "analysisConfigurations": {"canary": dict(STAT_CANARY_TEMPLATE)},
            "scopeName": "default",
        })
    return {
        "name": f"eval-{scenario.id}",
        "description": f"eval dataset scenario {scenario.id} (family={scenario.family}, truth={scenario.truth})",
        "configVersion": "1.0",
        "applications": ["ad-hoc"],
        "judge": judge,
        "metrics": metrics,
        "classifier": {
            "groupWeights": _group_weights(groups),
            "scoreThresholds": dict(SCORE_THRESHOLDS),
        },
    }


def build_pairs(scenario: Scenario, series: Dict[str, Dict[str, List[float]]],
                base_millis: int, step_millis: int = STEP_SECONDS * 1000) -> List[Dict[str, Any]]:
    """Build a Kayenta metricSetPairList from the scenario's series, in the same
    shape Kayenta POSTs to /judge (so the AI judge and the real NetflixACAJudge
    via /judges/judge both judge the same pairs)."""
    pairs = []
    for m in scenario.metrics:
        vals = series[m.name]
        scope = {"startTimeIso": _iso(base_millis), "startTimeMillis": base_millis, "stepMillis": step_millis}
        pairs.append({
            "name": m.name,
            "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{scenario.id}:{m.name}")),
            "tags": {},
            "values": {"control": list(vals["control"]), "experiment": list(vals["experiment"])},
            "scopes": {"control": dict(scope), "experiment": dict(scope)},
            "attributes": {
                "control": {"query": f'sum without(Canary) ({m.name}{{Canary="Control"}})'},
                "experiment": {"query": f'sum without(Canary) ({m.name}{{Canary="Experiment"}})'},
            },
        })
    return pairs


def _iso(millis: int) -> str:
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Manifest.
def scenario_to_manifest(scenario: Scenario) -> Dict[str, Any]:
    return {
        "id": scenario.id,
        "family": scenario.family,
        "gid": scenario.gid,
        "seed": scenario.seed,
        "truth": scenario.truth,
        "params": scenario.params,
        "metrics": [
            {"name": m.name, "base": m.base, "kind": m.kind, "group": m.group,
             "unit": m.unit, "direction": m.direction, "truth": m.truth}
            for m in scenario.metrics
        ],
    }


def build_manifest(master_seed: int, n_per_family: int, scenarios: List[Scenario],
                   base_millis: int) -> Dict[str, Any]:
    return {
        "master_seed": master_seed,
        "n_per_family": n_per_family,
        "families": FAMILIES,
        "window_mins": WINDOW_MINS,
        "step_seconds": STEP_SECONDS,
        "n_samples": N_SAMPLES,
        "base_millis": base_millis,
        "window_start_iso": _iso(base_millis),
        "window_end_iso": _iso(base_millis + (N_SAMPLES - 1) * STEP_SECONDS * 1000),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "statistical_template": {"canary": STAT_CANARY_TEMPLATE, "scoreThresholds": SCORE_THRESHOLDS},
        "scenarios": [scenario_to_manifest(s) for s in scenarios],
    }


def aligned_base_millis(now: float | None = None) -> int:
    """Step-aligned start time for the seeded window, ending ~now."""
    t = int(now if now is not None else time.time())
    t -= t % STEP_SECONDS
    start = t - (N_SAMPLES - 1) * STEP_SECONDS
    return start * 1000
