#!/usr/bin/env python3
"""The two control datasets of release line 1.2.0, and the two harness options
their runs use. No stack, no model, no judge: every check below is about the
generator or the harness code, never about a verdict.

  1. kind-controls-180 and heldout-benign-100 have the families, order, counts,
     truths, metric kinds, gid bases and token seeds the protocol fixes.
  2. Metric names are opaque: one random token per window, reproducible from the
     dataset's token seed, carrying neither the family nor the gid.
  3. No RNG seed is shared, per-metric offsets included, between a window of a
     new dataset and a window of original-180, generalisation-60 or another
     new window.
  4. The three drift controls and kc_tri_marginal produce exactly what the
     existing gradual_drift and cross_metric_marginal paths produce for the same
     kind and RNG.
  5. For every exchangeable PASS generator the deterministic component is
     identical on both sides (for isolated_spikes_both: the same levels, at
     isolated positions on each side), and the shared waveforms are those of
     gradual_drift, partial_recovery and late_transient.
  6. For the two recovered-transient generators the canary level equals the
     base level over at least the last seven samples.
  7. `--max-idx K` keeps the windows with index below K and leaves their seeds
     those of n = 20; `--skip-failed` writes a recorded failed call as an error
     row without calling the judge again.

Usage:
  python tools/test_r1_control_datasets.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "judge-service" / "app"))

import eval_dataset as ed  # noqa: E402

MASTER_SEED = 20260621
N = 20
KC = "kind-controls-180"
HB = "heldout-benign-100"

KC_EXPECTED = [
    ("kc_latency_nochange", "PASS", ["latency"], "exchangeable_nochange"),
    ("kc_memory_nochange", "PASS", ["memory"], "exchangeable_nochange"),
    ("kc_errors_nochange", "PASS", ["errors"], "exchangeable_nochange"),
    ("kc_tri_nochange", "PASS", ["latency", "memory", "errors"], "exchangeable_nochange"),
    ("kc_memory_shared_ramp", "PASS", ["memory"], "shared_ramp"),
    ("kc_latency_drift", "FAIL", ["latency"], "gradual_drift"),
    ("kc_memory_drift", "FAIL", ["memory"], "gradual_drift"),
    ("kc_errors_drift", "FAIL", ["errors"], "gradual_drift"),
    ("kc_tri_marginal", "FAIL", ["latency", "memory", "errors"], "cross_metric_marginal"),
]
HB_EXPECTED = [
    ("hb_shared_partial_recovery", "PASS", ["latency"], "shared_partial_recovery"),
    ("hb_shared_late_step", "PASS", ["latency"], "shared_late_step"),
    ("hb_isolated_spikes_both", "PASS", ["latency"], "isolated_spikes_both"),
    ("hb_recovered_mid_transient", "PASS", ["latency"], "recovered_mid_transient"),
    ("hb_brief_early_blip", "PASS", ["latency"], "brief_early_blip"),
]
EXCHANGEABLE = ("exchangeable_nochange", "shared_ramp", "shared_partial_recovery", "shared_late_step")
SHARED_TWIN = {"shared_ramp": "gradual_drift", "shared_partial_recovery": "partial_recovery",
               "shared_late_step": "late_transient"}
RECOVERED = ("recovered_mid_transient", "brief_early_blip")
TOKEN_RE = re.compile(r"^t[0-9a-f]{10}$")


class ZeroNoise(random.Random):
    """An RNG whose Gaussian draws are 0: a generator run on it returns its
    deterministic component (the level sequence) in place of each series."""

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:  # noqa: D401
        return mu


def metric_seeds(scenarios) -> List[int]:
    return [sc.seed + 1000 * mi for sc in scenarios for mi in range(len(sc.metrics))]


def main() -> int:
    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  {name:78s} {'ok' if cond else 'FAIL'}")
        if not cond:
            failures.append(name)

    orig = ed.build_scenarios(MASTER_SEED, N, "original-180")
    gen = ed.build_scenarios(MASTER_SEED, N, "generalisation-60")
    kc = ed.build_scenarios(MASTER_SEED, N, KC)
    hb = ed.build_scenarios(MASTER_SEED, N, HB)

    # 1. structure ---------------------------------------------------------------
    print("structure")
    for dsid, scs, expected, gid_base, token_seed in ((KC, kc, KC_EXPECTED, 2000, 20261122),
                                                      (HB, hb, HB_EXPECTED, 3000, 20261123)):
        ds = ed.DATASETS[dsid]
        check(f"{dsid}: {len(expected) * N} windows", len(scs) == len(expected) * N)
        check(f"{dsid}: families in the protocol's order", ds["families"] == [e[0] for e in expected])
        check(f"{dsid}: gid_base {gid_base}, gids sequential",
              ds["gid_base"] == gid_base and [s.gid for s in scs] == list(range(gid_base, gid_base + len(scs))))
        check(f"{dsid}: token_seed {token_seed}, opaque names (no metric_suffix key)",
              ds.get("token_seed") == token_seed and "metric_suffix" not in ds)
        for fam, truth, kinds, generator in expected:
            members = [s for s in scs if s.family == fam]
            check(f"{dsid}/{fam}: 20 windows, ids _000.._019",
                  [s.id for s in members] == [f"{fam}_{i:03d}" for i in range(N)])
            check(f"{dsid}/{fam}: truth {truth}, metric truths {'pass' if truth == 'PASS' else 'high'}",
                  all(s.truth == truth for s in members)
                  and all(m.truth == ("pass" if truth == "PASS" else "high") for s in members for m in s.metrics))
            check(f"{dsid}/{fam}: kinds {','.join(kinds)}",
                  all([m.kind for m in s.metrics] == kinds for s in members))
            check(f"{dsid}/{fam}: generator {generator}", ed.generator_of(fam) == generator)
            # the generator's own truth agrees with the label after generation
            for s in members:
                ed.series_for_scenario(s)
            check(f"{dsid}/{fam}: generated per-metric truth agrees with the label",
                  all(m.truth == ("pass" if truth == "PASS" else "high") for s in members for m in s.metrics))
    check("published families keep their own code path",
          all(ed.generator_of(f) == f for f in ed.FAMILIES + ed.GENERALISATION_FAMILIES))

    # 2. opaque names --------------------------------------------------------------
    print("metric names")
    for dsid, scs in ((KC, kc), (HB, hb)):
        toks = [s.token for s in scs]
        check(f"{dsid}: every window has a token of the form t + 10 hex digits",
              all(TOKEN_RE.match(t) for t in toks))
        check(f"{dsid}: tokens distinct", len(set(toks)) == len(toks))
        check(f"{dsid}: tokens reproduce from the token seed",
              toks == ed._opaque_tokens(dsid, ed.DATASETS[dsid]["token_seed"], len(scs)))
        check(f"{dsid}: every metric name is <base metric>_<token>",
              all(m.name == f"{ed.METRIC_KINDS[m.kind]['metric']}_{s.token}" for s in scs for m in s.metrics))
        check(f"{dsid}: no name carries a family name, a gid suffix or a label",
              all(s.family not in m.name and f"g{s.gid:04d}" not in m.name
                  and not re.search(r"(pass|fail|drift|nochange|ramp|spike|blip|transient)", m.name.split("_")[-1])
                  for s in scs for m in s.metrics))
        cfg = ed.build_canary_config(scs[0], {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}})
        check(f"{dsid}: canary config name and description carry only the token",
              cfg["name"] == f"eval-{scs[0].token}" and scs[0].family not in cfg["description"])
    check("tokens distinct across the two new datasets", not ({s.token for s in kc} & {s.token for s in hb}))

    # 3. RNG seeds -----------------------------------------------------------------
    print("RNG seeds")
    published = set(metric_seeds(orig)) | set(metric_seeds(gen))
    new_kc, new_hb = metric_seeds(kc), metric_seeds(hb)
    check("kind-controls-180: no per-metric seed shared with original-180 or generalisation-60",
          not (set(new_kc) & published))
    check("heldout-benign-100: no per-metric seed shared with original-180 or generalisation-60",
          not (set(new_hb) & published))
    check("no per-metric seed shared between two new windows (within and across the new datasets)",
          len(set(new_kc + new_hb)) == len(new_kc) + len(new_hb))
    check("no scenario seed shared between a new window and any other window",
          len({s.seed for s in orig + gen + kc + hb}) == len(orig + gen + kc + hb))

    # 4. reused code paths ---------------------------------------------------------
    print("reused code paths")
    for fam, path in (("kc_latency_drift", "gradual_drift"), ("kc_memory_drift", "gradual_drift"),
                      ("kc_errors_drift", "gradual_drift"), ("kc_tri_marginal", "cross_metric_marginal")):
        same = True
        for s in (x for x in kc if x.family == fam):
            series = ed.series_for_scenario(s)
            for mi, m in enumerate(s.metrics):
                c, e, t, p = ed._gen_kind(path, m.kind, random.Random(s.seed + 1000 * mi), ed.N_SAMPLES)
                same &= (series[m.name]["control"] == c and series[m.name]["experiment"] == e
                         and t == "high" and s.params[m.name] == p)
        check(f"{fam}: identical to the {path} path for the same kind and RNG (series, params)", same)

    # 5. exchangeable generators ---------------------------------------------------
    print("exchangeable generators (noise-free replay of every window's RNG stream)")
    for dsid, scs in ((KC, kc), (HB, hb)):
        for s in scs:
            g = ed.generator_of(s.family)
            if g not in EXCHANGEABLE + ("isolated_spikes_both",):
                continue
            for mi, m in enumerate(s.metrics):
                seed = s.seed + 1000 * mi
                c0, e0, _, _ = ed._gen_kind(g, m.kind, ZeroNoise(seed), ed.N_SAMPLES)
                base = ed.METRIC_KINDS[m.kind]["base"]
                if g in EXCHANGEABLE:
                    if c0 != e0:
                        failures.append(f"{s.id}/{m.kind}: deterministic component differs between sides")
                    if g == "exchangeable_nochange" and any(v != base for v in c0):
                        failures.append(f"{s.id}/{m.kind}: exchangeable_nochange level is not the base level")
                    if g in SHARED_TWIN:
                        _, twin_e, _, _ = ed._gen_kind(SHARED_TWIN[g], m.kind, ZeroNoise(seed), ed.N_SAMPLES)
                        if twin_e != c0:
                            failures.append(f"{s.id}: level sequence differs from {SHARED_TWIN[g]}'s canary levels")
                else:
                    if sorted(c0) != sorted(e0):
                        failures.append(f"{s.id}: isolated_spikes_both levels differ between sides")
    for g in EXCHANGEABLE:
        fams = [f for f, *_ in KC_EXPECTED + HB_EXPECTED if ed.generator_of(f) == g]
        bad = [x for x in failures if any(x.startswith(f) for f in fams)]
        check(f"{g}: same deterministic component on both sides in every window", not bad)
    for g, twin in SHARED_TWIN.items():
        fams = [f for f, *_ in KC_EXPECTED + HB_EXPECTED if ed.generator_of(f) == g]
        bad = [x for x in failures if twin in x and any(x.startswith(f) for f in fams)]
        check(f"{g}: level sequence equals {twin}'s canary levels for the same draws", not bad)
    spikes = [s for s in hb if s.family == "hb_isolated_spikes_both"]
    for s in spikes:
        ed.series_for_scenario(s)
    ok_iso = True
    repeats = []
    for s in spikes:
        p = s.params[s.metrics[0].name]
        for side in ("control", "experiment"):
            pos = p[f"elevated_positions_{side}"]
            ok_iso &= len(pos) == len(set(pos)) == p["n_elevated"] == p["n_runs"] * p["run_length"]
            ok_iso &= all(0 <= q < ed.N_SAMPLES for q in pos)
            ok_iso &= all(b - a > 1 for a, b in zip(pos, pos[1:]))
            ok_iso &= isinstance(p[f"placement_repeats_{side}"], int) and p[f"placement_repeats_{side}"] >= 0
        repeats.append((s.id, p["placement_repeats_control"], p["placement_repeats_experiment"]))
    check("isolated_spikes_both: n_elev isolated (never adjacent) elevated samples on each side", ok_iso)
    print("    placement repeats per window (control, experiment): "
          + "; ".join(f"{i[-3:]}={a},{b}" for i, a, b in repeats))
    real_diff = all(ed.series_for_scenario(s)[s.metrics[0].name]["control"]
                    != ed.series_for_scenario(s)[s.metrics[0].name]["experiment"] for s in kc + hb)
    check("with noise, the two sides of every window differ (independent noise)", real_diff)

    # 6. recovered transients --------------------------------------------------------
    print("recovered transients")
    for g in RECOVERED:
        fam = next(f for f, *_ in HB_EXPECTED if ed.generator_of(f) == g)
        tails, ctl_flat = [], True
        for s in (x for x in hb if x.family == fam):
            c0, e0, _, _ = ed._gen_kind(g, "latency", ZeroNoise(s.seed), ed.N_SAMPLES)
            base = ed.METRIC_KINDS["latency"]["base"]
            ctl_flat &= all(v == base for v in c0)
            k = 0
            for v in reversed(e0):
                if v != base:
                    break
                k += 1
            tails.append(k)
        check(f"{g}: control at base throughout; canary at base over the last >= 7 samples "
              f"(min {min(tails)})", ctl_flat and min(tails) >= 7)

    # 7. harness options ---------------------------------------------------------------
    print("harness options")
    import judge_clients  # noqa: E402
    import run_frontier_experiment as rfe  # noqa: E402

    kept = [s for s in ed.build_scenarios(MASTER_SEED, N, "original-180") if rfe.scenario_index(s) < 2]
    check("--max-idx 2 on original-180 keeps 18 windows, two per family",
          len(kept) == 18 and all(sum(1 for s in kept if s.family == f) == 2 for f in ed.FAMILIES))
    by_id = {s.id: s.seed for s in orig}
    check("--max-idx keeps the seeds of n = 20 (not those of n = 2)",
          all(by_id[s.id] == s.seed for s in kept)
          and any(s.seed != t.seed for s, t in zip(kept, ed.build_scenarios(MASTER_SEED, 2, "original-180"))))

    calls: List[str] = []
    saved = (judge_clients.run_default_judge, judge_clients.judge_ai)

    def fake_stat(pairs, cfg, pass_t=75.0, marginal_t=50.0, base_url=""):
        return {"verdict": "PASS", "classification": "Pass", "score": 100.0,
                "per_metric": [{"name": p["name"], "classification": "Pass", "reason": ""} for p in pairs],
                "rationale": "", "judge_name": "stub", "model": "", "latency": 0.0, "ok": True,
                "error": "", "error_kind": ""}

    def fake_ai(pairs, cfg, mode, model, pass_t=75.0, marginal_t=50.0, timeout=180, prompt_id=""):
        calls.append(mode)
        return {"verdict": "PASS", "classification": "Pass", "score": 90.0, "per_metric": [],
                "rationale": "stub", "judge_name": "stub", "model": model, "latency": 0.0, "ok": True,
                "error": "", "error_kind": "", "prompt_id": prompt_id, "prompt_hash": "h",
                "finish_reason": "stop", "tokens_in": 1, "tokens_out": 1, "truncated": False,
                "score_check": "ok"}

    try:
        judge_clients.run_default_judge, judge_clients.judge_ai = fake_stat, fake_ai
        sc = orig[0]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "failed.jsonl"
            ledger.write_text(json.dumps({"model": "m", "representation": "summary",
                                          "prompt_id": "v1-frozen-2026-06", "scenario_id": sc.id,
                                          "seed": sc.seed, "error_kind": "empty_completion",
                                          "error": "model returned an empty completion"}) + "\n")
            failed = rfe.load_failed_ledger(str(ledger))
        rows, long_rows = rfe.run_scenario(sc, "m", ["summary", "raw"], 0, False, "v1-frozen-2026-06",
                                           failed=failed)
        ai = {r["representation"]: r for r in long_rows if r["judge"] == "ai"}
        check("--skip-failed: the recorded key is not called again", calls == ["raw"])
        check("--skip-failed: it is written as an error row with the recorded error kind and no verdict",
              ai["summary"]["error_kind"] == "empty_completion" and ai["summary"]["verdict"] == ""
              and ai["summary"]["correct"] == "")
        check("--skip-failed: no hybrid is derived from a primary call that failed",
              not [r for r in long_rows if r["judge"] == "hybrid"])
        check("--skip-failed: the other representation is judged as usual",
              ai["raw"]["verdict"] == "PASS" and ai["raw"]["error_kind"] == "")
        calls.clear()
        rows2, long2 = rfe.run_scenario(sc, "m", ["summary", "raw"], 0, False, "v1-frozen-2026-06")
        check("without the option every key is called and three hybrids are derived",
              calls == ["summary", "raw"] and len([r for r in long2 if r["judge"] == "hybrid"]) == 3)
    finally:
        judge_clients.run_default_judge, judge_clients.judge_ai = saved

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures[:40]:
            print(f"  - {f}")
        return 1
    print("\nrelease 1.2.0 control-dataset and harness-option checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
