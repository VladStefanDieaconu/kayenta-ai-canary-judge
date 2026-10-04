#!/usr/bin/env python3
"""The bring-your-own-data path must be true where the article relies on it.

Contribution (2) and the Conclusions offer this testbed as the way to repeat the
evaluation on an organisation's own telemetry. That sentence is only honest if
six things hold, and each of them is a claim about this code rather than about
its documentation, so each is checked here by running it:

  * a window read from a file reaches the judges as the same thing a generated
    window reaches them as -- the whole seeded dataset is exported, re-imported,
    and the `metricSetPairList` and canary config are compared field by field
    over all 180 scenarios;
  * the seeded path did not move while that entry point was added -- v1.0.0's
    `tools/eval_dataset.py` is read back out of git at PRISTINE_REV, the commit
    the Data Availability Statement names, run beside the current one, and the
    two are compared over both datasets, with the one difference 1.1.0 makes on
    purpose asserted rather than waived: the canary configuration's `name` and
    `description` no longer carry the family and the ground-truth label, and
    must not;
  * a file that cannot be judged honestly is refused with the file, the line and
    the column, not repaired;
  * a window carrying no label is judged and written but enters no denominator,
    and the user's own window and metric names do not reach the prompt;
  * the study's published tables are still the study's -- the seeded run's
    roc_pr_auc.csv is the five configurations the article reports, in the seven
    columns it has always had, only an own-data run derives its own list, and a
    manifest left behind by some other run does not make a run one;
  * a run that finishes writes its reports -- an `--out` neither figure runner
    can reach costs the figures and nothing else.

`roundtrip_scenario_fields` is deliberately stricter than the contract: it
compares *every* Scenario and MetricSpec field, including the ones a supplied
window cannot carry, and reports each divergence. Its failures are the record of
what an export does not preserve, which is a thing the author should be able to
read off rather than rediscover.

No stack, no model, no network: every judge below is this repository's own pure
functions, and the only subprocess is `git show`.

Usage:
  python tools/test_own_data_contract.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import eval_dataset as ed  # noqa: E402
import own_data  # noqa: E402
import roc_pr_curves as roc  # noqa: E402
import run_experiment as exp  # noqa: E402

MASTER_SEED = ed.DEFAULT_MASTER_SEED
N_PER_FAMILY = 20                      # the article's dataset: 180 scenarios
STAT_JUDGE = {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}}
REMOTE_JUDGE = {"name": "RemoteJudge-v1.0", "judgeConfigurations": {"mode": "summary"}}
FIXED_BASE_MILLIS = 1_760_000_000_000  # so nothing below depends on the clock

# The v1.0.0 baseline `seeded_path_unchanged` reads its pre-change module from.
# A commit, never a branch ref: `a3707f5` is the archived v1.0.0 commit the
# article's Data Availability Statement names as tag v1.0.0, so this check
# verifies the deposited claim literally. `main` cannot be the baseline for a
# release that `main` is about to become -- the moment the tag is pushed the
# check would be comparing 1.1.0 with itself, and the two fields it asserts a
# difference over would be reported as unchanged, which is the inverse of what
# it exists to say. `a3707f5` is an ancestor of `main` before and after the
# merge, so it resolves in any fresh clone.
PRISTINE_REV = "a3707f5"

# What an export is known not to carry back, and why. These are asserted, not
# annotated: `roundtrip_scenario_fields` fails if a field outside this set
# diverges AND if a field inside it stops diverging, so the set is a statement
# about the format that goes stale loudly.
#
#   Scenario.gid     a supplied window has no place in the generated global
#                    index; the loader assigns it one from OWN_DATA_GID_BASE.
#   Scenario.seed    there is no master seed behind someone else's telemetry,
#                    and 0 is the loader saying so rather than a seed.
#   Scenario.params  the generator's arguments, replaced by the loader's
#                    provenance record (source name, window grid, per-side
#                    sample shapes) -- which is what params means for a window
#                    that was measured rather than generated.
#   MetricSpec.base  the user's own metric column, which for a re-imported
#                    export is the exported (gid-suffixed) name.
#
# None of the four reaches build_pairs() or build_canary_config(), which is what
# `roundtrip_judge_surface` below establishes rather than assumes.
KNOWN_LOSSY = {"Scenario.gid", "Scenario.seed", "Scenario.params", "MetricSpec.base"}

# Fields a round trip must preserve exactly, because a judge sees them.
SCENARIO_KEPT = ("id", "family", "truth")
METRIC_KEPT = ("name", "group", "unit", "direction", "truth")

# The two canary-configuration fields 1.1.0 changes on purpose, and the reason
# they are set aside from the identity comparisons below. v1.0.0 wrote the
# scenario id, its family and its ground-truth label into the config Kayenta
# stores and the judge service logs verbatim:
#
#   name         eval-no_change_000
#   description  eval dataset scenario no_change_000 (family=no_change, truth=PASS)
#
# From v1.1.0 both carry the window's opaque handle and nothing else --
# `eval-g0000`, `eval dataset scenario g0000`. Neither field is read by any
# judge and no verdict has ever depended on one, but a ground-truth label has no
# place in a request, and anyone running this testbed on their own telemetry
# would otherwise inherit the habit. Removing it is itself a leakage fix, so
# these two are *expected* to differ -- between v1.0.0 (PRISTINE_REV above) and
# this tree, and across an export and a re-import, which draws the window a
# fresh handle out of the own-data gid namespace.
#
# Set aside is not ignored. Like KNOWN_LOSSY above, this is asserted rather than
# annotated: every check that excludes these two also asserts, through
# `_handle_fields_changed`, that they differ on *every* scenario and that
# neither string carries a family or a label. A change that quietly put the
# label back into the configuration name fails here.
CONFIG_HANDLE_FIELDS = ("name", "description")


def _without_handle(config: Dict[str, Any]) -> Dict[str, Any]:
    """The canary config minus CONFIG_HANDLE_FIELDS: all of it a judge reads."""
    return {k: v for k, v in config.items() if k not in CONFIG_HANDLE_FIELDS}


def _handle_fields_changed(where: str, old: Dict[str, Any], new: Dict[str, Any],
                           family: str, truth: str) -> List[str]:
    """The other half of setting the two aside, so the exclusion cannot go stale.

    `old` is the labelled string -- v1.0.0's, or the one that went out through
    the exporter -- and `new` is the one this tree produces. The release's claim
    is that the second replaced the first and carries neither the family nor the
    label, so both halves are checked: unchanged is a failure, and so is a
    family or a truth value reappearing inside either string.
    """
    out: List[str] = []
    for field in CONFIG_HANDLE_FIELDS:
        if old[field] == new[field]:
            out.append(f"{where}: canary config {field} is unchanged at {new[field]!r}; "
                       f"1.1.0 replaces the labelled string with an opaque handle, so "
                       f"this field has to differ")
        for value, what in ((family, "family"), (truth, "ground-truth label")):
            if value and value in new[field]:
                out.append(f"{where}: canary config {field} is {new[field]!r}, which "
                           f"carries the {what} {value!r}; 1.1.0 takes it out")
    return out


# 1. Round trip over the whole dataset.
def _export_and_reload(tmp: Path) -> Tuple[List[ed.Scenario], List[ed.Scenario], Dict[str, Any]]:
    path = tmp / "seeded-180.csv"
    original = ed.build_scenarios(MASTER_SEED, N_PER_FAMILY)
    own_data.export(original, str(path), base_millis=own_data.EXPORT_BASE_MILLIS,
                    step_seconds=ed.STEP_SECONDS)
    # --keep-ids, so a reloaded window can be matched to the one it came from at
    # all. The renaming path is what `leakage_protection` below exercises.
    loaded, manifest = own_data.load(str(path), anonymise=False)
    return original, loaded, manifest


def _order_kept(declared: ed.Scenario, target: ed.Scenario) -> List[str]:
    """The round trip has to give back the metric order it was handed.

    It used not to. The exporter wrote a window's metrics sorted by name and the
    loader read them back sorted by name, and this file compared against the
    source scenario re-sorted the same way -- so a reordering was not a failure
    here, it was the fixture. `cross_metric_marginal` is the one family it bites:
    the generator declares latency, memory, errors and sorting the base names
    gives latency, errors, memory.

    Metric order is not decoration. It is the order the pairs reach the prompt
    in, and `eval_dataset._group_weights` hands the leftover 0.01 to the group of
    the last metric, so a round trip that reorders a window judges it under a
    different classifier configuration. `metric_order` carries it now, so the
    comparison is made in sequence with nothing canonicalised, and this is the
    assertion that would have caught the original defect.
    """
    want = [m.name for m in declared.metrics]
    got = [m.name for m in target.metrics]
    if want == got:
        return []
    if sorted(want) != sorted(got):
        return [f"{declared.id}: metrics out {want} came back {got}, which is not a "
                f"reordering of the same metrics"]
    return [f"{declared.id}: metric order {want} came back {got}; the file declares the "
            f"order in `metric_order` and the loader must honour it"]


def roundtrip_judge_surface(tmp: Path) -> List[str]:
    """The pairs and the canary config, over all 180, exported and read back.

    These two structures are everything a judge consumes: Kayenta's
    NetflixACAJudge reads the pairs and the per-metric analysis block, and every
    AI representation is built from the pairs alone. If they survive, a window
    from a file is judged as the same evidence as a generated one, which is the
    sentence the article wants to be able to write.
    """
    failures: List[str] = []
    original, loaded, manifest = _export_and_reload(tmp)

    if len(loaded) != len(original):
        return [f"exported {len(original)} scenarios, loaded {len(loaded)} back"]
    if manifest["n_labelled"] != len(original):
        failures.append(f"{manifest['n_labelled']} of {len(original)} windows came back labelled")
    if manifest["n_metrics"] != sum(len(s.metrics) for s in original):
        failures.append(f"manifest counts {manifest['n_metrics']} metric series, "
                        f"the dataset has {sum(len(s.metrics) for s in original)}")

    by_id = {s.id: s for s in loaded}
    for declared in original:
        target = by_id.get(declared.id)
        if target is None:
            failures.append(f"{declared.id}: did not come back")
            continue
        # The series have to be generated from the scenario as declared: a
        # generated metric's RNG is seeded from its *index* in scenario.metrics
        # (eval_dataset.series_for_scenario), so regenerating them for a
        # reordered copy would hand each metric another metric's numbers.
        out_series = ed.series_for_scenario(declared)
        source = declared
        failures.extend(_order_kept(declared, target))
        for field in SCENARIO_KEPT:
            if getattr(source, field) != getattr(target, field):
                failures.append(f"{source.id}: Scenario.{field} "
                                f"{getattr(source, field)!r} -> {getattr(target, field)!r}")
        if len(source.metrics) != len(target.metrics):
            failures.append(f"{source.id}: {len(source.metrics)} metric(s) out, "
                            f"{len(target.metrics)} back")
            continue
        for a, b in zip(source.metrics, target.metrics):
            for field in METRIC_KEPT:
                if getattr(a, field) != getattr(b, field):
                    failures.append(f"{source.id}/{a.name}: MetricSpec.{field} "
                                    f"{getattr(a, field)!r} -> {getattr(b, field)!r}")

        # The values themselves, compared as floats rather than as text: the
        # exporter writes repr(), which round-trips exactly, so anything but
        # equality here is a real loss of precision.
        back_series = ed.series_for_scenario(target)
        for name, sides in out_series.items():
            if name not in back_series:
                failures.append(f"{source.id}/{name}: series did not come back")
                continue
            for side in ("control", "experiment"):
                want, got = sides[side], back_series[name][side]
                if want == got:
                    continue
                if len(want) != len(got):
                    failures.append(f"{source.id}/{name}/{side}: {len(want)} sample(s) out, "
                                    f"{len(got)} back")
                    continue
                i, x, y = next((i, x, y) for i, (x, y) in enumerate(zip(want, got)) if x != y)
                failures.append(f"{source.id}/{name}/{side}[{i}]: {x!r} -> {y!r}")

        # What Kayenta and the AI judges are actually handed.
        want_pairs = ed.build_pairs(source, out_series, own_data.EXPORT_BASE_MILLIS,
                                    ed.STEP_SECONDS * 1000)
        got_pairs = ed.build_pairs(target, back_series, target.base_millis,
                                   target.step_seconds * 1000)
        failures.extend(_diff(f"{source.id}: metricSetPairList", want_pairs, got_pairs))
        for judge, label in ((STAT_JUDGE, "statistical"), (REMOTE_JUDGE, "remote")):
            # Every key but `name` and `description`, which name the window's
            # handle rather than describe its evidence and which a re-import
            # necessarily reassigns -- CONFIG_HANDLE_FIELDS above. That they
            # changed, and that neither carries the family or the label, is
            # asserted just below rather than left unsaid.
            want_config = ed.build_canary_config(source, judge)
            got_config = ed.build_canary_config(target, judge)
            failures.extend(_diff(f"{source.id}: canary config ({label})",
                                  _without_handle(want_config), _without_handle(got_config)))
            failures.extend(_handle_fields_changed(f"{source.id} ({label})", want_config,
                                                   got_config, source.family, source.truth))
    return failures


def _diff(where: str, want: Any, got: Any, path: str = "") -> List[str]:
    """The first few places two structures differ, located by key and index."""
    if isinstance(want, dict) and isinstance(got, dict):
        out: List[str] = []
        for key in sorted(set(want) | set(got)):
            if key not in want or key not in got:
                out.append(f"{where}{path}.{key}: "
                           f"{'absent after reload' if key in want else 'appeared after reload'}")
            else:
                out.extend(_diff(where, want[key], got[key], f"{path}.{key}"))
        return out[:5]
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [f"{where}{path}: {len(want)} item(s) -> {len(got)}"]
        out = []
        for i, (a, b) in enumerate(zip(want, got)):
            out.extend(_diff(where, a, b, f"{path}[{i}]"))
        return out[:5]
    return [] if want == got else [f"{where}{path}: {want!r} -> {got!r}"]


def roundtrip_scenario_fields(tmp: Path) -> List[str]:
    """Every Scenario and MetricSpec field, strictly, over all 180.

    Stricter than the contract on purpose: this is the inventory of what an
    export does not carry, and it is an assertion rather than a report. Four
    fields -- KNOWN_LOSSY above, each with the reason it cannot survive -- are
    *expected* to diverge, so this check passes only when exactly those four do.
    A fifth field appearing is a new loss and fails here; one of the four
    ceasing to diverge means the format now carries it and the set is out of
    date, which fails here too.

    Writing it the other way round -- listing the four as failures annotated
    "known" -- made the check red on a tree where nothing was wrong, which is a
    check that asserts something false about its own subject. What the article
    rests on is `roundtrip_judge_surface`: the metricSetPairList and the canary
    config are identical. The metadata below is provenance, and none of it
    reaches a judge.
    """
    original, loaded, _ = _export_and_reload(tmp)
    by_id = {s.id: s for s in loaded}
    seen: Dict[str, Tuple[int, str]] = {}

    for declared in original:
        target = by_id.get(declared.id)
        if target is None:
            continue
        source = declared               # compared in sequence: `metric_order` carries it
        for field in ("id", "family", "gid", "seed", "truth", "params"):
            a, b = getattr(source, field), getattr(target, field)
            if a != b:
                key = f"Scenario.{field}"
                count, _ = seen.get(key, (0, ""))
                seen[key] = (count + 1, f"{source.id}: {a!r} -> {b!r}")
        for ms, mt in zip(source.metrics, target.metrics):
            for field in ("name", "base", "kind", "group", "unit", "direction", "truth"):
                a, b = getattr(ms, field), getattr(mt, field)
                if a != b:
                    key = f"MetricSpec.{field}"
                    count, _ = seen.get(key, (0, ""))
                    seen[key] = (count + 1, f"{source.id}/{ms.name}: {a!r} -> {b!r}")

    failures: List[str] = []
    for key in sorted(set(seen) - KNOWN_LOSSY):
        count, example = seen[key]
        failures.append(f"{key} differs on {count} scenario(s) and is not a known loss; "
                        f"e.g. {example}")
    for key in sorted(KNOWN_LOSSY - set(seen)):
        failures.append(f"{key} is listed in KNOWN_LOSSY but survived the round trip; "
                        f"the export got better and the set should shrink")
    return failures


# 2. The seeded path did not move.
def _pristine_module(tmp: Path) -> Tuple[Optional[Any], str]:
    """`tools/eval_dataset.py` as v1.0.0 has it, imported under its own name.

    Read out of git rather than out of a file somebody left in a temp directory,
    so this check does not depend on an artefact of the session that made the
    change; and read at PRISTINE_REV -- the commit -- rather than at a branch
    ref, because the branch this release is merged into stops being 1.0.0 the
    moment it is.
    """
    rev = f"{PRISTINE_REV}:tools/eval_dataset.py"
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), "show", rev],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"git show {rev} failed: {proc.stderr.strip()[:200]}"
    source = tmp / "eval_dataset_v100.py"
    source.write_text(proc.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("eval_dataset_v100", source)
    if spec is None or spec.loader is None:
        return None, "could not load the pre-change eval_dataset.py"
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: @dataclass resolves a field's annotation through
    # sys.modules[cls.__module__], and an unregistered module makes that None.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, ""


def _seeded_digest(module: Any) -> Tuple[Dict[str, str], str, Dict[str, Dict[str, Any]]]:
    """Per-scenario digests of everything the seeded path produces, plus one for
    the module's constants. Keyed by scenario id so a difference has a name.

    The canary config goes into the digest without `name` and `description`
    (CONFIG_HANDLE_FIELDS): those two are the one thing 1.1.0 changes here on
    purpose, so digesting them would make this check assert the opposite of what
    the CHANGELOG says. They are returned alongside instead, and
    `seeded_path_unchanged` asserts over them directly.
    """
    digests: Dict[str, str] = {}
    handles: Dict[str, Dict[str, Any]] = {}
    for dataset, n in (("original-180", N_PER_FAMILY), ("generalisation-60", N_PER_FAMILY)):
        scenarios = module.build_scenarios(MASTER_SEED, n, dataset)
        for s in scenarios:
            series = module.series_for_scenario(s)
            statistical = module.build_canary_config(s, STAT_JUDGE)
            remote = module.build_canary_config(s, REMOTE_JUDGE)
            blob = json.dumps({
                "id": s.id, "family": s.family, "gid": s.gid, "seed": s.seed,
                "truth": s.truth, "params": s.params,
                "metrics": [{"name": m.name, "base": m.base, "kind": m.kind,
                             "group": m.group, "unit": m.unit,
                             "direction": m.direction, "truth": m.truth}
                            for m in s.metrics],
                "series": {k: {side: [repr(x) for x in v[side]]
                               for side in ("control", "experiment")}
                           for k, v in series.items()},
                "statistical": _without_handle(statistical),
                "remote": _without_handle(remote),
                "pairs": module.build_pairs(s, series, FIXED_BASE_MILLIS),
            }, sort_keys=True)
            digests[f"{dataset}/{s.id}"] = hashlib.sha256(blob.encode()).hexdigest()
            handles[f"{dataset}/{s.id}"] = {"statistical": statistical, "remote": remote,
                                            "family": s.family, "truth": s.truth}
        manifest = module.build_manifest(MASTER_SEED, n, scenarios, FIXED_BASE_MILLIS)
        manifest.pop("generated_at", None)
        digests[f"{dataset}/_manifest"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    families = list(module.FAMILIES) + list(module.GENERALISATION_FAMILIES)
    constants = json.dumps({
        "STAT_CANARY_TEMPLATE": module.STAT_CANARY_TEMPLATE,
        "SCORE_THRESHOLDS": module.SCORE_THRESHOLDS,
        "STEP_SECONDS": module.STEP_SECONDS, "N_SAMPLES": module.N_SAMPLES,
        "WINDOW_MINS": module.WINDOW_MINS, "FAMILIES": module.FAMILIES,
        "GENERALISATION_FAMILIES": module.GENERALISATION_FAMILIES,
        "DEFAULT_MASTER_SEED": module.DEFAULT_MASTER_SEED,
        "DEFAULT_N_PER_FAMILY": module.DEFAULT_N_PER_FAMILY,
        "family_order": {f: module.family_order(f) for f in families},
        "family_dataset": {f: module.family_dataset(f) for f in families},
        "aligned_base_millis": module.aligned_base_millis(1_760_000_000.0),
    }, sort_keys=True)
    whole = hashlib.sha256("".join(digests[k] for k in sorted(digests)).encode()
                           + constants.encode()).hexdigest()
    digests["_constants"] = hashlib.sha256(constants.encode()).hexdigest()
    return digests, whole, handles


def seeded_path_unchanged(tmp: Path) -> List[str]:
    """Both seeded datasets, every scenario, against v1.0.0's module.

    The property, stated so the assertion can be read against it: the generated
    series, the metric names, the metricSetPairList, the ground-truth labels,
    the manifest and every canary-configuration field other than `name` and
    `description` are identical to v1.0.0 (PRISTINE_REV), over all 240 scenarios
    of both seeded datasets and both manifests.

    Those two fields are the exception, and the exception is asserted rather
    than waived: 1.1.0 takes the family and the label out of them
    (CONFIG_HANDLE_FIELDS), so they must differ from v1.0.0's on every scenario
    and the new strings must carry neither. Asserting identity over them as well
    would be asserting something this release makes false by design, which is
    worse than not checking at all; asserting nothing over them would let a
    later change put the label back without a word.
    """
    pristine, error = _pristine_module(tmp)
    if pristine is None:
        return [error]
    before, before_all, before_handles = _seeded_digest(pristine)
    after, after_all, after_handles = _seeded_digest(ed)

    failures: List[str] = []
    if before_all != after_all:
        failures.append(f"the seeded path changed: {len(before)} digest(s) before, "
                        f"{len(after)} after; whole-dataset digest "
                        f"{before_all[:16]} -> {after_all[:16]}")
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                failures.append(f"  {key}: {before.get(key, '(absent)')[:16]} -> "
                                f"{after.get(key, '(absent)')[:16]}")
        return failures[:12]

    # The intended difference, over the same 240 scenarios and both judges.
    # Reported as a count of the comparisons that went wrong with the first few
    # lines behind it, rather than as several hundred lines.
    lines: List[str] = []
    bad = 0
    for key in sorted(after_handles):
        old, new = before_handles.get(key), after_handles[key]
        if old is None:
            failures.append(f"{key}: present now, absent at v1.0.0 ({PRISTINE_REV})")
            continue
        for label in ("statistical", "remote"):
            said = _handle_fields_changed(f"{key} ({label})", old[label], new[label],
                                          new["family"], new["truth"])
            if said:
                bad += 1
                lines.extend(said)
    if lines:
        failures.append(f"the canary config's name and description are not what 1.1.0 "
                        f"makes them on {bad} of {2 * len(after_handles)} comparison(s); "
                        f"the family and the truth label have to be gone from both, and "
                        f"they are not")
        failures.extend(f"  {line}" for line in lines[:8])
    return failures[:12]


# 3. Validation: one broken file per rule.
HEADER = "scenario,metric,side,timestamp,value,kind,direction,label\n"


def _rows(scenario: str = "win-a", metric: str = "checkout_latency", n: int = 12,
          label: str = "FAIL", sides: Tuple[str, ...] = ("control", "canary"),
          kind: str = "latency", direction: str = "increase") -> str:
    return "".join(
        f"{scenario},{metric},{side},2026-06-21T09:{i:02d}:00Z,{100 + i},{kind},{direction},{label}\n"
        for side in sides for i in range(n))


def _write(tmp: Path, name: str, body: str, header: str = HEADER) -> Path:
    path = tmp / name
    path.write_text(header + body, encoding="utf-8")
    return path


def _refusal(path: Path) -> Tuple[Optional[List[str]], str]:
    """The messages a load produced, or a note that it did not refuse at all."""
    try:
        scenarios, _ = own_data.load(str(path))
    except own_data.OwnDataError as e:
        return e.messages, ""
    return None, f"loaded {len(scenarios)} window(s) instead of refusing"


def validation_rules(tmp: Path) -> List[str]:
    """Every rule in the spec, each broken on purpose.

    What is asserted is not that a load fails -- a bare exception is not a usable
    error -- but that the message names the file, the line and the column, which
    is the difference between fixing an export and guessing at it.
    """
    failures: List[str] = []

    # The control: the same generator, unbroken, must load. Without it every
    # assertion below could be passing because the fixture itself is invalid.
    ok_path = _write(tmp, "valid.csv", _rows())
    try:
        scenarios, manifest = own_data.load(str(ok_path))
        if len(scenarios) != 1 or manifest["n_labelled"] != 1:
            failures.append(f"control: a valid file gave {len(scenarios)} window(s), "
                            f"{manifest['n_labelled']} labelled")
    except own_data.OwnDataError as e:
        failures.append(f"control: a valid file was refused: {e.messages[0]}")

    cases: List[Tuple[str, Path, Tuple[str, ...]]] = [
        # (rule, file, substrings every one of which must appear in one message)
        ("missing required column",
         _write(tmp, "missing_column.csv", _rows(),
                "scenario,metric,side,timestamp,value,kind,directoin,label\n"),
         ("missing_column.csv:1:", "direction", "directoin")),
        ("non-numeric value",
         _write(tmp, "non_numeric.csv",
                _rows().replace("2026-06-21T09:05:00Z,105", "2026-06-21T09:05:00Z,not_a_number")),
         ("non_numeric.csv:7:", "'value'", "not_a_number")),
        ("unequal control and canary lengths",
         _write(tmp, "unequal.csv",
                _rows(sides=("control",)) + _rows(sides=("canary",), n=4)),
         ("unequal.csv:", "'win-a'", "'checkout_latency'", "control", "canary")),
        ("unknown direction",
         _write(tmp, "bad_direction.csv", _rows(direction="sideways")),
         ("bad_direction.csv:2:", "'direction'", "'sideways'", "increase|decrease|either")),
        ("unknown label",
         _write(tmp, "bad_label.csv", _rows(label="ROLLEDBACK")),
         ("bad_label.csv:2:", "'label'", "'ROLLEDBACK'", "PASS|FAIL")),
        ("duplicate (scenario, metric, index)",
         _write(tmp, "duplicate.csv", _rows()
                + "win-a,checkout_latency,control,2026-06-21T09:05:00Z,999,latency,increase,FAIL\n"),
         ("duplicate.csv:26:", "'win-a'", "'checkout_latency'", "'control'",
          "2026-06-21T09:05:00Z", "duplicate.csv:7")),
        ("empty file",
         _write(tmp, "empty.csv", "", ""),
         ("empty.csv:", "header")),
        ("header but no rows",
         _write(tmp, "header_only.csv", ""),
         ("header_only.csv:", "no data")),
    ]

    for rule, path, needles in cases:
        messages, note = _refusal(path)
        if messages is None:
            failures.append(f"{rule}: {note}")
            continue
        if not any(all(needle in message for needle in needles) for message in messages):
            failures.append(f"{rule}: no message carries all of {list(needles)}; "
                            f"{len(messages)} message(s), first is {messages[0]!r}")
    return failures


# 4. Unlabelled windows are judged and not scored.
def _unlabelled_file(tmp: Path, name: str, labelled: int, total: int) -> Path:
    body = "".join(_rows(scenario=f"win-{i:02d}", label="FAIL" if i < labelled else "")
                   for i in range(total))
    return _write(tmp, name, body)


def _fake_rows(scenarios: List[ed.Scenario], verdict: str = "PASS") -> List[Dict[str, Any]]:
    """Result rows through run_experiment's own `_row`, so the `correct` field is
    the one the runner writes rather than one this test made up."""
    return [exp._row(s, "statistical", "", "-", verdict, 100.0, 0.5, "")
            for s in scenarios]


def _write_outputs(tmp: Path, rows, manifest, name: str) -> Path:
    out = tmp / name
    out.mkdir(parents=True, exist_ok=True)
    present = [{"alias": "stub-llm", "modality": "text", "tag": "stub"}]
    det = {"scenario": "own_0000", "model": "stub-llm", "run1": "PASS",
           "run2": "PASS", "identical": True}
    saved_results, saved_ailog = exp.RESULTS_DIR, exp.AILOG_DIR
    exp.RESULTS_DIR, exp.AILOG_DIR = out, out / "ai-logs"
    try:
        exp.write_outputs(rows, present, [], det, manifest, argparse.Namespace(), 1,
                          True, 60.0)
    finally:
        exp.RESULTS_DIR, exp.AILOG_DIR = saved_results, saved_ailog
    return out


def unlabelled_mode(tmp: Path) -> List[str]:
    """A file with no labels loads, is judgeable, and produces no accuracy."""
    failures: List[str] = []
    path = _unlabelled_file(tmp, "unlabelled.csv", labelled=0, total=4)
    scenarios, manifest = own_data.load(str(path))

    if len(scenarios) != 4:
        failures.append(f"an unlabelled file gave {len(scenarios)} window(s), expected 4")
    if manifest["n_labelled"] != 0:
        failures.append(f"manifest claims {manifest['n_labelled']} labelled window(s)")
    for s in scenarios:
        if s.truth != "":
            failures.append(f"{s.id}: unlabelled window came back with truth={s.truth!r}")
        for m in s.metrics:
            if m.truth != "":
                failures.append(f"{s.id}/{m.name}: metric acquired truth={m.truth!r}")

    # The judges can be built for it: pairs and a full canary config, the two
    # things a judge is handed.
    for s in scenarios:
        pairs = ed.build_pairs(s, ed.series_for_scenario(s), s.base_millis,
                               s.step_seconds * 1000)
        if len(pairs) != len(s.metrics) or not pairs[0]["values"]["experiment"]:
            failures.append(f"{s.id}: no judgeable pairs were built")
        config = ed.build_canary_config(s, STAT_JUDGE)
        if not config["metrics"] or "canary" not in config["metrics"][0]["analysisConfigurations"]:
            failures.append(f"{s.id}: no statistical configuration was built")

    # And the scoreboard carries no numbers rather than zeroes.
    rows = _fake_rows(scenarios)
    for r in rows:
        if r["correct"] != "":
            failures.append(f"{r['scenario']}: `correct` is {r['correct']!r} for an "
                            f"unlabelled window; it must be empty, not 0")
    out = _write_outputs(tmp, rows, manifest, "out-unlabelled")
    summary = (out / "summary.csv").read_text()
    if "nothing to score" not in summary:
        failures.append(f"summary.csv for an unlabelled run does not say so: {summary!r}")
    if "accuracy" in summary:
        failures.append("summary.csv for an unlabelled run carries an accuracy column")
    results_md = (out / "results.md").read_text()
    if "no accuracy" not in results_md.replace("there is no\naccuracy", "no accuracy"):
        failures.append("results.md for an unlabelled run does not say there is no accuracy")
    raw = (out / "results_raw.csv").read_text().splitlines()
    if len(raw) != len(rows) + 1:
        failures.append(f"results_raw.csv holds {len(raw) - 1} row(s) for {len(rows)} judgement(s); "
                        f"an unlabelled verdict must still be written")

    # Partially labelled: only the labelled windows may reach a denominator.
    part_path = _unlabelled_file(tmp, "partial.csv", labelled=1, total=4)
    part_scenarios, part_manifest = own_data.load(str(part_path))
    part_rows = _fake_rows(part_scenarios)
    part_out = _write_outputs(tmp, part_rows, part_manifest, "out-partial")
    part_summary = [line.split(",") for line in
                    (part_out / "summary.csv").read_text().strip().splitlines()]
    n_column = part_summary[0].index("n")
    counted = [int(row[n_column]) for row in part_summary[1:]]
    if counted != [1]:
        failures.append(f"a run with 1 labelled window of 4 scored n={counted}, expected [1]")
    return failures


def accuracy_guard(tmp: Path) -> List[str]:
    """An unlabelled row must not be able to become an accuracy.

    `write_outputs` filters to the labelled rows before it aggregates, and that
    filter is the whole protection. This checks the layer underneath it as well,
    because `confusion()` and `aggregate()` are module-level and a later caller
    will reach for them: a row whose truth is "" must not be silently counted as
    a PASS-truth row, which would make a judge that says PASS to everything look
    perfect on data nobody labelled.
    """
    failures: List[str] = []
    path = _unlabelled_file(tmp, "guard.csv", labelled=0, total=4)
    scenarios, _ = own_data.load(str(path))
    rows = _fake_rows(scenarios, verdict="PASS")

    try:
        agg = exp.aggregate(rows)
    except Exception:  # noqa: BLE001 -- refusing loudly is the desired behaviour
        return failures
    reported = {key: value["accuracy"] for key, value in agg.items()}
    if reported:
        failures.append(
            f"aggregate() on {len(rows)} unlabelled row(s) returned an accuracy "
            f"{reported} instead of refusing; confusion() reads truth != 'FAIL' as a "
            f"PASS-truth window, so an all-PASS judge scores 1.0 on data carrying no "
            f"labels. write_outputs() filters these rows out first, so no shipped "
            f"command reaches this, but the guard is the caller's, not the function's")
    return failures


# 5. The user's own names must not reach the prompt.
LEAKY_SCENARIO = "2026-03-14-rollback-FAIL"
LEAKY_METRIC = "checkout_latency_REGRESSION"
LEAKY_TOKENS = ("REGRESSION", "regression", "rollback", "checkout", "2026-03-14")


def leakage_protection(tmp: Path) -> List[str]:
    """A window that names its own answer must not hand it to the model.

    Asserted on the two structures that leave this process: the pairs, which are
    the only thing a representation is built from, and the canary config. The
    manifest is checked in the opposite direction -- the real names have to
    survive *somewhere*, or the run is not traceable back to the user's data.
    """
    failures: List[str] = []
    path = _write(tmp, "leaky.csv", _rows(scenario=LEAKY_SCENARIO, metric=LEAKY_METRIC))
    scenarios, manifest = own_data.load(str(path))   # renaming is the default
    s = scenarios[0]

    pairs = ed.build_pairs(s, ed.series_for_scenario(s), s.base_millis, s.step_seconds * 1000)
    config = ed.build_canary_config(s, STAT_JUDGE)
    prompt_visible = json.dumps(pairs)
    config_text = json.dumps(config)

    for token in LEAKY_TOKENS:
        if token in prompt_visible:
            failures.append(f"{token!r} reaches build_pairs() output, which is what the "
                            f"representation the model reads is built from")
        if token in config_text:
            failures.append(f"{token!r} reaches the canary config")
    if s.id == LEAKY_SCENARIO:
        failures.append(f"the window kept its own name {s.id!r}")
    for m in s.metrics:
        if m.name == LEAKY_METRIC:
            failures.append(f"the metric kept its own name {m.name!r}")
        if not m.name.endswith(f"_g{s.gid:04d}"):
            failures.append(f"{m.name!r} does not carry the opaque gid suffix")

    # The pair id is derived from the scenario id and the metric name, so it is
    # only opaque if both of those are.
    for pair in pairs:
        for token in LEAKY_TOKENS:
            if token in pair["id"] or token in json.dumps(pair["attributes"]):
                failures.append(f"{token!r} survives in a pair id or PromQL attribute")

    provenance = manifest["scenarios"][0]["params"]["own_data"]
    if provenance["scenario"] != LEAKY_SCENARIO:
        failures.append("the manifest lost the window's real name; the run cannot be "
                        "traced back to the user's data")
    if LEAKY_METRIC not in json.dumps(provenance["metrics"]):
        failures.append("the manifest lost the metric's real name")
    return failures


# 5b. R2-3: the opaque suffix must not encode the sort order of the window ids.
#
# The defect this replaced assigned `gid = OWN_DATA_GID_BASE + index` over
# `sorted(windows)`, so the suffix was monotone in the alphabetical order of the
# practitioner's own window names. A practitioner whose window ids sort by
# outcome -- and plenty do, because the outcome is often in the name -- therefore
# got an opaque name that ranks the windows by their answer. That is the same
# leak the article measures on its own metric names (RQ2), shipped as the feature
# that is supposed to prevent it.
#
# The check below is written as a comparison against that defect rather than as
# an assertion about the fix, so it is a test that fails on the code it was
# written for: `_positional_gids` is the old rule, it is run over the same files,
# and it is *asserted to leak*. If the instrument stopped detecting the leak, the
# instrument would fail first.
LEAK_WINDOWS = 40            # per dataset; 20 PASS then 20 FAIL by name
LEAK_DATASETS = 40           # independent files, so the claim is distributional


def _outcome_sorted_file(tmp: Path, tag: str, bump: float = 0.0,
                         suffix: str = "") -> Tuple[Path, Dict[str, str]]:
    """A file whose window ids sort by outcome: every PASS name before every FAIL.

    `a-deploy-NNN-ok` and `z-deploy-NNN-rollback`, which is what a practitioner
    who names windows after what happened ends up with. `tag` varies the content
    between datasets, so each one draws its own salt.
    """
    rows = ["scenario,metric,side,timestamp,value,kind,direction,label"]
    truth: Dict[str, str] = {}
    half = LEAK_WINDOWS // 2
    for i in range(LEAK_WINDOWS):
        passing = i < half
        name = (f"a-deploy-{i:03d}-ok-{tag}" if passing
                else f"z-deploy-{i:03d}-rollback-{tag}")
        label = "PASS" if passing else "FAIL"
        truth[name] = label
        for j in range(12):
            stamp = f"2026-03-14T09:{j:02d}:00Z"
            base = 0.20 + 0.001 * i + 0.0001 * j + bump
            canary = base * (1.0 if passing else 2.0)
            rows.append(f"{name},checkout_latency,control,{stamp},{base!r},"
                        f"latency,increase,{label}")
            rows.append(f"{name},checkout_latency,canary,{stamp},{canary!r},"
                        f"latency,increase,{label}")
    path = tmp / f"outcome-sorted-{tag}{suffix}.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path, truth


def _positional_gids(names: List[str]) -> Dict[str, int]:
    """The defect, kept here as the instrument's control: gid by sort position."""
    return {name: own_data.OWN_DATA_GID_BASE + i for i, name in enumerate(sorted(names))}


def _best_single_cut(gids: List[int], labels: List[str]) -> float:
    """The best accuracy any rule of the form `gid < t -> PASS` can reach.

    Both orientations, every cut between two observed gids and the two degenerate
    ones. This is the rule Reviewer 2 ran on the study's own metric names -- one
    cut at 0060 reproduces all 405 labels -- so it is the rule the anonymiser's
    output has to defeat.
    """
    order = sorted(zip(gids, labels))
    n = len(order)
    best = 0.0
    for cut in range(n + 1):
        low = order[:cut]
        high = order[cut:]
        for pass_side, fail_side in ((low, high), (high, low)):
            right = sum(1 for _, y in pass_side if y == "PASS")
            right += sum(1 for _, y in fail_side if y == "FAIL")
            best = max(best, right / n)
    return best


def _auc(gids: List[int], labels: List[str]) -> float:
    """P(gid of a FAIL window > gid of a PASS one), ties at a half.

    0.5 is no information; 1.0 or 0.0 is a suffix that ranks the windows by their
    answer. Computed directly rather than through roc_pr_curves so this check
    does not depend on the module it is checking the study's tables with.
    """
    pos = [g for g, y in zip(gids, labels) if y == "FAIL"]
    neg = [g for g, y in zip(gids, labels) if y == "PASS"]
    if not pos or not neg:
        return 0.5
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def gid_carries_no_order(tmp: Path) -> List[str]:
    """R2-3. Window ids that sort by outcome must not produce suffixes that do.

    Three claims, all asserted over the same files:

      1. the old positional rule leaks completely -- one cut on the suffix
         reproduces every label, and the rank correlation with the label is
         perfect. This is the control: it establishes that the two statistics
         below can see the defect at all;
      2. the shipped anonymiser does not -- no cut reproduces every label in any
         of the datasets, and the mean rank correlation with the label sits
         inside the interval a suffix drawn independently of the label would;
      3. and the suffix carries nothing about the *name* either: the mean rank
         correlation between a window's place in the sorted file and its gid is
         inside the same interval. The label is one thing a name can sort by; the
         claim the format needs is the stronger one, that it sorts by none.

    The intervals are the null's own: a mean over `LEAK_DATASETS` independent
    files, four standard errors wide. Nothing here is eyeballed and nothing is a
    magic constant read off a run.
    """
    failures: List[str] = []
    n = LEAK_WINDOWS
    # SE of the Mann-Whitney statistic for n/2 vs n/2 under the null, then of a
    # mean of LEAK_DATASETS of them.
    half = n // 2
    se_one = math.sqrt((half + half + 1) / (12.0 * half * half))
    se_mean = se_one / math.sqrt(LEAK_DATASETS)
    band = 4.0 * se_mean

    label_aucs: List[float] = []
    position_aucs: List[float] = []
    leaked: List[str] = []
    control_checked = False

    for d in range(LEAK_DATASETS):
        path, truth = _outcome_sorted_file(tmp, f"{d:03d}")
        scenarios, manifest = own_data.load(str(path))
        names = [s.params["own_data"]["scenario"] for s in scenarios]
        gids = [s.gid for s in scenarios]
        labels = [truth[name] for name in names]

        if len(set(gids)) != len(gids):
            failures.append(f"dataset {d}: two windows share a gid; "
                            f"{len(gids) - len(set(gids))} collision(s) went unresolved")
        for gid in gids:
            if not own_data.OWN_DATA_GID_BASE <= gid < own_data.OWN_DATA_GID_LIMIT:
                failures.append(f"dataset {d}: gid {gid} is outside the own-data range")
                break
        if not manifest.get("gid_salt"):
            failures.append(f"dataset {d}: the manifest records no gid_salt, so the "
                            f"names this run was judged under cannot be re-derived")

        # 1. The control: the rule that was shipped, over the same windows.
        if not control_checked:
            control_checked = True
            positional = _positional_gids(names)
            old_gids = [positional[name] for name in names]
            if _best_single_cut(old_gids, labels) != 1.0:
                failures.append("the instrument is broken: the positional gid rule this "
                                "check exists to detect is not detected by it")
            if _auc(old_gids, labels) != 1.0:
                failures.append("the instrument is broken: the positional gid rule does "
                                "not rank the windows by their label under _auc")

        # 2. The fix, on the same statistics.
        cut = _best_single_cut(gids, labels)
        if cut == 1.0:
            leaked.append(f"dataset {d}: a single cut on the opaque suffix reproduces "
                          f"all {n} labels")
        label_aucs.append(_auc(gids, labels))
        # 3. The suffix against the name's own sort position, label aside.
        position = {name: i for i, name in enumerate(sorted(names))}
        first_half = ["PASS" if position[name] < half else "FAIL" for name in names]
        position_aucs.append(_auc(gids, first_half))

    failures.extend(leaked[:5])
    mean_label = sum(label_aucs) / len(label_aucs)
    mean_position = sum(position_aucs) / len(position_aucs)
    if abs(mean_label - 0.5) > band:
        failures.append(f"the opaque suffix ranks the windows by their label: mean AUC "
                        f"{mean_label:.4f} over {LEAK_DATASETS} datasets, outside "
                        f"0.5 +/- {band:.4f} (4 SE)")
    if abs(mean_position - 0.5) > band:
        failures.append(f"the opaque suffix ranks the windows by where their name sorts: "
                        f"mean AUC {mean_position:.4f} over {LEAK_DATASETS} datasets, "
                        f"outside 0.5 +/- {band:.4f} (4 SE)")

    # 4. Two datasets must not share a namespace: the same window name in two
    #    files gets different suffixes, so a suffix seen in one run says nothing
    #    about a window in another.
    # The same window names, the same shape, different numbers.
    a, _ = _outcome_sorted_file(tmp, "salt", suffix="-a")
    b_path, _ = _outcome_sorted_file(tmp, "salt", bump=0.5, suffix="-b")
    sa, ma = own_data.load(str(a))
    sb, mb = own_data.load(str(b_path))
    if [s.params["own_data"]["scenario"] for s in sa] != \
            [s.params["own_data"]["scenario"] for s in sb]:
        failures.append("the salt fixtures do not carry the same window names, so the "
                        "cross-dataset check below is not the check it claims to be")
    if ma["gid_salt"] == mb["gid_salt"]:
        failures.append("two datasets differing in a value share a gid salt, so the same "
                        "window name gets the same opaque suffix in both")
    same = sum(1 for x, y in zip([s.gid for s in sa], [s.gid for s in sb]) if x == y)
    if same == len(sa):
        failures.append("two different datasets produced an identical gid assignment")

    # 5. Determinism, which the round trip rests on: one file, two loads.
    again, _ = own_data.load(str(a))
    if [s.gid for s in sa] != [s.gid for s in again]:
        failures.append("two loads of one file assigned different gids")
    return failures


# 6. The same file, twice.
def determinism(tmp: Path) -> List[str]:
    """Two loads of one file must be the same run.

    Nothing here is seeded, so the only sources of variation are dictionary and
    file order; both are the file's own, and if either leaked through, two runs
    over one export would not be comparable with each other.
    """
    failures: List[str] = []
    body = "".join(_rows(scenario=name, metric=metric, kind=kind)
                   for name, metric, kind in (("win-b", "queue_depth", "saturation"),
                                              ("win-a", "checkout_latency", "latency"),
                                              ("win-c", "error_rate", "errors")))
    # A second metric in one window, so group weights and their rounding are
    # exercised rather than assumed.
    body += _rows(scenario="win-a", metric="cart_latency", kind="latency")
    path = _write(tmp, "twice.csv", body)

    first, first_manifest = own_data.load(str(path))
    second, second_manifest = own_data.load(str(path))

    if [s.id for s in first] != [s.id for s in second]:
        failures.append(f"window ids differ between loads: {[s.id for s in first]} vs "
                        f"{[s.id for s in second]}")
        return failures
    for a, b in zip(first, second):
        failures.extend(_diff(f"{a.id}: manifest entry", ed.scenario_to_manifest(a),
                              ed.scenario_to_manifest(b)))
        sa, sb = ed.series_for_scenario(a), ed.series_for_scenario(b)
        if sa != sb:
            failures.append(f"{a.id}: the series differ between two loads of one file")
        failures.extend(_diff(f"{a.id}: metricSetPairList",
                              ed.build_pairs(a, sa, a.base_millis, a.step_seconds * 1000),
                              ed.build_pairs(b, sb, b.base_millis, b.step_seconds * 1000)))
        failures.extend(_diff(f"{a.id}: canary config",
                              ed.build_canary_config(a, STAT_JUDGE),
                              ed.build_canary_config(b, STAT_JUDGE)))

    for key in ("n_windows", "n_labelled", "n_metrics", "families", "scenarios",
                "statistical_template", "anonymised"):
        if first_manifest[key] != second_manifest[key]:
            failures.append(f"manifest[{key!r}] differs between two loads of one file")

    # Window ids follow the user's own identifiers, not the order they appear in.
    if [s.id for s in first] != sorted(s.id for s in first):
        failures.append(f"window ids are not assigned in sorted order: {[s.id for s in first]}")
    if [s.params['own_data']['scenario'] for s in first] != ["win-a", "win-b", "win-c"]:
        failures.append("windows are not ordered by the user's own identifier")
    return failures



# 6b. The file declares the metric order, and the loader honours it.
ORDER_HEADER = ("scenario,metric,side,timestamp,value,kind,direction,label,group,"
                "metric_order\n")

# A three-metric window whose NAME order and KIND order disagree, which is the
# shape the defect needed. Declared: latency, memory, errors -- the generator's
# order for `cross_metric_marginal`. Sorted by name: errors, latency, memory.
# So the two orders put a different metric last, and `_group_weights` hands the
# leftover 0.01 to the group of the last one.
ORDER_METRICS = (
    # (metric name, kind, group, declared rank)
    ("p99_seconds", "latency", "latency", 1),
    ("rss_bytes", "memory", "resources", 2),
    ("errors_per_second", "errors", "errors", 3),
)
ORDER_BY_NAME = ["errors_per_second", "p99_seconds", "rss_bytes"]
ORDER_DECLARED = ["p99_seconds", "rss_bytes", "errors_per_second"]


def _order_rows(declare: bool) -> List[str]:
    """One row per (metric, side, sample), as a list so a caller can shuffle it."""
    out: List[str] = []
    for rank, (metric, kind, group, order) in enumerate(ORDER_METRICS, start=1):
        for side in ("control", "canary"):
            for i in range(12):
                out.append(
                    f"win-a,{metric},{side},2026-06-21T09:{i:02d}:00Z,"
                    f"{100 * rank + i},{kind},increase,FAIL,{group},"
                    f"{order if declare else ''}\n")
    return out


def _order_load(tmp: Path, name: str, rows: List[str], *,
                anonymise: bool = False) -> Tuple[ed.Scenario, Dict[str, Any]]:
    path = _write(tmp, name, "".join(rows), header=ORDER_HEADER)
    scenarios, manifest = own_data.load(str(path), anonymise=anonymise)
    return scenarios[0], manifest


def metric_order_is_declared(tmp: Path) -> List[str]:
    """The order a window's metrics reach the judge in is the file's to state.

    This is the test the original defect needed and did not have. The loader
    ordered a window's metrics by the practitioner's metric name, which for a
    window whose names and kinds disagree is not the order its owner had; and
    because `eval_dataset._group_weights` gives the leftover 0.01 to the group of
    the LAST metric, a silent reordering is a silent change to the classifier
    configuration as well as to the prompt.

    Two properties, and they have to hold together:

      * order-faithful -- a file that declares an order gets that order, whatever
        sequence its rows are in;
      * deterministic -- two loads of one file, and of that file shuffled, are one
        run, declared or not.

    A file that declares nothing keeps the old sorted-by-name rule, which is the
    only order two ends can agree on without being told.
    """
    failures: List[str] = []

    declared, manifest = _order_load(tmp, "declared.csv", _order_rows(True))
    got = [m.name for m in declared.metrics]
    if got != ORDER_DECLARED:
        failures.append(f"declared order {ORDER_DECLARED} came back {got}")
    if manifest["n_windows_with_declared_metric_order"] != 1:
        failures.append(f"manifest records "
                        f"{manifest['n_windows_with_declared_metric_order']} window(s) with a "
                        f"declared metric order, not 1")

    # The prompt. The only field of a pair that reaches the model is `name`
    # (judge-service/app/representations.py), so the pair order IS the order the
    # model reads the metrics in.
    pairs = ed.build_pairs(declared, ed.series_for_scenario(declared),
                           declared.base_millis, declared.step_seconds * 1000)
    if [p["name"] for p in pairs] != ORDER_DECLARED:
        failures.append(f"metricSetPairList order {[p['name'] for p in pairs]} is not the "
                        f"declared {ORDER_DECLARED}")

    # The group weights. Last declared metric is the errors one, so `errors`
    # takes the 33.34 and the other two take 33.33.
    weights = ed.build_canary_config(declared, STAT_JUDGE)["classifier"]["groupWeights"]
    if weights != {"latency": 33.33, "resources": 33.33, "errors": 33.34}:
        failures.append(f"declared order gives groupWeights {weights}, not "
                        f"{{'latency': 33.33, 'resources': 33.33, 'errors': 33.34}}")
    if list(weights) != ["latency", "resources", "errors"]:
        failures.append(f"groupWeights are keyed in {list(weights)}, not the declared "
                        f"['latency', 'resources', 'errors']")

    # Order-faithful under shuffling: the declaration is a column, not a row
    # sequence, and a file is the one thing a user is free to re-sort.
    rows = _order_rows(True)
    for seed in (1, 2, 3):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        other, _ = _order_load(tmp, f"declared-{seed}.csv", shuffled)
        failures.extend(_diff(f"shuffled(seed={seed}): metricSetPairList", pairs,
                              ed.build_pairs(other, ed.series_for_scenario(other),
                                             other.base_millis, other.step_seconds * 1000)))
        failures.extend(_diff(f"shuffled(seed={seed}): canary config",
                              ed.build_canary_config(declared, STAT_JUDGE),
                              ed.build_canary_config(other, STAT_JUDGE)))

    # Two loads of one file are one run.
    again, _ = _order_load(tmp, "declared.csv", _order_rows(True))
    failures.extend(_diff("second load: canary config",
                          ed.build_canary_config(declared, STAT_JUDGE),
                          ed.build_canary_config(again, STAT_JUDGE)))

    # Declaring an order must not rename anything: the gid salt is taken over the
    # values, not over the declaration. That is what lets a dataset state its
    # order while carrying names it was judged under.
    anon_declared, man_a = _order_load(tmp, "declared.csv", _order_rows(True), anonymise=True)
    anon_bare, man_b = _order_load(tmp, "bare.csv", _order_rows(False), anonymise=True)
    if man_a["gid_salt"] != man_b["gid_salt"]:
        failures.append("declaring the metric order changed the gid salt; it must be a "
                        "digest of the values, not of the declaration")
    if sorted(m.name for m in anon_declared.metrics) != sorted(m.name for m in anon_bare.metrics):
        failures.append(f"declaring the metric order renamed the series: "
                        f"{sorted(m.name for m in anon_declared.metrics)} vs "
                        f"{sorted(m.name for m in anon_bare.metrics)}")
    if [m.kind for m in anon_declared.metrics] != ["latency", "memory", "errors"]:
        failures.append(f"anonymised, the declared order came back "
                        f"{[m.kind for m in anon_declared.metrics]}, not "
                        f"['latency', 'memory', 'errors']")

    # A file that declares nothing falls back to sorting on the name, unchanged.
    bare, man_bare = _order_load(tmp, "bare.csv", _order_rows(False))
    if [m.name for m in bare.metrics] != ORDER_BY_NAME:
        failures.append(f"with no declaration the order is {[m.name for m in bare.metrics]}, "
                        f"not the sorted-by-name {ORDER_BY_NAME}")
    if man_bare["n_windows_with_declared_metric_order"] != 0:
        failures.append("a file that declares no order is recorded as declaring one")
    bare_weights = ed.build_canary_config(bare, STAT_JUDGE)["classifier"]["groupWeights"]
    if bare_weights.get("resources") != 33.34:
        failures.append(f"with no declaration the leftover 0.01 should go to the group of "
                        f"the last sorted name (resources); weights are {bare_weights}")
    if bare_weights == weights:
        failures.append("the declared and undeclared orders produce the same group "
                        "weights, so this fixture no longer exercises the defect")
    for seed in (4, 5):
        shuffled = _order_rows(False)
        random.Random(seed).shuffle(shuffled)
        other, _ = _order_load(tmp, f"bare-{seed}.csv", shuffled)
        failures.extend(_diff(f"undeclared, shuffled(seed={seed}): canary config",
                              ed.build_canary_config(bare, STAT_JUDGE),
                              ed.build_canary_config(other, STAT_JUDGE)))

    # And a declaration that is not a declaration is refused rather than
    # half-honoured: a partial one, and one that ranks two metrics the same.
    partial = [r.replace(",2\n", ",\n") for r in _order_rows(True)]
    messages, note = _refusal(_write(tmp, "partial.csv", "".join(partial), header=ORDER_HEADER))
    if messages is None:
        failures.append(f"a window declaring metric_order for two of its three metrics: {note}")
    elif not any("all of its metrics or of none" in m for m in messages):
        failures.append(f"the partial declaration was refused without saying why: {messages}")
    clash = [r.replace(",3\n", ",1\n") for r in _order_rows(True)]
    messages, note = _refusal(_write(tmp, "clash.csv", "".join(clash), header=ORDER_HEADER))
    if messages is None:
        failures.append(f"a window giving two metrics the same metric_order: {note}")
    elif not any("must be distinct" in m for m in messages):
        failures.append(f"the duplicate rank was refused without saying why: {messages}")
    text = [r.replace(",2\n", ",second\n") for r in _order_rows(True)]
    messages, note = _refusal(_write(tmp, "text.csv", "".join(text), header=ORDER_HEADER))
    if messages is None:
        failures.append(f"a metric_order that is not an integer: {note}")
    elif not any("not an" in m and "integer" in m for m in messages):
        failures.append(f"the non-integer rank was refused without saying why: {messages}")

    return failures


# 7. The published tables stay published.
def _auc_fixture(tmp: Path, name: str, own_data_run: bool) -> Path:
    """One results directory shaped like a real run, for roc_pr_curves.py.

    Three untagged frontier files, which is what the study's own archived agg/
    holds; a glob over them is what turned the published five-row table into a
    nine-row one.
    """
    out = tmp / name
    (out / "agg").mkdir(parents=True, exist_ok=True)
    raw = ["scenario,family,truth,judge,kind,rep_or_policy,model,verdict,score,correct"]
    summary = ["judge,model,n,accuracy"]
    for judge, model, acc in (("statistical", "-", 0.5),
                              ("ai:summary", "phi4-llm", 0.9),
                              ("ai:summary", "weaker-llm", 0.4),
                              ("hybrid:gated", "phi4-llm", 0.7)):
        summary.append(f"{judge},{model},4,{acc}")
        for i, (truth, score) in enumerate((("FAIL", 10), ("PASS", 90),
                                            ("FAIL", 20), ("PASS", 80))):
            raw.append(f"s{i},fam,{truth},{judge},k,r,{model},{truth},{score},1")
        if own_data_run:            # one judged window nobody labelled
            raw.append(f"s9,fam,,{judge},k,r,{model},PASS,50,")
    (out / "results_raw.csv").write_text("\n".join(raw) + "\n")
    (out / "summary.csv").write_text("\n".join(summary) + "\n")
    for alias in ("claude-opus-4-8-vlm", "gpt-oss-120b-llm", "qwen3-vl-235b-vlm"):
        rows = ["scenario,family,truth,judge,kind,rep_or_policy,model,verdict,score,correct"]
        for judge in ("ai:raw", "hybrid:gated"):
            for i, (truth, score) in enumerate((("FAIL", 10), ("PASS", 90),
                                                ("FAIL", 20), ("PASS", 80))):
                rows.append(f"s{i},fam,{truth},{judge},k,r,{alias},{truth},{score},1")
        (out / "agg" / f"frontier_long_{alias}.csv").write_text("\n".join(rows) + "\n")
    # And a tagged re-run beside the real one, which is not the real one.
    (out / "agg" / "frontier_long_phi4-llm__defect-rerun.csv").write_text(raw[0] + "\n")
    if own_data_run:
        # The manifest of this run: the windows it judged, which is what makes it
        # a record of the directory it sits in rather than a flag.
        _write_own_data_manifest(out, [r.split(",")[0] for r in raw[1:]])
    return out


def _write_own_data_manifest(out: Path, scenario_ids: List[str]) -> None:
    """own_data_manifest.json, shaped the way tools/own_data.py shapes it.

    Only the keys tools/roc_pr_curves.py reads are filled in; the rest of what
    own_data.build_manifest() writes is the record of the load and plays no part
    in deciding what kind of run this is.
    """
    manifest = {"own_data": True,
                "scenarios": [{"id": sid} for sid in dict.fromkeys(scenario_ids)]}
    (out / "own_data_manifest.json").write_text(json.dumps(manifest, indent=2))


def published_tables(tmp: Path) -> List[str]:
    """The seeded run's ROC table is the published one; an own-data run's is its own.

    Two claims, and the first is about bytes. tools/roc_pr_curves.py derived its
    configuration list from `agg/frontier_long_*.csv`, which is three files in the
    study's own archive and not the one the fixture had: the published five-row
    table came out with nine rows, a column inserted at position 4, and
    fig13_threshold_summaries.png -- laid out for five groups -- came out with
    overprinted labels on numbers that had not moved.
    """
    failures: List[str] = []
    published = ["statistical", "ai:summary/phi4-llm (best local)", "hybrid:gated/phi4-llm",
                 "ai:raw/claude-opus-4-8-vlm (frontier best rep)",
                 "hybrid:gated/claude-opus-4-8-vlm"]

    seeded = _auc_fixture(tmp, "auc-seeded", own_data_run=False)
    names = [src["name"] for src in roc.sources_for(seeded)]
    if names != published:
        failures.append(f"a run that is not an own-data run swept {names}, expected the "
                        f"five published configurations {published}")
    if roc.is_own_data_run(seeded):
        failures.append("a directory with no own_data_manifest.json was read as own-data")

    header = _roc_header(seeded)
    want = ["judge", "model", "n", "auc_roc", "auc_pr", "average_precision", "note"]
    if header != want:
        failures.append(f"the seeded roc_pr_auc.csv header is {header}, expected {want} -- "
                        f"the published file has seven columns in that order")

    own = _auc_fixture(tmp, "auc-own", own_data_run=True)
    own_names = [src["name"] for src in roc.sources_for(own)]
    if own_names == published:
        failures.append("an own-data run was given the published five configurations "
                        "instead of its own")
    for alias in ("gpt-oss-120b-llm", "qwen3-vl-235b-vlm"):
        if not any(alias in name for name in own_names):
            failures.append(f"an own-data run holding frontier_long_{alias}.csv did not "
                            f"sweep it: {own_names}")
    if any("__defect-rerun" in name for name in own_names):
        failures.append(f"a `__`-tagged re-run was swept as the run: {own_names}")

    own_header = _roc_header(own)
    if own_header != want + ["n_unlabelled_dropped"]:
        failures.append(f"the own-data roc_pr_auc.csv header is {own_header}; the "
                        f"unlabelled count belongs after {want[-1]!r}, not before "
                        f"auc_roc")
    else:
        rows = list(csv.DictReader((own / "agg" / "roc_pr_auc.csv").open()))
        dropped = {r["n_unlabelled_dropped"] for r in rows if r["model"] == "phi4-llm"}
        if dropped != {"1"}:
            failures.append(f"the unlabelled window was counted as {dropped}, expected 1")

    # A manifest left behind by somebody else's run must not decide this one.
    # Nothing removed own_data_manifest.json, so one `--own-data --out results`
    # put one in the study's own directory and every later seeded run into
    # results/ inherited it: the published five-row, seven-column table came back
    # as the derived nine-row, eight-column one, and fig13_threshold_summaries.png
    # with it. The manifest below is a real own-data manifest -- it is simply
    # about other windows than the ones results_raw.csv holds.
    stale = _auc_fixture(tmp, "auc-seeded-stale", own_data_run=False)
    _write_own_data_manifest(stale, ["own_0000", "own_0001"])
    if roc.is_own_data_run(stale):
        failures.append("a seeded run was read as own-data because a manifest naming "
                        "other windows had been left in the directory")
    stale_names = [src["name"] for src in roc.sources_for(stale)]
    if stale_names != published:
        failures.append(f"a stale own_data_manifest.json changed the swept "
                        f"configurations to {stale_names}, expected the five published "
                        f"ones {published}")
    stale_header = _roc_header(stale)
    if stale_header != want:
        failures.append(f"a stale own_data_manifest.json changed the seeded "
                        f"roc_pr_auc.csv header to {stale_header}, expected {want}")
    return failures


def _roc_header(results_dir: Path) -> List[str]:
    """roc_pr_curves.py's own main(), run over one directory, and its header."""
    argv = sys.argv
    sys.argv = ["roc_pr_curves.py", "--results", str(results_dir)]
    try:
        roc.main()
    finally:
        sys.argv = argv
    with (results_dir / "agg" / "roc_pr_auc.csv").open() as f:
        return next(csv.reader(f))


# 8. The reports are written whatever happens to the figures.
def reports_survive_figures(tmp: Path) -> List[str]:
    """A results directory outside the repository loses figures, not reports.

    `--out` is documented and may point anywhere. Neither figure runner can read
    a spec outside the repository -- the running judge-service has only
    data/ai-logs mounted, the one-off container mounts the repository -- and the
    unconditional `.relative_to(REPO_ROOT)` that built the second runner's
    argument raised ValueError there. It raised inside write_outputs(), after
    every window had been judged and every table written, and before results.md
    and summary.md were: a completed run, exit 1, and neither report.
    """
    failures: List[str] = []
    outside = Path(tempfile.mkdtemp(prefix="outside-repo-"))
    try:
        spec, scratch = outside / "_exp_figdata.json", outside / "_exp_figs"
        if exp._bind_mount_args(spec, scratch) is not None:
            failures.append("a spec outside the repository was passed to the bind-mount "
                            "runner as a relative path")
        if exp._mounted_args(spec, scratch) is not None:
            failures.append("a spec outside data/ai-logs was passed to the running service")

        saved = (exp.RESULTS_DIR, exp.FIG_DIR, exp.FIGDATA_PATH, exp.FIGSCRATCH_DIR)
        exp.RESULTS_DIR, exp.FIG_DIR = outside, outside / "figures"
        exp.FIGDATA_PATH, exp.FIGSCRATCH_DIR = spec, scratch
        try:
            try:
                if exp.render_figures() is not False:
                    failures.append("render_figures() claimed figures it cannot have drawn")
            except Exception as e:  # noqa: BLE001
                failures.append(f"render_figures() raised {type(e).__name__}: {e}; a path "
                                f"outside the bind mount must be a stated skip")

            # And the whole of write_outputs, with the figure step left enabled
            # and made to fail the way a broken renderer would.
            scenarios, manifest = own_data.load(str(_unlabelled_file(tmp, "figs.csv",
                                                                     labelled=2, total=2)))
            rows = _fake_rows(scenarios, verdict="FAIL")
            present = [{"alias": "stub-llm", "modality": "text", "tag": "stub"}]
            det = {"scenario": scenarios[0].id, "model": "stub-llm", "run1": "FAIL",
                   "run2": "FAIL", "identical": True}
            real_figdata = exp._write_figdata
            exp._write_figdata = _raise_figdata
            try:
                exp.write_outputs(rows, present, [], det, manifest,
                                  argparse.Namespace(), 1, False, 60.0)
            finally:
                exp._write_figdata = real_figdata
            for name in ("results.md", "summary.md", "summary.csv"):
                if not (outside / name).is_file():
                    failures.append(f"{name} was not written when the figure step failed; "
                                    f"the reports are what the run is for")
        finally:
            (exp.RESULTS_DIR, exp.FIG_DIR, exp.FIGDATA_PATH, exp.FIGSCRATCH_DIR) = saved
    finally:
        shutil.rmtree(outside, ignore_errors=True)
    return failures


def _raise_figdata(*_args, **_kwargs):
    raise RuntimeError("the renderer is broken in this test, deliberately")


CHECKS = (
    ("roundtrip_judge_surface", roundtrip_judge_surface),
    ("roundtrip_scenario_fields", roundtrip_scenario_fields),
    ("seeded_path_unchanged", seeded_path_unchanged),
    ("validation_rules", validation_rules),
    ("unlabelled_mode", unlabelled_mode),
    ("accuracy_guard", accuracy_guard),
    ("leakage_protection", leakage_protection),
    ("gid_carries_no_order", gid_carries_no_order),
    ("determinism", determinism),
    ("metric_order_is_declared", metric_order_is_declared),
    ("published_tables", published_tables),
    ("reports_survive_figures", reports_survive_figures),
)


def main() -> int:
    all_failures: List[Tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        for name, check in CHECKS:
            failures = check(tmp)
            print(f"  {name:26s} {'FAIL' if failures else 'ok'}")
            all_failures.extend((name, f) for f in failures)

    if all_failures:
        print("\nFAILED:")
        for name, failure in all_failures:
            print(f"  [{name}] {failure}")
        return 1
    print(f"\nThe seeded dataset survives an export and a re-import with an identical\n"
          f"metricSetPairList and an identical canary config on all "
          f"{N_PER_FAMILY * len(ed.FAMILIES)} scenarios, save for\nthe configuration's name "
          f"and description, which carry an opaque handle rather\nthan the window's family "
          f"and ground-truth label, as they must; the generated\npath is unchanged against "
          f"v1.0.0 ({PRISTINE_REV}) on those same terms; a broken file\nis "
          f"refused by file, line and column;\nan unlabelled window is judged and scored "
          f"against nothing; a window that\nnames its own answer does not hand it to "
          f"the model; the seeded ROC table is still the\npublished five in the published "
          f"columns; and a run whose figures cannot be drawn\nstill writes its reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
