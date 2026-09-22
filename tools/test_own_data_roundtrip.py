#!/usr/bin/env python3
"""Your windows must reach the judges exactly the way the study's do.

`docs/own-data.md` and the article's contribution both rest on one claim: that
the released testbed runs the same judges, under the same configuration and the
same scoring code, over an organisation's own windows as over the seeded
dataset. That is a claim about this code, so it is checked here rather than
asserted.

The check is a round trip. The seeded scenarios are written out through the
documented exporter, read back through the documented loader, and the two things
the judges actually consume are compared:

  * the `metricSetPairList` -- the names, the ids, the values, the scopes and the
    attributes that Kayenta's NetflixACAJudge and every AI representation see;
  * the canary config -- the per-metric statistical block, the group weights and
    the score thresholds the verdict is cut at.

Nothing is approximated: floats are compared as floats and the structures are
compared whole. If the two ever differ, the claim is false and this test says
which scenario, which metric and which field.

The config's `name` and `description` are the one exception, and the exception
is asserted rather than waived: from v1.0.1 they carry the window's opaque
handle instead of its id, family and ground-truth label, so a re-imported
window's must differ from the one it went out with, and neither may carry a
family or a label. See CONFIG_HANDLE_FIELDS below.

The order of the metrics within a window is carried, not canonicalised. The
exporter writes each metric's place in `metric_order` and the loader honours it,
so the comparison below is made in sequence against the source scenario with
nothing re-sorted. For eight of the nine families the order is the same list
either way; the ninth, `cross_metric_marginal`, declares latency, memory, errors
where sorting the names would give latency, errors, memory -- and that order
reaches the prompt and decides which group takes the leftover 0.01 of the
classifier's group weights, so it is not a presentational difference.

It also checks the parts a round trip cannot: that an unlabelled window survives
with `truth == ""` rather than acquiring a verdict it was never given, that
renaming is on by default and hides the window's own name from the prompt, and
that a broken file is rejected by line number rather than repaired.

No stack, no model, no network.

Usage:
  python tools/test_own_data_roundtrip.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import dataclasses  # noqa: E402

import eval_dataset as ed  # noqa: E402
import own_data  # noqa: E402

N_PER_FAMILY = 2
JUDGE = {"name": "NetflixACAJudge-v1.0", "judgeConfigurations": {}}

# The two canary-configuration fields a round trip is not asked to preserve, and
# why. v1.0.0 wrote the scenario id, its family and its ground-truth label into
# the config Kayenta stores and the judge service logs verbatim --
# `eval-no_change_000`, `eval dataset scenario no_change_000 (family=no_change,
# truth=PASS)`. From v1.0.1 both carry the window's opaque handle and nothing
# else, `eval-g0000` and `eval dataset scenario g0000`, so no ground-truth label
# travels in a request. No judge reads either field; `build_pairs` and the rest
# of the config, which every judge does read, are compared whole below.
#
# A re-imported window draws a fresh handle out of the own-data gid namespace
# (own_data.OWN_DATA_GID_BASE), so these two cannot come back as they went out
# and should not: the reloaded window is a different window as far as its
# provenance goes. That is asserted, not assumed -- `_handle_fields_changed`
# fails if either field survives unchanged, which is what would happen if a
# later change put `eval-{scenario.id}` and its label back, and fails if either
# string carries a family or a label at all.
CONFIG_HANDLE_FIELDS = ("name", "description")


def _without_handle(config: Dict[str, Any]) -> Dict[str, Any]:
    """The canary config minus CONFIG_HANDLE_FIELDS: all of it a judge reads."""
    return {k: v for k, v in config.items() if k not in CONFIG_HANDLE_FIELDS}


def _handle_fields_changed(where: str, out: Dict[str, Any], back: Dict[str, Any],
                           family: str, truth: str) -> List[str]:
    """The other half of setting the two aside, so the exclusion cannot go stale."""
    failures: List[str] = []
    for field in CONFIG_HANDLE_FIELDS:
        if out[field] == back[field]:
            failures.append(f"{where}: canary config {field} came back unchanged at "
                            f"{back[field]!r}; a re-imported window is handed a new "
                            f"opaque handle, so this field has to differ")
        for value, what in ((family, "family"), (truth, "ground-truth label")):
            if value and value in back[field]:
                failures.append(f"{where}: canary config {field} is {back[field]!r}, "
                                f"which carries the {what} {value!r}; from v1.0.1 "
                                f"neither field carries one")
    return failures


def _pairs_and_config(scenario: ed.Scenario, series, base_millis: int, step_seconds: int):
    """The two structures a judge consumes, from a series dict the caller owns.

    The series are passed in rather than regenerated here because
    `ed.series_for_scenario` seeds a generated metric's RNG from its *index* in
    `scenario.metrics` (`scenario.seed + 1000 * mi`). Regenerating them for a
    scenario whose metrics have been reordered would hand each metric another
    metric's numbers, which is a different dataset, not a different order.
    """
    pairs = ed.build_pairs(scenario, series, base_millis, step_seconds * 1000)
    return pairs, ed.build_canary_config(scenario, JUDGE)


def _order_kept(source: ed.Scenario, target: ed.Scenario) -> List[str]:
    """A window must come back in the order it went out in.

    `own_data.export` writes `metric_order`, so the order is carried by the file
    and not re-derived from the names. Nothing below is canonicalised: a
    reordering is a failure, which is what it always should have been.
    `cross_metric_marginal` is the family that proves it -- the generator
    declares latency, memory, errors, and sorting the base names gives latency,
    errors, memory.
    """
    want = [m.name for m in source.metrics]
    got = [m.name for m in target.metrics]
    if want == got:
        return []
    if sorted(want) != sorted(got):
        return [f"{source.id}: metrics out {want} came back {got}, which is not a "
                f"reordering of the same metrics"]
    return [f"{source.id}: metric order {want} came back {got}"]


def roundtrip(tmp: Path) -> List[str]:
    """Export the seeded dataset, read it back, compare what the judges consume."""
    failures: List[str] = []
    path = tmp / "seeded.csv"

    original = ed.build_scenarios(ed.DEFAULT_MASTER_SEED, N_PER_FAMILY)
    own_data.export(original, str(path), base_millis=own_data.EXPORT_BASE_MILLIS,
                    step_seconds=ed.STEP_SECONDS)

    # --keep-ids, so the reloaded windows carry the seeded ids and metric names
    # and can be matched to their originals at all. The renaming path is checked
    # separately below.
    loaded, manifest = own_data.load(str(path), anonymise=False)

    if len(loaded) != len(original):
        failures.append(f"exported {len(original)} scenarios, loaded {len(loaded)}")
        return failures
    if manifest["n_labelled"] != len(original):
        failures.append(f"{manifest['n_labelled']} of {len(original)} windows came back labelled")

    by_id = {s.id: s for s in loaded}
    for source in original:
        target = by_id.get(source.id)
        if target is None:
            failures.append(f"{source.id}: did not survive the round trip")
            continue

        # Generated from the seed in the declaration order, then read in the
        # canonical order: same numbers under the same metric names, reordered.
        source_series = ed.series_for_scenario(source)
        failures.extend(_order_kept(source, target))
        want_pairs, want_config = _pairs_and_config(source, source_series,
                                                    own_data.EXPORT_BASE_MILLIS,
                                                    ed.STEP_SECONDS)
        got_pairs, got_config = _pairs_and_config(target, ed.series_for_scenario(target),
                                                  target.base_millis, target.step_seconds)

        failures.extend(_diff(f"{source.id}: metricSetPairList", want_pairs, got_pairs))
        failures.extend(_diff(f"{source.id}: canaryConfig",
                              _without_handle(want_config), _without_handle(got_config)))
        failures.extend(_handle_fields_changed(source.id, want_config, got_config,
                                               source.family, source.truth))
        if source.truth != target.truth:
            failures.append(f"{source.id}: truth {source.truth!r} came back {target.truth!r}")
        if source.family != target.family:
            failures.append(f"{source.id}: family {source.family!r} came back {target.family!r}")
    return failures


def _diff(where: str, want: Any, got: Any, path: str = "") -> List[str]:
    """Structural comparison that names the field, not just the object.

    A diff of two 31-element float arrays printed whole is unreadable at 02:00;
    this reports the first index that differs and both values.
    """
    if isinstance(want, dict) and isinstance(got, dict):
        out: List[str] = []
        for key in sorted(set(want) | set(got)):
            if key not in want:
                out.append(f"{where}{path}.{key}: unexpected key in the reloaded copy")
            elif key not in got:
                out.append(f"{where}{path}.{key}: missing from the reloaded copy")
            else:
                out.extend(_diff(where, want[key], got[key], f"{path}.{key}"))
        return out
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return [f"{where}{path}: {len(want)} element(s) exported, {len(got)} reloaded"]
        out = []
        for i, (a, b) in enumerate(zip(want, got)):
            out.extend(_diff(where, a, b, f"{path}[{i}]"))
            if len(out) > 3:
                break
        return out
    if isinstance(want, float) and isinstance(got, float):
        # Exact, not approximate: repr() round-trips a float without loss, so
        # anything but equality here means the format lost a digit.
        return [] if want == got else [f"{where}{path}: {want!r} exported, {got!r} reloaded"]
    return [] if want == got else [f"{where}{path}: {want!r} exported, {got!r} reloaded"]


def anonymised(tmp: Path) -> List[str]:
    """The round trip has to survive the renaming, not only `--keep-ids`.

    `roundtrip` above loads with `anonymise=False`, so every name in the file is
    carried through untouched and the loader's renaming is never exercised. The
    default is the other way round, and it is the one a practitioner runs: the
    loader derives an opaque name for each metric from the *sorted order* of the
    names the file carries. Export such a window and re-import it and the names
    the loader now sorts are the opaque ones it invented last time, so the
    renaming has to be a fixed point -- sorting its own output must give back the
    order it assigned.

    It was not. `latency_gNNNN`, `latency_2_gNNNN`, `latency_3_gNNNN` sort as
    `_2`, `_3`, then the unnumbered one, because a digit sorts before a letter,
    so a three-latency window came back with each series under its neighbour's
    name, rotating with period three and never returning to itself. Nothing in
    the harness reaches this today -- `--export` only exports seeded scenarios,
    whose windows carry one metric per kind -- which is exactly why it is checked
    here rather than left to be found by the first person whose window has two
    latencies in it.

    Three windows, so the failure cannot hide: one with three metrics of one
    kind, one mixing kinds, and one with a single metric (the shape every seeded
    window has, whose name must not have changed at all).
    """
    failures: List[str] = []
    path = tmp / "anonymised.csv"
    rows = ["scenario,metric,side,timestamp,value,kind,direction,label,unit"]
    # Distinct constants per series, so a rotation moves a number that is visible
    # in the failure message rather than one that has to be inferred.
    window_metrics = {
        "win-multi-latency": [("auth_p99", "latency"), ("checkout_p99", "latency"),
                              ("search_p99", "latency")],
        "win-mixed": [("rss_bytes", "memory"), ("p99", "latency"), ("rate_5xx", "errors")],
        "win-single": [("only_latency", "latency")],
    }
    value = 10.0
    for window, metrics in window_metrics.items():
        for metric, kind in metrics:
            value += 1.0
            for i in range(12):
                stamp = f"2026-03-14T09:{i:02d}:00Z"
                rows.append(f"{window},{metric},control,{stamp},{value},{kind},increase,FAIL,s")
                rows.append(f"{window},{metric},canary,{stamp},{value * 2},{kind},increase,FAIL,s")
    path.write_text("\n".join(rows) + "\n")

    first, _ = own_data.load(str(path))          # anonymise=True, the default
    names = sorted(m.name for s in first for m in s.metrics)
    # The gid is a salted hash of the window's own name (own_data._assign_gids),
    # so the number cannot be written down here; what is asserted is the shape --
    # a window with one metric of a kind is `latency_gNNNN`, unnumbered.
    only = next((s for s in first if s.params["own_data"]["scenario"] == "win-single"), None)
    if only is None:
        failures.append("win-single did not come back at all")
    elif [m.name for m in only.metrics] != [f"latency_g{only.gid:04d}"]:
        failures.append(f"a window with one metric of a kind must still be named "
                        f"'latency_g{only.gid:04d}', unnumbered; got "
                        f"{[m.name for m in only.metrics]}")
    if not all(re.fullmatch(r"[a-z]+(_\d+)?_g\d{4}", n) for n in names):
        failures.append(f"every opaque name must be '<kind>[_n]_gNNNN'; got {names}")

    # Export it, read it back the same way, and compare what the judges consume.
    # Two cycles rather than one: a permutation of period two would survive a
    # single round trip undetected, and the defect this guards against was a
    # permutation.
    previous = first
    for cycle in (1, 2):
        exported = tmp / f"anonymised_cycle{cycle}.csv"
        own_data.export(previous, str(exported))
        reloaded, _ = own_data.load(str(exported))
        if len(reloaded) != len(previous):
            failures.append(f"cycle {cycle}: {len(previous)} window(s) exported, "
                            f"{len(reloaded)} reloaded")
            return failures
        by_id = {s.id: s for s in reloaded}
        for source in previous:
            target = by_id.get(source.id)
            if target is None:
                failures.append(f"cycle {cycle}: {source.id} did not survive the round trip")
                continue
            want = _pairs_and_config(source, ed.series_for_scenario(source),
                                     source.base_millis, source.step_seconds)
            got = _pairs_and_config(target, ed.series_for_scenario(target),
                                    target.base_millis, target.step_seconds)
            failures.extend(_diff(f"{source.id} (cycle {cycle}): metricSetPairList",
                                  want[0], got[0]))
            failures.extend(_diff(f"{source.id} (cycle {cycle}): canaryConfig",
                                  want[1], got[1]))
        previous = reloaded
    return failures


def unlabelled(tmp: Path) -> List[str]:
    """A window with no label keeps no verdict, and is still judged."""
    failures: List[str] = []
    path = tmp / "unlabelled.csv"
    scenarios = ed.build_scenarios(ed.DEFAULT_MASTER_SEED, 1)[:2]
    own_data.export(scenarios, str(path), base_millis=own_data.EXPORT_BASE_MILLIS,
                    step_seconds=ed.STEP_SECONDS)
    # Blank every label cell, the way a file for windows nobody has labelled yet
    # actually looks -- the per-metric truth as well as the window verdict, since
    # someone who has not decided the verdict has not decided which metric caused
    # it either.
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("scenario,"))
    columns = lines[header].split(",")
    blank = [columns.index("label"), columns.index("metric_label")]
    for i in range(header + 1, len(lines)):
        cells = lines[i].split(",")
        for column in blank:
            cells[column] = ""
        lines[i] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")

    loaded, manifest = own_data.load(str(path), anonymise=False)
    if manifest["n_labelled"] != 0:
        failures.append(f"an unlabelled file reports {manifest['n_labelled']} labelled window(s)")
    for scenario in loaded:
        if scenario.truth != "":
            failures.append(f"{scenario.id}: unlabelled window came back truth={scenario.truth!r}")
        for metric in scenario.metrics:
            if metric.truth != "":
                failures.append(f"{scenario.id}/{metric.name}: unlabelled metric came back "
                                f"truth={metric.truth!r}")
        series = ed.series_for_scenario(scenario)
        if not ed.build_pairs(scenario, series, scenario.base_millis,
                              scenario.step_seconds * 1000):
            failures.append(f"{scenario.id}: an unlabelled window produced no pairs to judge")
    return failures


def anonymisation(tmp: Path) -> List[str]:
    """The default must keep the window's own name away from the prompt.

    The only field of a pair that reaches the model is `name`
    (judge-service/app/representations.py), so this checks that field and the
    scenario id that Kayenta's config carries alongside it.
    """
    failures: List[str] = []
    path = tmp / "leaky.csv"
    rows = [
        "scenario,metric,side,timestamp,value,kind,direction,label",
    ]
    for i in range(12):
        stamp = f"2026-03-14T09:{i:02d}:00Z"
        rows.append(f"2026-03-14-rollback-FAIL,checkout_latency_during_the_bad_deploy,"
                    f"control,{stamp},0.2{i:02d},latency,increase,FAIL")
        rows.append(f"2026-03-14-rollback-FAIL,checkout_latency_during_the_bad_deploy,"
                    f"canary,{stamp},0.3{i:02d},latency,increase,FAIL")
    path.write_text("\n".join(rows) + "\n")

    loaded, manifest = own_data.load(str(path))
    scenario = loaded[0]
    leaks = ("rollback", "FAIL", "checkout", "bad_deploy", "2026-03-14")
    series = ed.series_for_scenario(scenario)
    pairs = ed.build_pairs(scenario, series, scenario.base_millis,
                           scenario.step_seconds * 1000)
    for pair in pairs:
        for leak in leaks:
            if leak.lower() in pair["name"].lower():
                failures.append(f"the metric name the model sees ({pair['name']!r}) still "
                                f"carries {leak!r}")
    if scenario.id != "own_0000":
        failures.append(f"window id is {scenario.id!r}, expected 'own_0000'")
    # The gid is drawn from the own-data range by a salted hash of the window
    # name, so what is asserted is the range -- disjoint from the seeded 0... and
    # 1000... ranges -- and not a particular number. Asserting the number is what
    # the old `OWN_DATA_GID_BASE + index` scheme allowed, and that scheme is the
    # defect R2-3 reported; `test_own_data_contract.py::gid_carries_no_order` is
    # the check that it has not come back.
    if not own_data.OWN_DATA_GID_BASE <= scenario.gid < own_data.OWN_DATA_GID_LIMIT:
        failures.append(f"gid is {scenario.gid}, expected it in "
                        f"[{own_data.OWN_DATA_GID_BASE}, {own_data.OWN_DATA_GID_LIMIT}) "
                        f"(disjoint from the seeded 0... and 1000... ranges)")
    if not scenario.metrics[0].name.endswith(f"_g{scenario.gid:04d}"):
        failures.append(f"the metric name {scenario.metrics[0].name!r} does not carry "
                        f"the window's gid {scenario.gid}")
    # The real names must survive somewhere, or the run is unreadable afterwards.
    recorded = json.dumps(manifest)
    for original in ("2026-03-14-rollback-FAIL", "checkout_latency_during_the_bad_deploy"):
        if original not in recorded:
            failures.append(f"{original!r} is in neither the manifest nor the mapping; "
                            f"the run cannot be traced back to your data")
    if not manifest["anonymised"]:
        failures.append("the manifest does not record that renaming was on")
    return failures


def rejection(tmp: Path) -> List[str]:
    """A broken file is refused by line number, and never repaired."""
    failures: List[str] = []
    cases = {
        "missing direction column": (
            "scenario,metric,side,timestamp,value,kind\n"
            "w,m,control,2026-03-14T09:00:00Z,1.0,latency\n", "direction"),
        "bad kind": (
            "scenario,metric,side,timestamp,value,kind,direction\n"
            "w,m,control,2026-03-14T09:00:00Z,1.0,disk_queue,increase\n", "disk_queue"),
        "one-sided metric": (
            "scenario,metric,side,timestamp,value,kind,direction\n"
            + "".join(f"w,m,control,2026-03-14T09:{i:02d}:00Z,1.0,latency,increase\n"
                      for i in range(12)), "canary"),
        "too few samples": (
            "scenario,metric,side,timestamp,value,kind,direction\n"
            + "".join(f"w,m,{s},2026-03-14T09:{i:02d}:00Z,1.0,latency,increase\n"
                      for i in range(4) for s in ("control", "canary")), "at least 10"),
        "irregular spacing": (
            "scenario,metric,side,timestamp,value,kind,direction\n"
            + "".join(f"w,m,{s},2026-03-14T09:{i * 2 if i < 6 else i * 2 + 1:02d}:00Z,"
                      f"1.0,latency,increase\n"
                      for i in range(12) for s in ("control", "canary")), "not a multiple"),
        "duplicate timestamp": (
            "scenario,metric,side,timestamp,value,kind,direction\n"
            + "".join(f"w,m,{s},2026-03-14T09:{min(i, 9):02d}:00Z,1.0,latency,increase\n"
                      for i in range(12) for s in ("control", "canary")), "duplicate"),
    }
    for name, (text, expected) in cases.items():
        path = tmp / f"{name.replace(' ', '_')}.csv"
        path.write_text(text)
        try:
            own_data.load(str(path))
        except own_data.OwnDataError as e:
            joined = "\n".join(e.messages)
            if expected not in joined:
                failures.append(f"{name}: rejected, but no message mentioned {expected!r}: "
                                f"{joined[:200]}")
            if not any(path.name in m for m in e.messages):
                failures.append(f"{name}: no message names the file it is in")
        else:
            failures.append(f"{name}: loaded without complaint; it should have been rejected")
    return failures


def overrides(tmp: Path) -> List[str]:
    """A statistical override reaches the config; a typo in one is an error."""
    failures: List[str] = []
    path = tmp / "override.json"
    path.write_text(json.dumps({"effectSize": {"allowedIncrease": 1.10}}))
    scenario = ed.build_scenarios(ed.DEFAULT_MASTER_SEED, 1)[0]
    ed.series_for_scenario(scenario)
    scenario.canary_overrides = own_data.read_stat_config(str(path))
    block = ed.build_canary_config(scenario, JUDGE)["metrics"][0]["analysisConfigurations"]["canary"]
    if block["effectSize"]["allowedIncrease"] != 1.10:
        failures.append(f"allowedIncrease came through as {block['effectSize']['allowedIncrease']}")
    if block["effectSize"]["criticalIncrease"] != 1.25:
        failures.append("overriding one effectSize gate disturbed the other four")
    if ed.STAT_CANARY_TEMPLATE["effectSize"]["allowedIncrease"] != 1.05:
        failures.append("an override mutated STAT_CANARY_TEMPLATE itself")

    for bad, why in (({"scoreThresholds": {"pass": 60}}, "scoreThresholds"),
                     ({"direction": "decrease"}, "direction"),
                     ({"effectSize": {"allowedIncrase": 1.1}}, "a misspelled gate"),
                     ({"outlierz": {}}, "an unknown key")):
        path.write_text(json.dumps(bad))
        try:
            own_data.read_stat_config(str(path))
        except own_data.OwnDataError:
            pass
        else:
            failures.append(f"{why} was accepted silently; a typo must not be a no-op")
    return failures


def main() -> int:
    all_failures: List[str] = []
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for name, check in (("round trip", roundtrip), ("anonymised", anonymised),
                            ("unlabelled", unlabelled),
                            ("anonymisation", anonymisation), ("rejection", rejection),
                            ("stat overrides", overrides)):
            failures = check(tmp)
            print(f"  {name:16s} {'FAIL' if failures else 'ok'}")
            all_failures.extend(failures)

    if all_failures:
        print("\nFAILED:")
        for failure in all_failures[:40]:
            print(f"  - {failure}")
        if len(all_failures) > 40:
            print(f"  ... and {len(all_failures) - 40} more")
        return 1
    print(f"\nThe seeded dataset survives an export and a re-import with an identical\n"
          f"metricSetPairList and an identical canary config, over "
          f"{len(ed.FAMILIES) * N_PER_FAMILY} scenarios,\n"
          f"with the metrics of a window in the order the file declares for them.\n"
          f"The config's name and description are the exception, and they carry an\n"
          f"opaque handle rather than the window's family and label, as they must.\n"
          f"Your own windows reach the judges the way the study's do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
