#!/usr/bin/env python3
"""Tests of the D3 claim extraction and checks on hand-written sentences and synthetic windows.

The sentences are written for the test; none is taken from a model output.
Run: python -I -B analysis/r1/posthoc/test_d3_extraction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d3_rationale_check as d3  # noqa: E402
import posthoc_common as pc  # noqa: E402

CFG = json.loads((HERE / "d3_rationale_config.json").read_text(encoding="utf-8"))
EX = d3.Extractor(CFG)
MK = pc.repo_modules()["ensemble_judge"].mann_kendall_p


def kinds(text):
    return [(c["type"], c["direction"]) for c in EX.extract(text) if c["type"] != "number"]


def numbers(text):
    return [(c["value"], c["unit"]) for c in EX.extract(text) if c["type"] == "number"]


CASES = [
    ("The canary mean is higher than control.", [("level", "up")]),
    ("Canary median is lower.", [("level", "down")]),
    ("Means are comparable on both sides.", [("level", "same")]),
    ("No increase in the mean.", [("level", "not_up")]),
    ("The canary shows much higher variance than the control.", [("spread", "up")]),
    ("Spread is similar.", [("spread", "same")]),
    ("The canary is erratic.", [("spread", "up")]),
    ("Variability is reduced in the canary.", [("spread", "down")]),
    ("There is an upward trend in the canary.", [("trend", "up")]),
    ("No upward trend is visible.", [("trend", "not_up")]),
    ("No trend.", [("trend", "same")]),
    ("The slope is flat.", [("trend", "same")]),
    ("Memory creeps up over time.", [("trend", "up")]),
    ("Worse tail latency at p99.", [("tail", "up")]),
    ("Several spikes in the canary.", [("tail", "up")]),
    ("No spikes or outliers.", [("tail", "not_up")]),
    ("Tail percentiles are comparable.", [("tail", "same")]),
    ("p95 is lower in the canary.", [("tail", "down")]),
    ("Latency is 30% higher in the canary.", [("level", "up")]),
    ("Memory usage is comparable.", [("level", "same")]),
    ("The canary mean is only 0.5% higher, negligible.", [("level", "up")]),
    ("Variance is lower and the tail improved.", [("spread", "down")]),
    ("Worse tail latencies and no improvement trend.", [("tail", "up"), ("trend", "not_down")]),
    ("Memory increases steadily toward the end of the window.", [("trend", "up")]),
    ("A larger standard deviation, indicating less stable performance.", [("spread", "up")]),
]


def test_directions():
    bad = []
    for text, exp in CASES:
        got = kinds(text)
        for e in exp:
            if e not in got:
                bad.append((text, exp, got))
    assert not bad, "\n".join(f"{t!r}: expected {e}, got {g}" for t, e, g in bad)


def test_numbers():
    assert (34.0, "%") in numbers("The canary mean is 34% higher.")
    assert (36.0, "%") in numbers("an increase of approximately 36%.")
    assert (1.34, "x") in numbers("Latency is 1.34x the control.")
    assert (0.0081, "") in numbers("stddev 0.0081 vs 0.0080")
    assert numbers("p95 and p99 are worse") == []
    assert numbers("a score of 85 was given") == []
    assert numbers("under 60 words") == []


def window(control, canary, kind="latency"):
    return d3.Window("w", "f", "PASS", [{"name": "m", "kind": kind, "control": control, "canary": canary}], MK)


def test_checks():
    b = [1.0 + 0.01 * ((i * 7) % 5 - 2) for i in range(31)]
    up = [v * 1.3 for v in b]
    w = window(b, up)
    c = {"type": "level", "direction": "up", "named": "mean", "kinds": []}
    assert d3.check(c, w, [], CFG)[0] == "consistent"
    c["direction"] = "down"
    assert d3.check(c, w, [], CFG)[0] == "inconsistent"
    c["direction"] = "same"
    assert d3.check(c, w, [], CFG)[0] == "inconsistent"
    ramp = [v + (0.02 * (i - 24) if i >= 24 else 0.0) for i, v in enumerate(b)]
    w2 = window(b, ramp)
    t = {"type": "trend", "direction": "up", "kinds": []}
    assert d3.check(t, w2, [], CFG)[0] == "consistent"
    t["direction"] = "same"
    assert d3.check(t, w2, [], CFG)[0] == "inconsistent"
    n = {"type": "number", "direction": "value", "value": 30.0, "unit": "%", "decimals": 0, "kinds": []}
    assert d3.check(n, w, [], CFG)[0] == "consistent"
    n["value"] = 60.0
    assert d3.check(n, w, [], CFG)[0] == "inconsistent"


if __name__ == "__main__":
    test_directions()
    test_numbers()
    test_checks()
    print("test_d3_extraction: all passed")
